#include "bf_cuda_transfusion.h"
#include "bf_cuda_ops.h"

#include <cuda_runtime.h>
#ifdef BF_CUDA_VENDOR
#include <cublas_v2.h>
#endif
#include <cub/cub.cuh>

#include <cmath>
#include <cfloat>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHANNELS 128
#define HEADS 8
#define HEAD_CHANNELS 16

struct device_linear { float *weight, *bias; int in_features, out_features; };
struct position_weights { device_linear first, second; };
struct head_weights { device_linear hidden, output; int channels; };

struct bf_cuda_transfusion {
#ifdef BF_CUDA_VENDOR
    cublasHandle_t cublas;
#endif
    size_t height, width, keys, proposals, candidates;
    size_t resident_bytes, arena_bytes, cub_bytes;
    unsigned char *arena;
    void *cub_workspace;
    device_linear class_encoding;
    device_linear self_q, self_k, self_v, self_out;
    device_linear cross_q, cross_k, cross_v, cross_out;
    device_linear ffn1, ffn2;
    float *norm_scale[3], *norm_bias[3];
    position_weights self_position, cross_position;
    head_weights heads[6];
    bf_detections *device_detections;
};

struct stage_profiler {
    bool enabled; cudaEvent_t event[7];
    stage_profiler(cudaStream_t stream):enabled(std::getenv("BF_CUDA_TRANSFUSION_PROFILE")!=nullptr),event{} {
        if(enabled){for(int i=0;i<7;++i)cudaEventCreateWithFlags(&event[i],cudaEventDefault);cudaEventRecord(event[0],stream);}
    }
    void mark(int index,cudaStream_t stream){if(enabled)cudaEventRecord(event[index],stream);}
    void finish(){if(!enabled)return;cudaEventSynchronize(event[6]);float value[6]={};
        for(int i=0;i<6;++i)cudaEventElapsedTime(&value[i],event[i],event[i+1]);
        std::fprintf(stderr,"cuda_transfusion stages select=%.3f self=%.3f keys=%.3f cross=%.3f ffn=%.3f heads=%.3f ms\n",
            value[0],value[1],value[2],value[3],value[4],value[5]);}
    ~stage_profiler(){for(int i=0;enabled&&i<7;++i)cudaEventDestroy(event[i]);}
};

static int fail(char *error,size_t cap,const char *format,...) {
    if(error&&cap){va_list args;va_start(args,format);std::vsnprintf(error,cap,format,args);va_end(args);}return 0;
}
static int cuda_ok(cudaError_t s,char *e,size_t c,const char *w){return s==cudaSuccess?1:fail(e,c,"%s: %s",w,cudaGetErrorString(s));}
#ifdef BF_CUDA_VENDOR
static int blas_ok(cublasStatus_t s,char *e,size_t c,const char *w){return s==CUBLAS_STATUS_SUCCESS?1:fail(e,c,"%s: cuBLAS status %d",w,(int)s);}
#endif

static const float *find_f32(const bf_model *m,const char *name,uint32_t rank,
                             const uint32_t *dims,char *error,size_t cap){
    const bf_tensor *t=bf_model_find(m,name);
    if(!t||t->dtype!=BF_DTYPE_F32||t->rank!=rank){fail(error,cap,"%s contract mismatch",name);return nullptr;}
    for(uint32_t i=0;i<rank;++i)if(t->dims[i]!=dims[i]){fail(error,cap,"%s shape mismatch",name);return nullptr;}
    return (const float *)t->data;
}

static int upload_array(float **out,const float *host,size_t count,
                        bf_cuda_transfusion *d,char *error,size_t cap){
    size_t bytes=count*sizeof(float);
    if(!cuda_ok(cudaMalloc((void **)out,bytes),error,cap,"allocate decoder weight")||
       !cuda_ok(cudaMemcpy(*out,host,bytes,cudaMemcpyHostToDevice),error,cap,"upload decoder weight"))return 0;
    d->resident_bytes+=bytes;return 1;
}

static int bind_linear(bf_cuda_transfusion *d,const bf_model *m,device_linear *l,
                       const char *weight_name,const char *bias_name,
                       int outputs,int inputs,char *error,size_t cap){
    uint32_t wd[2]={(uint32_t)outputs,(uint32_t)inputs},bd[1]={(uint32_t)outputs};
    const float *w=find_f32(m,weight_name,2,wd,error,cap);
    const float *b=bias_name?find_f32(m,bias_name,1,bd,error,cap):nullptr;
    if(!w||(bias_name&&!b)||!upload_array(&l->weight,w,(size_t)outputs*inputs,d,error,cap))return 0;
    if(b){if(!upload_array(&l->bias,b,outputs,d,error,cap))return 0;}
    l->in_features=inputs;l->out_features=outputs;return 1;
}

static int bind_slice(bf_cuda_transfusion *d,const bf_model *m,device_linear *l,
                      const char *weight_name,const char *bias_name,int slice,
                      char *error,size_t cap){
    uint32_t wd[2]={3*CHANNELS,CHANNELS},bd[1]={3*CHANNELS};
    const float *w=find_f32(m,weight_name,2,wd,error,cap);
    const float *b=find_f32(m,bias_name,1,bd,error,cap);
    if(!w||!b||!upload_array(&l->weight,w+(size_t)slice*CHANNELS*CHANNELS,
                             CHANNELS*CHANNELS,d,error,cap)||
       !upload_array(&l->bias,b+(size_t)slice*CHANNELS,CHANNELS,d,error,cap))return 0;
    l->in_features=l->out_features=CHANNELS;return 1;
}

static int bind_folded_1d(bf_cuda_transfusion *d,const bf_model *m,device_linear *l,
    const char *weight_name,const char *bn_prefix,int outputs,int inputs,
    char *error,size_t cap){
    uint32_t wd[3]={(uint32_t)outputs,(uint32_t)inputs,1},cd[1]={(uint32_t)outputs};
    const float *source=find_f32(m,weight_name,3,wd,error,cap);
    const char *suffix[4]={"weight","bias","running_mean","running_var"};
    const float *bn[4]={};char name[320];
    for(int i=0;i<4;++i){std::snprintf(name,sizeof(name),"%s.%s",bn_prefix,suffix[i]);bn[i]=find_f32(m,name,1,cd,error,cap);}
    if(!source||!bn[0]||!bn[1]||!bn[2]||!bn[3])return 0;
    size_t count=(size_t)outputs*inputs;float *w=(float *)std::malloc(count*sizeof(float));
    float *b=(float *)std::malloc((size_t)outputs*sizeof(float));
    if(!w||!b){std::free(w);std::free(b);return fail(error,cap,"decoder fold allocation failed");}
    for(int o=0;o<outputs;++o){float factor=bn[0][o]/std::sqrt(bn[3][o]+1e-5f);
        b[o]=bn[1][o]-bn[2][o]*factor;
        for(int i=0;i<inputs;++i)w[(size_t)o*inputs+i]=source[(size_t)o*inputs+i]*factor;}
    int ok=upload_array(&l->weight,w,count,d,error,cap)&&upload_array(&l->bias,b,outputs,d,error,cap);
    std::free(w);std::free(b);l->in_features=inputs;l->out_features=outputs;return ok;
}

static int bind_position(bf_cuda_transfusion *d,const bf_model *m,const char *base,
                         position_weights *p,char *error,size_t cap){
    char w[224],b[224],bn[224];uint32_t wd[3]={CHANNELS,2,1},bd[1]={CHANNELS};
    std::snprintf(w,sizeof(w),"%s.position_embedding_head.0.weight",base);
    std::snprintf(b,sizeof(b),"%s.position_embedding_head.0.bias",base);
    std::snprintf(bn,sizeof(bn),"%s.position_embedding_head.1",base);
    const float *sw=find_f32(m,w,3,wd,error,cap),*sb=find_f32(m,b,1,bd,error,cap);
    const char *suffix[4]={"weight","bias","running_mean","running_var"};const float *v[4]={};char n[320];
    for(int i=0;i<4;++i){std::snprintf(n,sizeof(n),"%s.%s",bn,suffix[i]);v[i]=find_f32(m,n,1,bd,error,cap);}
    if(!sw||!sb||!v[0]||!v[1]||!v[2]||!v[3])return 0;
    float fw[CHANNELS*2],fb[CHANNELS];
    for(int o=0;o<CHANNELS;++o){float factor=v[0][o]/std::sqrt(v[3][o]+1e-5f);
        fb[o]=(sb[o]-v[2][o])*factor+v[1][o];fw[o*2]=sw[o*2]*factor;fw[o*2+1]=sw[o*2+1]*factor;}
    if(!upload_array(&p->first.weight,fw,CHANNELS*2,d,error,cap)||
       !upload_array(&p->first.bias,fb,CHANNELS,d,error,cap))return 0;
    p->first.in_features=2;p->first.out_features=CHANNELS;
    std::snprintf(w,sizeof(w),"%s.position_embedding_head.3.weight",base);
    std::snprintf(b,sizeof(b),"%s.position_embedding_head.3.bias",base);
    uint32_t second_w[3]={CHANNELS,CHANNELS,1};
    const float *second_weight=find_f32(m,w,3,second_w,error,cap);
    const float *second_bias=find_f32(m,b,1,bd,error,cap);
    if(!second_weight||!second_bias||
       !upload_array(&p->second.weight,second_weight,CHANNELS*CHANNELS,d,error,cap)||
       !upload_array(&p->second.bias,second_bias,CHANNELS,d,error,cap))return 0;
    p->second.in_features=p->second.out_features=CHANNELS;return 1;
}

static int bind_head(bf_cuda_transfusion *d,const bf_model *m,const char *name,
                     int channels,head_weights *h,char *error,size_t cap){
    char w[224],b[224],bn[224];
    std::snprintf(w,sizeof(w),"dense_head.prediction_head.%s.0.0.weight",name);
    std::snprintf(bn,sizeof(bn),"dense_head.prediction_head.%s.0.1",name);
    if(!bind_folded_1d(d,m,&h->hidden,w,bn,64,CHANNELS,error,cap))return 0;
    uint32_t wd[3]={(uint32_t)channels,64,1},bd[1]={(uint32_t)channels};
    std::snprintf(w,sizeof(w),"dense_head.prediction_head.%s.1.weight",name);
    std::snprintf(b,sizeof(b),"dense_head.prediction_head.%s.1.bias",name);
    const float *sw=find_f32(m,w,3,wd,error,cap),*sb=find_f32(m,b,1,bd,error,cap);
    if(!sw||!sb||!upload_array(&h->output.weight,sw,(size_t)channels*64,d,error,cap)||
       !upload_array(&h->output.bias,sb,channels,d,error,cap))return 0;
    h->output.in_features=64;h->output.out_features=channels;h->channels=channels;return 1;
}

__global__ static void suppress_kernel(const float *logits,float *output,int h,int w){
    size_t flat=(size_t)blockIdx.x*blockDim.x+threadIdx.x,spatial=(size_t)h*w,total=10*spatial;
    if(flat>=total)return;int cls=(int)(flat/spatial),position=(int)(flat%spatial),y=position/w,x=position-y*w;
    float v=1.0f/(1.0f+expf(-logits[flat]));bool keep=cls==8||cls==9;
    if(!keep&&y>0&&x>0&&y+1<h&&x+1<w){float maximum=-FLT_MAX;
        for(int ky=-1;ky<=1;++ky)for(int kx=-1;kx<=1;++kx){float q=1.0f/(1.0f+expf(-logits[(size_t)cls*spatial+(y+ky)*w+x+kx]));maximum=fmaxf(maximum,q);}keep=v==maximum;}
    output[flat]=keep?v:0.0f;
}
__global__ static void iota_kernel(uint32_t *values,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)values[i]=(uint32_t)i;}
__global__ static void select_kernel(const uint32_t *flat,const float *suppressed,
    const float *shared,const float *class_w,const float *class_b,float *query,
    int64_t *labels,int64_t *indices,float *query_scores,int keys,int proposals){
    int p=blockIdx.x,c=threadIdx.x;if(p>=proposals)return;uint32_t f=flat[p];int label=f/(uint32_t)keys,index=f%(uint32_t)keys;
    if(c==0){labels[p]=label;indices[p]=index;}
    if(c<CHANNELS)query[(size_t)p*CHANNELS+c]=shared[(size_t)c*keys+index]+class_b[c]+class_w[(size_t)c*10+label];
    if(c<10)query_scores[(size_t)c*proposals+p]=suppressed[(size_t)c*keys+index];
}
__global__ static void position_first_kernel(float *out,const float *weight,const float *bias,
    const int64_t *indices,int rows,int width){int row=blockIdx.x,c=threadIdx.x;if(row<rows&&c<CHANNELS){
    int index=indices?(int)indices[row]:row;float x=(float)(index%width)+0.5f,y=(float)(index/width)+0.5f;
    out[(size_t)row*CHANNELS+c]=fmaxf(0.0f,bias[c]+weight[c*2]*x+weight[c*2+1]*y);}}
__global__ static void add_shared_kernel(float *tokens,const float *shared,int keys){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)keys*CHANNELS){int k=i/CHANNELS,c=i%CHANNELS;tokens[i]+=shared[(size_t)c*keys+k];}}
__global__ static void add_kernel(const float *a,const float *b,float *out,size_t n,bool relu){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=a[i]+b[i];out[i]=relu?fmaxf(v,0.0f):v;}}
#ifdef BF_CUDA_VENDOR
__global__ static void bias_activation_kernel(float *x,const float *bias,int rows,int cols,bool relu){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<(size_t)rows*cols){float v=x[i]+bias[i%cols];x[i]=relu?fmaxf(v,0.0f):v;}}
#endif
__global__ static void relu_kernel(float *x,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=fmaxf(x[i],0.0f);}

__global__ static void layer_norm_kernel(const float *residual,const float *update,float *out,
    const float *scale,const float *bias,int rows){int row=blockIdx.x,lane=threadIdx.x;__shared__ float sum[CHANNELS],sq[CHANNELS];
    float v=residual[(size_t)row*CHANNELS+lane]+update[(size_t)row*CHANNELS+lane];sum[lane]=v;sq[lane]=v*v;__syncthreads();
    for(int stride=64;stride;stride>>=1){if(lane<stride){sum[lane]+=sum[lane+stride];sq[lane]+=sq[lane+stride];}__syncthreads();}
    float mean=sum[0]/CHANNELS,var=fmaxf(0.0f,sq[0]/CHANNELS-mean*mean);out[(size_t)row*CHANNELS+lane]=(v-mean)*rsqrtf(var+1e-5f)*scale[lane]+bias[lane];}

__global__ static void flash_attention_kernel(const float *q,const float *k,const float *v,
    float *out,int queries,int keys){int query=blockIdx.x,head=blockIdx.y,lane=threadIdx.x;if(query>=queries)return;
    float qv=lane<HEAD_CHANNELS?q[((size_t)query*CHANNELS+head*HEAD_CHANNELS+lane)]:0.0f;
    float accumulator=0.0f,m=-FLT_MAX,l=0.0f;
    for(int key=0;key<keys;++key){float dot=lane<HEAD_CHANNELS?qv*k[(size_t)key*CHANNELS+head*HEAD_CHANNELS+lane]:0.0f;
        for(int offset=16;offset;offset>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,offset);
        float alpha=1.0f,beta=0.0f;
        if(lane==0){float score=dot*0.25f,new_m=fmaxf(m,score);alpha=expf(m-new_m);beta=expf(score-new_m);l=l*alpha+beta;m=new_m;}
        alpha=__shfl_sync(0xffffffffu,alpha,0);beta=__shfl_sync(0xffffffffu,beta,0);
        if(lane<HEAD_CHANNELS)accumulator=accumulator*alpha+beta*v[(size_t)key*CHANNELS+head*HEAD_CHANNELS+lane];}
    l=__shfl_sync(0xffffffffu,l,0);if(lane<HEAD_CHANNELS)out[(size_t)query*CHANNELS+head*HEAD_CHANNELS+lane]=accumulator/l;}

template <bool FastExp> __global__ static void flash_attention_query_tile_kernel(const float *q,const float *k,const float *v,
    float *out,int queries,int keys){int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int warps=blockDim.x>>5,query=blockIdx.x*warps+warp,head=blockIdx.y;if(query>=queries)return;
    float qv=lane<HEAD_CHANNELS?q[(size_t)query*CHANNELS+head*HEAD_CHANNELS+lane]:0.0f;
    float accumulator=0.0f,m=-FLT_MAX,l=0.0f;
    for(int key=0;key<keys;++key){float dot=lane<HEAD_CHANNELS?qv*k[(size_t)key*CHANNELS+head*HEAD_CHANNELS+lane]:0.0f;
        for(int offset=16;offset;offset>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,offset);
        float alpha=1.0f,beta=0.0f;if(lane==0){float score=dot*0.25f,new_m=fmaxf(m,score);
            alpha=FastExp?__expf(m-new_m):expf(m-new_m);
            beta=FastExp?__expf(score-new_m):expf(score-new_m);l=l*alpha+beta;m=new_m;}
        alpha=__shfl_sync(0xffffffffu,alpha,0);beta=__shfl_sync(0xffffffffu,beta,0);
        if(lane<HEAD_CHANNELS)accumulator=accumulator*alpha+beta*v[(size_t)key*CHANNELS+head*HEAD_CHANNELS+lane];}
    l=__shfl_sync(0xffffffffu,l,0);if(lane<HEAD_CHANNELS)out[(size_t)query*CHANNELS+head*HEAD_CHANNELS+lane]=accumulator/l;}

static void launch_attention(const float *q,const float *k,const float *v,float *out,
                             int queries,int keys,cudaStream_t stream){
    if(std::getenv("BF_CUDA_TRANSFUSION_ONE_QUERY"))
        flash_attention_kernel<<<dim3((unsigned)queries,HEADS),32,0,stream>>>(q,k,v,out,queries,keys);
    else if(std::getenv("BF_CUDA_TRANSFUSION_TILE16"))
        flash_attention_query_tile_kernel<false><<<dim3((unsigned)((queries+15)/16),HEADS),512,0,stream>>>(q,k,v,out,queries,keys);
    else if(std::getenv("BF_CUDA_TRANSFUSION_FAST_EXP"))
        flash_attention_query_tile_kernel<true><<<dim3((unsigned)((queries+7)/8),HEADS),256,0,stream>>>(q,k,v,out,queries,keys);
    else
        flash_attention_query_tile_kernel<false><<<dim3((unsigned)((queries+7)/8),HEADS),256,0,stream>>>(q,k,v,out,queries,keys);
}

__global__ static void transpose_head_kernel(const float *rows,float *channels,int proposals,int count,int center,const int64_t *indices,int width){
    int p=blockIdx.x,c=threadIdx.x;if(p<proposals&&c<count){float v=rows[(size_t)p*count+c];if(center)v+=(float)((c?indices[p]/width:indices[p]%width))+0.5f;channels[(size_t)c*proposals+p]=v;}}

static int linear(bf_cuda_transfusion *d,const device_linear &l,const float *input,float *output,
                  int rows,cudaStream_t stream,char *error,size_t cap){
#ifdef BF_CUDA_VENDOR
    if(!blas_ok(cublasSetStream(d->cublas,stream),error,cap,"set decoder stream"))return 0;
    const float one=1.0f,zero=0.0f;
    if(!blas_ok(cublasSgemm(d->cublas,CUBLAS_OP_T,CUBLAS_OP_N,l.out_features,rows,l.in_features,
        &one,l.weight,l.in_features,input,l.in_features,&zero,output,l.out_features),error,cap,"decoder GEMM"))return 0;
    if(l.bias)bias_activation_kernel<<<((size_t)rows*l.out_features+255)/256,256,0,stream>>>(output,l.bias,rows,l.out_features,false);
    return cuda_ok(cudaGetLastError(),error,cap,"decoder linear kernel");
#else
    (void)d;
    return bf_cuda_gemm_f32(input,l.weight,l.bias,output,rows,l.in_features,
                            l.out_features,0,reinterpret_cast<void *>(stream),
                            error,cap);
#endif
}

typedef struct {float *suppressed,*sort_a,*sort_b;uint32_t *index_a,*index_b;float *key_value,*key_tmp,*key_proj,*value_proj;
    float *query,*query_pos_hidden,*query_pos,*temporary,*attention,*q_proj,*ffn,*head_hidden,*head_rows;} work_views;

static work_views views(bf_cuda_transfusion *d){unsigned char *p=d->arena;work_views v={};
#define TAKE(field,type,count) do{v.field=(type *)p;p+=(size_t)(count)*sizeof(type);}while(0)
    TAKE(suppressed,float,d->candidates);TAKE(sort_a,float,d->candidates);TAKE(sort_b,float,d->candidates);
    TAKE(index_a,uint32_t,d->candidates);TAKE(index_b,uint32_t,d->candidates);
    TAKE(key_value,float,d->keys*CHANNELS);TAKE(key_tmp,float,d->keys*CHANNELS);
    TAKE(key_proj,float,d->keys*CHANNELS);TAKE(value_proj,float,d->keys*CHANNELS);
    TAKE(query,float,d->proposals*CHANNELS);TAKE(query_pos_hidden,float,d->proposals*CHANNELS);
    TAKE(query_pos,float,d->proposals*CHANNELS);TAKE(temporary,float,d->proposals*CHANNELS);
    TAKE(attention,float,d->proposals*CHANNELS);TAKE(q_proj,float,d->proposals*CHANNELS);
    TAKE(ffn,float,d->proposals*256);TAKE(head_hidden,float,d->proposals*64);TAKE(head_rows,float,d->proposals*10);
#undef TAKE
    return v;}

extern "C" int bf_cuda_transfusion_create(const bf_model *m,size_t h,size_t w,size_t proposals,
    bf_cuda_transfusion **out,char *error,size_t cap){if(out)*out=nullptr;
    if(!m||!out||!h||!w||!proposals||proposals>200||h>INT_MAX||w>INT_MAX||h>SIZE_MAX/w||10*h*w>UINT32_MAX)
        return fail(error,cap,"invalid CUDA TransFusion contract");
    bf_cuda_transfusion *d=(bf_cuda_transfusion *)std::calloc(1,sizeof(*d));if(!d)return fail(error,cap,"decoder context allocation failed");
    d->height=h;d->width=w;d->keys=h*w;d->proposals=proposals;d->candidates=10*d->keys;
#ifdef BF_CUDA_VENDOR
    if(!blas_ok(cublasCreate(&d->cublas),error,cap,"create decoder cuBLAS")||
       !blas_ok(cublasSetMathMode(d->cublas,CUBLAS_PEDANTIC_MATH),error,cap,"set pedantic GEMM")||
       !blas_ok(cublasSetAtomicsMode(d->cublas,CUBLAS_ATOMICS_NOT_ALLOWED),error,cap,"disable GEMM atomics"))goto failure;
#endif
    {uint32_t wd[3]={CHANNELS,10,1},bd[1]={CHANNELS};const float *sw=find_f32(m,"dense_head.class_encoding.weight",3,wd,error,cap);
     const float *sb=find_f32(m,"dense_head.class_encoding.bias",1,bd,error,cap);if(!sw||!sb||
       !upload_array(&d->class_encoding.weight,sw,CHANNELS*10,d,error,cap)||!upload_array(&d->class_encoding.bias,sb,CHANNELS,d,error,cap))goto failure;
     d->class_encoding.in_features=10;d->class_encoding.out_features=CHANNELS;}
#define SLICE(dst,prefix,s) if(!bind_slice(d,m,&dst,"dense_head.decoder." prefix ".in_proj_weight","dense_head.decoder." prefix ".in_proj_bias",s,error,cap))goto failure
    SLICE(d->self_q,"self_attn",0);SLICE(d->self_k,"self_attn",1);SLICE(d->self_v,"self_attn",2);
    SLICE(d->cross_q,"multihead_attn",0);SLICE(d->cross_k,"multihead_attn",1);SLICE(d->cross_v,"multihead_attn",2);
#undef SLICE
    if(!bind_linear(d,m,&d->self_out,"dense_head.decoder.self_attn.out_proj.weight","dense_head.decoder.self_attn.out_proj.bias",128,128,error,cap)||
       !bind_linear(d,m,&d->cross_out,"dense_head.decoder.multihead_attn.out_proj.weight","dense_head.decoder.multihead_attn.out_proj.bias",128,128,error,cap)||
       !bind_linear(d,m,&d->ffn1,"dense_head.decoder.linear1.weight","dense_head.decoder.linear1.bias",256,128,error,cap)||
       !bind_linear(d,m,&d->ffn2,"dense_head.decoder.linear2.weight","dense_head.decoder.linear2.bias",128,256,error,cap)||
       !bind_position(d,m,"dense_head.decoder.self_posembed",&d->self_position,error,cap)||
       !bind_position(d,m,"dense_head.decoder.cross_posembed",&d->cross_position,error,cap))goto failure;
    for(int i=0;i<3;++i){char n[128];uint32_t dims[1]={128};std::snprintf(n,sizeof(n),"dense_head.decoder.norm%d.weight",i+1);
        const float *scale=find_f32(m,n,1,dims,error,cap);std::snprintf(n,sizeof(n),"dense_head.decoder.norm%d.bias",i+1);
        const float *bias=find_f32(m,n,1,dims,error,cap);if(!scale||!bias||!upload_array(&d->norm_scale[i],scale,128,d,error,cap)||
           !upload_array(&d->norm_bias[i],bias,128,d,error,cap))goto failure;}
    {const char *names[6]={"center","height","dim","rot","vel","heatmap"};const int channels[6]={2,1,3,2,2,10};
     for(int i=0;i<6;++i)if(!bind_head(d,m,names[i],channels[i],&d->heads[i],error,cap))goto failure;}
    {size_t floats=3*d->candidates+4*d->keys*CHANNELS+d->proposals*(6*CHANNELS+256+64+10);
     size_t integers=2*d->candidates*sizeof(uint32_t);if(floats>SIZE_MAX/sizeof(float)||floats*sizeof(float)>SIZE_MAX-integers)goto failure;
     d->arena_bytes=floats*sizeof(float)+integers;if(!cuda_ok(cudaMalloc((void **)&d->arena,d->arena_bytes),error,cap,"allocate decoder arena"))goto failure;
     d->resident_bytes+=d->arena_bytes;}
    {work_views v=views(d);cub::DeviceRadixSort::SortPairsDescending(nullptr,d->cub_bytes,v.sort_a,v.sort_b,v.index_a,v.index_b,d->candidates);
     if(d->cub_bytes&&!cuda_ok(cudaMalloc(&d->cub_workspace,d->cub_bytes),error,cap,"allocate CUB workspace"))goto failure;
     d->resident_bytes+=d->cub_bytes;}
    if(!cuda_ok(cudaMalloc((void **)&d->device_detections,sizeof(bf_detections)),error,cap,
                "allocate canonical detection boundary"))goto failure;
    d->resident_bytes+=sizeof(bf_detections);
    *out=d;return 1;
failure:bf_cuda_transfusion_destroy(d);return 0;}

static void free_linear(device_linear *l){cudaFree(l->weight);cudaFree(l->bias);}
extern "C" void bf_cuda_transfusion_destroy(bf_cuda_transfusion *d){if(!d)return;
    free_linear(&d->class_encoding);free_linear(&d->self_q);free_linear(&d->self_k);free_linear(&d->self_v);free_linear(&d->self_out);
    free_linear(&d->cross_q);free_linear(&d->cross_k);free_linear(&d->cross_v);free_linear(&d->cross_out);free_linear(&d->ffn1);free_linear(&d->ffn2);
    free_linear(&d->self_position.first);free_linear(&d->self_position.second);free_linear(&d->cross_position.first);free_linear(&d->cross_position.second);
    for(int i=0;i<3;++i){cudaFree(d->norm_scale[i]);cudaFree(d->norm_bias[i]);}for(int i=0;i<6;++i){free_linear(&d->heads[i].hidden);free_linear(&d->heads[i].output);}
    cudaFree(d->arena);cudaFree(d->cub_workspace);cudaFree(d->device_detections);
#ifdef BF_CUDA_VENDOR
    if(d->cublas)cublasDestroy(d->cublas);
#endif
    std::free(d);}

extern "C" int bf_cuda_transfusion_forward(bf_cuda_transfusion *d,const float *shared,const float *heatmap,
    bf_transfusion_raw_outputs *out,void *stream_value,char *error,size_t cap){
    if(!d||!shared||!heatmap||!out||!out->center_b2p||!out->height_b1p||!out->dimension_log_b3p||!out->rotation_sincos_b2p||
       !out->velocity_b2p||!out->heatmap_logits_b10p||!out->query_heatmap_scores_b10p||!out->query_labels_bp||!out->query_indices_bp)
       return fail(error,cap,"invalid CUDA TransFusion buffers");
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_value);work_views v=views(d);size_t p=d->proposals,k=d->keys;
    stage_profiler profiler(stream);
    suppress_kernel<<<(d->candidates+255)/256,256,0,stream>>>(heatmap,v.suppressed,(int)d->height,(int)d->width);
    if(!cuda_ok(cudaMemcpyAsync(v.sort_a,v.suppressed,d->candidates*sizeof(float),
        cudaMemcpyDeviceToDevice,stream),error,cap,"copy suppressed proposal scores"))return 0;
    iota_kernel<<<(d->candidates+255)/256,256,0,stream>>>(v.index_a,d->candidates);
    if(cub::DeviceRadixSort::SortPairsDescending(d->cub_workspace,d->cub_bytes,v.sort_a,v.sort_b,v.index_a,v.index_b,d->candidates,0,32,stream)!=cudaSuccess)
        return fail(error,cap,"CUB proposal sort failed");
    select_kernel<<<p,128,0,stream>>>(v.index_b,v.suppressed,shared,d->class_encoding.weight,d->class_encoding.bias,v.query,
        out->query_labels_bp,out->query_indices_bp,out->query_heatmap_scores_b10p,(int)k,(int)p);
    profiler.mark(1,stream);
    position_first_kernel<<<p,128,0,stream>>>(v.query_pos_hidden,d->self_position.first.weight,d->self_position.first.bias,
        out->query_indices_bp,(int)p,(int)d->width);
    if(!linear(d,d->self_position.second,v.query_pos_hidden,v.query_pos,(int)p,stream,error,cap))return 0;
    add_kernel<<<(p*CHANNELS+255)/256,256,0,stream>>>(v.query,v.query_pos,v.temporary,p*CHANNELS,false);
    if(!linear(d,d->self_q,v.temporary,v.q_proj,(int)p,stream,error,cap)||!linear(d,d->self_k,v.temporary,v.key_proj,(int)p,stream,error,cap)||
       !linear(d,d->self_v,v.temporary,v.value_proj,(int)p,stream,error,cap))return 0;
    launch_attention(v.q_proj,v.key_proj,v.value_proj,v.attention,(int)p,(int)p,stream);
    if(!linear(d,d->self_out,v.attention,v.temporary,(int)p,stream,error,cap))return 0;
    layer_norm_kernel<<<p,128,0,stream>>>(v.query,v.temporary,v.query,d->norm_scale[0],d->norm_bias[0],(int)p);
    profiler.mark(2,stream);
    position_first_kernel<<<k,128,0,stream>>>(v.key_tmp,d->cross_position.first.weight,d->cross_position.first.bias,nullptr,(int)k,(int)d->width);
    if(!linear(d,d->cross_position.second,v.key_tmp,v.key_value,(int)k,stream,error,cap))return 0;
    add_shared_kernel<<<(k*CHANNELS+255)/256,256,0,stream>>>(v.key_value,shared,(int)k);
    add_kernel<<<(p*CHANNELS+255)/256,256,0,stream>>>(v.query,v.query_pos,v.temporary,p*CHANNELS,false);
    if(!linear(d,d->cross_q,v.temporary,v.q_proj,(int)p,stream,error,cap)||!linear(d,d->cross_k,v.key_value,v.key_proj,(int)k,stream,error,cap)||
       !linear(d,d->cross_v,v.key_value,v.value_proj,(int)k,stream,error,cap))return 0;
    profiler.mark(3,stream);
    launch_attention(v.q_proj,v.key_proj,v.value_proj,v.attention,(int)p,(int)k,stream);
    if(!linear(d,d->cross_out,v.attention,v.temporary,(int)p,stream,error,cap))return 0;
    layer_norm_kernel<<<p,128,0,stream>>>(v.query,v.temporary,v.query,d->norm_scale[1],d->norm_bias[1],(int)p);
    profiler.mark(4,stream);
    if(!linear(d,d->ffn1,v.query,v.ffn,(int)p,stream,error,cap))return 0;
    relu_kernel<<<(p*256+255)/256,256,0,stream>>>(v.ffn,p*256);
    if(!linear(d,d->ffn2,v.ffn,v.temporary,(int)p,stream,error,cap))return 0;
    layer_norm_kernel<<<p,128,0,stream>>>(v.query,v.temporary,v.query,d->norm_scale[2],d->norm_bias[2],(int)p);
    profiler.mark(5,stream);
    float *outputs[6]={out->center_b2p,out->height_b1p,out->dimension_log_b3p,out->rotation_sincos_b2p,out->velocity_b2p,out->heatmap_logits_b10p};
    for(int head=0;head<6;++head){if(!linear(d,d->heads[head].hidden,v.query,v.head_hidden,(int)p,stream,error,cap))return 0;
        relu_kernel<<<(p*64+255)/256,256,0,stream>>>(v.head_hidden,p*64);
        if(!linear(d,d->heads[head].output,v.head_hidden,v.head_rows,(int)p,stream,error,cap))return 0;
        transpose_head_kernel<<<p,16,0,stream>>>(v.head_rows,outputs[head],(int)p,d->heads[head].channels,head==0,out->query_indices_bp,(int)d->width);}
    profiler.mark(6,stream);profiler.finish();
    return cuda_ok(cudaGetLastError(),error,cap,"launch CUDA TransFusion graph");}

extern "C" size_t bf_cuda_transfusion_resident_bytes(const bf_cuda_transfusion *d){return d?d->resident_bytes:0;}

__global__ static void decode_detections_kernel(
    const float *center,const float *height,const float *dimension,const float *rotation,
    const float *velocity,const float *heatmap,const float *query_scores,
    const int64_t *labels,int proposals,float threshold,bf_detections *output){
    int p=threadIdx.x;__shared__ int offsets[BF_MAX_PROPOSALS];bf_detection detection={};int keep=0;
    if(p<proposals){int label=(int)labels[p];float logit=heatmap[(size_t)label*proposals+p];
        float score=(1.0f/(1.0f+expf(-logit)))*query_scores[(size_t)label*proposals+p];
        detection.x=center[p]*0.6f-54.0f;detection.y=center[proposals+p]*0.6f-54.0f;
        detection.z=height[p];detection.width=expf(dimension[p]);
        detection.length=expf(dimension[proposals+p]);detection.height=expf(dimension[2*proposals+p]);
        detection.yaw=atan2f(rotation[p],rotation[proposals+p]);
        detection.velocity_x=velocity[p];detection.velocity_y=velocity[proposals+p];
        detection.score=score;detection.class_id=label;
        keep=isfinite(score)&&score>threshold&&isfinite(detection.x)&&isfinite(detection.y)&&isfinite(detection.z)&&
            detection.x>=-61.2f&&detection.x<=61.2f&&detection.y>=-61.2f&&detection.y<=61.2f&&
            detection.z>=-10.0f&&detection.z<=10.0f;offsets[p]=keep;}
    __syncthreads();if(p==0){int count=0;for(int i=0;i<proposals;++i){if(offsets[i])offsets[i]=count++;else offsets[i]=-1;}output->count=count;}
    __syncthreads();if(p<proposals&&offsets[p]>=0)output->items[offsets[p]]=detection;
}

extern "C" int bf_cuda_transfusion_decode_detections(bf_cuda_transfusion *d,
    const bf_transfusion_raw_outputs *out,float threshold,bf_detections *host,
    void *stream_value,char *error,size_t cap){
    if(!d||!out||!host||!std::isfinite(threshold)||!out->center_b2p||!out->height_b1p||
       !out->dimension_log_b3p||!out->rotation_sincos_b2p||!out->velocity_b2p||
       !out->heatmap_logits_b10p||!out->query_heatmap_scores_b10p||!out->query_labels_bp)
        return fail(error,cap,"invalid CUDA detection decode contract");
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_value);
    decode_detections_kernel<<<1,256,0,stream>>>(out->center_b2p,out->height_b1p,
        out->dimension_log_b3p,out->rotation_sincos_b2p,out->velocity_b2p,
        out->heatmap_logits_b10p,out->query_heatmap_scores_b10p,out->query_labels_bp,
        (int)d->proposals,threshold,d->device_detections);
    if(!cuda_ok(cudaGetLastError(),error,cap,"launch CUDA canonical decode")||
       !cuda_ok(cudaMemcpyAsync(host,d->device_detections,sizeof(*host),cudaMemcpyDeviceToHost,stream),
                error,cap,"copy canonical detections")||
       !cuda_ok(cudaStreamSynchronize(stream),error,cap,"synchronize canonical detections"))return 0;
    return host->count>=0&&host->count<=BF_MAX_PROPOSALS;
}
