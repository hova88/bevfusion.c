#include "bf_cuda_swin.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cfloat>

struct swin_linear { float *weight,*bias; int input,output; size_t bytes; };
struct swin_block {
    float *norm1_scale,*norm1_bias,*relative_bias;
    long long *relative_index;
    swin_linear qkv,projection;
    float *norm2_scale,*norm2_bias;
    swin_linear ffn1,ffn2;
};
struct swin_merge { float *scale,*bias; swin_linear reduction; };

struct bf_cuda_swin {
    size_t batches,input_h,input_w,resident_bytes;
    int h[4],w[4],channels[4],heads[4],depths[4];
    float *patch_weight,*patch_bias,*patch_scale,*patch_norm_bias;
    swin_block blocks[4][6]; swin_merge merges[3];
    float *output_scale[3],*output_bias[3];
    float *current_buffer,*normal_buffer,*ffn_buffer,*windows,*qkv_buffer;
    size_t token_capacity,ffn_capacity,window_capacity;
    size_t ffn_chunk_rows,window_chunk;
    void *cudnn_workspace;size_t cudnn_workspace_bytes;
    cudnnHandle_t cudnn;cublasHandle_t cublas;
    cudnnTensorDescriptor_t patch_input_desc,patch_output_desc,patch_bias_desc;
    cudnnFilterDescriptor_t patch_filter_desc;
    cudnnConvolutionDescriptor_t patch_conv_desc;
    cudnnActivationDescriptor_t patch_activation_desc;
    cudnnConvolutionFwdAlgo_t patch_algorithm;
};

static int fail(char *error,size_t cap,const char *format,...) {
    if(error&&cap){va_list args;va_start(args,format);std::vsnprintf(error,cap,format,args);va_end(args);}return 0;
}
static int cuda_ok(cudaError_t s,char *e,size_t c,const char *w){return s==cudaSuccess?1:fail(e,c,"%s: %s",w,cudaGetErrorString(s));}
static int cudnn_ok(cudnnStatus_t s,char *e,size_t c,const char *w){return s==CUDNN_STATUS_SUCCESS?1:fail(e,c,"%s: %s",w,cudnnGetErrorString(s));}
static int blas_ok(cublasStatus_t s,char *e,size_t c,const char *w){return s==CUBLAS_STATUS_SUCCESS?1:fail(e,c,"%s: cuBLAS status %d",w,(int)s);}

static const void *tensor_data(const bf_model *model,const char *name,uint32_t dtype,
    uint32_t rank,const uint32_t *dims,char *error,size_t cap) {
    const bf_tensor *t=bf_model_find(model,name);
    if(!t||t->dtype!=dtype||t->rank!=rank)return fail(error,cap,"%s contract mismatch",name),nullptr;
    for(uint32_t i=0;i<rank;++i)if(t->dims[i]!=dims[i])return fail(error,cap,"%s shape mismatch",name),nullptr;
    return t->data;
}
static int upload_raw(void **device,const void *host,size_t bytes,size_t *resident,
    char *error,size_t cap) {
    if(!cuda_ok(cudaMalloc(device,bytes),error,cap,"allocate Swin parameter")||
       !cuda_ok(cudaMemcpy(*device,host,bytes,cudaMemcpyHostToDevice),error,cap,"upload Swin parameter"))return 0;
    *resident+=bytes;return 1;
}
static int bind_float(bf_cuda_swin *s,const bf_model *m,const char *name,uint32_t rank,
    const uint32_t *dims,float **out,char *error,size_t cap) {
    const float *sour=(const float*)tensor_data(m,name,BF_DTYPE_F32,rank,dims,error,cap);
    size_t count=1;for(uint32_t i=0;i<rank;++i)count*=dims[i];
    return sour&&upload_raw((void**)out,sour,count*sizeof(float),&s->resident_bytes,error,cap);
}
static int bind_i64(bf_cuda_swin *s,const bf_model *m,const char *name,uint32_t rank,
    const uint32_t *dims,long long **out,char *error,size_t cap) {
    const void *source=tensor_data(m,name,BF_DTYPE_I64,rank,dims,error,cap);
    size_t count=1;for(uint32_t i=0;i<rank;++i)count*=dims[i];
    return source&&upload_raw((void**)out,source,count*sizeof(long long),&s->resident_bytes,error,cap);
}
static int bind_linear(bf_cuda_swin *s,const bf_model *m,const char *weight_name,
    const char *bias_name,int input,int output,swin_linear *linear,char *error,size_t cap) {
    uint32_t wd[2]={(uint32_t)output,(uint32_t)input};
    if(!bind_float(s,m,weight_name,2,wd,&linear->weight,error,cap))return 0;
    if(bias_name){uint32_t bd[1]={(uint32_t)output};if(!bind_float(s,m,bias_name,1,bd,&linear->bias,error,cap))return 0;}
    linear->input=input;linear->output=output;linear->bytes=(size_t)input*output*sizeof(float)+(bias_name?(size_t)output*sizeof(float):0);return 1;
}

__global__ static void bias_kernel(float *values,const float *bias,size_t count,int columns) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)values[i]+=bias[i%columns];
}
__global__ static void layer_norm_kernel(const float *input,float *output,
    const float *scale,const float *bias,int rows,int channels) {
    int row=blockIdx.x;__shared__ float sums[256],squares[256];
    float sum=0.0f,square=0.0f;
    for(int c=threadIdx.x;c<channels;c+=blockDim.x){float v=input[(size_t)row*channels+c];sum+=v;square+=v*v;}
    sums[threadIdx.x]=sum;squares[threadIdx.x]=square;__syncthreads();
    for(int stride=128;stride;stride>>=1){if(threadIdx.x<stride){sums[threadIdx.x]+=sums[threadIdx.x+stride];squares[threadIdx.x]+=squares[threadIdx.x+stride];}__syncthreads();}
    float mean=sums[0]/channels;float variance=fmaxf(0.0f,squares[0]/channels-mean*mean);float inverse=rsqrtf(variance+1e-5f);
    for(int c=threadIdx.x;c<channels;c+=blockDim.x){float v=input[(size_t)row*channels+c];output[(size_t)row*channels+c]=(v-mean)*inverse*scale[c]+bias[c];}
}
__global__ static void patch_transpose_norm_kernel(const float *input,float *output,
    const float *scale,const float *bias,int batches,int height,int width) {
    int row=blockIdx.x;__shared__ float values[96],sums[128],squares[128];int lane=threadIdx.x;
    int position=row%(height*width),batch=row/(height*width);float v=0.0f;
    if(lane<96){v=input[((size_t)batch*96+lane)*height*width+position];values[lane]=v;}
    sums[lane]=lane<96?v:0.0f;squares[lane]=lane<96?v*v:0.0f;__syncthreads();
    for(int stride=64;stride;stride>>=1){if(lane<stride){sums[lane]+=sums[lane+stride];squares[lane]+=squares[lane+stride];}__syncthreads();}
    if(lane<96){float mean=sums[0]/96.0f,var=fmaxf(0.0f,squares[0]/96.0f-mean*mean);output[(size_t)row*96+lane]=(values[lane]-mean)*rsqrtf(var+1e-5f)*scale[lane]+bias[lane];}
}
__global__ static void add_kernel(float *current,const float *update,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)current[i]+=update[i];
}
__global__ static void gelu_kernel(float *values,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float x=values[i];values[i]=0.5f*x*(1.0f+erff(x*0.7071067811865475244f));}
}
__global__ static void gather_windows_kernel(const float *tokens,float *windows,
    int batches,int height,int width,int channels,int windows_y,int windows_x,
    int window_offset,int window_count,int shift) {
    size_t total=(size_t)window_count*49*channels;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
    int channel=i%channels;size_t q=i/channels;int token=q%49;q/=49;
    int window=(int)q+window_offset;int wx=window%windows_x;window/=windows_x;
    int wy=window%windows_y;int batch=window/windows_y;
    int py=wy*7+token/7,px=wx*7+token%7;int padded_h=windows_y*7,padded_w=windows_x*7;
    int sy=(py+shift)%padded_h,sx=(px+shift)%padded_w;
    windows[i]=(sy<height&&sx<width)?tokens[((size_t)batch*height*width+sy*width+sx)*channels+channel]:0.0f;
}
__device__ static int mask_region_device(int position,int extent,int shift) {
    if(position<extent-7)return 0;if(position<extent-shift)return 1;return 2;
}
__global__ static void window_attention_kernel(const float *qkv,float *output,
    const float *relative_bias,const long long *relative_index,int total_windows,
    int channels,int heads,int windows_y,int windows_x,int window_offset,int shift) {
    int item=blockIdx.x,query=item%49;item/=49;int head=item%heads;int window=item/heads;
    if(window>=total_windows)return;int lane=threadIdx.x;int head_base=head*32;
    const float scale=0.1767766952966369f;float qv=qkv[((size_t)window*49+query)*3*channels+head_base+lane];
    float accumulator=0.0f,maximum=-FLT_MAX,denominator=0.0f;
    int spatial_window=(window+window_offset)%(windows_y*windows_x),wy=spatial_window/windows_x,wx=spatial_window%windows_x;
    int qy=query/7,qx=query%7;int qlabel=0;
    if(shift)qlabel=mask_region_device(wy*7+qy,windows_y*7,shift)*3+mask_region_device(wx*7+qx,windows_x*7,shift);
    for(int key=0;key<49;++key){float dot=qv*qkv[((size_t)window*49+key)*3*channels+channels+head_base+lane];
        for(int offset=16;offset;offset>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,offset);
        float alpha=1.0f,beta=0.0f;
        if(lane==0){float score=dot*scale+relative_bias[(size_t)relative_index[query*49+key]*heads+head];
            if(shift){int ky=key/7,kx=key%7;int label=mask_region_device(wy*7+ky,windows_y*7,shift)*3+mask_region_device(wx*7+kx,windows_x*7,shift);if(label!=qlabel)score-=100.0f;}
            float next=fmaxf(maximum,score);alpha=expf(maximum-next);beta=expf(score-next);denominator=denominator*alpha+beta;maximum=next;}
        alpha=__shfl_sync(0xffffffffu,alpha,0);beta=__shfl_sync(0xffffffffu,beta,0);
        accumulator=accumulator*alpha+beta*qkv[((size_t)window*49+key)*3*channels+2*channels+head_base+lane];
    }
    denominator=__shfl_sync(0xffffffffu,denominator,0);
    output[((size_t)window*49+query)*channels+head_base+lane]=accumulator/denominator;
}
template <int QueriesPerBlock>
__global__ static void window_attention_tiled_kernel(const float *qkv,float *output,
    const float *relative_bias,const long long *relative_index,int total_windows,
    int channels,int heads,int windows_y,int windows_x,int window_offset,int shift) {
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,item=blockIdx.x;
    constexpr int groups=(49+QueriesPerBlock-1)/QueriesPerBlock;
    int group=item%groups;item/=groups;int head=item%heads;int window=item/heads;
    int query=group*QueriesPerBlock+warp;
    if(window>=total_windows||query>=49)return;
    int head_base=head*32;const float scale=0.1767766952966369f;
    float qv=qkv[((size_t)window*49+query)*3*channels+head_base+lane];
    float accumulator=0.0f,maximum=-FLT_MAX,denominator=0.0f;
    int spatial_window=(window+window_offset)%(windows_y*windows_x),wy=spatial_window/windows_x,wx=spatial_window%windows_x;
    int qy=query/7,qx=query%7,qlabel=0;
    if(shift)qlabel=mask_region_device(wy*7+qy,windows_y*7,shift)*3+mask_region_device(wx*7+qx,windows_x*7,shift);
    for(int key=0;key<49;++key){float dot=qv*qkv[((size_t)window*49+key)*3*channels+channels+head_base+lane];
        for(int offset=16;offset;offset>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,offset);
        float alpha=1.0f,beta=0.0f;if(lane==0){float score=dot*scale+relative_bias[(size_t)relative_index[query*49+key]*heads+head];
            if(shift){int ky=key/7,kx=key%7;int label=mask_region_device(wy*7+ky,windows_y*7,shift)*3+mask_region_device(wx*7+kx,windows_x*7,shift);if(label!=qlabel)score-=100.0f;}
            float next=fmaxf(maximum,score);alpha=expf(maximum-next);beta=expf(score-next);denominator=denominator*alpha+beta;maximum=next;}
        alpha=__shfl_sync(0xffffffffu,alpha,0);beta=__shfl_sync(0xffffffffu,beta,0);
        accumulator=accumulator*alpha+beta*qkv[((size_t)window*49+key)*3*channels+2*channels+head_base+lane];}
    denominator=__shfl_sync(0xffffffffu,denominator,0);
    output[((size_t)window*49+query)*channels+head_base+lane]=accumulator/denominator;
}
__global__ static void window_attention_shared_kernel(const float *qkv,float *output,
    const float *relative_bias,const long long *relative_index,int total_windows,
    int channels,int heads,int windows_y,int windows_x,int window_offset,int shift) {
    int item=blockIdx.x,head=item%heads,window=item/heads;
    if(window>=total_windows)return;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    extern __shared__ float keys_values[];int head_base=head*32;
    for(int i=threadIdx.x;i<49*64;i+=blockDim.x){int key=i/64,component=i%64;
        int offset=component<32?channels+head_base+component:2*channels+head_base+component-32;
        keys_values[i]=qkv[((size_t)window*49+key)*3*channels+offset];}
    __syncthreads();const float scale=0.1767766952966369f;
    int spatial_window=(window+window_offset)%(windows_y*windows_x),wy=spatial_window/windows_x,wx=spatial_window%windows_x;
    int warps=blockDim.x>>5;
    for(int query=warp;query<49;query+=warps){float qv=qkv[((size_t)window*49+query)*3*channels+head_base+lane];
        float accumulator=0.0f,maximum=-FLT_MAX,denominator=0.0f;int qy=query/7,qx=query%7,qlabel=0;
        if(shift)qlabel=mask_region_device(wy*7+qy,windows_y*7,shift)*3+mask_region_device(wx*7+qx,windows_x*7,shift);
        for(int key=0;key<49;++key){float dot=qv*keys_values[key*64+lane];
            for(int offset=16;offset;offset>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,offset);
            float alpha=1.0f,beta=0.0f;if(lane==0){float score=dot*scale+relative_bias[(size_t)relative_index[query*49+key]*heads+head];
                if(shift){int ky=key/7,kx=key%7;int label=mask_region_device(wy*7+ky,windows_y*7,shift)*3+mask_region_device(wx*7+kx,windows_x*7,shift);if(label!=qlabel)score-=100.0f;}
                float next=fmaxf(maximum,score);alpha=expf(maximum-next);beta=expf(score-next);denominator=denominator*alpha+beta;maximum=next;}
            alpha=__shfl_sync(0xffffffffu,alpha,0);beta=__shfl_sync(0xffffffffu,beta,0);
            accumulator=accumulator*alpha+beta*keys_values[key*64+32+lane];}
        denominator=__shfl_sync(0xffffffffu,denominator,0);output[((size_t)window*49+query)*channels+head_base+lane]=accumulator/denominator;}
}
__global__ static void scatter_windows_kernel(const float *windows,float *tokens,
    int batches,int height,int width,int channels,int windows_y,int windows_x,
    int window_offset,int window_count,int shift) {
    size_t total=(size_t)window_count*49*channels;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
    int channel=i%channels;size_t q=i/channels;int token=q%49;q/=49;
    int window=(int)q+window_offset;int wx=window%windows_x;window/=windows_x;
    int wy=window%windows_y;int batch=window/windows_y;
    int py=wy*7+token/7,px=wx*7+token%7;int padded_h=windows_y*7,padded_w=windows_x*7;
    int dy=(py+shift)%padded_h,dx=(px+shift)%padded_w;
    if(dy<height&&dx<width)tokens[((size_t)batch*height*width+dy*width+dx)*channels+channel]=windows[i];
}
__global__ static void patch_merge_kernel(const float *input,float *output,
    int batches,int height,int width,int channels) {
    int out_h=(height+1)/2,out_w=(width+1)/2;size_t total=(size_t)batches*out_h*out_w*4*channels;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
    int element=i%(4*channels);size_t row=i/(4*channels);int ox=row%out_w;size_t q=row/out_w;int oy=q%out_h;int batch=q/out_h;
    int channel=element/4,corner=element%4,y=2*oy+corner/2,x=2*ox+corner%2;
    output[i]=(y<height&&x<width)?input[((size_t)batch*height*width+y*width+x)*channels+channel]:0.0f;
}
__global__ static void output_transpose_kernel(const float *tokens,float *output,
    size_t count,int channels,int spatial) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    int position=i%spatial;size_t q=i/spatial;int channel=q%channels;int batch=q/channels;
    output[i]=tokens[((size_t)batch*spatial+position)*channels+channel];
}

static int linear(bf_cuda_swin *s,const swin_linear &l,const float *input,
    float *output,int rows,cudaStream_t stream,char *error,size_t cap) {
    if(!blas_ok(cublasSetStream(s->cublas,stream),error,cap,"set Swin cuBLAS stream"))return 0;
    const float one=1.0f,zero=0.0f;
    if(!blas_ok(cublasSgemm(s->cublas,CUBLAS_OP_T,CUBLAS_OP_N,l.output,rows,l.input,
        &one,l.weight,l.input,input,l.input,&zero,output,l.output),error,cap,"Swin GEMM"))return 0;
    if(l.bias)bias_kernel<<<((size_t)rows*l.output+255)/256,256,0,stream>>>(output,l.bias,(size_t)rows*l.output,l.output);
    return cuda_ok(cudaGetLastError(),error,cap,"Swin linear launch");
}

extern "C" void bf_cuda_swin_destroy(bf_cuda_swin *s) {
    if(!s)return;
#define FREE(P) cudaFree(P)
    FREE(s->patch_weight);FREE(s->patch_bias);FREE(s->patch_scale);FREE(s->patch_norm_bias);
    for(int st=0;st<4;++st)for(int b=0;b<s->depths[st];++b){swin_block &x=s->blocks[st][b];
        FREE(x.norm1_scale);FREE(x.norm1_bias);FREE(x.relative_bias);FREE(x.relative_index);
        FREE(x.qkv.weight);FREE(x.qkv.bias);FREE(x.projection.weight);FREE(x.projection.bias);
        FREE(x.norm2_scale);FREE(x.norm2_bias);FREE(x.ffn1.weight);FREE(x.ffn1.bias);FREE(x.ffn2.weight);FREE(x.ffn2.bias);}
    for(int i=0;i<3;++i){FREE(s->merges[i].scale);FREE(s->merges[i].bias);FREE(s->merges[i].reduction.weight);FREE(s->output_scale[i]);FREE(s->output_bias[i]);}
    FREE(s->current_buffer);FREE(s->normal_buffer);FREE(s->ffn_buffer);FREE(s->windows);FREE(s->qkv_buffer);FREE(s->cudnn_workspace);
#undef FREE
    if(s->patch_input_desc)cudnnDestroyTensorDescriptor(s->patch_input_desc);if(s->patch_output_desc)cudnnDestroyTensorDescriptor(s->patch_output_desc);
    if(s->patch_bias_desc)cudnnDestroyTensorDescriptor(s->patch_bias_desc);if(s->patch_filter_desc)cudnnDestroyFilterDescriptor(s->patch_filter_desc);
    if(s->patch_conv_desc)cudnnDestroyConvolutionDescriptor(s->patch_conv_desc);if(s->patch_activation_desc)cudnnDestroyActivationDescriptor(s->patch_activation_desc);
    if(s->cudnn)cudnnDestroy(s->cudnn);if(s->cublas)cublasDestroy(s->cublas);std::free(s);
}

extern "C" int bf_cuda_swin_create(const bf_model *model,size_t batches,size_t input_h,
    size_t input_w,bf_cuda_swin **out,char *error,size_t cap) {
    if(out)*out=nullptr;if(!model||!out||!batches||!input_h||!input_w||(input_h%4)||(input_w%4)||
       batches>INT_MAX||input_h>INT_MAX||input_w>INT_MAX)return fail(error,cap,"invalid CUDA Swin contract");
    bf_cuda_swin *s=(bf_cuda_swin*)std::calloc(1,sizeof(*s));if(!s)return fail(error,cap,"Swin host allocation failed");
    cudnnStatus_t cs=CUDNN_STATUS_SUCCESS;
    const char *chunk_text=nullptr;
    const char *window_text=nullptr;
    s->batches=batches;s->input_h=input_h;s->input_w=input_w;
    int channels[4]={96,192,384,768},heads[4]={3,6,12,24},depths[4]={2,2,6,2};
    s->h[0]=input_h/4;s->w[0]=input_w/4;for(int i=1;i<4;++i){s->h[i]=(s->h[i-1]+1)/2;s->w[i]=(s->w[i-1]+1)/2;}
    std::memcpy(s->channels,channels,sizeof(channels));std::memcpy(s->heads,heads,sizeof(heads));std::memcpy(s->depths,depths,sizeof(depths));
    if(!cudnn_ok(cudnnCreate(&s->cudnn),error,cap,"create Swin cuDNN")||!blas_ok(cublasCreate(&s->cublas),error,cap,"create Swin cuBLAS")||
       !blas_ok(cublasSetMathMode(s->cublas,std::getenv("BF_CUDA_SWIN_TF32")?
            CUBLAS_TF32_TENSOR_OP_MATH:CUBLAS_PEDANTIC_MATH),error,cap,"set Swin math mode")||
       !blas_ok(cublasSetAtomicsMode(s->cublas,CUBLAS_ATOMICS_NOT_ALLOWED),error,cap,"disable Swin atomics"))goto failure;
    {uint32_t wd[4]={96,3,4,4},d[1]={96};
     if(!bind_float(s,model,"image_backbone.patch_embed.projection.weight",4,wd,&s->patch_weight,error,cap)||
        !bind_float(s,model,"image_backbone.patch_embed.projection.bias",1,d,&s->patch_bias,error,cap)||
        !bind_float(s,model,"image_backbone.patch_embed.norm.weight",1,d,&s->patch_scale,error,cap)||
        !bind_float(s,model,"image_backbone.patch_embed.norm.bias",1,d,&s->patch_norm_bias,error,cap))goto failure;}
    for(int st=0;st<4;++st)for(int b=0;b<depths[st];++b){swin_block &x=s->blocks[st][b];int c=channels[st],h=heads[st];char prefix[160],name[224];
        std::snprintf(prefix,sizeof(prefix),"image_backbone.stages.%d.blocks.%d",st,b);uint32_t dc[1]={(uint32_t)c},rel[2]={169,(uint32_t)h},idx[2]={49,49};
#define BF_BIND(F,SFX,R,D) do{std::snprintf(name,sizeof(name),"%s.%s",prefix,SFX);if(!bind_float(s,model,name,R,D,&x.F,error,cap))goto failure;}while(0)
        BF_BIND(norm1_scale,"norm1.weight",1,dc);BF_BIND(norm1_bias,"norm1.bias",1,dc);
        BF_BIND(relative_bias,"attn.w_msa.relative_position_bias_table",2,rel);
        std::snprintf(name,sizeof(name),"%s.attn.w_msa.relative_position_index",prefix);if(!bind_i64(s,model,name,2,idx,&x.relative_index,error,cap))goto failure;
        char wname[224],bname[224];std::snprintf(wname,sizeof(wname),"%s.attn.w_msa.qkv.weight",prefix);std::snprintf(bname,sizeof(bname),"%s.attn.w_msa.qkv.bias",prefix);
        if(!bind_linear(s,model,wname,bname,c,3*c,&x.qkv,error,cap))goto failure;
        std::snprintf(wname,sizeof(wname),"%s.attn.w_msa.proj.weight",prefix);std::snprintf(bname,sizeof(bname),"%s.attn.w_msa.proj.bias",prefix);
        if(!bind_linear(s,model,wname,bname,c,c,&x.projection,error,cap))goto failure;
        BF_BIND(norm2_scale,"norm2.weight",1,dc);BF_BIND(norm2_bias,"norm2.bias",1,dc);
        std::snprintf(wname,sizeof(wname),"%s.ffn.layers.0.0.weight",prefix);std::snprintf(bname,sizeof(bname),"%s.ffn.layers.0.0.bias",prefix);
        if(!bind_linear(s,model,wname,bname,c,4*c,&x.ffn1,error,cap))goto failure;
        std::snprintf(wname,sizeof(wname),"%s.ffn.layers.1.weight",prefix);std::snprintf(bname,sizeof(bname),"%s.ffn.layers.1.bias",prefix);
        if(!bind_linear(s,model,wname,bname,4*c,c,&x.ffn2,error,cap))goto failure;
#undef BF_BIND
    }
    for(int st=0;st<3;++st){int c=channels[st];uint32_t d4[1]={(uint32_t)(4*c)};char name[192];
        std::snprintf(name,sizeof(name),"image_backbone.stages.%d.downsample.norm.weight",st);if(!bind_float(s,model,name,1,d4,&s->merges[st].scale,error,cap))goto failure;
        std::snprintf(name,sizeof(name),"image_backbone.stages.%d.downsample.norm.bias",st);if(!bind_float(s,model,name,1,d4,&s->merges[st].bias,error,cap))goto failure;
        std::snprintf(name,sizeof(name),"image_backbone.stages.%d.downsample.reduction.weight",st);if(!bind_linear(s,model,name,nullptr,4*c,2*c,&s->merges[st].reduction,error,cap))goto failure;
        uint32_t dc[1]={(uint32_t)channels[st+1]};std::snprintf(name,sizeof(name),"image_backbone.norm%d.weight",st+1);if(!bind_float(s,model,name,1,dc,&s->output_scale[st],error,cap))goto failure;
        std::snprintf(name,sizeof(name),"image_backbone.norm%d.bias",st+1);if(!bind_float(s,model,name,1,dc,&s->output_bias[st],error,cap))goto failure;}
    s->token_capacity=batches*(size_t)s->h[0]*s->w[0]*96;
    s->ffn_chunk_rows=4096;
    chunk_text=std::getenv("BF_CUDA_SWIN_FFN_CHUNK_ROWS");
    if(chunk_text){char *end=nullptr;unsigned long long value=std::strtoull(chunk_text,&end,10);if(end&&!*end&&value)s->ffn_chunk_rows=value;}
    s->window_chunk=256;window_text=std::getenv("BF_CUDA_SWIN_WINDOW_CHUNK");
    if(window_text){char *end=nullptr;unsigned long long value=std::strtoull(window_text,&end,10);if(end&&!*end&&value)s->window_chunk=value;}
    s->ffn_capacity=s->token_capacity;s->window_capacity=0;
    for(int st=0;st<4;++st){size_t rows=batches*(size_t)s->h[st]*s->w[st],chunk=rows<s->ffn_chunk_rows?rows:s->ffn_chunk_rows;size_t values=chunk*4*channels[st];if(values>s->ffn_capacity)s->ffn_capacity=values;}
    for(int st=0;st<4;++st){size_t window_count=batches*(size_t)((s->h[st]+6)/7)*((s->w[st]+6)/7);if(window_count>s->window_chunk)window_count=s->window_chunk;size_t win=window_count*49*channels[st];if(win>s->window_capacity)s->window_capacity=win;}
#define ALLOC(P,N,L) if(!cuda_ok(cudaMalloc(&s->P,(N)*sizeof(float)),error,cap,L))goto failure;else s->resident_bytes+=(N)*sizeof(float)
    ALLOC(current_buffer,s->token_capacity,"allocate Swin current");ALLOC(normal_buffer,s->token_capacity,"allocate Swin normal");
    ALLOC(ffn_buffer,s->ffn_capacity,"allocate Swin FFN");ALLOC(windows,s->window_capacity,"allocate Swin windows");
    ALLOC(qkv_buffer,3*s->window_capacity,"allocate Swin QKV");
#undef ALLOC
    cs=cudnnCreateTensorDescriptor(&s->patch_input_desc);if(cs==CUDNN_STATUS_SUCCESS)cs=cudnnCreateTensorDescriptor(&s->patch_output_desc);
    if(cs==CUDNN_STATUS_SUCCESS)cs=cudnnCreateTensorDescriptor(&s->patch_bias_desc);if(cs==CUDNN_STATUS_SUCCESS)cs=cudnnCreateFilterDescriptor(&s->patch_filter_desc);
    if(cs==CUDNN_STATUS_SUCCESS)cs=cudnnCreateConvolutionDescriptor(&s->patch_conv_desc);if(cs==CUDNN_STATUS_SUCCESS)cs=cudnnCreateActivationDescriptor(&s->patch_activation_desc);
    if(!cudnn_ok(cs,error,cap,"create Swin patch descriptors")||
       !cudnn_ok(cudnnSetTensor4dDescriptor(s->patch_input_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batches,3,input_h,input_w),error,cap,"set patch input")||
       !cudnn_ok(cudnnSetTensor4dDescriptor(s->patch_output_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batches,96,s->h[0],s->w[0]),error,cap,"set patch output")||
       !cudnn_ok(cudnnSetTensor4dDescriptor(s->patch_bias_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,96,1,1),error,cap,"set patch bias")||
       !cudnn_ok(cudnnSetFilter4dDescriptor(s->patch_filter_desc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,96,3,4,4),error,cap,"set patch filter")||
       !cudnn_ok(cudnnSetConvolution2dDescriptor(s->patch_conv_desc,0,0,4,4,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT),error,cap,"set patch convolution")||
       !cudnn_ok(cudnnSetConvolutionMathType(s->patch_conv_desc,CUDNN_FMA_MATH),error,cap,"set patch math")||
       !cudnn_ok(cudnnSetActivationDescriptor(s->patch_activation_desc,CUDNN_ACTIVATION_IDENTITY,CUDNN_NOT_PROPAGATE_NAN,0),error,cap,"set patch activation"))goto failure;
    {cudnnConvolutionFwdAlgoPerf_t perf[8];int returned=0,selected=-1;if(!cudnn_ok(cudnnGetConvolutionForwardAlgorithm_v7(s->cudnn,s->patch_input_desc,s->patch_filter_desc,s->patch_conv_desc,s->patch_output_desc,8,&returned,perf),error,cap,"select patch algorithm"))goto failure;
     for(int i=0;i<returned;++i)if(perf[i].status==CUDNN_STATUS_SUCCESS&&perf[i].determinism==CUDNN_DETERMINISTIC&&perf[i].memory<=16ull*1024*1024){selected=i;break;}
     if(selected<0){fail(error,cap,"no bounded deterministic patch algorithm");goto failure;}s->patch_algorithm=perf[selected].algo;
     if(!cudnn_ok(cudnnGetConvolutionForwardWorkspaceSize(s->cudnn,s->patch_input_desc,s->patch_filter_desc,s->patch_conv_desc,s->patch_output_desc,s->patch_algorithm,&s->cudnn_workspace_bytes),error,cap,"query patch workspace"))goto failure;
     if(s->cudnn_workspace_bytes&&!cuda_ok(cudaMalloc(&s->cudnn_workspace,s->cudnn_workspace_bytes),error,cap,"allocate patch workspace"))goto failure;s->resident_bytes+=s->cudnn_workspace_bytes;}
    *out=s;return 1;
failure:bf_cuda_swin_destroy(s);return 0;
}

extern "C" int bf_cuda_swin_forward(bf_cuda_swin *s,const float *images,float *out1,
    float *out2,float *out3,void *stream_value,char *error,size_t cap) {
    if(!s||!images||!out1||!out2||!out3)return fail(error,cap,"invalid CUDA Swin buffers");
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_value);
    bool profile=std::getenv("BF_CUDA_SWIN_PROFILE")!=nullptr;cudaEvent_t events[6]={};
    cudaEvent_t detail[60]={};int detail_count=0;
    if(profile){for(int i=0;i<6;++i)cudaEventCreate(&events[i]);for(int i=0;i<60;++i)cudaEventCreate(&detail[i]);cudaEventRecord(events[0],stream);}
    if(!cudnn_ok(cudnnSetStream(s->cudnn,stream),error,cap,"set Swin cuDNN stream"))return 0;
    const float one=1.0f,zero=0.0f;cudnnStatus_t status=cudnnConvolutionBiasActivationForward(s->cudnn,&one,s->patch_input_desc,images,
        s->patch_filter_desc,s->patch_weight,s->patch_conv_desc,s->patch_algorithm,s->cudnn_workspace,s->cudnn_workspace_bytes,&zero,
        s->patch_output_desc,s->normal_buffer,s->patch_bias_desc,s->patch_bias,s->patch_activation_desc,s->patch_output_desc,s->normal_buffer);
    if(!cudnn_ok(status,error,cap,"execute Swin patch embedding"))return 0;
    int rows=(int)(s->batches*(size_t)s->h[0]*s->w[0]);patch_transpose_norm_kernel<<<rows,128,0,stream>>>(s->normal_buffer,s->current_buffer,s->patch_scale,s->patch_norm_bias,(int)s->batches,s->h[0],s->w[0]);
    if(profile)cudaEventRecord(events[1],stream);
    float *current=s->current_buffer,*normal=s->normal_buffer;float *outputs[3]={out1,out2,out3};
    for(int st=0;st<4;++st){int c=s->channels[st],heads=s->heads[st],height=s->h[st],width=s->w[st];rows=(int)(s->batches*(size_t)height*width);size_t count=(size_t)rows*c;
        int wy=(height+6)/7,wx=(width+6)/7,total_windows=(int)(s->batches*(size_t)wy*wx);
        for(int b=0;b<s->depths[st];++b){swin_block &x=s->blocks[st][b];int shift=(b&1)?3:0;
            if(profile)cudaEventRecord(detail[detail_count++],stream);
            layer_norm_kernel<<<rows,256,0,stream>>>(current,normal,x.norm1_scale,x.norm1_bias,rows,c);
            if(profile)cudaEventRecord(detail[detail_count++],stream);
            for(int window_begin=0;window_begin<total_windows;window_begin+=(int)s->window_chunk){int window_count=(int)((size_t)(total_windows-window_begin)<s->window_chunk?(size_t)(total_windows-window_begin):s->window_chunk);int window_rows=window_count*49;
                size_t window_values=(size_t)window_rows*c;gather_windows_kernel<<<(window_values+255)/256,256,0,stream>>>(normal,s->windows,(int)s->batches,height,width,c,wy,wx,window_begin,window_count,shift);
                if(!linear(s,x.qkv,s->windows,s->qkv_buffer,window_rows,stream,error,cap))return 0;
                if(!std::getenv("BF_CUDA_SWIN_NO_SHARED_KV")){
                    int threads=std::getenv("BF_CUDA_SWIN_SHARED16")?512:(std::getenv("BF_CUDA_SWIN_SHARED14")?448:224);
                    window_attention_shared_kernel<<<window_count*heads,threads,49*64*sizeof(float),stream>>>(s->qkv_buffer,s->windows,x.relative_bias,x.relative_index,window_count,c,heads,wy,wx,window_begin,shift);}
                else if(std::getenv("BF_CUDA_SWIN_ONE_QUERY"))
                    window_attention_kernel<<<window_count*heads*49,32,0,stream>>>(s->qkv_buffer,s->windows,x.relative_bias,x.relative_index,window_count,c,heads,wy,wx,window_begin,shift);
                else if(std::getenv("BF_CUDA_SWIN_TILE16"))
                    window_attention_tiled_kernel<16><<<window_count*heads*4,512,0,stream>>>(s->qkv_buffer,s->windows,x.relative_bias,x.relative_index,window_count,c,heads,wy,wx,window_begin,shift);
                else window_attention_tiled_kernel<8><<<window_count*heads*7,256,0,stream>>>(s->qkv_buffer,s->windows,x.relative_bias,x.relative_index,window_count,c,heads,wy,wx,window_begin,shift);
                if(!linear(s,x.projection,s->windows,s->qkv_buffer,window_rows,stream,error,cap))return 0;
                scatter_windows_kernel<<<(window_values+255)/256,256,0,stream>>>(s->qkv_buffer,normal,(int)s->batches,height,width,c,wy,wx,window_begin,window_count,shift);}
            if(profile)cudaEventRecord(detail[detail_count++],stream);
            add_kernel<<<(count+255)/256,256,0,stream>>>(current,normal,count);
            if(profile)cudaEventRecord(detail[detail_count++],stream);
            layer_norm_kernel<<<rows,256,0,stream>>>(current,normal,x.norm2_scale,x.norm2_bias,rows,c);
            for(size_t begin=0;begin<(size_t)rows;begin+=s->ffn_chunk_rows){int chunk=(int)fmin((double)s->ffn_chunk_rows,(double)((size_t)rows-begin));
                if(!linear(s,x.ffn1,normal+begin*c,s->ffn_buffer,chunk,stream,error,cap))return 0;size_t hidden=(size_t)chunk*4*c;
                gelu_kernel<<<(hidden+255)/256,256,0,stream>>>(s->ffn_buffer,hidden);
                if(!linear(s,x.ffn2,s->ffn_buffer,normal+begin*c,chunk,stream,error,cap))return 0;
                add_kernel<<<((size_t)chunk*c+255)/256,256,0,stream>>>(current+begin*c,normal+begin*c,(size_t)chunk*c);}
            if(profile)cudaEventRecord(detail[detail_count++],stream);
        }
        if(st>0){layer_norm_kernel<<<rows,256,0,stream>>>(current,normal,s->output_scale[st-1],s->output_bias[st-1],rows,c);
            output_transpose_kernel<<<(count+255)/256,256,0,stream>>>(normal,outputs[st-1],count,c,height*width);}
        if(st<3){int next_h=s->h[st+1],next_w=s->w[st+1],merge_rows=(int)(s->batches*(size_t)next_h*next_w);size_t unfolded=(size_t)merge_rows*4*c;
            patch_merge_kernel<<<(unfolded+255)/256,256,0,stream>>>(current,s->ffn_buffer,(int)s->batches,height,width,c);
            layer_norm_kernel<<<merge_rows,256,0,stream>>>(s->ffn_buffer,s->ffn_buffer,s->merges[st].scale,s->merges[st].bias,merge_rows,4*c);
            if(!linear(s,s->merges[st].reduction,s->ffn_buffer,normal,merge_rows,stream,error,cap))return 0;
            float *swap=current;current=normal;normal=swap;}
        if(profile)cudaEventRecord(events[st+2],stream);
    }
    if(profile){cudaEventSynchronize(events[5]);float value=0.0f,parts[4]={};std::fprintf(stderr,"cuda_swin stages");for(int i=1;i<6;++i){cudaEventElapsedTime(&value,events[i-1],events[i]);std::fprintf(stderr," %s=%.3f",i==1?"patch":(i==2?"s0":(i==3?"s1":(i==4?"s2":"s3"))),value);}for(int i=0;i<detail_count;i+=5)for(int p=0;p<4;++p){cudaEventElapsedTime(&value,detail[i+p],detail[i+p+1]);parts[p]+=value;}std::fprintf(stderr," | norm1=%.3f window_path=%.3f residual=%.3f norm2+ffn=%.3f\n",parts[0],parts[1],parts[2],parts[3]);for(int i=0;i<6;++i)cudaEventDestroy(events[i]);for(int i=0;i<60;++i)cudaEventDestroy(detail[i]);}
    return cuda_ok(cudaGetLastError(),error,cap,"launch CUDA Swin graph");
}

extern "C" size_t bf_cuda_swin_resident_bytes(const bf_cuda_swin *s){return s?s->resident_bytes:0;}
