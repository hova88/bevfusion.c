#include "bf_cuda_lidar.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

struct gpu_sparse_layer {
    float *weight,*bias;int ci,co,kd,kh,kw,sd,sh,sw,pd,ph,pw,submanifold;
    size_t weight_bytes,bias_bytes;
};

struct bf_cuda_lidar_backbone {
    gpu_sparse_layer input,residual[4][2][2],down[3],output;
    size_t batches,capacity,hash_capacity,resident_bytes;
    int input_d,input_h,input_w,output_d,output_h,output_w;
    bf_coord4 *coords[2];float *features[3];
    unsigned long long *hash_keys[2];int *hash_values[2];
    unsigned *counts,*debug_counts;int *overflow,*neighbors;
};

static int fail(char *error,size_t cap,const char *format,...){if(error&&cap){va_list args;va_start(args,format);std::vsnprintf(error,cap,format,args);va_end(args);}return 0;}
static int cuda_ok(cudaError_t status,char *error,size_t cap,const char *where){return status==cudaSuccess?1:fail(error,cap,"%s: %s",where,cudaGetErrorString(status));}
static const float *tensor_f32(const bf_model *model,const char *name,uint32_t rank,const uint32_t *dims,char *error,size_t cap){const bf_tensor *t=bf_model_find(model,name);if(!t||t->dtype!=BF_DTYPE_F32||t->rank!=rank)return fail(error,cap,"%s contract mismatch",name),nullptr;for(uint32_t i=0;i<rank;++i)if(t->dims[i]!=dims[i])return fail(error,cap,"%s shape mismatch",name),nullptr;return(const float*)t->data;}
static int upload(float **device,const float *host,size_t count,size_t *resident,char *error,size_t cap){size_t bytes=count*sizeof(float);if(!cuda_ok(cudaMalloc(device,bytes),error,cap,"allocate LiDAR parameter")||!cuda_ok(cudaMemcpy(*device,host,bytes,cudaMemcpyHostToDevice),error,cap,"upload LiDAR parameter"))return 0;*resident+=bytes;return 1;}

static int bind_layer(bf_cuda_lidar_backbone *net,gpu_sparse_layer *layer,const bf_model *model,
    const char *weight_name,const char *bn_prefix,int ci,int co,int kd,int kh,int kw,
    int sd,int sh,int sw,int pd,int ph,int pw,int subm,char *error,size_t cap){
    uint32_t wd[5]={(uint32_t)kd,(uint32_t)kh,(uint32_t)kw,(uint32_t)ci,(uint32_t)co};
    const float *source=tensor_f32(model,weight_name,5,wd,error,cap);if(!source)return 0;
    uint32_t bd[1]={(uint32_t)co};const char *suffix[4]={"weight","bias","running_mean","running_var"};const float *bn[4]={};char name[192];
    for(int i=0;i<4;++i){std::snprintf(name,sizeof(name),"%s.%s",bn_prefix,suffix[i]);bn[i]=tensor_f32(model,name,1,bd,error,cap);if(!bn[i])return 0;}
    size_t weight_count=(size_t)kd*kh*kw*ci*co;float *weight=(float*)std::malloc(weight_count*4),*bias=(float*)std::malloc((size_t)co*4);
    if(!weight||!bias){std::free(weight);std::free(bias);return fail(error,cap,"LiDAR fold allocation failed");}
    std::memcpy(weight,source,weight_count*4);for(int o=0;o<co;++o){float factor=bn[0][o]/std::sqrt(bn[3][o]+1e-3f);bias[o]=bn[1][o]-bn[2][o]*factor;for(size_t i=o;i<weight_count;i+=co)weight[i]*=factor;}
    layer->ci=ci;layer->co=co;layer->kd=kd;layer->kh=kh;layer->kw=kw;layer->sd=sd;layer->sh=sh;layer->sw=sw;layer->pd=pd;layer->ph=ph;layer->pw=pw;layer->submanifold=subm;
    int ok=upload(&layer->weight,weight,weight_count,&net->resident_bytes,error,cap)&&upload(&layer->bias,bias,co,&net->resident_bytes,error,cap);
    layer->weight_bytes=weight_count*4;layer->bias_bytes=(size_t)co*4;std::free(weight);std::free(bias);return ok;
}

__device__ static unsigned long long mix64_device(unsigned long long value){value^=value>>30;value*=0xbf58476d1ce4e5b9ull;value^=value>>27;value*=0x94d049bb133111ebull;return value^(value>>31);}
__device__ static unsigned long long coord_key(int b,int z,int y,int x,int d,int h,int w){return(((unsigned long long)b*d+z)*h*w+(unsigned long long)y*w+x)+1;}
__device__ static int hash_lookup(const unsigned long long *keys,const int *values,size_t capacity,unsigned long long key){size_t slot=(size_t)mix64_device(key)&(capacity-1);for(size_t probe=0;probe<capacity;++probe){unsigned long long found=keys[slot];if(found==key)return values[slot]-1;if(!found)return-1;slot=(slot+1)&(capacity-1);}return-1;}
__device__ static bool hash_insert(unsigned long long *keys,int *values,size_t capacity,unsigned long long key,int value){size_t slot=(size_t)mix64_device(key)&(capacity-1);for(size_t probe=0;probe<capacity;++probe){unsigned long long old=atomicCAS(keys+slot,0ull,key);if(!old||old==key){if(!old)values[slot]=value+1;return old==0;}slot=(slot+1)&(capacity-1);}return false;}

__global__ static void build_hash_kernel(const bf_coord4 *coords,const unsigned *count,
    unsigned long long *keys,int *values,size_t hash_capacity,int d,int h,int w){unsigned n=*count;for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){bf_coord4 c=coords[i];hash_insert(keys,values,hash_capacity,coord_key(c.batch,c.z,c.y,c.x,d,h,w),(int)i);}}

__global__ static void generate_output_coords_kernel(const bf_coord4 *input,const unsigned *input_count,
    bf_coord4 *output,unsigned *output_count,unsigned long long *keys,int *values,size_t hash_capacity,
    size_t capacity,int id,int ih,int iw,int od,int oh,int ow,gpu_sparse_layer layer,int *overflow){
    unsigned n=*input_count;for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){bf_coord4 c=input[i];
        for(int kz=0;kz<layer.kd;++kz){int nz=c.z+layer.pd-kz;if(nz<0||nz%layer.sd)continue;int z=nz/layer.sd;if((unsigned)z>=(unsigned)od)continue;
        for(int ky=0;ky<layer.kh;++ky){int ny=c.y+layer.ph-ky;if(ny<0||ny%layer.sh)continue;int y=ny/layer.sh;if((unsigned)y>=(unsigned)oh)continue;
        for(int kx=0;kx<layer.kw;++kx){int nx=c.x+layer.pw-kx;if(nx<0||nx%layer.sw)continue;int x=nx/layer.sw;if((unsigned)x>=(unsigned)ow)continue;
            unsigned long long key=coord_key(c.batch,z,y,x,od,oh,ow);size_t slot=(size_t)mix64_device(key)&(hash_capacity-1);
            for(size_t probe=0;probe<hash_capacity;++probe){unsigned long long old=atomicCAS(keys+slot,0ull,key);if(!old){unsigned index=atomicAdd(output_count,1u);if(index>=capacity){*overflow=1;values[slot]=0;}else{output[index]={c.batch,z,y,x};values[slot]=(int)index+1;}break;}if(old==key)break;slot=(slot+1)&(hash_capacity-1);}
        }}}
    }}

__global__ static void build_neighbors_kernel(const bf_coord4 *out_coords,const unsigned *out_count,
    int *neighbors,const unsigned long long *input_keys,const int *input_values,size_t hash_capacity,
    int id,int ih,int iw,gpu_sparse_layer layer,size_t capacity){unsigned n=*out_count;int volume=layer.kd*layer.kh*layer.kw;size_t total=(size_t)n*volume;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=(size_t)gridDim.x*blockDim.x){unsigned row=i/volume;int k=i%volume,kx=k%layer.kw,ky=(k/layer.kw)%layer.kh,kz=k/(layer.kw*layer.kh);bf_coord4 out=out_coords[row];int z=out.z*layer.sd-layer.pd+kz,y=out.y*layer.sh-layer.ph+ky,x=out.x*layer.sw-layer.pw+kx;neighbors[(size_t)row*27+k]=(unsigned)z<(unsigned)id&&(unsigned)y<(unsigned)ih&&(unsigned)x<(unsigned)iw?hash_lookup(input_keys,input_values,hash_capacity,coord_key(out.batch,z,y,x,id,ih,iw)):-1;}}

__global__ static void rulebook_convolution_kernel(const unsigned *out_count,const int *neighbors,
    const float *input,float *output,gpu_sparse_layer layer,int relu){unsigned n=*out_count;int volume=layer.kd*layer.kh*layer.kw;
    for(unsigned row=blockIdx.x;row<n;row+=gridDim.x)for(int oc=threadIdx.x;oc<layer.co;oc+=blockDim.x){float sum=layer.bias[oc];for(int k=0;k<volume;++k){int source=neighbors[(size_t)row*27+k];if(source<0)continue;size_t weight_base=(size_t)k*layer.ci*layer.co+oc;for(int ic=0;ic<layer.ci;++ic)sum+=input[(size_t)source*layer.ci+ic]*layer.weight[weight_base+(size_t)ic*layer.co];}output[(size_t)row*layer.co+oc]=relu?fmaxf(sum,0.0f):sum;}}

__global__ static void residual_add_kernel(const float *identity,float *update,const unsigned *count,int channels){unsigned n=*count;size_t total=(size_t)n*channels;for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=(size_t)gridDim.x*blockDim.x)update[i]=fmaxf(update[i]+identity[i],0.0f);}
__global__ static void dense_scatter_kernel(const bf_coord4 *coords,const float *features,const unsigned *count,float *dense,int channels,int d,int h,int w){unsigned n=*count;for(unsigned row=blockIdx.x;row<n;row+=gridDim.x){bf_coord4 c=coords[row];for(int channel=threadIdx.x;channel<channels;channel+=blockDim.x)dense[(((size_t)c.batch*channels+channel)*d+c.z)*h*w+(size_t)c.y*w+c.x]=features[(size_t)row*channels+channel];}}
__global__ static void record_count_kernel(const unsigned *source,unsigned *destination){if(!threadIdx.x&&!blockIdx.x)*destination=*source;}
__global__ static void clamp_count_kernel(unsigned *count,unsigned capacity){if(!threadIdx.x&&!blockIdx.x&&*count>capacity)*count=capacity;}

static void free_layer(gpu_sparse_layer *l){cudaFree(l->weight);cudaFree(l->bias);}
extern "C" void bf_cuda_lidar_backbone_destroy(bf_cuda_lidar_backbone *n){if(!n)return;free_layer(&n->input);for(int s=0;s<4;++s)for(int b=0;b<2;++b)for(int c=0;c<2;++c)free_layer(&n->residual[s][b][c]);for(int i=0;i<3;++i)free_layer(&n->down[i]);free_layer(&n->output);for(int i=0;i<2;++i){cudaFree(n->coords[i]);cudaFree(n->hash_keys[i]);cudaFree(n->hash_values[i]);}for(int i=0;i<3;++i)cudaFree(n->features[i]);cudaFree(n->counts);cudaFree(n->debug_counts);cudaFree(n->overflow);cudaFree(n->neighbors);std::free(n);}

static int output_axis(int input,int kernel,int stride,int pad){return(input+2*pad-kernel)/stride+1;}
extern "C" int bf_cuda_lidar_backbone_create(const bf_model *model,size_t batches,size_t d,size_t h,size_t w,size_t capacity,bf_cuda_lidar_backbone **out,char *error,size_t cap){
    if(out)*out=nullptr;if(!model||!out||!batches||!d||!h||!w||!capacity||batches>INT_MAX||d>INT_MAX||h>INT_MAX||w>INT_MAX||capacity>UINT_MAX)return fail(error,cap,"invalid CUDA LiDAR contract");
    bf_cuda_lidar_backbone *n=(bf_cuda_lidar_backbone*)std::calloc(1,sizeof(*n));if(!n)return fail(error,cap,"LiDAR host allocation failed");
    int channels[4]={16,32,64,128};
    n->batches=batches;n->capacity=capacity;n->input_d=d;n->input_h=h;n->input_w=w;n->hash_capacity=2;while(n->hash_capacity<capacity*2)n->hash_capacity<<=1;
#define LAYER(DST,W,BN,CI,CO,KD,KH,KW,SD,SH,SW,PD,PH,PW,SUB) if(!bind_layer(n,DST,model,W,BN,CI,CO,KD,KH,KW,SD,SH,SW,PD,PH,PW,SUB,error,cap))goto failure
    LAYER(&n->input,"backbone_3d.conv_input.0.weight","backbone_3d.conv_input.1",5,16,3,3,3,1,1,1,1,1,1,1);
    for(int stage=0;stage<4;++stage)for(int block=0;block<2;++block)for(int conv=0;conv<2;++conv){char weight[160],bn[160];std::snprintf(weight,sizeof(weight),"backbone_3d.conv%d.%d.conv%d.weight",stage+1,block+(stage?1:0),conv+1);std::snprintf(bn,sizeof(bn),"backbone_3d.conv%d.%d.bn%d",stage+1,block+(stage?1:0),conv+1);LAYER(&n->residual[stage][block][conv],weight,bn,channels[stage],channels[stage],3,3,3,1,1,1,1,1,1,1);}
    for(int transition=0;transition<3;++transition){int stage=transition+2;char weight[128],bn[128];std::snprintf(weight,sizeof(weight),"backbone_3d.conv%d.0.0.weight",stage);std::snprintf(bn,sizeof(bn),"backbone_3d.conv%d.0.1",stage);int pz=stage==4?0:1;LAYER(&n->down[transition],weight,bn,channels[transition],channels[transition+1],3,3,3,2,2,2,pz,1,1,0);}
    LAYER(&n->output,"backbone_3d.conv_out.0.weight","backbone_3d.conv_out.1",128,128,3,1,1,2,1,1,0,0,0,0);
#undef LAYER
    {int td=d,th=h,tw=w;for(int i=0;i<3;++i){td=output_axis(td,3,2,n->down[i].pd);th=output_axis(th,3,2,1);tw=output_axis(tw,3,2,1);}n->output_d=output_axis(td,3,2,0);n->output_h=th;n->output_w=tw;if(n->output_d<=0||th<=0||tw<=0){fail(error,cap,"invalid CUDA LiDAR output shape");goto failure;}}
    for(int i=0;i<2;++i){if(!cuda_ok(cudaMalloc(&n->coords[i],capacity*sizeof(bf_coord4)),error,cap,"allocate LiDAR coords")||!cuda_ok(cudaMalloc(&n->hash_keys[i],n->hash_capacity*sizeof(unsigned long long)),error,cap,"allocate LiDAR hash keys")||!cuda_ok(cudaMalloc(&n->hash_values[i],n->hash_capacity*sizeof(int)),error,cap,"allocate LiDAR hash values"))goto failure;n->resident_bytes+=capacity*sizeof(bf_coord4)+n->hash_capacity*(sizeof(unsigned long long)+sizeof(int));}
    for(int i=0;i<3;++i){if(!cuda_ok(cudaMalloc(&n->features[i],capacity*128*sizeof(float)),error,cap,"allocate LiDAR features"))goto failure;n->resident_bytes+=capacity*128*sizeof(float);}
    if(!cuda_ok(cudaMalloc(&n->counts,2*sizeof(unsigned)),error,cap,"allocate LiDAR counts")||!cuda_ok(cudaMalloc(&n->debug_counts,5*sizeof(unsigned)),error,cap,"allocate LiDAR debug counts")||!cuda_ok(cudaMalloc(&n->overflow,sizeof(int)),error,cap,"allocate LiDAR overflow")||!cuda_ok(cudaMalloc(&n->neighbors,capacity*27*sizeof(int)),error,cap,"allocate LiDAR rulebook"))goto failure;n->resident_bytes+=7*sizeof(unsigned)+sizeof(int)+capacity*27*sizeof(int);*out=n;return 1;
failure:bf_cuda_lidar_backbone_destroy(n);return 0;
}

static void build_neighbors(bf_cuda_lidar_backbone *n,const bf_coord4 *coords,const unsigned *count,
    const unsigned long long *keys,const int *values,int d,int h,int w,gpu_sparse_layer layer,cudaStream_t stream){build_neighbors_kernel<<<512,256,0,stream>>>(coords,count,n->neighbors,keys,values,n->hash_capacity,d,h,w,layer,n->capacity);}
static void launch_rulebook(bf_cuda_lidar_backbone *n,const unsigned *count,const float *input,float *output,
    gpu_sparse_layer layer,int relu,cudaStream_t stream){size_t target=8192;const char *text=std::getenv("BF_CUDA_LIDAR_BLOCKS");if(text){char *end=nullptr;unsigned long value=std::strtoul(text,&end,10);if(end&&!*end&&value)target=value;}unsigned blocks=(unsigned)(n->capacity<target?n->capacity:target);int threads=std::getenv("BF_CUDA_LIDAR_FIXED128")?128:(layer.co<=32?32:(layer.co<=64?64:128));rulebook_convolution_kernel<<<blocks,threads,0,stream>>>(count,n->neighbors,input,output,layer,relu);}

static int lidar_forward_prepared(bf_cuda_lidar_backbone *n,const bf_coord4 *voxel_coords,
    const float *voxel_features,float *dense,cudaStream_t stream,char *error,size_t cap){
    bool profile=std::getenv("BF_CUDA_LIDAR_PROFILE")!=nullptr;cudaEvent_t events[7]={};if(profile){for(int i=0;i<7;++i)cudaEventCreate(&events[i]);cudaEventRecord(events[0],stream);}
    cudaMemsetAsync(n->hash_keys[0],0,n->hash_capacity*sizeof(unsigned long long),stream);cudaMemsetAsync(n->hash_values[0],0,n->hash_capacity*sizeof(int),stream);cudaMemsetAsync(n->overflow,0,sizeof(int),stream);
    build_hash_kernel<<<256,256,0,stream>>>(voxel_coords,n->counts,n->hash_keys[0],n->hash_values[0],n->hash_capacity,n->input_d,n->input_h,n->input_w);
    const bf_coord4 *coords=voxel_coords;const float *features=voxel_features;unsigned *count=n->counts;int hash_slot=0,coord_slot=0,feature_slot=0;int d=n->input_d,h=n->input_h,w=n->input_w;
    build_neighbors(n,coords,count,n->hash_keys[0],n->hash_values[0],d,h,w,n->input,stream);launch_rulebook(n,count,features,n->features[0],n->input,1,stream);features=n->features[0];
    if(profile)cudaEventRecord(events[1],stream);
    for(int stage=0;stage<4;++stage){if(stage){gpu_sparse_layer layer=n->down[stage-1];int od=output_axis(d,3,2,layer.pd),oh=output_axis(h,3,2,1),ow=output_axis(w,3,2,1);int next_hash=hash_slot^1;unsigned *next_count=n->counts+(count==n->counts?1:0);cudaMemsetAsync(next_count,0,sizeof(unsigned),stream);cudaMemsetAsync(n->hash_keys[next_hash],0,n->hash_capacity*sizeof(unsigned long long),stream);cudaMemsetAsync(n->hash_values[next_hash],0,n->hash_capacity*sizeof(int),stream);
            if(std::getenv("BF_CUDA_LIDAR_SERIAL_COORDS"))generate_output_coords_kernel<<<1,1,0,stream>>>(coords,count,n->coords[coord_slot],next_count,n->hash_keys[next_hash],n->hash_values[next_hash],n->hash_capacity,n->capacity,d,h,w,od,oh,ow,layer,n->overflow);else generate_output_coords_kernel<<<256,256,0,stream>>>(coords,count,n->coords[coord_slot],next_count,n->hash_keys[next_hash],n->hash_values[next_hash],n->hash_capacity,n->capacity,d,h,w,od,oh,ow,layer,n->overflow);
            clamp_count_kernel<<<1,1,0,stream>>>(next_count,(unsigned)n->capacity);
            int next_feature=(feature_slot+1)%3;build_neighbors(n,n->coords[coord_slot],next_count,n->hash_keys[hash_slot],n->hash_values[hash_slot],d,h,w,layer,stream);launch_rulebook(n,next_count,features,n->features[next_feature],layer,1,stream);
            coords=n->coords[coord_slot];coord_slot^=1;features=n->features[next_feature];feature_slot=next_feature;count=next_count;hash_slot=next_hash;d=od;h=oh;w=ow;}
        build_neighbors(n,coords,count,n->hash_keys[hash_slot],n->hash_values[hash_slot],d,h,w,n->residual[stage][0][0],stream);
        for(int block=0;block<2;++block){int first=(feature_slot+1)%3,second=(feature_slot+2)%3;launch_rulebook(n,count,features,n->features[first],n->residual[stage][block][0],1,stream);launch_rulebook(n,count,n->features[first],n->features[second],n->residual[stage][block][1],0,stream);residual_add_kernel<<<256,256,0,stream>>>(features,n->features[second],count,n->residual[stage][block][1].co);features=n->features[second];feature_slot=second;}
        if(profile){record_count_kernel<<<1,1,0,stream>>>(count,n->debug_counts+stage);cudaEventRecord(events[stage+2],stream);}
    }
    {gpu_sparse_layer layer=n->output;int od=output_axis(d,3,2,0),oh=h,ow=w;int next_hash=hash_slot^1;unsigned *next_count=n->counts+(count==n->counts?1:0);cudaMemsetAsync(next_count,0,sizeof(unsigned),stream);cudaMemsetAsync(n->hash_keys[next_hash],0,n->hash_capacity*sizeof(unsigned long long),stream);cudaMemsetAsync(n->hash_values[next_hash],0,n->hash_capacity*sizeof(int),stream);if(std::getenv("BF_CUDA_LIDAR_SERIAL_COORDS"))generate_output_coords_kernel<<<1,1,0,stream>>>(coords,count,n->coords[coord_slot],next_count,n->hash_keys[next_hash],n->hash_values[next_hash],n->hash_capacity,n->capacity,d,h,w,od,oh,ow,layer,n->overflow);else generate_output_coords_kernel<<<256,256,0,stream>>>(coords,count,n->coords[coord_slot],next_count,n->hash_keys[next_hash],n->hash_values[next_hash],n->hash_capacity,n->capacity,d,h,w,od,oh,ow,layer,n->overflow);clamp_count_kernel<<<1,1,0,stream>>>(next_count,(unsigned)n->capacity);int next_feature=(feature_slot+1)%3;build_neighbors(n,n->coords[coord_slot],next_count,n->hash_keys[hash_slot],n->hash_values[hash_slot],d,h,w,layer,stream);launch_rulebook(n,next_count,features,n->features[next_feature],layer,1,stream);size_t dense_count=n->batches*(size_t)128*od*oh*ow;cudaMemsetAsync(dense,0,dense_count*sizeof(float),stream);dense_scatter_kernel<<<4096,128,0,stream>>>(n->coords[coord_slot],n->features[next_feature],next_count,dense,128,od,oh,ow);}
    if(profile){record_count_kernel<<<1,1,0,stream>>>(n->counts+(count==n->counts?1:0),n->debug_counts+4);cudaEventRecord(events[6],stream);cudaEventSynchronize(events[6]);unsigned counts[5]={};cudaMemcpy(counts,n->debug_counts,sizeof(counts),cudaMemcpyDeviceToHost);float ms=0;std::fprintf(stderr,"cuda_lidar stages");for(int i=1;i<7;++i){cudaEventElapsedTime(&ms,events[i-1],events[i]);std::fprintf(stderr," %s=%.3f",i==1?"input":(i==6?"output":(i==2?"s0":(i==3?"s1":(i==4?"s2":"s3")))),ms);}std::fprintf(stderr," counts=%u/%u/%u/%u/%u\n",counts[0],counts[1],counts[2],counts[3],counts[4]);for(int i=0;i<7;++i)cudaEventDestroy(events[i]);}
    return cuda_ok(cudaGetLastError(),error,cap,"launch CUDA LiDAR graph");
}

extern "C" int bf_cuda_lidar_backbone_forward(bf_cuda_lidar_backbone *n,const bf_coord4 *voxel_coords,
    const float *voxel_features,size_t voxel_count,float *dense,void *stream_value,char *error,size_t cap){
    if(!n||!voxel_coords||!voxel_features||!voxel_count||voxel_count>n->capacity||!dense)return fail(error,cap,"invalid CUDA LiDAR buffers");cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_value);unsigned initial=(unsigned)voxel_count;if(!cuda_ok(cudaMemcpyAsync(n->counts,&initial,sizeof(initial),cudaMemcpyHostToDevice,stream),error,cap,"upload CUDA LiDAR count"))return 0;return lidar_forward_prepared(n,voxel_coords,voxel_features,dense,stream,error,cap);
}

extern "C" int bf_cuda_lidar_backbone_forward_count_device(bf_cuda_lidar_backbone *n,const bf_coord4 *voxel_coords,
    const float *voxel_features,const unsigned *voxel_count,float *dense,void *stream_value,char *error,size_t cap){
    if(!n||!voxel_coords||!voxel_features||!voxel_count||!dense)return fail(error,cap,"invalid device-count CUDA LiDAR buffers");cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_value);if(!cuda_ok(cudaMemcpyAsync(n->counts,voxel_count,sizeof(unsigned),cudaMemcpyDeviceToDevice,stream),error,cap,"copy device CUDA LiDAR count"))return 0;clamp_count_kernel<<<1,1,0,stream>>>(n->counts,(unsigned)n->capacity);return lidar_forward_prepared(n,voxel_coords,voxel_features,dense,stream,error,cap);
}

extern "C" size_t bf_cuda_lidar_backbone_resident_bytes(const bf_cuda_lidar_backbone *n){return n?n->resident_bytes:0;}
