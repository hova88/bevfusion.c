#include "bf_cuda.h"
#include "bf_cuda_camera.h"
#include "bf_cuda_swin.h"
#include "bf_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const bf_tensor *tensor(const bf_model *m,const char *name){const bf_tensor *t=bf_model_find(m,name);if(!t)std::fprintf(stderr,"missing %s\n",name);return t;}
static float *upload(const bf_tensor *t){float *d=nullptr;if(!t||cudaMalloc(&d,t->nbytes)!=cudaSuccess||cudaMemcpy(d,t->data,t->nbytes,cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(d);return nullptr;}return d;}
static int compare(const char *name,const float *device,const bf_tensor *expected){float *actual=(float*)std::malloc(expected->nbytes);if(!actual||cudaMemcpy(actual,device,expected->nbytes,cudaMemcpyDeviceToHost)!=cudaSuccess){std::free(actual);return 0;}const float *reference=(const float*)expected->data;size_t count=expected->nbytes/4;float maximum=0;double mean=0;int ok=1;for(size_t i=0;i<count;++i){float d=std::fabs(actual[i]-reference[i]);maximum=fmaxf(maximum,d);mean+=d;if(!(d<=8e-5f+8e-5f*std::fabs(reference[i])))ok=0;}std::printf("%-34s max_abs=%.3g mean_abs=%.3g\n",name,maximum,mean/count);std::free(actual);return ok;}
__global__ static void initialize(float *v,size_t n,float scale){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v[i]=((int)(i%31)-15)*scale;}
__global__ static void identity(float *v,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count*9)v[i]=(i%9)%4==0?1.0f:0.0f;}

static int compact_oracle(const bf_model *weights,const bf_model *oracle,
                          cudaStream_t stream,char *error,size_t cap){
    bf_cuda_swin *swin=nullptr;bf_cuda_camera_neck *neck=nullptr;
    const bf_tensor *image=tensor(oracle,"cuda_camera_full.image");
    const bf_tensor *depth=tensor(oracle,"cuda_camera_full.dense_depth");
    const bf_tensor *logits=tensor(oracle,"cuda_camera_full.logits");
    const bf_tensor *context=tensor(oracle,"cuda_camera_full.context");
    size_t batches=image?image->dims[0]:0;
    int ok=batches&&bf_cuda_swin_create(weights,batches,32,48,&swin,error,cap)&&
           bf_cuda_camera_neck_create(weights,batches,4,6,8,10,&neck,error,cap);
    float *di=upload(image),*dd=upload(depth),*s0=nullptr,*s1=nullptr,*s2=nullptr,*dl=nullptr,*dc=nullptr;
    ok=ok&&di&&dd&&cudaMalloc(&s0,batches*192*4*6*4)==cudaSuccess&&cudaMalloc(&s1,batches*384*2*3*4)==cudaSuccess&&
       cudaMalloc(&s2,batches*768*1*2*4)==cudaSuccess&&cudaMalloc(&dl,logits->nbytes)==cudaSuccess&&
       cudaMalloc(&dc,context->nbytes)==cudaSuccess&&
       bf_cuda_swin_forward(swin,di,s0,s1,s2,(void*)stream,error,cap)&&
       bf_cuda_camera_neck_forward(neck,s0,s1,s2,dd,dl,dc,(void*)stream,error,cap)&&
       cudaStreamSynchronize(stream)==cudaSuccess&&
       compare("cuda_camera_full.logits",dl,logits)&&compare("cuda_camera_full.context",dc,context);
    cudaFree(dc);cudaFree(dl);cudaFree(s2);cudaFree(s1);cudaFree(s0);cudaFree(dd);cudaFree(di);
    bf_cuda_camera_neck_destroy(neck);bf_cuda_swin_destroy(swin);return ok;
}

static int production(const bf_model *weights,char *error,size_t cap){
    bf_cuda_swin *swin=nullptr;bf_cuda_camera_neck *neck=nullptr;bf_cuda_lss_plan *lss=nullptr;
    bf_lss_desc desc={1,6,118,32,88,80,{-54,-54,-10},{0.3f,0.3f,20},{360,360,1}};
    int ok=bf_cuda_swin_create(weights,6,256,704,&swin,error,cap)&&
        bf_cuda_camera_neck_create(weights,6,32,88,360,360,&neck,error,cap)&&
        bf_cuda_lss_plan_create(&desc,&lss,error,cap);
    size_t image_n=6ull*3*256*704,depth_n=6ull*256*704,s0_n=6ull*192*32*88;
    size_t s1_n=6ull*384*16*44,s2_n=6ull*768*8*22,logit_n=6ull*118*32*88;
    size_t context_n=6ull*80*32*88,full_n=80ull*360*360,out_n=80ull*180*180;
    float *images=nullptr,*depth=nullptr,*s0=nullptr,*s1=nullptr,*s2=nullptr,*logits=nullptr,*context=nullptr,*full=nullptr,*output=nullptr;
    float *frustum=upload(tensor(weights,"vtransform.frustum")),*cr=nullptr,*ct=nullptr,*intr=nullptr,*pr=nullptr,*pt=nullptr,*er=nullptr,*et=nullptr;
    ok=ok&&frustum&&cudaMalloc(&images,image_n*4)==cudaSuccess&&cudaMalloc(&depth,depth_n*4)==cudaSuccess&&
       cudaMalloc(&s0,s0_n*4)==cudaSuccess&&cudaMalloc(&s1,s1_n*4)==cudaSuccess&&cudaMalloc(&s2,s2_n*4)==cudaSuccess&&
       cudaMalloc(&logits,logit_n*4)==cudaSuccess&&cudaMalloc(&context,context_n*4)==cudaSuccess&&
       cudaMalloc(&full,full_n*4)==cudaSuccess&&cudaMalloc(&output,out_n*4)==cudaSuccess&&
       cudaMalloc(&cr,6*9*4)==cudaSuccess&&cudaMalloc(&ct,6*3*4)==cudaSuccess&&cudaMalloc(&intr,6*9*4)==cudaSuccess&&
       cudaMalloc(&pr,6*9*4)==cudaSuccess&&cudaMalloc(&pt,6*3*4)==cudaSuccess&&cudaMalloc(&er,9*4)==cudaSuccess&&cudaMalloc(&et,3*4)==cudaSuccess;
    if(ok){initialize<<<(image_n+255)/256,256>>>(images,image_n,0.01f);initialize<<<(depth_n+255)/256,256>>>(depth,depth_n,0.01f);
        identity<<<1,64>>>(cr,6);identity<<<1,64>>>(intr,6);identity<<<1,64>>>(pr,6);identity<<<1,32>>>(er,1);
        cudaMemset(ct,0,6*3*4);cudaMemset(pt,0,6*3*4);cudaMemset(et,0,3*4);ok=cudaDeviceSynchronize()==cudaSuccess;}
    cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);float cold=0,warm=0;
#define FORWARD() (bf_cuda_swin_forward(swin,images,s0,s1,s2,nullptr,error,cap)&& \
 bf_cuda_camera_neck_forward(neck,s0,s1,s2,depth,logits,context,nullptr,error,cap)&& \
 bf_cuda_lss_plan_prepare_calibration_f32(lss,frustum,cr,ct,intr,pr,pt,er,et,nullptr,error,cap)&& \
 bf_cuda_lss_plan_forward_f32(lss,logits,context,full,nullptr,error,cap)&& \
 bf_cuda_camera_downsample_forward(neck,full,output,nullptr,error,cap))
    if(ok){cudaEventRecord(begin);ok=FORWARD();cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&cold,begin,end);}
    if(ok){cudaEventRecord(begin);for(int i=0;i<5;++i)ok=ok&&FORWARD();cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&warm,begin,end);}
    float *first=(float*)std::malloc(out_n*4),*second=(float*)std::malloc(out_n*4);
    if(ok)ok=first&&second&&cudaMemcpy(first,output,out_n*4,cudaMemcpyDeviceToHost)==cudaSuccess&&FORWARD()&&
        cudaDeviceSynchronize()==cudaSuccess&&cudaMemcpy(second,output,out_n*4,cudaMemcpyDeviceToHost)==cudaSuccess&&
        std::memcmp(first,second,out_n*4)==0;
    size_t resident=bf_cuda_swin_resident_bytes(swin)+bf_cuda_camera_neck_resident_bytes(neck)+bf_cuda_lss_plan_resident_bytes(lss);
    double boundary=(image_n+depth_n+s0_n+s1_n+s2_n+logit_n+context_n+full_n+out_n)*4.0/(1024*1024);
    std::printf("cuda_camera_full production cold=%.3f ms warm=%.3f ms resident=%.2f MiB boundary=%.2f MiB repeat=%s\n",cold,warm/5,resident/(1024.0*1024.0),boundary,ok?"exact":"FAIL");
#undef FORWARD
    std::free(second);std::free(first);cudaEventDestroy(end);cudaEventDestroy(begin);cudaFree(et);cudaFree(er);cudaFree(pt);cudaFree(pr);cudaFree(intr);cudaFree(ct);cudaFree(cr);cudaFree(frustum);
    cudaFree(output);cudaFree(full);cudaFree(context);cudaFree(logits);cudaFree(s2);cudaFree(s1);cudaFree(s0);cudaFree(depth);cudaFree(images);
    bf_cuda_lss_plan_destroy(lss);bf_cuda_camera_neck_destroy(neck);bf_cuda_swin_destroy(swin);return ok;
}

int main(int argc,char **argv){if(argc!=3)return 2;char error[256]={0};bf_model *weights=nullptr,*oracle=nullptr;
 int ok=bf_model_open(argv[1],&weights,error,sizeof(error))&&bf_model_open(argv[2],&oracle,error,sizeof(error));cudaStream_t stream=nullptr;
 if(ok)ok=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess&&compact_oracle(weights,oracle,stream,error,sizeof(error));
 if(ok)ok=production(weights,error,sizeof(error));if(!ok)std::fprintf(stderr,"CUDA full camera failure: %s\n",error);
 if(stream)cudaStreamDestroy(stream);bf_model_close(oracle);bf_model_close(weights);return ok?0:5;}
