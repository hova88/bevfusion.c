#include "bf_cuda_camera.h"
#include "bf_cuda.h"
#include "bf_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const bf_tensor *tensor(const bf_model *model,const char *name) {
    const bf_tensor *value=bf_model_find(model,name);
    if(!value)std::fprintf(stderr,"missing fixture: %s\n",name);
    return value;
}

static float *upload(const bf_tensor *value) {
    float *device=nullptr;
    if(!value||cudaMalloc(&device,value->nbytes)!=cudaSuccess||
       cudaMemcpy(device,value->data,value->nbytes,cudaMemcpyHostToDevice)!=cudaSuccess) {
        cudaFree(device);return nullptr;
    }
    return device;
}

static int compare_device(const char *name,const float *device,
                          const bf_tensor *expected,float atol,float rtol) {
    float *actual=(float*)std::malloc(expected->nbytes);
    if(!actual||cudaMemcpy(actual,device,expected->nbytes,
                           cudaMemcpyDeviceToHost)!=cudaSuccess) { std::free(actual);return 0; }
    const float *reference=(const float*)expected->data;
    size_t count=expected->nbytes/sizeof(float);float maximum=0.0f;double mean=0.0;int ok=1;
    for(size_t i=0;i<count;++i) {
        float difference=std::fabs(actual[i]-reference[i]);
        maximum=fmaxf(maximum,difference);mean+=difference;
        if(!(difference<=atol+rtol*std::fabs(reference[i])))ok=0;
    }
    std::printf("%-34s max_abs=%.3g mean_abs=%.3g\n",name,maximum,mean/count);
    std::free(actual);return ok;
}

__global__ static void initialize(float *values,size_t count,float scale) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)values[i]=((int)(i%29)-14)*scale;
}

__global__ static void identity_matrices(float *values,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count*9)values[i]=(i%9)%4==0?1.0f:0.0f;
}

static int test_fpn(const bf_model *weights,const bf_model *oracle,
                    cudaStream_t stream,char *error,size_t cap) {
    bf_cuda_camera_neck *neck=nullptr;
    int ok=bf_cuda_camera_neck_create(weights,1,4,6,8,10,&neck,error,cap);
    const bf_tensor *i0=tensor(oracle,"image_fpn.input0");
    const bf_tensor *i1=tensor(oracle,"image_fpn.input1");
    const bf_tensor *i2=tensor(oracle,"image_fpn.input2");
    const bf_tensor *o0=tensor(oracle,"image_fpn.output0");
    const bf_tensor *o1=tensor(oracle,"image_fpn.output1");
    float *d0=upload(i0),*d1=upload(i1),*d2=upload(i2),*a0=nullptr,*a1=nullptr;
    ok=ok&&d0&&d1&&d2&&o0&&o1&&cudaMalloc(&a0,o0->nbytes)==cudaSuccess&&
       cudaMalloc(&a1,o1->nbytes)==cudaSuccess&&
       bf_cuda_camera_fpn_forward(neck,d0,d1,d2,a0,a1,(void*)stream,error,cap)&&
       cudaStreamSynchronize(stream)==cudaSuccess&&
       compare_device("cuda_camera.fpn1",a1,o1,2e-5f,2e-5f)&&
       compare_device("cuda_camera.fpn0",a0,o0,2e-5f,2e-5f);
    cudaFree(a1);cudaFree(a0);cudaFree(d2);cudaFree(d1);cudaFree(d0);
    bf_cuda_camera_neck_destroy(neck);return ok;
}

static int test_depth(const bf_model *weights,const bf_model *oracle,
                      cudaStream_t stream,char *error,size_t cap) {
    bf_cuda_camera_neck *neck=nullptr;
    int ok=bf_cuda_camera_neck_create(weights,1,2,3,8,10,&neck,error,cap);
    const bf_tensor *features=tensor(oracle,"depth_head.features");
    const bf_tensor *depth=tensor(oracle,"depth_head.dense_depth");
    const bf_tensor *logits=tensor(oracle,"depth_head.logits");
    const bf_tensor *context=tensor(oracle,"depth_head.context");
    float *df=upload(features),*dd=upload(depth),*dl=nullptr,*dc=nullptr;
    ok=ok&&df&&dd&&logits&&context&&cudaMalloc(&dl,logits->nbytes)==cudaSuccess&&
       cudaMalloc(&dc,context->nbytes)==cudaSuccess&&
       bf_cuda_camera_depth_forward(neck,df,dd,dl,dc,(void*)stream,error,cap)&&
       cudaStreamSynchronize(stream)==cudaSuccess&&
       compare_device("cuda_camera.depth_logits",dl,logits,2e-5f,2e-5f)&&
       compare_device("cuda_camera.context",dc,context,2e-5f,2e-5f);
    cudaFree(dc);cudaFree(dl);cudaFree(dd);cudaFree(df);
    bf_cuda_camera_neck_destroy(neck);return ok;
}

static int test_downsample(const bf_model *weights,const bf_model *oracle,
                           cudaStream_t stream,char *error,size_t cap) {
    bf_cuda_camera_neck *neck=nullptr;
    int ok=bf_cuda_camera_neck_create(weights,1,2,3,8,10,&neck,error,cap);
    const bf_tensor *input=tensor(oracle,"lss_downsample.input");
    const bf_tensor *output=tensor(oracle,"lss_downsample.output");
    float *di=upload(input),*actual=nullptr;
    ok=ok&&di&&output&&cudaMalloc(&actual,output->nbytes)==cudaSuccess&&
       bf_cuda_camera_downsample_forward(neck,di,actual,(void*)stream,error,cap)&&
       cudaStreamSynchronize(stream)==cudaSuccess&&
       compare_device("cuda_camera.downsample",actual,output,2e-5f,2e-5f);
    cudaFree(actual);cudaFree(di);bf_cuda_camera_neck_destroy(neck);return ok;
}

static int benchmark(const bf_model *weights,char *error,size_t cap) {
    bf_cuda_camera_neck *neck=nullptr;
    int ok=bf_cuda_camera_neck_create(weights,6,32,88,360,360,&neck,error,cap);
    size_t n0=6ull*192*32*88,n1=6ull*384*16*44,n2=6ull*768*8*22;
    size_t nd=6ull*256*704,nl=6ull*118*32*88,nc=6ull*80*32*88;
    size_t full=80ull*360*360,half=80ull*180*180;
    float *s0=nullptr,*s1=nullptr,*s2=nullptr,*depth=nullptr,*logits=nullptr,*context=nullptr;
    float *full_bev=nullptr,*image_bev=nullptr;
    float *frustum=upload(tensor(weights,"vtransform.frustum"));
    float *camera_rotation=nullptr,*camera_translation=nullptr,*intrinsics=nullptr;
    float *post_rotation=nullptr,*post_translation=nullptr,*extra_rotation=nullptr,*extra_translation=nullptr;
    ok=ok&&cudaMalloc(&s0,n0*4)==cudaSuccess&&cudaMalloc(&s1,n1*4)==cudaSuccess&&
       cudaMalloc(&s2,n2*4)==cudaSuccess&&cudaMalloc(&depth,nd*4)==cudaSuccess&&
       cudaMalloc(&logits,nl*4)==cudaSuccess&&cudaMalloc(&context,nc*4)==cudaSuccess&&
       cudaMalloc(&full_bev,full*4)==cudaSuccess&&cudaMalloc(&image_bev,half*4)==cudaSuccess&&frustum&&
       cudaMalloc(&camera_rotation,6*9*4)==cudaSuccess&&cudaMalloc(&camera_translation,6*3*4)==cudaSuccess&&
       cudaMalloc(&intrinsics,6*9*4)==cudaSuccess&&cudaMalloc(&post_rotation,6*9*4)==cudaSuccess&&
       cudaMalloc(&post_translation,6*3*4)==cudaSuccess&&cudaMalloc(&extra_rotation,9*4)==cudaSuccess&&
       cudaMalloc(&extra_translation,3*4)==cudaSuccess;
    if(ok) {
        initialize<<<(n0+255)/256,256>>>(s0,n0,0.002f);
        initialize<<<(n1+255)/256,256>>>(s1,n1,0.002f);
        initialize<<<(n2+255)/256,256>>>(s2,n2,0.002f);
        initialize<<<(nd+255)/256,256>>>(depth,nd,0.01f);
        initialize<<<(full+255)/256,256>>>(full_bev,full,0.002f);
        identity_matrices<<<1,64>>>(camera_rotation,6);identity_matrices<<<1,64>>>(intrinsics,6);
        identity_matrices<<<1,64>>>(post_rotation,6);identity_matrices<<<1,32>>>(extra_rotation,1);
        cudaMemset(camera_translation,0,6*3*4);cudaMemset(post_translation,0,6*3*4);
        cudaMemset(extra_translation,0,3*4);
        ok=cudaDeviceSynchronize()==cudaSuccess;
    }
    cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
    float camera_cold=0,camera_warm=0,down_cold=0,down_warm=0;
    if(ok) { cudaEventRecord(begin);ok=bf_cuda_camera_neck_forward(neck,s0,s1,s2,depth,
        logits,context,nullptr,error,cap);cudaEventRecord(end);cudaEventSynchronize(end);
        cudaEventElapsedTime(&camera_cold,begin,end); }
    if(ok) { cudaEventRecord(begin);for(int i=0;i<10;++i)ok=ok&&bf_cuda_camera_neck_forward(
        neck,s0,s1,s2,depth,logits,context,nullptr,error,cap);
        cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&camera_warm,begin,end); }
    if(ok) { cudaEventRecord(begin);ok=bf_cuda_camera_downsample_forward(neck,full_bev,image_bev,
        nullptr,error,cap);cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&down_cold,begin,end); }
    if(ok) { cudaEventRecord(begin);for(int i=0;i<10;++i)ok=ok&&bf_cuda_camera_downsample_forward(
        neck,full_bev,image_bev,nullptr,error,cap);cudaEventRecord(end);cudaEventSynchronize(end);
        cudaEventElapsedTime(&down_warm,begin,end); }
    bf_lss_desc lss_desc={1,6,118,32,88,80,
        {-54.0f,-54.0f,-10.0f},{0.3f,0.3f,20.0f},{360,360,1}};
    bf_cuda_lss_plan *lss=nullptr;float slice_cold=0,slice_warm=0;
    if(ok)ok=bf_cuda_lss_plan_create(&lss_desc,&lss,error,cap);
    if(ok) { cudaEventRecord(begin);
        ok=bf_cuda_camera_neck_forward(neck,s0,s1,s2,depth,logits,context,nullptr,error,cap)&&
           bf_cuda_lss_plan_prepare_calibration_f32(lss,frustum,camera_rotation,
             camera_translation,intrinsics,post_rotation,post_translation,
             extra_rotation,extra_translation,nullptr,error,cap)&&
           bf_cuda_lss_plan_forward_f32(lss,logits,context,full_bev,nullptr,error,cap)&&
           bf_cuda_camera_downsample_forward(neck,full_bev,image_bev,nullptr,error,cap);
        cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&slice_cold,begin,end); }
    if(ok) { cudaEventRecord(begin);for(int i=0;i<10;++i)ok=ok&&
        bf_cuda_camera_neck_forward(neck,s0,s1,s2,depth,logits,context,nullptr,error,cap)&&
        bf_cuda_lss_plan_prepare_calibration_f32(lss,frustum,camera_rotation,
          camera_translation,intrinsics,post_rotation,post_translation,
          extra_rotation,extra_translation,nullptr,error,cap)&&
        bf_cuda_lss_plan_forward_f32(lss,logits,context,full_bev,nullptr,error,cap)&&
        bf_cuda_camera_downsample_forward(neck,full_bev,image_bev,nullptr,error,cap);
        cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&slice_warm,begin,end); }
    double boundary=(n0+n1+n2+nd+nl+nc+full+half)*4.0/(1024.0*1024.0);
    std::printf("cuda_camera production camera_cold=%.3f ms camera_warm=%.3f ms "
                "down_cold=%.3f ms down_warm=%.3f ms slice_cold=%.3f ms slice_warm=%.3f ms "
                "resident=%.2f MiB boundary=%.2f MiB\n",
        camera_cold,camera_warm/10,down_cold,down_warm/10,
        slice_cold,slice_warm/10,
        (bf_cuda_camera_neck_resident_bytes(neck)+bf_cuda_lss_plan_resident_bytes(lss))/(1024.0*1024.0),boundary);
    bf_cuda_lss_plan_destroy(lss);
    cudaEventDestroy(end);cudaEventDestroy(begin);cudaFree(image_bev);cudaFree(full_bev);
    cudaFree(extra_translation);cudaFree(extra_rotation);cudaFree(post_translation);cudaFree(post_rotation);
    cudaFree(intrinsics);cudaFree(camera_translation);cudaFree(camera_rotation);cudaFree(frustum);
    cudaFree(context);cudaFree(logits);cudaFree(depth);cudaFree(s2);cudaFree(s1);cudaFree(s0);
    bf_cuda_camera_neck_destroy(neck);return ok;
}

int main(int argc,char **argv) {
    if(argc!=5)return 2;
    char error[256]={0};bf_model *weights=nullptr,*fpn=nullptr,*depth=nullptr,*down=nullptr;
    int ok=bf_model_open(argv[1],&weights,error,sizeof(error))&&
        bf_model_open(argv[2],&fpn,error,sizeof(error))&&
        bf_model_open(argv[3],&depth,error,sizeof(error))&&
        bf_model_open(argv[4],&down,error,sizeof(error));
    cudaStream_t stream=nullptr;if(ok)ok=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess;
    if(ok)ok=test_fpn(weights,fpn,stream,error,sizeof(error));
    if(ok)ok=test_depth(weights,depth,stream,error,sizeof(error));
    if(ok)ok=test_downsample(weights,down,stream,error,sizeof(error));
    if(ok)ok=benchmark(weights,error,sizeof(error));
    if(!ok)std::fprintf(stderr,"CUDA camera failure: %s\n",error);
    if(stream)cudaStreamDestroy(stream);bf_model_close(down);bf_model_close(depth);
    bf_model_close(fpn);bf_model_close(weights);return ok?0:5;
}
