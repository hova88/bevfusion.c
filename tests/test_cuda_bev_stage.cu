#include "bf_cuda_bev.h"
#include "bf_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) std::fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

static uint64_t hash_bytes(const void *data,size_t bytes) {
    const unsigned char *p=(const unsigned char *)data;uint64_t h=1469598103934665603ull;
    for(size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ull;}return h;
}

static int compare(const char *name, const float *actual,
                   const bf_tensor *expected, float atol, float rtol) {
    if (!expected || expected->dtype != BF_DTYPE_F32) return 0;
    const float *reference = (const float *)expected->data;
    size_t count = (size_t)expected->nbytes / sizeof(float);
    float maximum = 0.0f; double mean = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float difference = std::fabs(actual[i] - reference[i]);
        if (difference > maximum) maximum = difference;
        mean += difference;
        if (!(difference <= atol + rtol * std::fabs(reference[i]))) {
            std::fprintf(stderr, "%s[%zu] got %.9g expected %.9g diff %.9g\n",
                         name, i, actual[i], reference[i], difference);
            return 0;
        }
    }
    std::printf("%-24s max_abs=%.3g mean_abs=%.3g\n", name, maximum,
                mean / (double)count);
    return 1;
}

static int oracle_test(const bf_model *weights, const bf_model *oracle,
                       char *error, size_t cap) {
    const size_t h=8,w=8,hw=h*w;
    bf_cuda_bev_stage *stage=nullptr;
    if (!bf_cuda_bev_stage_create(weights,h,w,&stage,error,cap)) return 0;
    float *input=nullptr,*spatial=nullptr,*shared=nullptr,*heatmap=nullptr;
    cudaStream_t stream=nullptr;
    cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    cudaMalloc((void **)&input,336*hw*sizeof(float));
    cudaMalloc((void **)&spatial,512*hw*sizeof(float));
    cudaMalloc((void **)&shared,128*hw*sizeof(float));
    cudaMalloc((void **)&heatmap,10*hw*sizeof(float));
    float *host_spatial=(float *)std::malloc(512*hw*sizeof(float));
    float *host_shared=(float *)std::malloc(128*hw*sizeof(float));
    float *host_heatmap=(float *)std::malloc(10*hw*sizeof(float));
    const bf_tensor *source=tensor(oracle,"bev_stage.input");
    int ok=input&&spatial&&shared&&heatmap&&host_spatial&&host_shared&&host_heatmap&&source;
    if (ok) ok = cudaMemcpy(input,source->data,(size_t)source->nbytes,
                             cudaMemcpyHostToDevice)==cudaSuccess &&
        bf_cuda_bev_stage_forward(stage,input,spatial,shared,heatmap,stream,error,cap) &&
        cudaStreamSynchronize(stream)==cudaSuccess &&
        cudaMemcpy(host_spatial,spatial,512*hw*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess &&
        cudaMemcpy(host_shared,shared,128*hw*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess &&
        cudaMemcpy(host_heatmap,heatmap,10*hw*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess &&
        compare("cuda_bev.spatial",host_spatial,tensor(oracle,"bev_stage.spatial"),2e-4f,2e-4f) &&
        compare("cuda_bev.shared",host_shared,tensor(oracle,"bev_stage.shared"),2e-4f,2e-4f) &&
        compare("cuda_bev.heatmap",host_heatmap,tensor(oracle,"bev_stage.heatmap"),2e-4f,2e-4f) &&
        bf_cuda_bev_stage_resident_bytes(stage)>0;
    uint64_t first_hash[3]={0,0,0};
    if(ok){first_hash[0]=hash_bytes(host_spatial,512*hw*sizeof(float));
        first_hash[1]=hash_bytes(host_shared,128*hw*sizeof(float));
        first_hash[2]=hash_bytes(host_heatmap,10*hw*sizeof(float));}
    if(ok) ok=bf_cuda_bev_stage_forward(stage,input,spatial,shared,heatmap,stream,error,cap)&&
        cudaStreamSynchronize(stream)==cudaSuccess&&
        cudaMemcpy(host_spatial,spatial,512*hw*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess&&
        cudaMemcpy(host_shared,shared,128*hw*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess&&
        cudaMemcpy(host_heatmap,heatmap,10*hw*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess&&
        first_hash[0]==hash_bytes(host_spatial,512*hw*sizeof(float))&&
        first_hash[1]==hash_bytes(host_shared,128*hw*sizeof(float))&&
        first_hash[2]==hash_bytes(host_heatmap,10*hw*sizeof(float));
    cudaFree(input);cudaFree(spatial);cudaFree(shared);cudaFree(heatmap);
    cudaStreamDestroy(stream);
    std::free(host_spatial);std::free(host_shared);std::free(host_heatmap);
    bf_cuda_bev_stage_destroy(stage);
    return ok;
}

static int production_benchmark(const bf_model *weights, char *error, size_t cap) {
    const size_t h=180,w=180,hw=h*w;
    cudaEvent_t begin,end; cudaEventCreate(&begin); cudaEventCreate(&end);
    cudaEventRecord(begin);
    bf_cuda_bev_stage *stage=nullptr;
    int ok=bf_cuda_bev_stage_create(weights,h,w,&stage,error,cap);
    cudaEventRecord(end);cudaEventSynchronize(end);
    float create_ms=0;cudaEventElapsedTime(&create_ms,begin,end);
    float *input=nullptr,*spatial=nullptr,*shared=nullptr,*heatmap=nullptr;
    if(ok) {
        ok=cudaMalloc((void **)&input,336*hw*sizeof(float))==cudaSuccess&&
           cudaMalloc((void **)&spatial,512*hw*sizeof(float))==cudaSuccess&&
           cudaMalloc((void **)&shared,128*hw*sizeof(float))==cudaSuccess&&
           cudaMalloc((void **)&heatmap,10*hw*sizeof(float))==cudaSuccess;
    }
    if(ok) cudaMemset(input,0,336*hw*sizeof(float));
    float cold=0,warm=0;
    if(ok) {
        cudaEventRecord(begin);
        ok=bf_cuda_bev_stage_forward(stage,input,spatial,shared,heatmap,nullptr,error,cap);
        cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&cold,begin,end);
    }
    if(ok) {
        cudaEventRecord(begin);
        for(int i=0;i<10&&ok;++i)
            ok=bf_cuda_bev_stage_forward(stage,input,spatial,shared,heatmap,nullptr,error,cap);
        cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&warm,begin,end);
        warm/=10.0f;
    }
    size_t external=(336+512+128+10)*hw*sizeof(float);
    std::printf("cuda_bev production create=%.3f ms cold=%.3f ms warm=%.3f ms "
                "resident=%.2f MiB external=%.2f MiB\n",create_ms,cold,warm,
                bf_cuda_bev_stage_resident_bytes(stage)/(1024.0*1024.0),
                external/(1024.0*1024.0));
    cudaFree(input);cudaFree(spatial);cudaFree(shared);cudaFree(heatmap);
    bf_cuda_bev_stage_destroy(stage);cudaEventDestroy(begin);cudaEventDestroy(end);
    return ok;
}

int main(int argc,char **argv) {
    if(argc!=3)return 2;
    char error[256]={0};bf_model *weights=nullptr,*oracle=nullptr;
    if(!bf_model_open(argv[1],&weights,error,sizeof(error))||
       !bf_model_open(argv[2],&oracle,error,sizeof(error))) {
        std::fprintf(stderr,"%s\n",error);return 3;
    }
    bf_cuda_bev_stage *invalid=nullptr;
    int ok=!bf_cuda_bev_stage_create(weights,7,8,&invalid,error,sizeof(error))&&
           oracle_test(weights,oracle,error,sizeof(error))&&
           production_benchmark(weights,error,sizeof(error));
    if(!ok)std::fprintf(stderr,"CUDA BEV stage failure: %s\n",error);
    bf_cuda_bev_stage_destroy(invalid);bf_model_close(oracle);bf_model_close(weights);
    return ok?0:4;
}
