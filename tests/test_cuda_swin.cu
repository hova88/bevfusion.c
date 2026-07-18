#include "bf_cuda_swin.h"
#include "bf_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const bf_tensor *tensor(const bf_model *m,const char *name){const bf_tensor *t=bf_model_find(m,name);if(!t)std::fprintf(stderr,"missing %s\n",name);return t;}
static float *upload(const bf_tensor *t){float *d=nullptr;if(!t||cudaMalloc(&d,t->nbytes)!=cudaSuccess||cudaMemcpy(d,t->data,t->nbytes,cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(d);return nullptr;}return d;}
static int compare(const char *name,const float *device,const bf_tensor *expected){float *actual=(float*)std::malloc(expected->nbytes);if(!actual||cudaMemcpy(actual,device,expected->nbytes,cudaMemcpyDeviceToHost)!=cudaSuccess){std::free(actual);return 0;}size_t n=expected->nbytes/4;const float *ref=(const float*)expected->data;float maximum=0;double mean=0;int ok=1;for(size_t i=0;i<n;++i){float d=std::fabs(actual[i]-ref[i]);maximum=fmaxf(maximum,d);mean+=d;if(!(d<=6e-5f+6e-5f*std::fabs(ref[i])))ok=0;}std::printf("%-28s max_abs=%.3g mean_abs=%.3g\n",name,maximum,mean/n);std::free(actual);return ok;}
__global__ static void initialize(float *values,size_t count){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)values[i]=((int)(i%31)-15)*0.01f;}

int main(int argc,char **argv){if(argc!=3)return 2;char error[256]={0};bf_model *weights=nullptr,*oracle=nullptr;
 int ok=bf_model_open(argv[1],&weights,error,sizeof(error))&&bf_model_open(argv[2],&oracle,error,sizeof(error));
 bf_cuda_swin *small=nullptr;cudaStream_t stream=nullptr;
 if(std::getenv("BF_CUDA_SWIN_TEST_CHUNK_BOUNDARY"))setenv("BF_CUDA_SWIN_WINDOW_CHUNK","2",1);
 if(ok)ok=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess&&bf_cuda_swin_create(weights,1,32,48,&small,error,sizeof(error));
 const bf_tensor *input=tensor(oracle,"swin_backbone.input"),*expected[3]={tensor(oracle,"swin_backbone.output0"),tensor(oracle,"swin_backbone.output1"),tensor(oracle,"swin_backbone.output2")};
 float *di=upload(input),*out[3]={nullptr,nullptr,nullptr};for(int i=0;i<3&&ok;++i)ok=expected[i]&&cudaMalloc(&out[i],expected[i]->nbytes)==cudaSuccess;
 if(ok)ok=di&&bf_cuda_swin_forward(small,di,out[0],out[1],out[2],(void*)stream,error,sizeof(error))&&cudaStreamSynchronize(stream)==cudaSuccess;
 for(int i=0;i<3&&ok;++i){char name[40];std::snprintf(name,sizeof(name),"cuda_swin.output%d",i);ok=compare(name,out[i],expected[i]);}
 for(int i=0;i<3;++i)cudaFree(out[i]);cudaFree(di);bf_cuda_swin_destroy(small);
 if(std::getenv("BF_CUDA_SWIN_TEST_CHUNK_BOUNDARY"))unsetenv("BF_CUDA_SWIN_WINDOW_CHUNK");
 bf_cuda_swin *production=nullptr;float *images=nullptr,*p0=nullptr,*p1=nullptr,*p2=nullptr;
 size_t ni=6ull*3*256*704,n0=6ull*192*32*88,n1=6ull*384*16*44,n2=6ull*768*8*22;
 if(ok)ok=bf_cuda_swin_create(weights,6,256,704,&production,error,sizeof(error))&&cudaMalloc(&images,ni*4)==cudaSuccess&&cudaMalloc(&p0,n0*4)==cudaSuccess&&cudaMalloc(&p1,n1*4)==cudaSuccess&&cudaMalloc(&p2,n2*4)==cudaSuccess;
 if(ok){initialize<<<(ni+255)/256,256>>>(images,ni);ok=cudaDeviceSynchronize()==cudaSuccess;}
 cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);float cold=0,warm=0;
 if(ok){cudaEventRecord(begin);ok=bf_cuda_swin_forward(production,images,p0,p1,p2,nullptr,error,sizeof(error));cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&cold,begin,end);}
 if(ok){cudaEventRecord(begin);for(int i=0;i<5;++i)ok=ok&&bf_cuda_swin_forward(production,images,p0,p1,p2,nullptr,error,sizeof(error));cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&warm,begin,end);}
 std::printf("cuda_swin production cold=%.3f ms warm=%.3f ms resident=%.2f MiB boundary=%.2f MiB\n",cold,warm/5,bf_cuda_swin_resident_bytes(production)/(1024.0*1024.0),(ni+n0+n1+n2)*4.0/(1024.0*1024.0));
 if(!ok)std::fprintf(stderr,"CUDA Swin failure: %s\n",error);cudaEventDestroy(end);cudaEventDestroy(begin);cudaFree(p2);cudaFree(p1);cudaFree(p0);cudaFree(images);bf_cuda_swin_destroy(production);if(stream)cudaStreamDestroy(stream);bf_model_close(oracle);bf_model_close(weights);return ok?0:5;}
