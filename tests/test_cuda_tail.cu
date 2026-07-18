#include "bf_cuda_bev.h"
#include "bf_cuda_transfusion.h"
#include "bf_model.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const bf_tensor *get(const bf_model *m,const char *n){const bf_tensor *t=bf_model_find(m,n);if(!t)std::fprintf(stderr,"missing %s\n",n);return t;}
static int close_f32(const char *n,const float *a,const bf_tensor *e){if(!e||e->dtype!=BF_DTYPE_F32)return 0;const float *r=(const float *)e->data;size_t count=(size_t)e->nbytes/4;float maximum=0;double mean=0;
    for(size_t i=0;i<count;++i){float d=std::fabs(a[i]-r[i]);maximum=fmaxf(maximum,d);mean+=d;if(!(d<=6e-5f+6e-5f*std::fabs(r[i]))){std::fprintf(stderr,"%s[%zu] diff %.9g\n",n,i,d);return 0;}}
    std::printf("%-24s max_abs=%.3g mean_abs=%.3g\n",n,maximum,mean/count);return 1;}
static int close_i64(const int64_t *a,const bf_tensor *e){if(!e||e->dtype!=BF_DTYPE_I64)return 0;const int64_t *r=(const int64_t *)e->data;size_t n=(size_t)e->nbytes/8;for(size_t i=0;i<n;++i)if(a[i]!=r[i])return 0;return 1;}
struct output_buffers{float *center,*height,*dim,*rot,*vel,*heat,*scores;int64_t *labels,*indices;};
static int alloc_outputs(output_buffers *b,size_t p){return cudaMalloc((void **)&b->center,2*p*4)==cudaSuccess&&cudaMalloc((void **)&b->height,p*4)==cudaSuccess&&cudaMalloc((void **)&b->dim,3*p*4)==cudaSuccess&&cudaMalloc((void **)&b->rot,2*p*4)==cudaSuccess&&cudaMalloc((void **)&b->vel,2*p*4)==cudaSuccess&&cudaMalloc((void **)&b->heat,10*p*4)==cudaSuccess&&cudaMalloc((void **)&b->scores,10*p*4)==cudaSuccess&&cudaMalloc((void **)&b->labels,p*8)==cudaSuccess&&cudaMalloc((void **)&b->indices,p*8)==cudaSuccess;}
static void free_outputs(output_buffers *b){cudaFree(b->center);cudaFree(b->height);cudaFree(b->dim);cudaFree(b->rot);cudaFree(b->vel);cudaFree(b->heat);cudaFree(b->scores);cudaFree(b->labels);cudaFree(b->indices);}
static bf_transfusion_raw_outputs raw(output_buffers *b){return {b->center,b->height,b->dim,b->rot,b->vel,b->heat,b->scores,b->labels,b->indices};}

static int oracle_test(const bf_model *weights,const bf_model *oracle,char *error,size_t cap){const size_t h=8,w=8,hw=64,p=20;
    bf_cuda_bev_stage *bev=nullptr;bf_cuda_transfusion *decoder=nullptr;float *input=nullptr,*spatial=nullptr,*shared=nullptr,*dense=nullptr;output_buffers outputs={};cudaStream_t stream=nullptr;
    int ok=bf_cuda_bev_stage_create(weights,h,w,&bev,error,cap)&&bf_cuda_transfusion_create(weights,h,w,p,&decoder,error,cap)&&cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess&&
        cudaMalloc((void **)&input,336*hw*4)==cudaSuccess&&cudaMalloc((void **)&spatial,512*hw*4)==cudaSuccess&&cudaMalloc((void **)&shared,128*hw*4)==cudaSuccess&&cudaMalloc((void **)&dense,10*hw*4)==cudaSuccess&&alloc_outputs(&outputs,p);
    const bf_tensor *source=get(oracle,"cuda_tail.input");bf_transfusion_raw_outputs device_raw=raw(&outputs);
    if(ok)ok=source&&cudaMemcpyAsync(input,source->data,(size_t)source->nbytes,cudaMemcpyHostToDevice,stream)==cudaSuccess&&
        bf_cuda_bev_stage_forward(bev,input,spatial,shared,dense,stream,error,cap)&&bf_cuda_transfusion_forward(decoder,shared,dense,&device_raw,stream,error,cap)&&cudaStreamSynchronize(stream)==cudaSuccess;
    size_t float_count=(512+128+10)*hw+(2+1+3+2+2+10+10)*p;float *host=(float *)std::malloc(float_count*4);int64_t *host_i=(int64_t *)std::malloc(2*p*8);if(!host||!host_i)ok=0;
    float *device_arrays[10]={spatial,shared,dense,outputs.center,outputs.height,outputs.dim,outputs.rot,outputs.vel,outputs.heat,outputs.scores};
    size_t counts[10]={512*hw,128*hw,10*hw,2*p,p,3*p,2*p,2*p,10*p,10*p};const char *names[10]={"spatial","shared","dense_heatmap","center","height","dim","rot","vel","heatmap","query_scores"};
    size_t cursor=0;if(ok)for(int i=0;i<10;++i){ok=ok&&cudaMemcpy(host+cursor,device_arrays[i],counts[i]*4,cudaMemcpyDeviceToHost)==cudaSuccess;cursor+=counts[i];}
    if(ok)ok=cudaMemcpy(host_i,outputs.labels,p*8,cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(host_i+p,outputs.indices,p*8,cudaMemcpyDeviceToHost)==cudaSuccess;
    cursor=0;if(ok)for(int i=0;i<10;++i){char full[96];std::snprintf(full,sizeof(full),"cuda_tail.%s",names[i]);ok=ok&&close_f32(full,host+cursor,get(oracle,full));cursor+=counts[i];}
    if(ok)ok=close_i64(host_i,get(oracle,"cuda_tail.labels"))&&close_i64(host_i+p,get(oracle,"cuda_tail.indices"));
    bf_detections detections={};if(ok)ok=bf_cuda_transfusion_decode_detections(decoder,&device_raw,0.0f,&detections,stream,error,cap);
    if(ok){const float *boxes=(const float *)get(oracle,"cuda_tail.boxes")->data,*scores=(const float *)get(oracle,"cuda_tail.scores")->data;
        const int64_t *labels=(const int64_t *)get(oracle,"cuda_tail.final_labels")->data;int expected=0;
        for(size_t proposal=0;proposal<p;++proposal){const float *box=boxes+proposal*9;int keep=scores[proposal]>0&&box[0]>=-61.2f&&box[0]<=61.2f&&box[1]>=-61.2f&&box[1]<=61.2f&&box[2]>=-10&&box[2]<=10;
            if(!keep)continue;if(expected>=detections.count){ok=0;break;}const bf_detection *got=&detections.items[expected++];float values[10]={got->x,got->y,got->z,got->width,got->length,got->height,got->yaw,got->velocity_x,got->velocity_y,got->score};
            for(int axis=0;axis<9;++axis)if(std::fabs(values[axis]-box[axis])>7e-5f+7e-5f*std::fabs(box[axis]))ok=0;
            if(std::fabs(values[9]-scores[proposal])>7e-6f||got->class_id!=labels[proposal])ok=0;}ok=ok&&expected==detections.count;}
    std::free(host);std::free(host_i);free_outputs(&outputs);cudaFree(input);cudaFree(spatial);cudaFree(shared);cudaFree(dense);cudaStreamDestroy(stream);bf_cuda_transfusion_destroy(decoder);bf_cuda_bev_stage_destroy(bev);return ok;}

static int benchmark(const bf_model *weights,char *error,size_t cap){const size_t h=180,w=180,hw=h*w,p=200;bf_cuda_bev_stage *bev=nullptr;bf_cuda_transfusion *decoder=nullptr;
    int ok=bf_cuda_bev_stage_create(weights,h,w,&bev,error,cap)&&bf_cuda_transfusion_create(weights,h,w,p,&decoder,error,cap);float *input=nullptr,*spatial=nullptr,*shared=nullptr,*dense=nullptr;output_buffers outputs={};
    if(ok)ok=cudaMalloc((void **)&input,336*hw*4)==cudaSuccess&&cudaMalloc((void **)&spatial,512*hw*4)==cudaSuccess&&cudaMalloc((void **)&shared,128*hw*4)==cudaSuccess&&cudaMalloc((void **)&dense,10*hw*4)==cudaSuccess&&alloc_outputs(&outputs,p);
    if(ok)cudaMemset(input,0,336*hw*4);bf_transfusion_raw_outputs device_raw=raw(&outputs);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);float cold=0,warm=0;
    bf_detections detections={};
    if(ok){cudaEventRecord(a);ok=bf_cuda_bev_stage_forward(bev,input,spatial,shared,dense,nullptr,error,cap)&&bf_cuda_transfusion_forward(decoder,shared,dense,&device_raw,nullptr,error,cap)&&bf_cuda_transfusion_decode_detections(decoder,&device_raw,0.0f,&detections,nullptr,error,cap);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&cold,a,b);}
    if(ok){cudaEventRecord(a);for(int i=0;i<10&&ok;++i)ok=bf_cuda_bev_stage_forward(bev,input,spatial,shared,dense,nullptr,error,cap)&&bf_cuda_transfusion_forward(decoder,shared,dense,&device_raw,nullptr,error,cap)&&bf_cuda_transfusion_decode_detections(decoder,&device_raw,0.0f,&detections,nullptr,error,cap);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&warm,a,b);warm/=10;}
    std::printf("cuda_tail production cold=%.3f ms warm=%.3f ms contexts=%.2f MiB intermediate_d2h=0 final_d2h=%zu\n",cold,warm,(bf_cuda_bev_stage_resident_bytes(bev)+bf_cuda_transfusion_resident_bytes(decoder))/(1024.0*1024.0),sizeof(bf_detections));
    cudaEventDestroy(a);cudaEventDestroy(b);free_outputs(&outputs);cudaFree(input);cudaFree(spatial);cudaFree(shared);cudaFree(dense);bf_cuda_transfusion_destroy(decoder);bf_cuda_bev_stage_destroy(bev);return ok;}

int main(int argc,char **argv){if(argc!=3)return 2;char error[256]={};bf_model *weights=nullptr,*oracle=nullptr;if(!bf_model_open(argv[1],&weights,error,sizeof(error))||!bf_model_open(argv[2],&oracle,error,sizeof(error))){std::fprintf(stderr,"%s\n",error);return 3;}
    int ok=oracle_test(weights,oracle,error,sizeof(error))&&benchmark(weights,error,sizeof(error));if(!ok)std::fprintf(stderr,"CUDA tail failure: %s\n",error);bf_model_close(oracle);bf_model_close(weights);return ok?0:4;}
