#include "bf_cuda_transfusion.h"
#include "bf_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const bf_tensor *tensor(const bf_model *model,const char *name){const bf_tensor *t=bf_model_find(model,name);if(!t)std::fprintf(stderr,"missing %s\n",name);return t;}
static int close_f32(const char *name,const float *actual,const bf_tensor *expected,float atol,float rtol){
    if(!expected||expected->dtype!=BF_DTYPE_F32)return 0;const float *reference=(const float *)expected->data;size_t count=(size_t)expected->nbytes/4;
    float maximum=0;double mean=0;for(size_t i=0;i<count;++i){float difference=std::fabs(actual[i]-reference[i]);maximum=fmaxf(maximum,difference);mean+=difference;
        if(!(difference<=atol+rtol*std::fabs(reference[i]))){std::fprintf(stderr,"%s[%zu] got %.9g expected %.9g diff %.9g\n",name,i,actual[i],reference[i],difference);return 0;}}
    std::printf("%-28s max_abs=%.3g mean_abs=%.3g\n",name,maximum,mean/count);return 1;}
static int close_i64(const char *name,const int64_t *actual,const bf_tensor *expected){if(!expected||expected->dtype!=BF_DTYPE_I64)return 0;
    size_t count=(size_t)expected->nbytes/8;const int64_t *reference=(const int64_t *)expected->data;for(size_t i=0;i<count;++i)if(actual[i]!=reference[i]){
        std::fprintf(stderr,"%s[%zu] got %lld expected %lld\n",name,i,(long long)actual[i],(long long)reference[i]);return 0;}return 1;}

struct buffers {float *center,*height,*dim,*rot,*vel,*heatmap,*scores;int64_t *labels,*indices;};
static int allocate(buffers *b,size_t p){return cudaMalloc((void **)&b->center,2*p*4)==cudaSuccess&&cudaMalloc((void **)&b->height,p*4)==cudaSuccess&&
    cudaMalloc((void **)&b->dim,3*p*4)==cudaSuccess&&cudaMalloc((void **)&b->rot,2*p*4)==cudaSuccess&&cudaMalloc((void **)&b->vel,2*p*4)==cudaSuccess&&
    cudaMalloc((void **)&b->heatmap,10*p*4)==cudaSuccess&&cudaMalloc((void **)&b->scores,10*p*4)==cudaSuccess&&
    cudaMalloc((void **)&b->labels,p*8)==cudaSuccess&&cudaMalloc((void **)&b->indices,p*8)==cudaSuccess;}
static void release(buffers *b){cudaFree(b->center);cudaFree(b->height);cudaFree(b->dim);cudaFree(b->rot);cudaFree(b->vel);cudaFree(b->heatmap);cudaFree(b->scores);cudaFree(b->labels);cudaFree(b->indices);}
static bf_transfusion_raw_outputs raw(buffers *b){return {b->center,b->height,b->dim,b->rot,b->vel,b->heatmap,b->scores,b->labels,b->indices};}

static int oracle_test(const bf_model *weights,const bf_model *oracle,char *error,size_t cap){const size_t h=5,w=5,p=12,k=h*w;
    bf_cuda_transfusion *decoder=nullptr;if(!bf_cuda_transfusion_create(weights,h,w,p,&decoder,error,cap))return 0;
    float *shared=nullptr,*dense=nullptr;buffers device={};cudaStream_t stream=nullptr;cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    const bf_tensor *shared_source=tensor(oracle,"transfusion_decoder.shared"),*dense_source=tensor(oracle,"transfusion_decoder.dense_heatmap");
    int ok=shared_source&&dense_source&&cudaMalloc((void **)&shared,128*k*4)==cudaSuccess&&cudaMalloc((void **)&dense,10*k*4)==cudaSuccess&&allocate(&device,p);
    bf_transfusion_raw_outputs output=raw(&device);
    if(ok)ok=cudaMemcpyAsync(shared,shared_source->data,(size_t)shared_source->nbytes,cudaMemcpyHostToDevice,stream)==cudaSuccess&&
        cudaMemcpyAsync(dense,dense_source->data,(size_t)dense_source->nbytes,cudaMemcpyHostToDevice,stream)==cudaSuccess&&
        bf_cuda_transfusion_forward(decoder,shared,dense,&output,stream,error,cap)&&cudaStreamSynchronize(stream)==cudaSuccess;
    float *host=(float *)std::malloc(30*p*4);int64_t *host_i=(int64_t *)std::malloc(2*p*8);if(!host||!host_i)ok=0;
    size_t offsets[7]={0,2*p,3*p,6*p,8*p,10*p,20*p};float *sources[7]={device.center,device.height,device.dim,device.rot,device.vel,device.heatmap,device.scores};
    size_t counts[7]={2*p,p,3*p,2*p,2*p,10*p,10*p};const char *names[7]={"center","height","dim","rot","vel","heatmap","query_scores"};
    if(ok)for(int i=0;i<7;++i)ok=ok&&cudaMemcpy(host+offsets[i],sources[i],counts[i]*4,cudaMemcpyDeviceToHost)==cudaSuccess;
    if(ok)ok=cudaMemcpy(host_i,device.labels,p*8,cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(host_i+p,device.indices,p*8,cudaMemcpyDeviceToHost)==cudaSuccess;
    if(ok)for(int i=0;i<7;++i){char full[96];std::snprintf(full,sizeof(full),"transfusion_decoder.%s",names[i]);ok=ok&&close_f32(full,host+offsets[i],tensor(oracle,full),5e-5f,5e-5f);}
    if(ok)ok=close_i64("labels",host_i,tensor(oracle,"transfusion_decoder.labels"))&&close_i64("indices",host_i+p,tensor(oracle,"transfusion_decoder.indices"));
    float *boxes=(float *)std::malloc(p*9*4),*final_scores=(float *)std::malloc(p*4);
    if(!boxes||!final_scores)ok=0;
    if(ok)for(size_t proposal=0;proposal<p;++proposal){int64_t label=host_i[proposal];
        boxes[proposal*9]=(host[offsets[0]+proposal])*0.6f-54.0f;
        boxes[proposal*9+1]=(host[offsets[0]+p+proposal])*0.6f-54.0f;
        boxes[proposal*9+2]=host[offsets[1]+proposal];
        for(int axis=0;axis<3;++axis)boxes[proposal*9+3+axis]=std::exp(host[offsets[2]+axis*p+proposal]);
        boxes[proposal*9+6]=std::atan2(host[offsets[3]+proposal],host[offsets[3]+p+proposal]);
        boxes[proposal*9+7]=host[offsets[4]+proposal];boxes[proposal*9+8]=host[offsets[4]+p+proposal];
        float logit=host[offsets[5]+(size_t)label*p+proposal];
        final_scores[proposal]=(1.0f/(1.0f+std::exp(-logit)))*host[offsets[6]+(size_t)label*p+proposal];}
    if(ok)ok=close_f32("transfusion_decoder.boxes",boxes,tensor(oracle,"transfusion_decoder.boxes"),5e-5f,5e-5f)&&
        close_f32("transfusion_decoder.scores",final_scores,tensor(oracle,"transfusion_decoder.scores"),5e-6f,5e-5f)&&
        close_i64("final_labels",host_i,tensor(oracle,"transfusion_decoder.final_labels"));
    bf_detections detections={};
    if(ok)ok=bf_cuda_transfusion_decode_detections(decoder,&output,0.0f,&detections,stream,error,cap);
    if(ok){const float *oracle_boxes=(const float *)tensor(oracle,"transfusion_decoder.boxes")->data;
        const float *oracle_scores=(const float *)tensor(oracle,"transfusion_decoder.scores")->data;
        const int64_t *oracle_labels=(const int64_t *)tensor(oracle,"transfusion_decoder.final_labels")->data;int expected=0;
        for(size_t proposal=0;proposal<p;++proposal){const float *box=oracle_boxes+proposal*9;
            int keep=oracle_scores[proposal]>0&&box[0]>=-61.2f&&box[0]<=61.2f&&box[1]>=-61.2f&&box[1]<=61.2f&&box[2]>=-10&&box[2]<=10;
            if(!keep)continue;if(expected>=detections.count){ok=0;break;}const bf_detection *got=&detections.items[expected++];
            float values[10]={got->x,got->y,got->z,got->width,got->length,got->height,got->yaw,got->velocity_x,got->velocity_y,got->score};
            for(int axis=0;axis<9;++axis)if(std::fabs(values[axis]-box[axis])>6e-5f+6e-5f*std::fabs(box[axis]))ok=0;
            if(std::fabs(values[9]-oracle_scores[proposal])>6e-6f||got->class_id!=oracle_labels[proposal])ok=0;}
        ok=ok&&expected==detections.count;}
    std::free(boxes);std::free(final_scores);
    std::free(host);std::free(host_i);release(&device);cudaFree(shared);cudaFree(dense);cudaStreamDestroy(stream);bf_cuda_transfusion_destroy(decoder);return ok;}

static int benchmark(const bf_model *weights,char *error,size_t cap){const size_t h=180,w=180,p=200,k=h*w;
    bf_cuda_transfusion *decoder=nullptr;cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
    int ok=bf_cuda_transfusion_create(weights,h,w,p,&decoder,error,cap);cudaEventRecord(b);cudaEventSynchronize(b);float create=0;cudaEventElapsedTime(&create,a,b);
    float *shared=nullptr,*dense=nullptr;buffers device={};if(ok)ok=cudaMalloc((void **)&shared,128*k*4)==cudaSuccess&&cudaMalloc((void **)&dense,10*k*4)==cudaSuccess&&allocate(&device,p);
    if(ok){cudaMemset(shared,0,128*k*4);cudaMemset(dense,0,10*k*4);}bf_transfusion_raw_outputs output=raw(&device);float cold=0,warm=0;
    if(ok){cudaEventRecord(a);ok=bf_cuda_transfusion_forward(decoder,shared,dense,&output,nullptr,error,cap);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&cold,a,b);}
    if(ok){cudaEventRecord(a);for(int i=0;i<10&&ok;++i)ok=bf_cuda_transfusion_forward(decoder,shared,dense,&output,nullptr,error,cap);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&warm,a,b);warm/=10;}
    std::printf("cuda_transfusion production create=%.3f ms cold=%.3f ms warm=%.3f ms resident=%.2f MiB\n",create,cold,warm,
        bf_cuda_transfusion_resident_bytes(decoder)/(1024.0*1024.0));
    release(&device);cudaFree(shared);cudaFree(dense);bf_cuda_transfusion_destroy(decoder);cudaEventDestroy(a);cudaEventDestroy(b);return ok;}

int main(int argc,char **argv){if(argc!=3)return 2;char error[256]={};bf_model *weights=nullptr,*oracle=nullptr;
    if(!bf_model_open(argv[1],&weights,error,sizeof(error))||!bf_model_open(argv[2],&oracle,error,sizeof(error))){std::fprintf(stderr,"%s\n",error);return 3;}
    bf_cuda_transfusion *invalid=nullptr;int ok=!bf_cuda_transfusion_create(weights,180,180,201,&invalid,error,sizeof(error))&&oracle_test(weights,oracle,error,sizeof(error))&&benchmark(weights,error,sizeof(error));
    if(!ok)std::fprintf(stderr,"CUDA TransFusion failure: %s\n",error);bf_cuda_transfusion_destroy(invalid);bf_model_close(oracle);bf_model_close(weights);return ok?0:4;}
