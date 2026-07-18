#include "bf_cuda_runtime.h"
#include "bf_frame.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static float detection_drift(const bf_detections *a,const bf_detections *b){
    if(a->count!=b->count)return INFINITY;float maximum=0;bool used[BF_MAX_PROPOSALS]={};
    for(int i=0;i<a->count;++i){int match=-1;float best=INFINITY;
        for(int j=0;j<b->count;++j)if(!used[j]&&a->items[i].class_id==b->items[j].class_id){
            float dx=a->items[i].x-b->items[j].x,dy=a->items[i].y-b->items[j].y,dz=a->items[i].z-b->items[j].z;
            float distance=dx*dx+dy*dy+dz*dz;if(distance<best){best=distance;match=j;}}
        if(match<0)return INFINITY;used[match]=true;const float *x=&a->items[i].x,*y=&b->items[match].x;
        for(int j=0;j<10;++j)maximum=fmaxf(maximum,fabsf(x[j]-y[j]));}
    return maximum;
}

int main(int argc,char **argv){
    if(argc!=3)return 2;char error[256]={};bf_frame_file *file=nullptr;
    bf_cuda_runtime *runtime=nullptr;
    int ok=bf_frame_open(argv[2],&file,error,sizeof(error))&&
        bf_cuda_runtime_create(argv[1],300000,160000,160000,&runtime,error,sizeof(error));
    bf_detections first={},second={};double cold=0,warm=0;unsigned long long first_hashes[11]={},second_hashes[11]={};
    if(ok){auto begin=std::chrono::steady_clock::now();ok=bf_cuda_runtime_infer(runtime,bf_frame_input_view(file),&first,error,sizeof(error));auto end=std::chrono::steady_clock::now();cold=std::chrono::duration<double,std::milli>(end-begin).count();if(ok)ok=bf_cuda_runtime_debug_hashes(runtime,first_hashes,error,sizeof(error));}
    int warm_runs=1,warm_repeat=1,exact_repeat=1,warm_mismatch_at=-1,executed=0;float max_drift=0;if(const char *text=std::getenv("BF_CUDA_RUNTIME_WARM_RUNS")){int parsed=std::atoi(text);if(parsed>0&&parsed<=100)warm_runs=parsed;}
    if(ok){auto begin=std::chrono::steady_clock::now();for(int i=0;i<warm_runs&&ok;++i){bf_detections current={};ok=bf_cuda_runtime_infer(runtime,bf_frame_input_view(file),&current,error,sizeof(error));if(ok&&std::getenv("BF_CUDA_RUNTIME_TRACE_HASHES")){unsigned long long current_hashes[11]={};ok=bf_cuda_runtime_debug_hashes(runtime,current_hashes,error,sizeof(error));if(ok){std::fprintf(stderr,"warm[%d]",i);for(int h=0;h<11;++h)if(current_hashes[h]!=first_hashes[h])std::fprintf(stderr," %d=%016llx",h,current_hashes[h]);std::fputc('\n',stderr);}}if(ok){++executed;if(i==0)second=current;else{float drift=detection_drift(&second,&current);max_drift=fmaxf(max_drift,drift);if(std::memcmp(&second,&current,sizeof(second))!=0)exact_repeat=0;if(!(drift<=2.5e-2f)){warm_repeat=0;warm_mismatch_at=i;second=current;break;}}}}auto end=std::chrono::steady_clock::now();warm=executed?std::chrono::duration<double,std::milli>(end-begin).count()/executed:0;if(ok)ok=bf_cuda_runtime_debug_hashes(runtime,second_hashes,error,sizeof(error));}
    {float drift=detection_drift(&first,&second);max_drift=fmaxf(max_drift,drift);if(std::memcmp(&first,&second,sizeof(first))!=0)exact_repeat=0;if(!(drift<=2.5e-2f))warm_repeat=0;}
    int repeat=ok&&warm_repeat;float stats[4]={};
    if(ok&&!repeat){
        const char *names[11]={"depth","swin0","swin1","swin2","logits","context","full_image","image","lidar","shared","heat"};
        for(int i=0;i<11;++i)if(first_hashes[i]!=second_hashes[i])std::fprintf(stderr,"hash mismatch %s=%016llx/%016llx\n",names[i],first_hashes[i],second_hashes[i]);
        std::fprintf(stderr,"repeat mismatch warm_iteration=%d count=%d/%d\n",warm_mismatch_at,first.count,second.count);
        for(int i=0;i<first.count&&i<second.count;++i)
            if(std::memcmp(&first.items[i],&second.items[i],sizeof(first.items[i]))!=0){
                const bf_detection &a=first.items[i],&b=second.items[i];
                std::fprintf(stderr,"first mismatch i=%d class=%d/%d score=%.9g/%.9g xyz=(%.9g,%.9g,%.9g)/(%.9g,%.9g,%.9g)\n",
                    i,a.class_id,b.class_id,a.score,b.score,a.x,a.y,a.z,b.x,b.y,b.z);break;
            }
    }
    if(ok)ok=bf_cuda_runtime_debug_bev_stats(runtime,stats,error,sizeof(error));
    int invalid_gates=0;if(ok){bf_frame_input oversized=*bf_frame_input_view(file);
        oversized.point_count=300001;bf_detections ignored={};char rejected[128]={};
        int rejects_oversized=!bf_cuda_runtime_infer(runtime,&oversized,&ignored,rejected,sizeof(rejected));
        bf_frame_input singular=*bf_frame_input_view(file);float zero_matrix[16]={};
        singular.lidar_augmentation_16=zero_matrix;rejected[0]=0;
        int rejects_singular=!bf_cuda_runtime_infer(runtime,&singular,&ignored,rejected,sizeof(rejected));
        invalid_gates=rejects_oversized&&rejects_singular;ok=ok&&invalid_gates;}
    std::printf("cuda_runtime real_frame detections=%d cold_wall=%.3f ms warm_wall=%.3f ms resident=%.2f MiB repeat=%s max_drift=%.3g image_l1/max=%.3g/%.3g lidar_l1/max=%.3g/%.3g\n",first.count,cold,warm,bf_cuda_runtime_resident_bytes(runtime)/(1024.0*1024.0),repeat&&invalid_gates?(exact_repeat?"bit-exact+invalid-gates":"bounded-drift+invalid-gates"):"FAIL",max_drift,stats[0],stats[1],stats[2],stats[3]);
    if(!ok||!repeat)std::fprintf(stderr,"CUDA runtime failure: %s\n",error);
    bf_cuda_runtime_destroy(runtime);bf_frame_close(file);return ok&&repeat?0:5;
}
