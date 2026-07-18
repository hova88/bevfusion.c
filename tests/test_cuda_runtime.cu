#include "bf_cuda_runtime.h"
#include "bf_frame.h"

#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc,char **argv){
    if(argc!=3)return 2;char error[256]={};bf_frame_file *file=nullptr;
    bf_cuda_runtime *runtime=nullptr;
    int ok=bf_frame_open(argv[2],&file,error,sizeof(error))&&
        bf_cuda_runtime_create(argv[1],300000,160000,160000,&runtime,error,sizeof(error));
    bf_detections first={},second={};double cold=0,warm=0;
    if(ok){auto begin=std::chrono::steady_clock::now();ok=bf_cuda_runtime_infer(runtime,bf_frame_input_view(file),&first,error,sizeof(error));auto end=std::chrono::steady_clock::now();cold=std::chrono::duration<double,std::milli>(end-begin).count();}
    if(ok){auto begin=std::chrono::steady_clock::now();ok=bf_cuda_runtime_infer(runtime,bf_frame_input_view(file),&second,error,sizeof(error));auto end=std::chrono::steady_clock::now();warm=std::chrono::duration<double,std::milli>(end-begin).count();}
    int repeat=ok&&std::memcmp(&first,&second,sizeof(first))==0;float stats[4]={};
    if(ok)ok=bf_cuda_runtime_debug_bev_stats(runtime,stats,error,sizeof(error));
    int invalid_gates=0;if(ok){bf_frame_input oversized=*bf_frame_input_view(file);
        oversized.point_count=300001;bf_detections ignored={};char rejected[128]={};
        int rejects_oversized=!bf_cuda_runtime_infer(runtime,&oversized,&ignored,rejected,sizeof(rejected));
        bf_frame_input singular=*bf_frame_input_view(file);float zero_matrix[16]={};
        singular.lidar_augmentation_16=zero_matrix;rejected[0]=0;
        int rejects_singular=!bf_cuda_runtime_infer(runtime,&singular,&ignored,rejected,sizeof(rejected));
        invalid_gates=rejects_oversized&&rejects_singular;ok=ok&&invalid_gates;}
    std::printf("cuda_runtime real_frame detections=%d cold_wall=%.3f ms warm_wall=%.3f ms resident=%.2f MiB repeat=%s image_l1/max=%.3g/%.3g lidar_l1/max=%.3g/%.3g\n",first.count,cold,warm,bf_cuda_runtime_resident_bytes(runtime)/(1024.0*1024.0),repeat&&invalid_gates?"bit-exact+invalid-gates":"FAIL",stats[0],stats[1],stats[2],stats[3]);
    if(!ok||!repeat)std::fprintf(stderr,"CUDA runtime failure: %s\n",error);
    bf_cuda_runtime_destroy(runtime);bf_frame_close(file);return ok&&repeat?0:5;
}
