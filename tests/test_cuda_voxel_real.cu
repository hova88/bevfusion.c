#include "bf_cuda_voxel.h"
#include "bf_frame.h"
#include "bf_voxel.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc,char **argv){
    if(argc!=2)return 2;char error[256]={};bf_frame_file *file=nullptr;
    int ok=bf_frame_open(argv[1],&file,error,sizeof(error));
    const bf_frame_input *frame=bf_frame_input_view(file);
    bf_voxel_config config={{-54,-54,-5},{54,54,3},{.075f,.075f,.2f},5,10,160000};
    size_t wb=bf_voxelize_workspace_bytes(&config),cpu_count=0;
    float *voxels=(float*)std::malloc(160000ull*10*5*4);
    float *cpu_features=(float*)std::malloc(160000*5*4);
    bf_coord4 *cpu_coords=(bf_coord4*)std::malloc(160000*sizeof(bf_coord4));
    int64_t *counts=(int64_t*)std::malloc(160000*8);void *workspace=std::malloc(wb);
    bf_voxel_stats stats={};
    if(ok)ok=voxels&&cpu_features&&cpu_coords&&counts&&workspace&&
        bf_voxelize_f32_workspace_ref(frame->points,frame->point_count,5,0,&config,
            voxels,cpu_coords,counts,&cpu_count,&stats,workspace,wb);
    if(ok)bf_mean_vfe_f32_ref(voxels,counts,cpu_features,cpu_count,10,5);

    bf_cuda_voxelizer *voxelizer=nullptr;float *points=nullptr,*features=nullptr;
    bf_coord4 *coords=nullptr;unsigned *count=nullptr;
    if(ok)ok=bf_cuda_voxelizer_create(&config,300000,&voxelizer,error,sizeof(error))&&
        cudaMalloc(&points,frame->point_count*5*4)==cudaSuccess&&
        cudaMalloc(&features,160000*5*4)==cudaSuccess&&
        cudaMalloc(&coords,160000*sizeof(bf_coord4))==cudaSuccess&&
        cudaMalloc(&count,4)==cudaSuccess&&
        cudaMemcpy(points,frame->points,frame->point_count*5*4,cudaMemcpyHostToDevice)==cudaSuccess&&
        bf_cuda_voxelize_mean_f32(voxelizer,points,frame->point_count,coords,
            features,count,nullptr,error,sizeof(error))&&cudaDeviceSynchronize()==cudaSuccess;
    unsigned gpu_count=0;bf_coord4 *gpu_coords=(bf_coord4*)std::malloc(160000*sizeof(bf_coord4));
    float *gpu_features=(float*)std::malloc(160000*5*4);
    if(ok)ok=gpu_coords&&gpu_features&&cudaMemcpy(&gpu_count,count,4,cudaMemcpyDeviceToHost)==cudaSuccess&&
        cudaMemcpy(gpu_coords,coords,gpu_count*sizeof(bf_coord4),cudaMemcpyDeviceToHost)==cudaSuccess&&
        cudaMemcpy(gpu_features,features,gpu_count*5*4,cudaMemcpyDeviceToHost)==cudaSuccess;
    size_t coord_diff=0,feature_diff=0;float max_abs=0;
    if(ok){for(size_t i=0;i<cpu_count&&i<gpu_count;++i){
        if(std::memcmp(cpu_coords+i,gpu_coords+i,sizeof(bf_coord4)))++coord_diff;
        for(int c=0;c<5;++c){float d=fabsf(cpu_features[i*5+c]-gpu_features[i*5+c]);
            max_abs=fmaxf(max_abs,d);if(d>2e-5f)++feature_diff;}}
        ok=cpu_count==gpu_count&&!coord_diff&&!feature_diff;}
    std::printf("cuda_voxel_real cpu=%zu gpu=%u coord_diff=%zu feature_diff=%zu max_abs=%.3g\n",
        cpu_count,gpu_count,coord_diff,feature_diff,max_abs);
    if(!ok)std::fprintf(stderr,"CUDA real voxel failure: %s\n",error);
    std::free(gpu_features);std::free(gpu_coords);cudaFree(count);cudaFree(coords);
    cudaFree(features);cudaFree(points);bf_cuda_voxelizer_destroy(voxelizer);
    std::free(workspace);std::free(counts);std::free(cpu_coords);
    std::free(cpu_features);std::free(voxels);bf_frame_close(file);return ok?0:5;
}
