#include "bf_cuda_depth_raster.h"
#include "bf_depth_raster.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__ static void make_points(float *points,size_t count){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count){points[i*5]=(float)((i*17)%1000)*.01f-5;points[i*5+1]=(float)((i*29)%1000)*.01f-5;points[i*5+2]=(float)((i%80)+1)*.25f;points[i*5+3]=.5f;points[i*5+4]=0;}}

int main(){
    const size_t cameras=2,height=8,width=12;
    float points[][5]={{1,1,2,.1f,0},{2,2,4,.2f,0},{-1,1,2,.3f,0},{3,-1,3,.4f,0},{NAN,0,1,0,0}};
    float batch_points[30];for(size_t i=0;i<5;++i){batch_points[i*6]=0;for(int c=0;c<5;++c)batch_points[i*6+c+1]=points[i][c];}
    float lidar_aug[16]={1,0,0,.25f,0,1,0,-.5f,0,0,1,0,0,0,0,1};
    float projection[32]={},image_aug[32]={};
    for(size_t c=0;c<cameras;++c){projection[c*16]=6+c;projection[c*16+5]=5;projection[c*16+10]=1;projection[c*16+3]=4+c;projection[c*16+7]=3;projection[c*16+15]=1;image_aug[c*16]=c?.9f:1;image_aug[c*16+5]=c?1.1f:1;image_aug[c*16+10]=1;image_aug[c*16+15]=1;image_aug[c*16+3]=c?.5f:0;}
    float expected[cameras*height*width];int ok=bf_depth_rasterize_f32_ref(batch_points,5,6,lidar_aug,projection,image_aug,expected,1,cameras,height,width);char error[256]={};
    bf_cuda_depth_raster *raster=nullptr;float *dp=nullptr,*dla=nullptr,*dproj=nullptr,*dia=nullptr,*depth=nullptr;
    if(ok)ok=bf_cuda_depth_raster_create(100000,cameras,height,width,&raster,error,sizeof(error))&&cudaMalloc(&dp,100000*5*4)==cudaSuccess&&cudaMalloc(&dla,16*4)==cudaSuccess&&cudaMalloc(&dproj,32*4)==cudaSuccess&&cudaMalloc(&dia,32*4)==cudaSuccess&&cudaMalloc(&depth,sizeof(expected))==cudaSuccess&&cudaMemcpy(dp,points,sizeof(points),cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(dla,lidar_aug,sizeof(lidar_aug),cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(dproj,projection,sizeof(projection),cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(dia,image_aug,sizeof(image_aug),cudaMemcpyHostToDevice)==cudaSuccess&&bf_cuda_depth_rasterize_f32(raster,dp,5,dla,dproj,dia,depth,nullptr,error,sizeof(error))&&cudaDeviceSynchronize()==cudaSuccess;
    float actual[cameras*height*width],maximum=0;size_t differences=0;if(ok)ok=cudaMemcpy(actual,depth,sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess;if(ok)for(size_t i=0;i<cameras*height*width;++i){float d=fabsf(actual[i]-expected[i]);maximum=fmaxf(maximum,d);if(d>2e-6f)++differences;}

    bf_cuda_depth_raster *production=nullptr;float *production_depth=nullptr,*production_projection=nullptr,*production_aug=nullptr;float hp[96]={},ha[96]={};
    for(int c=0;c<6;++c){hp[c*16]=20;hp[c*16+3]=352;hp[c*16+5]=20;hp[c*16+7]=128;hp[c*16+10]=1;hp[c*16+15]=1;ha[c*16]=ha[c*16+5]=ha[c*16+10]=ha[c*16+15]=1;}
    if(ok)ok=bf_cuda_depth_raster_create(100000,6,256,704,&production,error,sizeof(error))&&cudaMalloc(&production_depth,6ull*256*704*4)==cudaSuccess&&cudaMalloc(&production_projection,sizeof(hp))==cudaSuccess&&cudaMalloc(&production_aug,sizeof(ha))==cudaSuccess&&cudaMemcpy(production_projection,hp,sizeof(hp),cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(production_aug,ha,sizeof(ha),cudaMemcpyHostToDevice)==cudaSuccess;
    cudaEvent_t begin=nullptr,end=nullptr;cudaEventCreate(&begin);cudaEventCreate(&end);float production_ms=0;
    if(ok){make_points<<<391,256>>>(dp,100000);cudaEventRecord(begin);for(int i=0;i<20;++i)ok=ok&&bf_cuda_depth_rasterize_f32(production,dp,100000,dla,production_projection,production_aug,production_depth,nullptr,error,sizeof(error));cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&production_ms,begin,end);}
    std::printf("cuda_depth_raster max_abs=%.3g differences=%zu production_warm=%.3f ms resident=%.2f MiB\n",maximum,differences,production_ms/20,bf_cuda_depth_raster_resident_bytes(production)/(1024.0*1024.0));
    if(!ok||differences)std::fprintf(stderr,"CUDA depth raster failure: %s\n",error);
    cudaEventDestroy(end);cudaEventDestroy(begin);cudaFree(production_aug);cudaFree(production_projection);cudaFree(production_depth);bf_cuda_depth_raster_destroy(production);cudaFree(depth);cudaFree(dia);cudaFree(dproj);cudaFree(dla);cudaFree(dp);bf_cuda_depth_raster_destroy(raster);return ok&&!differences?0:5;
}
