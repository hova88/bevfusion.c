#include "bf_cuda_depth_raster.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

struct bf_cuda_depth_raster {
    size_t max_points, cameras, height, width, resident_bytes;
    unsigned *owners;
    float *distances, *inverse;
    int *valid_inverse;
};

static int fail(char *error,size_t cap,const char *format,...){if(error&&cap){va_list args;va_start(args,format);std::vsnprintf(error,cap,format,args);va_end(args);}return 0;}
static int cuda_ok(cudaError_t status,char *error,size_t cap,const char *where){return status==cudaSuccess?1:fail(error,cap,"%s: %s",where,cudaGetErrorString(status));}

__global__ static void inverse_lidar_aug_kernel(const float *m,float *out,int *valid){if(blockIdx.x||threadIdx.x)return;float a00=m[0],a01=m[1],a02=m[2],a10=m[4],a11=m[5],a12=m[6],a20=m[8],a21=m[9],a22=m[10];float c00=a11*a22-a12*a21,c01=a02*a21-a01*a22,c02=a01*a12-a02*a11,c10=a12*a20-a10*a22,c11=a00*a22-a02*a20,c12=a02*a10-a00*a12,c20=a10*a21-a11*a20,c21=a01*a20-a00*a21,c22=a00*a11-a01*a10;float determinant=a00*c00+a01*c10+a02*c20;if(!isfinite(determinant)||fabsf(determinant)<=1.17549435e-38f){*valid=0;return;}float scale=1.0f/determinant;out[0]=c00*scale;out[1]=c01*scale;out[2]=c02*scale;out[3]=c10*scale;out[4]=c11*scale;out[5]=c12*scale;out[6]=c20*scale;out[7]=c21*scale;out[8]=c22*scale;*valid=1;}

__global__ static void project_owner_kernel(const float *points,unsigned point_count,
    size_t cameras,size_t height,size_t width,const float *lidar_aug,
    const float *inverse,const int *valid_inverse,const float *lidar_to_image,
    const float *image_aug,unsigned *owners,float *distances){size_t total=(size_t)point_count*cameras;for(size_t item=(size_t)blockIdx.x*blockDim.x+threadIdx.x;item<total;item+=(size_t)gridDim.x*blockDim.x){if(!*valid_inverse)continue;unsigned p=(unsigned)(item/cameras);size_t camera=item%cameras;const float *point=points+(size_t)p*5;if(!isfinite(point[0])||!isfinite(point[1])||!isfinite(point[2]))continue;float centered[3]={point[0]-lidar_aug[3],point[1]-lidar_aug[7],point[2]-lidar_aug[11]};float xyz[3]={inverse[0]*centered[0]+inverse[1]*centered[1]+inverse[2]*centered[2],inverse[3]*centered[0]+inverse[4]*centered[1]+inverse[5]*centered[2],inverse[6]*centered[0]+inverse[7]*centered[1]+inverse[8]*centered[2]};const float *projection=lidar_to_image+camera*16;float projected[3]={projection[0]*xyz[0]+projection[1]*xyz[1]+projection[2]*xyz[2]+projection[3],projection[4]*xyz[0]+projection[5]*xyz[1]+projection[6]*xyz[2]+projection[7],projection[8]*xyz[0]+projection[9]*xyz[1]+projection[10]*xyz[2]+projection[11]};if(!isfinite(projected[0])||!isfinite(projected[1])||!isfinite(projected[2]))continue;float distance=fminf(fmaxf(projected[2],1e-5f),1e5f);float normalized[3]={projected[0]/distance,projected[1]/distance,distance};const float *post=image_aug+camera*16;float x=post[0]*normalized[0]+post[1]*normalized[1]+post[2]*normalized[2]+post[3],y=post[4]*normalized[0]+post[5]*normalized[1]+post[6]*normalized[2]+post[7];if(!isfinite(x)||!isfinite(y)||x<0||x>=(float)width||y<0||y>=(float)height)continue;size_t pixel=(camera*height+(size_t)y)*width+(size_t)x;distances[item]=distance;atomicMax(owners+pixel,p+1);}}

__global__ static void materialize_depth_kernel(const unsigned *owners,
    const float *distances,size_t cameras,size_t height,size_t width,
    float *depth){size_t count=cameras*height*width;for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(size_t)gridDim.x*blockDim.x){unsigned owner=owners[i];size_t camera=i/(height*width);depth[i]=owner?distances[(size_t)(owner-1)*cameras+camera]:0.0f;}}

extern "C" void bf_cuda_depth_raster_destroy(bf_cuda_depth_raster *r){if(!r)return;cudaFree(r->owners);cudaFree(r->distances);cudaFree(r->inverse);cudaFree(r->valid_inverse);std::free(r);}

extern "C" int bf_cuda_depth_raster_create(size_t max_points,size_t cameras,size_t height,size_t width,bf_cuda_depth_raster **out,char *error,size_t cap){if(out)*out=nullptr;if(!out||!max_points||max_points>UINT_MAX||!cameras||!height||!width||cameras>SIZE_MAX/height||cameras*height>SIZE_MAX/width||max_points>SIZE_MAX/cameras)return fail(error,cap,"invalid CUDA depth raster contract");bf_cuda_depth_raster *r=(bf_cuda_depth_raster*)std::calloc(1,sizeof(*r));if(!r)return fail(error,cap,"CUDA depth raster host allocation failed");r->max_points=max_points;r->cameras=cameras;r->height=height;r->width=width;size_t owner_bytes=cameras*height*width*sizeof(unsigned),distance_bytes=max_points*cameras*sizeof(float);if(!cuda_ok(cudaMalloc(&r->owners,owner_bytes),error,cap,"allocate CUDA depth owners")||!cuda_ok(cudaMalloc(&r->distances,distance_bytes),error,cap,"allocate CUDA depth distances")||!cuda_ok(cudaMalloc(&r->inverse,9*sizeof(float)),error,cap,"allocate CUDA lidar inverse")||!cuda_ok(cudaMalloc(&r->valid_inverse,sizeof(int)),error,cap,"allocate CUDA inverse status")){bf_cuda_depth_raster_destroy(r);return 0;}r->resident_bytes=owner_bytes+distance_bytes+9*sizeof(float)+sizeof(int);*out=r;return 1;}

extern "C" int bf_cuda_depth_rasterize_f32(bf_cuda_depth_raster *r,const float *points,size_t point_count,const float *lidar_aug,const float *lidar_to_image,const float *image_aug,float *depth,void *stream_value,char *error,size_t cap){if(!r||!points||point_count>r->max_points||!lidar_aug||!lidar_to_image||!image_aug||!depth)return fail(error,cap,"invalid CUDA depth raster buffers");cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_value);size_t pixels=r->cameras*r->height*r->width;if(!cuda_ok(cudaMemsetAsync(r->owners,0,pixels*sizeof(unsigned),stream),error,cap,"clear CUDA depth owners"))return 0;inverse_lidar_aug_kernel<<<1,1,0,stream>>>(lidar_aug,r->inverse,r->valid_inverse);if(point_count){size_t total=point_count*r->cameras;unsigned blocks=(unsigned)((total+255)/256);if(blocks>4096)blocks=4096;project_owner_kernel<<<blocks,256,0,stream>>>(points,(unsigned)point_count,r->cameras,r->height,r->width,lidar_aug,r->inverse,r->valid_inverse,lidar_to_image,image_aug,r->owners,r->distances);}unsigned pixel_blocks=(unsigned)((pixels+255)/256);if(pixel_blocks>4096)pixel_blocks=4096;materialize_depth_kernel<<<pixel_blocks,256,0,stream>>>(r->owners,r->distances,r->cameras,r->height,r->width,depth);return cuda_ok(cudaGetLastError(),error,cap,"launch CUDA depth raster graph");}

extern "C" size_t bf_cuda_depth_raster_resident_bytes(const bf_cuda_depth_raster *r){return r?r->resident_bytes:0;}
