#ifndef BF_CUDA_DEPTH_RASTER_H
#define BF_CUDA_DEPTH_RASTER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_depth_raster bf_cuda_depth_raster;

int bf_cuda_depth_raster_create(size_t max_points, size_t cameras,
                                size_t image_height, size_t image_width,
                                bf_cuda_depth_raster **out,
                                char *error, size_t error_cap);
void bf_cuda_depth_raster_destroy(bf_cuda_depth_raster *raster);
size_t bf_cuda_depth_raster_resident_bytes(const bf_cuda_depth_raster *raster);

/* Batch-one frame contract. points are [P,5] x/y/z/intensity/time. Matrices
 * and output are device-resident; output is [N,H,W]. Later input points win
 * pixel collisions, exactly matching the ordered scalar assignment. */
int bf_cuda_depth_rasterize_f32(bf_cuda_depth_raster *raster,
                                const float *points_p5_device,
                                size_t point_count,
                                const float *lidar_augmentation_16_device,
                                const float *lidar_to_image_n16_device,
                                const float *image_augmentation_n16_device,
                                float *dense_depth_nhw_device,
                                void *stream,
                                char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
