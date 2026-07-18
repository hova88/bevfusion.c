#ifndef BF_CUDA_LIDAR_H
#define BF_CUDA_LIDAR_H

#include "bf_kernels.h"
#include "bf_model.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_lidar_backbone bf_cuda_lidar_backbone;

int bf_cuda_lidar_backbone_create(const bf_model *model,
                                  size_t batches,
                                  size_t input_depth,
                                  size_t input_height,
                                  size_t input_width,
                                  size_t sparse_capacity,
                                  bf_cuda_lidar_backbone **out,
                                  char *error, size_t error_cap);
void bf_cuda_lidar_backbone_destroy(bf_cuda_lidar_backbone *backbone);
size_t bf_cuda_lidar_backbone_resident_bytes(const bf_cuda_lidar_backbone *backbone);

/* Coordinates and MeanVFE features are device-resident. Output is checkpoint
 * layout [B,128,D,H,W], logically [B,256,H,W] for production D=2. */
int bf_cuda_lidar_backbone_forward(bf_cuda_lidar_backbone *backbone,
                                   const bf_coord4 *voxel_coords_device,
                                   const float *voxel_features_v5_device,
                                   size_t voxel_count,
                                   float *dense_bev_device,
                                   void *stream,
                                   char *error, size_t error_cap);

int bf_cuda_lidar_backbone_forward_count_device(
                                   bf_cuda_lidar_backbone *backbone,
                                   const bf_coord4 *voxel_coords_device,
                                   const float *voxel_features_v5_device,
                                   const unsigned *voxel_count_device,
                                   float *dense_bev_device,
                                   void *stream,
                                   char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
