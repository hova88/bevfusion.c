#ifndef BF_CUDA_VOXEL_H
#define BF_CUDA_VOXEL_H

#include "bf_kernels.h"
#include "bf_voxel.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_voxelizer bf_cuda_voxelizer;

int bf_cuda_voxelizer_create(const bf_voxel_config *config,
                             size_t max_input_points,
                             bf_cuda_voxelizer **out,
                             char *error, size_t error_cap);
void bf_cuda_voxelizer_destroy(bf_cuda_voxelizer *voxelizer);
size_t bf_cuda_voxelizer_resident_bytes(const bf_cuda_voxelizer *voxelizer);

/* Stable radix grouping preserves input order inside each voxel and emits
 * voxels in first-encounter order, including CPU-identical capacity truncation.
 * voxel_count is a single device uint32. */
int bf_cuda_voxelize_mean_f32(bf_cuda_voxelizer *voxelizer,
                              const float *points_p5_device,
                              size_t point_count,
                              bf_coord4 *coords_device,
                              float *mean_features_v5_device,
                              unsigned *voxel_count_device,
                              void *stream,
                              char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
