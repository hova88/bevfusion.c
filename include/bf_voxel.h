#ifndef BF_VOXEL_H
#define BF_VOXEL_H

#include "bf_kernels.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float minimum[3];
    float maximum[3];
    float voxel_size[3];
    size_t point_features;
    size_t max_points_per_voxel;
    size_t max_voxels;
} bf_voxel_config;

typedef struct {
    size_t input_points;
    size_t accepted_points;
    size_t rejected_nonfinite;
    size_t rejected_out_of_range;
    size_t dropped_voxel_capacity;
    size_t dropped_point_capacity;
} bf_voxel_stats;

int bf_voxel_grid_shape(const bf_voxel_config *config,
                        size_t grid_xyz[3]);
size_t bf_voxelize_workspace_bytes(const bf_voxel_config *config);
int bf_voxelize_f32_workspace_ref(const float *points, size_t point_count,
                        size_t point_stride, int32_t batch_index,
                        const bf_voxel_config *config, float *voxels,
                        bf_coord4 *coords, int64_t *counts,
                        size_t *voxel_count, bf_voxel_stats *stats,
                        void *workspace, size_t workspace_bytes);
int bf_voxelize_f32_ref(const float *points, size_t point_count,
                        size_t point_stride, int32_t batch_index,
                        const bf_voxel_config *config, float *voxels,
                        bf_coord4 *coords, int64_t *counts,
                        size_t *voxel_count, bf_voxel_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
