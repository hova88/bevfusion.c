#ifndef BF_LIDAR_BACKBONE_H
#define BF_LIDAR_BACKBONE_H

#include "bf_kernels.h"
#include "bf_model.h"

#include <stddef.h>

typedef struct bf_lidar_backbone bf_lidar_backbone;

int bf_lidar_backbone_create(const bf_model *model, bf_lidar_backbone **out,
                             char *error, size_t error_cap);
void bf_lidar_backbone_destroy(bf_lidar_backbone *backbone);

size_t bf_lidar_backbone_workspace_bytes(size_t sparse_capacity);
int bf_lidar_backbone_output_shape(size_t input_depth, size_t input_height,
                                   size_t input_width, size_t *output_depth,
                                   size_t *output_height, size_t *output_width);

/* Input MeanVFE features [V,5], coordinates [V,4] in [batch,z,y,x]. */
int bf_lidar_backbone_forward_ref(const bf_lidar_backbone *backbone,
                                  const bf_coord4 *voxel_coords,
                                  const float *voxel_features,
                                  size_t voxel_count, size_t batches,
                                  size_t input_depth, size_t input_height,
                                  size_t input_width, size_t sparse_capacity,
                                  float *dense_bev_b128d_hw,
                                  void *workspace, size_t workspace_bytes,
                                  char *error, size_t error_cap);

#endif
