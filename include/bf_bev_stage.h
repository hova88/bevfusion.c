#ifndef BF_BEV_STAGE_H
#define BF_BEV_STAGE_H

#include "bf_model.h"

#include <stddef.h>

typedef struct bf_bev_stage bf_bev_stage;

/* The model mapping must outlive the bound stage. */
int bf_bev_stage_create(const bf_model *model, bf_bev_stage **out,
                        char *error, size_t error_cap);
void bf_bev_stage_destroy(bf_bev_stage *stage);

size_t bf_bev_stage_workspace_bytes(size_t height, size_t width);

/*
 * input:   [B,336,H,W] (image 80 channels followed by lidar 256)
 * spatial: [B,512,H,W]
 * shared:  [B,128,H,W]
 * heatmap: [B,10,H,W] raw logits
 */
int bf_bev_stage_forward_ref(const bf_bev_stage *stage,
                             const float *input_b336hw,
                             size_t batches, size_t height, size_t width,
                             float *spatial_b512hw,
                             float *shared_b128hw,
                             float *heatmap_b10hw,
                             void *workspace, size_t workspace_bytes,
                             char *error, size_t error_cap);

#endif
