#ifndef BF_DEPTH_HEAD_H
#define BF_DEPTH_HEAD_H

#include "bf_model.h"

#include <stddef.h>

#define BF_DEPTH_BINS 118u
#define BF_CONTEXT_CHANNELS 80u

typedef struct bf_depth_head bf_depth_head;

int bf_depth_head_create(const bf_model *model, bf_depth_head **out,
                         char *error, size_t error_cap);
void bf_depth_head_destroy(bf_depth_head *head);

size_t bf_depth_head_workspace_bytes(size_t feature_height,
                                     size_t feature_width);

/*
 * image_features: [BN,256,H,W]
 * dense_depth:    [BN,1,8H,8W]
 * depth_logits:  [BN,118,H,W]
 * context:       [BN,80,H,W]
 */
int bf_depth_head_forward_ref(const bf_depth_head *head,
                              const float *image_features,
                              const float *dense_depth,
                              size_t camera_batches,
                              size_t feature_height, size_t feature_width,
                              float *depth_logits, float *context,
                              void *workspace, size_t workspace_bytes,
                              char *error, size_t error_cap);

#endif
