#ifndef BF_SWIN_BACKBONE_H
#define BF_SWIN_BACKBONE_H

#include "bf_model.h"

#include <stddef.h>

typedef struct bf_swin_backbone bf_swin_backbone;

typedef struct {
    size_t height[3];
    size_t width[3];
    size_t channels[3];
} bf_swin_backbone_shapes;

/* The model mapping must outlive the bound backbone. */
int bf_swin_backbone_create(const bf_model *model, bf_swin_backbone **out,
                            char *error, size_t error_cap);
void bf_swin_backbone_destroy(bf_swin_backbone *backbone);

int bf_swin_backbone_output_shapes(size_t input_height, size_t input_width,
                                   bf_swin_backbone_shapes *shapes);
size_t bf_swin_backbone_workspace_bytes(size_t batches,
                                        size_t input_height, size_t input_width);

/*
 * input:  [B,3,H,W]
 * output0 [B,192,ceil(H/8),ceil(W/8)]
 * output1 [B,384,ceil(H/16),ceil(W/16)]
 * output2 [B,768,ceil(H/32),ceil(W/32)]
 *
 * This is the exact scalar inference graph for the checkpoint Swin-T image
 * backbone. Dropout and stochastic depth are identities in evaluation mode.
 */
int bf_swin_backbone_forward_ref(const bf_swin_backbone *backbone,
                                 const float *input_b3hw,
                                 size_t batches, size_t input_height,
                                 size_t input_width,
                                 float *output_stage1,
                                 float *output_stage2,
                                 float *output_stage3,
                                 void *workspace, size_t workspace_bytes,
                                 char *error, size_t error_cap);

#endif
