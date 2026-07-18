#ifndef BF_IMAGE_FPN_H
#define BF_IMAGE_FPN_H

#include "bf_model.h"

#include <stddef.h>

typedef struct bf_image_fpn bf_image_fpn;

int bf_image_fpn_create(const bf_model *model, bf_image_fpn **out,
                        char *error, size_t error_cap);
void bf_image_fpn_destroy(bf_image_fpn *fpn);

size_t bf_image_fpn_workspace_bytes(size_t level0_height, size_t level0_width);

/*
 * inputs: [B,192,H,W], [B,384,ceil(H/2),ceil(W/2)],
 *         [B,768,ceil(H/4),ceil(W/4)]
 * outputs: [B,256,H,W], [B,256,ceil(H/2),ceil(W/2)]
 */
int bf_image_fpn_forward_ref(const bf_image_fpn *fpn,
                             const float *input_level0,
                             const float *input_level1,
                             const float *input_level2,
                             size_t batches, size_t level0_height,
                             size_t level0_width,
                             float *output_level0, float *output_level1,
                             void *workspace, size_t workspace_bytes,
                             char *error, size_t error_cap);

#endif
