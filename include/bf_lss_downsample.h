#ifndef BF_LSS_DOWNSAMPLE_H
#define BF_LSS_DOWNSAMPLE_H

#include "bf_model.h"

#include <stddef.h>

typedef struct bf_lss_downsample bf_lss_downsample;

int bf_lss_downsample_create(const bf_model *model, bf_lss_downsample **out,
                             char *error, size_t error_cap);
void bf_lss_downsample_destroy(bf_lss_downsample *downsample);
size_t bf_lss_downsample_workspace_bytes(size_t input_height, size_t input_width);

/* Input [B,80,X,Y], output [B,80,Y/2,X/2], including final x/y permute. */
int bf_lss_downsample_forward_ref(const bf_lss_downsample *downsample,
                                  const float *input_b80xy,
                                  size_t batches, size_t input_height,
                                  size_t input_width,
                                  float *output_b80yx,
                                  void *workspace, size_t workspace_bytes,
                                  char *error, size_t error_cap);

#endif
