#ifndef BF_CUDA_SWIN_H
#define BF_CUDA_SWIN_H

#include "bf_model.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_swin bf_cuda_swin;

/* Fixed-shape strict Swin-T context. Input dimensions must be multiples of 4,
 * as in the production 256x704 checkpoint boundary. */
int bf_cuda_swin_create(const bf_model *model, size_t batches,
                        size_t input_height, size_t input_width,
                        bf_cuda_swin **out,
                        char *error, size_t error_cap);
void bf_cuda_swin_destroy(bf_cuda_swin *swin);
size_t bf_cuda_swin_resident_bytes(const bf_cuda_swin *swin);

/* Input [B,3,H,W]; outputs [B,192,H/8,W/8], [B,384,H/16,W/16],
 * [B,768,H/32,ceil(W/32)] for the production shape. */
int bf_cuda_swin_forward(bf_cuda_swin *swin,
                         const float *images_b3hw_device,
                         float *stage1_b192hw_device,
                         float *stage2_b384hw_device,
                         float *stage3_b768hw_device,
                         void *stream,
                         char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
