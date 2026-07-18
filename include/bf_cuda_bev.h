#ifndef BF_CUDA_BEV_H
#define BF_CUDA_BEV_H

#include "bf_model.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_bev_stage bf_cuda_bev_stage;

/* Fixed-shape, batch-one context. Weights and reusable scratch remain on GPU. */
int bf_cuda_bev_stage_create(const bf_model *model, size_t height, size_t width,
                             bf_cuda_bev_stage **out,
                             char *error, size_t error_cap);
void bf_cuda_bev_stage_destroy(bf_cuda_bev_stage *stage);

/* All tensor pointers are device pointers. stream is cudaStream_t or NULL. */
int bf_cuda_bev_stage_forward(bf_cuda_bev_stage *stage,
                              const float *input_b336hw_device,
                              float *spatial_b512hw_device,
                              float *shared_b128hw_device,
                              float *heatmap_b10hw_device,
                              void *stream,
                              char *error, size_t error_cap);

size_t bf_cuda_bev_stage_resident_bytes(const bf_cuda_bev_stage *stage);

#ifdef __cplusplus
}
#endif

#endif
