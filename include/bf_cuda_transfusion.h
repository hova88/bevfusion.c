#ifndef BF_CUDA_TRANSFUSION_H
#define BF_CUDA_TRANSFUSION_H

#include "bf_model.h"
#include "bf_transfusion_decoder.h"
#include "bevfusion.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_transfusion bf_cuda_transfusion;

int bf_cuda_transfusion_create(const bf_model *model,
                               size_t height, size_t width, size_t proposals,
                               bf_cuda_transfusion **out,
                               char *error, size_t error_cap);
void bf_cuda_transfusion_destroy(bf_cuda_transfusion *decoder);

/* Input and every pointer inside outputs are device pointers. Batch is one. */
int bf_cuda_transfusion_forward(bf_cuda_transfusion *decoder,
                                const float *shared_b128hw_device,
                                const float *dense_heatmap_b10hw_device,
                                bf_transfusion_raw_outputs *outputs_device,
                                void *stream,
                                char *error, size_t error_cap);

/* Decodes/range-filters on GPU, then synchronously copies one canonical result. */
int bf_cuda_transfusion_decode_detections(
                                bf_cuda_transfusion *decoder,
                                const bf_transfusion_raw_outputs *outputs_device,
                                float score_threshold,
                                bf_detections *detections_host,
                                void *stream,
                                char *error, size_t error_cap);

size_t bf_cuda_transfusion_resident_bytes(const bf_cuda_transfusion *decoder);

#ifdef __cplusplus
}
#endif

#endif
