#ifndef BF_CUDA_RUNTIME_H
#define BF_CUDA_RUNTIME_H

#include "bf_runtime.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_runtime bf_cuda_runtime;

int bf_cuda_runtime_create(const char *model_path,
                           size_t point_capacity,
                           size_t voxel_capacity,
                           size_t sparse_capacity,
                           bf_cuda_runtime **out,
                           char *error, size_t error_cap);
void bf_cuda_runtime_destroy(bf_cuda_runtime *runtime);

/* Host BFI boundary -> canonical host detections. Inputs are copied H2D once;
 * every graph intermediate and sparse count remains device-resident. */
int bf_cuda_runtime_infer(bf_cuda_runtime *runtime,
                          const bf_frame_input *frame,
                          bf_detections *detections,
                          char *error, size_t error_cap);

/* Includes contexts, parameters, reusable graph intermediates, and owned input
 * boundary buffers. It excludes the caller's BFI mapping and detections. */
size_t bf_cuda_runtime_resident_bytes(const bf_cuda_runtime *runtime);

/* Out-of-band diagnostic: sum(abs(x)) and max(abs(x)) for image and LiDAR BEV. */
int bf_cuda_runtime_debug_bev_stats(const bf_cuda_runtime *runtime,
                                    float stats4_host[4],
                                    char *error, size_t error_cap);

/* Out-of-band bitwise hashes for eleven major device graph boundaries. */
int bf_cuda_runtime_debug_hashes(const bf_cuda_runtime *runtime,
                                 unsigned long long hashes11_host[11],
                                 char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
