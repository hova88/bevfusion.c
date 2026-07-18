#ifndef BF_CUDA_H
#define BF_CUDA_H

#include "bf_lss.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int bf_cuda_available(char *error, size_t error_cap);

/* All tensor pointers are device pointers. stream is cudaStream_t or NULL. */
int bf_cuda_lss_lift_pool_f32(
    const float *depth_logits_device,
    const float *context_device,
    const float *geometry_device,
    float *bev_device,
    const bf_lss_desc *desc,
    void *stream,
    char *error, size_t error_cap);

/* Convert [B,N,D,H,W,3] metric geometry to compact flattened BEV-cell ranks.
 * Invalid/out-of-grid samples are encoded as -1. All pointers remain device
 * pointers and ranks contains B*N*D*H*W int32_t values. */
int bf_cuda_lss_geometry_to_ranks_f32(
    const float *geometry_device,
    int *ranks_device,
    const bf_lss_desc *desc,
    void *stream,
    char *error, size_t error_cap);

/* Lift and pool from precomputed cell ranks, avoiding repeated metric-to-cell
 * conversion in the channel loop. */
int bf_cuda_lss_lift_pool_ranks_f32(
    const float *depth_logits_device,
    const float *context_device,
    const int *ranks_device,
    float *bev_device,
    const bf_lss_desc *desc,
    void *stream,
    char *error, size_t error_cap);

typedef struct bf_cuda_lss_plan bf_cuda_lss_plan;

/* Deterministic, atomics-free BEVPool plan. The fixed-shape plan owns the
 * radix-sort buffers, compact cell intervals, and depth-probability scratch. */
int bf_cuda_lss_plan_create(
    const bf_lss_desc *desc,
    bf_cuda_lss_plan **out,
    char *error, size_t error_cap);
void bf_cuda_lss_plan_destroy(bf_cuda_lss_plan *plan);
size_t bf_cuda_lss_plan_resident_bytes(const bf_cuda_lss_plan *plan);

/* Geometry preparation performs metric-to-cell conversion and a stable sort.
 * It is separate because calibration geometry can be reused when unchanged. */
int bf_cuda_lss_plan_prepare_geometry_f32(
    bf_cuda_lss_plan *plan,
    const float *geometry_device,
    void *stream,
    char *error, size_t error_cap);

/* Direct calibration route: computes cell keys without materializing the
 * [B,N,D,H,W,3] geometry tensor. Optional extra transforms may be NULL. */
int bf_cuda_lss_plan_prepare_calibration_f32(
    bf_cuda_lss_plan *plan,
    const float *frustum_dhw3_device,
    const float *camera_rotation_bn33_device,
    const float *camera_translation_bn3_device,
    const float *intrinsics_bn33_device,
    const float *post_rotation_bn33_device,
    const float *post_translation_bn3_device,
    const float *extra_rotation_b33_device,
    const float *extra_translation_b3_device,
    void *stream,
    char *error, size_t error_cap);

int bf_cuda_lss_plan_forward_f32(
    bf_cuda_lss_plan *plan,
    const float *depth_logits_device,
    const float *context_device,
    float *bev_device,
    void *stream,
    char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
