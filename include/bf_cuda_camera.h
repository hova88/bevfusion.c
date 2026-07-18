#ifndef BF_CUDA_CAMERA_H
#define BF_CUDA_CAMERA_H

#include "bf_model.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_cuda_camera_neck bf_cuda_camera_neck;

/* Fixed-shape context for GeneralizedLSSFPN, DepthLSS head, and the post-LSS
 * downsample block. All evaluation BatchNorm parameters are folded at create. */
int bf_cuda_camera_neck_create(const bf_model *model,
                               size_t camera_batches,
                               size_t feature_height,
                               size_t feature_width,
                               size_t bev_height,
                               size_t bev_width,
                               bf_cuda_camera_neck **out,
                               char *error, size_t error_cap);
void bf_cuda_camera_neck_destroy(bf_cuda_camera_neck *neck);
size_t bf_cuda_camera_neck_resident_bytes(const bf_cuda_camera_neck *neck);

/* Swin inputs are [BN,192,H,W], [BN,384,ceil(H/2),ceil(W/2)], and
 * [BN,768,ceil(H/4),ceil(W/4)]. Outputs are the two 256-channel FPN levels. */
int bf_cuda_camera_fpn_forward(bf_cuda_camera_neck *neck,
                               const float *swin0_device,
                               const float *swin1_device,
                               const float *swin2_device,
                               float *fpn0_device,
                               float *fpn1_device,
                               void *stream,
                               char *error, size_t error_cap);

/* Standalone oracle boundary. dense_depth is [BN,1,8H,8W]. */
int bf_cuda_camera_depth_forward(bf_cuda_camera_neck *neck,
                                 const float *fpn0_device,
                                 const float *dense_depth_device,
                                 float *depth_logits_device,
                                 float *context_device,
                                 void *stream,
                                 char *error, size_t error_cap);

/* Production fused scheduling: FPN intermediates remain context-owned. */
int bf_cuda_camera_neck_forward(bf_cuda_camera_neck *neck,
                                const float *swin0_device,
                                const float *swin1_device,
                                const float *swin2_device,
                                const float *dense_depth_device,
                                float *depth_logits_device,
                                float *context_device,
                                void *stream,
                                char *error, size_t error_cap);

/* [B,80,X,Y] -> [B,80,Y/2,X/2], including the checkpoint x/y transpose. */
int bf_cuda_camera_downsample_forward(bf_cuda_camera_neck *neck,
                                      const float *full_bev_device,
                                      float *image_bev_device,
                                      void *stream,
                                      char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
