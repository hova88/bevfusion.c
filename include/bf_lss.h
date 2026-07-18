#ifndef BF_LSS_H
#define BF_LSS_H

#include <stddef.h>

typedef struct {
    size_t batches;
    size_t cameras;
    size_t depth_bins;
    size_t feature_height;
    size_t feature_width;
    size_t channels;
    float grid_minimum[3];
    float grid_step[3];
    size_t grid_cells[3];
} bf_lss_desc;

int bf_lss_geometry_f32_ref(const float *frustum_dhw3,
                            const float *camera_to_lidar_rotation,
                            const float *camera_to_lidar_translation,
                            const float *intrinsics,
                            const float *post_rotation,
                            const float *post_translation,
                            const float *extra_rotation,
                            const float *extra_translation,
                            float *geometry_bndhw3,
                            size_t batches, size_t cameras,
                            size_t depth_bins, size_t height, size_t width);

int bf_lss_lift_f32_ref(const float *depth_logits_bndhw,
                        const float *context_bnchw,
                        float *lifted_bndhwc,
                        const bf_lss_desc *desc);

/* Output layout matches OpenPCDet before its final x/y permute: [B,C*Z,X,Y]. */
int bf_lss_bev_pool_f32_ref(const float *lifted_bndhwc,
                            const float *geometry_bndhw3,
                            float *bev_bczxy,
                            const bf_lss_desc *desc);

/* Fused strict route: avoids materializing the full B*N*D*H*W*C lift tensor. */
int bf_lss_lift_pool_f32_ref(const float *depth_logits_bndhw,
                             const float *context_bnchw,
                             const float *geometry_bndhw3,
                             float *bev_bczxy,
                             const bf_lss_desc *desc);

#endif
