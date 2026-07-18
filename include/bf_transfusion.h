#ifndef BF_TRANSFUSION_H
#define BF_TRANSFUSION_H

#include "bevfusion.h"

#include <stddef.h>
#include <stdint.h>

int bf_transfusion_select_proposals_f32_ref(
    const float *dense_heatmap_logits_bchw,
    float *suppressed_heatmap_bchw,
    float *proposal_scores_bk,
    int64_t *proposal_classes_bk,
    int64_t *proposal_indices_bk,
    size_t batches, size_t classes, size_t height, size_t width,
    size_t proposals, size_t nms_kernel_size);

int bf_transfusion_decode_raw_f32_ref(
    const float *heatmap_logits_bcp,
    const float *query_heatmap_scores_bcp,
    const int64_t *query_labels_bp,
    const float *center_b2p, const float *height_b1p,
    const float *dimension_log_b3p, const float *rotation_sincos_b2p,
    const float *velocity_b2p,
    float *boxes_bp9, float *scores_bp, int64_t *labels_bp,
    size_t batches, size_t classes, size_t proposals,
    float feature_stride, const float voxel_size_xy[2],
    const float point_cloud_minimum_xy[2]);

int bf_transfusion_filter_detections(
    const float *boxes_bp9, const float *scores_bp, const int64_t *labels_bp,
    size_t batches, size_t proposals, float score_threshold,
    const float post_center_range[6], bf_detections *output);

#endif
