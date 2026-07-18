#include "bf_transfusion.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float sigmoid(float value) {
    if (value >= 0.0f) {
        float exponent = expf(-value);
        return 1.0f / (1.0f + exponent);
    }
    float exponent = expf(value);
    return exponent / (1.0f + exponent);
}

int bf_transfusion_select_proposals_f32_ref(
    const float *logits, float *suppressed, float *top_scores,
    int64_t *top_classes, int64_t *top_indices,
    size_t batches, size_t classes, size_t height, size_t width,
    size_t proposals, size_t kernel) {
    if (!logits || !suppressed || !top_scores || !top_classes || !top_indices ||
        !batches || !classes || !height || !width || !proposals ||
        proposals > classes * height * width || !kernel || !(kernel & 1)) return 0;
    size_t spatial = height * width;
    size_t padding = kernel / 2;
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t class_id = 0; class_id < classes; ++class_id) {
            int bypass_nms = classes == 10 && (class_id == 8 || class_id == 9);
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x) {
                    size_t offset = ((batch * classes + class_id) * height + y) * width + x;
                    float value = sigmoid(logits[offset]);
                    int keep = bypass_nms;
                    if (!bypass_nms && y >= padding && x >= padding &&
                        y + padding < height && x + padding < width) {
                        float maximum = -FLT_MAX;
                        for (size_t ky = 0; ky < kernel; ++ky)
                            for (size_t kx = 0; kx < kernel; ++kx) {
                                size_t iy = y + ky - padding;
                                size_t ix = x + kx - padding;
                                float neighbor = sigmoid(logits[
                                    ((batch * classes + class_id) * height + iy) * width + ix]);
                                if (neighbor > maximum) maximum = neighbor;
                            }
                        keep = value == maximum;
                    }
                    suppressed[offset] = keep ? value : 0.0f;
                }
        }
        for (size_t rank = 0; rank < proposals; ++rank) {
            size_t best = SIZE_MAX;
            float best_score = -FLT_MAX;
            for (size_t flat = 0; flat < classes * spatial; ++flat) {
                int used = 0;
                for (size_t prior = 0; prior < rank; ++prior) {
                    size_t prior_flat = (size_t)top_classes[batch * proposals + prior] * spatial +
                                        (size_t)top_indices[batch * proposals + prior];
                    used |= prior_flat == flat;
                }
                float value = suppressed[batch * classes * spatial + flat];
                if (!used && (best == SIZE_MAX || value > best_score)) {
                    best = flat;
                    best_score = value;
                }
            }
            top_scores[batch * proposals + rank] = best_score;
            top_classes[batch * proposals + rank] = (int64_t)(best / spatial);
            top_indices[batch * proposals + rank] = (int64_t)(best % spatial);
        }
    }
    return 1;
}

int bf_transfusion_decode_raw_f32_ref(
    const float *heatmap, const float *query_scores, const int64_t *query_labels,
    const float *center, const float *height, const float *dimension_log,
    const float *rotation, const float *velocity,
    float *boxes, float *scores, int64_t *labels,
    size_t batches, size_t classes, size_t proposals,
    float feature_stride, const float voxel_size[2], const float minimum[2]) {
    if (!heatmap || !query_scores || !query_labels || !center || !height ||
        !dimension_log || !rotation || !boxes || !scores || !labels ||
        !batches || !classes || !proposals || !isfinite(feature_stride) ||
        feature_stride <= 0.0f || !voxel_size || !minimum ||
        voxel_size[0] <= 0.0f || voxel_size[1] <= 0.0f) return 0;
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t proposal = 0; proposal < proposals; ++proposal) {
            int64_t class_id = query_labels[batch * proposals + proposal];
            if (class_id < 0 || (uint64_t)class_id >= classes) return 0;
            size_t class_offset = (batch * classes + (size_t)class_id) * proposals + proposal;
            scores[batch * proposals + proposal] = sigmoid(heatmap[class_offset]) *
                                                   query_scores[class_offset];
            labels[batch * proposals + proposal] = class_id;
            float *box = boxes + (batch * proposals + proposal) * 9;
            box[0] = center[(batch * 2) * proposals + proposal] *
                     feature_stride * voxel_size[0] + minimum[0];
            box[1] = center[(batch * 2 + 1) * proposals + proposal] *
                     feature_stride * voxel_size[1] + minimum[1];
            box[2] = height[batch * proposals + proposal];
            for (size_t axis = 0; axis < 3; ++axis)
                box[3 + axis] = expf(dimension_log[(batch * 3 + axis) * proposals + proposal]);
            float sine = rotation[(batch * 2) * proposals + proposal];
            float cosine = rotation[(batch * 2 + 1) * proposals + proposal];
            box[6] = atan2f(sine, cosine);
            box[7] = velocity ? velocity[(batch * 2) * proposals + proposal] : 0.0f;
            box[8] = velocity ? velocity[(batch * 2 + 1) * proposals + proposal] : 0.0f;
        }
    return 1;
}

int bf_transfusion_filter_detections(
    const float *boxes, const float *scores, const int64_t *labels,
    size_t batches, size_t proposals, float threshold,
    const float range[6], bf_detections *output) {
    if (!boxes || !scores || !labels || !batches || !proposals ||
        proposals > BF_MAX_PROPOSALS || !isfinite(threshold) || !range || !output)
        return 0;
    for (size_t batch = 0; batch < batches; ++batch) {
        output[batch].count = 0;
        for (size_t proposal = 0; proposal < proposals; ++proposal) {
            const float *box = boxes + (batch * proposals + proposal) * 9;
            float score = scores[batch * proposals + proposal];
            int keep = isfinite(score) && score > threshold;
            for (size_t axis = 0; axis < 3; ++axis)
                keep &= isfinite(box[axis]) && box[axis] >= range[axis] && box[axis] <= range[axis + 3];
            if (!keep) continue;
            bf_detection *detection = &output[batch].items[output[batch].count++];
            detection->x = box[0]; detection->y = box[1]; detection->z = box[2];
            detection->width = box[3]; detection->length = box[4]; detection->height = box[5];
            detection->yaw = box[6]; detection->velocity_x = box[7]; detection->velocity_y = box[8];
            detection->score = score; detection->class_id = (int)labels[batch * proposals + proposal];
        }
    }
    return 1;
}
