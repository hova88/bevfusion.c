#include "bf_lss.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int matrix_inverse_3x3(const float *m, float *out) {
    float c00 = m[4] * m[8] - m[5] * m[7];
    float c01 = m[2] * m[7] - m[1] * m[8];
    float c02 = m[1] * m[5] - m[2] * m[4];
    float c10 = m[5] * m[6] - m[3] * m[8];
    float c11 = m[0] * m[8] - m[2] * m[6];
    float c12 = m[2] * m[3] - m[0] * m[5];
    float c20 = m[3] * m[7] - m[4] * m[6];
    float c21 = m[1] * m[6] - m[0] * m[7];
    float c22 = m[0] * m[4] - m[1] * m[3];
    float determinant = m[0] * c00 + m[1] * c10 + m[2] * c20;
    if (!isfinite(determinant) || fabsf(determinant) <= FLT_MIN) return 0;
    float inverse = 1.0f / determinant;
    out[0] = c00 * inverse; out[1] = c01 * inverse; out[2] = c02 * inverse;
    out[3] = c10 * inverse; out[4] = c11 * inverse; out[5] = c12 * inverse;
    out[6] = c20 * inverse; out[7] = c21 * inverse; out[8] = c22 * inverse;
    return 1;
}

static void matrix_multiply_3x3(const float *a, const float *b, float *out) {
    for (size_t row = 0; row < 3; ++row)
        for (size_t column = 0; column < 3; ++column) {
            float sum = 0.0f;
            for (size_t k = 0; k < 3; ++k) sum += a[row * 3 + k] * b[k * 3 + column];
            out[row * 3 + column] = sum;
        }
}

static void matrix_vector_3x3(const float *matrix, const float *vector, float *out) {
    for (size_t row = 0; row < 3; ++row)
        out[row] = matrix[row * 3] * vector[0] +
                   matrix[row * 3 + 1] * vector[1] +
                   matrix[row * 3 + 2] * vector[2];
}

static int multiply_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int lss_contract(const bf_lss_desc *d, size_t *samples,
                        size_t *lifted_count, size_t *bev_count) {
    if (!d || !samples || !lifted_count || !bev_count || !d->batches ||
        !d->cameras || !d->depth_bins || !d->feature_height ||
        !d->feature_width || !d->channels) return 0;
    for (size_t axis = 0; axis < 3; ++axis)
        if (!isfinite(d->grid_minimum[axis]) || !isfinite(d->grid_step[axis]) ||
            d->grid_step[axis] <= 0.0f || !d->grid_cells[axis] ||
            d->grid_cells[axis] > INT32_MAX) return 0;
    size_t value = d->batches;
    if (!multiply_size(value, d->cameras, &value) ||
        !multiply_size(value, d->depth_bins, &value) ||
        !multiply_size(value, d->feature_height, &value) ||
        !multiply_size(value, d->feature_width, samples) ||
        !multiply_size(*samples, d->channels, lifted_count)) return 0;
    value = d->batches;
    if (!multiply_size(value, d->channels, &value) ||
        !multiply_size(value, d->grid_cells[2], &value) ||
        !multiply_size(value, d->grid_cells[0], &value) ||
        !multiply_size(value, d->grid_cells[1], bev_count)) return 0;
    return 1;
}

int bf_lss_geometry_f32_ref(const float *frustum, const float *camera_rotation,
                            const float *camera_translation, const float *intrinsics,
                            const float *post_rotation, const float *post_translation,
                            const float *extra_rotation, const float *extra_translation,
                            float *geometry, size_t batches, size_t cameras,
                            size_t depth_bins, size_t height, size_t width) {
    if (!frustum || !camera_rotation || !camera_translation || !intrinsics ||
        !post_rotation || !post_translation || !geometry || !batches || !cameras ||
        !depth_bins || !height || !width) return 0;
    for (size_t batch = 0; batch < batches; ++batch) {
        const float *augmentation_rotation = extra_rotation ? extra_rotation + batch * 9 : NULL;
        const float *augmentation_translation = extra_translation ? extra_translation + batch * 3 : NULL;
        for (size_t camera = 0; camera < cameras; ++camera) {
            size_t camera_index = batch * cameras + camera;
            const float *cam_rot = camera_rotation + camera_index * 9;
            const float *cam_trans = camera_translation + camera_index * 3;
            const float *intrinsic = intrinsics + camera_index * 9;
            const float *post_rot = post_rotation + camera_index * 9;
            const float *post_trans = post_translation + camera_index * 3;
            float inverse_post[9], inverse_intrinsic[9], combined[9];
            if (!matrix_inverse_3x3(post_rot, inverse_post) ||
                !matrix_inverse_3x3(intrinsic, inverse_intrinsic)) return 0;
            matrix_multiply_3x3(cam_rot, inverse_intrinsic, combined);
            for (size_t depth = 0; depth < depth_bins; ++depth)
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x) {
                        size_t frustum_index = ((depth * height + y) * width + x) * 3;
                        float point[3] = {
                            frustum[frustum_index] - post_trans[0],
                            frustum[frustum_index + 1] - post_trans[1],
                            frustum[frustum_index + 2] - post_trans[2]
                        };
                        float undone[3], projected[3], lidar[3];
                        matrix_vector_3x3(inverse_post, point, undone);
                        projected[0] = undone[0] * undone[2];
                        projected[1] = undone[1] * undone[2];
                        projected[2] = undone[2];
                        matrix_vector_3x3(combined, projected, lidar);
                        lidar[0] += cam_trans[0];
                        lidar[1] += cam_trans[1];
                        lidar[2] += cam_trans[2];
                        float augmented[3];
                        if (augmentation_rotation)
                            matrix_vector_3x3(augmentation_rotation, lidar, augmented);
                        else
                            memcpy(augmented, lidar, sizeof(augmented));
                        if (augmentation_translation) {
                            augmented[0] += augmentation_translation[0];
                            augmented[1] += augmentation_translation[1];
                            augmented[2] += augmentation_translation[2];
                        }
                        size_t output_index = (((((batch * cameras + camera) * depth_bins + depth) *
                                                  height + y) * width + x) * 3);
                        memcpy(geometry + output_index, augmented, sizeof(augmented));
                    }
        }
    }
    return 1;
}

static size_t depth_index(const bf_lss_desc *d, size_t b, size_t n,
                          size_t depth, size_t y, size_t x) {
    return (((((b * d->cameras + n) * d->depth_bins + depth) *
               d->feature_height + y) * d->feature_width + x));
}

static size_t context_index(const bf_lss_desc *d, size_t b, size_t n,
                            size_t channel, size_t y, size_t x) {
    return (((((b * d->cameras + n) * d->channels + channel) *
               d->feature_height + y) * d->feature_width + x));
}

static int softmax_terms(const float *logits, const bf_lss_desc *d,
                         size_t b, size_t n, size_t y, size_t x,
                         float *maximum, float *inverse_sum) {
    *maximum = -FLT_MAX;
    for (size_t depth = 0; depth < d->depth_bins; ++depth) {
        float value = logits[depth_index(d, b, n, depth, y, x)];
        if (!isfinite(value)) return 0;
        if (value > *maximum) *maximum = value;
    }
    double sum = 0.0;
    for (size_t depth = 0; depth < d->depth_bins; ++depth)
        sum += expf(logits[depth_index(d, b, n, depth, y, x)] - *maximum);
    *inverse_sum = (float)(1.0 / sum);
    return isfinite(*inverse_sum);
}

int bf_lss_lift_f32_ref(const float *logits, const float *context,
                        float *lifted, const bf_lss_desc *d) {
    size_t samples, lifted_count, bev_count;
    if (!logits || !context || !lifted ||
        !lss_contract(d, &samples, &lifted_count, &bev_count)) return 0;
    (void)samples; (void)lifted_count; (void)bev_count;
    for (size_t b = 0; b < d->batches; ++b)
        for (size_t n = 0; n < d->cameras; ++n)
            for (size_t y = 0; y < d->feature_height; ++y)
                for (size_t x = 0; x < d->feature_width; ++x) {
                    float maximum, inverse_sum;
                    if (!softmax_terms(logits, d, b, n, y, x, &maximum, &inverse_sum)) return 0;
                    for (size_t depth = 0; depth < d->depth_bins; ++depth) {
                        float probability = expf(logits[depth_index(d, b, n, depth, y, x)] - maximum) * inverse_sum;
                        size_t base = depth_index(d, b, n, depth, y, x) * d->channels;
                        for (size_t channel = 0; channel < d->channels; ++channel)
                            lifted[base + channel] = probability * context[context_index(d, b, n, channel, y, x)];
                    }
                }
    return 1;
}

static int geometry_cell(const float *point, const bf_lss_desc *d, int32_t cell[3]) {
    for (size_t axis = 0; axis < 3; ++axis) {
        float coordinate = (point[axis] - d->grid_minimum[axis]) / d->grid_step[axis];
        if (!isfinite(coordinate) || coordinate < (float)INT32_MIN ||
            coordinate > (float)INT32_MAX) return 0;
        cell[axis] = (int32_t)coordinate; /* torch .long(): truncate toward zero */
        if (cell[axis] < 0 || (size_t)cell[axis] >= d->grid_cells[axis]) return 0;
    }
    return 1;
}

static size_t bev_index(const bf_lss_desc *d, size_t b, size_t channel,
                        const int32_t cell[3]) {
    size_t collapsed_channel = channel * d->grid_cells[2] + (size_t)cell[2];
    return (((b * (d->channels * d->grid_cells[2]) + collapsed_channel) *
             d->grid_cells[0] + (size_t)cell[0]) * d->grid_cells[1] + (size_t)cell[1]);
}

int bf_lss_bev_pool_f32_ref(const float *lifted, const float *geometry,
                            float *bev, const bf_lss_desc *d) {
    size_t samples, lifted_count, bev_count;
    if (!lifted || !geometry || !bev ||
        !lss_contract(d, &samples, &lifted_count, &bev_count)) return 0;
    memset(bev, 0, bev_count * sizeof(*bev));
    for (size_t sample = 0; sample < samples; ++sample) {
        int32_t cell[3];
        if (!geometry_cell(geometry + sample * 3, d, cell)) continue;
        size_t b = sample / (d->cameras * d->depth_bins * d->feature_height * d->feature_width);
        for (size_t channel = 0; channel < d->channels; ++channel)
            bev[bev_index(d, b, channel, cell)] += lifted[sample * d->channels + channel];
    }
    return 1;
}

int bf_lss_lift_pool_f32_ref(const float *logits, const float *context,
                             const float *geometry, float *bev,
                             const bf_lss_desc *d) {
    size_t samples, lifted_count, bev_count;
    if (!logits || !context || !geometry || !bev ||
        !lss_contract(d, &samples, &lifted_count, &bev_count)) return 0;
    (void)samples; (void)lifted_count;
    memset(bev, 0, bev_count * sizeof(*bev));
    for (size_t b = 0; b < d->batches; ++b)
        for (size_t n = 0; n < d->cameras; ++n)
            for (size_t y = 0; y < d->feature_height; ++y)
                for (size_t x = 0; x < d->feature_width; ++x) {
                    float maximum, inverse_sum;
                    if (!softmax_terms(logits, d, b, n, y, x, &maximum, &inverse_sum)) return 0;
                    for (size_t depth = 0; depth < d->depth_bins; ++depth) {
                        size_t sample = depth_index(d, b, n, depth, y, x);
                        int32_t cell[3];
                        if (!geometry_cell(geometry + sample * 3, d, cell)) continue;
                        float probability = expf(logits[sample] - maximum) * inverse_sum;
                        for (size_t channel = 0; channel < d->channels; ++channel)
                            bev[bev_index(d, b, channel, cell)] +=
                                probability * context[context_index(d, b, n, channel, y, x)];
                    }
                }
    return 1;
}
