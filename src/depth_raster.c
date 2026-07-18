#include "bf_depth_raster.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int inverse_3x3_from_4x4(const float *m, float *out) {
    float a00 = m[0], a01 = m[1], a02 = m[2];
    float a10 = m[4], a11 = m[5], a12 = m[6];
    float a20 = m[8], a21 = m[9], a22 = m[10];
    float c00 = a11 * a22 - a12 * a21;
    float c01 = a02 * a21 - a01 * a22;
    float c02 = a01 * a12 - a02 * a11;
    float c10 = a12 * a20 - a10 * a22;
    float c11 = a00 * a22 - a02 * a20;
    float c12 = a02 * a10 - a00 * a12;
    float c20 = a10 * a21 - a11 * a20;
    float c21 = a01 * a20 - a00 * a21;
    float c22 = a00 * a11 - a01 * a10;
    float determinant = a00 * c00 + a01 * c10 + a02 * c20;
    if (!isfinite(determinant) || fabsf(determinant) <= FLT_MIN) return 0;
    float scale = 1.0f / determinant;
    out[0] = c00 * scale; out[1] = c01 * scale; out[2] = c02 * scale;
    out[3] = c10 * scale; out[4] = c11 * scale; out[5] = c12 * scale;
    out[6] = c20 * scale; out[7] = c21 * scale; out[8] = c22 * scale;
    return 1;
}

static void mat3_vec(const float *matrix4, const float *v, float *out) {
    out[0] = matrix4[0] * v[0] + matrix4[1] * v[1] + matrix4[2] * v[2];
    out[1] = matrix4[4] * v[0] + matrix4[5] * v[1] + matrix4[6] * v[2];
    out[2] = matrix4[8] * v[0] + matrix4[9] * v[1] + matrix4[10] * v[2];
}

static void compact_mat3_vec(const float *matrix, const float *v, float *out) {
    out[0] = matrix[0] * v[0] + matrix[1] * v[1] + matrix[2] * v[2];
    out[1] = matrix[3] * v[0] + matrix[4] * v[1] + matrix[5] * v[2];
    out[2] = matrix[6] * v[0] + matrix[7] * v[1] + matrix[8] * v[2];
}

int bf_depth_rasterize_f32_ref(const float *points, size_t point_count,
                               size_t point_stride, const float *lidar_aug,
                               const float *lidar_to_image,
                               const float *image_aug, float *depth,
                               size_t batches, size_t cameras,
                               size_t height, size_t width) {
    if ((!points && point_count) || point_stride < 4 || !lidar_aug ||
        !lidar_to_image || !image_aug || !depth || !batches || !cameras ||
        !height || !width) return 0;
    size_t count;
    if (!checked_mul(batches, cameras, &count) ||
        !checked_mul(count, height, &count) ||
        !checked_mul(count, width, &count)) return 0;
    memset(depth, 0, count * sizeof(*depth));
    for (size_t batch = 0; batch < batches; ++batch) {
        const float *augmentation = lidar_aug + batch * 16;
        float inverse[9];
        if (!inverse_3x3_from_4x4(augmentation, inverse)) return 0;
        for (size_t p = 0; p < point_count; ++p) {
            const float *point = points + p * point_stride;
            if (!isfinite(point[0]) || point[0] != (float)batch) continue;
            float centered[3] = {point[1] - augmentation[3],
                                 point[2] - augmentation[7],
                                 point[3] - augmentation[11]};
            if (!isfinite(centered[0]) || !isfinite(centered[1]) ||
                !isfinite(centered[2])) continue;
            float unaugmented[3];
            compact_mat3_vec(inverse, centered, unaugmented);
            for (size_t camera = 0; camera < cameras; ++camera) {
                size_t camera_index = batch * cameras + camera;
                const float *projection = lidar_to_image + camera_index * 16;
                const float *post = image_aug + camera_index * 16;
                float projected[3];
                mat3_vec(projection, unaugmented, projected);
                projected[0] += projection[3];
                projected[1] += projection[7];
                projected[2] += projection[11];
                if (!isfinite(projected[0]) || !isfinite(projected[1]) ||
                    !isfinite(projected[2])) continue;
                float distance = projected[2];
                if (distance < 1e-5f) distance = 1e-5f;
                if (distance > 1e5f) distance = 1e5f;
                float normalized[3] = {projected[0] / distance,
                                       projected[1] / distance, distance};
                float augmented[3];
                mat3_vec(post, normalized, augmented);
                augmented[0] += post[3];
                augmented[1] += post[7];
                if (!isfinite(augmented[0]) || !isfinite(augmented[1]) ||
                    augmented[0] < 0.0f || augmented[0] >= (float)width ||
                    augmented[1] < 0.0f || augmented[1] >= (float)height) continue;
                size_t x = (size_t)augmented[0], y = (size_t)augmented[1];
                depth[(camera_index * height + y) * width + x] = distance;
            }
        }
    }
    return 1;
}
