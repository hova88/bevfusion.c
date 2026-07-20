#include "bf_lss_downsample.h"
#include "bf_kernels.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const float *scale, *bias, *mean, *variance;
} bf_bound_bn;

struct bf_lss_downsample {
    const float *weight[3];
    bf_bound_bn bn[3];
};

static int fail(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, cap, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static const float *bind_f32(const bf_model *model, const char *name,
                             uint32_t rank, const uint32_t *dims,
                             char *error, size_t cap) {
    const bf_tensor *tensor = bf_model_find(model, name);
    if (!tensor) {
        fail(error, cap, "missing LSS downsample tensor %s", name);
        return NULL;
    }
    if (tensor->dtype != BF_DTYPE_F32 || tensor->rank != rank) {
        fail(error, cap, "%s: dtype/rank mismatch", name);
        return NULL;
    }
    for (uint32_t axis = 0; axis < rank; ++axis)
        if (tensor->dims[axis] != dims[axis]) {
            fail(error, cap, "%s: dimension %u is %u, expected %u",
                 name, axis, tensor->dims[axis], dims[axis]);
            return NULL;
        }
    return (const float *)tensor->data;
}

static int bind_bn(const bf_model *model, const char *prefix, bf_bound_bn *bn,
                   char *error, size_t cap) {
    const char *suffixes[4] = {"weight", "bias", "running_mean", "running_var"};
    const float **destinations[4] = {&bn->scale, &bn->bias, &bn->mean, &bn->variance};
    uint32_t dims[1] = {80};
    for (size_t i = 0; i < 4; ++i) {
        char name[128];
        int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[i]);
        if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "LSS downsample name overflow");
        *destinations[i] = bind_f32(model, name, 1, dims, error, cap);
        if (!*destinations[i]) return 0;
    }
    return 1;
}

int bf_lss_downsample_create(const bf_model *model, bf_lss_downsample **out,
                             char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid LSS downsample arguments");
    bf_lss_downsample *down = (bf_lss_downsample *)calloc(1, sizeof(*down));
    if (!down) return fail(error, cap, "LSS downsample allocation failed");
    const size_t conv_indices[3] = {0, 3, 6}, bn_indices[3] = {1, 4, 7};
    uint32_t wdims[4] = {80, 80, 3, 3};
    for (size_t layer = 0; layer < 3; ++layer) {
        char name[128], prefix[128];
        snprintf(name, sizeof(name), "vtransform.downsample.%zu.weight", conv_indices[layer]);
        down->weight[layer] = bind_f32(model, name, 4, wdims, error, cap);
        snprintf(prefix, sizeof(prefix), "vtransform.downsample.%zu", bn_indices[layer]);
        if (!down->weight[layer] || !bind_bn(model, prefix, &down->bn[layer], error, cap)) {
            free(down);
            return 0;
        }
    }
    *out = down;
    return 1;
}

void bf_lss_downsample_destroy(bf_lss_downsample *downsample) { free(downsample); }

size_t bf_lss_downsample_workspace_bytes(size_t height, size_t width) {
    if (!height || !width || (height & 1) || (width & 1)) return 0;
    size_t full, half;
    if (!checked_mul(height, width, &full) || !checked_mul(full, 80, &full) ||
        !checked_mul(height / 2, width / 2, &half) || !checked_mul(half, 80, &half) ||
        full > SIZE_MAX - half || !checked_mul(full + half, sizeof(float), &full)) return 0;
    return full;
}

static void bn_relu(float *values, const bf_bound_bn *bn,
                    size_t height, size_t width) {
    bf_batch_norm_2d_f32(values, bn->scale, bn->bias, bn->mean,
                             bn->variance, 1e-5f, values, 1, 80, height, width);
    bf_relu_f32(values, values, 80 * height * width);
}

static int conv(const float *input, const float *weight, float *output,
                size_t height, size_t width, size_t stride) {
    bf_conv2d_desc desc = {1, 80, height, width, 80, 3, 3,
                           stride, stride, 1, 1, 1, 1, 1};
    return bf_conv2d_f32(input, weight, NULL, output, &desc);
}

int bf_lss_downsample_forward_ref(const bf_lss_downsample *down,
                                  const float *input, size_t batches,
                                  size_t height, size_t width, float *output,
                                  void *workspace, size_t workspace_bytes,
                                  char *error, size_t cap) {
    size_t required = bf_lss_downsample_workspace_bytes(height, width);
    if (!down || !input || !batches || !height || !width || !output ||
        !workspace || !required || workspace_bytes < required)
        return fail(error, cap, "invalid LSS downsample buffers or dimensions");
    size_t full_hw = height * width, half_h = height / 2, half_w = width / 2;
    size_t half_hw = half_h * half_w;
    float *full = (float *)workspace;
    float *half = full + 80 * full_hw;
    for (size_t b = 0; b < batches; ++b) {
        if (!conv(input + b * 80 * full_hw, down->weight[0], full,
                  height, width, 1)) return fail(error, cap, "LSS downsample convolution 0 failed");
        bn_relu(full, &down->bn[0], height, width);
        if (!conv(full, down->weight[1], half, height, width, 2))
            return fail(error, cap, "LSS downsample convolution 1 failed");
        bn_relu(half, &down->bn[1], half_h, half_w);
        /* Reuse the beginning of the full-resolution buffer for the half-size result. */
        if (!conv(half, down->weight[2], full, half_h, half_w, 1))
            return fail(error, cap, "LSS downsample convolution 2 failed");
        bn_relu(full, &down->bn[2], half_h, half_w);
        float *batch_output = output + b * 80 * half_hw;
        for (size_t c = 0; c < 80; ++c)
            for (size_t x = 0; x < half_h; ++x)
                for (size_t y = 0; y < half_w; ++y)
                    batch_output[(c * half_w + y) * half_h + x] =
                        full[(c * half_h + x) * half_w + y];
    }
    return 1;
}
