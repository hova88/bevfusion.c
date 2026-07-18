#include "bf_depth_head.h"
#include "bf_kernels.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *scale, *bias, *mean, *variance;
} bf_bound_bn;

typedef struct {
    const float *weight, *bias;
    bf_bound_bn bn;
} bf_bound_conv_bn;

struct bf_depth_head {
    bf_bound_conv_bn depth[3];
    bf_bound_conv_bn trunk[2];
    const float *output_weight, *output_bias;
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
        fail(error, cap, "missing depth-head tensor %s", name);
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

static int bind_bn(const bf_model *model, const char *prefix, size_t channels,
                   bf_bound_bn *bn, char *error, size_t cap) {
    const char *suffixes[4] = {"weight", "bias", "running_mean", "running_var"};
    const float **destinations[4] = {&bn->scale, &bn->bias, &bn->mean, &bn->variance};
    uint32_t dims[1] = {(uint32_t)channels};
    for (size_t i = 0; i < 4; ++i) {
        char name[128];
        int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[i]);
        if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "depth-head name overflow");
        *destinations[i] = bind_f32(model, name, 1, dims, error, cap);
        if (!*destinations[i]) return 0;
    }
    return 1;
}

static int bind_conv_bn(const bf_model *model, const char *base,
                        size_t conv_index, size_t bn_index,
                        size_t co, size_t ci, size_t kernel,
                        bf_bound_conv_bn *bound, char *error, size_t cap) {
    char name[128], prefix[128];
    uint32_t wdims[4] = {(uint32_t)co, (uint32_t)ci,
                         (uint32_t)kernel, (uint32_t)kernel};
    uint32_t bdims[1] = {(uint32_t)co};
    int n = snprintf(name, sizeof(name), "%s.%zu.weight", base, conv_index);
    if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "depth-head name overflow");
    bound->weight = bind_f32(model, name, 4, wdims, error, cap);
    n = snprintf(name, sizeof(name), "%s.%zu.bias", base, conv_index);
    if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "depth-head name overflow");
    bound->bias = bind_f32(model, name, 1, bdims, error, cap);
    n = snprintf(prefix, sizeof(prefix), "%s.%zu", base, bn_index);
    if (n < 0 || (size_t)n >= sizeof(prefix)) return fail(error, cap, "depth-head name overflow");
    return bound->weight && bound->bias && bind_bn(model, prefix, co, &bound->bn, error, cap);
}

int bf_depth_head_create(const bf_model *model, bf_depth_head **out,
                         char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid depth-head arguments");
    bf_depth_head *head = (bf_depth_head *)calloc(1, sizeof(*head));
    if (!head) return fail(error, cap, "depth-head allocation failed");
    if (!bind_conv_bn(model, "vtransform.dtransform", 0, 1, 8, 1, 1,
                      &head->depth[0], error, cap) ||
        !bind_conv_bn(model, "vtransform.dtransform", 3, 4, 32, 8, 5,
                      &head->depth[1], error, cap) ||
        !bind_conv_bn(model, "vtransform.dtransform", 6, 7, 64, 32, 5,
                      &head->depth[2], error, cap) ||
        !bind_conv_bn(model, "vtransform.depthnet", 0, 1, 256, 320, 3,
                      &head->trunk[0], error, cap) ||
        !bind_conv_bn(model, "vtransform.depthnet", 3, 4, 256, 256, 3,
                      &head->trunk[1], error, cap)) goto failure;
    {
        uint32_t wdims[4] = {198, 256, 1, 1}, bdims[1] = {198};
        head->output_weight = bind_f32(model, "vtransform.depthnet.6.weight", 4, wdims, error, cap);
        head->output_bias = bind_f32(model, "vtransform.depthnet.6.bias", 1, bdims, error, cap);
    }
    if (!head->output_weight || !head->output_bias) goto failure;
    *out = head;
    return 1;
failure:
    free(head);
    return 0;
}

void bf_depth_head_destroy(bf_depth_head *head) { free(head); }

size_t bf_depth_head_workspace_bytes(size_t height, size_t width) {
    if (!height || !width || height > SIZE_MAX / 8 || width > SIZE_MAX / 8) return 0;
    size_t hw, values;
    if (!checked_mul(height, width, &hw) || !checked_mul(hw, 768, &values) ||
        !checked_mul(values, sizeof(float), &values)) return 0;
    return values;
}

static void bn_relu(float *values, const bf_bound_bn *bn,
                    size_t channels, size_t height, size_t width) {
    bf_batch_norm_2d_f32_ref(values, bn->scale, bn->bias, bn->mean,
                             bn->variance, 1e-5f, values, 1, channels, height, width);
    bf_relu_f32_ref(values, values, channels * height * width);
}

static int conv_bn_relu(const float *input, const bf_bound_conv_bn *bound,
                        float *output, size_t ci, size_t co,
                        size_t height, size_t width, size_t kernel,
                        size_t stride, size_t padding) {
    bf_conv2d_desc desc = {1, ci, height, width, co, kernel, kernel,
                           stride, stride, padding, padding, 1, 1, 1};
    if (!bf_conv2d_f32_ref(input, bound->weight, bound->bias, output, &desc)) return 0;
    size_t out_h, out_w;
    if (!bf_conv2d_output_shape(&desc, &out_h, &out_w)) return 0;
    bn_relu(output, &bound->bn, co, out_h, out_w);
    return 1;
}

int bf_depth_head_forward_ref(const bf_depth_head *head,
                              const float *features, const float *depth,
                              size_t camera_batches, size_t height, size_t width,
                              float *logits, float *context, void *workspace,
                              size_t workspace_bytes, char *error, size_t cap) {
    size_t required = bf_depth_head_workspace_bytes(height, width);
    if (!head || !features || !depth || !camera_batches || !height || !width ||
        !logits || !context || !workspace || !required || workspace_bytes < required)
        return fail(error, cap, "invalid depth-head buffers or dimensions");
    size_t hw = height * width, image_h = 8 * height, image_w = 8 * width;
    float *buffer_a = (float *)workspace;
    float *buffer_b = buffer_a + 512 * hw;
    for (size_t b = 0; b < camera_batches; ++b) {
        if (!conv_bn_relu(depth + b * image_h * image_w, &head->depth[0],
                          buffer_a, 1, 8, image_h, image_w, 1, 1, 0) ||
            !conv_bn_relu(buffer_a, &head->depth[1], buffer_b,
                          8, 32, image_h, image_w, 5, 4, 2) ||
            !conv_bn_relu(buffer_b, &head->depth[2], buffer_a,
                          32, 64, 2 * height, 2 * width, 5, 2, 2))
            return fail(error, cap, "depth image encoder failed at camera %zu", b);
        memmove(buffer_a + 64 * hw, features + b * 256 * hw,
                256 * hw * sizeof(float));
        if (!conv_bn_relu(buffer_a, &head->trunk[0], buffer_b,
                          320, 256, height, width, 3, 1, 1) ||
            !conv_bn_relu(buffer_b, &head->trunk[1], buffer_a,
                          256, 256, height, width, 3, 1, 1))
            return fail(error, cap, "depth prediction trunk failed at camera %zu", b);
        bf_conv2d_desc final = {1, 256, height, width, 198, 1, 1,
                                1, 1, 0, 0, 1, 1, 1};
        if (!bf_conv2d_f32_ref(buffer_a, head->output_weight, head->output_bias,
                               buffer_b, &final))
            return fail(error, cap, "depth prediction output failed at camera %zu", b);
        memcpy(logits + b * 118 * hw, buffer_b, 118 * hw * sizeof(float));
        memcpy(context + b * 80 * hw, buffer_b + 118 * hw, 80 * hw * sizeof(float));
    }
    return 1;
}
