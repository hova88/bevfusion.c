#include "bf_image_fpn.h"
#include "bf_kernels.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const float *scale, *bias, *mean, *variance;
} bf_bound_bn;

struct bf_image_fpn {
    const float *lateral_weight[2];
    bf_bound_bn lateral_bn[2];
    const float *output_weight[2];
    bf_bound_bn output_bn[2];
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
        fail(error, cap, "missing image FPN tensor %s", name);
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
    uint32_t dims[1] = {256};
    for (size_t i = 0; i < 4; ++i) {
        char name[128];
        int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[i]);
        if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "image FPN name overflow");
        *destinations[i] = bind_f32(model, name, 1, dims, error, cap);
        if (!*destinations[i]) return 0;
    }
    return 1;
}

int bf_image_fpn_create(const bf_model *model, bf_image_fpn **out,
                        char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid image FPN arguments");
    bf_image_fpn *fpn = (bf_image_fpn *)calloc(1, sizeof(*fpn));
    if (!fpn) return fail(error, cap, "image FPN allocation failed");
    const uint32_t lateral_inputs[2] = {448, 1152};
    for (size_t level = 0; level < 2; ++level) {
        char name[128], prefix[128];
        uint32_t lateral_dims[4] = {256, lateral_inputs[level], 1, 1};
        uint32_t output_dims[4] = {256, 256, 3, 3};
        snprintf(name, sizeof(name), "neck.lateral_convs.%zu.conv.weight", level);
        fpn->lateral_weight[level] = bind_f32(model, name, 4, lateral_dims, error, cap);
        snprintf(prefix, sizeof(prefix), "neck.lateral_convs.%zu.bn", level);
        if (!fpn->lateral_weight[level] ||
            !bind_bn(model, prefix, &fpn->lateral_bn[level], error, cap)) goto failure;
        snprintf(name, sizeof(name), "neck.fpn_convs.%zu.conv.weight", level);
        fpn->output_weight[level] = bind_f32(model, name, 4, output_dims, error, cap);
        snprintf(prefix, sizeof(prefix), "neck.fpn_convs.%zu.bn", level);
        if (!fpn->output_weight[level] ||
            !bind_bn(model, prefix, &fpn->output_bn[level], error, cap)) goto failure;
    }
    *out = fpn;
    return 1;
failure:
    free(fpn);
    return 0;
}

void bf_image_fpn_destroy(bf_image_fpn *fpn) { free(fpn); }

size_t bf_image_fpn_workspace_bytes(size_t height, size_t width) {
    if (!height || !width || height > SIZE_MAX - 3 || width > SIZE_MAX - 3) return 0;
    size_t h1 = (height + 1) / 2, w1 = (width + 1) / 2;
    size_t level0, level1, concat;
    if (!checked_mul(height, width, &level0) || !checked_mul(level0, 448, &level0) ||
        !checked_mul(h1, w1, &level1) || !checked_mul(level1, 1152, &level1)) return 0;
    concat = level0 > level1 ? level0 : level1;
    if (!checked_mul(height, width, &level0) || !checked_mul(level0, 256, &level0) ||
        concat > SIZE_MAX - level0 || !checked_mul(concat + level0, sizeof(float), &concat)) return 0;
    return concat;
}

static void bn_relu(float *values, const bf_bound_bn *bn,
                    size_t height, size_t width) {
    bf_batch_norm_2d_f32_ref(values, bn->scale, bn->bias, bn->mean,
                             bn->variance, 1e-5f, values, 1, 256, height, width);
    bf_relu_f32_ref(values, values, 256 * height * width);
}

static void bilinear_channel(const float *input, size_t in_h, size_t in_w,
                             float *output, size_t out_h, size_t out_w) {
    for (size_t y = 0; y < out_h; ++y) {
        float source_y = ((float)y + 0.5f) * (float)in_h / (float)out_h - 0.5f;
        if (source_y < 0.0f) source_y = 0.0f;
        size_t y0 = (size_t)floorf(source_y);
        size_t y1 = y0 + 1 < in_h ? y0 + 1 : y0;
        float fy = source_y - (float)y0;
        for (size_t x = 0; x < out_w; ++x) {
            float source_x = ((float)x + 0.5f) * (float)in_w / (float)out_w - 0.5f;
            if (source_x < 0.0f) source_x = 0.0f;
            size_t x0 = (size_t)floorf(source_x);
            size_t x1 = x0 + 1 < in_w ? x0 + 1 : x0;
            float fx = source_x - (float)x0;
            float top = input[y0 * in_w + x0] * (1.0f - fx) + input[y0 * in_w + x1] * fx;
            float bottom = input[y1 * in_w + x0] * (1.0f - fx) + input[y1 * in_w + x1] * fx;
            output[y * out_w + x] = top * (1.0f - fy) + bottom * fy;
        }
    }
}

static int fuse_level(const bf_image_fpn *fpn, size_t level,
                      const float *lower, size_t lower_channels,
                      size_t out_h, size_t out_w,
                      const float *upper, size_t upper_channels,
                      size_t in_h, size_t in_w, float *output,
                      float *concat, float *temporary) {
    size_t out_hw = out_h * out_w, in_hw = in_h * in_w;
    for (size_t c = 0; c < lower_channels; ++c)
        for (size_t i = 0; i < out_hw; ++i)
            concat[c * out_hw + i] = lower[c * out_hw + i];
    for (size_t c = 0; c < upper_channels; ++c)
        bilinear_channel(upper + c * in_hw, in_h, in_w,
                         concat + (lower_channels + c) * out_hw, out_h, out_w);
    bf_conv2d_desc lateral = {1, lower_channels + upper_channels, out_h, out_w,
                              256, 1, 1, 1, 1, 0, 0, 1, 1, 1};
    if (!bf_conv2d_f32_ref(concat, fpn->lateral_weight[level], NULL,
                           temporary, &lateral)) return 0;
    bn_relu(temporary, &fpn->lateral_bn[level], out_h, out_w);
    bf_conv2d_desc final = {1, 256, out_h, out_w, 256, 3, 3,
                            1, 1, 1, 1, 1, 1, 1};
    if (!bf_conv2d_f32_ref(temporary, fpn->output_weight[level], NULL,
                           output, &final)) return 0;
    bn_relu(output, &fpn->output_bn[level], out_h, out_w);
    return 1;
}

int bf_image_fpn_forward_ref(const bf_image_fpn *fpn,
                             const float *input0, const float *input1,
                             const float *input2, size_t batches,
                             size_t h0, size_t w0, float *output0,
                             float *output1, void *workspace,
                             size_t workspace_bytes, char *error, size_t cap) {
    size_t required = bf_image_fpn_workspace_bytes(h0, w0);
    if (!fpn || !input0 || !input1 || !input2 || !batches || !h0 || !w0 ||
        !output0 || !output1 || !workspace || !required || workspace_bytes < required)
        return fail(error, cap, "invalid image FPN buffers or dimensions");
    size_t h1 = (h0 + 1) / 2, w1 = (w0 + 1) / 2;
    size_t h2 = (h1 + 1) / 2, w2 = (w1 + 1) / 2;
    size_t concat0 = 448 * h0 * w0, concat1 = 1152 * h1 * w1;
    size_t concat_count = concat0 > concat1 ? concat0 : concat1;
    float *concat = (float *)workspace;
    float *temporary = concat + concat_count;
    for (size_t b = 0; b < batches; ++b) {
        float *batch_out1 = output1 + b * 256 * h1 * w1;
        if (!fuse_level(fpn, 1, input1 + b * 384 * h1 * w1, 384, h1, w1,
                        input2 + b * 768 * h2 * w2, 768, h2, w2,
                        batch_out1, concat, temporary))
            return fail(error, cap, "image FPN level 1 failed");
        if (!fuse_level(fpn, 0, input0 + b * 192 * h0 * w0, 192, h0, w0,
                        batch_out1, 256, h1, w1,
                        output0 + b * 256 * h0 * w0, concat, temporary))
            return fail(error, cap, "image FPN level 0 failed");
    }
    return 1;
}
