#include "bf_bev_stage.h"
#include "bf_kernels.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *scale;
    const float *bias;
    const float *mean;
    const float *variance;
} bf_bound_bn;

struct bf_bev_stage {
    const float *fuser_weight;
    bf_bound_bn fuser_bn;
    const float *block_weight[2][6];
    bf_bound_bn block_bn[2][6];
    const float *deblock_weight[2];
    bf_bound_bn deblock_bn[2];
    const float *shared_weight;
    const float *shared_bias;
    const float *heatmap_mid_weight;
    bf_bound_bn heatmap_mid_bn;
    const float *heatmap_out_weight;
    const float *heatmap_out_bias;
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

static const float *bind_f32(const bf_model *model, const char *name,
                             uint32_t rank, const uint32_t *dims,
                             char *error, size_t cap) {
    const bf_tensor *tensor = bf_model_find(model, name);
    if (!tensor) {
        fail(error, cap, "missing BEV stage tensor %s", name);
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
    char name[128];
    const char *suffixes[4] = {"weight", "bias", "running_mean", "running_var"};
    const float **destinations[4] = {&bn->scale, &bn->bias, &bn->mean, &bn->variance};
    uint32_t dims[1] = {(uint32_t)channels};
    for (size_t i = 0; i < 4; ++i) {
        int written = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[i]);
        if (written < 0 || (size_t)written >= sizeof(name)) return fail(error, cap, "BN name overflow");
        *destinations[i] = bind_f32(model, name, 1, dims, error, cap);
        if (!*destinations[i]) return 0;
    }
    return 1;
}

static const float *bind_conv(const bf_model *model, const char *name,
                              size_t co, size_t ci, size_t kh, size_t kw,
                              char *error, size_t cap) {
    uint32_t dims[4] = {(uint32_t)co, (uint32_t)ci, (uint32_t)kh, (uint32_t)kw};
    return bind_f32(model, name, 4, dims, error, cap);
}

int bf_bev_stage_create(const bf_model *model, bf_bev_stage **out,
                        char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid BEV stage arguments");
    bf_bev_stage *stage = (bf_bev_stage *)calloc(1, sizeof(*stage));
    if (!stage) return fail(error, cap, "BEV stage allocation failed");
    stage->fuser_weight = bind_conv(model, "fuser.conv.0.weight", 256, 336, 3, 3, error, cap);
    if (!stage->fuser_weight || !bind_bn(model, "fuser.conv.1", 256, &stage->fuser_bn, error, cap)) goto failure;
    const int conv_indices[6] = {1, 4, 7, 10, 13, 16};
    const int bn_indices[6] = {2, 5, 8, 11, 14, 17};
    for (size_t block = 0; block < 2; ++block) {
        size_t channels = block ? 256 : 128;
        size_t first_input = block ? 128 : 256;
        for (size_t layer = 0; layer < 6; ++layer) {
            char name[128], prefix[128];
            snprintf(name, sizeof(name), "backbone_2d.blocks.%zu.%d.weight", block, conv_indices[layer]);
            stage->block_weight[block][layer] = bind_conv(
                model, name, channels, layer ? channels : first_input, 3, 3, error, cap);
            snprintf(prefix, sizeof(prefix), "backbone_2d.blocks.%zu.%d", block, bn_indices[layer]);
            if (!stage->block_weight[block][layer] ||
                !bind_bn(model, prefix, channels, &stage->block_bn[block][layer], error, cap))
                goto failure;
        }
    }
    stage->deblock_weight[0] = bind_conv(model, "backbone_2d.deblocks.0.0.weight",
                                         256, 128, 1, 1, error, cap);
    /* ConvTranspose2d is stored [input,output,kH,kW]. */
    stage->deblock_weight[1] = bind_conv(model, "backbone_2d.deblocks.1.0.weight",
                                         256, 256, 2, 2, error, cap);
    if (!stage->deblock_weight[0] || !stage->deblock_weight[1] ||
        !bind_bn(model, "backbone_2d.deblocks.0.1", 256, &stage->deblock_bn[0], error, cap) ||
        !bind_bn(model, "backbone_2d.deblocks.1.1", 256, &stage->deblock_bn[1], error, cap))
        goto failure;
    stage->shared_weight = bind_conv(model, "dense_head.shared_conv.weight", 128, 512, 3, 3, error, cap);
    { uint32_t dims[1] = {128}; stage->shared_bias = bind_f32(model, "dense_head.shared_conv.bias", 1, dims, error, cap); }
    stage->heatmap_mid_weight = bind_conv(model, "dense_head.heatmap_head.0.conv.weight", 128, 128, 3, 3, error, cap);
    if (!stage->heatmap_mid_weight ||
        !bind_bn(model, "dense_head.heatmap_head.0.bn", 128, &stage->heatmap_mid_bn, error, cap)) goto failure;
    stage->heatmap_out_weight = bind_conv(model, "dense_head.heatmap_head.1.weight", 10, 128, 3, 3, error, cap);
    { uint32_t dims[1] = {10}; stage->heatmap_out_bias = bind_f32(model, "dense_head.heatmap_head.1.bias", 1, dims, error, cap); }
    if (!stage->shared_weight || !stage->shared_bias ||
        !stage->heatmap_out_weight || !stage->heatmap_out_bias) goto failure;
    *out = stage;
    return 1;
failure:
    free(stage);
    return 0;
}

void bf_bev_stage_destroy(bf_bev_stage *stage) { free(stage); }

size_t bf_bev_stage_workspace_bytes(size_t height, size_t width) {
    if (!height || !width || height > SIZE_MAX / width) return 0;
    size_t spatial = height * width;
    if (spatial > SIZE_MAX / (2 * 256 * sizeof(float))) return 0;
    return 2 * 256 * spatial * sizeof(float);
}

static int conv(const float *input, const float *weight, const float *bias,
                float *output, size_t ci, size_t co, size_t height, size_t width,
                size_t stride, size_t padding) {
    bf_conv2d_desc desc = {1, ci, height, width, co, 3, 3,
                           stride, stride, padding, padding, 1, 1, 1};
    return bf_conv2d_f32_ref(input, weight, bias, output, &desc);
}

static void bn_relu(float *values, const bf_bound_bn *bn,
                    size_t channels, size_t height, size_t width) {
    bf_batch_norm_2d_f32_ref(values, bn->scale, bn->bias, bn->mean, bn->variance,
                             1e-3f, values, 1, channels, height, width);
    bf_relu_f32_ref(values, values, channels * height * width);
}

int bf_bev_stage_forward_ref(const bf_bev_stage *stage, const float *input,
                             size_t batches, size_t height, size_t width,
                             float *spatial, float *shared, float *heatmap,
                             void *workspace, size_t workspace_bytes,
                             char *error, size_t cap) {
    size_t required = bf_bev_stage_workspace_bytes(height, width);
    if (!stage || !input || !batches || !height || !width || (height & 1) || (width & 1) ||
        !spatial || !shared || !heatmap || !workspace || !required || workspace_bytes < required)
        return fail(error, cap, "invalid BEV stage buffers or dimensions");
    size_t hw = height * width, half_h = height / 2, half_w = width / 2;
    float *buffer_a = (float *)workspace;
    float *buffer_b = buffer_a + 256 * hw;
    for (size_t batch = 0; batch < batches; ++batch) {
        const float *batch_input = input + batch * 336 * hw;
        float *batch_spatial = spatial + batch * 512 * hw;
        float *batch_shared = shared + batch * 128 * hw;
        float *batch_heatmap = heatmap + batch * 10 * hw;
        if (!conv(batch_input, stage->fuser_weight, NULL, buffer_a, 336, 256,
                  height, width, 1, 1)) return fail(error, cap, "fuser convolution failed");
        bn_relu(buffer_a, &stage->fuser_bn, 256, height, width);
        const float *current = buffer_a;
        float *next = buffer_b;
        for (size_t layer = 0; layer < 6; ++layer) {
            size_t ci = layer ? 128 : 256;
            if (!conv(current, stage->block_weight[0][layer], NULL, next,
                      ci, 128, height, width, 1, 1))
                return fail(error, cap, "BEV block 0 convolution %zu failed", layer);
            bn_relu(next, &stage->block_bn[0][layer], 128, height, width);
            current = next;
            next = next == buffer_a ? buffer_b : buffer_a;
        }
        bf_conv2d_desc up0_desc = {1, 128, height, width, 256, 1, 1,
                                   1, 1, 0, 0, 1, 1, 1};
        if (!bf_conv2d_f32_ref(current, stage->deblock_weight[0], NULL,
                               batch_spatial, &up0_desc))
            return fail(error, cap, "BEV deblock 0 failed");
        bn_relu(batch_spatial, &stage->deblock_bn[0], 256, height, width);
        next = current == buffer_a ? buffer_b : buffer_a;
        for (size_t layer = 0; layer < 6; ++layer) {
            size_t ci = layer ? 256 : 128;
            size_t in_h = layer ? half_h : height;
            size_t in_w = layer ? half_w : width;
            size_t stride = layer ? 1 : 2;
            if (!conv(current, stage->block_weight[1][layer], NULL, next,
                      ci, 256, in_h, in_w, stride, 1))
                return fail(error, cap, "BEV block 1 convolution %zu failed", layer);
            bn_relu(next, &stage->block_bn[1][layer], 256, half_h, half_w);
            current = next;
            next = next == buffer_a ? buffer_b : buffer_a;
        }
        bf_conv_transpose2d_desc up1_desc = {
            1, 256, half_h, half_w, 256, 2, 2, 2, 2,
            0, 0, 0, 0, 1, 1, 1
        };
        if (!bf_conv_transpose2d_f32_ref(current, stage->deblock_weight[1], NULL,
                                         batch_spatial + 256 * hw, &up1_desc))
            return fail(error, cap, "BEV deblock 1 failed");
        bn_relu(batch_spatial + 256 * hw, &stage->deblock_bn[1], 256, height, width);
        if (!conv(batch_spatial, stage->shared_weight, stage->shared_bias,
                  batch_shared, 512, 128, height, width, 1, 1))
            return fail(error, cap, "TransFusion shared convolution failed");
        if (!conv(batch_shared, stage->heatmap_mid_weight, NULL,
                  buffer_a, 128, 128, height, width, 1, 1))
            return fail(error, cap, "TransFusion heatmap middle convolution failed");
        bn_relu(buffer_a, &stage->heatmap_mid_bn, 128, height, width);
        if (!conv(buffer_a, stage->heatmap_out_weight, stage->heatmap_out_bias,
                  batch_heatmap, 128, 10, height, width, 1, 1))
            return fail(error, cap, "TransFusion heatmap output convolution failed");
    }
    return 1;
}
