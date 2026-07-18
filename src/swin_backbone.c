#include "bf_swin_backbone.h"
#include "bf_kernels.h"
#include "bf_swin.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BF_SWIN_STAGES 4u
#define BF_SWIN_MAX_BLOCKS 6u

typedef struct {
    const float *norm1_scale, *norm1_bias;
    const float *relative_bias;
    const int64_t *relative_index;
    const float *qkv_weight, *qkv_bias;
    const float *projection_weight, *projection_bias;
    const float *norm2_scale, *norm2_bias;
    const float *ffn1_weight, *ffn1_bias;
    const float *ffn2_weight, *ffn2_bias;
} bf_bound_swin_block;

typedef struct {
    const float *norm_scale, *norm_bias;
    const float *reduction_weight;
} bf_bound_patch_merge;

struct bf_swin_backbone {
    const float *patch_weight, *patch_bias;
    const float *patch_norm_scale, *patch_norm_bias;
    bf_bound_swin_block blocks[BF_SWIN_STAGES][BF_SWIN_MAX_BLOCKS];
    bf_bound_patch_merge merging[3];
    const float *output_norm_scale[3], *output_norm_bias[3];
};

static const size_t stage_channels[4] = {96, 192, 384, 768};
static const size_t stage_heads[4] = {3, 6, 12, 24};
static const size_t stage_depths[4] = {2, 2, 6, 2};

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

static const void *bind_tensor(const bf_model *model, const char *name,
                               uint32_t dtype, uint32_t rank,
                               const uint32_t *dims, char *error, size_t cap) {
    const bf_tensor *tensor = bf_model_find(model, name);
    if (!tensor) {
        fail(error, cap, "missing Swin tensor %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->rank != rank) {
        fail(error, cap, "%s: dtype/rank mismatch", name);
        return NULL;
    }
    for (uint32_t axis = 0; axis < rank; ++axis)
        if (tensor->dims[axis] != dims[axis]) {
            fail(error, cap, "%s: dimension %u is %u, expected %u",
                 name, axis, tensor->dims[axis], dims[axis]);
            return NULL;
        }
    return tensor->data;
}

static const float *bind_f32(const bf_model *model, const char *name,
                             uint32_t rank, const uint32_t *dims,
                             char *error, size_t cap) {
    return (const float *)bind_tensor(model, name, BF_DTYPE_F32, rank, dims,
                                      error, cap);
}

static int make_name(char *name, size_t cap, char *error, size_t error_cap,
                     const char *format, size_t a, size_t b) {
    int n = snprintf(name, cap, format, a, b);
    return n >= 0 && (size_t)n < cap ? 1 : fail(error, error_cap, "Swin tensor name overflow");
}

static const float *bind_block_f32(const bf_model *model, size_t stage,
                                   size_t block, const char *suffix,
                                   uint32_t rank, const uint32_t *dims,
                                   char *error, size_t cap) {
    char prefix[96], name[160];
    if (!make_name(prefix, sizeof(prefix), error, cap,
                   "image_backbone.stages.%zu.blocks.%zu", stage, block)) return NULL;
    int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        fail(error, cap, "Swin tensor name overflow");
        return NULL;
    }
    return bind_f32(model, name, rank, dims, error, cap);
}

int bf_swin_backbone_create(const bf_model *model, bf_swin_backbone **out,
                            char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid Swin backbone arguments");
    bf_swin_backbone *net = (bf_swin_backbone *)calloc(1, sizeof(*net));
    if (!net) return fail(error, cap, "Swin backbone allocation failed");
    const uint32_t patch_dims[4] = {96, 3, 4, 4}, d96[1] = {96};
    net->patch_weight = bind_f32(model, "image_backbone.patch_embed.projection.weight", 4, patch_dims, error, cap);
    net->patch_bias = bind_f32(model, "image_backbone.patch_embed.projection.bias", 1, d96, error, cap);
    net->patch_norm_scale = bind_f32(model, "image_backbone.patch_embed.norm.weight", 1, d96, error, cap);
    net->patch_norm_bias = bind_f32(model, "image_backbone.patch_embed.norm.bias", 1, d96, error, cap);
    if (!net->patch_weight || !net->patch_bias || !net->patch_norm_scale || !net->patch_norm_bias) goto failure;

    for (size_t stage = 0; stage < 4; ++stage) {
        uint32_t c = (uint32_t)stage_channels[stage];
        uint32_t h = (uint32_t)stage_heads[stage];
        uint32_t dc[1] = {c}, qkvw[2] = {3 * c, c}, qkvb[1] = {3 * c};
        uint32_t square[2] = {c, c}, rel[2] = {169, h}, index[2] = {49, 49};
        uint32_t ffn1w[2] = {4 * c, c}, ffn1b[1] = {4 * c};
        uint32_t ffn2w[2] = {c, 4 * c};
        for (size_t block = 0; block < stage_depths[stage]; ++block) {
            bf_bound_swin_block *bound = &net->blocks[stage][block];
#define BIND_BLOCK(field, suffix, rank, dims) do { \
    bound->field = bind_block_f32(model, stage, block, suffix, rank, dims, error, cap); \
    if (!bound->field) goto failure; \
} while (0)
            BIND_BLOCK(norm1_scale, "norm1.weight", 1, dc);
            BIND_BLOCK(norm1_bias, "norm1.bias", 1, dc);
            BIND_BLOCK(relative_bias, "attn.w_msa.relative_position_bias_table", 2, rel);
            {
                char name[160];
                int n = snprintf(name, sizeof(name),
                    "image_backbone.stages.%zu.blocks.%zu.attn.w_msa.relative_position_index", stage, block);
                if (n < 0 || (size_t)n >= sizeof(name)) goto failure;
                bound->relative_index = (const int64_t *)bind_tensor(model, name, BF_DTYPE_I64, 2, index, error, cap);
                if (!bound->relative_index) goto failure;
            }
            BIND_BLOCK(qkv_weight, "attn.w_msa.qkv.weight", 2, qkvw);
            BIND_BLOCK(qkv_bias, "attn.w_msa.qkv.bias", 1, qkvb);
            BIND_BLOCK(projection_weight, "attn.w_msa.proj.weight", 2, square);
            BIND_BLOCK(projection_bias, "attn.w_msa.proj.bias", 1, dc);
            BIND_BLOCK(norm2_scale, "norm2.weight", 1, dc);
            BIND_BLOCK(norm2_bias, "norm2.bias", 1, dc);
            BIND_BLOCK(ffn1_weight, "ffn.layers.0.0.weight", 2, ffn1w);
            BIND_BLOCK(ffn1_bias, "ffn.layers.0.0.bias", 1, ffn1b);
            BIND_BLOCK(ffn2_weight, "ffn.layers.1.weight", 2, ffn2w);
            BIND_BLOCK(ffn2_bias, "ffn.layers.1.bias", 1, dc);
#undef BIND_BLOCK
        }
    }
    for (size_t stage = 0; stage < 3; ++stage) {
        uint32_t c = (uint32_t)stage_channels[stage];
        uint32_t four_c[1] = {4 * c}, reduction[2] = {2 * c, 4 * c};
        char name[128];
#define BIND_MERGE(field, suffix, rank, dims) do { \
    int n = snprintf(name, sizeof(name), "image_backbone.stages.%zu.downsample.%s", stage, suffix); \
    if (n < 0 || (size_t)n >= sizeof(name)) goto failure; \
    net->merging[stage].field = bind_f32(model, name, rank, dims, error, cap); \
    if (!net->merging[stage].field) goto failure; \
} while (0)
        BIND_MERGE(norm_scale, "norm.weight", 1, four_c);
        BIND_MERGE(norm_bias, "norm.bias", 1, four_c);
        BIND_MERGE(reduction_weight, "reduction.weight", 2, reduction);
#undef BIND_MERGE
    }
    for (size_t output = 0; output < 3; ++output) {
        uint32_t dims[1] = {(uint32_t)stage_channels[output + 1]};
        char name[96];
        int n = snprintf(name, sizeof(name), "image_backbone.norm%zu.weight", output + 1);
        if (n < 0 || (size_t)n >= sizeof(name)) goto failure;
        net->output_norm_scale[output] = bind_f32(model, name, 1, dims, error, cap);
        n = snprintf(name, sizeof(name), "image_backbone.norm%zu.bias", output + 1);
        if (n < 0 || (size_t)n >= sizeof(name)) goto failure;
        net->output_norm_bias[output] = bind_f32(model, name, 1, dims, error, cap);
        if (!net->output_norm_scale[output] || !net->output_norm_bias[output]) goto failure;
    }
    *out = net;
    return 1;
failure:
    free(net);
    return 0;
}

void bf_swin_backbone_destroy(bf_swin_backbone *backbone) { free(backbone); }

int bf_swin_backbone_output_shapes(size_t input_height, size_t input_width,
                                   bf_swin_backbone_shapes *shapes) {
    if (!input_height || !input_width || !shapes ||
        input_height > SIZE_MAX - 31 || input_width > SIZE_MAX - 31) return 0;
    for (size_t i = 0; i < 3; ++i) {
        size_t divisor = (size_t)8 << i;
        shapes->height[i] = (input_height + divisor - 1) / divisor;
        shapes->width[i] = (input_width + divisor - 1) / divisor;
        shapes->channels[i] = stage_channels[i + 1];
    }
    return 1;
}

size_t bf_swin_backbone_workspace_bytes(size_t batches,
                                        size_t input_height, size_t input_width) {
    if (!batches || !input_height || !input_width ||
        input_height > SIZE_MAX - 3 || input_width > SIZE_MAX - 3) return 0;
    size_t h = (input_height + 3) / 4, w = (input_width + 3) / 4, values;
    if (!checked_mul(batches, h, &values) || !checked_mul(values, w, &values) ||
        !checked_mul(values, 96, &values) || !checked_mul(values, 6, &values) ||
        !checked_mul(values, sizeof(float), &values)) return 0;
    bf_swin_window_desc largest = {1, 1, 1, 768, 24, 7, 3};
    size_t attention = bf_swin_shifted_window_workspace_bytes(&largest);
    return attention && values <= SIZE_MAX - attention ? values + attention : 0;
}

static void patch_embed(const bf_swin_backbone *net, const float *input,
                        float *output, size_t batches, size_t input_h,
                        size_t input_w, size_t output_h, size_t output_w) {
    size_t input_hw = input_h * input_w;
    for (size_t b = 0; b < batches; ++b)
        for (size_t oy = 0; oy < output_h; ++oy)
            for (size_t ox = 0; ox < output_w; ++ox)
                for (size_t co = 0; co < 96; ++co) {
                    float sum = net->patch_bias[co];
                    for (size_t ci = 0; ci < 3; ++ci)
                        for (size_t ky = 0; ky < 4; ++ky)
                            for (size_t kx = 0; kx < 4; ++kx) {
                                size_t y = 4 * oy + ky, x = 4 * ox + kx;
                                if (y < input_h && x < input_w)
                                    sum += input[(b * 3 + ci) * input_hw + y * input_w + x] *
                                           net->patch_weight[((co * 3 + ci) * 4 + ky) * 4 + kx];
                            }
                    output[((b * output_h + oy) * output_w + ox) * 96 + co] = sum;
                }
}

static int run_block(const bf_bound_swin_block *block, float *current,
                     float *normal, float *hidden, size_t batches,
                     size_t height, size_t width, size_t channels,
                     size_t heads, size_t shift,
                     void *attention_workspace, size_t attention_bytes) {
    size_t rows = batches * height * width, count = rows * channels;
    bf_layer_norm_f32_ref(current, block->norm1_scale, block->norm1_bias,
                          1e-5f, normal, rows, channels);
    bf_swin_window_desc desc = {batches, height, width, channels, heads, 7, shift};
    if (!bf_swin_shifted_window_f32_workspace_ref(
            normal, block->qkv_weight, block->qkv_bias, block->relative_bias,
            block->relative_index, block->projection_weight,
            block->projection_bias, hidden, &desc,
            attention_workspace, attention_bytes)) return 0;
    for (size_t i = 0; i < count; ++i) current[i] += hidden[i];
    bf_layer_norm_f32_ref(current, block->norm2_scale, block->norm2_bias,
                          1e-5f, normal, rows, channels);
    bf_linear_f32_ref(normal, block->ffn1_weight, block->ffn1_bias,
                      hidden, rows, channels, 4 * channels);
    bf_gelu_f32_ref(hidden, hidden, rows * 4 * channels);
    bf_linear_f32_ref(hidden, block->ffn2_weight, block->ffn2_bias,
                      normal, rows, 4 * channels, channels);
    for (size_t i = 0; i < count; ++i) current[i] += normal[i];
    return 1;
}

static void store_output(const float *tokens, float *output,
                         const float *scale, const float *bias,
                         float *normal, size_t batches, size_t height,
                         size_t width, size_t channels) {
    size_t rows = batches * height * width, spatial = height * width;
    bf_layer_norm_f32_ref(tokens, scale, bias, 1e-5f, normal, rows, channels);
    for (size_t b = 0; b < batches; ++b)
        for (size_t p = 0; p < spatial; ++p)
            for (size_t c = 0; c < channels; ++c)
                output[(b * channels + c) * spatial + p] =
                    normal[(b * spatial + p) * channels + c];
}

static void patch_merge(const bf_bound_patch_merge *merge, const float *input,
                        float *unfolded, float *output, size_t batches,
                        size_t height, size_t width, size_t channels) {
    size_t out_h = (height + 1) / 2, out_w = (width + 1) / 2;
    size_t rows = batches * out_h * out_w;
    for (size_t b = 0; b < batches; ++b)
        for (size_t oy = 0; oy < out_h; ++oy)
            for (size_t ox = 0; ox < out_w; ++ox) {
                float *row = unfolded + ((b * out_h + oy) * out_w + ox) * 4 * channels;
                /* nn.Unfold order: channel, then (ky,kx). */
                for (size_t c = 0; c < channels; ++c)
                    for (size_t ky = 0; ky < 2; ++ky)
                        for (size_t kx = 0; kx < 2; ++kx) {
                            size_t y = 2 * oy + ky, x = 2 * ox + kx;
                            row[c * 4 + ky * 2 + kx] =
                                y < height && x < width
                                ? input[((b * height + y) * width + x) * channels + c]
                                : 0.0f;
                        }
            }
    bf_layer_norm_f32_ref(unfolded, merge->norm_scale, merge->norm_bias,
                          1e-5f, unfolded, rows, 4 * channels);
    bf_linear_f32_ref(unfolded, merge->reduction_weight, NULL, output,
                      rows, 4 * channels, 2 * channels);
}

int bf_swin_backbone_forward_ref(const bf_swin_backbone *net,
                                 const float *input, size_t batches,
                                 size_t input_h, size_t input_w,
                                 float *out1, float *out2, float *out3,
                                 void *workspace, size_t workspace_bytes,
                                 char *error, size_t cap) {
    size_t required = bf_swin_backbone_workspace_bytes(batches, input_h, input_w);
    if (!net || !input || !batches || !input_h || !input_w || !out1 || !out2 ||
        !out3 || !workspace || !required || workspace_bytes < required)
        return fail(error, cap, "invalid Swin backbone buffers or dimensions");
    size_t h = (input_h + 3) / 4, w = (input_w + 3) / 4, capacity;
    if (!checked_mul(h, w, &capacity) || !checked_mul(batches, capacity, &capacity) ||
        !checked_mul(capacity, 96, &capacity))
        return fail(error, cap, "Swin backbone shape overflow");
    float *buffer_a = (float *)workspace;
    float *buffer_b = buffer_a + capacity;
    float *hidden = buffer_b + capacity;
    void *attention_workspace = buffer_a + 6 * capacity;
    size_t attention_bytes = required - 6 * capacity * sizeof(float);
    patch_embed(net, input, buffer_a, batches, input_h, input_w, h, w);
    bf_layer_norm_f32_ref(buffer_a, net->patch_norm_scale, net->patch_norm_bias,
                          1e-5f, buffer_a, batches * h * w, 96);
    float *current = buffer_a, *other = buffer_b;
    float *outputs[3] = {out1, out2, out3};
    for (size_t stage = 0; stage < 4; ++stage) {
        size_t channels = stage_channels[stage];
        for (size_t block = 0; block < stage_depths[stage]; ++block)
            if (!run_block(&net->blocks[stage][block], current, other, hidden,
                           batches, h, w, channels, stage_heads[stage],
                           block & 1 ? 3 : 0,
                           attention_workspace, attention_bytes))
                return fail(error, cap, "Swin stage %zu block %zu failed", stage, block);
        if (stage > 0)
            store_output(current, outputs[stage - 1],
                         net->output_norm_scale[stage - 1],
                         net->output_norm_bias[stage - 1], other,
                         batches, h, w, channels);
        if (stage < 3) {
            size_t next_h = (h + 1) / 2, next_w = (w + 1) / 2;
            patch_merge(&net->merging[stage], current, hidden, other,
                        batches, h, w, channels);
            current = other;
            other = current == buffer_a ? buffer_b : buffer_a;
            h = next_h;
            w = next_w;
        }
    }
    return 1;
}
