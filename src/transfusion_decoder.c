#include "bf_transfusion_decoder.h"

#include "bf_kernels.h"
#include "bf_transfusion.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define C BF_TRANSFUSION_CHANNELS
#define HEADS 8u
#define HEAD_C (C / HEADS)

typedef struct {
    const float *scale, *bias, *mean, *variance;
} bf_bound_bn1d;

typedef struct {
    const float *first_weight, *first_bias;
    bf_bound_bn1d bn;
    const float *second_weight, *second_bias;
} bf_bound_position;

typedef struct {
    const float *hidden_weight;
    bf_bound_bn1d bn;
    const float *output_weight, *output_bias;
    size_t output_channels;
} bf_bound_head;

struct bf_transfusion_decoder {
    const float *class_weight, *class_bias;
    const float *self_in_weight, *self_in_bias;
    const float *self_out_weight, *self_out_bias;
    const float *cross_in_weight, *cross_in_bias;
    const float *cross_out_weight, *cross_out_bias;
    const float *linear1_weight, *linear1_bias;
    const float *linear2_weight, *linear2_bias;
    const float *norm_scale[3], *norm_bias[3];
    bf_bound_position self_position, cross_position;
    bf_bound_head heads[6];
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
    if (!tensor) { fail(error, cap, "missing TransFusion tensor %s", name); return NULL; }
    if (tensor->dtype != BF_DTYPE_F32 || tensor->rank != rank) {
        fail(error, cap, "%s: dtype/rank mismatch", name); return NULL;
    }
    for (uint32_t axis = 0; axis < rank; ++axis)
        if (tensor->dims[axis] != dims[axis]) {
            fail(error, cap, "%s: dimension %u is %u, expected %u",
                 name, axis, tensor->dims[axis], dims[axis]); return NULL;
        }
    return (const float *)tensor->data;
}

static int bind_bn(const bf_model *model, const char *prefix, size_t channels,
                   bf_bound_bn1d *bn, char *error, size_t cap) {
    const char *suffixes[4] = {"weight", "bias", "running_mean", "running_var"};
    const float **destinations[4] = {&bn->scale, &bn->bias, &bn->mean, &bn->variance};
    uint32_t dims[1] = {(uint32_t)channels};
    for (size_t i = 0; i < 4; ++i) {
        char name[192];
        int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[i]);
        if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "TransFusion name overflow");
        *destinations[i] = bind_f32(model, name, 1, dims, error, cap);
        if (!*destinations[i]) return 0;
    }
    return 1;
}

static int bind_position(const bf_model *model, const char *base,
                         bf_bound_position *position, char *error, size_t cap) {
    char name[192], prefix[192];
    uint32_t first[3] = {C, 2, 1}, cdim[1] = {C}, second[3] = {C, C, 1};
#define POS(field, suffix, rank, dims) do { \
    int n = snprintf(name, sizeof(name), "%s.position_embedding_head.%s", base, suffix); \
    if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "position name overflow"); \
    position->field = bind_f32(model, name, rank, dims, error, cap); \
    if (!position->field) return 0; \
} while (0)
    POS(first_weight, "0.weight", 3, first);
    POS(first_bias, "0.bias", 1, cdim);
    int n = snprintf(prefix, sizeof(prefix), "%s.position_embedding_head.1", base);
    if (n < 0 || (size_t)n >= sizeof(prefix) ||
        !bind_bn(model, prefix, C, &position->bn, error, cap)) return 0;
    POS(second_weight, "3.weight", 3, second);
    POS(second_bias, "3.bias", 1, cdim);
#undef POS
    return 1;
}

static int bind_head(const bf_model *model, const char *name,
                     size_t output_channels, bf_bound_head *head,
                     char *error, size_t cap) {
    char tensor_name[192], prefix[192];
    uint32_t hidden[3] = {64, C, 1}, output[3] = {(uint32_t)output_channels, 64, 1};
    uint32_t output_bias[1] = {(uint32_t)output_channels};
    int n = snprintf(tensor_name, sizeof(tensor_name),
                     "dense_head.prediction_head.%s.0.0.weight", name);
    if (n < 0 || (size_t)n >= sizeof(tensor_name)) return fail(error, cap, "head name overflow");
    head->hidden_weight = bind_f32(model, tensor_name, 3, hidden, error, cap);
    n = snprintf(prefix, sizeof(prefix), "dense_head.prediction_head.%s.0.1", name);
    if (n < 0 || (size_t)n >= sizeof(prefix) ||
        !head->hidden_weight || !bind_bn(model, prefix, 64, &head->bn, error, cap)) return 0;
    n = snprintf(tensor_name, sizeof(tensor_name),
                 "dense_head.prediction_head.%s.1.weight", name);
    if (n < 0 || (size_t)n >= sizeof(tensor_name)) return fail(error, cap, "head name overflow");
    head->output_weight = bind_f32(model, tensor_name, 3, output, error, cap);
    n = snprintf(tensor_name, sizeof(tensor_name),
                 "dense_head.prediction_head.%s.1.bias", name);
    if (n < 0 || (size_t)n >= sizeof(tensor_name)) return fail(error, cap, "head name overflow");
    head->output_bias = bind_f32(model, tensor_name, 1, output_bias, error, cap);
    head->output_channels = output_channels;
    return head->output_weight && head->output_bias;
}

int bf_transfusion_decoder_create(const bf_model *model,
                                  bf_transfusion_decoder **out,
                                  char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid TransFusion arguments");
    bf_transfusion_decoder *net = (bf_transfusion_decoder *)calloc(1, sizeof(*net));
    if (!net) return fail(error, cap, "TransFusion allocation failed");
    uint32_t class_w[3] = {C, 10, 1}, cdim[1] = {C};
    uint32_t in_w[2] = {3 * C, C}, in_b[1] = {3 * C}, square[2] = {C, C};
    uint32_t ffn1[2] = {256, C}, ffn1b[1] = {256}, ffn2[2] = {C, 256};
#define BIND(field, name, rank, dims) do { \
    net->field = bind_f32(model, name, rank, dims, error, cap); \
    if (!net->field) goto failure; \
} while (0)
    BIND(class_weight, "dense_head.class_encoding.weight", 3, class_w);
    BIND(class_bias, "dense_head.class_encoding.bias", 1, cdim);
    BIND(self_in_weight, "dense_head.decoder.self_attn.in_proj_weight", 2, in_w);
    BIND(self_in_bias, "dense_head.decoder.self_attn.in_proj_bias", 1, in_b);
    BIND(self_out_weight, "dense_head.decoder.self_attn.out_proj.weight", 2, square);
    BIND(self_out_bias, "dense_head.decoder.self_attn.out_proj.bias", 1, cdim);
    BIND(cross_in_weight, "dense_head.decoder.multihead_attn.in_proj_weight", 2, in_w);
    BIND(cross_in_bias, "dense_head.decoder.multihead_attn.in_proj_bias", 1, in_b);
    BIND(cross_out_weight, "dense_head.decoder.multihead_attn.out_proj.weight", 2, square);
    BIND(cross_out_bias, "dense_head.decoder.multihead_attn.out_proj.bias", 1, cdim);
    BIND(linear1_weight, "dense_head.decoder.linear1.weight", 2, ffn1);
    BIND(linear1_bias, "dense_head.decoder.linear1.bias", 1, ffn1b);
    BIND(linear2_weight, "dense_head.decoder.linear2.weight", 2, ffn2);
    BIND(linear2_bias, "dense_head.decoder.linear2.bias", 1, cdim);
    for (size_t i = 0; i < 3; ++i) {
        char name[96];
        snprintf(name, sizeof(name), "dense_head.decoder.norm%zu.weight", i + 1);
        net->norm_scale[i] = bind_f32(model, name, 1, cdim, error, cap);
        snprintf(name, sizeof(name), "dense_head.decoder.norm%zu.bias", i + 1);
        net->norm_bias[i] = bind_f32(model, name, 1, cdim, error, cap);
        if (!net->norm_scale[i] || !net->norm_bias[i]) goto failure;
    }
#undef BIND
    if (!bind_position(model, "dense_head.decoder.self_posembed", &net->self_position, error, cap) ||
        !bind_position(model, "dense_head.decoder.cross_posembed", &net->cross_position, error, cap)) goto failure;
    const char *head_names[6] = {"center", "height", "dim", "rot", "vel", "heatmap"};
    const size_t head_channels[6] = {2, 1, 3, 2, 2, 10};
    for (size_t i = 0; i < 6; ++i)
        if (!bind_head(model, head_names[i], head_channels[i], &net->heads[i], error, cap)) goto failure;
    *out = net;
    return 1;
failure:
    free(net);
    return 0;
}

void bf_transfusion_decoder_destroy(bf_transfusion_decoder *decoder) { free(decoder); }

size_t bf_transfusion_decoder_workspace_bytes(size_t height, size_t width,
                                               size_t proposals) {
    size_t keys, key_floats, proposal_floats, floats, bytes;
    if (!height || !width || !proposals || proposals > 200 ||
        !checked_mul(height, width, &keys) || proposals > 10 * keys ||
        !checked_mul(keys, 3 * C + 11, &key_floats) ||
        !checked_mul(proposals, 8 * C + 1, &proposal_floats) ||
        key_floats > SIZE_MAX - proposal_floats) return 0;
    floats = key_floats + proposal_floats;
    if (!checked_mul(floats, sizeof(float), &bytes)) return 0;
    return bytes;
}

static void bn_relu_rows(float *values, size_t rows, size_t channels,
                         const bf_bound_bn1d *bn) {
    for (size_t row = 0; row < rows; ++row)
        for (size_t c = 0; c < channels; ++c) {
            float factor = bn->scale[c] / sqrtf(bn->variance[c] + 1e-5f);
            float value = (values[row * channels + c] - bn->mean[c]) * factor + bn->bias[c];
            values[row * channels + c] = value > 0.0f ? value : 0.0f;
        }
}

static void position_embedding(const bf_bound_position *position,
                               const int64_t *indices, size_t count,
                               size_t width, float *hidden, float *output) {
    for (size_t row = 0; row < count; ++row) {
        float x = indices ? (float)((size_t)indices[row] % width) + 0.5f
                          : (float)(row % width) + 0.5f;
        float y = indices ? (float)((size_t)indices[row] / width) + 0.5f
                          : (float)(row / width) + 0.5f;
        for (size_t c = 0; c < C; ++c)
            hidden[row * C + c] = position->first_bias[c] +
                position->first_weight[c * 2] * x + position->first_weight[c * 2 + 1] * y;
    }
    bn_relu_rows(hidden, count, C, &position->bn);
    bf_linear_f32_ref(hidden, position->second_weight, position->second_bias,
                      output, count, C, C);
}

static void project_slice(const float *input, size_t rows,
                          const float *weight, const float *bias,
                          size_t slice, float *output) {
    bf_linear_f32_ref(input, weight + slice * C * C, bias + slice * C,
                      output, rows, C, C);
}

static int multihead_attention(const float *query, size_t query_count,
                               const float *key_value, size_t key_count,
                               const float *in_weight, const float *in_bias,
                               const float *out_weight, const float *out_bias,
                               float *q_projection, float *k_projection,
                               float *v_projection, float *scores,
                               float *head_output, float *output) {
    if (!query_count || !key_count) return 0;
    project_slice(query, query_count, in_weight, in_bias, 0, q_projection);
    project_slice(key_value, key_count, in_weight, in_bias, 1, k_projection);
    project_slice(key_value, key_count, in_weight, in_bias, 2, v_projection);
    const float scale = 1.0f / sqrtf((float)HEAD_C);
    for (size_t q = 0; q < query_count; ++q) {
        for (size_t c = 0; c < C; ++c) head_output[q * C + c] = 0.0f;
        for (size_t head = 0; head < HEADS; ++head) {
            float maximum = -FLT_MAX;
            for (size_t k = 0; k < key_count; ++k) {
                float score = 0.0f;
                for (size_t lane = 0; lane < HEAD_C; ++lane) {
                    size_t c = head * HEAD_C + lane;
                    score += q_projection[q * C + c] * k_projection[k * C + c];
                }
                score *= scale;
                scores[k] = score;
                if (score > maximum) maximum = score;
            }
            double sum = 0.0;
            for (size_t k = 0; k < key_count; ++k) {
                scores[k] = expf(scores[k] - maximum);
                sum += scores[k];
            }
            float inverse = (float)(1.0 / sum);
            for (size_t k = 0; k < key_count; ++k) {
                float probability = scores[k] * inverse;
                for (size_t lane = 0; lane < HEAD_C; ++lane) {
                    size_t c = head * HEAD_C + lane;
                    head_output[q * C + c] += probability * v_projection[k * C + c];
                }
            }
        }
    }
    bf_linear_f32_ref(head_output, out_weight, out_bias, output,
                      query_count, C, C);
    return 1;
}

static void prediction_head(const bf_bound_head *head, const float *query,
                            size_t proposals, float *hidden, float *channel_major) {
    bf_linear_f32_ref(query, head->hidden_weight, NULL, hidden,
                      proposals, C, 64);
    bn_relu_rows(hidden, proposals, 64, &head->bn);
    /* Write row-major to the tail of hidden, then transpose to OpenPCDet [C,P]. */
    float *rows = hidden + proposals * 64;
    bf_linear_f32_ref(hidden, head->output_weight, head->output_bias, rows,
                      proposals, 64, head->output_channels);
    for (size_t p = 0; p < proposals; ++p)
        for (size_t c = 0; c < head->output_channels; ++c)
            channel_major[c * proposals + p] = rows[p * head->output_channels + c];
}

int bf_transfusion_decoder_forward_ref(
    const bf_transfusion_decoder *net, const float *shared,
    const float *dense_heatmap, size_t batches, size_t height, size_t width,
    size_t proposals, bf_transfusion_raw_outputs *out, void *workspace,
    size_t workspace_bytes, char *error, size_t cap) {
    size_t required = bf_transfusion_decoder_workspace_bytes(height, width, proposals);
    if (!net || !shared || !dense_heatmap || !batches || !out ||
        !out->center_b2p || !out->height_b1p || !out->dimension_log_b3p ||
        !out->rotation_sincos_b2p || !out->velocity_b2p ||
        !out->heatmap_logits_b10p || !out->query_heatmap_scores_b10p ||
        !out->query_labels_bp || !out->query_indices_bp || !workspace ||
        !required || workspace_bytes < required)
        return fail(error, cap, "invalid TransFusion buffers or dimensions");
    size_t keys = height * width;
    float *cursor = (float *)workspace;
    float *key_value = cursor; cursor += keys * C;
    float *key_projection = cursor; cursor += keys * C;
    float *value_projection = cursor; cursor += keys * C;
    float *suppressed = cursor; cursor += keys * 10;
    float *scores = cursor; cursor += keys;
    float *query = cursor; cursor += proposals * C;
    float *query_position = cursor; cursor += proposals * C;
    float *temporary = cursor; cursor += proposals * C;
    float *attention = cursor; cursor += proposals * C;
    float *query_projection = cursor; cursor += proposals * C;
    float *ffn = cursor; cursor += proposals * 256;
    float *head_hidden = cursor; cursor += proposals * 128;
    float *top_scores = cursor; cursor += proposals;
    for (size_t b = 0; b < batches; ++b) {
        const float *batch_shared = shared + b * C * keys;
        const float *batch_heatmap = dense_heatmap + b * 10 * keys;
        int64_t *labels = out->query_labels_bp + b * proposals;
        int64_t *indices = out->query_indices_bp + b * proposals;
        if (!bf_transfusion_select_proposals_f32_ref(
                batch_heatmap, suppressed, top_scores, labels,
                indices, 1, 10, height, width, proposals, 3))
            return fail(error, cap, "TransFusion proposal selection failed");
        for (size_t p = 0; p < proposals; ++p) {
            size_t index = (size_t)indices[p], class_id = (size_t)labels[p];
            for (size_t c = 0; c < C; ++c)
                query[p * C + c] = batch_shared[c * keys + index] +
                    net->class_bias[c] + net->class_weight[c * 10 + class_id];
        }
        position_embedding(&net->self_position, indices, proposals, width,
                           temporary, query_position);
        position_embedding(&net->cross_position, NULL, keys, width,
                           key_projection, key_value);
        for (size_t k = 0; k < keys; ++k)
            for (size_t c = 0; c < C; ++c)
                key_value[k * C + c] += batch_shared[c * keys + k];
        for (size_t i = 0; i < proposals * C; ++i) temporary[i] = query[i] + query_position[i];
        if (!multihead_attention(temporary, proposals, temporary, proposals,
                net->self_in_weight, net->self_in_bias,
                net->self_out_weight, net->self_out_bias,
                query_projection, key_projection, value_projection, scores,
                attention, temporary)) return fail(error, cap, "TransFusion self attention failed");
        for (size_t i = 0; i < proposals * C; ++i) query[i] += temporary[i];
        bf_layer_norm_f32_ref(query, net->norm_scale[0], net->norm_bias[0],
                              1e-5f, query, proposals, C);
        for (size_t i = 0; i < proposals * C; ++i) temporary[i] = query[i] + query_position[i];
        if (!multihead_attention(temporary, proposals, key_value, keys,
                net->cross_in_weight, net->cross_in_bias,
                net->cross_out_weight, net->cross_out_bias,
                query_projection, key_projection, value_projection, scores,
                attention, temporary)) return fail(error, cap, "TransFusion cross attention failed");
        for (size_t i = 0; i < proposals * C; ++i) query[i] += temporary[i];
        bf_layer_norm_f32_ref(query, net->norm_scale[1], net->norm_bias[1],
                              1e-5f, query, proposals, C);
        bf_linear_f32_ref(query, net->linear1_weight, net->linear1_bias,
                          ffn, proposals, C, 256);
        bf_relu_f32_ref(ffn, ffn, proposals * 256);
        bf_linear_f32_ref(ffn, net->linear2_weight, net->linear2_bias,
                          temporary, proposals, 256, C);
        for (size_t i = 0; i < proposals * C; ++i) query[i] += temporary[i];
        bf_layer_norm_f32_ref(query, net->norm_scale[2], net->norm_bias[2],
                              1e-5f, query, proposals, C);
        float *head_outputs[6] = {
            out->center_b2p + b * 2 * proposals,
            out->height_b1p + b * proposals,
            out->dimension_log_b3p + b * 3 * proposals,
            out->rotation_sincos_b2p + b * 2 * proposals,
            out->velocity_b2p + b * 2 * proposals,
            out->heatmap_logits_b10p + b * 10 * proposals
        };
        for (size_t head = 0; head < 6; ++head)
            prediction_head(&net->heads[head], query, proposals, head_hidden,
                            head_outputs[head]);
        for (size_t p = 0; p < proposals; ++p) {
            size_t index = (size_t)indices[p];
            out->center_b2p[(b * 2) * proposals + p] +=
                (float)(index % width) + 0.5f;
            out->center_b2p[(b * 2 + 1) * proposals + p] +=
                (float)(index / width) + 0.5f;
            for (size_t c = 0; c < 10; ++c)
                out->query_heatmap_scores_b10p[(b * 10 + c) * proposals + p] =
                    suppressed[c * keys + index];
        }
    }
    return 1;
}
