#include "bf_lidar_backbone.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *scale, *bias, *mean, *variance;
} bf_sparse_bn;

typedef struct {
    const float *weight;
    bf_sparse_bn bn;
    size_t ci, co, kd, kh, kw;
    size_t sd, sh, sw, pd, ph, pw;
    int submanifold;
} bf_sparse_layer;

struct bf_lidar_backbone {
    bf_sparse_layer input;
    bf_sparse_layer residual[4][2][2];
    bf_sparse_layer down[3];
    bf_sparse_layer output;
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
    if (!tensor) { fail(error, cap, "missing lidar backbone tensor %s", name); return NULL; }
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
                   bf_sparse_bn *bn, char *error, size_t cap) {
    const char *suffixes[4] = {"weight", "bias", "running_mean", "running_var"};
    const float **destinations[4] = {&bn->scale, &bn->bias, &bn->mean, &bn->variance};
    uint32_t dims[1] = {(uint32_t)channels};
    for (size_t i = 0; i < 4; ++i) {
        char name[160];
        int n = snprintf(name, sizeof(name), "%s.%s", prefix, suffixes[i]);
        if (n < 0 || (size_t)n >= sizeof(name)) return fail(error, cap, "lidar tensor name overflow");
        *destinations[i] = bind_f32(model, name, 1, dims, error, cap);
        if (!*destinations[i]) return 0;
    }
    return 1;
}

static int bind_layer(const bf_model *model, bf_sparse_layer *layer,
                      const char *weight_name, const char *bn_prefix,
                      size_t ci, size_t co, size_t kd, size_t kh, size_t kw,
                      size_t sd, size_t sh, size_t sw,
                      size_t pd, size_t ph, size_t pw, int submanifold,
                      char *error, size_t cap) {
    uint32_t dims[5] = {(uint32_t)kd, (uint32_t)kh, (uint32_t)kw,
                        (uint32_t)ci, (uint32_t)co};
    layer->weight = bind_f32(model, weight_name, 5, dims, error, cap);
    if (!layer->weight || !bind_bn(model, bn_prefix, co, &layer->bn, error, cap)) return 0;
    layer->ci = ci; layer->co = co; layer->kd = kd; layer->kh = kh; layer->kw = kw;
    layer->sd = sd; layer->sh = sh; layer->sw = sw;
    layer->pd = pd; layer->ph = ph; layer->pw = pw; layer->submanifold = submanifold;
    return 1;
}

int bf_lidar_backbone_create(const bf_model *model, bf_lidar_backbone **out,
                             char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model || !out) return fail(error, cap, "invalid lidar backbone arguments");
    bf_lidar_backbone *net = (bf_lidar_backbone *)calloc(1, sizeof(*net));
    if (!net) return fail(error, cap, "lidar backbone allocation failed");
#define LAYER(dst, weight, bn, ci, co, kd, kh, kw, sd, sh, sw, pd, ph, pw, subm) \
    if (!bind_layer(model, dst, weight, bn, ci, co, kd, kh, kw, sd, sh, sw, pd, ph, pw, subm, error, cap)) goto failure
    LAYER(&net->input, "backbone_3d.conv_input.0.weight", "backbone_3d.conv_input.1",
          5, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1);
    const size_t channels[4] = {16, 32, 64, 128};
    for (size_t stage = 0; stage < 4; ++stage)
        for (size_t block = 0; block < 2; ++block)
            for (size_t convolution = 0; convolution < 2; ++convolution) {
                char weight[160], bn[160];
                snprintf(weight, sizeof(weight), "backbone_3d.conv%zu.%zu.conv%zu.weight",
                         stage + 1, block + (stage ? 1 : 0), convolution + 1);
                snprintf(bn, sizeof(bn), "backbone_3d.conv%zu.%zu.bn%zu",
                         stage + 1, block + (stage ? 1 : 0), convolution + 1);
                LAYER(&net->residual[stage][block][convolution], weight, bn,
                      channels[stage], channels[stage], 3, 3, 3,
                      1, 1, 1, 1, 1, 1, 1);
            }
    for (size_t transition = 0; transition < 3; ++transition) {
        size_t stage = transition + 2;
        char weight[128], bn[128];
        snprintf(weight, sizeof(weight), "backbone_3d.conv%zu.0.0.weight", stage);
        snprintf(bn, sizeof(bn), "backbone_3d.conv%zu.0.1", stage);
        size_t pad_z = stage == 4 ? 0 : 1;
        LAYER(&net->down[transition], weight, bn, channels[transition],
              channels[transition + 1], 3, 3, 3, 2, 2, 2,
              pad_z, 1, 1, 0);
    }
    LAYER(&net->output, "backbone_3d.conv_out.0.weight", "backbone_3d.conv_out.1",
          128, 128, 3, 1, 1, 2, 1, 1, 0, 0, 0, 0);
#undef LAYER
    *out = net;
    return 1;
failure:
    free(net);
    return 0;
}

void bf_lidar_backbone_destroy(bf_lidar_backbone *backbone) { free(backbone); }

size_t bf_lidar_backbone_workspace_bytes(size_t capacity) {
    size_t features, coords, total, sparse;
    if (!capacity || !checked_mul(capacity, 3 * 128 * sizeof(float), &features) ||
        !checked_mul(capacity, 2 * sizeof(bf_coord4), &coords) ||
        features > SIZE_MAX - coords) return 0;
    total = features + coords;
    sparse = bf_sparse_conv3d_workspace_bytes(capacity, capacity);
    return sparse && total <= SIZE_MAX - sparse ? total + sparse : 0;
}

int bf_lidar_backbone_output_shape(size_t d, size_t h, size_t w,
                                   size_t *od, size_t *oh, size_t *ow) {
    if (!d || !h || !w || !od || !oh || !ow) return 0;
    for (size_t stage = 0; stage < 3; ++stage) {
        size_t pad_z = stage == 2 ? 0 : 1;
        if (d + 2 * pad_z < 3 || h > SIZE_MAX - 2 || w > SIZE_MAX - 2) return 0;
        d = (d + 2 * pad_z - 3) / 2 + 1;
        h = (h + 2 - 3) / 2 + 1;
        w = (w + 2 - 3) / 2 + 1;
    }
    if (d < 3) return 0;
    *od = (d - 3) / 2 + 1; *oh = h; *ow = w;
    return *od && *oh && *ow;
}

static void bn_relu(float *features, size_t rows, size_t channels,
                    const bf_sparse_bn *bn, int relu) {
    for (size_t row = 0; row < rows; ++row)
        for (size_t c = 0; c < channels; ++c) {
            float factor = bn->scale[c] / sqrtf(bn->variance[c] + 1e-3f);
            float value = (features[row * channels + c] - bn->mean[c]) * factor + bn->bias[c];
            features[row * channels + c] = relu && value < 0.0f ? 0.0f : value;
        }
}

static int run_layer(const bf_sparse_layer *layer,
                     const bf_coord4 *in_coords, const float *in_features,
                     size_t in_count, bf_coord4 *out_coords, float *out_features,
                     size_t capacity, size_t batches, size_t d, size_t h, size_t w,
                     size_t *out_count, size_t *od, size_t *oh, size_t *ow,
                     int relu, void *sparse_workspace, size_t sparse_bytes) {
    bf_sparse_conv3d_desc desc = {batches, d, h, w, layer->ci, layer->co,
        layer->kd, layer->kh, layer->kw, layer->sd, layer->sh, layer->sw,
        layer->pd, layer->ph, layer->pw, 1, 1, 1, layer->submanifold};
    if (!bf_sparse_conv3d_output_shape(&desc, od, oh, ow) ||
        !bf_sparse_conv3d_f32_workspace_ref(in_coords, in_features, in_count,
                                  layer->weight, NULL, out_coords, out_features,
                                  capacity, out_count, &desc,
                                  sparse_workspace, sparse_bytes)) return 0;
    bn_relu(out_features, *out_count, layer->co, &layer->bn, relu);
    return 1;
}

int bf_lidar_backbone_forward_ref(const bf_lidar_backbone *net,
                                  const bf_coord4 *voxel_coords,
                                  const float *voxel_features, size_t voxel_count,
                                  size_t batches, size_t d, size_t h, size_t w,
                                  size_t capacity, float *dense, void *workspace,
                                  size_t workspace_bytes, char *error, size_t cap) {
    size_t required = bf_lidar_backbone_workspace_bytes(capacity), od, oh, ow;
    if (!net || !voxel_coords || !voxel_features || !voxel_count ||
        voxel_count > capacity || !batches || !d || !h || !w || !dense ||
        !workspace || !required || workspace_bytes < required ||
        !bf_lidar_backbone_output_shape(d, h, w, &od, &oh, &ow))
        return fail(error, cap, "invalid lidar backbone buffers or dimensions");
    bf_coord4 *coords[2];
    coords[0] = (bf_coord4 *)workspace;
    coords[1] = coords[0] + capacity;
    float *feature_base = (float *)(coords[1] + capacity);
    float *features[3] = {feature_base, feature_base + capacity * 128,
                          feature_base + capacity * 256};
    void *sparse_workspace = feature_base + capacity * 384;
    size_t sparse_bytes = required - (size_t)((unsigned char *)sparse_workspace -
                                               (unsigned char *)workspace);
    const bf_coord4 *current_coords = voxel_coords;
    const float *current_features = voxel_features;
    size_t current_count = voxel_count, coord_slot = 1, feature_slot = 0;
    size_t nd, nh, nw;
    if (!run_layer(&net->input, current_coords, current_features, current_count,
                   coords[0], features[0], capacity, batches, d, h, w,
                   &current_count, &nd, &nh, &nw, 1, sparse_workspace, sparse_bytes))
        return fail(error, cap, "lidar input sparse convolution failed");
    current_coords = coords[0]; current_features = features[0]; coord_slot = 0;
    for (size_t stage = 0; stage < 4; ++stage) {
        if (stage) {
            size_t next_coord = coord_slot ^ 1, next_feature = (feature_slot + 1) % 3;
            if (!run_layer(&net->down[stage - 1], current_coords, current_features,
                           current_count, coords[next_coord], features[next_feature],
                           capacity, batches, d, h, w, &current_count,
                           &nd, &nh, &nw, 1, sparse_workspace, sparse_bytes))
                return fail(error, cap, "lidar stage %zu downsample failed", stage);
            d = nd; h = nh; w = nw; coord_slot = next_coord; feature_slot = next_feature;
            current_coords = coords[coord_slot]; current_features = features[feature_slot];
        }
        for (size_t block = 0; block < 2; ++block) {
            size_t first_coord = coord_slot ^ 1;
            size_t first_feature = (feature_slot + 1) % 3;
            size_t second_feature = (feature_slot + 2) % 3;
            size_t count1, count2;
            if (!run_layer(&net->residual[stage][block][0], current_coords,
                           current_features, current_count, coords[first_coord],
                           features[first_feature], capacity, batches, d, h, w,
                           &count1, &nd, &nh, &nw, 1, sparse_workspace, sparse_bytes) ||
                !run_layer(&net->residual[stage][block][1], coords[first_coord],
                           features[first_feature], count1, coords[coord_slot],
                           features[second_feature], capacity, batches, d, h, w,
                           &count2, &nd, &nh, &nw, 0, sparse_workspace, sparse_bytes) || count2 != current_count)
                return fail(error, cap, "lidar stage %zu residual block %zu failed", stage, block);
            for (size_t i = 0; i < current_count * net->residual[stage][block][1].co; ++i) {
                float value = features[second_feature][i] + current_features[i];
                features[second_feature][i] = value > 0.0f ? value : 0.0f;
            }
            feature_slot = second_feature;
            current_coords = coords[coord_slot]; current_features = features[feature_slot];
        }
    }
    size_t final_coord = coord_slot ^ 1, final_feature = (feature_slot + 1) % 3;
    size_t final_count;
    if (!run_layer(&net->output, current_coords, current_features, current_count,
                   coords[final_coord], features[final_feature], capacity,
                   batches, d, h, w, &final_count, &nd, &nh, &nw, 1,
                   sparse_workspace, sparse_bytes))
        return fail(error, cap, "lidar output sparse convolution failed");
    size_t dense_count;
    if (!checked_mul(batches, 128, &dense_count) || !checked_mul(dense_count, nd, &dense_count) ||
        !checked_mul(dense_count, nh, &dense_count) || !checked_mul(dense_count, nw, &dense_count))
        return fail(error, cap, "lidar dense output overflow");
    memset(dense, 0, dense_count * sizeof(float));
    size_t spatial = nh * nw;
    for (size_t i = 0; i < final_count; ++i) {
        bf_coord4 c = coords[final_coord][i];
        for (size_t channel = 0; channel < 128; ++channel)
            dense[(((size_t)c.batch * 128 + channel) * nd + (size_t)c.z) * spatial +
                  (size_t)c.y * nw + (size_t)c.x] = features[final_feature][i * 128 + channel];
    }
    return 1;
}
