#include "bf_kernels.h"
#include "bf_model.h"
#include "bf_voxel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

static int close_array(const char *name, const float *actual,
                       const bf_tensor *expected, float atol, float rtol) {
    if (!expected || expected->dtype != BF_DTYPE_F32) return 0;
    size_t count = (size_t)(expected->nbytes / sizeof(float));
    const float *reference = (const float *)expected->data;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float difference = fabsf(actual[i] - reference[i]);
        float tolerance = atol + rtol * fabsf(reference[i]);
        if (difference > max_abs) max_abs = difference;
        if (!(difference <= tolerance)) {
            fprintf(stderr, "%s[%zu]: got %.9g expected %.9g diff %.9g tolerance %.9g\n",
                    name, i, actual[i], reference[i], difference, tolerance);
            return 0;
        }
    }
    printf("%-12s max_abs=%.3g\n", name, max_abs);
    return 1;
}

static float *allocate_like(const bf_tensor *value) {
    return value ? (float *)malloc((size_t)value->nbytes) : NULL;
}

static int test_conv(const bf_model *model) {
    const bf_tensor *input = tensor(model, "conv.input");
    const bf_tensor *weight = tensor(model, "conv.weight");
    const bf_tensor *bias = tensor(model, "conv.bias");
    const bf_tensor *expected = tensor(model, "conv.output");
    float *output = allocate_like(expected);
    bf_conv2d_desc desc = {2, 4, 5, 6, 6, 3, 3, 2, 2, 1, 1, 1, 1, 2};
    int ok = output && bf_conv2d_f32(input->data, weight->data, bias->data,
                                     output, &desc) &&
             close_array("conv2d", output, expected, 2e-5f, 2e-6f);
    free(output);
    return ok;
}

static int test_linear(const bf_model *model) {
    const bf_tensor *expected = tensor(model, "linear.output");
    float *output = allocate_like(expected);
    bf_linear_f32(tensor(model, "linear.input")->data,
                  tensor(model, "linear.weight")->data,
                  tensor(model, "linear.bias")->data, output, 7, 5, 4);
    int ok = output && close_array("linear", output, expected, 4e-6f, 2e-6f);
    free(output);
    return ok;
}

static int test_norms(const bf_model *model) {
    const bf_tensor *bn_expected = tensor(model, "bn.output");
    const bf_tensor *ln_expected = tensor(model, "layer_norm.output");
    float *bn = allocate_like(bn_expected);
    float *ln = allocate_like(ln_expected);
    bf_batch_norm_2d_f32(tensor(model, "bn.input")->data,
                             tensor(model, "bn.scale")->data,
                             tensor(model, "bn.bias")->data,
                             tensor(model, "bn.mean")->data,
                             tensor(model, "bn.variance")->data,
                             1e-3f, bn, 2, 3, 4, 5);
    bf_layer_norm_f32(tensor(model, "layer_norm.input")->data,
                          tensor(model, "layer_norm.scale")->data,
                          tensor(model, "layer_norm.bias")->data,
                          1e-5f, ln, 5, 7);
    int ok = bn && ln && close_array("batch_norm", bn, bn_expected, 2e-6f, 2e-6f) &&
             close_array("layer_norm", ln, ln_expected, 2e-6f, 2e-6f);
    free(bn);
    free(ln);
    return ok;
}

static int test_activations(const bf_model *model) {
    const bf_tensor *softmax_expected = tensor(model, "softmax.output");
    const bf_tensor *gelu_expected = tensor(model, "gelu.output");
    const bf_tensor *relu_expected = tensor(model, "relu.output");
    float *softmax = allocate_like(softmax_expected);
    float *gelu = allocate_like(gelu_expected);
    float *relu = allocate_like(relu_expected);
    bf_softmax_f32_ref(tensor(model, "softmax.input")->data, softmax, 4, 9);
    const bf_tensor *activation_input = tensor(model, "gelu.input");
    bf_gelu_f32(activation_input->data, gelu, 65);
    bf_relu_f32(activation_input->data, relu, 65);
    int ok = softmax && gelu && relu &&
             close_array("softmax", softmax, softmax_expected, 2e-7f, 2e-6f) &&
             close_array("gelu", gelu, gelu_expected, 3e-7f, 3e-6f) &&
             close_array("relu", relu, relu_expected, 0.0f, 0.0f);
    free(softmax);
    free(gelu);
    free(relu);
    return ok;
}

static int test_vfe_topk(const bf_model *model) {
    const bf_tensor *vfe_expected = tensor(model, "vfe.output");
    const bf_tensor *topk_expected = tensor(model, "topk.values");
    float *vfe = allocate_like(vfe_expected);
    float *values = allocate_like(topk_expected);
    int64_t indices[15];
    bf_mean_vfe_f32_ref(tensor(model, "vfe.points")->data,
                        tensor(model, "vfe.counts")->data, vfe, 4, 6, 5);
    int ok = bf_topk_f32_ref(tensor(model, "topk.input")->data,
                             values, indices, 3, 17, 5) &&
             close_array("mean_vfe", vfe, vfe_expected, 3e-7f, 2e-6f) &&
             close_array("topk", values, topk_expected, 0.0f, 0.0f) &&
             memcmp(indices, tensor(model, "topk.indices")->data, sizeof(indices)) == 0;
    free(vfe);
    free(values);
    return ok;
}

static int test_invalid_shapes(void) {
    bf_conv2d_desc desc = {1, 3, 2, 2, 4, 5, 5, 1, 1, 0, 0, 1, 1, 1};
    bf_sparse_conv3d_desc sparse = {
        1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1
    };
    bf_coord4 duplicate[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    bf_coord4 sparse_output[2];
    float features[2] = {1.0f, 2.0f}, sparse_values[2], weight = 1.0f;
    size_t sparse_count;
    size_t h, w;
    float one = 1.0f;
    int64_t index;
    return !bf_conv2d_output_shape(&desc, &h, &w) &&
           !bf_topk_f32_ref(&one, &one, &index, 1, 1, 2) &&
           !bf_sparse_conv3d_f32_ref(duplicate, features, 2, &weight, NULL,
                                     sparse_output, sparse_values, 2,
                                     &sparse_count, &sparse);
}

static int coords_from_i64(const bf_tensor *source, bf_coord4 *destination, size_t count) {
    if (!source || source->dtype != BF_DTYPE_I64 || source->rank != 2 ||
        source->dims[0] != count || source->dims[1] != 4) return 0;
    const int64_t *values = (const int64_t *)source->data;
    for (size_t i = 0; i < count; ++i) {
        destination[i].batch = (int32_t)values[i * 4];
        destination[i].z = (int32_t)values[i * 4 + 1];
        destination[i].y = (int32_t)values[i * 4 + 2];
        destination[i].x = (int32_t)values[i * 4 + 3];
    }
    return 1;
}

static int compare_coords(const bf_coord4 *actual, const bf_tensor *expected, size_t count) {
    bf_coord4 *reference = (bf_coord4 *)malloc(count * sizeof(*reference));
    int ok = reference && coords_from_i64(expected, reference, count) &&
             memcmp(actual, reference, count * sizeof(*actual)) == 0;
    if (!ok) fprintf(stderr, "sparse output coordinates differ\n");
    free(reference);
    return ok;
}

static int test_sparse(const bf_model *model) {
    bf_coord4 input_coords[8], output_coords[128];
    size_t output_count = 0;
    const bf_tensor *features = tensor(model, "sparse.features");
    const bf_tensor *subm_expected = tensor(model, "sparse.subm.output");
    const bf_tensor *stride_expected = tensor(model, "sparse.stride.output");
    float *output = (float *)malloc(128 * 5 * sizeof(*output));
    if (!output || !coords_from_i64(tensor(model, "sparse.coords"), input_coords, 8)) {
        free(output);
        return 0;
    }
    bf_sparse_conv3d_desc subm = {
        1, 4, 5, 6, 3, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    int ok = bf_sparse_conv3d_f32_ref(
        input_coords, features->data, 8, tensor(model, "sparse.subm.weight")->data,
        tensor(model, "sparse.subm.bias")->data, output_coords, output, 128,
        &output_count, &subm) && output_count == 8 &&
        compare_coords(output_coords, tensor(model, "sparse.coords"), output_count) &&
        close_array("sparse_subm", output, subm_expected, 3e-6f, 2e-6f);
    bf_sparse_conv3d_desc stride = {
        1, 4, 5, 6, 3, 5, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0
    };
    ok = ok && bf_sparse_conv3d_f32_ref(
        input_coords, features->data, 8, tensor(model, "sparse.stride.weight")->data,
        tensor(model, "sparse.stride.bias")->data, output_coords, output, 128,
        &output_count, &stride) && output_count == stride_expected->dims[0] &&
        compare_coords(output_coords, tensor(model, "sparse.stride.coords"), output_count) &&
        close_array("sparse_stride", output, stride_expected, 3e-6f, 2e-6f);
    free(output);
    return ok;
}

static int test_voxel(const bf_model *model) {
    bf_voxel_config config = {
        {-2.0f, -2.0f, -1.0f}, {2.0f, 2.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
        5, 3, 5
    };
    float values[5 * 3 * 5];
    bf_coord4 coords[5], expected_coords[5];
    int64_t counts[5];
    size_t voxel_count;
    bf_voxel_stats stats;
    const bf_tensor *points = tensor(model, "voxel.points");
    int ok = bf_voxelize_f32_ref(points->data, points->dims[0], 5, 0, &config,
                                 values, coords, counts, &voxel_count, &stats) &&
             voxel_count == 5 &&
             coords_from_i64(tensor(model, "voxel.coords"), expected_coords, 5) &&
             memcmp(coords, expected_coords, sizeof(coords)) == 0 &&
             memcmp(counts, tensor(model, "voxel.counts")->data, sizeof(counts)) == 0 &&
             memcmp(&stats, tensor(model, "voxel.stats")->data, sizeof(stats)) == 0 &&
             close_array("voxelize", values, tensor(model, "voxel.values"), 0.0f, 0.0f);
    size_t grid[3];
    ok = ok && bf_voxel_grid_shape(&config, grid) &&
         grid[0] == 4 && grid[1] == 4 && grid[2] == 2;
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[256];
    bf_model *model = NULL;
    if (!bf_model_open(argv[1], &model, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 3;
    }
    int ok = test_conv(model) && test_linear(model) && test_norms(model) &&
             test_activations(model) && test_vfe_topk(model) && test_sparse(model) &&
             test_voxel(model) &&
             test_invalid_shapes();
    bf_model_close(model);
    if (!ok) return 4;
    puts("scalar kernels match deterministic PyTorch oracle");
    return 0;
}
