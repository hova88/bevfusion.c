#include "bf_lidar_backbone.h"
#include "bf_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[256] = {0};
    bf_model *weights = NULL, *oracle = NULL;
    if (!bf_model_open(argv[1], &weights, error, sizeof(error)) ||
        !bf_model_open(argv[2], &oracle, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle); return 3;
    }
    bf_lidar_backbone *backbone = NULL;
    if (!bf_lidar_backbone_create(weights, &backbone, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle); return 4;
    }
    size_t od, oh, ow;
    int ok = bf_lidar_backbone_output_shape(41, 8, 8, &od, &oh, &ow) &&
             od == 2 && oh == 1 && ow == 1;
    const size_t capacity = 3000;
    size_t workspace_bytes = bf_lidar_backbone_workspace_bytes(capacity);
    void *workspace = malloc(workspace_bytes);
    float *output = (float *)malloc(256 * sizeof(float));
    const bf_tensor *coords = tensor(oracle, "lidar_backbone.coords");
    bf_coord4 input_coords[8];
    const int64_t *raw_coords = coords ? (const int64_t *)coords->data : NULL;
    for (size_t i = 0; i < 8 && raw_coords; ++i) {
        input_coords[i].batch = (int32_t)raw_coords[i * 4];
        input_coords[i].z = (int32_t)raw_coords[i * 4 + 1];
        input_coords[i].y = (int32_t)raw_coords[i * 4 + 2];
        input_coords[i].x = (int32_t)raw_coords[i * 4 + 3];
    }
    ok = ok && workspace && output &&
        bf_lidar_backbone_forward_ref(
            backbone, input_coords, tensor(oracle, "lidar_backbone.features")->data,
            8, 1, 41, 8, 8, capacity, output, workspace, workspace_bytes,
            error, sizeof(error));
    const bf_tensor *expected = tensor(oracle, "lidar_backbone.output");
    if (ok) {
        const float *reference = (const float *)expected->data;
        float max_absolute = 0.0f;
        double mean_absolute = 0.0;
        for (size_t i = 0; i < 256; ++i) {
            float difference = fabsf(output[i] - reference[i]);
            if (difference > max_absolute) max_absolute = difference;
            mean_absolute += difference;
            if (!(difference <= 5e-5f + 5e-5f * fabsf(reference[i]))) {
                fprintf(stderr, "lidar output[%zu]: got %.9g expected %.9g diff %.9g\n",
                        i, output[i], reference[i], difference);
                ok = 0; break;
            }
        }
        printf("%-22s max_abs=%.3g mean_abs=%.3g\n", "lidar_backbone.output",
               max_absolute, mean_absolute / 256.0);
    }
    ok = ok && !bf_lidar_backbone_forward_ref(
        backbone, input_coords, tensor(oracle, "lidar_backbone.features")->data,
        8, 1, 41, 8, 8, capacity, output, workspace, workspace_bytes - 1,
        error, sizeof(error));
    if (!ok) fprintf(stderr, "lidar backbone failure: %s\n", error);
    free(workspace); free(output);
    bf_lidar_backbone_destroy(backbone);
    bf_model_close(weights); bf_model_close(oracle);
    if (!ok) return 5;
    puts("real-checkpoint VoxelResBackBone8x matches dense PyTorch oracle");
    return 0;
}
