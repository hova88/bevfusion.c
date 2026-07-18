#include "bf_lss_downsample.h"
#include "bf_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

static int close_array(const float *actual, const bf_tensor *expected) {
    if (!expected || expected->dtype != BF_DTYPE_F32) return 0;
    const float *reference = (const float *)expected->data;
    size_t count = (size_t)(expected->nbytes / sizeof(float));
    float max_absolute = 0.0f;
    double mean_absolute = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float difference = fabsf(actual[i] - reference[i]);
        float tolerance = 4e-5f + 4e-5f * fabsf(reference[i]);
        if (difference > max_absolute) max_absolute = difference;
        mean_absolute += difference;
        if (!(difference <= tolerance)) {
            fprintf(stderr, "lss_downsample.output[%zu]: got %.9g expected %.9g diff %.9g\n",
                    i, actual[i], reference[i], difference);
            return 0;
        }
    }
    printf("%-22s max_abs=%.3g mean_abs=%.3g\n", "lss_downsample.output",
           max_absolute, mean_absolute / (double)count);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[256] = {0};
    bf_model *weights = NULL, *oracle = NULL;
    if (!bf_model_open(argv[1], &weights, error, sizeof(error)) ||
        !bf_model_open(argv[2], &oracle, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle);
        return 3;
    }
    bf_lss_downsample *down = NULL;
    if (!bf_lss_downsample_create(weights, &down, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle);
        return 4;
    }
    size_t workspace_bytes = bf_lss_downsample_workspace_bytes(8, 10);
    void *workspace = malloc(workspace_bytes);
    float *output = (float *)malloc(80 * 4 * 5 * sizeof(float));
    int ok = workspace && output &&
        bf_lss_downsample_forward_ref(
            down, tensor(oracle, "lss_downsample.input")->data,
            1, 8, 10, output, workspace, workspace_bytes,
            error, sizeof(error)) &&
        close_array(output, tensor(oracle, "lss_downsample.output")) &&
        !bf_lss_downsample_forward_ref(
            down, tensor(oracle, "lss_downsample.input")->data,
            1, 8, 10, output, workspace, workspace_bytes - 1,
            error, sizeof(error));
    if (!ok) fprintf(stderr, "LSS downsample failure: %s\n", error);
    free(workspace); free(output);
    bf_lss_downsample_destroy(down);
    bf_model_close(weights); bf_model_close(oracle);
    if (!ok) return 5;
    puts("real-checkpoint LSS downsample and x/y permute match PyTorch oracle");
    return 0;
}
