#include "bf_image_fpn.h"
#include "bf_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

static int close_array(const char *name, const float *actual,
                       const bf_tensor *expected, float atol, float rtol) {
    if (!expected || expected->dtype != BF_DTYPE_F32) return 0;
    const float *reference = (const float *)expected->data;
    size_t count = (size_t)(expected->nbytes / sizeof(float));
    float max_absolute = 0.0f;
    double mean_absolute = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float difference = fabsf(actual[i] - reference[i]);
        float tolerance = atol + rtol * fabsf(reference[i]);
        if (difference > max_absolute) max_absolute = difference;
        mean_absolute += difference;
        if (!(difference <= tolerance)) {
            fprintf(stderr, "%s[%zu]: got %.9g expected %.9g diff %.9g tolerance %.9g\n",
                    name, i, actual[i], reference[i], difference, tolerance);
            return 0;
        }
    }
    printf("%-18s max_abs=%.3g mean_abs=%.3g\n", name, max_absolute,
           mean_absolute / (double)count);
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
    bf_image_fpn *fpn = NULL;
    if (!bf_image_fpn_create(weights, &fpn, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle);
        return 4;
    }
    size_t workspace_bytes = bf_image_fpn_workspace_bytes(4, 6);
    void *workspace = malloc(workspace_bytes);
    float *output0 = (float *)malloc(256 * 4 * 6 * sizeof(float));
    float *output1 = (float *)malloc(256 * 2 * 3 * sizeof(float));
    int ok = workspace && output0 && output1 &&
        bf_image_fpn_forward_ref(
            fpn, tensor(oracle, "image_fpn.input0")->data,
            tensor(oracle, "image_fpn.input1")->data,
            tensor(oracle, "image_fpn.input2")->data,
            1, 4, 6, output0, output1, workspace, workspace_bytes,
            error, sizeof(error)) &&
        close_array("image_fpn.output0", output0,
                    tensor(oracle, "image_fpn.output0"), 3e-5f, 3e-5f) &&
        close_array("image_fpn.output1", output1,
                    tensor(oracle, "image_fpn.output1"), 3e-5f, 3e-5f) &&
        !bf_image_fpn_forward_ref(
            fpn, tensor(oracle, "image_fpn.input0")->data,
            tensor(oracle, "image_fpn.input1")->data,
            tensor(oracle, "image_fpn.input2")->data,
            1, 4, 6, output0, output1, workspace, workspace_bytes - 1,
            error, sizeof(error));
    if (!ok) fprintf(stderr, "image FPN failure: %s\n", error);
    free(workspace); free(output0); free(output1);
    bf_image_fpn_destroy(fpn);
    bf_model_close(weights); bf_model_close(oracle);
    if (!ok) return 5;
    puts("real-checkpoint GeneralizedLSSFPN matches PyTorch oracle");
    return 0;
}
