#include "bf_model.h"
#include "bf_swin_backbone.h"

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
    printf("%-22s max_abs=%.3g mean_abs=%.3g\n", name, max_absolute,
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
    bf_swin_backbone *backbone = NULL;
    if (!bf_swin_backbone_create(weights, &backbone, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle);
        return 4;
    }
    bf_swin_backbone_shapes shapes;
    int ok = bf_swin_backbone_output_shapes(32, 48, &shapes) &&
             shapes.height[0] == 4 && shapes.width[0] == 6 &&
             shapes.height[1] == 2 && shapes.width[1] == 3 &&
             shapes.height[2] == 1 && shapes.width[2] == 2;
    size_t workspace_bytes = bf_swin_backbone_workspace_bytes(1, 32, 48);
    void *workspace = malloc(workspace_bytes);
    float *outputs[3] = {NULL, NULL, NULL};
    for (size_t i = 0; i < 3; ++i)
        outputs[i] = (float *)malloc(shapes.channels[i] * shapes.height[i] *
                                     shapes.width[i] * sizeof(float));
    ok = ok && workspace && outputs[0] && outputs[1] && outputs[2] &&
        bf_swin_backbone_forward_ref(
            backbone, tensor(oracle, "swin_backbone.input")->data, 1, 32, 48,
            outputs[0], outputs[1], outputs[2], workspace, workspace_bytes,
            error, sizeof(error));
    for (size_t i = 0; i < 3 && ok; ++i) {
        char name[40];
        snprintf(name, sizeof(name), "swin_backbone.output%zu", i);
        ok = close_array(name, outputs[i], tensor(oracle, name), 4e-5f, 4e-5f);
    }
    ok = ok && !bf_swin_backbone_forward_ref(
        backbone, tensor(oracle, "swin_backbone.input")->data, 1, 32, 48,
        outputs[0], outputs[1], outputs[2], workspace, workspace_bytes - 1,
        error, sizeof(error));
    if (!ok) fprintf(stderr, "Swin backbone failure: %s\n", error);
    free(workspace);
    for (size_t i = 0; i < 3; ++i) free(outputs[i]);
    bf_swin_backbone_destroy(backbone);
    bf_model_close(weights);
    bf_model_close(oracle);
    if (!ok) return 5;
    puts("real-checkpoint four-stage Swin-T matches PyTorch oracle");
    return 0;
}
