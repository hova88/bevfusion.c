#include "bf_bev_stage.h"
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
    char error[256];
    bf_model *weights = NULL, *oracle = NULL;
    if (!bf_model_open(argv[1], &weights, error, sizeof(error)) ||
        !bf_model_open(argv[2], &oracle, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle);
        return 3;
    }
    bf_bev_stage *stage = NULL;
    if (!bf_bev_stage_create(weights, &stage, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle);
        return 4;
    }
    const size_t height = 8, width = 8, hw = height * width;
    size_t workspace_bytes = bf_bev_stage_workspace_bytes(height, width);
    void *workspace = malloc(workspace_bytes);
    float *spatial = (float *)malloc(512 * hw * sizeof(*spatial));
    float *shared = (float *)malloc(128 * hw * sizeof(*shared));
    float *heatmap = (float *)malloc(10 * hw * sizeof(*heatmap));
    int ok = workspace && spatial && shared && heatmap &&
        bf_bev_stage_forward_ref(stage, tensor(oracle, "bev_stage.input")->data,
                                 1, height, width, spatial, shared, heatmap,
                                 workspace, workspace_bytes, error, sizeof(error)) &&
        close_array("bev_stage.spatial", spatial, tensor(oracle, "bev_stage.spatial"), 3e-5f, 3e-5f) &&
        close_array("bev_stage.shared", shared, tensor(oracle, "bev_stage.shared"), 3e-5f, 3e-5f) &&
        close_array("bev_stage.heatmap", heatmap, tensor(oracle, "bev_stage.heatmap"), 3e-5f, 3e-5f) &&
        !bf_bev_stage_forward_ref(stage, tensor(oracle, "bev_stage.input")->data,
                                  1, height, width, spatial, shared, heatmap,
                                  workspace, workspace_bytes - 1, error, sizeof(error));
    if (!ok) fprintf(stderr, "BEV stage failure: %s\n", error);
    free(workspace); free(spatial); free(shared); free(heatmap);
    bf_bev_stage_destroy(stage);
    bf_model_close(weights); bf_model_close(oracle);
    if (!ok) return 5;
    puts("real-checkpoint BEV fusion/backbone/heatmap stage matches PyTorch oracle");
    return 0;
}
