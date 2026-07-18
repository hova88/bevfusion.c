#include "bf_model.h"
#include "bf_transfusion.h"
#include "bf_transfusion_decoder.h"

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
    float maximum = 0.0f;
    double mean = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float difference = fabsf(actual[i] - reference[i]);
        if (difference > maximum) maximum = difference;
        mean += difference;
        if (!(difference <= atol + rtol * fabsf(reference[i]))) {
            fprintf(stderr, "%s[%zu]: got %.9g expected %.9g diff %.9g\n",
                    name, i, actual[i], reference[i], difference);
            return 0;
        }
    }
    printf("%-30s max_abs=%.3g mean_abs=%.3g\n", name, maximum,
           mean / (double)count);
    return 1;
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
    bf_transfusion_decoder *decoder = NULL;
    if (!bf_transfusion_decoder_create(weights, &decoder, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        bf_model_close(weights); bf_model_close(oracle); return 4;
    }
    const size_t proposals = 12;
    size_t workspace_bytes = bf_transfusion_decoder_workspace_bytes(5, 5, proposals);
    void *workspace = malloc(workspace_bytes);
    float *center = malloc(2 * proposals * sizeof(float));
    float *height = malloc(proposals * sizeof(float));
    float *dimension = malloc(3 * proposals * sizeof(float));
    float *rotation = malloc(2 * proposals * sizeof(float));
    float *velocity = malloc(2 * proposals * sizeof(float));
    float *heatmap = malloc(10 * proposals * sizeof(float));
    float *scores = malloc(10 * proposals * sizeof(float));
    int64_t *labels = malloc(proposals * sizeof(int64_t));
    int64_t *indices = malloc(proposals * sizeof(int64_t));
    bf_transfusion_raw_outputs outputs = {
        center, height, dimension, rotation, velocity, heatmap, scores,
        labels, indices
    };
    int ok = workspace && center && height && dimension && rotation && velocity &&
        heatmap && scores && labels && indices &&
        bf_transfusion_decoder_forward_ref(
            decoder, tensor(oracle, "transfusion_decoder.shared")->data,
            tensor(oracle, "transfusion_decoder.dense_heatmap")->data,
            1, 5, 5, proposals, &outputs, workspace, workspace_bytes,
            error, sizeof(error));
    const int64_t *expected_labels = tensor(oracle, "transfusion_decoder.labels")->data;
    const int64_t *expected_indices = tensor(oracle, "transfusion_decoder.indices")->data;
    for (size_t i = 0; i < proposals && ok; ++i)
        ok = labels[i] == expected_labels[i] && indices[i] == expected_indices[i];
    struct { const char *name; float *value; } checks[] = {
        {"center", center}, {"height", height}, {"dim", dimension},
        {"rot", rotation}, {"vel", velocity}, {"heatmap", heatmap},
        {"query_scores", scores}
    };
    for (size_t i = 0; i < sizeof(checks) / sizeof(checks[0]) && ok; ++i) {
        char name[96];
        snprintf(name, sizeof(name), "transfusion_decoder.%s", checks[i].name);
        ok = close_array(name, checks[i].value, tensor(oracle, name), 8e-5f, 8e-5f);
    }
    float boxes[12 * 9], final_scores[12];
    int64_t final_labels[12];
    const float voxel_size[2] = {0.075f, 0.075f};
    const float minimum[2] = {-54.0f, -54.0f};
    ok = ok && bf_transfusion_decode_raw_f32_ref(
        heatmap, scores, labels, center, height, dimension, rotation, velocity,
        boxes, final_scores, final_labels, 1, 10, proposals, 8.0f,
        voxel_size, minimum) &&
        close_array("transfusion_decoder.boxes", boxes,
                    tensor(oracle, "transfusion_decoder.boxes"), 8e-5f, 8e-5f) &&
        close_array("transfusion_decoder.scores", final_scores,
                    tensor(oracle, "transfusion_decoder.scores"), 8e-5f, 8e-5f);
    const int64_t *oracle_final_labels =
        tensor(oracle, "transfusion_decoder.final_labels")->data;
    for (size_t i = 0; i < proposals && ok; ++i)
        ok = final_labels[i] == oracle_final_labels[i];
    bf_detections detections;
    const float range[6] = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
    ok = ok && bf_transfusion_filter_detections(
        boxes, final_scores, final_labels, 1, proposals, 0.0f, range, &detections) &&
        detections.count >= 0 && (size_t)detections.count <= proposals;
    ok = ok && !bf_transfusion_decoder_forward_ref(
        decoder, tensor(oracle, "transfusion_decoder.shared")->data,
        tensor(oracle, "transfusion_decoder.dense_heatmap")->data,
        1, 5, 5, proposals, &outputs, workspace, workspace_bytes - 1,
        error, sizeof(error));
    if (!ok) fprintf(stderr, "TransFusion decoder failure: %s\n", error);
    free(workspace); free(center); free(height); free(dimension); free(rotation);
    free(velocity); free(heatmap); free(scores); free(labels); free(indices);
    bf_transfusion_decoder_destroy(decoder);
    bf_model_close(weights); bf_model_close(oracle);
    if (!ok) return 5;
    puts("real-checkpoint TransFusion decoder and all heads match PyTorch oracle");
    return 0;
}
