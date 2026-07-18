#include "bf_model.h"
#include "bf_transfusion.h"

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
    const float *reference = (const float *)expected->data;
    size_t count = (size_t)(expected->nbytes / sizeof(float));
    float max_absolute = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float difference = fabsf(actual[i] - reference[i]);
        float tolerance = atol + rtol * fabsf(reference[i]);
        if (difference > max_absolute) max_absolute = difference;
        if (!(difference <= tolerance)) {
            fprintf(stderr, "%s[%zu]: got %.9g expected %.9g diff %.9g tolerance %.9g\n",
                    name, i, actual[i], reference[i], difference, tolerance);
            return 0;
        }
    }
    printf("%-22s max_abs=%.3g\n", name, max_absolute);
    return 1;
}

static int test_proposals(const bf_model *model) {
    float suppressed[2 * 10 * 8 * 9], scores[2 * 12];
    int64_t classes[2 * 12], indices[2 * 12];
    return bf_transfusion_select_proposals_f32_ref(
        tensor(model, "transfusion.dense_logits")->data, suppressed, scores,
        classes, indices, 2, 10, 8, 9, 12, 3) &&
        close_array("transfusion.suppressed", suppressed,
                    tensor(model, "transfusion.suppressed"), 2e-7f, 2e-6f) &&
        close_array("transfusion.proposals", scores,
                    tensor(model, "transfusion.proposal_scores"), 2e-7f, 2e-6f) &&
        memcmp(classes, tensor(model, "transfusion.proposal_classes")->data,
               sizeof(classes)) == 0 &&
        memcmp(indices, tensor(model, "transfusion.proposal_indices")->data,
               sizeof(indices)) == 0;
}

static int test_decode(const bf_model *model) {
    float boxes[2 * 12 * 9], scores[2 * 12];
    int64_t labels[2 * 12];
    const float voxel_size[2] = {0.075f, 0.075f};
    const float minimum[2] = {-54.0f, -54.0f};
    int ok = bf_transfusion_decode_raw_f32_ref(
        tensor(model, "transfusion.prediction_logits")->data,
        tensor(model, "transfusion.query_scores")->data,
        tensor(model, "transfusion.proposal_classes")->data,
        tensor(model, "transfusion.center")->data,
        tensor(model, "transfusion.height")->data,
        tensor(model, "transfusion.dimension_log")->data,
        tensor(model, "transfusion.rotation")->data,
        tensor(model, "transfusion.velocity")->data,
        boxes, scores, labels, 2, 10, 12, 8.0f, voxel_size, minimum) &&
        close_array("transfusion.boxes", boxes,
                    tensor(model, "transfusion.boxes"), 4e-6f, 3e-6f) &&
        close_array("transfusion.final_scores", scores,
                    tensor(model, "transfusion.final_scores"), 2e-7f, 2e-6f) &&
        memcmp(labels, tensor(model, "transfusion.proposal_classes")->data,
               sizeof(labels)) == 0;
    const float range[6] = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
    bf_detections detections[2];
    ok = ok && bf_transfusion_filter_detections(boxes, scores, labels, 2, 12,
                                                 0.1f, range, detections);
    const int64_t *keep = tensor(model, "transfusion.keep")->data;
    const float *expected_boxes = tensor(model, "transfusion.boxes")->data;
    const float *expected_scores = tensor(model, "transfusion.final_scores")->data;
    for (size_t batch = 0; batch < 2 && ok; ++batch) {
        int expected_count = 0;
        for (size_t proposal = 0; proposal < 12; ++proposal) expected_count += keep[batch * 12 + proposal] != 0;
        ok = detections[batch].count == expected_count;
        int at = 0;
        for (size_t proposal = 0; proposal < 12 && ok; ++proposal) {
            if (!keep[batch * 12 + proposal]) continue;
            const bf_detection *detection = &detections[batch].items[at++];
            const float *box = expected_boxes + (batch * 12 + proposal) * 9;
            ok = fabsf(detection->x - box[0]) <= 4e-6f &&
                 fabsf(detection->y - box[1]) <= 4e-6f &&
                 fabsf(detection->score - expected_scores[batch * 12 + proposal]) <= 2e-7f &&
                 detection->class_id == labels[batch * 12 + proposal];
        }
    }
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
    int ok = test_proposals(model) && test_decode(model);
    bf_model_close(model);
    if (!ok) return 4;
    puts("TransFusion proposal selection and canonical decode match PyTorch oracle");
    return 0;
}
