#include "bf_lss.h"
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
    printf("%-16s max_abs=%.3g\n", name, max_absolute);
    return 1;
}

static int test_geometry(const bf_model *model, float *geometry) {
    int ok = bf_lss_geometry_f32_ref(
        tensor(model, "lss.frustum")->data,
        tensor(model, "lss.camera_rotation")->data,
        tensor(model, "lss.camera_translation")->data,
        tensor(model, "lss.intrinsics")->data,
        tensor(model, "lss.post_rotation")->data,
        tensor(model, "lss.post_translation")->data,
        tensor(model, "lss.extra_rotation")->data,
        tensor(model, "lss.extra_translation")->data,
        geometry, 2, 2, 4, 3, 5) &&
        close_array("lss_geometry", geometry, tensor(model, "lss.geometry"), 3e-6f, 3e-6f);
    float singular[9] = {0};
    float vector[3] = {0};
    float one_point[3] = {0};
    ok = ok && !bf_lss_geometry_f32_ref(one_point, singular, vector, singular,
                                         singular, vector, NULL, NULL,
                                         one_point, 1, 1, 1, 1, 1);
    return ok;
}

static int test_lift_pool(const bf_model *model, const float *geometry) {
    bf_lss_desc desc = {
        2, 2, 4, 3, 5, 3,
        {-6.0f, -6.0f, -3.0f}, {2.0f, 2.0f, 2.0f}, {6, 6, 4}
    };
    const bf_tensor *lifted_expected = tensor(model, "lss.lifted");
    const bf_tensor *bev_expected = tensor(model, "lss.bev");
    float *lifted = (float *)malloc((size_t)lifted_expected->nbytes);
    float *bev = (float *)malloc((size_t)bev_expected->nbytes);
    float *fused = (float *)malloc((size_t)bev_expected->nbytes);
    if (!lifted || !bev || !fused) {
        free(lifted); free(bev); free(fused);
        return 0;
    }
    const float *logits = tensor(model, "lss.depth_logits")->data;
    const float *context = tensor(model, "lss.context")->data;
    int ok = bf_lss_lift_f32_ref(logits, context, lifted, &desc) &&
        close_array("lss_lift", lifted, lifted_expected, 2e-7f, 3e-6f) &&
        bf_lss_bev_pool_f32_ref(lifted, geometry, bev, &desc) &&
        close_array("lss_pool", bev, bev_expected, 5e-7f, 4e-6f) &&
        bf_lss_lift_pool_f32_ref(logits, context, geometry, fused, &desc) &&
        close_array("lss_fused", fused, bev_expected, 5e-7f, 4e-6f);
    float invalid_logits[4] = {0.0f, 0.0f, INFINITY, 0.0f};
    bf_lss_desc tiny = {1, 1, 4, 1, 1, 1, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}};
    float tiny_context = 1.0f, tiny_geometry[12] = {0}, tiny_output;
    ok = ok && !bf_lss_lift_pool_f32_ref(invalid_logits, &tiny_context,
                                          tiny_geometry, &tiny_output, &tiny);
    free(lifted); free(bev); free(fused);
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
    const bf_tensor *expected = tensor(model, "lss.geometry");
    float *geometry = expected ? (float *)malloc((size_t)expected->nbytes) : NULL;
    int ok = geometry && test_geometry(model, geometry) && test_lift_pool(model, geometry);
    free(geometry);
    bf_model_close(model);
    if (!ok) return 4;
    puts("LSS geometry, lift, pool, and fused route match PyTorch oracle");
    return 0;
}
