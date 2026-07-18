#include "bf_model.h"
#include "bf_swin.h"

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

static int run_route(const bf_model *model, size_t shift,
                     const char *expected_name, float *output) {
    bf_swin_window_desc desc = {2, 5, 7, 8, 2, 3, shift};
    return bf_swin_shifted_window_f32_ref(
        tensor(model, "swin.input")->data,
        tensor(model, "swin.qkv_weight")->data,
        tensor(model, "swin.qkv_bias")->data,
        tensor(model, "swin.relative_bias")->data,
        tensor(model, "swin.relative_index")->data,
        tensor(model, "swin.projection_weight")->data,
        tensor(model, "swin.projection_bias")->data,
        output, &desc) &&
        close_array(expected_name, output, tensor(model, expected_name), 3e-6f, 4e-6f);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[256];
    bf_model *model = NULL;
    if (!bf_model_open(argv[1], &model, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 3;
    }
    const bf_tensor *expected = tensor(model, "swin.shifted_output");
    float *output = expected ? (float *)malloc((size_t)expected->nbytes) : NULL;
    int ok = output && run_route(model, 0, "swin.regular_output", output) &&
             run_route(model, 1, "swin.shifted_output", output);
    bf_swin_window_desc invalid = {1, 2, 2, 4, 3, 2, 0};
    float dummy = 0.0f;
    ok = ok && !bf_swin_shifted_window_f32_ref(
        &dummy, &dummy, &dummy, &dummy, (const int64_t *)&dummy,
        &dummy, &dummy, &dummy, &invalid);
    free(output);
    bf_model_close(model);
    if (!ok) return 4;
    puts("regular and shifted Swin window attention match PyTorch oracle");
    return 0;
}
