#define _POSIX_C_SOURCE 200809L
#include "bf_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill(float *values, size_t count, float phase) {
    for (size_t i = 0; i < count; ++i)
        values[i] = sinf((float)i * 0.173f + phase) * 0.25f;
}

static int close_values(const float *a, const float *b, size_t count,
                        float tolerance, float *maximum) {
    *maximum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float difference = fabsf(a[i] - b[i]);
        if (difference > *maximum) *maximum = difference;
        if (!(difference <= tolerance)) return 0;
    }
    return 1;
}

int main(void) {
    enum { ROWS = 31, IN = 64, OUT = 48, H = 5, W = 7, CO = 11 };
    float *linear_input = malloc(ROWS * IN * sizeof(float));
    float *linear_weight = malloc(OUT * IN * sizeof(float));
    float *linear_bias = malloc(OUT * sizeof(float));
    float *linear_ref = malloc(ROWS * OUT * sizeof(float));
    float *linear_fast = malloc(ROWS * OUT * sizeof(float));
    float *conv_input = malloc(IN * H * W * sizeof(float));
    float *conv_weight = malloc(CO * IN * sizeof(float));
    float *conv_bias = malloc(CO * sizeof(float));
    float *conv_ref = malloc(CO * H * W * sizeof(float));
    float *conv_fast = malloc(CO * H * W * sizeof(float));
    int ok = linear_input && linear_weight && linear_bias && linear_ref &&
        linear_fast && conv_input && conv_weight && conv_bias && conv_ref && conv_fast;
    if (!ok) return 2;
    fill(linear_input, ROWS * IN, 0.1f); fill(linear_weight, OUT * IN, 0.2f);
    fill(linear_bias, OUT, 0.3f); fill(conv_input, IN * H * W, 0.4f);
    fill(conv_weight, CO * IN, 0.5f); fill(conv_bias, CO, 0.6f);

    bf_linear_f32_ref(linear_input, linear_weight, linear_bias, linear_ref,
                      ROWS, IN, OUT);
    bf_linear_f32(linear_input, linear_weight, linear_bias, linear_fast,
                  ROWS, IN, OUT);
    bf_conv2d_desc one = {1, IN, H, W, CO, 1, 1, 1, 1, 0, 0, 1, 1, 1};
    ok = bf_conv2d_f32_ref(conv_input, conv_weight, conv_bias, conv_ref, &one) &&
         bf_conv2d_f32(conv_input, conv_weight, conv_bias, conv_fast, &one);
    float linear_max = 0.0f, conv_max = 0.0f;
    ok = ok && close_values(linear_ref, linear_fast, ROWS * OUT, 2e-5f, &linear_max) &&
         close_values(conv_ref, conv_fast, CO * H * W, 2e-5f, &conv_max);

    setenv("BF_CPU_SCALAR", "1", 1);
    memset(linear_fast, 0, ROWS * OUT * sizeof(float));
    bf_linear_f32(linear_input, linear_weight, linear_bias, linear_fast,
                  ROWS, IN, OUT);
    float forced_max = 0.0f;
    ok = ok && strcmp(bf_cpu_kernel_backend(), "scalar-forced") == 0 &&
         close_values(linear_ref, linear_fast, ROWS * OUT, 0.0f, &forced_max);
    unsetenv("BF_CPU_SCALAR");

    printf("cpu dispatch backend=%s linear_max=%.3g conv1x1_max=%.3g forced_max=%.3g\n",
           bf_cpu_kernel_backend(), linear_max, conv_max, forced_max);
    free(linear_input); free(linear_weight); free(linear_bias); free(linear_ref);
    free(linear_fast); free(conv_input); free(conv_weight); free(conv_bias);
    free(conv_ref); free(conv_fast);
    return ok ? 0 : 3;
}
