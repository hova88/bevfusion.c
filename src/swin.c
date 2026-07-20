#include "bf_swin.h"
#include "bf_kernels.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int multiply_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int attention_core(const float *windows, const float *qkv_weight,
    const float *qkv_bias, const float *relative_bias,
    const int64_t *relative_index, const float *mask, size_t mask_windows,
    const float *projection_weight, const float *projection_bias, float *output,
    size_t total_windows, size_t tokens, size_t channels, size_t heads,
    size_t relative_bias_rows, float *qkv, float *attention, float *projected) {
    const size_t head_channels = channels / heads;
    const float scale = 1.0f / sqrtf((float)head_channels);
    int ok = 1;
    for (size_t window = 0; window < total_windows && ok; ++window) {
        const float *input = windows + window * tokens * channels;
        bf_linear_f32(input, qkv_weight, qkv_bias, qkv,
                      tokens, channels, 3 * channels);
        memset(projected, 0, tokens * channels * sizeof(*projected));
        for (size_t head = 0; head < heads && ok; ++head)
            for (size_t query = 0; query < tokens; ++query) {
                float maximum = -FLT_MAX;
                for (size_t key = 0; key < tokens; ++key) {
                    int64_t bias_row = relative_index[query * tokens + key];
                    if (bias_row < 0 || (uint64_t)bias_row >= relative_bias_rows) {
                        ok = 0; break;
                    }
                    float score = relative_bias[(size_t)bias_row * heads + head];
#if defined(BF_WITH_OPENMP)
#pragma omp simd reduction(+:score)
#endif
                    for (size_t lane = 0; lane < head_channels; ++lane) {
                        size_t channel = head * head_channels + lane;
                        score += qkv[query * 3 * channels + channel] * scale *
                                 qkv[key * 3 * channels + channels + channel];
                    }
                    if (mask)
                        score += mask[((window % mask_windows) * tokens + query) * tokens + key];
                    attention[query * tokens + key] = score;
                    if (score > maximum) maximum = score;
                }
                if (!ok) break;
                double sum = 0.0;
                for (size_t key = 0; key < tokens; ++key) {
                    float value = expf(attention[query * tokens + key] - maximum);
                    attention[query * tokens + key] = value; sum += value;
                }
                float inverse_sum = (float)(1.0 / sum);
                for (size_t key = 0; key < tokens; ++key) {
                    float probability = attention[query * tokens + key] * inverse_sum;
                    for (size_t lane = 0; lane < head_channels; ++lane) {
                        size_t channel = head * head_channels + lane;
                        projected[query * channels + channel] += probability *
                            qkv[key * 3 * channels + 2 * channels + channel];
                    }
                }
            }
        if (ok)
            bf_linear_f32(projected, projection_weight, projection_bias,
                          output + window * tokens * channels,
                          tokens, channels, channels);
    }
    return ok;
}

int bf_swin_window_attention_f32_ref(
    const float *windows, const float *qkv_weight, const float *qkv_bias,
    const float *relative_bias, const int64_t *relative_index,
    const float *mask, size_t mask_windows,
    const float *projection_weight, const float *projection_bias,
    float *output, size_t total_windows, size_t tokens, size_t channels,
    size_t heads, size_t relative_bias_rows) {
    if (!windows || !qkv_weight || !qkv_bias || !relative_bias ||
        !relative_index || !projection_weight || !projection_bias || !output ||
        !total_windows || !tokens || !channels || !heads ||
        channels % heads || !relative_bias_rows || (mask && !mask_windows)) return 0;
    size_t qkv_count, attention_count, projected_count;
    if (!multiply_size(tokens, 3 * channels, &qkv_count) ||
        !multiply_size(tokens, tokens, &attention_count) ||
        !multiply_size(tokens, channels, &projected_count)) return 0;
    float *qkv = (float *)malloc(qkv_count * sizeof(*qkv));
    float *attention = (float *)malloc(attention_count * sizeof(*attention));
    float *projected = (float *)malloc(projected_count * sizeof(*projected));
    if (!qkv || !attention || !projected) {
        free(qkv); free(attention); free(projected);
        return 0;
    }
    int ok = attention_core(windows, qkv_weight, qkv_bias, relative_bias,
        relative_index, mask, mask_windows, projection_weight, projection_bias,
        output, total_windows, tokens, channels, heads, relative_bias_rows,
        qkv, attention, projected);
    free(qkv); free(attention); free(projected);
    return ok;
}

static size_t cyclic_source(size_t position, size_t extent, size_t shift, int reverse) {
    if (!shift) return position;
    if (!reverse) return (position + shift) % extent;
    return (position + extent - shift) % extent;
}

static size_t mask_region(size_t position, size_t extent,
                          size_t window, size_t shift) {
    if (position < extent - window) return 0;
    if (position < extent - shift) return 1;
    return 2;
}

int bf_swin_shifted_window_f32_ref(
    const float *query, const float *qkv_weight, const float *qkv_bias,
    const float *relative_bias, const int64_t *relative_index,
    const float *projection_weight, const float *projection_bias,
    float *output, const bf_swin_window_desc *d) {
    if (!query || !qkv_weight || !qkv_bias || !relative_bias || !relative_index ||
        !projection_weight || !projection_bias || !output || !d || !d->batches ||
        !d->height || !d->width || !d->channels || !d->heads ||
        d->channels % d->heads || !d->window_size || d->shift_size >= d->window_size)
        return 0;
    if (d->height > SIZE_MAX - (d->window_size - 1) ||
        d->width > SIZE_MAX - (d->window_size - 1)) return 0;
    size_t padded_h = ((d->height + d->window_size - 1) / d->window_size) * d->window_size;
    size_t padded_w = ((d->width + d->window_size - 1) / d->window_size) * d->window_size;
    size_t windows_y = padded_h / d->window_size;
    size_t windows_x = padded_w / d->window_size;
    size_t windows_per_image, total_windows, tokens, window_values;
    if (!multiply_size(windows_y, windows_x, &windows_per_image) ||
        !multiply_size(d->batches, windows_per_image, &total_windows) ||
        !multiply_size(d->window_size, d->window_size, &tokens) ||
        !multiply_size(total_windows, tokens, &window_values) ||
        !multiply_size(window_values, d->channels, &window_values)) return 0;
    float *windows = (float *)calloc(window_values, sizeof(*windows));
    float *attended = (float *)malloc(window_values * sizeof(*attended));
    size_t mask_values;
    if (!multiply_size(windows_per_image, tokens, &mask_values) ||
        !multiply_size(mask_values, tokens, &mask_values)) {
        free(windows); free(attended);
        return 0;
    }
    float *mask = d->shift_size ? (float *)malloc(mask_values * sizeof(*mask)) : NULL;
    if (!windows || !attended || (d->shift_size && !mask)) {
        free(windows); free(attended); free(mask);
        return 0;
    }
    for (size_t b = 0; b < d->batches; ++b)
        for (size_t wy = 0; wy < windows_y; ++wy)
            for (size_t wx = 0; wx < windows_x; ++wx) {
                size_t window = (b * windows_y + wy) * windows_x + wx;
                for (size_t iy = 0; iy < d->window_size; ++iy)
                    for (size_t ix = 0; ix < d->window_size; ++ix) {
                        size_t shifted_y = wy * d->window_size + iy;
                        size_t shifted_x = wx * d->window_size + ix;
                        size_t source_y = cyclic_source(shifted_y, padded_h, d->shift_size, 0);
                        size_t source_x = cyclic_source(shifted_x, padded_w, d->shift_size, 0);
                        if (source_y >= d->height || source_x >= d->width) continue;
                        size_t token = iy * d->window_size + ix;
                        memcpy(windows + (window * tokens + token) * d->channels,
                               query + (b * d->height * d->width + source_y * d->width + source_x) * d->channels,
                               d->channels * sizeof(*windows));
                    }
            }
    if (mask) {
        for (size_t wy = 0; wy < windows_y; ++wy)
            for (size_t wx = 0; wx < windows_x; ++wx) {
                size_t window = wy * windows_x + wx;
                for (size_t ay = 0; ay < d->window_size; ++ay)
                    for (size_t ax = 0; ax < d->window_size; ++ax) {
                        size_t a = ay * d->window_size + ax;
                        size_t a_label = mask_region(wy * d->window_size + ay, padded_h,
                                                     d->window_size, d->shift_size) * 3 +
                                         mask_region(wx * d->window_size + ax, padded_w,
                                                     d->window_size, d->shift_size);
                        for (size_t by = 0; by < d->window_size; ++by)
                            for (size_t bx = 0; bx < d->window_size; ++bx) {
                                size_t b = by * d->window_size + bx;
                                size_t b_label = mask_region(wy * d->window_size + by, padded_h,
                                                             d->window_size, d->shift_size) * 3 +
                                                 mask_region(wx * d->window_size + bx, padded_w,
                                                             d->window_size, d->shift_size);
                                mask[(window * tokens + a) * tokens + b] =
                                    a_label == b_label ? 0.0f : -100.0f;
                            }
                    }
            }
    }
    size_t bias_side = 2 * d->window_size - 1;
    size_t bias_rows;
    int ok = multiply_size(bias_side, bias_side, &bias_rows) &&
        bf_swin_window_attention_f32_ref(
            windows, qkv_weight, qkv_bias, relative_bias, relative_index,
            mask, mask ? windows_per_image : 0, projection_weight, projection_bias,
            attended, total_windows, tokens, d->channels, d->heads, bias_rows);
    if (ok) {
        for (size_t b = 0; b < d->batches; ++b)
            for (size_t y = 0; y < d->height; ++y)
                for (size_t x = 0; x < d->width; ++x) {
                    size_t shifted_y = cyclic_source(y, padded_h, d->shift_size, 1);
                    size_t shifted_x = cyclic_source(x, padded_w, d->shift_size, 1);
                    size_t wy = shifted_y / d->window_size, iy = shifted_y % d->window_size;
                    size_t wx = shifted_x / d->window_size, ix = shifted_x % d->window_size;
                    size_t window = (b * windows_y + wy) * windows_x + wx;
                    size_t token = iy * d->window_size + ix;
                    memcpy(output + (b * d->height * d->width + y * d->width + x) * d->channels,
                           attended + (window * tokens + token) * d->channels,
                           d->channels * sizeof(*output));
                }
    }
    free(windows); free(attended); free(mask);
    return ok;
}

size_t bf_swin_shifted_window_workspace_bytes(const bf_swin_window_desc *d) {
    if (!d || !d->channels || !d->window_size) return 0;
    size_t tokens, tc, values;
    if (!multiply_size(d->window_size, d->window_size, &tokens) ||
        !multiply_size(tokens, d->channels, &tc) || tc > SIZE_MAX / 6 ||
        !multiply_size(tokens, tokens, &values) || values > SIZE_MAX / 2 ||
        6 * tc > SIZE_MAX - 2 * values ||
        6 * tc + 2 * values > SIZE_MAX / sizeof(float)) return 0;
    return (6 * tc + 2 * values) * sizeof(float);
}

int bf_swin_shifted_window_f32_workspace_ref(
    const float *query, const float *qkv_weight, const float *qkv_bias,
    const float *relative_bias, const int64_t *relative_index,
    const float *projection_weight, const float *projection_bias,
    float *output, const bf_swin_window_desc *d,
    void *workspace, size_t workspace_bytes) {
    size_t required = bf_swin_shifted_window_workspace_bytes(d);
    if (!query || !qkv_weight || !qkv_bias || !relative_bias || !relative_index ||
        !projection_weight || !projection_bias || !output || !d || !workspace ||
        workspace_bytes < required || !d->batches || !d->height || !d->width ||
        !d->heads || d->channels % d->heads ||
        d->shift_size >= d->window_size ||
        d->height > SIZE_MAX - (d->window_size - 1) ||
        d->width > SIZE_MAX - (d->window_size - 1)) return 0;
    size_t padded_h = ((d->height + d->window_size - 1) / d->window_size) * d->window_size;
    size_t padded_w = ((d->width + d->window_size - 1) / d->window_size) * d->window_size;
    size_t windows_y = padded_h / d->window_size, windows_x = padded_w / d->window_size;
    size_t tokens = d->window_size * d->window_size, tc = tokens * d->channels;
    float *window = workspace;
    float *attended = window + tc;
    float *qkv = attended + tc;
    float *attention = qkv + 3 * tc;
    float *projected = attention + tokens * tokens;
    float *mask = projected + tc;
    size_t bias_side = 2 * d->window_size - 1, bias_rows = bias_side * bias_side;
    for (size_t b = 0; b < d->batches; ++b)
        for (size_t wy = 0; wy < windows_y; ++wy)
            for (size_t wx = 0; wx < windows_x; ++wx) {
                memset(window, 0, tc * sizeof(*window));
                for (size_t iy = 0; iy < d->window_size; ++iy)
                    for (size_t ix = 0; ix < d->window_size; ++ix) {
                        size_t shifted_y = wy * d->window_size + iy;
                        size_t shifted_x = wx * d->window_size + ix;
                        size_t source_y = cyclic_source(shifted_y, padded_h, d->shift_size, 0);
                        size_t source_x = cyclic_source(shifted_x, padded_w, d->shift_size, 0);
                        if (source_y < d->height && source_x < d->width)
                            memcpy(window + (iy * d->window_size + ix) * d->channels,
                                   query + ((b * d->height + source_y) * d->width + source_x) * d->channels,
                                   d->channels * sizeof(*window));
                    }
                const float *mask_value = NULL;
                if (d->shift_size) {
                    mask_value = mask;
                    for (size_t ay = 0; ay < d->window_size; ++ay)
                        for (size_t ax = 0; ax < d->window_size; ++ax) {
                            size_t a = ay * d->window_size + ax;
                            size_t al = mask_region(wy * d->window_size + ay, padded_h,
                                d->window_size, d->shift_size) * 3 +
                                mask_region(wx * d->window_size + ax, padded_w,
                                d->window_size, d->shift_size);
                            for (size_t by = 0; by < d->window_size; ++by)
                                for (size_t bx = 0; bx < d->window_size; ++bx) {
                                    size_t key = by * d->window_size + bx;
                                    size_t bl = mask_region(wy * d->window_size + by, padded_h,
                                        d->window_size, d->shift_size) * 3 +
                                        mask_region(wx * d->window_size + bx, padded_w,
                                        d->window_size, d->shift_size);
                                    mask[a * tokens + key] = al == bl ? 0.0f : -100.0f;
                                }
                        }
                }
                if (!attention_core(window, qkv_weight, qkv_bias, relative_bias,
                        relative_index, mask_value, mask_value ? 1 : 0,
                        projection_weight, projection_bias, attended, 1, tokens,
                        d->channels, d->heads, bias_rows, qkv, attention, projected))
                    return 0;
                for (size_t iy = 0; iy < d->window_size; ++iy)
                    for (size_t ix = 0; ix < d->window_size; ++ix) {
                        size_t shifted_y = wy * d->window_size + iy;
                        size_t shifted_x = wx * d->window_size + ix;
                        size_t destination_y = cyclic_source(shifted_y, padded_h, d->shift_size, 0);
                        size_t destination_x = cyclic_source(shifted_x, padded_w, d->shift_size, 0);
                        if (destination_y < d->height && destination_x < d->width)
                            memcpy(output + ((b * d->height + destination_y) * d->width + destination_x) * d->channels,
                                   attended + (iy * d->window_size + ix) * d->channels,
                                   d->channels * sizeof(*output));
                    }
            }
    return 1;
}
