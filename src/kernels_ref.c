#include "bf_kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(BF_WITH_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(BF_WITH_CBLAS)
#include <cblas.h>
#endif

static int force_scalar(void) {
    const char *value = getenv("BF_CPU_SCALAR");
    return value && value[0] && strcmp(value, "0") != 0;
}

const char *bf_cpu_kernel_backend(void) {
    if (force_scalar()) return "scalar-forced";
#if defined(BF_WITH_ACCELERATE) && defined(BF_WITH_OPENMP)
    return "accelerate+openmp";
#elif defined(BF_WITH_ACCELERATE)
    return "accelerate";
#elif defined(BF_WITH_CBLAS) && defined(BF_WITH_OPENMP)
    return "cblas+openmp";
#elif defined(BF_WITH_CBLAS)
    return "cblas";
#elif defined(BF_WITH_OPENMP)
    return "openmp";
#else
    return "scalar";
#endif
}

int bf_conv2d_output_shape(const bf_conv2d_desc *d,
                           size_t *out_height, size_t *out_width) {
    if (!d || !out_height || !out_width || !d->n || !d->in_channels ||
        !d->in_height || !d->in_width || !d->out_channels ||
        !d->kernel_height || !d->kernel_width || !d->stride_height ||
        !d->stride_width || !d->dilation_height || !d->dilation_width ||
        !d->groups || d->in_channels % d->groups ||
        d->out_channels % d->groups)
        return 0;
    if (d->kernel_height > (SIZE_MAX - 1) / d->dilation_height + 1 ||
        d->kernel_width > (SIZE_MAX - 1) / d->dilation_width + 1)
        return 0;
    size_t effective_h = d->dilation_height * (d->kernel_height - 1) + 1;
    size_t effective_w = d->dilation_width * (d->kernel_width - 1) + 1;
    if (d->pad_height > (SIZE_MAX - d->in_height) / 2 ||
        d->pad_width > (SIZE_MAX - d->in_width) / 2)
        return 0;
    size_t padded_h = d->in_height + 2 * d->pad_height;
    size_t padded_w = d->in_width + 2 * d->pad_width;
    if (effective_h > padded_h || effective_w > padded_w) return 0;
    *out_height = (padded_h - effective_h) / d->stride_height + 1;
    *out_width = (padded_w - effective_w) / d->stride_width + 1;
    return *out_height && *out_width;
}

int bf_conv2d_f32_ref(const float *input, const float *weight,
                      const float *bias, float *output,
                      const bf_conv2d_desc *d) {
    size_t oh, ow;
    if (!input || !weight || !output ||
        !bf_conv2d_output_shape(d, &oh, &ow)) return 0;
    const size_t ci_group = d->in_channels / d->groups;
    const size_t co_group = d->out_channels / d->groups;
    for (size_t n = 0; n < d->n; ++n) {
        for (size_t co = 0; co < d->out_channels; ++co) {
            const size_t group = co / co_group;
            const size_t ci_begin = group * ci_group;
            for (size_t y = 0; y < oh; ++y) {
                for (size_t x = 0; x < ow; ++x) {
                    float sum = bias ? bias[co] : 0.0f;
                    for (size_t ci_local = 0; ci_local < ci_group; ++ci_local) {
                        size_t ci = ci_begin + ci_local;
                        for (size_t ky = 0; ky < d->kernel_height; ++ky) {
                            size_t padded_y = y * d->stride_height + ky * d->dilation_height;
                            if (padded_y < d->pad_height) continue;
                            size_t iy = padded_y - d->pad_height;
                            if (iy >= d->in_height) continue;
                            for (size_t kx = 0; kx < d->kernel_width; ++kx) {
                                size_t padded_x = x * d->stride_width + kx * d->dilation_width;
                                if (padded_x < d->pad_width) continue;
                                size_t ix = padded_x - d->pad_width;
                                if (ix >= d->in_width) continue;
                                size_t ii = ((n * d->in_channels + ci) * d->in_height + iy) * d->in_width + ix;
                                size_t wi = ((co * ci_group + ci_local) * d->kernel_height + ky) * d->kernel_width + kx;
                                sum += input[ii] * weight[wi];
                            }
                        }
                    }
                    output[((n * d->out_channels + co) * oh + y) * ow + x] = sum;
                }
            }
        }
    }
    return 1;
}

static void valid_output_range(size_t outputs, size_t input, size_t stride,
                               size_t pad, size_t kernel_offset,
                               size_t *begin, size_t *end) {
    size_t first = 0;
    if (kernel_offset < pad) {
        size_t delta = pad - kernel_offset;
        first = delta / stride + (delta % stride != 0);
    }
    size_t limit = pad + input;
    size_t last = 0;
    if (kernel_offset < limit) {
        size_t delta = limit - kernel_offset;
        last = delta / stride + (delta % stride != 0);
    }
    *begin = first < outputs ? first : outputs;
    *end = last < outputs ? last : outputs;
    if (*end < *begin) *end = *begin;
}

/* Spatial sweep is the better fallback for patch embedding, strided kernels,
 * and small feature maps: it amortizes setup across the complete output plane. */
static int conv2d_spatial_sweep(const float *restrict input,
                                const float *restrict weight,
                                const float *bias, float *restrict output,
                                const bf_conv2d_desc *d,
                                size_t oh, size_t ow) {
    const size_t ci_group = d->in_channels / d->groups;
    const size_t co_group = d->out_channels / d->groups;
    const size_t tasks = d->n * d->out_channels;
#if defined(BF_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t task = 0; task < tasks; ++task) {
        const size_t n = task / d->out_channels;
        const size_t co = task % d->out_channels;
        const size_t group = co / co_group;
        const size_t ci_begin = group * ci_group;
        float *destination = output + (n * d->out_channels + co) * oh * ow;
        const float initial = bias ? bias[co] : 0.0f;
        for (size_t i = 0; i < oh * ow; ++i) destination[i] = initial;
        for (size_t ci_local = 0; ci_local < ci_group; ++ci_local) {
            const size_t ci = ci_begin + ci_local;
            const float *source = input + (n * d->in_channels + ci) *
                                  d->in_height * d->in_width;
            for (size_t ky = 0; ky < d->kernel_height; ++ky) {
                size_t y_begin, y_end;
                const size_t y_offset = ky * d->dilation_height;
                valid_output_range(oh, d->in_height, d->stride_height,
                                   d->pad_height, y_offset, &y_begin, &y_end);
                for (size_t kx = 0; kx < d->kernel_width; ++kx) {
                    const float coefficient = weight[((co * ci_group + ci_local) *
                        d->kernel_height + ky) * d->kernel_width + kx];
                    size_t x_begin, x_end;
                    const size_t x_offset = kx * d->dilation_width;
                    valid_output_range(ow, d->in_width, d->stride_width,
                                       d->pad_width, x_offset, &x_begin, &x_end);
                    for (size_t y = y_begin; y < y_end; ++y) {
                        const size_t iy = y * d->stride_height + y_offset -
                                          d->pad_height;
                        const float *in_row = source + iy * d->in_width;
                        float *out_row = destination + y * ow;
#if defined(BF_WITH_OPENMP)
#pragma omp simd
#endif
                        for (size_t x = x_begin; x < x_end; ++x) {
                            const size_t ix = x * d->stride_width + x_offset -
                                              d->pad_width;
                            out_row[x] += in_row[ix] * coefficient;
                        }
                    }
                }
            }
        }
    }
    return 1;
}

int bf_conv2d_f32(const float *restrict input,
                  const float *restrict weight,
                  const float *bias, float *restrict output,
                  const bf_conv2d_desc *d) {
    size_t oh, ow;
    if (!input || !weight || !output ||
        !bf_conv2d_output_shape(d, &oh, &ow)) return 0;
    if (force_scalar()) return bf_conv2d_f32_ref(input, weight, bias, output, d);
#if defined(BF_WITH_ACCELERATE) || defined(BF_WITH_CBLAS)
    if (d->kernel_height == 1 && d->kernel_width == 1 &&
        d->stride_height == 1 && d->stride_width == 1 &&
        d->pad_height == 0 && d->pad_width == 0 && d->groups == 1 &&
        d->n <= INT_MAX && d->out_channels <= INT_MAX &&
        d->in_channels <= INT_MAX && oh <= SIZE_MAX / ow && oh * ow <= INT_MAX) {
        const size_t spatial = oh * ow;
        for (size_t n = 0; n < d->n; ++n) {
            float *destination = output + n * d->out_channels * spatial;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)d->out_channels, (int)spatial, (int)d->in_channels,
                        1.0f, weight, (int)d->in_channels,
                        input + n * d->in_channels * spatial, (int)spatial,
                        0.0f, destination, (int)spatial);
            if (bias)
                for (size_t co = 0; co < d->out_channels; ++co)
                    for (size_t i = 0; i < spatial; ++i)
                        destination[co * spatial + i] += bias[co];
        }
        return 1;
    }
#endif
    return conv2d_spatial_sweep(input, weight, bias, output, d, oh, ow);
}

static int transpose_axis(size_t input, size_t kernel, size_t stride,
                          size_t pad, size_t output_pad, size_t dilation,
                          size_t *output) {
    if (!input || !kernel || !stride || !dilation || output_pad >= stride ||
        kernel > (SIZE_MAX - 1) / dilation + 1) return 0;
    size_t effective = dilation * (kernel - 1) + 1;
    if (effective > SIZE_MAX - output_pad) return 0;
    size_t tail = effective + output_pad;
    if (input - 1 > (SIZE_MAX - tail) / stride) return 0;
    size_t expanded = (input - 1) * stride + tail;
    if (pad > expanded / 2 || expanded - 2 * pad == 0) return 0;
    *output = expanded - 2 * pad;
    return 1;
}

int bf_conv_transpose2d_output_shape(const bf_conv_transpose2d_desc *d,
                                     size_t *out_height, size_t *out_width) {
    if (!d || !out_height || !out_width || !d->n || !d->in_channels ||
        !d->out_channels || !d->groups || d->in_channels % d->groups ||
        d->out_channels % d->groups) return 0;
    return transpose_axis(d->in_height, d->kernel_height, d->stride_height,
                          d->pad_height, d->output_pad_height,
                          d->dilation_height, out_height) &&
           transpose_axis(d->in_width, d->kernel_width, d->stride_width,
                          d->pad_width, d->output_pad_width,
                          d->dilation_width, out_width);
}

int bf_conv_transpose2d_f32_ref(const float *input, const float *weight,
                                const float *bias, float *output,
                                const bf_conv_transpose2d_desc *d) {
    size_t oh, ow;
    if (!input || !weight || !output ||
        !bf_conv_transpose2d_output_shape(d, &oh, &ow)) return 0;
    if (oh > SIZE_MAX / ow) return 0;
    size_t output_spatial = oh * ow;
    if (d->n > SIZE_MAX / d->out_channels ||
        d->n * d->out_channels > SIZE_MAX / output_spatial) return 0;
    size_t output_count = d->n * d->out_channels * output_spatial;
    if (bias) {
        for (size_t n = 0; n < d->n; ++n)
            for (size_t co = 0; co < d->out_channels; ++co)
                for (size_t spatial = 0; spatial < output_spatial; ++spatial)
                    output[(n * d->out_channels + co) * output_spatial + spatial] = bias[co];
    } else {
        memset(output, 0, output_count * sizeof(*output));
    }
    size_t ci_group = d->in_channels / d->groups;
    size_t co_group = d->out_channels / d->groups;
    for (size_t n = 0; n < d->n; ++n)
        for (size_t ci = 0; ci < d->in_channels; ++ci) {
            size_t group = ci / ci_group;
            for (size_t iy = 0; iy < d->in_height; ++iy)
                for (size_t ix = 0; ix < d->in_width; ++ix) {
                    float value = input[((n * d->in_channels + ci) * d->in_height + iy) * d->in_width + ix];
                    for (size_t co_local = 0; co_local < co_group; ++co_local) {
                        size_t co = group * co_group + co_local;
                        for (size_t ky = 0; ky < d->kernel_height; ++ky) {
                            size_t py = iy * d->stride_height + ky * d->dilation_height;
                            if (py < d->pad_height) continue;
                            size_t oy = py - d->pad_height;
                            if (oy >= oh) continue;
                            for (size_t kx = 0; kx < d->kernel_width; ++kx) {
                                size_t px = ix * d->stride_width + kx * d->dilation_width;
                                if (px < d->pad_width) continue;
                                size_t ox = px - d->pad_width;
                                if (ox >= ow) continue;
                                size_t wi = ((ci * co_group + co_local) * d->kernel_height + ky) *
                                            d->kernel_width + kx;
                                output[((n * d->out_channels + co) * oh + oy) * ow + ox] += value * weight[wi];
                            }
                        }
                    }
                }
        }
    return 1;
}

void bf_linear_f32_ref(const float *input, const float *weight,
                       const float *bias, float *output,
                       size_t rows, size_t in_features, size_t out_features) {
    for (size_t row = 0; row < rows; ++row)
        for (size_t out = 0; out < out_features; ++out) {
            float sum = bias ? bias[out] : 0.0f;
            for (size_t in = 0; in < in_features; ++in)
                sum += input[row * in_features + in] * weight[out * in_features + in];
            output[row * out_features + out] = sum;
        }
}

void bf_linear_f32(const float *input, const float *weight,
                   const float *bias, float *output,
                   size_t rows, size_t in_features, size_t out_features) {
    if (force_scalar()) {
        bf_linear_f32_ref(input, weight, bias, output, rows, in_features, out_features);
        return;
    }
#if defined(BF_WITH_ACCELERATE) || defined(BF_WITH_CBLAS)
    if (rows <= INT_MAX && in_features <= INT_MAX && out_features <= INT_MAX) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)rows, (int)out_features, (int)in_features,
                    1.0f, input, (int)in_features, weight, (int)in_features,
                    0.0f, output, (int)out_features);
        if (bias)
            for (size_t row = 0; row < rows; ++row)
                for (size_t out = 0; out < out_features; ++out)
                    output[row * out_features + out] += bias[out];
        return;
    }
#endif
#if defined(BF_WITH_OPENMP)
    const size_t tasks = rows * out_features;
#pragma omp parallel for schedule(static)
    for (size_t task = 0; task < tasks; ++task) {
        const size_t row = task / out_features;
        const size_t out = task % out_features;
        float sum = bias ? bias[out] : 0.0f;
#pragma omp simd reduction(+:sum)
        for (size_t in = 0; in < in_features; ++in)
            sum += input[row * in_features + in] * weight[out * in_features + in];
        output[row * out_features + out] = sum;
    }
#else
    bf_linear_f32_ref(input, weight, bias, output, rows, in_features, out_features);
#endif
}

void bf_batch_norm_2d_f32_ref(const float *input, const float *scale,
                              const float *bias, const float *mean,
                              const float *variance, float epsilon,
                              float *output, size_t n, size_t channels,
                              size_t height, size_t width) {
    const size_t spatial = height * width;
    for (size_t batch = 0; batch < n; ++batch)
        for (size_t channel = 0; channel < channels; ++channel) {
            float factor = scale[channel] / sqrtf(variance[channel] + epsilon);
            float offset = bias[channel] - mean[channel] * factor;
            size_t base = (batch * channels + channel) * spatial;
            for (size_t i = 0; i < spatial; ++i)
                output[base + i] = input[base + i] * factor + offset;
        }
}

void bf_batch_norm_2d_f32(const float *input, const float *scale,
                          const float *bias, const float *mean,
                          const float *variance, float epsilon,
                          float *output, size_t n, size_t channels,
                          size_t height, size_t width) {
    if (force_scalar()) {
        bf_batch_norm_2d_f32_ref(input, scale, bias, mean, variance, epsilon,
                                 output, n, channels, height, width);
        return;
    }
    const size_t spatial = height * width;
    const size_t tasks = n * channels;
#if defined(BF_WITH_OPENMP)
#pragma omp parallel for schedule(static) if(tasks * spatial >= 4096)
#endif
    for (size_t task = 0; task < tasks; ++task) {
        const size_t channel = task % channels;
        const float factor = scale[channel] / sqrtf(variance[channel] + epsilon);
        const float offset = bias[channel] - mean[channel] * factor;
        const size_t base = task * spatial;
        for (size_t i = 0; i < spatial; ++i)
            output[base + i] = input[base + i] * factor + offset;
    }
}

void bf_layer_norm_f32_ref(const float *input, const float *scale,
                           const float *bias, float epsilon, float *output,
                           size_t rows, size_t channels) {
    for (size_t row = 0; row < rows; ++row) {
        const float *source = input + row * channels;
        float *destination = output + row * channels;
        double sum = 0.0;
        for (size_t i = 0; i < channels; ++i) sum += source[i];
        float mean = (float)(sum / (double)channels);
        double square_sum = 0.0;
        for (size_t i = 0; i < channels; ++i) {
            double centered = (double)source[i] - mean;
            square_sum += centered * centered;
        }
        float inverse_std = 1.0f / sqrtf((float)(square_sum / (double)channels) + epsilon);
        for (size_t i = 0; i < channels; ++i)
            destination[i] = (source[i] - mean) * inverse_std * scale[i] + bias[i];
    }
}

void bf_layer_norm_f32(const float *input, const float *scale,
                       const float *bias, float epsilon, float *output,
                       size_t rows, size_t channels) {
    if (force_scalar()) {
        bf_layer_norm_f32_ref(input, scale, bias, epsilon, output, rows, channels);
        return;
    }
#if defined(BF_WITH_OPENMP)
#pragma omp parallel for schedule(static) if(rows * channels >= 4096)
#endif
    for (size_t row = 0; row < rows; ++row) {
        const float *source = input + row * channels;
        float *destination = output + row * channels;
        double sum = 0.0;
        for (size_t i = 0; i < channels; ++i) sum += source[i];
        const float mean_value = (float)(sum / (double)channels);
        double square_sum = 0.0;
        for (size_t i = 0; i < channels; ++i) {
            const double centered = (double)source[i] - mean_value;
            square_sum += centered * centered;
        }
        const float inverse_std = 1.0f /
            sqrtf((float)(square_sum / (double)channels) + epsilon);
        for (size_t i = 0; i < channels; ++i)
            destination[i] = (source[i] - mean_value) * inverse_std * scale[i] + bias[i];
    }
}

void bf_softmax_f32_ref(const float *input, float *output,
                        size_t rows, size_t columns) {
    for (size_t row = 0; row < rows; ++row) {
        const float *source = input + row * columns;
        float *destination = output + row * columns;
        float maximum = -FLT_MAX;
        for (size_t i = 0; i < columns; ++i)
            if (source[i] > maximum) maximum = source[i];
        double sum = 0.0;
        for (size_t i = 0; i < columns; ++i) {
            destination[i] = expf(source[i] - maximum);
            sum += destination[i];
        }
        float inverse_sum = (float)(1.0 / sum);
        for (size_t i = 0; i < columns; ++i) destination[i] *= inverse_sum;
    }
}

void bf_gelu_f32_ref(const float *input, float *output, size_t count) {
    const float inverse_sqrt_two = 0.7071067811865475244f;
    for (size_t i = 0; i < count; ++i)
        output[i] = 0.5f * input[i] * (1.0f + erff(input[i] * inverse_sqrt_two));
}

void bf_gelu_f32(const float *input, float *output, size_t count) {
    if (force_scalar()) { bf_gelu_f32_ref(input, output, count); return; }
    const float inverse_sqrt_two = 0.7071067811865475244f;
#if defined(BF_WITH_OPENMP)
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
    for (size_t i = 0; i < count; ++i)
        output[i] = 0.5f * input[i] * (1.0f + erff(input[i] * inverse_sqrt_two));
}

void bf_relu_f32_ref(const float *input, float *output, size_t count) {
    for (size_t i = 0; i < count; ++i) output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

void bf_relu_f32(const float *input, float *output, size_t count) {
    if (force_scalar()) { bf_relu_f32_ref(input, output, count); return; }
#if defined(BF_WITH_OPENMP)
#pragma omp parallel for schedule(static) if(count >= 4096)
#endif
    for (size_t i = 0; i < count; ++i)
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

void bf_mean_vfe_f32_ref(const float *points, const int64_t *counts,
                         float *output, size_t voxels, size_t max_points,
                         size_t channels) {
    for (size_t voxel = 0; voxel < voxels; ++voxel) {
        size_t live = counts[voxel] <= 0 ? 0 : (size_t)counts[voxel];
        if (live > max_points) live = max_points;
        for (size_t channel = 0; channel < channels; ++channel) {
            float sum = 0.0f;
            for (size_t point = 0; point < live; ++point)
                sum += points[(voxel * max_points + point) * channels + channel];
            output[voxel * channels + channel] = live ? sum / (float)live : 0.0f;
        }
    }
}

int bf_topk_f32_ref(const float *input, float *values, int64_t *indices,
                    size_t rows, size_t columns, size_t k) {
    if (!input || !values || !indices || !rows || !columns || !k || k > columns) return 0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t rank = 0; rank < k; ++rank) {
            size_t best = SIZE_MAX;
            float best_value = -FLT_MAX;
            for (size_t column = 0; column < columns; ++column) {
                int used = 0;
                for (size_t prior = 0; prior < rank; ++prior)
                    used |= indices[row * k + prior] == (int64_t)column;
                float value = input[row * columns + column];
                if (!used && (best == SIZE_MAX || value > best_value)) {
                    best = column;
                    best_value = value;
                }
            }
            values[row * k + rank] = best_value;
            indices[row * k + rank] = (int64_t)best;
        }
    }
    return 1;
}

typedef struct {
    uint64_t key;
    size_t value;
} bf_coord_slot;

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

static int coord_key(bf_coord4 coord, size_t batches, size_t depth,
                     size_t height, size_t width, uint64_t *key) {
    if (coord.batch < 0 || coord.z < 0 || coord.y < 0 || coord.x < 0 ||
        (size_t)coord.batch >= batches || (size_t)coord.z >= depth ||
        (size_t)coord.y >= height || (size_t)coord.x >= width)
        return 0;
    uint64_t value = (uint64_t)coord.batch;
    if (depth && value > UINT64_MAX / depth) return 0;
    value = value * depth + (uint32_t)coord.z;
    if (height && value > UINT64_MAX / height) return 0;
    value = value * height + (uint32_t)coord.y;
    if (width && value > (UINT64_MAX - 1) / width) return 0;
    *key = value * width + (uint32_t)coord.x + 1;
    return 1;
}

static size_t hash_capacity(size_t count) {
    size_t capacity = 2;
    if (count > SIZE_MAX / 2) return 0;
    while (capacity < count * 2) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity <<= 1;
    }
    return capacity;
}

static int hash_insert(bf_coord_slot *slots, size_t capacity,
                       uint64_t key, size_t value) {
    size_t slot = (size_t)mix64(key) & (capacity - 1);
    while (slots[slot].key) {
        if (slots[slot].key == key) return 0;
        slot = (slot + 1) & (capacity - 1);
    }
    slots[slot].key = key;
    slots[slot].value = value;
    return 1;
}

static size_t hash_find(const bf_coord_slot *slots, size_t capacity,
                        uint64_t key) {
    size_t slot = (size_t)mix64(key) & (capacity - 1);
    while (slots[slot].key) {
        if (slots[slot].key == key) return slots[slot].value;
        slot = (slot + 1) & (capacity - 1);
    }
    return SIZE_MAX;
}

static int output_axis(size_t input, size_t kernel, size_t stride,
                       size_t pad, size_t dilation, size_t *output) {
    if (!input || !kernel || !stride || !dilation ||
        kernel > (SIZE_MAX - 1) / dilation + 1 ||
        pad > (SIZE_MAX - input) / 2)
        return 0;
    size_t effective = dilation * (kernel - 1) + 1;
    size_t padded = input + 2 * pad;
    if (effective > padded) return 0;
    *output = (padded - effective) / stride + 1;
    return *output != 0;
}

int bf_sparse_conv3d_output_shape(const bf_sparse_conv3d_desc *d,
                                  size_t *depth, size_t *height, size_t *width) {
    if (!d || !depth || !height || !width || !d->batches ||
        !d->in_channels || !d->out_channels || d->batches > INT32_MAX ||
        d->in_depth > INT32_MAX || d->in_height > INT32_MAX ||
        d->in_width > INT32_MAX || d->kernel_depth > INT64_MAX ||
        d->kernel_height > INT64_MAX || d->kernel_width > INT64_MAX ||
        d->stride_depth > INT64_MAX || d->stride_height > INT64_MAX ||
        d->stride_width > INT64_MAX || d->pad_depth > INT64_MAX ||
        d->pad_height > INT64_MAX || d->pad_width > INT64_MAX ||
        d->dilation_depth > INT64_MAX || d->dilation_height > INT64_MAX ||
        d->dilation_width > INT64_MAX) return 0;
    if (d->submanifold) {
        if (d->stride_depth != 1 || d->stride_height != 1 || d->stride_width != 1)
            return 0;
        *depth = d->in_depth;
        *height = d->in_height;
        *width = d->in_width;
        return *depth && *height && *width;
    }
    return output_axis(d->in_depth, d->kernel_depth, d->stride_depth,
                       d->pad_depth, d->dilation_depth, depth) &&
           output_axis(d->in_height, d->kernel_height, d->stride_height,
                       d->pad_height, d->dilation_height, height) &&
           output_axis(d->in_width, d->kernel_width, d->stride_width,
                       d->pad_width, d->dilation_width, width);
}

static int coord_compare(const void *left, const void *right) {
    const bf_coord4 *a = (const bf_coord4 *)left;
    const bf_coord4 *b = (const bf_coord4 *)right;
    if (a->batch != b->batch) return a->batch < b->batch ? -1 : 1;
    if (a->z != b->z) return a->z < b->z ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    return (a->x > b->x) - (a->x < b->x);
}

static int candidate_axis(int32_t input, size_t kernel_index, size_t pad,
                          size_t dilation, size_t stride, size_t limit,
                          int32_t *output) {
    if (pad > INT64_MAX || kernel_index > INT64_MAX || dilation > INT64_MAX ||
        stride > INT64_MAX || limit > INT32_MAX) return 0;
    int64_t numerator = (int64_t)input + (int64_t)pad -
                        (int64_t)kernel_index * (int64_t)dilation;
    if (numerator < 0 || numerator % (int64_t)stride) return 0;
    int64_t value = numerator / (int64_t)stride;
    if (value >= (int64_t)limit) return 0;
    *output = (int32_t)value;
    return 1;
}

size_t bf_sparse_conv3d_workspace_bytes(size_t input_capacity,
                                        size_t output_capacity) {
    if (!input_capacity || !output_capacity) return 0;
    size_t a = hash_capacity(input_capacity), b = hash_capacity(output_capacity);
    if (!a || !b || a > SIZE_MAX - b || a + b > SIZE_MAX / sizeof(bf_coord_slot))
        return 0;
    return (a + b) * sizeof(bf_coord_slot);
}

int bf_sparse_conv3d_f32_workspace_ref(const bf_coord4 *input_coords,
                             const float *input_features, size_t input_count,
                             const float *weight, const float *bias,
                             bf_coord4 *output_coords, float *output_features,
                             size_t output_capacity, size_t *output_count,
                             const bf_sparse_conv3d_desc *d,
                             void *workspace, size_t workspace_bytes) {
    if (output_count) *output_count = 0;
    size_t od, oh, ow;
    if (!input_coords || !input_features || !input_count || !weight ||
        !output_coords || !output_features || !output_count || !workspace ||
        !bf_sparse_conv3d_output_shape(d, &od, &oh, &ow)) return 0;
    size_t input_hash_cap = hash_capacity(input_count);
    size_t output_hash_cap = hash_capacity(output_capacity);
    size_t required = bf_sparse_conv3d_workspace_bytes(input_count, output_capacity);
    if (!input_hash_cap || !output_hash_cap || !required || workspace_bytes < required) return 0;
    bf_coord_slot *input_hash = workspace;
    bf_coord_slot *output_hash = input_hash + input_hash_cap;
    memset(workspace, 0, required);
    int ok = 1;
    for (size_t i = 0; i < input_count && ok; ++i) {
        uint64_t key;
        ok = coord_key(input_coords[i], d->batches, d->in_depth,
                       d->in_height, d->in_width, &key) &&
             hash_insert(input_hash, input_hash_cap, key, i);
    }
    size_t count = 0;
    if (ok && d->submanifold) {
        if (input_count > output_capacity) ok = 0;
        else {
            memcpy(output_coords, input_coords, input_count * sizeof(*output_coords));
            count = input_count;
        }
    } else if (ok) {
        for (size_t i = 0; i < input_count && ok; ++i)
            for (size_t kz = 0; kz < d->kernel_depth && ok; ++kz)
                for (size_t ky = 0; ky < d->kernel_height && ok; ++ky)
                    for (size_t kx = 0; kx < d->kernel_width && ok; ++kx) {
                        bf_coord4 candidate = {input_coords[i].batch, 0, 0, 0};
                        if (!candidate_axis(input_coords[i].z, kz, d->pad_depth,
                                            d->dilation_depth, d->stride_depth, od, &candidate.z) ||
                            !candidate_axis(input_coords[i].y, ky, d->pad_height,
                                            d->dilation_height, d->stride_height, oh, &candidate.y) ||
                            !candidate_axis(input_coords[i].x, kx, d->pad_width,
                                            d->dilation_width, d->stride_width, ow, &candidate.x))
                            continue;
                        uint64_t key;
                        if (!coord_key(candidate, d->batches, od, oh, ow, &key)) { ok = 0; break; }
                        if (hash_find(output_hash, output_hash_cap, key) == SIZE_MAX) {
                            if (count == output_capacity ||
                                !hash_insert(output_hash, output_hash_cap, key, count)) { ok = 0; break; }
                            output_coords[count++] = candidate;
                        }
                    }
    }
    if (ok) qsort(output_coords, count, sizeof(*output_coords), coord_compare);
    int compute_ok = ok;
#if defined(BF_WITH_OPENMP)
#pragma omp parallel for schedule(static) reduction(&:compute_ok)
#endif
    for (size_t out_index = 0; out_index < count; ++out_index) {
        bf_coord4 out_coord = output_coords[out_index];
        float *destination = output_features + out_index * d->out_channels;
        for (size_t co = 0; co < d->out_channels; ++co)
            destination[co] = bias ? bias[co] : 0.0f;
        for (size_t kz = 0; kz < d->kernel_depth; ++kz)
            for (size_t ky = 0; ky < d->kernel_height; ++ky)
                for (size_t kx = 0; kx < d->kernel_width; ++kx) {
                    int64_t iz = (int64_t)out_coord.z * (int64_t)d->stride_depth -
                                 (int64_t)d->pad_depth + (int64_t)kz * (int64_t)d->dilation_depth;
                    int64_t iy = (int64_t)out_coord.y * (int64_t)d->stride_height -
                                 (int64_t)d->pad_height + (int64_t)ky * (int64_t)d->dilation_height;
                    int64_t ix = (int64_t)out_coord.x * (int64_t)d->stride_width -
                                 (int64_t)d->pad_width + (int64_t)kx * (int64_t)d->dilation_width;
                    if (iz < 0 || iy < 0 || ix < 0 || iz >= (int64_t)d->in_depth ||
                        iy >= (int64_t)d->in_height || ix >= (int64_t)d->in_width) continue;
                    bf_coord4 source = {out_coord.batch, (int32_t)iz, (int32_t)iy, (int32_t)ix};
                    uint64_t key;
                    if (!coord_key(source, d->batches, d->in_depth,
                                   d->in_height, d->in_width, &key)) {
                        compute_ok = 0;
                        continue;
                    }
                    size_t in_index = hash_find(input_hash, input_hash_cap, key);
                    if (in_index == SIZE_MAX) continue;
                    const size_t kernel_index =
                        (kz * d->kernel_height + ky) * d->kernel_width + kx;
                    const float *source_features = input_features +
                        in_index * d->in_channels;
                    const float *kernel = weight + kernel_index *
                        d->in_channels * d->out_channels;
                    for (size_t co = 0; co < d->out_channels; ++co) {
                        float sum = destination[co];
#if defined(BF_WITH_OPENMP)
#pragma omp simd reduction(+:sum)
#endif
                        for (size_t ci = 0; ci < d->in_channels; ++ci)
                            sum += source_features[ci] *
                                   kernel[ci * d->out_channels + co];
                        destination[co] = sum;
                    }
                }
    }
    ok = ok && compute_ok;
    if (!ok) return 0;
    *output_count = count;
    return 1;
}

int bf_sparse_conv3d_f32_ref(const bf_coord4 *input_coords,
                             const float *input_features, size_t input_count,
                             const float *weight, const float *bias,
                             bf_coord4 *output_coords, float *output_features,
                             size_t output_capacity, size_t *output_count,
                             const bf_sparse_conv3d_desc *d) {
    size_t bytes = bf_sparse_conv3d_workspace_bytes(input_count, output_capacity);
    void *workspace = bytes ? malloc(bytes) : NULL;
    int ok = workspace && bf_sparse_conv3d_f32_workspace_ref(input_coords,
        input_features, input_count, weight, bias, output_coords,
        output_features, output_capacity, output_count, d, workspace, bytes);
    free(workspace);
    return ok;
}
