#include "bf_cuda.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>

struct bf_cuda_lss_plan {
    bf_lss_desc desc;
    size_t samples;
    size_t pixels;
    size_t cell_count;
    size_t cells_per_batch;
    size_t resident_bytes;
    int prepared;
    int *keys_in;
    int *keys_out;
    unsigned *values_in;
    unsigned *values_out;
    int *starts;
    int *ends;
    float *probabilities;
    float *context_nhwc;
    float *calibration;
    void *sort_scratch;
    size_t sort_scratch_bytes;
    int sort_end_bit;
};

static int fail(char *error, size_t cap, const char *operation,
                cudaError_t status) {
    if (error && cap)
        std::snprintf(error, cap, "%s: %s", operation, cudaGetErrorString(status));
    return 0;
}

static bool multiply_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool lss_contract(const bf_lss_desc *d) {
    return d && d->batches && d->cameras && d->depth_bins &&
        d->feature_height && d->feature_width && d->channels &&
        d->depth_bins <= 128 && d->batches <= INT32_MAX &&
        d->cameras <= INT32_MAX && d->feature_height <= INT32_MAX &&
        d->feature_width <= INT32_MAX && d->channels <= INT32_MAX &&
        d->grid_cells[0] && d->grid_cells[1] && d->grid_cells[2] &&
        d->grid_cells[0] <= INT32_MAX && d->grid_cells[1] <= INT32_MAX &&
        d->grid_cells[2] <= INT32_MAX && isfinite(d->grid_step[0]) &&
        isfinite(d->grid_step[1]) && isfinite(d->grid_step[2]) &&
        d->grid_step[0] > 0.0f && d->grid_step[1] > 0.0f &&
        d->grid_step[2] > 0.0f;
}

static bool lss_sizes(const bf_lss_desc *d, size_t *samples,
                      size_t *blocks, size_t *bev_bytes) {
    size_t bev_count;
    return multiply_size(d->batches, d->cameras, samples) &&
        multiply_size(*samples, d->depth_bins, samples) &&
        multiply_size(*samples, d->feature_height, samples) &&
        multiply_size(*samples, d->feature_width, samples) &&
        multiply_size(d->batches, d->cameras, blocks) &&
        multiply_size(*blocks, d->feature_height, blocks) &&
        multiply_size(*blocks, d->feature_width, blocks) &&
        *blocks <= UINT_MAX &&
        multiply_size(d->batches, d->channels, &bev_count) &&
        multiply_size(bev_count, d->grid_cells[2], &bev_count) &&
        multiply_size(bev_count, d->grid_cells[0], &bev_count) &&
        multiply_size(bev_count, d->grid_cells[1], &bev_count) &&
        multiply_size(bev_count, sizeof(float), bev_bytes);
}

extern "C" int bf_cuda_available(char *error, size_t cap) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) return fail(error, cap, "CUDA device query", status);
    if (!count) {
        if (error && cap) std::snprintf(error, cap, "no CUDA device available");
        return 0;
    }
    return 1;
}

__global__ static void lift_pool_kernel(
    const float *__restrict__ logits,
    const float *__restrict__ context,
    const float *__restrict__ geometry,
    float *__restrict__ bev,
    int cameras, int depths, int height, int width, int channels,
    float minimum_x, float minimum_y, float minimum_z,
    float step_x, float step_y, float step_z,
    int cells_x, int cells_y, int cells_z) {
    int pixel = blockIdx.x;
    int spatial = height * width;
    int camera_pixels = cameras * spatial;
    int batch = pixel / camera_pixels;
    int within_batch = pixel - batch * camera_pixels;
    int camera = within_batch / spatial;
    int position = within_batch - camera * spatial;
    int y = position / width, x = position - y * width;
    int camera_index = batch * cameras + camera;
    extern __shared__ float shared[];
    float *probabilities = shared;
    float *reduction = probabilities + depths;
    float local_maximum = -FLT_MAX;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x) {
        float value = logits[(camera_index * depths + depth) * spatial + position];
        local_maximum = fmaxf(local_maximum, value);
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                            reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    float maximum = reduction[0], local_sum = 0.0f;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        local_sum += expf(logits[(camera_index * depths + depth) * spatial + position] - maximum);
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse_sum = 1.0f / reduction[0];
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        probabilities[depth] = expf(
            logits[(camera_index * depths + depth) * spatial + position] - maximum) * inverse_sum;
    __syncthreads();
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
        float feature = context[(camera_index * channels + channel) * spatial + position];
        for (int depth = 0; depth < depths; ++depth) {
            size_t sample = (((size_t)camera_index * depths + depth) * height + y) * width + x;
            const float *point = geometry + sample * 3;
            float gx = (point[0] - minimum_x) / step_x;
            float gy = (point[1] - minimum_y) / step_y;
            float gz = (point[2] - minimum_z) / step_z;
            if (!isfinite(gx) || !isfinite(gy) || !isfinite(gz)) continue;
            int cell_x = (int)gx, cell_y = (int)gy, cell_z = (int)gz;
            if ((unsigned)cell_x >= (unsigned)cells_x ||
                (unsigned)cell_y >= (unsigned)cells_y ||
                (unsigned)cell_z >= (unsigned)cells_z) continue;
            size_t collapsed = (size_t)channel * cells_z + cell_z;
            size_t output = (((size_t)batch * channels * cells_z + collapsed) *
                              cells_x + cell_x) * cells_y + cell_y;
            atomicAdd(bev + output, probabilities[depth] * feature);
        }
    }
}

__global__ static void lift_pool_cached_geometry_kernel(
    const float *__restrict__ logits,
    const float *__restrict__ context,
    const float *__restrict__ geometry,
    float *__restrict__ bev,
    int cameras, int depths, int height, int width, int channels,
    float minimum_x, float minimum_y, float minimum_z,
    float step_x, float step_y, float step_z,
    int cells_x, int cells_y, int cells_z) {
    int pixel = blockIdx.x;
    int spatial = height * width;
    int camera_pixels = cameras * spatial;
    int batch = pixel / camera_pixels;
    int within_batch = pixel - batch * camera_pixels;
    int camera = within_batch / spatial;
    int position = within_batch - camera * spatial;
    int y = position / width, x = position - y * width;
    int camera_index = batch * cameras + camera;
    extern __shared__ float shared[];
    float *probabilities = shared;
    float *reduction = probabilities + depths;
    int *rank = reinterpret_cast<int *>(reduction + blockDim.x);
    float local_maximum = -FLT_MAX;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        local_maximum = fmaxf(local_maximum,
            logits[(camera_index * depths + depth) * spatial + position]);
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    float maximum = reduction[0], local_sum = 0.0f;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        local_sum += expf(logits[(camera_index * depths + depth) * spatial + position] - maximum);
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse_sum = 1.0f / reduction[0];
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        probabilities[depth] = expf(
            logits[(camera_index * depths + depth) * spatial + position] - maximum) * inverse_sum;
    __syncthreads();
    float feature = threadIdx.x < channels
        ? context[(camera_index * channels + threadIdx.x) * spatial + position] : 0.0f;
    for (int depth = 0; depth < depths; ++depth) {
        if (threadIdx.x == 0) {
            size_t sample = (((size_t)camera_index * depths + depth) * height + y) * width + x;
            const float *point = geometry + sample * 3;
            float gx = (point[0] - minimum_x) / step_x;
            float gy = (point[1] - minimum_y) / step_y;
            float gz = (point[2] - minimum_z) / step_z;
            int cx = (int)gx, cy = (int)gy, cz = (int)gz;
            *rank = isfinite(gx) && isfinite(gy) && isfinite(gz) &&
                    (unsigned)cx < (unsigned)cells_x &&
                    (unsigned)cy < (unsigned)cells_y &&
                    (unsigned)cz < (unsigned)cells_z
                ? (cz * cells_x + cx) * cells_y + cy : -1;
        }
        __syncthreads();
        if (threadIdx.x < channels && *rank >= 0) {
            int cell_z = *rank / (cells_x * cells_y);
            int xy = *rank - cell_z * cells_x * cells_y;
            int cell_x = xy / cells_y, cell_y = xy - cell_x * cells_y;
            size_t collapsed = (size_t)threadIdx.x * cells_z + cell_z;
            size_t output = (((size_t)batch * channels * cells_z + collapsed) *
                              cells_x + cell_x) * cells_y + cell_y;
            atomicAdd(bev + output, probabilities[depth] * feature);
        }
        __syncthreads();
    }
}

__global__ static void geometry_to_ranks_kernel(
    const float *__restrict__ geometry, int *__restrict__ ranks,
    size_t samples, float minimum_x, float minimum_y, float minimum_z,
    float step_x, float step_y, float step_z,
    int cells_x, int cells_y, int cells_z) {
    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    const float *point = geometry + sample * 3;
    float gx = (point[0] - minimum_x) / step_x;
    float gy = (point[1] - minimum_y) / step_y;
    float gz = (point[2] - minimum_z) / step_z;
    int cx = (int)gx, cy = (int)gy, cz = (int)gz;
    ranks[sample] = isfinite(gx) && isfinite(gy) && isfinite(gz) &&
        (unsigned)cx < (unsigned)cells_x &&
        (unsigned)cy < (unsigned)cells_y &&
        (unsigned)cz < (unsigned)cells_z
        ? (cz * cells_x + cx) * cells_y + cy : -1;
}

__global__ static void lift_pool_ranks_kernel(
    const float *__restrict__ logits,
    const float *__restrict__ context,
    const int *__restrict__ ranks,
    float *__restrict__ bev,
    int cameras, int depths, int height, int width, int channels,
    int cells_x, int cells_y, int cells_z) {
    int pixel = blockIdx.x;
    int spatial = height * width;
    int camera_pixels = cameras * spatial;
    int batch = pixel / camera_pixels;
    int within_batch = pixel - batch * camera_pixels;
    int camera = within_batch / spatial;
    int position = within_batch - camera * spatial;
    int y = position / width, x = position - y * width;
    int camera_index = batch * cameras + camera;
    extern __shared__ float shared[];
    float *probabilities = shared;
    float *reduction = probabilities + depths;
    int *cell_ranks = reinterpret_cast<int *>(reduction + blockDim.x);
    float local_maximum = -FLT_MAX;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x) {
        local_maximum = fmaxf(local_maximum,
            logits[(camera_index * depths + depth) * spatial + position]);
        size_t sample = (((size_t)camera_index * depths + depth) * height + y) * width + x;
        cell_ranks[depth] = ranks[sample];
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                            reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    float maximum = reduction[0], local_sum = 0.0f;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        local_sum += expf(logits[(camera_index * depths + depth) * spatial + position] - maximum);
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse_sum = 1.0f / reduction[0];
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        probabilities[depth] = expf(
            logits[(camera_index * depths + depth) * spatial + position] - maximum) * inverse_sum;
    __syncthreads();
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
        float feature = context[(camera_index * channels + channel) * spatial + position];
        for (int depth = 0; depth < depths; ++depth) {
            int rank = cell_ranks[depth];
            if (rank < 0) continue;
            size_t cell_count = (size_t)cells_z * cells_x * cells_y;
            size_t output = ((size_t)batch * channels + channel) * cell_count +
                            (unsigned)rank;
            atomicAdd(bev + output, probabilities[depth] * feature);
        }
    }
}

extern "C" int bf_cuda_lss_geometry_to_ranks_f32(
    const float *geometry, int *ranks, const bf_lss_desc *d,
    void *stream_value, char *error, size_t cap) {
    if (!geometry || !ranks || !lss_contract(d)) {
        if (error && cap) std::snprintf(error, cap, "invalid CUDA LSS rank contract");
        return 0;
    }
    size_t samples, blocks, bev_bytes;
    if (!lss_sizes(d, &samples, &blocks, &bev_bytes) ||
        samples > (size_t)UINT_MAX * 256) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS rank size overflow");
        return 0;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    geometry_to_ranks_kernel<<<(unsigned)((samples + 255) / 256), 256, 0, stream>>>(
        geometry, ranks, samples,
        d->grid_minimum[0], d->grid_minimum[1], d->grid_minimum[2],
        d->grid_step[0], d->grid_step[1], d->grid_step[2],
        (int)d->grid_cells[0], (int)d->grid_cells[1], (int)d->grid_cells[2]);
    cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? 1 : fail(error, cap, "CUDA LSS rank launch", status);
}

extern "C" int bf_cuda_lss_lift_pool_ranks_f32(
    const float *logits, const float *context, const int *ranks,
    float *bev, const bf_lss_desc *d, void *stream_value,
    char *error, size_t cap) {
    if (!logits || !context || !ranks || !bev || !lss_contract(d)) {
        if (error && cap) std::snprintf(error, cap, "invalid CUDA LSS ranked contract");
        return 0;
    }
    size_t samples, blocks, bev_bytes;
    if (!lss_sizes(d, &samples, &blocks, &bev_bytes)) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS ranked size overflow");
        return 0;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    cudaError_t status = cudaMemsetAsync(bev, 0, bev_bytes, stream);
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS ranked zero", status);
    size_t shared_bytes = (d->depth_bins + 128) * sizeof(float) +
                          d->depth_bins * sizeof(int);
    lift_pool_ranks_kernel<<<(unsigned)blocks, 128, shared_bytes, stream>>>(
        logits, context, ranks, bev, (int)d->cameras, (int)d->depth_bins,
        (int)d->feature_height, (int)d->feature_width, (int)d->channels,
        (int)d->grid_cells[0], (int)d->grid_cells[1], (int)d->grid_cells[2]);
    status = cudaGetLastError();
    return status == cudaSuccess ? 1 : fail(error, cap, "CUDA LSS ranked launch", status);
}

__global__ static void initialize_sort_values_kernel(
    int *__restrict__ keys, unsigned *__restrict__ values,
    size_t samples, int depths, int spatial, int channels,
    int samples_per_batch, int cells_per_batch, int invalid_key) {
    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    int key = keys[sample];
    keys[sample] = key < 0 ? invalid_key
        : key + (int)(sample / samples_per_batch) * cells_per_batch;
    size_t q = sample / spatial;
    int position = (int)(sample - q * spatial);
    int camera_index = (int)(q / depths);
    int depth = (int)(q - (size_t)camera_index * depths);
    values[sample] = ((unsigned)camera_index << 19) |
                     ((unsigned)depth << 12) | (unsigned)position;
}

__global__ static void build_intervals_kernel(
    const int *__restrict__ keys, int *__restrict__ starts,
    int *__restrict__ ends, size_t samples, int cells) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= samples) return;
    int key = keys[i];
    if ((unsigned)key >= (unsigned)cells) return;
    if (i == 0 || keys[i - 1] != key) starts[key] = (int)i;
    if (i + 1 == samples || keys[i + 1] != key) ends[key] = (int)(i + 1);
}

__global__ static void depth_softmax_kernel(
    const float *__restrict__ logits, float *__restrict__ probabilities,
    int depths, int spatial) {
    int pixel = blockIdx.x;
    int camera_index = pixel / spatial;
    int position = pixel - camera_index * spatial;
    __shared__ float reduction[128];
    float maximum = -FLT_MAX;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        maximum = fmaxf(maximum,
            logits[(camera_index * depths + depth) * spatial + position]);
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x],
                                            reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    float sum = 0.0f;
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x)
        sum += expf(logits[(camera_index * depths + depth) * spatial + position] - maximum);
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = 1.0f / reduction[0];
    for (int depth = threadIdx.x; depth < depths; depth += blockDim.x) {
        size_t sample = ((size_t)camera_index * depths + depth) * spatial + position;
        probabilities[sample] = expf(logits[sample] - maximum) * inverse;
    }
}

__global__ static void interval_pool_kernel(
    const float *__restrict__ probabilities,
    const float *__restrict__ context,
    const unsigned *__restrict__ sorted_values,
    const int *__restrict__ starts, const int *__restrict__ ends,
    float *__restrict__ bev, int cells, int channels, int depths, int spatial) {
    int output_cell = blockIdx.x;
    int batch = output_cell / cells;
    int cell = output_cell - batch * cells;
    int begin = starts[output_cell], end = ends[output_cell];
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
        float sum = 0.0f;
        if (begin >= 0) {
            for (int i = begin; i < end; ++i) {
                unsigned packed = sorted_values[i];
                unsigned position = packed & 4095u;
                unsigned depth = (packed >> 12) & 127u;
                unsigned camera_index = packed >> 19;
                size_t sample = ((size_t)camera_index * depths + depth) * spatial + position;
                size_t context_index = ((size_t)camera_index * channels + channel) * spatial + position;
                sum += probabilities[sample] * context[context_index];
            }
        }
        bev[((size_t)batch * channels + channel) * cells + cell] = sum;
    }
}

__global__ static void context_nchw_to_nhwc_kernel(
    const float *__restrict__ input, float *__restrict__ output,
    size_t count, int channels, int spatial) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int channel = (int)(i % channels);
    size_t pixel = i / channels;
    int camera = (int)(pixel / spatial);
    int position = (int)(pixel - (size_t)camera * spatial);
    output[i] = input[((size_t)camera * channels + channel) * spatial + position];
}

__global__ static void interval_pool_nhwc_kernel(
    const float *__restrict__ probabilities,
    const float *__restrict__ context,
    const unsigned *__restrict__ sorted_values,
    const int *__restrict__ starts, const int *__restrict__ ends,
    float *__restrict__ bev, int cells, int channels, int depths, int spatial) {
    int output_cell = blockIdx.x;
    int batch = output_cell / cells;
    int cell = output_cell - batch * cells;
    int begin = starts[output_cell], end = ends[output_cell];
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
        float sum = 0.0f;
        if (begin >= 0) {
            for (int i = begin; i < end; ++i) {
                unsigned packed = sorted_values[i];
                unsigned position = packed & 4095u;
                unsigned depth = (packed >> 12) & 127u;
                unsigned camera_index = packed >> 19;
                size_t sample = ((size_t)camera_index * depths + depth) * spatial + position;
                size_t feature = (size_t)camera_index * spatial + position;
                sum += probabilities[sample] * context[feature * channels + channel];
            }
        }
        bev[((size_t)batch * channels + channel) * cells + cell] = sum;
    }
}

__device__ static bool inverse_3x3_device(const float *m, float *out) {
    float c00 = m[4] * m[8] - m[5] * m[7];
    float c01 = m[2] * m[7] - m[1] * m[8];
    float c02 = m[1] * m[5] - m[2] * m[4];
    float c10 = m[5] * m[6] - m[3] * m[8];
    float c11 = m[0] * m[8] - m[2] * m[6];
    float c12 = m[2] * m[3] - m[0] * m[5];
    float c20 = m[3] * m[7] - m[4] * m[6];
    float c21 = m[1] * m[6] - m[0] * m[7];
    float c22 = m[0] * m[4] - m[1] * m[3];
    float determinant = m[0] * c00 + m[1] * c10 + m[2] * c20;
    if (!isfinite(determinant) || fabsf(determinant) <= FLT_MIN) return false;
    float inverse = 1.0f / determinant;
    out[0] = c00 * inverse; out[1] = c01 * inverse; out[2] = c02 * inverse;
    out[3] = c10 * inverse; out[4] = c11 * inverse; out[5] = c12 * inverse;
    out[6] = c20 * inverse; out[7] = c21 * inverse; out[8] = c22 * inverse;
    return true;
}

__global__ static void prepare_calibration_kernel(
    const float *__restrict__ camera_rotation,
    const float *__restrict__ camera_translation,
    const float *__restrict__ intrinsics,
    const float *__restrict__ post_rotation,
    const float *__restrict__ post_translation,
    const float *__restrict__ extra_rotation,
    const float *__restrict__ extra_translation,
    float *__restrict__ calibration, int cameras, int camera_count) {
    int camera_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (camera_index >= camera_count) return;
    int batch = camera_index / cameras;
    float *out = calibration + camera_index * 37;
    float inverse_intrinsic[9];
    bool valid_post = inverse_3x3_device(post_rotation + camera_index * 9, out);
    bool valid_intrinsic = inverse_3x3_device(intrinsics + camera_index * 9,
                                               inverse_intrinsic);
    bool valid = valid_post && valid_intrinsic;
    const float *rotation = camera_rotation + camera_index * 9;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k)
                sum += rotation[row * 3 + k] * inverse_intrinsic[k * 3 + column];
            out[9 + row * 3 + column] = sum;
        }
    for (int i = 0; i < 3; ++i) {
        out[18 + i] = camera_translation[camera_index * 3 + i];
        out[21 + i] = post_translation[camera_index * 3 + i];
        out[33 + i] = extra_translation ? extra_translation[batch * 3 + i] : 0.0f;
    }
    for (int i = 0; i < 9; ++i)
        out[24 + i] = extra_rotation ? extra_rotation[batch * 9 + i]
                                      : (i % 4 == 0 ? 1.0f : 0.0f);
    out[36] = valid ? 1.0f : 0.0f;
}

__device__ static void matrix_vector_device(const float *matrix,
                                             const float *vector, float *out) {
    for (int row = 0; row < 3; ++row)
        out[row] = matrix[row * 3] * vector[0] +
                   matrix[row * 3 + 1] * vector[1] +
                   matrix[row * 3 + 2] * vector[2];
}

__global__ static void calibration_to_ranks_kernel(
    const float *__restrict__ frustum,
    const float *__restrict__ calibration,
    int *__restrict__ ranks, size_t samples,
    int depths, int height, int width,
    float minimum_x, float minimum_y, float minimum_z,
    float step_x, float step_y, float step_z,
    int cells_x, int cells_y, int cells_z) {
    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    int spatial = height * width;
    size_t q = sample / spatial;
    int position = (int)(sample - q * spatial);
    int depth = (int)(q % depths);
    int camera_index = (int)(q / depths);
    const float *coefficients = calibration + camera_index * 37;
    if (coefficients[36] == 0.0f) { ranks[sample] = -1; return; }
    const float *source = frustum + ((size_t)depth * spatial + position) * 3;
    float point[3] = {source[0] - coefficients[21],
                      source[1] - coefficients[22],
                      source[2] - coefficients[23]};
    float undone[3], projected[3], lidar[3], augmented[3];
    matrix_vector_device(coefficients, point, undone);
    projected[0] = undone[0] * undone[2];
    projected[1] = undone[1] * undone[2];
    projected[2] = undone[2];
    matrix_vector_device(coefficients + 9, projected, lidar);
    for (int i = 0; i < 3; ++i) lidar[i] += coefficients[18 + i];
    matrix_vector_device(coefficients + 24, lidar, augmented);
    for (int i = 0; i < 3; ++i) augmented[i] += coefficients[33 + i];
    float gx = (augmented[0] - minimum_x) / step_x;
    float gy = (augmented[1] - minimum_y) / step_y;
    float gz = (augmented[2] - minimum_z) / step_z;
    int cx = (int)gx, cy = (int)gy, cz = (int)gz;
    ranks[sample] = isfinite(gx) && isfinite(gy) && isfinite(gz) &&
        (unsigned)cx < (unsigned)cells_x &&
        (unsigned)cy < (unsigned)cells_y &&
        (unsigned)cz < (unsigned)cells_z
        ? (cz * cells_x + cx) * cells_y + cy : -1;
}

extern "C" void bf_cuda_lss_plan_destroy(bf_cuda_lss_plan *plan) {
    if (!plan) return;
    cudaFree(plan->sort_scratch);
    cudaFree(plan->calibration);
    cudaFree(plan->context_nhwc);
    cudaFree(plan->probabilities);
    cudaFree(plan->ends);
    cudaFree(plan->starts);
    cudaFree(plan->values_out);
    cudaFree(plan->values_in);
    cudaFree(plan->keys_out);
    cudaFree(plan->keys_in);
    std::free(plan);
}

extern "C" int bf_cuda_lss_plan_create(
    const bf_lss_desc *d, bf_cuda_lss_plan **out, char *error, size_t cap) {
    if (out) *out = nullptr;
    if (!out || !lss_contract(d)) {
        if (error && cap) std::snprintf(error, cap, "invalid CUDA LSS plan contract");
        return 0;
    }
    size_t samples, pixels, bev_bytes;
    if (!lss_sizes(d, &samples, &pixels, &bev_bytes) || samples > INT_MAX ||
        d->feature_height * d->feature_width > 4096 ||
        d->batches * d->cameras > 8192) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS plan size overflow");
        return 0;
    }
    size_t cells = d->grid_cells[0] * d->grid_cells[1];
    if (!multiply_size(cells, d->grid_cells[2], &cells) || cells > INT_MAX) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS plan cell overflow");
        return 0;
    }
    bf_cuda_lss_plan *plan = (bf_cuda_lss_plan *)std::calloc(1, sizeof(*plan));
    if (!plan) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS plan host allocation failed");
        return 0;
    }
    size_t output_cells;
    if (!multiply_size(cells, d->batches, &output_cells) || output_cells > INT_MAX) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS plan output-cell overflow");
        std::free(plan); return 0;
    }
    plan->desc = *d; plan->samples = samples; plan->pixels = pixels;
    plan->cells_per_batch = cells; plan->cell_count = output_cells;
    plan->sort_end_bit = 1;
    while (((size_t)1 << plan->sort_end_bit) <= output_cells)
        ++plan->sort_end_bit;
    cudaError_t status = cudaMalloc(&plan->keys_in, samples * sizeof(int));
    if (status == cudaSuccess) status = cudaMalloc(&plan->keys_out, samples * sizeof(int));
    if (status == cudaSuccess) status = cudaMalloc(&plan->values_in, samples * sizeof(unsigned));
    if (status == cudaSuccess) status = cudaMalloc(&plan->values_out, samples * sizeof(unsigned));
    if (status == cudaSuccess) status = cudaMalloc(&plan->starts, output_cells * sizeof(int));
    if (status == cudaSuccess) status = cudaMalloc(&plan->ends, output_cells * sizeof(int));
    if (status == cudaSuccess) status = cudaMalloc(&plan->probabilities, samples * sizeof(float));
    size_t context_count = d->batches * d->cameras * d->channels *
                           d->feature_height * d->feature_width;
    if (status == cudaSuccess) status = cudaMalloc(&plan->context_nhwc,
                                                    context_count * sizeof(float));
    if (status == cudaSuccess) status = cudaMalloc(&plan->calibration,
                                                    d->batches * d->cameras * 37 * sizeof(float));
    if (status == cudaSuccess)
        status = cub::DeviceRadixSort::SortPairs(nullptr, plan->sort_scratch_bytes,
            plan->keys_in, plan->keys_out, plan->values_in, plan->values_out,
            (int)samples, 0, plan->sort_end_bit);
    if (status == cudaSuccess)
        status = cudaMalloc(&plan->sort_scratch, plan->sort_scratch_bytes);
    if (status != cudaSuccess) {
        fail(error, cap, "CUDA LSS plan allocation", status);
        bf_cuda_lss_plan_destroy(plan); return 0;
    }
    plan->resident_bytes = samples * (2 * sizeof(int) +
        2 * sizeof(unsigned) + sizeof(float)) +
        output_cells * 2 * sizeof(int) + plan->sort_scratch_bytes;
    plan->resident_bytes += d->batches * d->cameras * 37 * sizeof(float);
    plan->resident_bytes += context_count * sizeof(float);
    *out = plan;
    return 1;
}

extern "C" size_t bf_cuda_lss_plan_resident_bytes(const bf_cuda_lss_plan *plan) {
    return plan ? plan->resident_bytes : 0;
}

static int plan_sort_keys(bf_cuda_lss_plan *plan, cudaStream_t stream,
                          char *error, size_t cap) {
    const bf_lss_desc *d = &plan->desc;
    initialize_sort_values_kernel<<<(unsigned)((plan->samples + 255) / 256), 256, 0, stream>>>(
        plan->keys_in, plan->values_in, plan->samples, (int)d->depth_bins,
        (int)(d->feature_height * d->feature_width), (int)d->channels,
        (int)(d->cameras * d->depth_bins * d->feature_height * d->feature_width),
        (int)plan->cells_per_batch, (int)plan->cell_count);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS plan pair launch", status);
    status = cub::DeviceRadixSort::SortPairs(
        plan->sort_scratch, plan->sort_scratch_bytes,
        plan->keys_in, plan->keys_out, plan->values_in, plan->values_out,
        (int)plan->samples, 0, plan->sort_end_bit, stream);
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS plan radix sort", status);
    status = cudaMemsetAsync(plan->starts, 0xff,
                             plan->cell_count * sizeof(int), stream);
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS plan start reset", status);
    status = cudaMemsetAsync(plan->ends, 0xff,
                             plan->cell_count * sizeof(int), stream);
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS plan end reset", status);
    build_intervals_kernel<<<(unsigned)((plan->samples + 255) / 256), 256, 0, stream>>>(
        plan->keys_out, plan->starts, plan->ends, plan->samples,
        (int)plan->cell_count);
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS plan interval launch", status);
    plan->prepared = 1;
    return 1;
}

extern "C" int bf_cuda_lss_plan_prepare_geometry_f32(
    bf_cuda_lss_plan *plan, const float *geometry, void *stream_value,
    char *error, size_t cap) {
    if (!plan || !geometry) {
        if (error && cap) std::snprintf(error, cap, "invalid CUDA LSS plan geometry");
        return 0;
    }
    const bf_lss_desc *d = &plan->desc;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    geometry_to_ranks_kernel<<<(unsigned)((plan->samples + 255) / 256), 256, 0, stream>>>(
        geometry, plan->keys_in, plan->samples,
        d->grid_minimum[0], d->grid_minimum[1], d->grid_minimum[2],
        d->grid_step[0], d->grid_step[1], d->grid_step[2],
        (int)d->grid_cells[0], (int)d->grid_cells[1], (int)d->grid_cells[2]);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS plan rank launch", status);
    return plan_sort_keys(plan, stream, error, cap);
}

extern "C" int bf_cuda_lss_plan_prepare_calibration_f32(
    bf_cuda_lss_plan *plan, const float *frustum,
    const float *camera_rotation, const float *camera_translation,
    const float *intrinsics, const float *post_rotation,
    const float *post_translation, const float *extra_rotation,
    const float *extra_translation, void *stream_value,
    char *error, size_t cap) {
    if (!plan || !frustum || !camera_rotation || !camera_translation ||
        !intrinsics || !post_rotation || !post_translation) {
        if (error && cap) std::snprintf(error, cap, "invalid CUDA LSS calibration contract");
        return 0;
    }
    const bf_lss_desc *d = &plan->desc;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    int camera_count = (int)(d->batches * d->cameras);
    prepare_calibration_kernel<<<(unsigned)((camera_count + 31) / 32), 32, 0, stream>>>(
        camera_rotation, camera_translation, intrinsics, post_rotation,
        post_translation, extra_rotation, extra_translation, plan->calibration,
        (int)d->cameras, camera_count);
    calibration_to_ranks_kernel<<<(unsigned)((plan->samples + 255) / 256), 256, 0, stream>>>(
        frustum, plan->calibration, plan->keys_in, plan->samples,
        (int)d->depth_bins, (int)d->feature_height, (int)d->feature_width,
        d->grid_minimum[0], d->grid_minimum[1], d->grid_minimum[2],
        d->grid_step[0], d->grid_step[1], d->grid_step[2],
        (int)d->grid_cells[0], (int)d->grid_cells[1], (int)d->grid_cells[2]);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, cap, "CUDA LSS calibration rank launch", status);
    return plan_sort_keys(plan, stream, error, cap);
}

extern "C" int bf_cuda_lss_plan_forward_f32(
    bf_cuda_lss_plan *plan, const float *logits, const float *context,
    float *bev, void *stream_value, char *error, size_t cap) {
    if (!plan || !plan->prepared || !logits || !context || !bev) {
        if (error && cap) std::snprintf(error, cap, "invalid or unprepared CUDA LSS plan");
        return 0;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    int spatial = (int)(plan->desc.feature_height * plan->desc.feature_width);
    depth_softmax_kernel<<<(unsigned)plan->pixels, 128, 0, stream>>>(
        logits, plan->probabilities, (int)plan->desc.depth_bins, spatial);
    if (!std::getenv("BF_CUDA_LSS_PLAN_NCHW")) {
        size_t count = plan->desc.batches * plan->desc.cameras *
                       plan->desc.channels * spatial;
        context_nchw_to_nhwc_kernel<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            context, plan->context_nhwc, count, (int)plan->desc.channels, spatial);
        interval_pool_nhwc_kernel<<<(unsigned)plan->cell_count, 128, 0, stream>>>(
            plan->probabilities, plan->context_nhwc, plan->values_out,
            plan->starts, plan->ends, bev, (int)plan->cells_per_batch,
            (int)plan->desc.channels, (int)plan->desc.depth_bins, spatial);
    } else {
        interval_pool_kernel<<<(unsigned)plan->cell_count, 128, 0, stream>>>(
            plan->probabilities, context, plan->values_out, plan->starts, plan->ends,
            bev, (int)plan->cells_per_batch, (int)plan->desc.channels,
            (int)plan->desc.depth_bins, spatial);
    }
    cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? 1 : fail(error, cap, "CUDA LSS plan forward", status);
}

extern "C" int bf_cuda_lss_lift_pool_f32(
    const float *logits, const float *context, const float *geometry,
    float *bev, const bf_lss_desc *d, void *stream_value,
    char *error, size_t cap) {
    if (!logits || !context || !geometry || !bev || !lss_contract(d)) {
        if (error && cap) std::snprintf(error, cap, "invalid CUDA LSS contract");
        return 0;
    }
    size_t samples, blocks, bytes;
    if (!lss_sizes(d, &samples, &blocks, &bytes)) {
        if (error && cap) std::snprintf(error, cap, "CUDA LSS size overflow");
        return 0;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    cudaError_t status = cudaMemsetAsync(bev, 0, bytes, stream);
    if (status != cudaSuccess) return fail(error, cap, "CUDA LSS zero", status);
    size_t shared_bytes = (d->depth_bins + 128) * sizeof(float) + sizeof(int);
    const char *candidate = std::getenv("BF_CUDA_LSS_CACHED_GEOMETRY");
    if (!candidate)
        lift_pool_kernel<<<(unsigned)blocks, 128, shared_bytes, stream>>>(
        logits, context, geometry, bev, (int)d->cameras, (int)d->depth_bins,
        (int)d->feature_height, (int)d->feature_width, (int)d->channels,
        d->grid_minimum[0], d->grid_minimum[1], d->grid_minimum[2],
        d->grid_step[0], d->grid_step[1], d->grid_step[2],
        (int)d->grid_cells[0], (int)d->grid_cells[1], (int)d->grid_cells[2]);
    else
        lift_pool_cached_geometry_kernel<<<(unsigned)blocks, 128, shared_bytes, stream>>>(
        logits, context, geometry, bev, (int)d->cameras, (int)d->depth_bins,
        (int)d->feature_height, (int)d->feature_width, (int)d->channels,
        d->grid_minimum[0], d->grid_minimum[1], d->grid_minimum[2],
        d->grid_step[0], d->grid_step[1], d->grid_step[2],
        (int)d->grid_cells[0], (int)d->grid_cells[1], (int)d->grid_cells[2]);
    status = cudaGetLastError();
    return status == cudaSuccess ? 1 : fail(error, cap, "CUDA LSS launch", status);
}
