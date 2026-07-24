#include "bf_cuda_ops.h"

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <climits>

namespace {

constexpr int kTile = 16;

static int fail(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(error, cap, format, args);
        va_end(args);
    }
    return 0;
}

__device__ __forceinline__ float conv_input(
    const bf_cuda_conv2d_desc &d, const float *input,
    int row, int reduction) {
    int output_spatial = d.output_height * d.output_width;
    int batch = row / output_spatial;
    int position = row - batch * output_spatial;
    int oy = position / d.output_width;
    int ox = position - oy * d.output_width;
    int kernel_spatial = d.kernel_height * d.kernel_width;
    int channel = reduction / kernel_spatial;
    int kernel = reduction - channel * kernel_spatial;
    int ky = kernel / d.kernel_width;
    int kx = kernel - ky * d.kernel_width;
    int iy;
    int ix;
    if (!d.transpose) {
        iy = oy * d.stride_height - d.padding_height + ky;
        ix = ox * d.stride_width - d.padding_width + kx;
    } else {
        int sy = oy + d.padding_height - ky;
        int sx = ox + d.padding_width - kx;
        if (sy < 0 || sx < 0 ||
            sy % d.stride_height || sx % d.stride_width)
            return 0.0f;
        iy = sy / d.stride_height;
        ix = sx / d.stride_width;
    }
    if ((unsigned)iy >= (unsigned)d.input_height ||
        (unsigned)ix >= (unsigned)d.input_width)
        return 0.0f;
    return input[((size_t)batch * d.input_channels + channel) *
                 d.input_height * d.input_width +
                 (size_t)iy * d.input_width + ix];
}

__device__ __forceinline__ float conv_weight(
    const bf_cuda_conv2d_desc &d, const float *weight,
    int output_channel, int reduction) {
    int kernel_spatial = d.kernel_height * d.kernel_width;
    int input_channel = reduction / kernel_spatial;
    int kernel = reduction - input_channel * kernel_spatial;
    if (!d.transpose)
        return weight[((size_t)output_channel * d.input_channels +
                       input_channel) * kernel_spatial + kernel];
    return weight[((size_t)input_channel * d.output_channels +
                   output_channel) * kernel_spatial + kernel];
}

__global__ void conv2d_kernel(bf_cuda_conv2d_desc desc,
                              const float *__restrict__ input,
                              const float *__restrict__ weight,
                              const float *__restrict__ bias,
                              float *__restrict__ output) {
    __shared__ float tile_input[kTile][kTile];
    __shared__ float tile_weight[kTile][kTile];
    int row = (int)blockIdx.x * kTile + (int)threadIdx.y;
    int output_channel = (int)blockIdx.y * kTile + (int)threadIdx.x;
    int rows = desc.batches * desc.output_height * desc.output_width;
    int reduction = desc.input_channels *
                    desc.kernel_height * desc.kernel_width;
    float sum = 0.0f;
    for (int base = 0; base < reduction; base += kTile) {
        int input_k = base + (int)threadIdx.x;
        int weight_k = base + (int)threadIdx.y;
        tile_input[threadIdx.y][threadIdx.x] =
            row < rows && input_k < reduction
                ? conv_input(desc, input, row, input_k) : 0.0f;
        tile_weight[threadIdx.y][threadIdx.x] =
            output_channel < desc.output_channels && weight_k < reduction
                ? conv_weight(desc, weight, output_channel, weight_k) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k)
            sum = fmaf(tile_input[threadIdx.y][k],
                       tile_weight[k][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < rows && output_channel < desc.output_channels) {
        int spatial = desc.output_height * desc.output_width;
        int batch = row / spatial;
        int position = row - batch * spatial;
        float value = sum + (bias ? bias[output_channel] : 0.0f);
        if (desc.relu && value < 0.0f) value = 0.0f;
        output[((size_t)batch * desc.output_channels + output_channel) *
               spatial + position] = value;
    }
}

__global__ void gemm_kernel(const float *__restrict__ input,
                            const float *__restrict__ weight,
                            const float *__restrict__ bias,
                            float *__restrict__ output,
                            int rows, int input_features,
                            int output_features, int relu) {
    __shared__ float tile_input[kTile][kTile];
    __shared__ float tile_weight[kTile][kTile];
    int row = (int)blockIdx.x * kTile + (int)threadIdx.y;
    int column = (int)blockIdx.y * kTile + (int)threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < input_features; base += kTile) {
        int input_k = base + (int)threadIdx.x;
        int weight_k = base + (int)threadIdx.y;
        tile_input[threadIdx.y][threadIdx.x] =
            row < rows && input_k < input_features
                ? input[(size_t)row * input_features + input_k] : 0.0f;
        tile_weight[threadIdx.y][threadIdx.x] =
            column < output_features && weight_k < input_features
                ? weight[(size_t)column * input_features + weight_k] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k)
            sum = fmaf(tile_input[threadIdx.y][k],
                       tile_weight[k][threadIdx.x], sum);
        __syncthreads();
    }
    if (row < rows && column < output_features) {
        float value = sum + (bias ? bias[column] : 0.0f);
        if (relu && value < 0.0f) value = 0.0f;
        output[(size_t)row * output_features + column] = value;
    }
}

}  // namespace

extern "C" int bf_cuda_conv2d_f32(const bf_cuda_conv2d_desc *desc,
    const float *input, const float *weight, const float *bias, float *output,
    void *stream_value, char *error, size_t cap) {
    if (!desc || !input || !weight || !output || desc->batches <= 0 ||
        desc->input_channels <= 0 || desc->output_channels <= 0 ||
        desc->input_height <= 0 || desc->input_width <= 0 ||
        desc->kernel_height <= 0 || desc->kernel_width <= 0 ||
        desc->stride_height <= 0 || desc->stride_width <= 0 ||
        desc->padding_height < 0 || desc->padding_width < 0 ||
        desc->output_height <= 0 || desc->output_width <= 0 ||
        (desc->transpose != 0 && desc->transpose != 1) ||
        (desc->relu != 0 && desc->relu != 1))
        return fail(error, cap, "invalid custom CUDA convolution contract");
    long long batch_rows =
        (long long)desc->batches * desc->output_height;
    if (batch_rows > INT_MAX / desc->output_width)
        return fail(error, cap, "custom CUDA convolution is too large");
    int rows = (int)(batch_rows * desc->output_width);
    long long channel_reduction =
        (long long)desc->input_channels * desc->kernel_height;
    if (channel_reduction > INT_MAX / desc->kernel_width)
        return fail(error, cap, "custom CUDA convolution reduction is too large");
    if ((desc->output_channels + (long long)kTile - 1) / kTile > 65535)
        return fail(error, cap, "custom CUDA convolution has too many output channels");
    dim3 block(kTile, kTile);
    /*
     * Put the potentially very large flattened N*H*W dimension on grid.x.
     * The batch-six 8x upsampled depth feature has 67,584 row tiles, which is
     * one beyond the architectural grid.y limit of 65,535.
     */
    dim3 grid((unsigned)(rows + kTile - 1) / kTile,
              (unsigned)(desc->output_channels + kTile - 1) / kTile);
    conv2d_kernel<<<grid, block, 0,
        reinterpret_cast<cudaStream_t>(stream_value)>>>(
            *desc, input, weight, bias, output);
    cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? 1 :
        fail(error, cap, "custom CUDA convolution launch: %s",
             cudaGetErrorString(status));
}

extern "C" int bf_cuda_gemm_f32(const float *input, const float *weight,
    const float *bias, float *output, int rows, int input_features,
    int output_features, int relu, void *stream_value,
    char *error, size_t cap) {
    if (!input || !weight || !output || rows <= 0 || input_features <= 0 ||
        output_features <= 0 || (relu != 0 && relu != 1))
        return fail(error, cap, "invalid custom CUDA GEMM contract");
    if ((output_features + (long long)kTile - 1) / kTile > 65535)
        return fail(error, cap, "custom CUDA GEMM has too many output features");
    dim3 block(kTile, kTile);
    dim3 grid((unsigned)(rows + kTile - 1) / kTile,
              (unsigned)(output_features + kTile - 1) / kTile);
    gemm_kernel<<<grid, block, 0,
        reinterpret_cast<cudaStream_t>(stream_value)>>>(
            input, weight, bias, output, rows, input_features,
            output_features, relu);
    cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ? 1 :
        fail(error, cap, "custom CUDA GEMM launch: %s",
             cudaGetErrorString(status));
}
