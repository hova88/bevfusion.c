#include "bf_cuda_ops.h"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float pattern(size_t i) {
    return (float)((int)(i * 37u % 101u) - 50) / 37.0f;
}

static int same(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    if (std::isinf(a) || std::isinf(b)) return a == b;
    float difference = std::fabs(a - b);
    return difference <= 2e-5f + 2e-5f * std::fabs(b);
}

static void reference_conv(const bf_cuda_conv2d_desc &d,
                           const float *input, const float *weight,
                           const float *bias, float *output) {
    int ohw = d.output_height * d.output_width;
    int ihw = d.input_height * d.input_width;
    int ks = d.kernel_height * d.kernel_width;
    for (int n = 0; n < d.batches; ++n)
        for (int oc = 0; oc < d.output_channels; ++oc)
            for (int oy = 0; oy < d.output_height; ++oy)
                for (int ox = 0; ox < d.output_width; ++ox) {
                    float value = 0.0f;
                    for (int ic = 0; ic < d.input_channels; ++ic)
                        for (int ky = 0; ky < d.kernel_height; ++ky)
                            for (int kx = 0; kx < d.kernel_width; ++kx) {
                                int iy, ix;
                                if (!d.transpose) {
                                    iy = oy * d.stride_height -
                                         d.padding_height + ky;
                                    ix = ox * d.stride_width -
                                         d.padding_width + kx;
                                } else {
                                    int sy = oy + d.padding_height - ky;
                                    int sx = ox + d.padding_width - kx;
                                    if (sy < 0 || sx < 0 ||
                                        sy % d.stride_height ||
                                        sx % d.stride_width) continue;
                                    iy = sy / d.stride_height;
                                    ix = sx / d.stride_width;
                                }
                                if ((unsigned)iy >= (unsigned)d.input_height ||
                                    (unsigned)ix >= (unsigned)d.input_width)
                                    continue;
                                size_t wi = !d.transpose
                                    ? ((size_t)oc * d.input_channels + ic) *
                                          ks + ky * d.kernel_width + kx
                                    : ((size_t)ic * d.output_channels + oc) *
                                          ks + ky * d.kernel_width + kx;
                                value = std::fma(
                                    input[(size_t)n * d.input_channels * ihw +
                                          (size_t)ic * ihw +
                                          iy * d.input_width + ix],
                                    weight[wi], value);
                            }
                    value += bias ? bias[oc] : 0.0f;
                    if (d.relu && value < 0.0f) value = 0.0f;
                    output[(size_t)n * d.output_channels * ohw +
                           (size_t)oc * ohw + oy * d.output_width + ox] =
                        value;
                }
}

static int run_conv(const bf_cuda_conv2d_desc &d, int inject_special) {
    size_t inputs = (size_t)d.batches * d.input_channels *
                    d.input_height * d.input_width;
    size_t weights = (size_t)d.input_channels * d.output_channels *
                     d.kernel_height * d.kernel_width;
    size_t outputs = (size_t)d.batches * d.output_channels *
                     d.output_height * d.output_width;
    std::vector<float> hi(inputs), hw(weights), hb(d.output_channels);
    std::vector<float> expected(outputs), actual(outputs);
    for (size_t i = 0; i < inputs; ++i) hi[i] = pattern(i);
    for (size_t i = 0; i < weights; ++i) hw[i] = pattern(i + 11) * 0.1f;
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = pattern(i + 29) * 0.2f;
    if (inject_special) {
        hi[0] = INFINITY;
        hw[0] = 0.0f;
        hi[inputs - 1] = NAN;
    }
    reference_conv(d, hi.data(), hw.data(), hb.data(), expected.data());
    float *di = nullptr, *dw = nullptr, *db = nullptr, *out = nullptr;
    int ok = cudaMalloc(&di, inputs * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&dw, weights * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&db, hb.size() * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&out, outputs * sizeof(float)) == cudaSuccess &&
             cudaMemcpy(di, hi.data(), inputs * sizeof(float),
                        cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(dw, hw.data(), weights * sizeof(float),
                        cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(db, hb.data(), hb.size() * sizeof(float),
                        cudaMemcpyHostToDevice) == cudaSuccess;
    char error[256] = {};
    if (ok) ok = bf_cuda_conv2d_f32(&d, di, dw, db, out, nullptr,
                                     error, sizeof(error));
    if (ok) ok = cudaMemcpy(actual.data(), out, outputs * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    for (size_t i = 0; ok && i < outputs; ++i)
        if (!same(actual[i], expected[i])) {
            std::fprintf(stderr, "conv mismatch at %zu: %.9g != %.9g\n",
                         i, actual[i], expected[i]);
            ok = 0;
        }
    if (!ok && error[0]) std::fprintf(stderr, "%s\n", error);
    cudaFree(out); cudaFree(db); cudaFree(dw); cudaFree(di);
    return ok;
}

static int run_gemm(void) {
    const int rows = 19, inputs = 23, outputs = 17;
    std::vector<float> a(rows * inputs), w(outputs * inputs), b(outputs);
    std::vector<float> expected(rows * outputs), actual(rows * outputs);
    for (size_t i = 0; i < a.size(); ++i) a[i] = pattern(i) * 0.2f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = pattern(i + 7) * 0.1f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = pattern(i + 3);
    for (int m = 0; m < rows; ++m)
        for (int n = 0; n < outputs; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < inputs; ++k)
                sum = std::fma(a[m * inputs + k], w[n * inputs + k], sum);
            expected[m * outputs + n] = std::fmax(sum + b[n], 0.0f);
        }
    float *da = nullptr, *dw = nullptr, *db = nullptr, *out = nullptr;
    int ok = cudaMalloc(&da, a.size() * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&dw, w.size() * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&db, b.size() * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&out, actual.size() * sizeof(float)) == cudaSuccess &&
             cudaMemcpy(da, a.data(), a.size() * sizeof(float),
                        cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(dw, w.data(), w.size() * sizeof(float),
                        cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(db, b.data(), b.size() * sizeof(float),
                        cudaMemcpyHostToDevice) == cudaSuccess;
    char error[256] = {};
    if (ok) ok = bf_cuda_gemm_f32(da, dw, db, out, rows, inputs, outputs,
                                   1, nullptr, error, sizeof(error));
    if (ok) ok = cudaMemcpy(actual.data(), out, actual.size() * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    for (size_t i = 0; ok && i < actual.size(); ++i)
        if (!same(actual[i], expected[i])) ok = 0;
    if (!ok && error[0]) std::fprintf(stderr, "%s\n", error);
    cudaFree(out); cudaFree(db); cudaFree(dw); cudaFree(da);
    return ok;
}

int main(void) {
    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count < 1) {
        std::fprintf(stderr, "CUDA device unavailable: %s\n",
                     cudaGetErrorString(device_status));
        return 5;
    }
    char error[128] = {};
    bf_cuda_conv2d_desc invalid = {};
    if (bf_cuda_conv2d_f32(&invalid, nullptr, nullptr, nullptr, nullptr,
                           nullptr, error, sizeof(error)))
        return 2;
    bf_cuda_conv2d_desc oversized = {
        INT_MAX,1,1,INT_MAX,INT_MAX,1,1,1,1,0,0,INT_MAX,INT_MAX,0,0
    };
    float *sentinel = reinterpret_cast<float *>(1);
    if (bf_cuda_conv2d_f32(&oversized, sentinel, sentinel, nullptr, sentinel,
                           nullptr, error, sizeof(error)))
        return 3;
    bf_cuda_conv2d_desc stride = {2,3,5,6,7,3,3,2,2,1,1,3,4,0,1};
    bf_cuda_conv2d_desc wide = {1,2,3,9,10,5,5,4,4,2,2,3,3,0,0};
    bf_cuda_conv2d_desc transpose = {1,3,5,4,6,2,2,2,2,0,0,8,12,1,1};
    /* 65,536 row tiles: guards against placing flattened N*H*W on grid.y. */
    bf_cuda_conv2d_desc large_grid = {
        1,1,1,1,1048561,1,1,1,1,0,0,1,1048561,0,0
    };
    int ok = run_conv(stride, 0) && run_conv(wide, 1) &&
             run_conv(transpose, 0) && run_conv(large_grid, 0) && run_gemm();
    std::puts(ok ? "CUDA custom convolution/GEMM fixtures passed" :
                   "CUDA custom operator fixture failed");
    return ok ? 0 : 5;
}
