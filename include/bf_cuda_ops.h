#ifndef BF_CUDA_OPS_H
#define BF_CUDA_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal strict-FP32 CUDA operator boundary. All pointers are device
 * pointers and stream is a cudaStream_t (or NULL). Convolution weights use
 * OIHW layout; transposed-convolution weights use IOHW layout.
 */
typedef struct {
    int batches;
    int input_channels;
    int output_channels;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
    int stride_height;
    int stride_width;
    int padding_height;
    int padding_width;
    int output_height;
    int output_width;
    int transpose;
    int relu;
} bf_cuda_conv2d_desc;

int bf_cuda_conv2d_f32(const bf_cuda_conv2d_desc *desc,
                       const float *input,
                       const float *weight,
                       const float *bias,
                       float *output,
                       void *stream,
                       char *error, size_t error_cap);

/*
 * Row-major output = input[M,K] * weight[N,K]^T + bias[N].
 * The optional ReLU is fused into the writeback.
 */
int bf_cuda_gemm_f32(const float *input, const float *weight,
                     const float *bias, float *output,
                     int rows, int input_features, int output_features,
                     int relu, void *stream,
                     char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
