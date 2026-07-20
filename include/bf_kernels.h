#ifndef BF_KERNELS_H
#define BF_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t n, in_channels, in_height, in_width;
    size_t out_channels, kernel_height, kernel_width;
    size_t stride_height, stride_width;
    size_t pad_height, pad_width;
    size_t dilation_height, dilation_width;
    size_t groups;
} bf_conv2d_desc;

typedef struct {
    size_t n, in_channels, in_height, in_width;
    size_t out_channels, kernel_height, kernel_width;
    size_t stride_height, stride_width;
    size_t pad_height, pad_width;
    size_t output_pad_height, output_pad_width;
    size_t dilation_height, dilation_width;
    size_t groups;
} bf_conv_transpose2d_desc;

typedef struct {
    int32_t batch, z, y, x;
} bf_coord4;

typedef struct {
    size_t batches, in_depth, in_height, in_width;
    size_t in_channels, out_channels;
    size_t kernel_depth, kernel_height, kernel_width;
    size_t stride_depth, stride_height, stride_width;
    size_t pad_depth, pad_height, pad_width;
    size_t dilation_depth, dilation_height, dilation_width;
    int submanifold;
} bf_sparse_conv3d_desc;

int bf_conv2d_output_shape(const bf_conv2d_desc *desc,
                           size_t *out_height, size_t *out_width);
int bf_conv2d_f32_ref(const float *input, const float *weight,
                      const float *bias, float *output,
                      const bf_conv2d_desc *desc);
/* Production CPU dispatch: BLAS for eligible shapes, OpenMP when compiled,
   otherwise the scalar reference. BF_CPU_SCALAR=1 forces the reference. */
int bf_conv2d_f32(const float *input, const float *weight,
                  const float *bias, float *output,
                  const bf_conv2d_desc *desc);
int bf_conv_transpose2d_output_shape(const bf_conv_transpose2d_desc *desc,
                                     size_t *out_height, size_t *out_width);
int bf_conv_transpose2d_f32_ref(const float *input, const float *weight_iohw,
                                const float *bias, float *output,
                                const bf_conv_transpose2d_desc *desc);
void bf_linear_f32_ref(const float *input, const float *weight,
                       const float *bias, float *output,
                       size_t rows, size_t in_features, size_t out_features);
void bf_linear_f32(const float *input, const float *weight,
                   const float *bias, float *output,
                   size_t rows, size_t in_features, size_t out_features);
const char *bf_cpu_kernel_backend(void);
void bf_batch_norm_2d_f32_ref(const float *input, const float *scale,
                              const float *bias, const float *mean,
                              const float *variance, float epsilon,
                              float *output, size_t n, size_t channels,
                              size_t height, size_t width);
void bf_batch_norm_2d_f32(const float *input, const float *scale,
                          const float *bias, const float *mean,
                          const float *variance, float epsilon,
                          float *output, size_t n, size_t channels,
                          size_t height, size_t width);
void bf_layer_norm_f32_ref(const float *input, const float *scale,
                           const float *bias, float epsilon, float *output,
                           size_t rows, size_t channels);
void bf_layer_norm_f32(const float *input, const float *scale,
                       const float *bias, float epsilon, float *output,
                       size_t rows, size_t channels);
void bf_softmax_f32_ref(const float *input, float *output,
                        size_t rows, size_t columns);
void bf_gelu_f32_ref(const float *input, float *output, size_t count);
void bf_relu_f32_ref(const float *input, float *output, size_t count);
void bf_gelu_f32(const float *input, float *output, size_t count);
void bf_relu_f32(const float *input, float *output, size_t count);
void bf_mean_vfe_f32_ref(const float *points, const int64_t *counts,
                         float *output, size_t voxels, size_t max_points,
                         size_t channels);
int bf_topk_f32_ref(const float *input, float *values, int64_t *indices,
                    size_t rows, size_t columns, size_t k);
int bf_sparse_conv3d_output_shape(const bf_sparse_conv3d_desc *desc,
                                  size_t *depth, size_t *height, size_t *width);
size_t bf_sparse_conv3d_workspace_bytes(size_t input_capacity,
                                        size_t output_capacity);
int bf_sparse_conv3d_f32_workspace_ref(
                             const bf_coord4 *input_coords,
                             const float *input_features, size_t input_count,
                             const float *weight_kdhwio, const float *bias,
                             bf_coord4 *output_coords, float *output_features,
                             size_t output_capacity, size_t *output_count,
                             const bf_sparse_conv3d_desc *desc,
                             void *workspace, size_t workspace_bytes);
int bf_sparse_conv3d_f32_ref(const bf_coord4 *input_coords,
                             const float *input_features, size_t input_count,
                             const float *weight_kdhwio, const float *bias,
                             bf_coord4 *output_coords, float *output_features,
                             size_t output_capacity, size_t *output_count,
                             const bf_sparse_conv3d_desc *desc);

#ifdef __cplusplus
}
#endif

#endif
