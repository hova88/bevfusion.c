#ifndef BF_SWIN_H
#define BF_SWIN_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t batches;
    size_t height;
    size_t width;
    size_t channels;
    size_t heads;
    size_t window_size;
    size_t shift_size;
} bf_swin_window_desc;

int bf_swin_window_attention_f32_ref(
    const float *windows_bnc,
    const float *qkv_weight, const float *qkv_bias,
    const float *relative_position_bias,
    const int64_t *relative_position_index,
    const float *attention_mask,
    size_t mask_windows,
    const float *projection_weight, const float *projection_bias,
    float *output_bnc,
    size_t total_windows, size_t tokens, size_t channels,
    size_t heads, size_t relative_bias_rows);

int bf_swin_shifted_window_f32_ref(
    const float *query_blc,
    const float *qkv_weight, const float *qkv_bias,
    const float *relative_position_bias,
    const int64_t *relative_position_index,
    const float *projection_weight, const float *projection_bias,
    float *output_blc,
    const bf_swin_window_desc *desc);

size_t bf_swin_shifted_window_workspace_bytes(const bf_swin_window_desc *desc);
int bf_swin_shifted_window_f32_workspace_ref(
    const float *query_blc,
    const float *qkv_weight, const float *qkv_bias,
    const float *relative_position_bias,
    const int64_t *relative_position_index,
    const float *projection_weight, const float *projection_bias,
    float *output_blc, const bf_swin_window_desc *desc,
    void *workspace, size_t workspace_bytes);

#endif
