#ifndef BF_TRANSFUSION_DECODER_H
#define BF_TRANSFUSION_DECODER_H

#include "bf_model.h"

#include <stddef.h>
#include <stdint.h>

#define BF_TRANSFUSION_CLASSES 10u
#define BF_TRANSFUSION_CHANNELS 128u

typedef struct bf_transfusion_decoder bf_transfusion_decoder;

typedef struct {
    float *center_b2p;
    float *height_b1p;
    float *dimension_log_b3p;
    float *rotation_sincos_b2p;
    float *velocity_b2p;
    float *heatmap_logits_b10p;
    float *query_heatmap_scores_b10p;
    int64_t *query_labels_bp;
    int64_t *query_indices_bp;
} bf_transfusion_raw_outputs;

int bf_transfusion_decoder_create(const bf_model *model,
                                  bf_transfusion_decoder **out,
                                  char *error, size_t error_cap);
void bf_transfusion_decoder_destroy(bf_transfusion_decoder *decoder);

size_t bf_transfusion_decoder_workspace_bytes(size_t height, size_t width,
                                               size_t proposals);

/* shared [B,128,H,W], dense_heatmap_logits [B,10,H,W]. */
int bf_transfusion_decoder_forward_ref(
    const bf_transfusion_decoder *decoder,
    const float *shared_b128hw,
    const float *dense_heatmap_logits_b10hw,
    size_t batches, size_t height, size_t width, size_t proposals,
    bf_transfusion_raw_outputs *outputs,
    void *workspace, size_t workspace_bytes,
    char *error, size_t error_cap);

#endif
