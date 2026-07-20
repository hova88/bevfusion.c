#define _POSIX_C_SOURCE 200809L
#include "bf_model.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int require_shape(const bf_model *model, const char *name,
                         uint32_t rank, const uint32_t *dims) {
    const bf_tensor *tensor = bf_model_find(model, name);
    if (!tensor || tensor->rank != rank || tensor->dtype != BF_DTYPE_F32) return 0;
    for (uint32_t i = 0; i < rank; ++i)
        if (tensor->dims[i] != dims[i]) return 0;
    return 1;
}

static int malformed_header_test(void) {
    char path[] = "/tmp/bf-model-test-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    const char junk[] = "not a bevfusion model";
    int ok = write(fd, junk, sizeof(junk)) == (ssize_t)sizeof(junk);
    close(fd);
    bf_model *model = NULL;
    char error[128];
    ok = ok && !bf_model_open(path, &model, error, sizeof(error));
    unlink(path);
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        if (!malformed_header_test()) return 5;
        puts("model container malformed-header gate passes");
        return 0;
    }
    if (argc != 2) return 2;
    char error[256] = "unspecified contract failure";
    bf_model *model = NULL;
    if (!bf_model_open(argv[1], &model, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 3;
    }
    const uint32_t sparse_in[] = {3, 3, 3, 5, 16};
    const uint32_t patch[] = {96, 3, 4, 4};
    const uint32_t fusion[] = {256, 336, 3, 3};
    const uint32_t head[] = {10, 64, 1};
    int ok = bf_model_tensor_count(model) > 450 &&
        require_shape(model, "backbone_3d.conv_input.0.weight", 5, sparse_in) &&
        require_shape(model, "image_backbone.patch_embed.projection.weight", 4, patch) &&
        require_shape(model, "fuser.conv.0.weight", 4, fusion) &&
        require_shape(model, "dense_head.prediction_head.heatmap.1.weight", 3, head) &&
        bf_model_find(model, "does.not.exist") == NULL &&
        bf_model_tensor_at(model, bf_model_tensor_count(model)) == NULL &&
        bf_model_validate_all(model, error, sizeof(error));
    bf_model_close(model);
    if (!ok) {
        fprintf(stderr, "model contract test failed: %s\n", error);
        return 4;
    }
    if (!malformed_header_test()) return 5;
    puts("model container: contract, lookup, CRC, and malformed header pass");
    return 0;
}
