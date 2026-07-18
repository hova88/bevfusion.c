#ifndef BF_MODEL_H
#define BF_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BF_MODEL_VERSION 1u
#define BF_MODEL_MAX_RANK 8u
#define BF_MODEL_NAME_CAP 96u

typedef enum {
    BF_DTYPE_F32 = 1,
    BF_DTYPE_I64 = 2
} bf_dtype;

typedef struct {
    const char *name;
    const void *data;
    uint64_t nbytes;
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[BF_MODEL_MAX_RANK];
    uint32_t crc32;
} bf_tensor;

typedef struct bf_model bf_model;

int bf_model_open(const char *path, bf_model **out, char *error, size_t error_cap);
void bf_model_close(bf_model *model);
size_t bf_model_tensor_count(const bf_model *model);
uint64_t bf_model_file_bytes(const bf_model *model);
const bf_tensor *bf_model_tensor_at(const bf_model *model, size_t index);
const bf_tensor *bf_model_find(const bf_model *model, const char *name);
int bf_model_validate_tensor(const bf_model *model, const bf_tensor *tensor,
                             char *error, size_t error_cap);
int bf_model_validate_all(const bf_model *model, char *error, size_t error_cap);
const char *bf_dtype_name(uint32_t dtype);

#ifdef __cplusplus
}
#endif

#endif
