#define _POSIX_C_SOURCE 200809L
#include "bf_model.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define BF_MAGIC "BFW0001"

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t tensor_count;
    uint64_t directory_offset;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint32_t directory_crc32;
    uint32_t flags;
    uint8_t reserved[8];
} bf_disk_header;

typedef struct {
    char name[BF_MODEL_NAME_CAP];
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[BF_MODEL_MAX_RANK];
    uint64_t offset;
    uint64_t nbytes;
    uint32_t crc32;
    uint32_t flags;
} bf_disk_tensor;

struct bf_model {
    int fd;
    void *mapping;
    size_t mapping_bytes;
    bf_tensor *tensors;
    uint32_t *hash_slots;
    size_t tensor_count;
    size_t hash_cap;
};

_Static_assert(sizeof(bf_disk_header) == 64, "BFW header ABI changed");
_Static_assert(sizeof(bf_disk_tensor) == 160, "BFW tensor ABI changed");

static int fail(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list ap;
        va_start(ap, format);
        vsnprintf(error, cap, format, ap);
        va_end(ap);
    }
    return 0;
}

static int host_is_little_endian(void) {
    const uint16_t one = 1;
    return *(const uint8_t *)&one == 1;
}

static uint64_t checked_mul(uint64_t a, uint64_t b, int *ok) {
    if (a && b > UINT64_MAX / a) {
        *ok = 0;
        return 0;
    }
    return a * b;
}

static uint32_t crc32_bytes(const void *data, size_t size) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (uint32_t)-(int32_t)(crc & 1));
    }
    return ~crc;
}

static uint64_t name_hash(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static int tensor_contract(const bf_disk_tensor *disk, size_t file_bytes,
                           uint64_t data_offset,
                           char *error, size_t cap) {
    if (!memchr(disk->name, '\0', sizeof(disk->name)))
        return fail(error, cap, "tensor name is not terminated");
    if (!disk->name[0]) return fail(error, cap, "empty tensor name");
    if (disk->rank > BF_MODEL_MAX_RANK)
        return fail(error, cap, "%s: rank %u exceeds limit", disk->name, disk->rank);
    uint64_t item_bytes = disk->dtype == BF_DTYPE_F32 ? 4 :
                          disk->dtype == BF_DTYPE_I64 ? 8 : 0;
    if (!item_bytes) return fail(error, cap, "%s: unsupported dtype %u", disk->name, disk->dtype);
    int ok = 1;
    uint64_t elements = 1;
    for (uint32_t i = 0; i < disk->rank; ++i) {
        if (!disk->dims[i]) return fail(error, cap, "%s: zero dimension", disk->name);
        elements = checked_mul(elements, disk->dims[i], &ok);
    }
    uint64_t expected = checked_mul(elements, item_bytes, &ok);
    if (!ok || expected != disk->nbytes)
        return fail(error, cap, "%s: shape byte count mismatch", disk->name);
    if ((disk->offset & 63u) != 0)
        return fail(error, cap, "%s: data is not 64-byte aligned", disk->name);
    if (disk->offset < data_offset)
        return fail(error, cap, "%s: data overlaps model metadata", disk->name);
    if (disk->offset > file_bytes || disk->nbytes > file_bytes - disk->offset)
        return fail(error, cap, "%s: data range exceeds file", disk->name);
    return 1;
}

int bf_model_open(const char *path, bf_model **out, char *error, size_t cap) {
    if (out) *out = NULL;
    if (!path || !out) return fail(error, cap, "invalid model arguments");
    if (!host_is_little_endian()) return fail(error, cap, "BFW requires a little-endian host");
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(error, cap, "open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(bf_disk_header)) {
        close(fd);
        return fail(error, cap, "%s: truncated model", path);
    }
    size_t bytes = (size_t)st.st_size;
    void *mapping = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        close(fd);
        return fail(error, cap, "mmap %s: %s", path, strerror(errno));
    }
    const bf_disk_header *header = (const bf_disk_header *)mapping;
    int good = memcmp(header->magic, BF_MAGIC, 8) == 0 &&
               header->version == BF_MODEL_VERSION &&
               header->header_bytes == sizeof(*header) &&
               header->directory_offset == sizeof(*header) &&
               header->file_bytes == bytes &&
               header->tensor_count > 0 && header->tensor_count <= 1000000;
    uint64_t dir_bytes = 0;
    int arithmetic_ok = 1;
    dir_bytes = checked_mul(header->tensor_count, sizeof(bf_disk_tensor), &arithmetic_ok);
    if (!good || !arithmetic_ok || header->directory_offset > bytes ||
        dir_bytes > bytes - header->directory_offset || header->data_offset > bytes ||
        header->data_offset < header->directory_offset ||
        dir_bytes > header->data_offset - header->directory_offset ||
        (header->data_offset & 63u) != 0) {
        munmap(mapping, bytes);
        close(fd);
        return fail(error, cap, "%s: invalid BFW header", path);
    }
    const bf_disk_tensor *directory = (const bf_disk_tensor *)
        ((const uint8_t *)mapping + header->directory_offset);
    if (crc32_bytes(directory, (size_t)dir_bytes) != header->directory_crc32) {
        munmap(mapping, bytes);
        close(fd);
        return fail(error, cap, "%s: directory CRC mismatch", path);
    }
    bf_model *model = (bf_model *)calloc(1, sizeof(*model));
    if (!model) {
        munmap(mapping, bytes);
        close(fd);
        return fail(error, cap, "model allocation failed");
    }
    model->fd = fd;
    model->mapping = mapping;
    model->mapping_bytes = bytes;
    model->tensor_count = (size_t)header->tensor_count;
    model->tensors = (bf_tensor *)calloc(model->tensor_count, sizeof(*model->tensors));
    model->hash_cap = 1;
    while (model->hash_cap < model->tensor_count * 2) model->hash_cap <<= 1;
    model->hash_slots = (uint32_t *)calloc(model->hash_cap, sizeof(*model->hash_slots));
    if (!model->tensors || !model->hash_slots) {
        bf_model_close(model);
        return fail(error, cap, "tensor index allocation failed");
    }
    for (size_t i = 0; i < model->tensor_count; ++i) {
        if (!tensor_contract(&directory[i], bytes, header->data_offset, error, cap)) {
            bf_model_close(model);
            return 0;
        }
        bf_tensor *tensor = &model->tensors[i];
        tensor->name = directory[i].name;
        tensor->data = (const uint8_t *)mapping + directory[i].offset;
        tensor->nbytes = directory[i].nbytes;
        tensor->dtype = directory[i].dtype;
        tensor->rank = directory[i].rank;
        memcpy(tensor->dims, directory[i].dims, sizeof(tensor->dims));
        tensor->crc32 = directory[i].crc32;
        size_t slot = (size_t)name_hash(tensor->name) & (model->hash_cap - 1);
        while (model->hash_slots[slot]) {
            const bf_tensor *prior = &model->tensors[model->hash_slots[slot] - 1];
            if (strcmp(prior->name, tensor->name) == 0) {
                bf_model_close(model);
                return fail(error, cap, "duplicate tensor %s", tensor->name);
            }
            slot = (slot + 1) & (model->hash_cap - 1);
        }
        model->hash_slots[slot] = (uint32_t)i + 1;
    }
    *out = model;
    return 1;
}

void bf_model_close(bf_model *model) {
    if (!model) return;
    free(model->hash_slots);
    free(model->tensors);
    if (model->mapping && model->mapping != MAP_FAILED)
        munmap(model->mapping, model->mapping_bytes);
    if (model->fd >= 0) close(model->fd);
    free(model);
}

size_t bf_model_tensor_count(const bf_model *model) { return model ? model->tensor_count : 0; }
uint64_t bf_model_file_bytes(const bf_model *model) { return model ? model->mapping_bytes : 0; }

const bf_tensor *bf_model_tensor_at(const bf_model *model, size_t index) {
    return model && index < model->tensor_count ? &model->tensors[index] : NULL;
}

const bf_tensor *bf_model_find(const bf_model *model, const char *name) {
    if (!model || !name) return NULL;
    size_t slot = (size_t)name_hash(name) & (model->hash_cap - 1);
    while (model->hash_slots[slot]) {
        const bf_tensor *tensor = &model->tensors[model->hash_slots[slot] - 1];
        if (strcmp(tensor->name, name) == 0) return tensor;
        slot = (slot + 1) & (model->hash_cap - 1);
    }
    return NULL;
}

int bf_model_validate_tensor(const bf_model *model, const bf_tensor *tensor,
                             char *error, size_t cap) {
    if (!model || !tensor || tensor < model->tensors ||
        tensor >= model->tensors + model->tensor_count)
        return fail(error, cap, "tensor does not belong to model");
    uint32_t actual = crc32_bytes(tensor->data, (size_t)tensor->nbytes);
    if (actual != tensor->crc32)
        return fail(error, cap, "%s: data CRC mismatch", tensor->name);
    return 1;
}

int bf_model_validate_all(const bf_model *model, char *error, size_t cap) {
    if (!model) return fail(error, cap, "model is null");
    for (size_t i = 0; i < model->tensor_count; ++i)
        if (!bf_model_validate_tensor(model, &model->tensors[i], error, cap)) return 0;
    return 1;
}

const char *bf_dtype_name(uint32_t dtype) {
    return dtype == BF_DTYPE_F32 ? "f32" : dtype == BF_DTYPE_I64 ? "i64" : "unknown";
}
