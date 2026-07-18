#define _POSIX_C_SOURCE 200809L
#include "bf_frame.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define IMAGE_FLOATS (6u * 3u * 256u * 704u)
#define MATRIX_FLOATS (6u * 16u)

struct bf_frame_file {
    void *mapping;
    size_t bytes;
    bf_frame_input input;
};

static int fail(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return 0;
}

static uint32_t load_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

static uint64_t load_u64(const unsigned char *p) {
    return (uint64_t)load_u32(p) | (uint64_t)load_u32(p + 4) << 32;
}

static uint32_t crc32_bytes(const unsigned char *data, size_t bytes) {
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (uint32_t)-(int)(crc & 1));
    }
    return ~crc;
}

static int add_bytes(size_t offset, size_t count, size_t element, size_t *end) {
    if (count && element > SIZE_MAX / count) return 0;
    size_t bytes = count * element;
    if (offset > SIZE_MAX - bytes) return 0;
    *end = offset + bytes;
    return 1;
}

static int finite_floats(const float *values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

int bf_frame_open(const char *path, bf_frame_file **out,
                  char *error, size_t cap) {
    if (out) *out = NULL;
    if (!path || !out) return fail(error, cap, "invalid frame arguments");
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1)
        return fail(error, cap, "BFI requires a little-endian host");
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (error && cap) snprintf(error, cap, "cannot open frame: %s", strerror(errno));
        return 0;
    }
    struct stat status;
    if (fstat(fd, &status) != 0 || status.st_size < (off_t)BF_FRAME_HEADER_BYTES ||
        (uintmax_t)status.st_size > SIZE_MAX) {
        close(fd);
        return fail(error, cap, "invalid frame file size");
    }
    size_t bytes = (size_t)status.st_size;
    void *mapping = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return fail(error, cap, "frame mmap failed");
    const unsigned char *p = (const unsigned char *)mapping;
    uint64_t declared = load_u64(p + 16), point_count64 = load_u64(p + 24);
    uint64_t offsets64[7];
    for (size_t i = 0; i < 7; ++i) offsets64[i] = load_u64(p + 40 + i * 8);
    int header_ok = memcmp(p, "BFI1", 4) == 0 && load_u32(p + 4) == BF_FRAME_VERSION &&
        load_u32(p + 8) == BF_FRAME_HEADER_BYTES && declared == bytes &&
        point_count64 > 0 && point_count64 <= SIZE_MAX / 5u &&
        crc32_bytes(p + BF_FRAME_HEADER_BYTES, bytes - BF_FRAME_HEADER_BYTES) ==
            load_u32(p + 32);
    size_t expected[7], point_count = (size_t)point_count64, cursor = BF_FRAME_HEADER_BYTES;
    expected[0] = cursor;
    header_ok = header_ok && add_bytes(cursor, IMAGE_FLOATS, sizeof(float), &cursor);
    expected[1] = cursor;
    header_ok = header_ok && add_bytes(cursor, point_count * 5u, sizeof(float), &cursor);
    for (size_t i = 2; i < 5; ++i) {
        expected[i] = cursor;
        header_ok = header_ok && add_bytes(cursor, MATRIX_FLOATS, sizeof(float), &cursor);
    }
    expected[5] = cursor;
    header_ok = header_ok && add_bytes(cursor, 16, sizeof(float), &cursor);
    expected[6] = cursor;
    header_ok = header_ok && add_bytes(cursor, MATRIX_FLOATS, sizeof(float), &cursor) &&
        cursor == bytes;
    for (size_t i = 0; i < 7; ++i)
        header_ok = header_ok && offsets64[i] == expected[i] && !(expected[i] & 3u);
    if (!header_ok) {
        munmap(mapping, bytes);
        return fail(error, cap, "malformed, non-canonical, or corrupt BFI frame");
    }
    bf_frame_file *file = (bf_frame_file *)calloc(1, sizeof(*file));
    if (!file) {
        munmap(mapping, bytes);
        return fail(error, cap, "frame handle allocation failed");
    }
    file->mapping = mapping;
    file->bytes = bytes;
    file->input.camera_images = (const float *)(p + expected[0]);
    file->input.points = (const float *)(p + expected[1]);
    file->input.point_count = point_count;
    file->input.camera_intrinsics_6x16 = (const float *)(p + expected[2]);
    file->input.camera_to_lidar_6x16 = (const float *)(p + expected[3]);
    file->input.image_augmentation_6x16 = (const float *)(p + expected[4]);
    file->input.lidar_augmentation_16 = (const float *)(p + expected[5]);
    file->input.lidar_to_image_6x16 = (const float *)(p + expected[6]);
    if (!finite_floats(file->input.camera_images, IMAGE_FLOATS) ||
        !finite_floats(file->input.points, point_count * 5u) ||
        !finite_floats(file->input.camera_intrinsics_6x16, MATRIX_FLOATS) ||
        !finite_floats(file->input.camera_to_lidar_6x16, MATRIX_FLOATS) ||
        !finite_floats(file->input.image_augmentation_6x16, MATRIX_FLOATS) ||
        !finite_floats(file->input.lidar_augmentation_16, 16) ||
        !finite_floats(file->input.lidar_to_image_6x16, MATRIX_FLOATS)) {
        bf_frame_close(file);
        return fail(error, cap, "BFI frame contains non-finite values");
    }
    *out = file;
    return 1;
}

void bf_frame_close(bf_frame_file *file) {
    if (!file) return;
    if (file->mapping && file->bytes) munmap(file->mapping, file->bytes);
    free(file);
}

const bf_frame_input *bf_frame_input_view(const bf_frame_file *file) {
    return file ? &file->input : NULL;
}

size_t bf_frame_file_bytes(const bf_frame_file *file) {
    return file ? file->bytes : 0;
}
