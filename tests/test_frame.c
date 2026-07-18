#define _POSIX_C_SOURCE 200809L
#include "bf_frame.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define IMAGE_FLOATS (6u * 3u * 256u * 704u)

static void store_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}
static void store_u64(unsigned char *p, uint64_t v) {
    store_u32(p, (uint32_t)v); store_u32(p + 4, (uint32_t)(v >> 32));
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
static int write_all(int fd, const void *data, size_t bytes) {
    const unsigned char *p = data;
    while (bytes) {
        ssize_t n = write(fd, p, bytes);
        if (n <= 0) return 0;
        p += n; bytes -= (size_t)n;
    }
    return 1;
}
static void identity(float *matrix) {
    matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
}

int main(void) {
    const size_t point_count = 1;
    size_t sizes[7] = {IMAGE_FLOATS * sizeof(float), point_count * 5 * sizeof(float),
        96 * sizeof(float), 96 * sizeof(float), 96 * sizeof(float),
        16 * sizeof(float), 96 * sizeof(float)};
    size_t offsets[7], total = BF_FRAME_HEADER_BYTES;
    for (size_t i = 0; i < 7; ++i) { offsets[i] = total; total += sizes[i]; }
    unsigned char *file_data = calloc(total, 1);
    if (!file_data) return 2;
    memcpy(file_data, "BFI1", 4); store_u32(file_data + 4, BF_FRAME_VERSION);
    store_u32(file_data + 8, BF_FRAME_HEADER_BYTES); store_u64(file_data + 16, total);
    store_u64(file_data + 24, point_count);
    for (size_t i = 0; i < 7; ++i) store_u64(file_data + 40 + i * 8, offsets[i]);
    float *points = (float *)(file_data + offsets[1]);
    points[3] = 0.5f;
    for (size_t section = 2; section < 7; ++section) {
        size_t count = section == 5 ? 1 : 6;
        float *matrices = (float *)(file_data + offsets[section]);
        for (size_t i = 0; i < count; ++i) identity(matrices + i * 16);
    }
    store_u32(file_data + 32, crc32_bytes(file_data + BF_FRAME_HEADER_BYTES,
                                           total - BF_FRAME_HEADER_BYTES));
    char path[] = "/tmp/bf-frame-XXXXXX";
    int fd = mkstemp(path);
    int ok = fd >= 0 && write_all(fd, file_data, total) && close(fd) == 0;
    char error[256] = {0};
    bf_frame_file *frame = NULL;
    ok = ok && bf_frame_open(path, &frame, error, sizeof(error));
    const bf_frame_input *view = bf_frame_input_view(frame);
    ok = ok && view && view->point_count == point_count &&
         view->points[3] == 0.5f && view->lidar_to_image_6x16[15] == 1.0f &&
         bf_frame_file_bytes(frame) == total;
    bf_frame_close(frame); frame = NULL;
    file_data[offsets[0]] = 1;
    fd = open(path, O_WRONLY | O_TRUNC);
    ok = ok && fd >= 0 && write_all(fd, file_data, total) && close(fd) == 0 &&
         !bf_frame_open(path, &frame, error, sizeof(error));
    file_data[offsets[0]] = 0;
    float not_finite = NAN;
    memcpy(file_data + offsets[0], &not_finite, sizeof(not_finite));
    store_u32(file_data + 32, crc32_bytes(file_data + BF_FRAME_HEADER_BYTES,
                                           total - BF_FRAME_HEADER_BYTES));
    fd = open(path, O_WRONLY | O_TRUNC);
    ok = ok && fd >= 0 && write_all(fd, file_data, total) && close(fd) == 0 &&
         !bf_frame_open(path, &frame, error, sizeof(error));
    unlink(path); free(file_data);
    if (!ok) { fprintf(stderr, "frame test failed: %s\n", error); return 3; }
    puts("canonical frame mmap, CRC, layout, and finite-value gates pass");
    return 0;
}
