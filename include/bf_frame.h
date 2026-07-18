#ifndef BF_FRAME_H
#define BF_FRAME_H

#include "bf_runtime.h"

#include <stddef.h>

#define BF_FRAME_VERSION 1u
#define BF_FRAME_HEADER_BYTES 128u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_frame_file bf_frame_file;

/* Opens a canonical little-endian BFI frame and exposes mmap-backed tensors. */
int bf_frame_open(const char *path, bf_frame_file **out,
                  char *error, size_t error_cap);
void bf_frame_close(bf_frame_file *file);
const bf_frame_input *bf_frame_input_view(const bf_frame_file *file);
size_t bf_frame_file_bytes(const bf_frame_file *file);

#ifdef __cplusplus
}
#endif

#endif
