#ifndef BF_RUNTIME_H
#define BF_RUNTIME_H

#include "bevfusion.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bf_runtime bf_runtime;

typedef struct {
    /* [6,3,256,704], already resized and normalized with the YAML contract. */
    const float *camera_images;
    /* [point_count,5]: x,y,z,intensity,timestamp in the augmented lidar frame. */
    const float *points;
    size_t point_count;
    /* Row-major homogeneous matrices. */
    const float *camera_intrinsics_6x16;
    const float *camera_to_lidar_6x16;
    const float *image_augmentation_6x16;
    const float *lidar_augmentation_16;
    const float *lidar_to_image_6x16;
} bf_frame_input;

int bf_runtime_create(const char *model_path, bf_runtime **out,
                      char *error, size_t error_cap);
void bf_runtime_destroy(bf_runtime *runtime);

size_t bf_runtime_workspace_bytes(size_t point_capacity,
                                  size_t voxel_capacity,
                                  size_t sparse_capacity);

/* Strict scalar CPU reference, batch size one, six cameras. */
int bf_runtime_infer_cpu_ref(const bf_runtime *runtime,
                             const bf_frame_input *frame,
                             size_t point_capacity,
                             size_t voxel_capacity,
                             size_t sparse_capacity,
                             bf_detections *detections,
                             void *workspace, size_t workspace_bytes,
                             char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif
