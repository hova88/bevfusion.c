#ifndef BF_DEPTH_RASTER_H
#define BF_DEPTH_RASTER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * points: [P,point_stride], fields [batch,x,y,z,...]
 * matrices are row-major 4x4: lidar_aug [B,16], lidar_to_image and
 * image_aug [B,N,16]. Output is zero-initialized [B,N,H,W].
 * When points hit the same pixel, later input points overwrite earlier ones,
 * matching the ordered CPU reference assignment.
 */
int bf_depth_rasterize_f32_ref(const float *points, size_t point_count,
                               size_t point_stride,
                               const float *lidar_augmentation,
                               const float *lidar_to_image,
                               const float *image_augmentation,
                               float *dense_depth,
                               size_t batches, size_t cameras,
                               size_t image_height, size_t image_width);

#ifdef __cplusplus
}
#endif

#endif
