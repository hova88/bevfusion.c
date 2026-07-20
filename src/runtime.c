#include "bf_runtime.h"

#include "bf_bev_stage.h"
#include "bf_depth_head.h"
#include "bf_depth_raster.h"
#include "bf_image_fpn.h"
#include "bf_kernels.h"
#include "bf_lidar_backbone.h"
#include "bf_lss.h"
#include "bf_lss_downsample.h"
#include "bf_model.h"
#include "bf_swin_backbone.h"
#include "bf_transfusion.h"
#include "bf_transfusion_decoder.h"
#include "bf_voxel.h"

#include <stdarg.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAMERAS 6u
#define IMAGE_H 256u
#define IMAGE_W 704u
#define FEATURE_H 32u
#define FEATURE_W 88u
#define BEV_FULL 360u
#define BEV 180u

struct bf_runtime {
    bf_model *model;
    bf_swin_backbone *swin;
    bf_image_fpn *fpn;
    bf_depth_head *depth;
    bf_lss_downsample *lss_down;
    bf_lidar_backbone *lidar;
    bf_bev_stage *bev;
    bf_transfusion_decoder *decoder;
    const float *frustum;
};

typedef struct {
    unsigned char *base;
    size_t capacity;
    size_t used;
    size_t peak;
    int ok;
} bf_arena;

static int fail(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, cap, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static void *arena_take(bf_arena *arena, size_t bytes) {
    const size_t alignment = 64;
    if (!arena || !arena->ok || bytes > SIZE_MAX - (alignment - 1) ||
        arena->used > SIZE_MAX - (alignment - 1)) return NULL;
    size_t aligned = (arena->used + alignment - 1) & ~(alignment - 1);
    if (aligned < arena->used || bytes > SIZE_MAX - aligned ||
        aligned + bytes > arena->capacity) {
        arena->ok = 0;
        return NULL;
    }
    void *result = arena->base ? arena->base + aligned : (void *)(uintptr_t)alignment;
    arena->used = aligned + bytes;
    if (arena->used > arena->peak) arena->peak = arena->used;
    return result;
}

static void destroy_parts(bf_runtime *runtime) {
    if (!runtime) return;
    bf_transfusion_decoder_destroy(runtime->decoder);
    bf_bev_stage_destroy(runtime->bev);
    bf_lidar_backbone_destroy(runtime->lidar);
    bf_lss_downsample_destroy(runtime->lss_down);
    bf_depth_head_destroy(runtime->depth);
    bf_image_fpn_destroy(runtime->fpn);
    bf_swin_backbone_destroy(runtime->swin);
    bf_model_close(runtime->model);
}

int bf_runtime_create(const char *model_path, bf_runtime **out,
                      char *error, size_t cap) {
    if (out) *out = NULL;
    if (!model_path || !out) return fail(error, cap, "invalid runtime arguments");
    bf_runtime *runtime = (bf_runtime *)calloc(1, sizeof(*runtime));
    if (!runtime) return fail(error, cap, "runtime allocation failed");
    if (!bf_model_open(model_path, &runtime->model, error, cap) ||
        !bf_swin_backbone_create(runtime->model, &runtime->swin, error, cap) ||
        !bf_image_fpn_create(runtime->model, &runtime->fpn, error, cap) ||
        !bf_depth_head_create(runtime->model, &runtime->depth, error, cap) ||
        !bf_lss_downsample_create(runtime->model, &runtime->lss_down, error, cap) ||
        !bf_lidar_backbone_create(runtime->model, &runtime->lidar, error, cap) ||
        !bf_bev_stage_create(runtime->model, &runtime->bev, error, cap) ||
        !bf_transfusion_decoder_create(runtime->model, &runtime->decoder, error, cap))
        goto failure;
    const bf_tensor *frustum = bf_model_find(runtime->model, "vtransform.frustum");
    if (!frustum || frustum->dtype != BF_DTYPE_F32 || frustum->rank != 4 ||
        frustum->dims[0] != 118 || frustum->dims[1] != FEATURE_H ||
        frustum->dims[2] != FEATURE_W || frustum->dims[3] != 3) {
        fail(error, cap, "vtransform.frustum contract mismatch");
        goto failure;
    }
    runtime->frustum = (const float *)frustum->data;
    *out = runtime;
    return 1;
failure:
    destroy_parts(runtime);
    free(runtime);
    return 0;
}

void bf_runtime_destroy(bf_runtime *runtime) {
    destroy_parts(runtime);
    free(runtime);
}

static int take_floats(bf_arena *arena, size_t count) {
    size_t bytes;
    return checked_mul(count, sizeof(float), &bytes) && arena_take(arena, bytes) != NULL;
}

static size_t plan_workspace(size_t point_capacity, size_t voxel_capacity,
                             size_t sparse_capacity) {
    if (!point_capacity || point_capacity > SIZE_MAX / 6 ||
        !voxel_capacity || voxel_capacity > 160000 ||
        !sparse_capacity || sparse_capacity < voxel_capacity) return 0;
    bf_arena arena = {NULL, SIZE_MAX, 0, 0, 1};
#define FLOATS(count) do { if (!take_floats(&arena, (count))) return 0; } while (0)
#define BYTES(count) do { if (!arena_take(&arena, (count))) return 0; } while (0)
    /* Only the two modality BEV outputs survive across phases. */
    FLOATS(256 * BEV * BEV);
    FLOATS(80 * BEV * BEV);
    size_t persistent = arena.used;

    /* LiDAR phase. */
    FLOATS(voxel_capacity * 10 * 5);
    BYTES(voxel_capacity * sizeof(bf_coord4));
    BYTES(voxel_capacity * sizeof(int64_t));
    FLOATS(voxel_capacity * 5);
    bf_voxel_config voxel_plan = {
        {-54.0f, -54.0f, -5.0f}, {54.0f, 54.0f, 3.0f},
        {0.075f, 0.075f, 0.2f}, 5, 10, voxel_capacity
    };
    BYTES(bf_voxelize_workspace_bytes(&voxel_plan));
    BYTES(bf_lidar_backbone_workspace_bytes(sparse_capacity));
    arena.used = persistent;

    /* Camera depth and outputs needed through lift-splat. */
    FLOATS(CAMERAS * IMAGE_H * IMAGE_W);
    size_t dense_depth_mark = arena.used;
    FLOATS(point_capacity * 6);
    arena.used = dense_depth_mark;
    FLOATS(CAMERAS * 118 * 32 * 88);
    FLOATS(CAMERAS * 80 * 32 * 88);
    FLOATS(CAMERAS * 118 * 32 * 88 * 3);
    FLOATS(80 * BEV_FULL * BEV_FULL);
    size_t camera_fixed = arena.used;
    FLOATS(CAMERAS * 192 * 32 * 88);
    FLOATS(CAMERAS * 384 * 16 * 44);
    FLOATS(CAMERAS * 768 * 8 * 22);
    BYTES(bf_swin_backbone_workspace_bytes(CAMERAS, IMAGE_H, IMAGE_W));
    FLOATS(CAMERAS * 256 * 32 * 88);
    FLOATS(CAMERAS * 256 * 16 * 44);
    BYTES(bf_image_fpn_workspace_bytes(32, 88));
    BYTES(bf_depth_head_workspace_bytes(32, 88));
    arena.used = camera_fixed;
    BYTES(bf_lss_downsample_workspace_bytes(BEV_FULL, BEV_FULL));
    arena.used = persistent;

    /* Fusion, dense BEV backbone, decoder, and canonical decode. */
    FLOATS(336 * BEV * BEV);
    FLOATS(512 * BEV * BEV);
    FLOATS(128 * BEV * BEV);
    FLOATS(10 * BEV * BEV);
    BYTES(bf_bev_stage_workspace_bytes(BEV, BEV));
    BYTES(bf_transfusion_decoder_workspace_bytes(BEV, BEV, 200));
    FLOATS((2 + 1 + 3 + 2 + 2 + 10 + 10) * 200);
    BYTES(2 * 200 * sizeof(int64_t));
    FLOATS(200 * 9 + 200);
    BYTES(200 * sizeof(int64_t));
#undef FLOATS
#undef BYTES
    return arena.ok ? arena.peak : 0;
}

size_t bf_runtime_workspace_bytes(size_t point_capacity,
                                  size_t voxel_capacity,
                                  size_t sparse_capacity) {
    return plan_workspace(point_capacity, voxel_capacity, sparse_capacity);
}

static void extract_3x3_and_translation(const float *matrices4,
                                        float *rotations3, float *translations,
                                        size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float *m = matrices4 + i * 16;
        float *r = rotations3 + i * 9;
        r[0] = m[0]; r[1] = m[1]; r[2] = m[2];
        r[3] = m[4]; r[4] = m[5]; r[5] = m[6];
        r[6] = m[8]; r[7] = m[9]; r[8] = m[10];
        translations[i * 3] = m[3];
        translations[i * 3 + 1] = m[7];
        translations[i * 3 + 2] = m[11];
    }
}

static void debug_bev_stats(const char *name,const float *values,size_t count){double sum=0;float maximum=0;for(size_t i=0;i<count;++i){float value=fabsf(values[i]);sum+=value;if(value>maximum)maximum=value;}fprintf(stderr,"cpu_runtime %s l1=%.9g max=%.9g\n",name,sum,maximum);}

static double wall_milliseconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static void profile_stage(int enabled, const char *name,
                          double start, double *prior) {
    if (!enabled) return;
    double now = wall_milliseconds();
    fprintf(stderr, "cpu_profile %-18s stage=%10.3f ms total=%10.3f ms\n",
            name, now - *prior, now - start);
    *prior = now;
}

int bf_runtime_infer_cpu_ref(const bf_runtime *runtime,
                             const bf_frame_input *frame,
                             size_t point_capacity, size_t voxel_capacity,
                             size_t sparse_capacity, bf_detections *detections,
                             void *workspace, size_t workspace_bytes,
                             char *error, size_t cap) {
    size_t required = plan_workspace(point_capacity, voxel_capacity, sparse_capacity);
    if (!runtime || !frame || !frame->camera_images || !frame->points ||
        !frame->camera_intrinsics_6x16 || !frame->camera_to_lidar_6x16 ||
        !frame->image_augmentation_6x16 || !frame->lidar_augmentation_16 ||
        !frame->lidar_to_image_6x16 || !detections || !workspace ||
        !required || workspace_bytes < required || frame->point_count > point_capacity)
        return fail(error, cap, "invalid runtime frame, capacity, or workspace");
    const int profile = getenv("BF_CPU_PROFILE") != NULL;
    const double profile_start = wall_milliseconds();
    double profile_prior = profile_start;
    if (profile)
        fprintf(stderr, "cpu_profile backend=%s workspace=%.2f MiB points=%zu\n",
                bf_cpu_kernel_backend(), (double)required / (1024.0 * 1024.0),
                frame->point_count);
    bf_arena arena = {(unsigned char *)workspace, workspace_bytes, 0, 0, 1};
#define TAKE_FLOAT(name, count) float *name = arena_take(&arena, (count) * sizeof(float))
#define TAKE_BYTES(type, name, count) type *name = arena_take(&arena, (count) * sizeof(type))
    /* Cross-phase residents. */
    TAKE_FLOAT(lidar_bev, 256 * BEV * BEV);
    TAKE_FLOAT(image_bev, 80 * BEV * BEV);
    size_t persistent = arena.used;

    /* LiDAR phase: voxel buffers and sparse arena are released after BEV. */
    TAKE_FLOAT(voxels, voxel_capacity * 10 * 5);
    TAKE_BYTES(bf_coord4, voxel_coords, voxel_capacity);
    TAKE_BYTES(int64_t, voxel_counts, voxel_capacity);
    TAKE_FLOAT(voxel_features, voxel_capacity * 5);
    bf_voxel_config voxel_config = {
        {-54.0f, -54.0f, -5.0f}, {54.0f, 54.0f, 3.0f},
        {0.075f, 0.075f, 0.2f}, 5, 10, voxel_capacity
    };
    size_t voxel_workspace_bytes = bf_voxelize_workspace_bytes(&voxel_config);
    void *voxel_workspace = arena_take(&arena, voxel_workspace_bytes);
    void *lidar_workspace = arena_take(&arena, bf_lidar_backbone_workspace_bytes(sparse_capacity));
    if (!arena.ok) return fail(error, cap, "runtime arena planner mismatch");
    size_t voxel_count = 0;
    bf_voxel_stats voxel_stats;
    if (!bf_voxelize_f32_workspace_ref(frame->points, frame->point_count, 5, 0,
                             &voxel_config, voxels, voxel_coords, voxel_counts,
                             &voxel_count, &voxel_stats, voxel_workspace,
                             voxel_workspace_bytes) || !voxel_count)
        return fail(error, cap, "runtime voxelization produced no voxels");
    bf_mean_vfe_f32_ref(voxels, voxel_counts, voxel_features,
                        voxel_count, 10, 5);
    profile_stage(profile, "voxel+vfe", profile_start, &profile_prior);
    if (!bf_lidar_backbone_forward_ref(runtime->lidar, voxel_coords,
            voxel_features, voxel_count, 1, 41, 1440, 1440, sparse_capacity,
            lidar_bev, lidar_workspace,
            bf_lidar_backbone_workspace_bytes(sparse_capacity), error, cap)) return 0;
    profile_stage(profile, "lidar-backbone", profile_start, &profile_prior);
    arena.used = persistent;

    /* Camera phase. Dense point projection is discarded immediately. */
    TAKE_FLOAT(dense_depth, CAMERAS * IMAGE_H * IMAGE_W);
    size_t dense_depth_mark = arena.used;
    TAKE_FLOAT(batch_points, point_capacity * 6);
    if (!arena.ok) return fail(error, cap, "runtime camera input arena failed");
    for (size_t p = 0; p < frame->point_count; ++p) {
        batch_points[p * 6] = 0.0f;
        memcpy(batch_points + p * 6 + 1, frame->points + p * 5, 5 * sizeof(float));
    }
    if (!bf_depth_rasterize_f32_ref(
            batch_points, frame->point_count, 6, frame->lidar_augmentation_16,
            frame->lidar_to_image_6x16, frame->image_augmentation_6x16,
            dense_depth, 1, CAMERAS, IMAGE_H, IMAGE_W))
        return fail(error, cap, "runtime depth rasterization failed");
    profile_stage(profile, "depth-raster", profile_start, &profile_prior);
    arena.used = dense_depth_mark;
    TAKE_FLOAT(depth_logits, CAMERAS * 118 * 32 * 88);
    TAKE_FLOAT(context, CAMERAS * 80 * 32 * 88);
    TAKE_FLOAT(geometry, CAMERAS * 118 * 32 * 88 * 3);
    TAKE_FLOAT(full_image_bev, 80 * BEV_FULL * BEV_FULL);
    size_t camera_fixed = arena.used;
    TAKE_FLOAT(swin0, CAMERAS * 192 * 32 * 88);
    TAKE_FLOAT(swin1, CAMERAS * 384 * 16 * 44);
    TAKE_FLOAT(swin2, CAMERAS * 768 * 8 * 22);
    void *swin_workspace = arena_take(&arena, bf_swin_backbone_workspace_bytes(CAMERAS, IMAGE_H, IMAGE_W));
    if (!bf_swin_backbone_forward_ref(runtime->swin, frame->camera_images,
            CAMERAS, IMAGE_H, IMAGE_W, swin0, swin1, swin2,
            swin_workspace, bf_swin_backbone_workspace_bytes(CAMERAS, IMAGE_H, IMAGE_W),
            error, cap)) return 0;
    profile_stage(profile, "swin", profile_start, &profile_prior);
    TAKE_FLOAT(fpn0, CAMERAS * 256 * 32 * 88);
    TAKE_FLOAT(fpn1, CAMERAS * 256 * 16 * 44);
    void *fpn_workspace = arena_take(&arena, bf_image_fpn_workspace_bytes(32, 88));
    void *depth_workspace = arena_take(&arena, bf_depth_head_workspace_bytes(32, 88));
    if (!arena.ok || !bf_image_fpn_forward_ref(runtime->fpn, swin0, swin1, swin2,
            CAMERAS, 32, 88, fpn0, fpn1, fpn_workspace,
            bf_image_fpn_workspace_bytes(32, 88), error, cap) ||
        !bf_depth_head_forward_ref(runtime->depth, fpn0, dense_depth,
            CAMERAS, 32, 88, depth_logits, context, depth_workspace,
            bf_depth_head_workspace_bytes(32, 88), error, cap)) return 0;
    profile_stage(profile, "fpn+depth-head", profile_start, &profile_prior);
    arena.used = camera_fixed;
    float camera_rotation[CAMERAS * 9], camera_translation[CAMERAS * 3];
    float intrinsics[CAMERAS * 9], unused_translation[CAMERAS * 3];
    float post_rotation[CAMERAS * 9], post_translation[CAMERAS * 3];
    float extra_rotation[9], extra_translation[3];
    extract_3x3_and_translation(frame->camera_to_lidar_6x16,
                                camera_rotation, camera_translation, CAMERAS);
    extract_3x3_and_translation(frame->camera_intrinsics_6x16,
                                intrinsics, unused_translation, CAMERAS);
    extract_3x3_and_translation(frame->image_augmentation_6x16,
                                post_rotation, post_translation, CAMERAS);
    extract_3x3_and_translation(frame->lidar_augmentation_16,
                                extra_rotation, extra_translation, 1);
    if (!bf_lss_geometry_f32_ref(runtime->frustum, camera_rotation,
            camera_translation, intrinsics, post_rotation, post_translation,
            extra_rotation, extra_translation, geometry, 1, CAMERAS, 118, 32, 88))
        return fail(error, cap, "runtime LSS geometry failed");
    profile_stage(profile, "lss-geometry", profile_start, &profile_prior);
    bf_lss_desc lss = {1, CAMERAS, 118, 32, 88, 80,
        {-54.0f, -54.0f, -10.0f}, {0.3f, 0.3f, 20.0f}, {360, 360, 1}};
    if (!bf_lss_lift_pool_f32_ref(depth_logits, context, geometry,
                                   full_image_bev, &lss))
        return fail(error, cap, "runtime fused lift-splat failed");
    profile_stage(profile, "lss-lift-pool", profile_start, &profile_prior);
    void *lss_down_workspace = arena_take(&arena,
        bf_lss_downsample_workspace_bytes(BEV_FULL, BEV_FULL));
    if (!arena.ok || !bf_lss_downsample_forward_ref(runtime->lss_down, full_image_bev,
            1, BEV_FULL, BEV_FULL, image_bev, lss_down_workspace,
            bf_lss_downsample_workspace_bytes(BEV_FULL, BEV_FULL), error, cap)) return 0;
    profile_stage(profile, "lss-downsample", profile_start, &profile_prior);
    arena.used = persistent;

    /* Fusion/decoder phase. */
    TAKE_FLOAT(fusion, 336 * BEV * BEV);
    TAKE_FLOAT(spatial, 512 * BEV * BEV);
    TAKE_FLOAT(shared, 128 * BEV * BEV);
    TAKE_FLOAT(dense_heatmap, 10 * BEV * BEV);
    void *bev_workspace = arena_take(&arena, bf_bev_stage_workspace_bytes(BEV, BEV));
    void *decoder_workspace = arena_take(&arena, bf_transfusion_decoder_workspace_bytes(BEV, BEV, 200));
    TAKE_FLOAT(center, 2 * 200); TAKE_FLOAT(height_out, 200);
    TAKE_FLOAT(dimension, 3 * 200); TAKE_FLOAT(rotation, 2 * 200);
    TAKE_FLOAT(velocity, 2 * 200); TAKE_FLOAT(query_heatmap, 10 * 200);
    TAKE_FLOAT(query_scores, 10 * 200);
    TAKE_BYTES(int64_t, query_labels, 200); TAKE_BYTES(int64_t, query_indices, 200);
    TAKE_FLOAT(boxes, 200 * 9); TAKE_FLOAT(scores_out, 200);
    TAKE_BYTES(int64_t, labels_out, 200);
#undef TAKE_FLOAT
#undef TAKE_BYTES
    if (!arena.ok) return fail(error, cap, "runtime fusion arena failed");
    memcpy(fusion, image_bev, 80 * BEV * BEV * sizeof(float));
    memcpy(fusion + 80 * BEV * BEV, lidar_bev, 256 * BEV * BEV * sizeof(float));
    if(getenv("BF_CPU_RUNTIME_PROFILE")){debug_bev_stats("image_bev",image_bev,80*BEV*BEV);debug_bev_stats("lidar_bev",lidar_bev,256*BEV*BEV);}
    if (!bf_bev_stage_forward_ref(runtime->bev, fusion, 1, BEV, BEV,
            spatial, shared, dense_heatmap, bev_workspace,
            bf_bev_stage_workspace_bytes(BEV, BEV), error, cap)) return 0;
    profile_stage(profile, "bev-stage", profile_start, &profile_prior);
    bf_transfusion_raw_outputs raw = {center, height_out, dimension, rotation,
        velocity, query_heatmap, query_scores, query_labels, query_indices};
    if (!bf_transfusion_decoder_forward_ref(runtime->decoder, shared, dense_heatmap,
            1, BEV, BEV, 200, &raw, decoder_workspace,
            bf_transfusion_decoder_workspace_bytes(BEV, BEV, 200), error, cap)) return 0;
    profile_stage(profile, "transfusion", profile_start, &profile_prior);
    const float voxel_xy[2] = {0.075f, 0.075f}, minimum_xy[2] = {-54.0f, -54.0f};
    const float range[6] = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
    if (!bf_transfusion_decode_raw_f32_ref(query_heatmap, query_scores,
            query_labels, center, height_out, dimension, rotation, velocity,
            boxes, scores_out, labels_out, 1, 10, 200, 8.0f,
            voxel_xy, minimum_xy) ||
        !bf_transfusion_filter_detections(boxes, scores_out, labels_out,
            1, 200, 0.0f, range, detections))
        return fail(error, cap, "runtime canonical decode failed");
    profile_stage(profile, "decode", profile_start, &profile_prior);
    return 1;
}
