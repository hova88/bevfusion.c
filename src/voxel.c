#include "bf_voxel.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t key;
    size_t voxel;
} bf_voxel_slot;

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

static size_t table_capacity(size_t max_voxels) {
    if (!max_voxels || max_voxels > SIZE_MAX / 2) return 0;
    size_t capacity = 2;
    while (capacity < max_voxels * 2) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity <<= 1;
    }
    return capacity;
}

int bf_voxel_grid_shape(const bf_voxel_config *config,
                        size_t grid_xyz[3]) {
    if (!config || !grid_xyz || !config->point_features ||
        !config->max_points_per_voxel || !config->max_voxels) return 0;
    for (size_t axis = 0; axis < 3; ++axis) {
        float extent = config->maximum[axis] - config->minimum[axis];
        float size = config->voxel_size[axis];
        if (!isfinite(extent) || !isfinite(size) || extent <= 0.0f || size <= 0.0f)
            return 0;
        double cells = (double)extent / size;
        long long rounded = llround(cells);
        if (rounded <= 0 || (double)rounded > (double)SIZE_MAX ||
            fabs(cells - (double)rounded) > 1e-4 * fmax(1.0, cells)) return 0;
        grid_xyz[axis] = (size_t)rounded;
    }
    return 1;
}

size_t bf_voxelize_workspace_bytes(const bf_voxel_config *config) {
    size_t capacity = config ? table_capacity(config->max_voxels) : 0;
    return capacity && capacity <= SIZE_MAX / sizeof(bf_voxel_slot)
        ? capacity * sizeof(bf_voxel_slot) : 0;
}

int bf_voxelize_f32_workspace_ref(const float *points, size_t point_count,
                        size_t point_stride, int32_t batch_index,
                        const bf_voxel_config *config, float *voxels,
                        bf_coord4 *coords, int64_t *counts,
                        size_t *voxel_count, bf_voxel_stats *stats,
                        void *workspace, size_t workspace_bytes) {
    if (voxel_count) *voxel_count = 0;
    if (stats) memset(stats, 0, sizeof(*stats));
    size_t grid[3];
    if (!points || !config || !voxels || !coords || !counts || !voxel_count ||
        !stats || !workspace || batch_index < 0 || point_stride < config->point_features ||
        config->point_features < 3 || !bf_voxel_grid_shape(config, grid)) return 0;
    size_t capacity = table_capacity(config->max_voxels);
    size_t required = bf_voxelize_workspace_bytes(config);
    if (!capacity || !required || workspace_bytes < required) return 0;
    bf_voxel_slot *table = (bf_voxel_slot *)workspace;
    memset(table, 0, required);
    stats->input_points = point_count;
    size_t live_voxels = 0;
    for (size_t point_index = 0; point_index < point_count; ++point_index) {
        const float *point = points + point_index * point_stride;
        int finite = 1;
        for (size_t feature = 0; feature < config->point_features; ++feature)
            finite &= isfinite(point[feature]);
        if (!finite) {
            ++stats->rejected_nonfinite;
            continue;
        }
        int32_t cell[3];
        int inside = 1;
        for (size_t axis = 0; axis < 3; ++axis) {
            if (point[axis] < config->minimum[axis] || point[axis] >= config->maximum[axis]) {
                inside = 0;
                break;
            }
            cell[axis] = (int32_t)floorf((point[axis] - config->minimum[axis]) /
                                         config->voxel_size[axis]);
            if (cell[axis] < 0 || (size_t)cell[axis] >= grid[axis]) {
                inside = 0;
                break;
            }
        }
        if (!inside) {
            ++stats->rejected_out_of_range;
            continue;
        }
        uint64_t key = ((uint64_t)(uint32_t)cell[2] * grid[1] + (uint32_t)cell[1]) *
                       grid[0] + (uint32_t)cell[0] + 1;
        size_t slot = (size_t)mix64(key) & (capacity - 1);
        while (table[slot].key && table[slot].key != key)
            slot = (slot + 1) & (capacity - 1);
        size_t voxel;
        if (!table[slot].key) {
            if (live_voxels == config->max_voxels) {
                ++stats->dropped_voxel_capacity;
                continue;
            }
            voxel = live_voxels++;
            table[slot].key = key;
            table[slot].voxel = voxel;
            coords[voxel].batch = batch_index;
            coords[voxel].z = cell[2];
            coords[voxel].y = cell[1];
            coords[voxel].x = cell[0];
            counts[voxel] = 0;
            memset(voxels + voxel * config->max_points_per_voxel * config->point_features,
                   0, config->max_points_per_voxel * config->point_features * sizeof(*voxels));
        } else {
            voxel = table[slot].voxel;
        }
        if ((size_t)counts[voxel] == config->max_points_per_voxel) {
            ++stats->dropped_point_capacity;
            continue;
        }
        float *destination = voxels +
            (voxel * config->max_points_per_voxel + (size_t)counts[voxel]) * config->point_features;
        memcpy(destination, point, config->point_features * sizeof(*destination));
        ++counts[voxel];
        ++stats->accepted_points;
    }
    *voxel_count = live_voxels;
    return 1;
}

int bf_voxelize_f32_ref(const float *points, size_t point_count,
                        size_t point_stride, int32_t batch_index,
                        const bf_voxel_config *config, float *voxels,
                        bf_coord4 *coords, int64_t *counts,
                        size_t *voxel_count, bf_voxel_stats *stats) {
    size_t bytes = bf_voxelize_workspace_bytes(config);
    void *workspace = bytes ? malloc(bytes) : NULL;
    int ok = workspace && bf_voxelize_f32_workspace_ref(points, point_count,
        point_stride, batch_index, config, voxels, coords, counts,
        voxel_count, stats, workspace, bytes);
    free(workspace);
    return ok;
}
