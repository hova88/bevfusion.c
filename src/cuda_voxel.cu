#include "bf_cuda_voxel.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

struct bf_cuda_voxelizer {
    bf_voxel_config config;
    size_t max_points, resident_bytes;
    unsigned grid_x, grid_y, grid_z;
    uint64_t *keys[2];
    unsigned *indices[2], *flags, *segments, *starts, *ends;
    unsigned *first_indices[2], *segment_order[2];
    void *sort_scratch, *scan_scratch;
    size_t sort_scratch_bytes, scan_scratch_bytes;
};

static int fail(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(error, cap, format, args);
        va_end(args);
    }
    return 0;
}

static int cuda_ok(cudaError_t status, char *error, size_t cap,
                   const char *where) {
    return status == cudaSuccess ? 1 :
        fail(error, cap, "%s: %s", where, cudaGetErrorString(status));
}

__global__ static void make_keys_kernel(const float *points, unsigned count,
    uint64_t *keys, unsigned *indices, float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z, float size_x, float size_y,
    float size_z, unsigned grid_x, unsigned grid_y, unsigned grid_z) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        const float *p = points + (size_t)i * 5;
        bool finite = isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]) &&
                      isfinite(p[3]) && isfinite(p[4]);
        bool inside = finite && p[0] >= min_x && p[0] < max_x &&
                      p[1] >= min_y && p[1] < max_y &&
                      p[2] >= min_z && p[2] < max_z;
        uint64_t key = UINT64_MAX;
        if (inside) {
            unsigned x = (unsigned)floorf(__fdiv_rn(p[0] - min_x, size_x));
            unsigned y = (unsigned)floorf(__fdiv_rn(p[1] - min_y, size_y));
            unsigned z = (unsigned)floorf(__fdiv_rn(p[2] - min_z, size_z));
            if (x < grid_x && y < grid_y && z < grid_z)
                key = ((uint64_t)z * grid_y + y) * grid_x + x;
        }
        keys[i] = key;
        indices[i] = i;
    }
}

__global__ static void mark_segments_kernel(const uint64_t *keys,
                                             unsigned *flags,
                                             unsigned count) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x)
        flags[i] = keys[i] != UINT64_MAX && (i == 0 || keys[i - 1] != keys[i]);
}

__global__ static void write_intervals_kernel(const uint64_t *keys,
    const unsigned *flags, const unsigned *segments, unsigned count,
    const unsigned *sorted_indices, unsigned *starts, unsigned *ends,
    unsigned *first_indices, unsigned *segment_order, unsigned max_voxels,
    unsigned *voxel_count) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        uint64_t key = keys[i];
        unsigned segment = segments[i];
        if (key != UINT64_MAX && !flags[i]) --segment;
        if (flags[i]) {
            starts[segment] = i;
            first_indices[segment] = sorted_indices[i];
            segment_order[segment] = segment;
        }
        if (key != UINT64_MAX && (i + 1 == count || keys[i + 1] != key) &&
            segment < count)
            ends[segment] = i + 1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned total = segments[count - 1] + flags[count - 1];
        *voxel_count = total < max_voxels ? total : max_voxels;
    }
}

__global__ static void materialize_coords_kernel(const uint64_t *keys,
    const unsigned *starts, const unsigned *segment_order,
    const unsigned *voxel_count, unsigned max_voxels,
    unsigned grid_x, unsigned grid_y, bf_coord4 *coords) {
    for (unsigned voxel = blockIdx.x * blockDim.x + threadIdx.x;
         voxel < max_voxels; voxel += gridDim.x * blockDim.x) {
        if (voxel >= *voxel_count) continue;
        unsigned segment = segment_order[voxel];
        uint64_t decoded = keys[starts[segment]];
        unsigned x = (unsigned)(decoded % grid_x);
        decoded /= grid_x;
        unsigned y = (unsigned)(decoded % grid_y);
        unsigned z = (unsigned)(decoded / grid_y);
        coords[voxel] = {0, (int)z, (int)y, (int)x};
    }
}

__global__ static void mean_vfe_kernel(const float *points,
    const unsigned *sorted_indices, const unsigned *starts,
    const unsigned *ends, const unsigned *segment_order,
    const unsigned *voxel_count,
    unsigned max_voxels, unsigned max_points_per_voxel, float *means) {
    for (unsigned voxel = blockIdx.x * blockDim.x + threadIdx.x;
         voxel < max_voxels; voxel += gridDim.x * blockDim.x) {
        if (voxel >= *voxel_count) continue;
        unsigned segment = segment_order[voxel];
        unsigned begin = starts[segment], end = ends[segment];
        if (end - begin > max_points_per_voxel)
            end = begin + max_points_per_voxel;
        float sum[5] = {};
        for (unsigned i = begin; i < end; ++i) {
            const float *p = points + (size_t)sorted_indices[i] * 5;
#pragma unroll
            for (int c = 0; c < 5; ++c) sum[c] += p[c];
        }
        float scale = 1.0f / (float)(end - begin);
#pragma unroll
        for (int c = 0; c < 5; ++c)
            means[(size_t)voxel * 5 + c] = sum[c] * scale;
    }
}

extern "C" void bf_cuda_voxelizer_destroy(bf_cuda_voxelizer *v) {
    if (!v) return;
    for (int i = 0; i < 2; ++i) {
        cudaFree(v->keys[i]);
        cudaFree(v->indices[i]);
        cudaFree(v->first_indices[i]);
        cudaFree(v->segment_order[i]);
    }
    cudaFree(v->flags);
    cudaFree(v->segments);
    cudaFree(v->starts);
    cudaFree(v->ends);
    cudaFree(v->sort_scratch);
    cudaFree(v->scan_scratch);
    std::free(v);
}

extern "C" int bf_cuda_voxelizer_create(const bf_voxel_config *config,
    size_t max_input_points, bf_cuda_voxelizer **out,
    char *error, size_t cap) {
    if (out) *out = nullptr;
    if (!config || !out || !max_input_points || max_input_points > INT_MAX ||
        config->point_features != 5 || !config->max_points_per_voxel ||
        config->max_points_per_voxel > UINT_MAX || !config->max_voxels ||
        config->max_voxels > UINT_MAX)
        return fail(error, cap, "invalid CUDA voxelizer contract");
    unsigned grid[3];
    for (int axis = 0; axis < 3; ++axis) {
        float extent = config->maximum[axis] - config->minimum[axis];
        float cells = extent / config->voxel_size[axis];
        if (!std::isfinite(config->minimum[axis]) ||
            !std::isfinite(config->maximum[axis]) ||
            !(config->maximum[axis] > config->minimum[axis]) ||
            !std::isfinite(config->voxel_size[axis]) ||
            !(config->voxel_size[axis] > 0.0f) || cells < 1.0f ||
            cells > (float)INT_MAX)
            return fail(error, cap, "invalid CUDA voxel grid axis %d", axis);
        unsigned rounded = (unsigned)llround((double)cells);
        if (fabs((double)cells - rounded) > 1e-3)
            return fail(error, cap, "non-integral CUDA voxel grid axis %d", axis);
        grid[axis] = rounded;
    }
    if ((uint64_t)grid[0] > UINT64_MAX / grid[1] ||
        (uint64_t)grid[0] * grid[1] > UINT64_MAX / grid[2])
        return fail(error, cap, "CUDA voxel key range overflow");
    uint64_t volume = (uint64_t)grid[0] * grid[1] * grid[2];
    if (!volume || volume == UINT64_MAX)
        return fail(error, cap, "CUDA voxel key range overflow");
    bf_cuda_voxelizer *v =
        (bf_cuda_voxelizer *)std::calloc(1, sizeof(*v));
    if (!v) return fail(error, cap, "CUDA voxelizer host allocation failed");
    v->config = *config;
    v->max_points = max_input_points;
    v->grid_x = grid[0]; v->grid_y = grid[1]; v->grid_z = grid[2];
    cudaError_t cub_status;
#define ALLOC(PTR, BYTES, LABEL) do {                                      \
    size_t allocation_bytes = (BYTES);                                     \
    if (!cuda_ok(cudaMalloc((void **)&(PTR), allocation_bytes), error, cap, \
                 LABEL)) goto failure;                                     \
    v->resident_bytes += allocation_bytes;                                 \
} while (0)
    for (int i = 0; i < 2; ++i) {
        ALLOC(v->keys[i], max_input_points * sizeof(uint64_t),
              "allocate CUDA voxel keys");
        ALLOC(v->indices[i], max_input_points * sizeof(unsigned),
              "allocate CUDA voxel indices");
    }
    ALLOC(v->flags, max_input_points * sizeof(unsigned),
          "allocate CUDA voxel flags");
    ALLOC(v->segments, max_input_points * sizeof(unsigned),
          "allocate CUDA voxel segments");
    ALLOC(v->starts, max_input_points * sizeof(unsigned),
          "allocate CUDA voxel starts");
    ALLOC(v->ends, max_input_points * sizeof(unsigned),
          "allocate CUDA voxel ends");
    for (int i = 0; i < 2; ++i) {
        ALLOC(v->first_indices[i], max_input_points * sizeof(unsigned),
              "allocate CUDA voxel first indices");
        ALLOC(v->segment_order[i], max_input_points * sizeof(unsigned),
              "allocate CUDA voxel segment order");
    }
    cub_status = cub::DeviceRadixSort::SortPairs(nullptr, v->sort_scratch_bytes,
        v->keys[0], v->keys[1], v->indices[0], v->indices[1],
        (int)max_input_points);
    if (!cuda_ok(cub_status, error, cap, "size CUDA voxel sort scratch"))
        goto failure;
    {size_t encounter_sort_bytes=0;cub_status=cub::DeviceRadixSort::SortPairs(
        nullptr,encounter_sort_bytes,v->first_indices[0],v->first_indices[1],
        v->segment_order[0],v->segment_order[1],(int)max_input_points);
     if(!cuda_ok(cub_status,error,cap,"size CUDA voxel encounter sort scratch"))
        goto failure;
     if(encounter_sort_bytes>v->sort_scratch_bytes)
        v->sort_scratch_bytes=encounter_sort_bytes;}
    cub_status = cub::DeviceScan::ExclusiveSum(nullptr, v->scan_scratch_bytes,
        v->flags, v->segments, (int)max_input_points);
    if (!cuda_ok(cub_status, error, cap, "size CUDA voxel scan scratch"))
        goto failure;
    ALLOC(v->sort_scratch, v->sort_scratch_bytes,
          "allocate CUDA voxel sort scratch");
    ALLOC(v->scan_scratch, v->scan_scratch_bytes,
          "allocate CUDA voxel scan scratch");
#undef ALLOC
    *out = v;
    return 1;
failure:
    bf_cuda_voxelizer_destroy(v);
    return 0;
}

extern "C" int bf_cuda_voxelize_mean_f32(bf_cuda_voxelizer *v,
    const float *points, size_t point_count, bf_coord4 *coords,
    float *means, unsigned *voxel_count, void *stream_value,
    char *error, size_t cap) {
    if (!v || !points || point_count > v->max_points || !coords || !means ||
        !voxel_count)
        return fail(error, cap, "invalid CUDA voxelize buffers");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    if (!point_count) {
        return cuda_ok(cudaMemsetAsync(voxel_count, 0, sizeof(unsigned), stream),
                       error, cap, "clear CUDA voxel count");
    }
    unsigned count = (unsigned)point_count;
    unsigned blocks = (count + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    make_keys_kernel<<<blocks, 256, 0, stream>>>(points, count, v->keys[0],
        v->indices[0], v->config.minimum[0], v->config.minimum[1],
        v->config.minimum[2], v->config.maximum[0], v->config.maximum[1],
        v->config.maximum[2], v->config.voxel_size[0],
        v->config.voxel_size[1], v->config.voxel_size[2], v->grid_x,
        v->grid_y, v->grid_z);
    cudaError_t status = cub::DeviceRadixSort::SortPairs(v->sort_scratch,
        v->sort_scratch_bytes, v->keys[0], v->keys[1], v->indices[0],
        v->indices[1], (int)count, 0, 64, stream);
    if (!cuda_ok(status, error, cap, "sort CUDA voxel keys")) return 0;
    mark_segments_kernel<<<blocks, 256, 0, stream>>>(v->keys[1], v->flags,
                                                     count);
    status = cub::DeviceScan::ExclusiveSum(v->scan_scratch,
        v->scan_scratch_bytes, v->flags, v->segments, (int)count, stream);
    if (!cuda_ok(status, error, cap, "scan CUDA voxel segments")) return 0;
    if (!cuda_ok(cudaMemsetAsync(v->first_indices[0], 0xff,
            count * sizeof(unsigned), stream), error, cap,
            "clear CUDA voxel encounter indices")) return 0;
    write_intervals_kernel<<<blocks, 256, 0, stream>>>(v->keys[1], v->flags,
        v->segments, count, v->indices[1], v->starts, v->ends,
        v->first_indices[0], v->segment_order[0],
        (unsigned)v->config.max_voxels, voxel_count);
    status = cub::DeviceRadixSort::SortPairs(v->sort_scratch,
        v->sort_scratch_bytes, v->first_indices[0], v->first_indices[1],
        v->segment_order[0], v->segment_order[1], (int)count, 0, 32, stream);
    if (!cuda_ok(status, error, cap, "sort CUDA voxel encounter order")) return 0;
    unsigned voxel_blocks =
        ((unsigned)v->config.max_voxels + 255) / 256;
    if (voxel_blocks > 4096) voxel_blocks = 4096;
    materialize_coords_kernel<<<voxel_blocks,256,0,stream>>>(v->keys[1],
        v->starts,v->segment_order[1],voxel_count,
        (unsigned)v->config.max_voxels,v->grid_x,v->grid_y,coords);
    mean_vfe_kernel<<<voxel_blocks, 256, 0, stream>>>(points, v->indices[1],
        v->starts, v->ends, v->segment_order[1], voxel_count,
        (unsigned)v->config.max_voxels,
        (unsigned)v->config.max_points_per_voxel, means);
    return cuda_ok(cudaGetLastError(), error, cap,
                   "launch CUDA voxelize/MeanVFE graph");
}

extern "C" size_t bf_cuda_voxelizer_resident_bytes(
    const bf_cuda_voxelizer *v) {
    return v ? v->resident_bytes : 0;
}
