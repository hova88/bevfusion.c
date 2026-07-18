#include "bf_cuda_voxel.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

static int close5(const float *a, const float *b) {
    for (int i = 0; i < 5; ++i)
        if (std::fabs(a[i] - b[i]) > 1e-6f) return 0;
    return 1;
}

int main() {
    const float nan = NAN;
    const float points[][5] = {
        {1.2f, .1f, .1f, 10.f, 100.f},
        {.2f, .1f, .1f, 1.f, 2.f},
        {1.4f, .1f, .1f, 20.f, 200.f},
        {nan, .1f, .1f, 7.f, 8.f},
        {.3f, .1f, .1f, 3.f, 4.f},
        {1.6f, .1f, .1f, 30.f, 300.f},
        {1.8f, .1f, .1f, 99.f, 999.f},
        {2.2f, 1.2f, 1.2f, 5.f, 6.f},
        {4.0f, .1f, .1f, 7.f, 8.f}
    };
    const bf_coord4 expected_coords[] = {{0,0,0,1},{0,0,0,0},{0,1,1,2}};
    const float expected_means[][5] = {
        {1.4f,.1f,.1f,20.f,200.f},
        {.25f,.1f,.1f,2.f,3.f},
        {2.2f,1.2f,1.2f,5.f,6.f}
    };
    bf_voxel_config cfg = {{0,0,0},{4,3,2},{1,1,1},5,3,8};
    char error[256] = {};
    bf_cuda_voxelizer *voxelizer = nullptr;
    float *device_points = nullptr, *device_means = nullptr;
    bf_coord4 *device_coords = nullptr;
    unsigned *device_count = nullptr;
    int ok = bf_cuda_voxelizer_create(&cfg, 32, &voxelizer,
                                       error, sizeof(error)) &&
             cudaMalloc(&device_points, sizeof(points)) == cudaSuccess &&
             cudaMalloc(&device_means, cfg.max_voxels * 5 * sizeof(float)) == cudaSuccess &&
             cudaMalloc(&device_coords, cfg.max_voxels * sizeof(bf_coord4)) == cudaSuccess &&
             cudaMalloc(&device_count, sizeof(unsigned)) == cudaSuccess &&
             cudaMemcpy(device_points, points, sizeof(points),
                        cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) ok = bf_cuda_voxelize_mean_f32(voxelizer, device_points,
        sizeof(points) / sizeof(points[0]), device_coords, device_means,
        device_count, nullptr, error, sizeof(error)) &&
        cudaDeviceSynchronize() == cudaSuccess;
    unsigned count = 0;
    bf_coord4 coords[8] = {};
    float means[40] = {};
    if (ok) ok = cudaMemcpy(&count, device_count, sizeof(count),
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
                 cudaMemcpy(coords, device_coords, sizeof(coords),
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
                 cudaMemcpy(means, device_means, sizeof(means),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) {
        ok = count == 3;
        for (unsigned i = 0; i < count && ok; ++i)
            ok = std::memcmp(coords + i, expected_coords + i,
                             sizeof(bf_coord4)) == 0 &&
                 close5(means + i * 5, expected_means[i]);
    }
    if (ok) {
        ok = bf_cuda_voxelize_mean_f32(voxelizer, device_points, 0,
            device_coords, device_means, device_count, nullptr,
            error, sizeof(error)) && cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(&count, device_count, sizeof(count),
                        cudaMemcpyDeviceToHost) == cudaSuccess && count == 0;
    }
    std::printf("cuda_voxel compact=%s resident=%.3f MiB\n", ok ? "pass" : "FAIL",
                bf_cuda_voxelizer_resident_bytes(voxelizer) / (1024.0 * 1024.0));
    if (!ok) std::fprintf(stderr, "CUDA voxel failure: %s count=%u\n", error, count);
    cudaFree(device_count);
    cudaFree(device_coords);
    cudaFree(device_means);
    cudaFree(device_points);
    bf_cuda_voxelizer_destroy(voxelizer);
    return ok ? 0 : 5;
}
