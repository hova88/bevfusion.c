#include "bf_cuda.h"
#include "bf_model.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) std::fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

static float *upload(const bf_tensor *value) {
    float *device = nullptr;
    if (!value || cudaMalloc(&device, value->nbytes) != cudaSuccess ||
        cudaMemcpy(device, value->data, value->nbytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device); return nullptr;
    }
    return device;
}

__global__ static void initialize_values(float *values, size_t count, float scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = ((int)(i % 23) - 11) * scale;
}

__global__ static void initialize_geometry(float *geometry, size_t samples,
                                           int depths, int height, int width) {
    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    int x = sample % width;
    size_t q = sample / width;
    int y = q % height; q /= height;
    int depth = q % depths;
    int camera = (q / depths) % 6;
    int cell_x = (x * 4 + depth + camera * 13) % 360;
    int cell_y = (y * 9 + depth * 3 + camera * 17) % 360;
    geometry[sample * 3] = -54.0f + (cell_x + 0.5f) * 0.3f;
    geometry[sample * 3 + 1] = -54.0f + (cell_y + 0.5f) * 0.3f;
    geometry[sample * 3 + 2] = 0.0f;
}

__global__ static void initialize_frustum(float *frustum, size_t samples,
                                          int height, int width) {
    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    int x = sample % width;
    size_t q = sample / width;
    int y = q % height;
    int depth = q / height;
    int cell_x = (x * 4 + depth) % 360;
    int cell_y = (y * 9 + depth * 3) % 360;
    frustum[sample * 3] = -54.0f + (cell_x + 0.5f) * 0.3f;
    frustum[sample * 3 + 1] = -54.0f + (cell_y + 0.5f) * 0.3f;
    frustum[sample * 3 + 2] = 1.0f;
}

__global__ static void initialize_identity(float *matrices, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * 9) matrices[i] = i % 9 % 4 == 0 ? 1.0f : 0.0f;
}

static int benchmark_production(char *error, size_t cap) {
    bf_lss_desc desc = {1, 6, 118, 32, 88, 80,
        {-54.0f, -54.0f, -10.0f}, {0.3f, 0.3f, 20.0f}, {360, 360, 1}};
    size_t logits_count = 6ull * 118 * 32 * 88;
    size_t context_count = 6ull * 80 * 32 * 88;
    size_t samples = logits_count;
    size_t bev_count = 80ull * 360 * 360;
    float *logits = nullptr, *context = nullptr, *geometry = nullptr, *bev = nullptr;
    float *frustum = nullptr, *camera_rotation = nullptr, *camera_translation = nullptr;
    float *intrinsics = nullptr, *post_rotation = nullptr, *post_translation = nullptr;
    float *extra_rotation = nullptr, *extra_translation = nullptr;
    int *ranks = nullptr;
    int ok = cudaMalloc(&logits, logits_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&context, context_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&geometry, samples * 3 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&ranks, samples * sizeof(int)) == cudaSuccess &&
        cudaMalloc(&bev, bev_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&frustum, 118ull * 32 * 88 * 3 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&camera_rotation, 6 * 9 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&camera_translation, 6 * 3 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&intrinsics, 6 * 9 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&post_rotation, 6 * 9 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&post_translation, 6 * 3 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&extra_rotation, 9 * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&extra_translation, 3 * sizeof(float)) == cudaSuccess;
    if (ok) {
        initialize_values<<<(logits_count + 255) / 256, 256>>>(logits, logits_count, 0.03f);
        initialize_values<<<(context_count + 255) / 256, 256>>>(context, context_count, 0.01f);
        initialize_geometry<<<(samples + 255) / 256, 256>>>(geometry, samples, 118, 32, 88);
        initialize_frustum<<<(118 * 32 * 88 + 255) / 256, 256>>>(frustum, 118 * 32 * 88, 32, 88);
        initialize_identity<<<1, 64>>>(camera_rotation, 6);
        initialize_identity<<<1, 64>>>(intrinsics, 6);
        initialize_identity<<<1, 64>>>(post_rotation, 6);
        initialize_identity<<<1, 32>>>(extra_rotation, 1);
        cudaMemset(camera_translation, 0, 6 * 3 * sizeof(float));
        cudaMemset(post_translation, 0, 6 * 3 * sizeof(float));
        cudaMemset(extra_translation, 0, 3 * sizeof(float));
        ok = cudaDeviceSynchronize() == cudaSuccess;
    }
    cudaEvent_t begin, end; cudaEventCreate(&begin); cudaEventCreate(&end);
    float cold = 0.0f, warm = 0.0f, rank_build = 0.0f;
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_lift_pool_f32(logits, context, geometry, bev, &desc,
                                        nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&cold, begin, end);
    }
    if (ok) {
        cudaEventRecord(begin);
        for (int run = 0; run < 20; ++run)
            ok = ok && bf_cuda_lss_lift_pool_f32(logits, context, geometry, bev,
                                                  &desc, nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&warm, begin, end);
    }
    float ranked_cold = 0.0f, ranked_warm = 0.0f;
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_geometry_to_ranks_f32(geometry, ranks, &desc,
                                                nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&rank_build, begin, end);
    }
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_lift_pool_ranks_f32(logits, context, ranks, bev,
                                              &desc, nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&ranked_cold, begin, end);
    }
    if (ok) {
        cudaEventRecord(begin);
        for (int run = 0; run < 20; ++run)
            ok = ok && bf_cuda_lss_lift_pool_ranks_f32(
                logits, context, ranks, bev, &desc, nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&ranked_warm, begin, end);
    }
    bf_cuda_lss_plan *plan = nullptr;
    float plan_prepare = 0.0f, plan_cold = 0.0f, plan_warm = 0.0f;
    if (ok) ok = bf_cuda_lss_plan_create(&desc, &plan, error, cap);
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_plan_prepare_geometry_f32(plan, geometry, nullptr,
                                                   error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&plan_prepare, begin, end);
    }
    setenv("BF_CUDA_LSS_PLAN_NCHW", "1", 1);
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_plan_forward_f32(plan, logits, context, bev, nullptr,
                                           error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&plan_cold, begin, end);
    }
    if (ok) {
        cudaEventRecord(begin);
        for (int run = 0; run < 20; ++run)
            ok = ok && bf_cuda_lss_plan_forward_f32(
                plan, logits, context, bev, nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&plan_warm, begin, end);
    }
    unsetenv("BF_CUDA_LSS_PLAN_NCHW");
    float nhwc_cold = 0.0f, nhwc_warm = 0.0f;
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_plan_forward_f32(plan, logits, context, bev, nullptr,
                                           error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&nhwc_cold, begin, end);
    }
    if (ok) {
        cudaEventRecord(begin);
        for (int run = 0; run < 20; ++run)
            ok = ok && bf_cuda_lss_plan_forward_f32(
                plan, logits, context, bev, nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&nhwc_warm, begin, end);
    }
    float direct_prepare = 0.0f;
    if (ok) {
        cudaEventRecord(begin);
        ok = bf_cuda_lss_plan_prepare_calibration_f32(
            plan, frustum, camera_rotation, camera_translation, intrinsics,
            post_rotation, post_translation, extra_rotation, extra_translation,
            nullptr, error, cap);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&direct_prepare, begin, end);
    }
    std::printf("cuda_lss production baseline_cold=%.3f ms baseline_warm=%.3f ms "
                "rank_build=%.3f ms ranked_cold=%.3f ms ranked_warm=%.3f ms "
                "plan_prepare=%.3f ms plan_cold=%.3f ms plan_warm=%.3f ms "
                "nhwc_cold=%.3f ms nhwc_warm=%.3f ms "
                "direct_prepare=%.3f ms "
                "baseline_vram=%.2f MiB ranked_extra=%.2f MiB plan_resident=%.2f MiB\n",
                cold, warm / 20.0f, rank_build, ranked_cold, ranked_warm / 20.0f,
                plan_prepare, plan_cold, plan_warm / 20.0f,
                nhwc_cold, nhwc_warm / 20.0f,
                direct_prepare,
                (logits_count + context_count + samples * 3 + bev_count) *
                sizeof(float) / (1024.0 * 1024.0),
                samples * sizeof(int) / (1024.0 * 1024.0),
                bf_cuda_lss_plan_resident_bytes(plan) / (1024.0 * 1024.0));
    bf_cuda_lss_plan_destroy(plan);
    cudaEventDestroy(end); cudaEventDestroy(begin);
    cudaFree(extra_translation); cudaFree(extra_rotation); cudaFree(post_translation);
    cudaFree(post_rotation); cudaFree(intrinsics); cudaFree(camera_translation);
    cudaFree(camera_rotation); cudaFree(frustum); cudaFree(bev); cudaFree(ranks);
    cudaFree(geometry); cudaFree(context); cudaFree(logits);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[256] = {0};
    if (!bf_cuda_available(error, sizeof(error))) {
        std::fprintf(stderr, "%s\n", error); return 3;
    }
    bf_model *model = nullptr;
    if (!bf_model_open(argv[1], &model, error, sizeof(error))) {
        std::fprintf(stderr, "%s\n", error); return 4;
    }
    const bf_tensor *logits = tensor(model, "lss.depth_logits");
    const bf_tensor *context = tensor(model, "lss.context");
    const bf_tensor *geometry = tensor(model, "lss.geometry");
    const bf_tensor *expected = tensor(model, "lss.bev");
    const bf_tensor *frustum = tensor(model, "lss.frustum");
    const bf_tensor *camera_rotation = tensor(model, "lss.camera_rotation");
    const bf_tensor *camera_translation = tensor(model, "lss.camera_translation");
    const bf_tensor *intrinsics = tensor(model, "lss.intrinsics");
    const bf_tensor *post_rotation = tensor(model, "lss.post_rotation");
    const bf_tensor *post_translation = tensor(model, "lss.post_translation");
    const bf_tensor *extra_rotation = tensor(model, "lss.extra_rotation");
    const bf_tensor *extra_translation = tensor(model, "lss.extra_translation");
    float *d_logits = nullptr, *d_context = nullptr, *d_geometry = nullptr, *d_bev = nullptr;
    int *d_ranks = nullptr;
    float *d_frustum = upload(frustum), *d_camera_rotation = upload(camera_rotation);
    float *d_camera_translation = upload(camera_translation), *d_intrinsics = upload(intrinsics);
    float *d_post_rotation = upload(post_rotation), *d_post_translation = upload(post_translation);
    float *d_extra_rotation = upload(extra_rotation), *d_extra_translation = upload(extra_translation);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int ok = logits && context && geometry && expected && d_frustum &&
        d_camera_rotation && d_camera_translation && d_intrinsics &&
        d_post_rotation && d_post_translation && d_extra_rotation &&
        d_extra_translation &&
        cudaMalloc(&d_logits, logits->nbytes) == cudaSuccess &&
        cudaMalloc(&d_context, context->nbytes) == cudaSuccess &&
        cudaMalloc(&d_geometry, geometry->nbytes) == cudaSuccess &&
        cudaMalloc(&d_ranks, geometry->nbytes / 3) == cudaSuccess &&
        cudaMalloc(&d_bev, expected->nbytes) == cudaSuccess &&
        cudaMemcpy(d_logits, logits->data, logits->nbytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(d_context, context->data, context->nbytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(d_geometry, geometry->data, geometry->nbytes, cudaMemcpyHostToDevice) == cudaSuccess;
    bf_lss_desc desc = {2, 2, 4, 3, 5, 3,
        {-6.0f, -6.0f, -3.0f}, {2.0f, 2.0f, 2.0f}, {6, 6, 4}};
    if (ok) {
        ok = bf_cuda_lss_geometry_to_ranks_f32(
            d_geometry, d_ranks, &desc, nullptr, error, sizeof(error));
    }
    size_t samples = (size_t)(geometry->nbytes / (3 * sizeof(float)));
    int *host_ranks = (int *)std::malloc(samples * sizeof(int));
    if (ok) ok = host_ranks && cudaMemcpy(host_ranks, d_ranks,
                                           samples * sizeof(int),
                                           cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) {
        const float *points = (const float *)geometry->data;
        for (size_t i = 0; i < samples; ++i) {
            float gx = (points[i * 3] + 6.0f) / 2.0f;
            float gy = (points[i * 3 + 1] + 6.0f) / 2.0f;
            float gz = (points[i * 3 + 2] + 3.0f) / 2.0f;
            int x = (int)gx, y = (int)gy, z = (int)gz;
            int reference = std::isfinite(gx) && std::isfinite(gy) &&
                std::isfinite(gz) && (unsigned)x < 6 && (unsigned)y < 6 &&
                (unsigned)z < 4 ? (z * 6 + x) * 6 + y : -1;
            if (host_ranks[i] != reference) {
                std::fprintf(stderr, "rank[%zu]: got %d expected %d\n",
                             i, host_ranks[i], reference);
                ok = 0; break;
            }
        }
    }
    if (ok) {
        cudaEventRecord(start);
        for (int run = 0; run < 1000; ++run)
            ok = ok && bf_cuda_lss_lift_pool_ranks_f32(
                d_logits, d_context, d_ranks, d_bev, &desc, nullptr,
                error, sizeof(error));
        cudaEventRecord(stop); cudaEventSynchronize(stop);
    }
    float elapsed = 0.0f; cudaEventElapsedTime(&elapsed, start, stop);
    float *actual = (float *)std::malloc((size_t)expected->nbytes);
    ok = ok && actual && cudaMemcpy(actual, d_bev, expected->nbytes,
                                     cudaMemcpyDeviceToHost) == cudaSuccess;
    float maximum = 0.0f; double mean = 0.0;
    if (ok) {
        const float *reference = (const float *)expected->data;
        size_t count = (size_t)(expected->nbytes / sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            float difference = std::fabs(actual[i] - reference[i]);
            maximum = fmaxf(maximum, difference); mean += difference;
            if (!(difference <= 2e-6f + 8e-6f * std::fabs(reference[i]))) ok = 0;
        }
        std::printf("cuda_lss ranked max_abs=%.3g mean_abs=%.3g warm=%.4f ms\n",
                    maximum, mean / count, elapsed / 1000.0f);
    }
    bf_cuda_lss_plan *plan = nullptr;
    float plan_elapsed = 0.0f;
    if (ok) ok = bf_cuda_lss_plan_create(&desc, &plan, error, sizeof(error)) &&
        bf_cuda_lss_plan_prepare_geometry_f32(plan, d_geometry, nullptr,
                                               error, sizeof(error));
    if (ok) {
        cudaEventRecord(start);
        for (int run = 0; run < 1000; ++run)
            ok = ok && bf_cuda_lss_plan_forward_f32(
                plan, d_logits, d_context, d_bev, nullptr, error, sizeof(error));
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&plan_elapsed, start, stop);
    }
    if (ok) ok = cudaMemcpy(actual, d_bev, expected->nbytes,
                             cudaMemcpyDeviceToHost) == cudaSuccess;
    maximum = 0.0f; mean = 0.0;
    if (ok) {
        const float *reference = (const float *)expected->data;
        size_t count = (size_t)(expected->nbytes / sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            float difference = std::fabs(actual[i] - reference[i]);
            maximum = fmaxf(maximum, difference); mean += difference;
            if (!(difference <= 2e-6f + 8e-6f * std::fabs(reference[i]))) ok = 0;
        }
        std::printf("cuda_lss interval max_abs=%.3g mean_abs=%.3g warm=%.4f ms\n",
                    maximum, mean / count, plan_elapsed / 1000.0f);
    }
    if (ok) ok = bf_cuda_lss_plan_prepare_calibration_f32(
        plan, d_frustum, d_camera_rotation, d_camera_translation, d_intrinsics,
        d_post_rotation, d_post_translation, d_extra_rotation,
        d_extra_translation, nullptr, error, sizeof(error));
    if (ok) ok = bf_cuda_lss_plan_forward_f32(
        plan, d_logits, d_context, d_bev, nullptr, error, sizeof(error)) &&
        cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(actual, d_bev, expected->nbytes,
                   cudaMemcpyDeviceToHost) == cudaSuccess;
    maximum = 0.0f; mean = 0.0;
    if (ok) {
        const float *reference = (const float *)expected->data;
        size_t count = (size_t)(expected->nbytes / sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            float difference = std::fabs(actual[i] - reference[i]);
            maximum = fmaxf(maximum, difference); mean += difference;
            if (!(difference <= 2e-6f + 8e-6f * std::fabs(reference[i]))) ok = 0;
        }
        std::printf("cuda_lss direct-calibration+nhwc max_abs=%.3g mean_abs=%.3g\n",
                    maximum, mean / count);
    }
    bf_cuda_lss_plan_destroy(plan);
    if (ok) ok = benchmark_production(error, sizeof(error));
    if (!ok) std::fprintf(stderr, "CUDA LSS failure: %s\n", error);
    std::free(host_ranks); std::free(actual); cudaFree(d_bev); cudaFree(d_ranks); cudaFree(d_geometry);
    cudaFree(d_context); cudaFree(d_logits);
    cudaFree(d_extra_translation); cudaFree(d_extra_rotation);
    cudaFree(d_post_translation); cudaFree(d_post_rotation);
    cudaFree(d_intrinsics); cudaFree(d_camera_translation);
    cudaFree(d_camera_rotation); cudaFree(d_frustum);
    cudaEventDestroy(stop); cudaEventDestroy(start); bf_model_close(model);
    return ok ? 0 : 5;
}
