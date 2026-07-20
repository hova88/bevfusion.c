#define _POSIX_C_SOURCE 200809L
#include "bf_frame.h"
#include "bf_kernels.h"
#include "bf_model.h"
#include "bf_runtime.h"
#include "bf_tui.h"
#ifdef BF_WITH_CUDA
#include "bf_cuda_runtime.h"
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *program) {
    fprintf(stderr, "usage:\n  %s inspect MODEL.bfw\n"
                    "  %s frame-info FRAME.bfi\n"
                    "  %s plan MODEL.bfw POINT_CAP VOXEL_CAP SPARSE_CAP\n"
                    "  %s infer MODEL.bfw FRAME.bfi\n"
                    "  %s tui MODEL.bfw FRAME.bfi [FRAME.bfi ...]\n"
                    "  %s infer-cuda MODEL.bfw FRAME.bfi\n"
                    "  %s tui-cuda MODEL.bfw FRAME.bfi [FRAME.bfi ...]\n"
                    "  %s render-cuda MODEL.bfw INDEX TOTAL COLUMNS ROWS FRAME.bfi [FRAME.bfi ...]\n",
            program, program, program, program, program, program, program,
            program);
}

static double milliseconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec * 1000.0 + value.tv_nsec / 1000000.0;
}

static void close_frame_sequence(bf_frame_file **files, int count) {
    if (!files) return;
    for (int i = 0; i < count; ++i) bf_frame_close(files[i]);
    free(files);
}

static int run_frame(const bf_runtime *runtime, const bf_frame_file *file,
                     bf_detections *detections, double *elapsed,
                     char *error, size_t error_cap) {
    const bf_frame_input *input = bf_frame_input_view(file);
    size_t points = input->point_count > 300000 ? input->point_count : 300000;
    size_t bytes = bf_runtime_workspace_bytes(points, 160000, 160000);
    void *workspace = NULL;
    if (!bytes || posix_memalign(&workspace, 64, bytes) != 0) {
        snprintf(error, error_cap, "cannot allocate strict CPU workspace");
        return 0;
    }
    double start = milliseconds();
    int ok = bf_runtime_infer_cpu_ref(runtime, input, points, 160000, 160000,
                                      detections, workspace, bytes,
                                      error, error_cap);
    *elapsed = milliseconds() - start;
    free(workspace);
    return ok;
}

static void print_detections(const bf_detections *detections, double elapsed,
                             const char *backend) {
    printf("{\"backend\":\"%s\",\"inference_ms\":%.3f,\"detections\":[",
           backend, elapsed);
    for (int i = 0; i < detections->count; ++i) {
        const bf_detection *d = &detections->items[i];
        printf("%s{\"class_id\":%d,\"score\":%.9g,\"x\":%.9g,\"y\":%.9g,"
               "\"z\":%.9g,\"width\":%.9g,\"length\":%.9g,\"height\":%.9g,"
               "\"yaw\":%.9g,\"velocity_x\":%.9g,\"velocity_y\":%.9g}",
               i ? "," : "", d->class_id, d->score, d->x, d->y, d->z,
               d->width, d->length, d->height, d->yaw,
               d->velocity_x, d->velocity_y);
    }
    puts("]}");
}

static int infer_command(const char *model_path, const char *frame_path) {
    char error[256] = {0};
    bf_runtime *runtime = NULL;
    bf_frame_file *file = NULL;
    bf_detections detections = {0};
    double elapsed = 0.0;
    int ok = bf_runtime_create(model_path, &runtime, error, sizeof(error)) &&
        bf_frame_open(frame_path, &file, error, sizeof(error)) &&
        run_frame(runtime, file, &detections, &elapsed, error, sizeof(error));
    if (ok) print_detections(&detections, elapsed, "cpu-ref");
    else fprintf(stderr, "bevfusion: %s\n", error);
    bf_frame_close(file); bf_runtime_destroy(runtime);
    return ok ? 0 : 1;
}

#ifdef BF_WITH_CUDA
static int run_cuda_frame(bf_cuda_runtime *runtime, const bf_frame_file *file,
                          bf_detections *detections, double *elapsed,
                          char *error, size_t error_cap) {
    double start = milliseconds();
    int ok = bf_cuda_runtime_infer(runtime, bf_frame_input_view(file), detections,
                                   error, error_cap);
    *elapsed = milliseconds() - start;
    return ok;
}

static int infer_cuda_command(const char *model_path, const char *frame_path) {
    char error[256] = {0}; bf_cuda_runtime *runtime = NULL;
    bf_frame_file *file = NULL; bf_detections detections = {0}; double elapsed = 0;
    int ok = bf_cuda_runtime_create(model_path, 300000, 160000, 160000,
                                    &runtime, error, sizeof(error)) &&
        bf_frame_open(frame_path, &file, error, sizeof(error)) &&
        run_cuda_frame(runtime, file, &detections, &elapsed, error, sizeof(error));
    if (ok) print_detections(&detections, elapsed, "cuda-strict");
    else fprintf(stderr, "bevfusion: %s\n", error);
    bf_frame_close(file); bf_cuda_runtime_destroy(runtime); return ok ? 0 : 1;
}

static int tui_cuda_command(const char *model_path, int frame_count, char **paths) {
    char error[256] = {0}; bf_cuda_runtime *runtime = NULL;
    bf_detections *all = calloc((size_t)frame_count, sizeof(*all));
    double *timings = calloc((size_t)frame_count, sizeof(*timings));
    bf_frame_file **files = calloc((size_t)frame_count, sizeof(*files));
    if (!all || !timings || !files || !bf_cuda_runtime_create(model_path, 300000, 160000,
            160000, &runtime, error, sizeof(error))) goto failure;
    for (int i = 0; i < frame_count; ++i) {
        if (!bf_frame_open(paths[i], &files[i], error, sizeof(error)) ||
            !run_cuda_frame(runtime, files[i], &all[i], &timings[i], error, sizeof(error)))
            goto failure;
    }
    bf_tui_state state;
    if (!bf_tui_begin(&state)) { snprintf(error, sizeof(error), "TUI requires an interactive terminal"); goto failure; }
    size_t current = 0; bf_tui_update_tracks(&state, &all[current], current);
    const bf_frame_input *input = bf_frame_input_view(files[current]);
    bf_tui_render(input->points, input->point_count, 5, &all[current], current,
                  (size_t)frame_count, timings[current], "cuda-strict", &state);
    for (;;) {
        int action = bf_tui_poll(&state, state.paused ? -1 : 500);
        if (action == BF_TUI_QUIT) break;
        size_t prior = current;
        if (action == BF_TUI_NEXT || (action == BF_TUI_NONE && !state.paused)) current = (current + 1) % (size_t)frame_count;
        else if (action == BF_TUI_PREV) current = (current + (size_t)frame_count - 1) % (size_t)frame_count;
        if (current != prior) bf_tui_update_tracks(&state, &all[current], current);
        if (action != BF_TUI_NONE || current != prior) {
            input = bf_frame_input_view(files[current]);
            bf_tui_render(input->points, input->point_count, 5, &all[current],
                          current, (size_t)frame_count, timings[current],
                          "cuda-strict", &state);
        }
    }
    bf_tui_end(); close_frame_sequence(files, frame_count); free(timings);
    free(all); bf_cuda_runtime_destroy(runtime); return 0;
failure:
    fprintf(stderr, "bevfusion: %s\n", error[0] ? error : "allocation failed");
    close_frame_sequence(files, frame_count); free(timings); free(all);
    bf_cuda_runtime_destroy(runtime); return 1;
}

static int render_cuda_command(const char *model_path, int frame_index,
                               int frame_total, int columns, int rows,
                               int frame_count, char **paths) {
    if (columns < 32 || columns > 200 || rows < 10 || rows > 80 ||
        frame_count < 1 || frame_index < 0 || frame_total < 1 ||
        frame_index >= frame_total) {
        fputs("bevfusion: invalid render-cuda index, total, terminal size, or frames\n",
              stderr);
        return 2;
    }
    char error[256] = {0}; bf_cuda_runtime *runtime = NULL;
    bf_frame_file *file = NULL; bf_detections detections = {0};
    bf_tui_state state; bf_tui_state_init(&state);
    state.show_sidebar = 1; state.paused = 1;
    double elapsed = 0.0; int ok = bf_cuda_runtime_create(model_path, 300000,
        160000, 160000, &runtime, error, sizeof(error));
    for (int i = 0; i < frame_count && ok; ++i) {
        bf_frame_close(file); file = NULL;
        ok = bf_frame_open(paths[i], &file, error, sizeof(error)) &&
             run_cuda_frame(runtime, file, &detections, &elapsed,
                            error, sizeof(error));
        if (ok) bf_tui_update_tracks(&state, &detections, (size_t)i);
    }
    bf_tui_frame output = {0};
    if (ok) {
        const bf_frame_input *input = bf_frame_input_view(file);
        ok = bf_tui_compose(input->points, input->point_count, 5, &detections,
                            (size_t)frame_index, (size_t)frame_total,
                            elapsed, "cuda-strict / nuScenes", &state,
                            columns, rows, &output);
    }
    if (ok && fwrite(output.data, 1, output.length, stdout) != output.length)
        ok = 0;
    if (!ok) fprintf(stderr, "bevfusion: %s\n", error[0] ? error : "render failed");
    bf_tui_frame_free(&output); bf_frame_close(file);
    bf_cuda_runtime_destroy(runtime); return ok ? 0 : 1;
}
#endif

static int tui_command(const char *model_path, int frame_count, char **paths) {
    char error[256] = {0};
    bf_runtime *runtime = NULL;
    bf_detections *all = calloc((size_t)frame_count, sizeof(*all));
    double *timings = calloc((size_t)frame_count, sizeof(*timings));
    bf_frame_file **files = calloc((size_t)frame_count, sizeof(*files));
    if (!all || !timings || !files || !bf_runtime_create(model_path, &runtime,
                                                error, sizeof(error))) goto failure;
    for (int i = 0; i < frame_count; ++i) {
        if (!bf_frame_open(paths[i], &files[i], error, sizeof(error)) ||
            !run_frame(runtime, files[i], &all[i], &timings[i], error, sizeof(error)))
            goto failure;
    }
    bf_tui_state state;
    if (!bf_tui_begin(&state)) {
        snprintf(error, sizeof(error), "TUI requires an interactive terminal");
        goto failure;
    }
    size_t current = 0;
    bf_tui_update_tracks(&state, &all[current], current);
    const bf_frame_input *input = bf_frame_input_view(files[current]);
    bf_tui_render(input->points, input->point_count, 5, &all[current], current,
                  (size_t)frame_count, timings[current], "cpu-ref", &state);
    for (;;) {
        int action = bf_tui_poll(&state, state.paused ? -1 : 500);
        if (action == BF_TUI_QUIT) break;
        size_t prior = current;
        if (action == BF_TUI_NEXT || (action == BF_TUI_NONE && !state.paused))
            current = (current + 1) % (size_t)frame_count;
        else if (action == BF_TUI_PREV)
            current = (current + (size_t)frame_count - 1) % (size_t)frame_count;
        if (current != prior) bf_tui_update_tracks(&state, &all[current], current);
        if (action != BF_TUI_NONE || current != prior) {
            input = bf_frame_input_view(files[current]);
            bf_tui_render(input->points, input->point_count, 5, &all[current],
                          current, (size_t)frame_count, timings[current],
                          "cpu-ref", &state);
        }
    }
    bf_tui_end(); close_frame_sequence(files, frame_count); free(timings);
    free(all); bf_runtime_destroy(runtime); return 0;
failure:
    fprintf(stderr, "bevfusion: %s\n", error[0] ? error : "allocation failed");
    close_frame_sequence(files, frame_count); free(timings); free(all);
    bf_runtime_destroy(runtime); return 1;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "frame-info") == 0) {
        char error[256] = {0};
        bf_frame_file *file = NULL;
        if (!bf_frame_open(argv[2], &file, error, sizeof(error))) {
            fprintf(stderr, "bevfusion: %s\n", error); return 1;
        }
        printf("BFI v%u: %zu points, %.2f MiB, canonical layout and CRC valid\n",
               BF_FRAME_VERSION, bf_frame_input_view(file)->point_count,
               (double)bf_frame_file_bytes(file) / (1024.0 * 1024.0));
        bf_frame_close(file); return 0;
    }
    if (argc == 6 && strcmp(argv[1], "plan") == 0) {
        size_t values[3];
        for (size_t i = 0; i < 3; ++i) {
            char *end = NULL;
            errno = 0;
            unsigned long long parsed = strtoull(argv[i + 3], &end, 10);
            if (errno || !end || *end || parsed > SIZE_MAX) {
                fprintf(stderr, "bevfusion: invalid capacity %s\n", argv[i + 3]);
                return 2;
            }
            values[i] = (size_t)parsed;
        }
        char error[256];
        bf_runtime *runtime = NULL;
        if (!bf_runtime_create(argv[2], &runtime, error, sizeof(error))) {
            fprintf(stderr, "bevfusion: %s\n", error); return 1;
        }
        size_t bytes = bf_runtime_workspace_bytes(values[0], values[1], values[2]);
        bf_runtime_destroy(runtime);
        if (!bytes) { fprintf(stderr, "bevfusion: invalid or overflowing resource plan\n"); return 1; }
        printf("strict CPU workspace: %zu bytes (%.2f MiB)\n",
               bytes, (double)bytes / (1024.0 * 1024.0));
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "infer") == 0)
        return infer_command(argv[2], argv[3]);
    if (argc >= 4 && strcmp(argv[1], "tui") == 0)
        return tui_command(argv[2], argc - 3, argv + 3);
#ifdef BF_WITH_CUDA
    if (argc == 4 && strcmp(argv[1], "infer-cuda") == 0)
        return infer_cuda_command(argv[2], argv[3]);
    if (argc >= 4 && strcmp(argv[1], "tui-cuda") == 0)
        return tui_cuda_command(argv[2], argc - 3, argv + 3);
    if (argc >= 8 && strcmp(argv[1], "render-cuda") == 0) {
        char *ends[4] = {NULL, NULL, NULL, NULL}; long values[4];
        for (int i = 0; i < 4; ++i) values[i] = strtol(argv[i + 3], &ends[i], 10);
        if (!ends[0] || *ends[0] || !ends[1] || *ends[1] ||
            !ends[2] || *ends[2] || !ends[3] || *ends[3])
            return 2;
        return render_cuda_command(argv[2], (int)values[0], (int)values[1],
                                   (int)values[2], (int)values[3],
                                   argc - 7, argv + 7);
    }
#endif
    if (argc != 3 || strcmp(argv[1], "inspect") != 0) {
        usage(argv[0]); return 2;
    }
    char error[256];
    bf_model *model = NULL;
    if (!bf_model_open(argv[2], &model, error, sizeof(error)) ||
        !bf_model_validate_all(model, error, sizeof(error))) {
        fprintf(stderr, "bevfusion: %s\n", error); bf_model_close(model); return 1;
    }
    printf("BFW v%u: %zu tensors, %.2f MiB, all CRCs valid; CPU kernels: %s\n",
           BF_MODEL_VERSION, bf_model_tensor_count(model),
           (double)bf_model_file_bytes(model) / (1024.0 * 1024.0),
           bf_cpu_kernel_backend());
    bf_model_close(model); return 0;
}
