#include "bf_runtime.h"

#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[256] = {0};
    bf_runtime *runtime = NULL;
    if (!bf_runtime_create(argv[1], &runtime, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 3;
    }
    size_t bytes = bf_runtime_workspace_bytes(300000, 160000, 160000);
    int ok = bytes > 0 &&
        bf_runtime_workspace_bytes(300000, 160001, 160001) == 0 &&
        bf_runtime_workspace_bytes(SIZE_MAX, 160000, 160000) == 0 &&
        bf_runtime_workspace_bytes(300000, 160000, 159999) == 0;
    bf_frame_input invalid = {0};
    bf_detections detections;
    ok = ok && !bf_runtime_infer_cpu_ref(runtime, &invalid, 300000, 160000,
                                          160000, &detections, NULL, 0,
                                          error, sizeof(error));
    printf("strict CPU frame arena: %.2f MiB\n", (double)bytes / (1024.0 * 1024.0));
    bf_runtime_destroy(runtime);
    if (!ok) return 4;
    puts("complete runtime binding, resource preflight, and invalid-frame gate pass");
    return 0;
}
