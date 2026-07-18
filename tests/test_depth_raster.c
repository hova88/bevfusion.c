#include "bf_depth_raster.h"
#include "bf_model.h"

#include <stdio.h>
#include <stdlib.h>

static const bf_tensor *tensor(const bf_model *model, const char *name) {
    const bf_tensor *value = bf_model_find(model, name);
    if (!value) fprintf(stderr, "missing fixture: %s\n", name);
    return value;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[256] = {0};
    bf_model *oracle = NULL;
    if (!bf_model_open(argv[1], &oracle, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 3;
    }
    const bf_tensor *expected = tensor(oracle, "depth_raster.output");
    float *output = expected ? (float *)malloc((size_t)expected->nbytes) : NULL;
    int ok = output && bf_depth_rasterize_f32_ref(
        tensor(oracle, "depth_raster.points")->data, 6, 5,
        tensor(oracle, "depth_raster.lidar_aug")->data,
        tensor(oracle, "depth_raster.lidar2image")->data,
        tensor(oracle, "depth_raster.image_aug")->data,
        output, 2, 2, 8, 12);
    size_t mismatches = 0;
    if (ok) {
        const float *reference = (const float *)expected->data;
        size_t count = (size_t)(expected->nbytes / sizeof(float));
        for (size_t i = 0; i < count; ++i)
            if (output[i] != reference[i]) {
                if (mismatches < 4)
                    fprintf(stderr, "depth raster[%zu]: got %.9g expected %.9g\n",
                            i, output[i], reference[i]);
                ++mismatches;
            }
        ok = mismatches == 0;
    }
    ok = ok && !bf_depth_rasterize_f32_ref(
        tensor(oracle, "depth_raster.points")->data, 6, 3,
        tensor(oracle, "depth_raster.lidar_aug")->data,
        tensor(oracle, "depth_raster.lidar2image")->data,
        tensor(oracle, "depth_raster.image_aug")->data,
        output, 2, 2, 8, 12);
    free(output);
    bf_model_close(oracle);
    if (!ok) return 4;
    puts("OpenPCDet lidar-to-image depth raster matches deterministic oracle exactly");
    return 0;
}
