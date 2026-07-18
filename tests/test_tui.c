#include "bf_tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int validate_layout(const bf_tui_frame *frame, int columns, int rows) {
    int row = 0, width = 0;
    const unsigned char *cursor = (const unsigned char *)frame->data;
    const unsigned char *end = cursor + frame->length;
    while (cursor < end) {
        if (*cursor == 27 && cursor + 1 < end && cursor[1] == '[') {
            cursor += 2;
            while (cursor < end && !(*cursor >= '@' && *cursor <= '~')) ++cursor;
            if (cursor < end) ++cursor;
            continue;
        }
        if (*cursor == '\n') {
            if (width > columns) return 0;
            ++row; width = 0; ++cursor; continue;
        }
        if ((*cursor & 0xc0u) != 0x80u) ++width;
        ++cursor;
    }
    return row + 1 == rows && width <= columns;
}

int main(void) {
    float points[] = {
        12, 3, -1.5f, .7f, 0,
        12, 3, .2f, .8f, 0,
        -8, -7, 1.4f, .9f, 0,
        30, 20, 0, .03f, .8f,
        0, 0, -.2f, 1, 0
    };
    bf_detections detections = {0};
    detections.count = 3;
    detections.items[0] = (bf_detection){12, 3, 0, 2.0f, 4.5f, 1.7f,
        .2f, 4, .5f, .91f, 0};
    detections.items[1] = (bf_detection){-8, -7, 0, .7f, .8f, 1.8f,
        -.5f, 0, 0, .78f, 8};
    detections.items[2] = (bf_detection){30, 20, 0, 2.5f, 10, 3,
        1.2f, -1, 2, .12f, 1};
    bf_tui_state state;
    bf_tui_state_init(&state);
    bf_tui_update_tracks(&state, &detections, 0);
    detections.items[0].x += 1.0f;
    bf_tui_update_tracks(&state, &detections, 1);
    if (!state.track_count || state.tracks[0].length < 2) return 1;
    state.show_sidebar = 1;
    state.show_help = 0;
    bf_tui_frame first = {0}, second = {0}, compact = {0};
    int ok = bf_tui_compose(points, 5, 5, &detections, 1, 20, 18.25, "CPU strict",
                            &state, 140, 32, &first) &&
             bf_tui_compose(points, 5, 5, &detections, 1, 20, 18.25, "CPU strict",
                            &state, 140, 32, &second) &&
             bf_tui_compose(points, 5, 5, &detections, 1, 20, 18.25, "CPU strict",
                            &state, 36, 10, &compact) &&
             first.length == second.length &&
             !memcmp(first.data, second.data, first.length) &&
             validate_layout(&first, 140, 32) &&
             validate_layout(&compact, 36, 10) &&
             (strstr(first.data, "LiDAR OCC + BOXES") ||
              strstr(first.data, "LiDAR OCC + DETECTIONS")) &&
             strstr(first.data, "OCC 5") && strstr(first.data, "INSPECTOR") &&
             strstr(first.data, "FOCUS") && strstr(first.data, "position") &&
             strstr(first.data, "OCC height + return density") &&
             strstr(first.data, "velocity") &&
             strstr(compact.data, "BEVFUSION") &&
             !strstr(first.data, "point cloud") && !strstr(first.data, "perspective");
    const int widths[] = {20, 31, 32, 36, 79, 80, 110, 160};
    const int heights[] = {5, 9, 10, 23, 24, 32, 40};
    for (size_t wi = 0; ok && wi < sizeof(widths) / sizeof(widths[0]); ++wi)
        for (size_t hi = 0; ok && hi < sizeof(heights) / sizeof(heights[0]); ++hi) {
            bf_tui_frame responsive = {0};
            ok = bf_tui_compose(points, 5, 5, &detections, 1, 20, 18.25,
                                "CUDA strict", &state, widths[wi], heights[hi],
                                &responsive) &&
                 validate_layout(&responsive, widths[wi], heights[hi]);
            if (!ok) fprintf(stderr, "responsive fixture %dx%d failed\n",
                             widths[wi], heights[hi]);
            bf_tui_frame_free(&responsive);
        }
    bf_tui_state visual;
    bf_tui_state_init(&visual);
    visual.show_grid = visual.show_boxes = visual.show_velocity = visual.show_tracks = 0;
    float sparse_high[] = {8, 2, 1.5f, 1, 0};
    float dense_high[16 * 5];
    for (size_t i = 0; i < 16; ++i)
        memcpy(dense_high + i * 5, sparse_high, 5 * sizeof(float));
    bf_tui_frame sparse = {0}, dense = {0}, hidden = {0};
    ok = ok && bf_tui_compose(sparse_high, 1, 5, NULL, 0, 1, 1.0,
                              "CPU", &visual, 80, 24, &sparse) &&
         bf_tui_compose(dense_high, 16, 5, NULL, 0, 1, 1.0,
                        "CPU", &visual, 80, 24, &dense) &&
         strstr(sparse.data, "\033[2;38;5;118;48;5;233m") &&
         strstr(dense.data, "\033[1;38;5;118;48;5;233m");
    visual.show_occupancy = 0;
    ok = ok && bf_tui_compose(dense_high, 16, 5, NULL, 0, 1, 1.0,
                              "CPU", &visual, 80, 24, &hidden) &&
         !strstr(hidden.data, "\033[1;38;5;118;48;5;233m");
    bf_tui_frame_free(&sparse); bf_tui_frame_free(&dense); bf_tui_frame_free(&hidden);
    state.input[0] = 'b'; state.input_length = 1;
    ok = ok && bf_tui_poll(&state, 0) == BF_TUI_REDRAW && !state.show_boxes;
    state.input[0] = 'o'; state.input_length = 1;
    ok = ok && bf_tui_poll(&state, 0) == BF_TUI_REDRAW && !state.show_occupancy;
    state.input[0] = 'm'; state.input_length = 1;
    ok = ok && bf_tui_poll(&state, 0) == BF_TUI_NONE;
    ok = ok && !bf_tui_compose(points, 1, 2, &detections, 0, 1, 1.0,
                                "CPU", &state, 80, 24, &second);
    if (!ok) {
        fprintf(stderr, "tui fixture failed: first=%p second=%p compact=%p "
                "title=%d occ=%d inspector=%d focus=%d position=%d legend=%d "
                "velocity=%d compact-title=%d boxes=%d occupancy=%d\n",
                (void *)first.data, (void *)second.data, (void *)compact.data,
                first.data && !!(strstr(first.data, "LiDAR OCC + BOXES") ||
                                  strstr(first.data, "LiDAR OCC + DETECTIONS")),
                first.data && !!strstr(first.data, "OCC 5"),
                first.data && !!strstr(first.data, "INSPECTOR"),
                first.data && !!strstr(first.data, "FOCUS"),
                first.data && !!strstr(first.data, "position"),
                first.data && !!strstr(first.data, "OCC height + return density"),
                first.data && !!strstr(first.data, "velocity"),
                compact.data && !!strstr(compact.data, "BEVFUSION"),
                state.show_boxes, state.show_occupancy);
    }
    bf_tui_frame_free(&first); bf_tui_frame_free(&second); bf_tui_frame_free(&compact);
    if (!ok) return 2;
    puts("BEV occupancy, filled boxes, inspector, controls, and responsive layout pass");
    return 0;
}
