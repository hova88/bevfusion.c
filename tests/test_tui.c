#include "bf_tui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
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
    state.show_help = 1;
    bf_tui_frame first = {0}, second = {0}, compact = {0};
    int ok = bf_tui_compose(&detections, 1, 20, 18.25, "CPU strict",
                            &state, 110, 32, &first) &&
             bf_tui_compose(&detections, 1, 20, 18.25, "CPU strict",
                            &state, 110, 32, &second) &&
             bf_tui_compose(&detections, 1, 20, 18.25, "CPU strict",
                            &state, 36, 10, &compact) &&
             first.length == second.length &&
             !memcmp(first.data, second.data, first.length) &&
             strstr(first.data, "BEV ONLY") && strstr(first.data, "INSPECTOR") &&
             strstr(first.data, "velocity") && strstr(compact.data, "BEV ONLY") &&
             !strstr(first.data, "point cloud") && !strstr(first.data, "perspective");
    state.input[0] = 'b'; state.input_length = 1;
    ok = ok && bf_tui_poll(&state, 0) == BF_TUI_REDRAW && !state.show_boxes;
    state.input[0] = 'm'; state.input_length = 1;
    ok = ok && bf_tui_poll(&state, 0) == BF_TUI_NONE;
    bf_tui_frame_free(&first); bf_tui_frame_free(&second); bf_tui_frame_free(&compact);
    if (!ok) return 2;
    puts("BEV-only TUI composition, tracks, controls, and responsive layout pass");
    return 0;
}
