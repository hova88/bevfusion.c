#ifndef BF_TUI_H
#define BF_TUI_H

#include "bevfusion.h"

#include <stddef.h>

#define BF_TUI_MAX_TRACKS 64
#define BF_TUI_TRAIL 12

typedef struct {
    float x[BF_TUI_TRAIL], y[BF_TUI_TRAIL];
    unsigned char length, class_id, missed;
    int id;
} bf_tui_track;

typedef struct {
    float center_x, center_y, zoom, yaw, score_threshold;
    unsigned class_mask;
    int paused, show_occupancy, show_boxes, show_velocity, show_grid, show_tracks;
    int show_help, show_sidebar, selected;
    bf_tui_track tracks[BF_TUI_MAX_TRACKS];
    size_t track_count, last_frame;
    int next_track_id, have_last_frame;
    unsigned char input[32];
    size_t input_length;
} bf_tui_state;

typedef struct { char *data; size_t length; } bf_tui_frame;
enum { BF_TUI_NONE, BF_TUI_REDRAW, BF_TUI_NEXT, BF_TUI_PREV, BF_TUI_QUIT };

void bf_tui_state_init(bf_tui_state *state);
int bf_tui_begin(bf_tui_state *state);
void bf_tui_end(void);
int bf_tui_poll(bf_tui_state *state, int timeout_ms);
void bf_tui_update_tracks(bf_tui_state *state,
                          const bf_detections *detections, size_t frame);
int bf_tui_compose(const float *points, size_t point_count, size_t point_stride,
                   const bf_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const bf_tui_state *state,
                   int columns, int rows, bf_tui_frame *output);
void bf_tui_frame_free(bf_tui_frame *frame);
void bf_tui_render(const float *points, size_t point_count, size_t point_stride,
                   const bf_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const bf_tui_state *state);

#endif
