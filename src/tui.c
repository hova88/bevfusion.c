#define _POSIX_C_SOURCE 200809L
#include "bf_tui.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

static const char *const class_names[10] = {
    "car", "truck", "construction", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic cone"
};
static const int class_colors[10] = {51, 208, 220, 39, 141, 244, 201, 45, 82, 214};
enum {
    PIX_GRID = 1, PIX_AXIS, PIX_OCC_OLD, PIX_OCC_GROUND, PIX_OCC_LOW,
    PIX_OCC_HIGH, PIX_TRACK_BASE = 10, PIX_VELOCITY_BASE = 20,
    PIX_BOX_FILL_BASE = 30, PIX_BOX_EDGE_BASE = 40, PIX_BOX_FRONT_BASE = 50,
    PIX_SELECTED_FILL = 60, PIX_SELECTED_EDGE = 70, PIX_EGO = 80
};
static struct termios saved_termios;
static int terminal_active;
static volatile sig_atomic_t caught_signal, resize_pending;
static const char restore_sequence[] = "\033[0m\033[?25h\033[?1049l";

typedef struct { char *data; size_t length, capacity; int failed; } text_buffer;

static void write_all(int descriptor, const char *data, size_t length) {
    while (length) {
        ssize_t written = write(descriptor, data, length);
        if (written > 0) { data += written; length -= (size_t)written; }
        else if (written < 0 && errno == EINTR) continue;
        else break;
    }
}

static int reserve(text_buffer *b, size_t extra) {
    if (!b || b->failed || extra > SIZE_MAX - b->length - 1) return 0;
    size_t needed = b->length + extra + 1;
    if (needed <= b->capacity) return 1;
    size_t capacity = b->capacity ? b->capacity : 4096;
    while (capacity < needed && capacity <= SIZE_MAX / 2) capacity *= 2;
    if (capacity < needed) capacity = needed;
    char *grown = realloc(b->data, capacity);
    if (!grown) { b->failed = 1; return 0; }
    b->data = grown; b->capacity = capacity; return 1;
}

static void putn(text_buffer *b, const char *s, size_t n) {
    if (!reserve(b, n)) return;
    memcpy(b->data + b->length, s, n); b->length += n; b->data[b->length] = 0;
}
static void puts_b(text_buffer *b, const char *s) { putn(b, s, strlen(s)); }
static void spaces_b(text_buffer *b, int count) {
    for (int i = 0; i < count; ++i) putn(b, " ", 1);
}
static void fixed_ascii(text_buffer *b, const char *style, const char *s,
                        int width, int newline) {
    size_t length = strlen(s);
    size_t used = length < (size_t)width ? length : (size_t)width;
    puts_b(b, style); putn(b, s, used);
    spaces_b(b, width - (int)used); puts_b(b, "\033[K");
    if (newline) putn(b, "\n", 1);
}
static void printf_b(text_buffer *b, const char *format, ...) {
    va_list args, copy; va_start(args, format); va_copy(copy, args);
    int n = vsnprintf(NULL, 0, format, copy); va_end(copy);
    if (n < 0 || !reserve(b, (size_t)n)) { b->failed = 1; va_end(args); return; }
    vsnprintf(b->data + b->length, b->capacity - b->length, format, args);
    va_end(args); b->length += (size_t)n;
}

static void restore_terminal(void) {
    if (!terminal_active) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_termios);
    write_all(STDOUT_FILENO, restore_sequence, sizeof(restore_sequence) - 1);
    terminal_active = 0;
}
static void interrupted(int number) { caught_signal = number; }
static void resized(int number) { (void)number; resize_pending = 1; }

static void reset_view(bf_tui_state *s) {
    s->center_x = 0.0f; s->center_y = 0.0f; s->zoom = 1.0f; s->yaw = 0.0f;
}

void bf_tui_state_init(bf_tui_state *s) {
    if (!s) return;
    memset(s, 0, sizeof(*s)); reset_view(s);
    s->score_threshold = 0.2f; s->class_mask = 0x3ffu;
    s->show_occupancy = s->show_boxes = s->show_velocity = 1;
    s->show_grid = s->show_tracks = 1;
    s->next_track_id = 1;
}

int bf_tui_begin(bf_tui_state *s) {
    if (!s || !isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return 0;
    bf_tui_state_init(s);
    if (tcgetattr(STDIN_FILENO, &saved_termios)) return 0;
    struct termios raw = saved_termios;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL);
    raw.c_cc[VMIN] = raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)) return 0;
    terminal_active = 1; caught_signal = resize_pending = 0;
    atexit(restore_terminal);
    signal(SIGINT, interrupted); signal(SIGTERM, interrupted);
    signal(SIGHUP, interrupted); signal(SIGQUIT, interrupted);
    signal(SIGWINCH, resized);
    fputs("\033[?1049h\033[?25l\033[2J", stdout); fflush(stdout);
    return 1;
}
void bf_tui_end(void) { restore_terminal(); }

static void consume(bf_tui_state *s, size_t n) {
    s->input_length -= n; memmove(s->input, s->input + n, s->input_length);
}

static int incomplete_escape(const bf_tui_state *s) {
    return s->input_length && s->input[0] == 27 &&
        (s->input_length == 1 ||
         (s->input_length == 2 && s->input[1] == '['));
}

int bf_tui_poll(bf_tui_state *s, int timeout_ms) {
    if (!s || caught_signal) return BF_TUI_QUIT;
    if (resize_pending) { resize_pending = 0; return BF_TUI_REDRAW; }
    if (!s->input_length || incomplete_escape(s)) {
        struct pollfd fd = {STDIN_FILENO, POLLIN, 0};
        int result;
        do result = poll(&fd, 1, timeout_ms); while (result < 0 && errno == EINTR);
        if (caught_signal) return BF_TUI_QUIT;
        if (resize_pending) { resize_pending = 0; return BF_TUI_REDRAW; }
        if (result <= 0 || !(fd.revents & POLLIN)) {
            if (incomplete_escape(s)) s->input_length = 0;
            return BF_TUI_NONE;
        }
        size_t available = sizeof(s->input) - s->input_length;
        if (!available) { consume(s, 1); available = sizeof(s->input) - s->input_length; }
        ssize_t n = read(STDIN_FILENO, s->input + s->input_length, available);
        if (n <= 0) return BF_TUI_NONE;
        s->input_length += (size_t)n;
    }
    if (incomplete_escape(s)) return BF_TUI_NONE;
    unsigned char value = s->input[0], arrow = 0;
    if (value == 27 && s->input_length >= 3 && s->input[1] == '[') {
        arrow = s->input[2]; consume(s, 3);
    } else consume(s, 1);
    if (value == 3 || value == 'q' || value == 'Q') return BF_TUI_QUIT;
    if (value == ' ') { s->paused = !s->paused; return BF_TUI_REDRAW; }
    if (value == 'n' || value == 'N' || arrow == 'C') return BF_TUI_NEXT;
    if (value == 'p' || value == 'P' || arrow == 'D') return BF_TUI_PREV;
    float pan = 5.0f / s->zoom;
    if (value == 'w' || arrow == 'A') s->center_x += pan;
    else if (value == 's' || arrow == 'B') s->center_x -= pan;
    else if (value == 'a') s->center_y += pan;
    else if (value == 'd') s->center_y -= pan;
    else if (value == 'e') s->yaw += 0.1308997f;
    else if (value == 'z') s->yaw -= 0.1308997f;
    else if (value == '+' || value == '=') s->zoom = fminf(8.0f, s->zoom * 1.25f);
    else if (value == '-' || value == '_') s->zoom = fmaxf(0.3f, s->zoom / 1.25f);
    else if (value == 'b' || value == 'B') s->show_boxes = !s->show_boxes;
    else if (value == 'o' || value == 'O') s->show_occupancy = !s->show_occupancy;
    else if (value == 'v' || value == 'V') s->show_velocity = !s->show_velocity;
    else if (value == 'g' || value == 'G') s->show_grid = !s->show_grid;
    else if (value == 't' || value == 'T') s->show_tracks = !s->show_tracks;
    else if (value == 'i' || value == 'I') s->show_sidebar = !s->show_sidebar;
    else if (value == 'h' || value == 'H' || value == '?') { s->show_help = !s->show_help; s->show_sidebar = 1; }
    else if (value == 'c' || value == 'C') s->class_mask = 0x3ffu;
    else if (value >= '0' && value <= '9') s->class_mask ^= 1u << (value - '0');
    else if (value == '[' && s->selected > 0) --s->selected;
    else if (value == ']') ++s->selected;
    else if (value == ',') s->score_threshold = fmaxf(0.0f, s->score_threshold - 0.05f);
    else if (value == '.') s->score_threshold = fminf(0.95f, s->score_threshold + 0.05f);
    else if (value == 'r' || value == 'R') reset_view(s);
    else return BF_TUI_NONE;
    return BF_TUI_REDRAW;
}

static int visible(const bf_detection *d, const bf_tui_state *s) {
    return d && d->class_id >= 0 && d->class_id < 10 &&
        d->score >= s->score_threshold && (s->class_mask & (1u << d->class_id));
}

void bf_tui_update_tracks(bf_tui_state *s, const bf_detections *d, size_t frame) {
    if (!s || !d) return;
    if (s->have_last_frame && frame != s->last_frame + 1) s->track_count = 0;
    unsigned char used[BF_TUI_MAX_TRACKS] = {0};
    for (int i = 0; i < d->count; ++i) {
        const bf_detection *box = &d->items[i];
        size_t best = SIZE_MAX; float best_distance = 64.0f;
        for (size_t t = 0; t < s->track_count; ++t) {
            bf_tui_track *track = &s->tracks[t];
            if (used[t] || track->class_id != box->class_id || !track->length) continue;
            float dx = track->x[track->length - 1] - box->x;
            float dy = track->y[track->length - 1] - box->y;
            float distance = dx * dx + dy * dy;
            if (distance < best_distance) { best_distance = distance; best = t; }
        }
        if (best == SIZE_MAX && s->track_count < BF_TUI_MAX_TRACKS) {
            best = s->track_count++;
            memset(&s->tracks[best], 0, sizeof(s->tracks[best]));
            s->tracks[best].id = s->next_track_id++;
            s->tracks[best].class_id = (unsigned char)box->class_id;
        }
        if (best != SIZE_MAX) {
            bf_tui_track *track = &s->tracks[best];
            if (track->length == BF_TUI_TRAIL) {
                memmove(track->x, track->x + 1, (BF_TUI_TRAIL - 1) * sizeof(float));
                memmove(track->y, track->y + 1, (BF_TUI_TRAIL - 1) * sizeof(float));
                --track->length;
            }
            track->x[track->length] = box->x; track->y[track->length++] = box->y;
            track->missed = 0; used[best] = 1;
        }
    }
    for (size_t t = 0; t < s->track_count;) {
        if (!used[t] && ++s->tracks[t].missed > 3) {
            s->tracks[t] = s->tracks[--s->track_count]; continue;
        }
        ++t;
    }
    s->last_frame = frame; s->have_last_frame = 1;
}

static void plot(unsigned char *pixels, int width, int height,
                 int x, int y, unsigned char value) {
    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) return;
    size_t i = (size_t)y * (size_t)width + (size_t)x;
    if (pixels[i] < value) pixels[i] = value;
}

static void plot_occupancy(unsigned char *pixels, unsigned char *density,
                           int width, int height, int x, int y,
                           unsigned char value) {
    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) return;
    size_t i = (size_t)y * (size_t)width + (size_t)x;
    if (density[i] != UCHAR_MAX) ++density[i];
    if (pixels[i] < value) pixels[i] = value;
}

static void line(unsigned char *pixels, int width, int height,
                 int x0, int y0, int x1, int y1, unsigned char value) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1, error = dx + dy;
    for (int steps = 0; steps < width + height + dx - dy + 4; ++steps) {
        plot(pixels, width, height, x0, y0, value);
        if (x0 == x1 && y0 == y1) break;
        int twice = 2 * error;
        if (twice >= dy) { error += dy; x0 += sx; }
        if (twice <= dx) { error += dx; y0 += sy; }
    }
}

typedef struct { const bf_tui_state *state; float cosine, sine, scale; int w, h; } view;
static void project(float x, float y, const view *v, int *px, int *py) {
    float dx = x - v->state->center_x, dy = y - v->state->center_y;
    float forward = dx * v->cosine - dy * v->sine;
    float lateral = dx * v->sine + dy * v->cosine;
    *px = (int)lrintf(v->w * 0.5f - lateral * v->scale);
    *py = (int)lrintf(v->h * 0.5f - forward * v->scale);
}

static int inside_quad(int px, int py, const int x[4], const int y[4]) {
    int have_positive = 0, have_negative = 0;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) & 3;
        long long cross = (long long)(x[j] - x[i]) * (py - y[i]) -
                          (long long)(y[j] - y[i]) * (px - x[i]);
        have_positive |= cross > 0;
        have_negative |= cross < 0;
    }
    return !(have_positive && have_negative);
}

static void draw_box(unsigned char *pixels, const view *v,
                     const bf_detection *d, int selected) {
    float cs = cosf(d->yaw), sn = sinf(d->yaw);
    float local_x[4] = {d->length * .5f, d->length * .5f, -d->length * .5f, -d->length * .5f};
    float local_y[4] = {d->width * .5f, -d->width * .5f, -d->width * .5f, d->width * .5f};
    int x[4], y[4];
    for (size_t i = 0; i < 4; ++i)
        project(d->x + local_x[i] * cs - local_y[i] * sn,
                d->y + local_x[i] * sn + local_y[i] * cs, v, &x[i], &y[i]);
    unsigned char fill = selected ? PIX_SELECTED_FILL :
        (unsigned char)(PIX_BOX_FILL_BASE + d->class_id);
    unsigned char edge = selected ? PIX_SELECTED_EDGE :
        (unsigned char)(PIX_BOX_EDGE_BASE + d->class_id);
    unsigned char front = selected ? PIX_SELECTED_EDGE :
        (unsigned char)(PIX_BOX_FRONT_BASE + d->class_id);
    int min_x = x[0], max_x = x[0], min_y = y[0], max_y = y[0];
    for (int i = 1; i < 4; ++i) {
        if (x[i] < min_x) min_x = x[i];
        if (x[i] > max_x) max_x = x[i];
        if (y[i] < min_y) min_y = y[i];
        if (y[i] > max_y) max_y = y[i];
    }
    if (min_x < 0) min_x = 0;
    if (max_x >= v->w) max_x = v->w - 1;
    if (min_y < 0) min_y = 0;
    if (max_y >= v->h) max_y = v->h - 1;
    int coverage = 1 + (int)fminf(4.0f, fmaxf(0.0f, d->score) * 4.0f);
    for (int py = min_y; py <= max_y; ++py)
        for (int px = min_x; px <= max_x; ++px)
            if ((((px * 3 + py * 5) & 7) < coverage) &&
                inside_quad(px, py, x, y))
                plot(pixels, v->w, v->h, px, py, fill);
    for (size_t i = 0; i < 4; ++i)
        line(pixels, v->w, v->h, x[i], y[i], x[(i + 1) & 3], y[(i + 1) & 3],
             i == 0 ? front : edge);
    int center_x, center_y, nose_x, nose_y;
    project(d->x, d->y, v, &center_x, &center_y);
    project(d->x + d->length * .5f * cs, d->y + d->length * .5f * sn,
            v, &nose_x, &nose_y);
    line(pixels, v->w, v->h, center_x, center_y, nose_x, nose_y, front);
}

static void braille(unsigned codepoint, char out[4]) {
    out[0] = (char)(0xe0u | (codepoint >> 12));
    out[1] = (char)(0x80u | ((codepoint >> 6) & 63u));
    out[2] = (char)(0x80u | (codepoint & 63u)); out[3] = 0;
}

static int pixel_color(unsigned char value) {
    if (value == PIX_EGO) return 214;
    if (value >= PIX_SELECTED_EDGE) return 231;
    if (value >= PIX_SELECTED_FILL) return 226;
    if (value >= PIX_BOX_FRONT_BASE) return class_colors[(value - PIX_BOX_FRONT_BASE) % 10];
    if (value >= PIX_BOX_EDGE_BASE) return class_colors[(value - PIX_BOX_EDGE_BASE) % 10];
    if (value >= PIX_BOX_FILL_BASE) return class_colors[(value - PIX_BOX_FILL_BASE) % 10];
    if (value >= PIX_VELOCITY_BASE) return class_colors[(value - PIX_VELOCITY_BASE) % 10];
    if (value >= PIX_TRACK_BASE) return class_colors[(value - PIX_TRACK_BASE) % 10];
    if (value == PIX_OCC_HIGH) return 118;
    if (value == PIX_OCC_LOW) return 45;
    if (value == PIX_OCC_GROUND) return 246;
    if (value == PIX_OCC_OLD) return 31;
    return value == PIX_AXIS ? 243 : 239;
}

static int compose_minimal(text_buffer *text, const char *backend,
                           double inference_ms, int columns, int rows) {
    for (int row = 0; row < rows; ++row) {
        char line_text[160] = "";
        const char *style = "\033[0;38;5;248;48;5;233m";
        if (row == 0) {
            snprintf(line_text, sizeof(line_text), " BEVFUSION [%s]", backend);
            style = "\033[1;38;5;51;48;5;233m";
        } else if (row == 2) {
            snprintf(line_text, sizeof(line_text), " Terminal too small for BEV");
        } else if (row == 3) {
            snprintf(line_text, sizeof(line_text), " Need at least 32 x 10 | %.2f ms",
                     inference_ms);
        } else if (row == rows - 1) {
            snprintf(line_text, sizeof(line_text), " q quit");
        }
        if (row == 0) puts_b(text, "\033[H");
        fixed_ascii(text, style, line_text, columns, row + 1 < rows);
    }
    return !text->failed;
}

int bf_tui_compose(const float *points, size_t point_count, size_t point_stride,
                   const bf_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms, const char *backend,
                   const bf_tui_state *s, int columns, int rows,
                   bf_tui_frame *output) {
    if (!s || !output || columns < 1 || rows < 1 || !backend ||
        (point_count && (!points || point_stride < 3))) return 0;
    output->data = NULL; output->length = 0;
    if (columns < 32 || rows < 10) {
        text_buffer minimal = {0};
        if (!compose_minimal(&minimal, backend, inference_ms, columns, rows)) {
            free(minimal.data); return 0;
        }
        output->data = minimal.data; output->length = minimal.length; return 1;
    }
    int sidebar = s->show_sidebar && columns >= 80 && rows >= 24 ? 30 : 0;
    int canvas_columns = columns - sidebar, canvas_rows = rows - 2;
    int width = canvas_columns * 2, height = canvas_rows * 4;
    unsigned char *pixels = calloc((size_t)width * (size_t)height, 1);
    unsigned char *density = calloc((size_t)width * (size_t)height, 1);
    if (!pixels || !density) { free(density); free(pixels); return 0; }
    view v = {s, cosf(s->yaw), sinf(s->yaw),
              fminf((float)width, (float)height) / 120.0f * s->zoom, width, height};
    if (s->show_grid) {
        for (int radius = 10; radius <= 60; radius += 10) {
            int prior_x = 0, prior_y = 0;
            for (int segment = 0; segment <= 96; ++segment) {
                float angle = (float)segment * 6.28318530718f / 96.0f;
                int x, y; project(radius * cosf(angle), radius * sinf(angle), &v, &x, &y);
                if (segment) line(pixels, width, height, prior_x, prior_y, x, y, 1);
                prior_x = x; prior_y = y;
            }
        }
        int ax0, ay0, ax1, ay1;
        project(-60, 0, &v, &ax0, &ay0); project(60, 0, &v, &ax1, &ay1);
        line(pixels, width, height, ax0, ay0, ax1, ay1, 2);
        project(0, -60, &v, &ax0, &ay0); project(0, 60, &v, &ax1, &ay1);
        line(pixels, width, height, ax0, ay0, ax1, ay1, 2);
    }
    if (s->show_occupancy)
        for (size_t i = 0; i < point_count; ++i) {
            const float *point = points + i * point_stride;
            float intensity = point_stride >= 4 ? point[3] : 1.0f;
            float lag = point_stride >= 5 ? fabsf(point[4]) : 0.0f;
            unsigned char ink = lag > 0.55f || intensity < 0.05f ? PIX_OCC_OLD :
                point[2] < -1.0f ? PIX_OCC_GROUND :
                point[2] < 0.75f ? PIX_OCC_LOW : PIX_OCC_HIGH;
            int x, y;
            project(point[0], point[1], &v, &x, &y);
            plot_occupancy(pixels, density, width, height, x, y, ink);
        }
    if (s->show_tracks)
        for (size_t t = 0; t < s->track_count; ++t)
            for (size_t i = 1; i < s->tracks[t].length; ++i) {
                int x0, y0, x1, y1;
                project(s->tracks[t].x[i - 1], s->tracks[t].y[i - 1], &v, &x0, &y0);
                project(s->tracks[t].x[i], s->tracks[t].y[i], &v, &x1, &y1);
                line(pixels, width, height, x0, y0, x1, y1,
                     (unsigned char)(10 + s->tracks[t].class_id));
            }
    int visible_count = 0;
    if (detections)
        for (int i = 0; i < detections->count; ++i)
            if (visible(&detections->items[i], s)) ++visible_count;
    int selected_ordinal = visible_count ? s->selected % visible_count : -1;
    int selected_index = -1, ordinal = 0;
    if (detections)
        for (int i = 0; i < detections->count; ++i) {
            const bf_detection *d = &detections->items[i];
            if (!visible(d, s)) continue;
            int is_selected = ordinal == selected_ordinal;
            if (is_selected) selected_index = i;
            if (s->show_boxes) draw_box(pixels, &v, d, is_selected);
            if (s->show_velocity) {
                int x0, y0, x1, y1;
                project(d->x, d->y, &v, &x0, &y0);
                project(d->x + d->velocity_x * 2.0f, d->y + d->velocity_y * 2.0f, &v, &x1, &y1);
                line(pixels, width, height, x0, y0, x1, y1,
                     is_selected ? PIX_SELECTED_EDGE :
                     (unsigned char)(PIX_VELOCITY_BASE + d->class_id));
            }
            ++ordinal;
        }
    int ego_x, ego_y; project(0, 0, &v, &ego_x, &ego_y);
    plot(pixels, width, height, ego_x, ego_y, 80);
    text_buffer text = {0}; char header[512];
    if (columns < 80)
        snprintf(header, sizeof(header), " BEVFUSION [%s] %.2f ms | OCC %zu | %d boxes",
                 backend, inference_ms, point_count, visible_count);
    else if (columns < 120)
        snprintf(header, sizeof(header),
                 " BEVFUSION / LiDAR OCC + BOXES | %zu/%zu | %s | %.1f ms | %zu pts | %d boxes",
                 frame_count ? frame + 1 : 0, frame_count, backend, inference_ms,
                 point_count, visible_count);
    else
        snprintf(header, sizeof(header),
                 " BEVFUSION / LiDAR OCC + DETECTIONS | frame %zu/%zu | %s | %.2f ms | OCC %zu pts | %d boxes | score %.2f",
                 frame_count ? frame + 1 : 0, frame_count, backend, inference_ms,
                 point_count, visible_count, s->score_threshold);
    puts_b(&text, "\033[H");
    fixed_ascii(&text, "\033[1;38;5;51;48;5;233m", header, columns, 1);
    static const unsigned dots[4][2] = {{1, 8}, {2, 16}, {4, 32}, {64, 128}};
    for (int cy = 0; cy < canvas_rows; ++cy) {
        for (int cx = 0; cx < canvas_columns; ++cx) {
            unsigned code = 0, density_total = 0; unsigned char priority = 0;
            for (int dy = 0; dy < 4; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    unsigned char p = pixels[(size_t)(cy * 4 + dy) * width + cx * 2 + dx];
                    if (p) code |= dots[dy][dx];
                    if (p > priority) priority = p;
                    density_total += density[(size_t)(cy * 4 + dy) * width + cx * 2 + dx];
                }
            if (priority >= PIX_OCC_OLD && priority <= PIX_OCC_HIGH) {
                int mode = density_total <= 2 ? 2 : density_total <= 8 ? 0 : 1;
                printf_b(&text, "\033[%d;38;5;%d;48;5;233m", mode,
                         pixel_color(priority));
            } else if (priority)
                printf_b(&text, "\033[0;38;5;%d;48;5;233m",
                         pixel_color(priority));
            else puts_b(&text, "\033[0;38;5;238;48;5;233m");
            char glyph[4]; braille(0x2800u + code, glyph); puts_b(&text, glyph);
        }
        if (sidebar) {
            char side[128] = ""; int color = 250;
            const bf_detection *focus = selected_index >= 0 ?
                &detections->items[selected_index] : NULL;
            if (cy == 0) snprintf(side, sizeof(side), " INSPECTOR  %d/%d", visible_count ? selected_ordinal + 1 : 0, visible_count);
            else if (cy == 2) snprintf(side, sizeof(side), " center %+.1f %+.1f", s->center_x, s->center_y);
            else if (cy == 3) snprintf(side, sizeof(side), " zoom %.2fx yaw %+.0f deg", s->zoom, s->yaw * 57.29578f);
            else if (cy >= 5 && cy < 15) {
                int class_id = cy - 5;
                snprintf(side, sizeof(side), " %d %-14s %s", class_id,
                         class_names[class_id],
                         s->class_mask & (1u << class_id) ? "on" : "off");
                color = s->class_mask & (1u << class_id) ? class_colors[class_id] : 240;
            } else if (cy == 16) snprintf(side, sizeof(side), " FOCUS");
            else if (s->show_help && cy == 17) snprintf(side, sizeof(side), " WASD pan  +/- zoom");
            else if (s->show_help && cy == 18) snprintf(side, sizeof(side), " z/e rotate  o occupancy");
            else if (s->show_help && cy == 19) snprintf(side, sizeof(side), " b boxes v velocity g rings");
            else if (s->show_help && cy == 20) snprintf(side, sizeof(side), " t trails 0-9 classes");
            else if (s->show_help && cy == 21) snprintf(side, sizeof(side), " [ ] select ,/. score q quit");
            else if (focus && cy == 17) { snprintf(side, sizeof(side), " %s  %.1f%%", class_names[focus->class_id], focus->score * 100.0f); color = class_colors[focus->class_id]; }
            else if (focus && cy == 18) snprintf(side, sizeof(side), " position %+.1f %+.1f %+.1f m", focus->x, focus->y, focus->z);
            else if (focus && cy == 19) snprintf(side, sizeof(side), " size %.1f x %.1f x %.1f m", focus->length, focus->width, focus->height);
            else if (focus && cy == 20) snprintf(side, sizeof(side), " heading %+.0f deg", focus->yaw * 57.29578f);
            else if (focus && cy == 21) snprintf(side, sizeof(side), " velocity %+.1f %+.1f m/s", focus->velocity_x, focus->velocity_y);
            else if (cy == 23) snprintf(side, sizeof(side), " OCC height + return density");
            printf_b(&text, "\033[0;38;5;%d;48;5;235m", color);
            fixed_ascii(&text, "", side, sidebar, 0);
        }
        if (cy + 1 < canvas_rows) puts_b(&text, "\n");
    }
    char footer[512];
    if (columns < 80)
        snprintf(footer, sizeof(footer), " Space pause | arrows frames | h help | q quit");
    else
        snprintf(footer, sizeof(footer),
                 " layers %c %c %c %c %c | n/p frame | Space pause | WASD pan | +/- zoom | z/e rotate | i inspect | h help | q quit",
                 s->show_occupancy ? 'O' : '-', s->show_boxes ? 'B' : '-',
                 s->show_velocity ? 'V' : '-', s->show_grid ? 'G' : '-',
                 s->show_tracks ? 'T' : '-');
    putn(&text, "\n", 1);
    fixed_ascii(&text, "\033[0;38;5;245;48;5;233m", footer, columns, 0);
    free(density); free(pixels);
    if (text.failed) { free(text.data); return 0; }
    output->data = text.data; output->length = text.length; return 1;
}

void bf_tui_frame_free(bf_tui_frame *frame) {
    if (!frame) return;
    free(frame->data);
    frame->data = NULL;
    frame->length = 0;
}

void bf_tui_render(const float *points, size_t point_count, size_t point_stride,
                   const bf_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const bf_tui_state *state) {
    struct winsize size = {0};
    int columns = 80, rows = 24;
    if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) && size.ws_col && size.ws_row) {
        columns = size.ws_col; rows = size.ws_row;
    }
    bf_tui_frame composed = {0};
    if (bf_tui_compose(points, point_count, point_stride, detections, frame,
                       frame_count, inference_ms, backend, state, columns, rows,
                       &composed)) {
        write_all(STDOUT_FILENO, composed.data, composed.length);
        bf_tui_frame_free(&composed);
    }
}
