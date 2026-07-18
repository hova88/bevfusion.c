#define _POSIX_C_SOURCE 200809L
#include "bf_tui.h"

#include <errno.h>
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
    s->show_boxes = s->show_velocity = s->show_grid = s->show_tracks = 1;
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

int bf_tui_poll(bf_tui_state *s, int timeout_ms) {
    if (!s || caught_signal) return BF_TUI_QUIT;
    if (resize_pending) { resize_pending = 0; return BF_TUI_REDRAW; }
    if (!s->input_length) {
        struct pollfd fd = {STDIN_FILENO, POLLIN, 0};
        int result;
        do result = poll(&fd, 1, timeout_ms); while (result < 0 && errno == EINTR);
        if (caught_signal) return BF_TUI_QUIT;
        if (resize_pending) { resize_pending = 0; return BF_TUI_REDRAW; }
        if (result <= 0 || !(fd.revents & POLLIN)) return BF_TUI_NONE;
        ssize_t n = read(STDIN_FILENO, s->input, sizeof(s->input));
        if (n <= 0) return BF_TUI_NONE;
        s->input_length = (size_t)n;
    }
    if (s->input[0] == 27 && s->input_length < 3) return BF_TUI_NONE;
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

static void draw_box(unsigned char *pixels, const view *v,
                     const bf_detection *d, unsigned char value) {
    float cs = cosf(d->yaw), sn = sinf(d->yaw);
    float local_x[4] = {d->length * .5f, d->length * .5f, -d->length * .5f, -d->length * .5f};
    float local_y[4] = {d->width * .5f, -d->width * .5f, -d->width * .5f, d->width * .5f};
    int x[4], y[4];
    for (size_t i = 0; i < 4; ++i)
        project(d->x + local_x[i] * cs - local_y[i] * sn,
                d->y + local_x[i] * sn + local_y[i] * cs, v, &x[i], &y[i]);
    for (size_t i = 0; i < 4; ++i)
        line(pixels, v->w, v->h, x[i], y[i], x[(i + 1) & 3], y[(i + 1) & 3], value);
    line(pixels, v->w, v->h, (x[0] + x[1]) / 2, (y[0] + y[1]) / 2,
         (x[2] + x[3]) / 2, (y[2] + y[3]) / 2, value);
}

static void braille(unsigned codepoint, char out[4]) {
    out[0] = (char)(0xe0u | (codepoint >> 12));
    out[1] = (char)(0x80u | ((codepoint >> 6) & 63u));
    out[2] = (char)(0x80u | (codepoint & 63u)); out[3] = 0;
}

static int pixel_color(unsigned char value) {
    if (value >= 80) return 231;
    if (value >= 60) return 226;
    if (value >= 30) return class_colors[(value - 30) % 10];
    if (value >= 20) return class_colors[(value - 20) % 10];
    if (value >= 10) return class_colors[(value - 10) % 10];
    return value == 2 ? 250 : 238;
}

int bf_tui_compose(const bf_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms, const char *backend,
                   const bf_tui_state *s, int columns, int rows,
                   bf_tui_frame *output) {
    if (!s || !output || columns < 20 || rows < 5 || !backend) return 0;
    output->data = NULL; output->length = 0;
    int sidebar = s->show_sidebar && columns >= 72 ? 30 : 0;
    int canvas_columns = columns - sidebar, canvas_rows = rows - 2;
    int width = canvas_columns * 2, height = canvas_rows * 4;
    unsigned char *pixels = calloc((size_t)width * (size_t)height, 1);
    if (!pixels) return 0;
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
        for (int i = 0; i < detections->count; ++i) {
            const bf_detection *d = &detections->items[i];
            if (!visible(d, s)) continue;
            unsigned char code = (unsigned char)(30 + d->class_id);
            if (visible_count == s->selected) code = 60;
            if (s->show_boxes) draw_box(pixels, &v, d, code);
            if (s->show_velocity) {
                int x0, y0, x1, y1;
                project(d->x, d->y, &v, &x0, &y0);
                project(d->x + d->velocity_x * 2.0f, d->y + d->velocity_y * 2.0f, &v, &x1, &y1);
                line(pixels, width, height, x0, y0, x1, y1,
                     (unsigned char)(20 + d->class_id));
            }
            ++visible_count;
        }
    int ego_x, ego_y; project(0, 0, &v, &ego_x, &ego_y);
    plot(pixels, width, height, ego_x, ego_y, 80);
    text_buffer text = {0};
    printf_b(&text, "\033[H\033[0;48;5;233m\033[38;5;231m BEVFusion · BEV ONLY "
             " frame %zu/%zu  %s  %.2f ms  boxes %d  score ≥ %.2f\033[K\n",
             frame_count ? frame + 1 : 0, frame_count, backend, inference_ms,
             visible_count, s->score_threshold);
    static const unsigned dots[4][2] = {{1, 8}, {2, 16}, {4, 32}, {64, 128}};
    for (int cy = 0; cy < canvas_rows; ++cy) {
        for (int cx = 0; cx < canvas_columns; ++cx) {
            unsigned code = 0; unsigned char priority = 0;
            for (int dy = 0; dy < 4; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    unsigned char p = pixels[(size_t)(cy * 4 + dy) * width + cx * 2 + dx];
                    if (p) code |= dots[dy][dx];
                    if (p > priority) priority = p;
                }
            if (priority) printf_b(&text, "\033[38;5;%dm", pixel_color(priority));
            else puts_b(&text, "\033[38;5;238m");
            char glyph[4]; braille(0x2800u + code, glyph); puts_b(&text, glyph);
        }
        if (sidebar) {
            puts_b(&text, "\033[0;48;5;235m\033[38;5;250m ");
            if (cy == 0) printf_b(&text, "INSPECTOR  selected %d", s->selected + 1);
            else if (cy == 2) printf_b(&text, "center %+.1f %+.1f", s->center_x, s->center_y);
            else if (cy == 3) printf_b(&text, "zoom %.2fx yaw %+.0f°", s->zoom, s->yaw * 57.29578f);
            else if (cy >= 5 && cy < 15) {
                int class_id = cy - 5;
                printf_b(&text, "%d %-14s %s", class_id, class_names[class_id],
                         s->class_mask & (1u << class_id) ? "on" : "off");
            } else if (s->show_help && cy == 17) puts_b(&text, "WASD pan  +/- zoom");
            else if (s->show_help && cy == 18) puts_b(&text, "z/e rotate  b boxes");
            else if (s->show_help && cy == 19) puts_b(&text, "v velocity g rings t trails");
            else if (s->show_help && cy == 20) puts_b(&text, "0-9 classes ,/. score");
            else if (s->show_help && cy == 21) puts_b(&text, "[ ] select  q quit");
            puts_b(&text, "\033[K");
        }
        if (cy + 1 < canvas_rows) puts_b(&text, "\n");
    }
    puts_b(&text, "\n\033[0;48;5;233m\033[38;5;245m n/p frame · space pause · "
                  "WASD pan · +/- zoom · z/e rotate · i inspect · h help · q quit\033[K");
    free(pixels);
    if (text.failed) { free(text.data); return 0; }
    output->data = text.data; output->length = text.length; return 1;
}

void bf_tui_frame_free(bf_tui_frame *frame) {
    if (!frame) return;
    free(frame->data);
    frame->data = NULL;
    frame->length = 0;
}

void bf_tui_render(const bf_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const bf_tui_state *state) {
    struct winsize size = {0};
    int columns = 80, rows = 24;
    if (!ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) && size.ws_col && size.ws_row) {
        columns = size.ws_col; rows = size.ws_row;
    }
    bf_tui_frame composed = {0};
    if (bf_tui_compose(detections, frame, frame_count, inference_ms, backend,
                       state, columns, rows, &composed)) {
        write_all(STDOUT_FILENO, composed.data, composed.length);
        bf_tui_frame_free(&composed);
    }
}
