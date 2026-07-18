#ifndef BEVFUSION_H
#define BEVFUSION_H

#define BF_CLASS_COUNT 10
#define BF_CAMERA_COUNT 6
#define BF_MAX_PROPOSALS 200

typedef struct {
    float x, y, z;
    float width, length, height;
    float yaw;
    float velocity_x, velocity_y;
    float score;
    int class_id;
} bf_detection;

typedef struct {
    bf_detection items[BF_MAX_PROPOSALS];
    int count;
} bf_detections;

#endif
