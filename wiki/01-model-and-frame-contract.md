# Model and frame contract

The runtime intentionally supports one graph rather than a general model
interpreter. This keeps tensor names, dimensions, numeric behavior, and memory
bounds visible in C.

```text
BFI v1
  camera_images       float32 [6,3,256,704]
  points              float32 [P,5]
  calibration         float32 homogeneous matrices
        |
        v
BFW model             typed, shaped, checksummed tensors
        |
        v
bf_detections         at most 200 metric 3D boxes
```

The five point fields are `x, y, z, intensity, timestamp` in the augmented
LiDAR frame. The API contract is declared in
[`include/bf_runtime.h`](../include/bf_runtime.h), while
[`src/frame.c`](../src/frame.c) verifies little-endian layout, section order,
sizes, CRC32, finite values, and overflow before exposing mmap-backed pointers.

The output ABI in [`include/bevfusion.h`](../include/bevfusion.h) owns position,
dimensions, yaw, planar velocity, score, and class ID. JSON, CPU inference,
CUDA inference, and the TUI all consume this same representation. There is no
viewer-specific box decoder.

## Capacity is part of the call

CPU callers provide point, voxel, and sparse capacities together with one
workspace. CUDA callers freeze those capacities when creating the runtime and
allocate reusable device storage once. Oversized point counts and singular
calibration matrices fail before graph execution; they are tested as contract
errors rather than left as kernel preconditions.

The BFW container is bounds checked and memory mapped by
[`src/model.c`](../src/model.c). The exporter in
[`tools/export_checkpoint.py`](../tools/export_checkpoint.py) remains outside
the runtime trust boundary.
