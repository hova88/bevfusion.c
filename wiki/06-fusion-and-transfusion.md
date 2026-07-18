# Fusion and TransFusion

The image `[1,80,180,180]` and LiDAR `[1,256,180,180]` tensors concatenate on
device. The BEV stage produces a shared 128-channel feature and dense class
heatmap; TransFusion selects proposals, runs self/cross attention and an FFN,
then decodes six heads into metric boxes.

The decoder uses pedantic cuBLAS with atomics disabled. Custom online-softmax
attention keeps one stable `(m, l)` pair per query and head. The strict default
maps eight query warps per block. A sixteen-query tile regressed, while
`__expf` is retained only behind an explicitly approximate environment route.

| Attention candidate | Warm time | Decision |
|---|---:|---|
| One query per block | 22.467 ms | Strict fallback |
| Branched one-exp form | 31.770 ms | Reject regression |
| Eight-query warp tile | 17.164 ms | Strict default |
| Sixteen-query warp tile | 21.963 ms | Reject regression |
| Eight-query tile with `__expf` | 15.974 ms | Labeled approximate |

The custom forward checks kernel launch status, library calls, proposal-copy
status, and the final asynchronous detection copy. The public call synchronizes
only when materializing `bf_detections` on the host. See
[`src/cuda_transfusion.cu`](../src/cuda_transfusion.cu).

**Measured:** the complete BFI-to-detections graph reaches 142.381 ms warm on
the recorded real frame. All 200 CUDA detections match the scalar output by
class and nearest center; maxima are 1.823 mm for center, `9.674e-5` for score,
and 0.011014 rad for yaw. Raw rank order can differ at nearly tied scores, so
the validation compares the semantic set instead of assuming stable score
ties. Details are in
[`cuda_full_runtime.txt`](../runs/rtx4060ti-2026-07-18/cuda_full_runtime.txt).
