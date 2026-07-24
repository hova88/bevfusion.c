# Custom CUDA provider contract

Date frozen: 2026-07-25

This task removes cuDNN/cuBLAS from the default CUDA executable without
changing the public C API, BFW/BFI formats, CPU graph, tensor layouts, or
canonical detection output. The previous deterministic cuDNN/cuBLAS path is
retained as `bevfusion_cuda_vendor` and is compiled only when explicitly
requested.

## Frozen graph and numerical contract

- Input/output, preprocessing, graph shapes, and decoding are those in
  `docs/draft.md`.
- CUDA convolution inputs and outputs are NCHW FP32. Ordinary weights are OIHW;
  the single 2x2 stride-2 transpose convolution uses IOHW.
- Swin and TransFusion projections are row-major
  `output[M,N] = input[M,K] * weight[N,K]^T + bias[N]`.
- BatchNorm remains folded into FP32 convolution weights during context
  creation. Bias and optional ReLU are fused into custom kernel writeback.
- Default arithmetic is FP32 FMA with deterministic per-output reduction
  order. TF32, FP16, WMMA, fast math, and architecture-specific PTX are outside
  the strict provider.
- Context creation uploads all weights and allocates reusable scratch. Forward
  does not allocate and all intermediate tensors remain device resident.

## Real operator families

| Consumer | Frozen family |
|---|---|
| Swin | 4x4/stride-4 patch convolution; QKV, projection, FFN, merge GEMMs |
| Camera | 13 convolutions: 1x1, 3x3, 5x5; stride 1/2/4; batch six neck and batch-one BEV downsample |
| BEV | 18 convolutions: 1x1/3x3 and one non-overlapping 2x2/stride-2 transpose convolution |
| TransFusion | 128/256/64-channel projection and head GEMMs |

The initial candidate uses a shared-memory 16x16 implicit-GEMM convolution and
a 16x16 tiled SGEMM. Boundary tiles support nonmultiples. This is a
correctness-first candidate, not yet a performance promotion.

## Acceptance budget

The preserved RTX 4060 Ti vendor baseline is 142.381 ms warm median across
five fresh processes and 1118.71 MiB owned VRAM. Promotion requires all strict
operator/stage/full-frame oracles, no intermediate host transfers, stable
repeat output, warm median no higher than both 142.381 ms and a contemporaneous
vendor run, and owned VRAM no higher than 1118.71 MiB.

CUDA 12.4 execution is now available on the RTX 4060 Ti. Operator, BEV,
TransFusion, camera-full, and full-runtime correctness gates pass after moving
large flattened row grids from CUDA grid y to grid x. The current custom warm
runtime is `572.971 ms`, so it does not satisfy the `142.381 ms` performance
gate. Runtime counters report only the expected input H2D and detection D2H
boundaries, with no intermediate host transfers. Compute Sanitizer is installed
but cannot attach in this WSL environment (`Failed to initialize WDDM debugger
interface`; device unsupported), so sanitizer evidence, Nsight profiling, and
Turing runtime validation remain pending.
