# Portability and core review

## Review outcome

The runtime core is structurally portable across little-endian POSIX hosts:
the public ABI is C, the CPU graph uses C11 and the system math library, and BFW
and BFI are bounds-checked mmap containers. The main portability failures were
in build and workflow policy rather than tensor semantics.

This review changed the following defaults:

- CUDA changed from mandatory to auto-detected and explicitly controllable.
- `sm_89` changed from a global assumption to local detection or an override.
- `-march=native` changed from a release default to an opt-in local build.
- hot CPU dense/sparse kernels gained optional OpenMP, portable SIMD-friendly
  loops, and Accelerate/OpenBLAS dispatch behind a scalar oracle switch.
- dependency-free unit tests were separated from checkpoint/PyTorch oracles.
- `/data/nuscenes` changed from an implicit universal path to a configurable
  root with a home-directory default.
- CMake presets now cover portable release, required CUDA, and sanitizer builds.

## Runtime contract

The frozen graph accepts exactly six `[3,256,704]` normalized camera tensors,
`[P,5]` LiDAR points, and calibrated homogeneous transforms. It returns at most
200 canonical metric detections. Dynamic point, voxel, and sparse capacities
are bounded by a precomputed workspace; the strict graph performs no hidden
heap allocation after runtime creation.

Tensor residency follows reuse:

- BFW weights are permanently mmap-resident on CPU and uploaded into owning
  CUDA modules for the GPU backend.
- BFI input is mmap-resident on CPU for the frame lifetime.
- CPU activations use one caller-owned 64-byte-aligned arena.
- CUDA weights, activations, rulebooks, library workspaces, and staging buffers
  are persistent per runtime.
- one frame crosses H2D at the input boundary; only canonical detections return
  D2H at the output boundary.

## Platform acceptance matrix

| Host | Configure/build | Unit tests | Real inference | Current status |
|---|---|---|---|---|
| WSL2 x86-64 | Make + CMake CPU/CUDA | CPU/CUDA | CUDA production | tested locally for CPU CMake; CUDA revalidation required per driver |
| Linux x86-64 | Make + CMake CPU/CUDA | CPU/CUDA | CUDA production | primary measured platform |
| macOS arm64/x86-64 | Make + CMake CPU | CPU | Accelerate CPU path | CI matrix added; real frame still unmeasured |
| Jetson Orin/Xavier | Make + CMake CUDA | CPU/CUDA | intended CUDA path | architecture controls added; device validation still needed |

“Source-compatible” is not the same as measured. A platform should be marked
supported only after configure, compile, unit tests, model inspection, one real
frame, CPU/CUDA comparison where applicable, and peak memory are recorded.

## Core risks still open

1. **CPU performance.** OpenMP, SIMD-friendly direct convolution, sparse hash
   reuse, a bounded proposal heap, and threaded attention reduced one
   272,414-point WSL2 frame from 209.32 s scalar to 11.71 s on an i5-14600KF.
   The 342 MiB arena and model compute still prevent an honest real-time CPU
   claim. In that profile, dense BEV took 6.11 s, Swin-T 3.48 s, and sparse
   LiDAR 0.80 s; those three stages account for about 89% of total latency.
2. **Jetson memory.** The current CUDA plan owns about 1.1 GiB on the recorded
   desktop GPU and has no low-memory preset. Jetson needs measured cuDNN
   workspace selection and capacity budgets.
3. **Toolkit compatibility.** CUDA/cuDNN calls compile on the recorded CUDA 12.4
   stack. Older JetPack toolchains require an actual build matrix; architecture
   flags alone cannot guarantee API compatibility.
4. **macOS measurement.** GitHub Actions now builds and tests on macOS 14, but
   Accelerate real-frame latency and peak memory still need hardware evidence.
5. **Dataset accuracy.** BFI v1 lacks global sample identity and evaluation
   metadata. Add a versioned result adapter before claiming official mAP/NDS.
6. **Distribution.** The runtime source is MIT-licensed. Dataset, checkpoint,
   CUDA/cuDNN, PyTorch, and other external terms remain separate. The exact
   checkpoint download is documented, but its external license/provenance must
   remain distinct from the source distribution.

## CPU optimization boundary and model roadmap

The CPU backend is valuable as a complete portable implementation, a numerical
oracle, and a fallback for machines without NVIDIA CUDA. It is not presented as
an interactive backend for this FP32 checkpoint. The remaining cost is mostly
the architecture itself rather than container parsing, allocation, or proposal
selection:

| Frozen stage | Measured time | Primary cost |
|---|---:|---|
| BEV fusion + dense backbone | 6.11 s | wide 3×3 convolutions at 180×180 |
| six-camera Swin-T | 3.48 s | attention and dense projections for six views |
| sparse LiDAR backbone | 0.80 s | sparse convolution arithmetic and rule lookup |
| all other stages | 1.32 s | FPN/depth/LSS/decode combined |

Future CPU work should start at the model boundary: distill or replace Swin-T,
reduce image resolution or depth bins, coarsen the BEV grid, prune/narrow the
dense BEV backbone, and evaluate calibrated INT8/FP16 execution. Each proposal
changes accuracy or numerical behavior, so it needs a new versioned checkpoint,
export contract, stage oracles, end-to-end comparison, and official nuScenes
mAP/NDS measurement. Kernel work remains useful when profiling shows a portable
library or layout win, but it should not substitute for those model-level
tradeoffs.

## Recommended completion gates

For each supported platform, archive compiler/toolkit versions, exact build
commands, unit results, real-frame hashes, cold/warm end-to-end latency, peak
RAM/VRAM, and fallback behavior. Optimize only from those measurements. Keep
the scalar CPU path switchable so every SIMD, threading, quantization, and CUDA
change can be compared against the same oracle.
