# Custom CUDA build evidence — 2026-07-25

- Host toolkit: CUDA 12.4 / nvcc 12.4.131.
- GPU/NVML: blocked by the operating system.
- `make ENABLE_CUDA=1 CUDA_ARCH=sm_75 build/bevfusion_cuda`: passed.
- `make ENABLE_CUDA=1 ENABLE_CUDA_VENDOR=1 CUDA_ARCH=sm_75
  build/bevfusion_cuda_vendor`: passed.
- CMake custom CUDA configure/build with `BF_CUDA_ARCHS=75`: passed.
- CMake vendor configure/build with `BF_ENABLE_CUDA_VENDOR=ON`: passed and
  retains explicit `libcudnn.so.8`/`libcublas.so.12` dependencies.
- Default custom dynamic-link audit: no `libcudnn` or `libcublas` dependency.
- Focused custom operator and migrated stage fixtures compile.
- The shared custom operator source compiles for `sm_75`, `sm_80`, `sm_86`,
  `sm_89`, and `sm_90`; the portable executable contains `sm_75` cubins and
  `compute_75` PTX.
- Executing `test_cuda_ops` stops at device discovery with
  `CUDA driver version is insufficient for CUDA runtime version`, consistent
  with the host GPU permission block.
- Runtime oracle, Compute Sanitizer, Nsight profiling, cross-generation device
  execution, latency, VRAM, and transfer measurements: not run because device
  access is unavailable.

This record is build evidence only and is not a performance promotion.

## Device access restored

Device access was restored later on the same date on an RTX 4060 Ti. The first
real run exposed a launch-contract bug: batch-six DepthLSS creates 67,584
flattened row tiles, but v1 placed them on the CUDA grid y dimension whose
limit is 65,535. Candidate `cuda-custom-tile16-gridx-v2` moves row tiles to
grid x and adds a 65,536-row-tile regression fixture.

- Custom convolution/GEMM fixtures: passed.
- Exact 12-frame `tui-cuda` command: opened successfully and exited normally.
- BEV oracle maximum absolute error: `7.63e-6`.
- TransFusion oracle maximum absolute error: `9.06e-6`.
- Full camera oracle maximum absolute error: `5.96e-6`; repeat exact.
- Full real-frame runtime: `596.413 ms` cold, `572.971 ms` warm,
  `1102.63 MiB` resident; repeat bit-exact.
- Same-frame custom/vendor decoded comparison: 200/200 class IDs match;
  maximum center-coordinate difference `3.24e-5 m`.
- Runtime counters report `18,426,080` input H2D bytes and `8,804` detection
  D2H bytes, with no intermediate host transfers.
- Compute Sanitizer is installed but cannot attach to this WSL GPU: it reports
  `Failed to initialize WDDM debugger interface` and `Device not supported`.
  The fixture itself still exits successfully, but sanitizer validation is
  explicitly incomplete.

The launch fix is accepted for correctness. The custom path remains far above
the `142.381 ms` performance gate and is not a performance promotion.
