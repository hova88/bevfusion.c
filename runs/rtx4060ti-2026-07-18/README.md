# RTX 4060 Ti CUDA evidence

Command:

```sh
nsys profile --trace=cuda,cudnn --sample=none --cpuctxsw=none --stats=true \
  -o runs/rtx4060ti-2026-07-18/cuda_bev \
  ./build/test_cuda_bev_stage bevfusion.bfw build/bev_stage_oracle.bfw
```

Strict FP32, deterministic cuDNN algorithms:

- real-weight 8x8 oracle maximum error: `7.21e-6`
- production 180x180 cold forward: `18.422 ms`
- production 180x180 warm average: `19.587 ms`
- context creation and persistent upload: `22.885 ms`
- persistent weights, scratch, and cuDNN workspace: `94.75 MiB`
- device input/output boundary tensors: `121.87 MiB`
- host/device transfer within the measured stage: zero

Five independent unprofiled processes produced a median cold forward of
`20.056 ms` and median warm average of `19.612 ms`; `benchmark.csv` uses those
less tool-perturbed values.

Nsight Systems captured CUDA API timing but reported no kernel rows on this
driver/tool combination. The `.nsys-rep` and `.sqlite` files are retained as
negative profiler evidence; no per-kernel hotspot claim is derived from them.
Nsight Compute was also attempted, but the host returned
`ERR_NVGPUCTRPERM`; `cuda_bev_ncu.csv` preserves that permission failure.
Compute Sanitizer also ran the complete test payload but its debugger backend
reported this device unsupported, so that invocation is not counted as a
memory-safety pass. Oracle output and ordinary CUDA error checks still passed.

## LSS candidate funnel

Five independent `test_cuda_lss` processes at production shape produced these
medians:

- atomic geometry baseline: cold `86.011 ms`, warm `85.577 ms`
- precomputed rank plus the same atomics: cold `92.860 ms`, warm `92.977 ms`
  (rejected)
- stable interval pooling, NCHW context: cold `5.208 ms`, warm `5.549 ms`
- stable interval pooling, NHWC context: cold `1.266 ms`, warm `1.389 ms`
- direct calibration-to-interval preparation: `0.363 ms`

The promoted strict route is direct calibration plus NHWC interval pooling:
`1.752 ms` including preparation, or `1.389 ms` when an unchanged calibration
plan is reused. Its real oracle maximum error is `1.79e-7`. The fixed plan owns
`59.63 MiB`; with production logits, context, frustum, BEV output, and small
calibration tensors the direct boundary totals `115.75 MiB`. It performs no
H2D/D2H within the measured API and never materializes full XYZ geometry.

## CUDA camera neck and image-BEV slice

Real-weight compact oracle maxima were `2.98e-6` (FPN), `2.15e-6` (depth
logits), `1.19e-6` (context), and `1.01e-6` (post-pool downsample). The first
unconstrained deterministic cuDNN selection used 322.88 MiB and measured
9.47 ms for FPN+depth. A 16 MiB selection cap reduced that context to
96.70 MiB and is the promoted default.

Five independent processes with the cap produced these medians:

- FPN + DepthLSS: cold `5.475 ms`, warm `5.827 ms`
- post-LSS downsample: cold `2.245 ms`, warm `2.397 ms`
- chained camera features -> LSS -> image BEV: cold `9.682 ms`, warm `9.600 ms`

The chained contexts own 156.33 MiB; with all explicit synthetic production
boundary tensors the recorded total is 244.31 MiB. Transfers inside the slice
are zero. The test uses a nondefault stream for compact module oracles and the
production slice keeps every intermediate on device.

## CUDA Swin-T candidate funnel

The complete twelve-block backbone matches the compact real-weight oracle with
maximum error `2.86e-6`, including a forced two-window chunk that crosses
shifted-mask chunk offsets. Candidate results:

- one query per block: `87.749 ms`, 393.73 MiB
- eight query warps per block: `71.336 ms` (promoted intermediate)
- sixteen query warps: `75.229 ms` (rejected)
- one window/head block with shared K/V: `63.652 ms`
- shared K/V + 4,096-row FFN + 256-window chunks: `58.055 ms`, 216.27 MiB
- TF32: maximum error `1.87e-3` (rejected by strict oracle gate)

Five independent runs of the final strict route produced median cold/warm
times of `78.261 / 58.055 ms`. Its explicit input/output boundary is
34.03 MiB, and the API performs no host transfer.

## Complete CUDA camera branch

The composed real-weight Swin -> FPN -> DepthLSS oracle has maximum errors
`8.11e-6` for logits and `2.38e-6` for context. The production chain continues
through direct-calibration LSS and post-pool downsampling. Five independent
processes produced median cold/warm times of `67.800 / 67.007 ms`.

Swin, camera-neck, and LSS contexts total 372.59 MiB. All explicit input,
intermediate, and output boundary tensors add 100.36 MiB, for 472.95 MiB
recorded total. Two complete repeated outputs compare bit-for-bit equal. The
only D2H in the test is this out-of-band repeat check; the measured inference
chain performs none.

## Complete CUDA LiDAR branch

Stable radix grouping maps device-resident points to canonical voxel keys,
preserves point order within a voxel, applies the ten-point cap, and computes
MeanVFE without a host count readback. The device count feeds all 21 folded-BN
sparse convolutions directly. Persistent 27-neighbor rulebooks replace repeated
hash probes; an 8,192-block grid and channel-adaptive thread widths are promoted.

The compact real-weight oracle has `5.72e-6` maximum error. A production stress
case maps 100,000 points to 50,000 voxels and exercises sparse occupancies of
50,000 / 155,985 / 39,471 / 5,400 / 2,160. Five-process median cold/warm times
are `58.687 / 58.936 ms`, including a `0.381 ms` warm voxel/MeanVFE stage.
Contexts own 284.53 MiB and all explicit boundaries add 39.04 MiB. No inference
transfer occurs, and repeated 8,294,400-float outputs are bit-identical. Full
candidate evidence is in `cuda_lidar.txt`.

## Complete BFI-to-detections CUDA runtime

The promoted single-stream graph takes the mmap-backed BFI boundary through
depth rasterization, both sensor branches, BEV fusion, TransFusion, and canonical
detections. Five real-frame processes produced median cold/warm wall times of
`191.217 / 142.381 ms`; total owned residency is 1118.71 MiB. Per-frame boundary
traffic is 18,426,080 bytes H2D and only the final 8,804-byte detections D2H.
There is no graph-intermediate transfer, and repeat output is bit-identical.

The same BFI took `209320.586 ms` in the scalar oracle. Class/nearest-center
matching covers all 200 CUDA detections with maximum center error `0.001823 m`,
score error `9.674e-5`, and yaw error `0.011014 rad`. A two-stream camera/LiDAR
candidate measured `145.783 ms` warm and is rejected because resource contention
regressed the serial median by 2.4%. See `cuda_full_runtime.txt`.
