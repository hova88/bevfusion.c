# BEVFusion.c

An auditable C11 rewrite of the supplied OpenPCDet BEVFusion checkpoint. The
strict CPU graph covers six-camera Swin/LSS, sparse LiDAR encoding, BEV fusion,
TransFusion, metric decode, and rotated filtering. Its only visualization is a
metric bird's-eye view; there is no point-cloud rendering mode.

## Build and validate

```sh
make model
make -j2 test
make cuda-test
```

`make model` converts `ckpts/cbgs_bevfusion.pth` into the mmap-friendly,
versioned, checksummed `bevfusion.bfw` container. The test suite compares every
major module with saved PyTorch oracles. The CUDA target validates the
atomics-free LSS plan, complete 18-convolution BEV stage, TransFusion decoder,
and their chained canonical-detection tail.

## Frame input

The runtime consumes canonical BFI v1 files. Prepare normalized tensors in an
NPZ with these keys and shapes:

- `camera_images`: `[6,3,256,704]`
- `points`: `[P,5]` (`x,y,z,intensity,timestamp`)
- `camera_intrinsics`, `camera_to_lidar`, `image_augmentation`,
  `lidar_to_image`: `[6,4,4]`
- `lidar_augmentation`: `[4,4]`

All arrays are finite FP32 values in the augmented frame expected by the
checkpoint. Pack them with:

```sh
python3 tools/pack_frame.py frame.npz frame.bfi
```

BFI is little-endian, canonical-layout, CRC32-protected, and mmap-backed. The C
loader rejects overflow, truncation, reordered sections, checksum failures,
and non-finite tensors before inference.

## Run

```sh
./build/bevfusion inspect bevfusion.bfw
./build/bevfusion frame-info frame.bfi
./build/bevfusion plan bevfusion.bfw 300000 160000 160000
./build/bevfusion infer bevfusion.bfw frame.bfi
./build/bevfusion tui bevfusion.bfw frame0.bfi frame1.bfi
./build/bevfusion_cuda infer-cuda bevfusion.bfw frame.bfi
./build/bevfusion_cuda tui-cuda bevfusion.bfw frame0.bfi frame1.bfi
```

Both inference commands write canonical decoded detections as JSON; both TUI
commands render only those detections as rotated boxes, velocities, class
filters, and short tracks on a responsive Braille BEV. `build/bevfusion` is the
intentionally slow scalar oracle and `build/bevfusion_cuda` is the complete
device-resident backend. The scalar arena is 342.02 MiB at default capacities.
After runtime creation, neither strict inference graph performs hidden heap
allocation.

The CUDA BEV stage folds inference BatchNorm into persistent convolution
weights, keeps its two activation buffers and cuDNN workspace resident, accepts
only device tensors, and supports caller streams. Strict deterministic FP32
matches the real-weight oracle within `7.21e-6`; on the local RTX 4060 Ti its
production `180x180` warm time is about `19.6 ms`; the complete CUDA runtime
composes it after both sensor branches.

CUDA LSS now computes cell keys directly from the fixed frustum and device
calibration matrices, stable-sorts them into deterministic BEV intervals, and
pools without atomics. Its strict default transposes context once to NHWC for
coalesced channel access. The production-shape median is `0.363 ms` for
per-frame calibration preparation plus `1.389 ms` warm lift/pool, versus
`85.577 ms` for the original atomic baseline. The plan owns `59.63 MiB`; the
full direct route including input/output boundary tensors is `115.75 MiB` and
does not materialize the 22.8 MiB XYZ geometry tensor. Set
`BF_CUDA_LSS_PLAN_NCHW=1` only to reproduce the slower strict control.

The six-camera CUDA neck batches both FPN levels and the complete DepthLSS head
through strict deterministic cuDNN, then shares its buffers with the three-conv
post-LSS downsampler. Compact real-weight oracle errors are at most `2.98e-6`.
A measured 16 MiB algorithm-workspace cap is the default: it reduced FPN+depth
from 9.47 to `5.83 ms` while cutting context residency from 322.88 to
`96.70 MiB`. Chained from Swin feature maps through direct-calibration LSS to
the final 80-channel image BEV, the slice measures `9.60 ms` warm with no host
transfer. `BF_CUDA_CAMERA_WORKSPACE_MIB` remains an explicit tuning override.

The complete CUDA Swin-T backbone uses strict pedantic cuBLAS for every learned
projection, deterministic cuDNN for patch embedding, exact GELU, shifted masks,
and online-softmax window attention. A shared-K/V kernel caches each
window/head once, while 4,096-row FFN and 256-window chunks bound activations.
The real-weight oracle maximum is `2.86e-6`; production medians are `78.26 ms`
cold and `58.06 ms` warm at `216.27 MiB` resident. This is 33.8% faster and
45.1% smaller than the first strict CUDA layout. TF32 reached `1.87e-3` error
and is therefore rejected for strict inference.

Composed as one device chain, normalized six-camera images now pass through
Swin, FPN, DepthLSS, direct-calibration interval pooling, and post-pool
downsampling to the final `[1,80,180,180]` image BEV. A compact composed
real-weight oracle reaches `8.11e-6` maximum error. Production medians are
`67.80 ms` cold and `67.01 ms` warm; contexts own `372.59 MiB`, the complete
explicit boundary totals `472.95 MiB`, and repeated final BEV tensors are
bit-exact. No transfer occurs within inference.

The device-resident tail now continues from that shared BEV tensor through
proposal suppression/top-k, learned positions, self/cross attention, FFN, all
six TransFusion heads, metric decode, range filtering, and the same canonical
`bf_detections` consumed by the TUI. The combined production tail has no
intermediate D2H and copies only the final 8,804-byte detection structure;
strict warm latency is about `36.2 ms` with 168.67 MiB of context residency.
An eight-query warp tile improved strict attention by 23.6%; an explicit
`BF_CUDA_TRANSFUSION_FAST_EXP` route is faster but remains labeled approximate.

The CUDA LiDAR path is complete from device points through stable voxel grouping,
ten-point MeanVFE, all 21 folded-BN sparse convolutions, and dense 256-channel
BEV. Its persistent-rulebook route matches the real-weight oracle within
`5.72e-6`. A 100,000-point/50,000-voxel stress case measures `58.94 ms` warm
(five-process median), including `0.381 ms` voxelization, with 284.53 MiB of
contexts and no inference transfer. Repeated production outputs are bit-exact.

`build/bevfusion_cuda` now exposes `infer-cuda` and `tui-cuda`. The full strict
runtime consumes a BFI frame, keeps every graph intermediate on device, and
returns only the same canonical detections used by the BEV-only TUI. On the real
272,414-point mini frame, five-process medians are `191.22 ms` cold and
`142.38 ms` warm at 1118.71 MiB total residency, versus `209320.59 ms` for the
scalar reference. All 200 detections match by class/nearest center with maximum
center error 1.823 mm and score error `9.674e-5`. Input H2D is 18,426,080 bytes,
final D2H is 8,804 bytes, and intermediate transfer is zero.

The complete command/oracle/sanitizer audit is recorded under
`runs/rtx4060ti-2026-07-18/validation.txt`. Official nuScenes mAP/NDS is not
claimed from the three local mini frames because BFI deliberately omits the
lidar-to-global result metadata required by the official evaluator.
