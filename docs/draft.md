# BEVFusion.c implementation draft

## Frozen objective

Build a small, auditable C11 inference runtime for the supplied OpenPCDet
BEVFusion checkpoint. It must execute lidar, six-camera, LSS projection,
fusion, TransFusion decoding, and rotated NMS; provide complete CPU and CUDA
backends; and visualize canonical decoded detections in a metric BEV-only TUI.

## Runtime contract

- Input: six normalized `256x704` RGB camera images, calibration matrices,
  image augmentation matrices, and lidar points voxelized in
  `[-54,54] x [-54,54] x [-5,3]` at `0.075 x 0.075 x 0.2` metres.
- Lidar: MeanVFE and `VoxelResBackBone8x`, ending in 256 BEV channels.
- Camera: Swin-T `[2,2,6,2]`, LSS FPN and depth-aware lift-splat to 80 BEV
  channels on a `0.3 m` grid.
- Fusion: `336 -> 256`, two-stage BEV backbone, 512-channel concatenation.
- Output: TransFusion, 200 proposals, ten nuScenes classes, center/height/
  dimensions/yaw/velocity/score, decoded once into `bf_detections`.
- Visualization: metric BEV only; never a point-cloud view. Evaluation and TUI
  consume the same `bf_detections` representation.

The checkpoint contains 583 state entries and 163,487,704 tensor bytes before
removing training counters. The offline exporter preserves FP32 and required
int64 constants in the versioned BFW container. Runtime graph specialization
must use this checkpoint, not only the YAML names.

The architecture semantics are pinned to OpenPCDet commit
`233f849829b6ac19afb8af8837a0246890908755`. In particular, sparse coordinates
are `[batch,z,y,x]`; the supplied override produces a `1440x1440x40` voxel grid
and OpenPCDet's backbone adds the extra z cell required by its sparse shape.
Sparse checkpoint kernels are stored as `[kD,kH,kW,input,output]`.

LSS preserves OpenPCDet's pre-downsample `[B,C*Z,X,Y]` layout and final x/y
permute boundary. The strict fused lift-splat route computes depth softmax and
accumulates context directly into this BEV tensor, avoiding the approximately
608 MiB explicit `B*N*D*H*W*C` activation for one real frame. `bf_detection`
uses zero-based class IDs internally; dataset export is responsible for any
one-based compatibility conversion.

The first real-checkpoint dense stage binds ConvFuser, both BaseBEVBackbone
levels, both deblocks, and the TransFusion shared/heatmap convolutions by exact
name and shape. At `180x180`, its two-buffer scalar arena is 63.28 MiB; input,
canonical 512-channel spatial output, shared feature, and raw heatmap are
explicit pipeline boundaries rather than hidden workspace copies.

The real-checkpoint camera backbone now binds patch embedding, all twelve
Swin-T blocks, three patch-merging reductions, and output LayerNorms. It
preserves OpenPCDet's `nn.Unfold` patch order (channel first, then the 2x2
sample position), corner padding, alternating 0/3-pixel window shifts, exact
GELU, and evaluation-mode residual semantics. Its caller-provided scalar arena
is bounded by six times the stage-0 token storage (24.75 MiB for one
`256x704` camera); the current reference window-attention primitive still uses
short-lived internal allocations that the optimized CPU/CUDA implementations
must eliminate. All three output maps were checked through the complete
four-stage graph against actual checkpoint weights, including an odd-width
patch-merge case, with maximum absolute error below `4.8e-6`.

The image neck and view-transform boundaries are also executable scalar C.
GeneralizedLSSFPN performs the checkpoint's two bilinear top-down fusions and
returns only its `32x88` and `16x44` 256-channel maps. The DepthLSS head
constructs a 64-channel point-depth encoding, concatenates it before the two
256-channel trunk convolutions, and exposes 118 raw depth logits plus 80
context channels directly to fused lift-splat. A separate deterministic depth
rasterizer follows the exact inverse-lidar-augmentation, lidar-to-image, and
image-augmentation order. The post-pool three-convolution block downsamples
`[B,80,360,360]` to `[B,80,180,180]` and applies the final x/y transpose.
Real-weight oracle errors are below `5.5e-6` for FPN, `2.0e-6` for the
depth/context head, and `1.3e-6` for post-pool downsampling.

The complete real-checkpoint TransFusion boundary is executable rather than a
head-only approximation. It performs checkpoint proposal suppression and
top-k selection, class one-hot query initialization, learned query/key
positions, eight-head self/cross attention, residual LayerNorm and FFN, all six
prediction heads, metric decode, range filtering, and canonical output. At a
compact real-weight oracle shape its maximum absolute errors are `7.63e-6` for
center, `4.29e-6` for query heatmap, and `4.77e-6` for decoded boxes.

The LiDAR scalar graph binds all 21 sparse convolutions in
VoxelResBackBone8x. It preserves occupancy independently of feature values,
uses sorted `[batch,z,y,x]` coordinates between layers, applies BatchNorm only
to active sites, and keeps residual identities aligned by coordinate. The
three sparse transitions use z padding `1,1,0`, the output convolution uses a
`(3,1,1)` kernel with z stride 2, and height compression is the zero-copy
logical reshape from `[B,128,2,180,180]` to `[B,256,180,180]`. A complete
real-weight `41x8x8` dense-emulated oracle follows the same production z path
`41->21->11->5->2` and matches with maximum absolute error `9.54e-6`.

The public strict runtime composes these modules for batch one and six cameras.
It uses phase marks so only the 256-channel LiDAR BEV and 80-channel camera BEV
survive across modality phases; the honest explicit default arena is 342.02
MiB, 61.8% below the initial conservative lifetime plan. Production Swin attention
streams one padded window at a time through a bounded caller workspace, while
preserving the same `4.77e-6` real-weight maximum error. The remaining
voxel and sparse-coordinate hashes also reuse caller-owned regions. Thus the
strict inference call has no heap allocation after runtime creation; mapped
weights/input and the runtime's small binding objects remain separate from the
342.02 MiB activation/workspace figure.

BFI v1 is the executable input boundary: a canonical little-endian,
CRC32-protected, mmap-backed container for normalized images, five-component
points, and all calibration/augmentation matrices. The loader validates exact
section order and size arithmetic plus every input float before exposing a
`bf_frame_input` view. The CLI now supports model inspection, resource plans,
JSON inference, and sequences in the BEV-only TUI. The TUI never accepts a
point tensor; it consumes only `bf_detections` and draws metric rings, axes,
rotated boxes, velocity, filters, selection, and bounded nearest-class trails.

CUDA LSS began with a strict atomic baseline, then followed a measured
candidate funnel. Merely precomputing flattened cell ranks remained correct
but regressed to `92.977 ms` warm because the approximately 159 million atomic
updates still dominated. The promoted plan instead stable-sorts samples by
BEV cell, creates deterministic intervals, computes depth softmax once, and
lets one block write each cell without atomics. It also derives ranks directly
from the frustum and calibration matrices on device, eliminating the 22.8 MiB
XYZ geometry tensor. An NCHW interval control measured `5.549 ms`; transposing
context once to NHWC reduced strict warm pooling to `1.389 ms`. Including the
`0.363 ms` per-frame direct-calibration preparation gives `1.752 ms`, a 48.8x
improvement over the contemporaneous `85.577 ms` atomic baseline. All figures
are medians of five independent processes; the promoted path matches the
PyTorch oracle within `1.79e-7`, owns 59.63 MiB, and uses no transfers inside
the device boundary.

The CUDA camera neck folds all FPN, DepthLSS, and post-pool evaluation
BatchNorms into 13 persistent convolutions. Bilinear top-down concatenation,
depth-feature concatenation, output splitting, and the final x/y transpose are
custom stream-ordered kernels; all six cameras execute as one cuDNN batch.
Independent real-weight oracle maxima are `2.98e-6` for FPN, `2.15e-6` for
depth logits, `1.19e-6` for context, and `1.01e-6` for downsampling. An
unconstrained deterministic cuDNN selection measured 9.47 ms and 322.88 MiB;
the promoted 16 MiB workspace cap measures `5.827 ms` and 96.70 MiB. The
complete resident slice from Swin feature maps through LSS and downsampling
measures `9.600 ms` warm (five-process median), owns 156.33 MiB of contexts,
and performs no host transfer.

The CUDA Swin-T front end binds all twelve blocks, three patch merges, and
three output norms. Patch embedding uses bounded deterministic cuDNN; strict
pedantic cuBLAS evaluates QKV/projection/FFN/reduction matrices; custom kernels
implement LayerNorm, exact GELU, patch order, shifted/padded windows, relative
bias, masks, online softmax, residuals, and NCHW output conversion. The first
correct one-query layout measured `87.749 ms` warm and 393.73 MiB. Query tiling
reduced that to 71.336 ms; caching all 49 K/V vectors for one window/head in
shared memory reached 63.652 ms. Finally, 4,096-row FFN chunks, 256-window
chunks, and dead-QKV buffer reuse produce the promoted `58.055 ms`,
216.27 MiB context (five-process median) without changing the `2.86e-6`
real-weight oracle maximum. A TF32 experiment failed the strict gate at
`1.87e-3` and is not a production default.

The camera modules are also tested as one graph rather than only isolated
oracles. A saved PyTorch oracle composes real-weight Swin, FPN, and DepthLSS;
CUDA logits/context match within `8.11e-6`. The production chain continues
through direct-calibration stable-interval LSS and downsampling, from six
normalized images to `[1,80,180,180]` image BEV. Five-process medians are
`67.800 ms` cold and `67.007 ms` warm. Its three contexts own 372.59 MiB;
including every explicit boundary/intermediate tensor gives 472.95 MiB.
Repeated final outputs are bit-identical and no inference transfer is hidden.

The dense CUDA tail executes full BEV fusion, both six-layer
backbone blocks, both deblocks, shared projection, and heatmap head. At context
creation it folds every evaluation BatchNorm into its preceding convolution,
uploads all 18 weight/bias pairs once, selects deterministic strict-FP32 cuDNN
algorithms, and allocates reusable device scratch. Inputs, spatial/shared
features, and heatmap remain device pointers on the caller's stream. The
real-weight oracle maximum error is `7.21e-6`; a production `180x180` stage
measured `19.587 ms` warm with 94.75 MiB context residency and no transfers
inside the boundary. The complete runtime now composes this tail with camera,
sparse LiDAR, LSS, and decoder contexts.

TransFusion is now part of that device tail. CUB performs stable descending
proposal selection, strict pedantic cuBLAS handles every learned projection,
and a bounded online-softmax attention kernel avoids the approximately 198 MiB
production score tensor. Metric decode and range filtering execute on GPU; the
only tail D2H is the final 8,804-byte `bf_detections`. A chained real-weight
oracle from 336-channel fusion input through all raw heads and canonical output
matches within `8.58e-6`. At production shape the combined BEV-to-detections
tail measured `36.181 ms` warm (median of five processes) with zero intermediate
transfer and 168.67 MiB of persistent context storage. Query-tiling eight warps
per block improved strict decoder warm time by 23.6%; 16-query tiling and a
branched one-exp softmax regressed and remain rejected. Fast `__expf` is an
explicit approximate option, never the strict default.

## Correctness funnel

1. Container bounds, shape, CRC, duplicate-name, and malformed-input tests.
2. Scalar fixtures for voxelization, convolution, sparse convolution, layer
   norm, window attention, lift-splat, top-k, transformer attention, decode,
   and rotated NMS.
3. Saved PyTorch oracle tensors at every major stage.
4. Multiple real nuScenes frames: raw proposal allclose, decoded output match,
   then official task metric with the exact checkpoint/data pairing.
5. CPU and CUDA backends pass independently; approximate math is labeled and
   cannot replace the strict route.

## Performance loop

Each candidate records the bottleneck, expected mechanism, validation result,
cold/warm end-to-end latency, peak RAM/VRAM, transfer bytes, profiler evidence,
and promotion decision under `runs/` and `benchmark.csv`. CUDA weights and
intermediates remain device-resident, transfers occur at pipeline boundaries,
and kernel changes are promoted only when end-to-end results beat the current
strict baseline.

The complete CUDA LiDAR branch begins with deterministic device voxelization.
It stable-sorts flattened voxel keys, retains input order for the ten-point cap,
writes canonical `[batch,z,y,x]` coordinates, and computes five-channel MeanVFE.
The resulting device count flows into the sparse backbone without a host
readback. All 21 sparse convolutions use folded evaluation BatchNorm and a
persistent 27-neighbor rulebook. A compact real-weight oracle reaches `5.72e-6`
maximum error. For 100,000 stress points producing 50,000 voxels, five-process
medians are `58.687 ms` cold and `58.936 ms` warm from points through final
LiDAR BEV; voxelization itself is `0.381 ms` warm. Contexts total 284.53 MiB,
explicit boundaries add 39.04 MiB, inference transfers are zero, and repeated
production outputs compare bit-for-bit equal.

The complete CUDA runtime composes these branches with the resident BEV and
TransFusion tail. It uploads the BFI inputs once, carries sparse counts and all
intermediates on device, and copies only canonical detections. On the real mini
frame, five-process cold/warm wall medians are `191.217 / 142.381 ms` with
1118.71 MiB owned residency. The scalar oracle requires `209320.586 ms`.
Class-aware nearest-center matching covers all 200 detections; maximum center,
score, and yaw differences are `0.001823 m`, `9.674e-5`, and `0.011014 rad`.
A camera/LiDAR two-stream schedule is retained behind
`BF_CUDA_RUNTIME_PARALLEL=1` but rejected as default after a 2.4% warm regression.
