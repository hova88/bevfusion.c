# Real nuScenes KDA audit — RTX 4060 Ti

## Scope

- GPU: NVIDIA GeForce RTX 4060 Ti 16 GB, Ada `sm_89`
- CUDA: 12.4, strict FP32 cuDNN/cuBLAS graph
- Model: `/data/nuscenes/bevfusion-demo/bevfusion.bfw`
- Frame: `/data/nuscenes/bevfusion-demo/frame-000.bfi`, 272,414 points
- Demo: twelve deterministic nuScenes mini frames, ten LiDAR sweeps each

## Evidence ladder

Fresh stage timing identified LiDAR stage 3 (~25 ms), Swin window/FFN work
(~52 ms combined), and TransFusion cross attention (~15 ms) as the largest
visible slices. Nsight Compute was attempted, but the driver returned
`ERR_NVGPUCTRPERM`; no counter-derived stall, occupancy, or bandwidth claim is
made.

Candidates were gated in this order: production-shape oracle, repeated real
frame, isolated stage timing, then full-graph wall time.

| Candidate | Evidence | Decision |
|---|---:|---|
| Reuse calibration-derived LSS plan | avoids ~0.36 ms plan preparation when the 210-float key is unchanged | promote |
| 64-bit cell+ordinal LSS key | plan residency 59.63 → 82.71 MiB; rebuild ~1.17 ms | reject |
| Swin shared tiles 14/16 | slower than default in repeated stage scans | reject |
| LiDAR grids up to 32768 blocks | full-frame medians overlapped/noisy | reject |
| Parallel camera/LiDAR streams | historical warm median 145.783 vs 142.381 ms serial | reject |

The current seven-process real-frame samples had cache-path warm times
`147.669, 150.120, 135.069, 145.852, 146.134, 143.431, 145.254 ms`
(median 145.852 ms). Forced plan rebuild measured
`146.021, 145.261, 144.711, 134.656, 136.470, 145.938, 147.565 ms`
(median 145.261 ms). Process noise is larger than the isolated saving, so the
full-runtime result is reported as ~146 ms rather than as a claimed speedup.

## Commands

```sh
make build/test_cuda_runtime
./build/test_cuda_runtime \
  /data/nuscenes/bevfusion-demo/bevfusion.bfw \
  /data/nuscenes/bevfusion-demo/frame-000.bfi

BF_CUDA_LSS_REBUILD_PLAN=1 ./build/test_cuda_runtime \
  /data/nuscenes/bevfusion-demo/bevfusion.bfw \
  /data/nuscenes/bevfusion-demo/frame-000.bfi
```

The normal adjacent-repeat gate is bit-exact in 12 of the 14 recorded A/B
processes. Rare strict-cuDNN scheduling drift is bounded by the module oracle
tests; it is not attributed to LSS ordering because the sorted values and
interval tables remained identical during the diagnostic audit.
