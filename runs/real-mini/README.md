# Deterministic real-frame fixture

Generated from the locally installed nuScenes v1.0-mini dataset with:

```sh
python3 tools/prepare_nuscenes.py runs/real-mini/frame0.bfi --index 0 --seed 0
```

- sample token: `ca9a282c9e77460f8360f564131a8af5`
- sweeps: 10
- points after aggregation/shuffle: 272,414
- BFI size: 17.57 MiB
- payload CRC32: `9701f67a`

The generated BFI is intentionally ignored because it embeds dataset images
and LiDAR. Recreate it from a licensed local nuScenes copy. The container passes
the C mmap/layout/CRC/finite-value gates. Full scalar and CUDA inference were
both completed on this exact frame; scalar wall time was 209320.586 ms and the
promoted CUDA warm median was 142.381 ms.

Two additional multi-sweep frames were accepted by the same BFI/CUDA graph:

- index 1, token `39586f9d59004284a7114a68825e8eec`, 273,651 points,
  first-call wall time 282.103 ms, 200 detections
- index 2, token `356d81f38dd9473ba590f39e266f54e5`, 273,887 points,
  first-call wall time 271.422 ms, 198 detections
