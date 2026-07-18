# TUI-1: canonical LiDAR occupancy

## Decision

Promoted for integration. The viewer now renders the BFI LiDAR points and the
canonical decoded detections from the same frame through one metric BEV
transform. This is input occupancy, not a semantic occupancy prediction.

## Contract

- Input sequence: `/data/nuscenes/bevfusion-demo/frame-000.bfi` through
  `frame-011.bfi`, BFI v1, generated from nuScenes mini with ten sweeps.
- Model: `/data/nuscenes/bevfusion-demo/bevfusion.bfw`, exported from
  `/data/nuscenes/checkpoints/cbgs_bevfusion.pth`.
- Hardware: NVIDIA GeForce RTX 4060 Ti 16 GB, strict CUDA runtime.
- Visual layers: density/height-aware LiDAR occupancy, bold class-colored box
  outlines with a heading edge, velocity, trails, rings, and ego.
- Ownership: the CLI retains validated mmap-backed BFI files for the viewer
  lifetime; the compositor borrows pointers and frees only its terminal raster.

## Evidence

Focused functional and responsive-layout fixture:

```sh
make build/test_tui
./build/test_tui
```

Result:

```text
BEV occupancy, oriented box outlines, inspector, controls, and responsive layout pass
```

Address/undefined/leak sanitizer fixture:

```sh
cc -Iinclude -O1 -g -std=c11 -Wall -Wextra -Wpedantic \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/test_tui.c src/tui.c -lm -o /tmp/bevfusion-test-tui-sanitize
ASAN_OPTIONS=detect_leaks=1 /tmp/bevfusion-test-tui-sanitize
```

Result: pass with no sanitizer diagnostic.

The complete CPU/oracle suite also passes after the public compose/render API
change:

```sh
make -j2 test
```

This covers the BFW/BFI malformed-input gates, scalar PyTorch fixtures, every
major real-checkpoint module, runtime resource preflight, TUI, model inspection,
and the 342.02 MiB strict CPU workspace plan.

Real interactive command:

```sh
make build/bevfusion_cuda
./build/bevfusion_cuda tui-cuda \
  /data/nuscenes/bevfusion-demo/bevfusion.bfw \
  /data/nuscenes/bevfusion-demo/frame-*.bfi
```

The captured first frame reported `272414` occupancy points and `38` boxes at
the default `0.20` score gate. Its one-shot interactive wall time was
`252.20 ms`; this is a smoke-test observation, not a replacement for the
five-process cold/warm runtime benchmark. `q` restored cursor visibility,
terminal attributes, and the primary screen.

## Rejected baseline

The prior TUI accepted only `bf_detections`, closed each BFI before entering
the viewer, and drew unfilled outlines on range rings. It could not establish
sensor-to-box alignment and its long header/footer wrapped on narrow terminals.
The new layout emits exactly the requested row count and never exceeds the
reported terminal width across the tested `20..160` column and `5..40` row
matrix.

## Reproducible real-demo path

The promoted interaction has one real-data entry point rooted at
`/data/nuscenes`:

```sh
make -j2
make demo-data
make model
make demo
```

`tools/prepare_nuscenes_demo.py` records source tokens, point counts, byte
counts, and SHA-256 values in `manifest.json`. `render-cuda` is the deterministic
non-interactive boundary used by `scripts/render_demo.py`; it runs the real
CUDA graph before composing every frame. The checked-in 12-frame, 1000x570 GIF
is 588,455 bytes and has SHA-256
`42c7047dc2293c5f2847aa8e10cf66b610666a0856d08b6d092635be957a1fa1`.
The SVG mark has SHA-256
`fa7bced5fa241fe4069ee152975836cb4c16c2f196cc4c790e2812b3dd5826fe`;
its entire visual vocabulary is one plane, two lines, and one fusion point.
