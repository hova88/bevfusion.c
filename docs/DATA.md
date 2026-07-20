# Preparing nuScenes data and a BFW model

Data conversion is offline tooling. The inference runtime never imports
PyTorch, NumPy, Pillow, or the nuScenes devkit.

## 1. Download the required checkpoint

Download the exact checkpoint before following the export and full-validation
steps:

**[`cbgs_bevfusion.pth` — Google Drive](https://drive.google.com/file/d/1X50b-8immqlqD8VPAUkSKI0Ls-4k37g9/view?usp=share_link)**

```sh
export NUSCENES_ROOT=/data/nuscenes  # or another writable root
mkdir -p "$NUSCENES_ROOT/checkpoints"
# Save the downloaded file as:
# $NUSCENES_ROOT/checkpoints/cbgs_bevfusion.pth
```

The checkpoint is an external asset with its own terms; it is not redistributed
under the source repository's MIT License and is never downloaded implicitly.
The BFW exporter later verifies the expected tensor names and shapes.

To inspect the checkpoint before export, generate the human and web views:

```sh
make model-summary CHECKPOINT="$NUSCENES_ROOT/checkpoints/cbgs_bevfusion.pth"
```

The generated [`model-summary.md`](model-summary.md) lists every grouped
operator and stored tensor shape in a `torchsummary`-style table. The
interactive website reads the matching `model-graph.json`. Architecture names
are tied to the official OpenPCDet BEVFusion config; activation shapes are
identified separately as BEVFusion.c runtime contracts.

## 2. Create an isolated Python environment

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/demo.txt
```

The official devkit is also available as `pip install nuscenes-devkit` and its
[repository documents the expected dataset
layout](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup).

## 3. Verify the dataset layout

Choose any root; examples use `/data/nuscenes`:

```sh
export NUSCENES_ROOT=/data/nuscenes
test -d "$NUSCENES_ROOT/samples"
test -d "$NUSCENES_ROOT/sweeps"
test -d "$NUSCENES_ROOT/v1.0-mini"
python - <<'PY'
import os
from nuscenes.nuscenes import NuScenes
n = NuScenes(version="v1.0-mini", dataroot=os.environ["NUSCENES_ROOT"], verbose=False)
print(f"nuScenes mini: {len(n.sample)} samples")
PY
```

The demo uses camera images, the current LiDAR keyframe, and preceding sweeps.
Missing `sweeps/` may allow some early frames to pack but does not satisfy the
ten-sweep demo contract.

## 4. Prepare one frame or a sequence

```sh
python tools/prepare_nuscenes.py \
  --root "$NUSCENES_ROOT" --index 0 --sweeps 10 frame-000.bfi

make demo-data NUSCENES_ROOT="$NUSCENES_ROOT" DEMO_COUNT=12
```

The output BFI contract is little-endian FP32:

| Field | Shape | Meaning |
|---|---:|---|
| `camera_images` | `[6,3,256,704]` | resized, cropped, ImageNet-normalized |
| `points` | `[P,5]` | x, y, z, intensity, time lag |
| `camera_intrinsics` | `[6,4,4]` | homogeneous intrinsics |
| `camera_to_lidar` | `[6,4,4]` | camera to reference LiDAR |
| `image_augmentation` | `[6,4,4]` | resize/crop transform |
| `lidar_augmentation` | `[4,4]` | LiDAR augmentation transform |
| `lidar_to_image` | `[6,4,4]` | projection into each camera |

To pack tensors produced by another pipeline, write those keys to NPZ and run:

```sh
python tools/pack_frame.py frame.npz frame.bfi
```

The packer and C loader both enforce canonical order and shapes. BFI v1 stores
a CRC32; the loader also rejects truncation, overflow, reordered sections, and
NaN/Inf values.

## 5. Export the checkpoint

Install the platform-appropriate PyTorch package, then place the matching
checkpoint under `$NUSCENES_ROOT/checkpoints/cbgs_bevfusion.pth` or override
`CHECKPOINT` explicitly:

```sh
python -m pip install -r requirements/export.txt
make model \
  CHECKPOINT=/path/to/cbgs_bevfusion.pth \
  MODEL="$NUSCENES_ROOT/bevfusion-demo/bevfusion.bfw"
```

Export accepts `model_state`, `state_dict`, or a direct state dictionary. It
keeps contiguous FP32 and int64 tensors, aligns payloads to 64 bytes, records
per-tensor CRC32, and atomically replaces the destination. Model loading later
checks the header, directory, names, shapes, ranges, alignment, and checksums.

## 6. Inspect before inference

```sh
./build/bevfusion inspect "$NUSCENES_ROOT/bevfusion-demo/bevfusion.bfw"
./build/bevfusion frame-info "$NUSCENES_ROOT/bevfusion-demo/frame-000.bfi"
```

If model inspection fails on missing names or unexpected shapes, the checkpoint
does not match this frozen graph. Do not bypass the contract check: use the
checkpoint/config pairing for `cfgs/bevfusion.yaml`.
