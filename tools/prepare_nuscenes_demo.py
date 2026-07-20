#!/usr/bin/env python3
"""Prepare a deterministic real nuScenes sequence for the BEVFusion.c demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes

from pack_frame import FIELDS, canonical, write_bfi
from prepare_nuscenes import load_cameras, load_points


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    default_root = Path(os.environ.get(
        "NUSCENES_ROOT", Path.home() / "datasets" / "nuscenes"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path,
                        default=default_root / "bevfusion-demo")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.sweeps < 1:
        raise ValueError("start must be non-negative; count and sweeps must be positive")

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=False)
    stop = args.start + args.count
    if stop > len(nusc.sample):
        raise ValueError(f"requested samples [{args.start}, {stop}), dataset has {len(nusc.sample)}")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for sequence_index, sample_index in enumerate(range(args.start, stop)):
        sample = nusc.sample[sample_index]
        frame_seed = args.seed + sequence_index
        points = load_points(nusc, sample, args.sweeps, frame_seed)
        images, intrinsics, c2l, image_aug, l2i = load_cameras(nusc, sample)
        raw = (images, points, intrinsics, c2l, image_aug,
               np.eye(4, dtype=np.float32), l2i)
        arrays = [canonical(name, value, shape)
                  for (name, shape), value in zip(FIELDS, raw)]
        output = args.output / f"frame-{sequence_index:03d}.bfi"
        write_bfi(arrays, output)
        records.append({
            "frame": sequence_index,
            "sample_index": sample_index,
            "sample_token": sample["token"],
            "seed": frame_seed,
            "sweeps": args.sweeps,
            "points": int(points.shape[0]),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "path": str(output),
        })
    manifest = {
        "format": "bevfusion.c-nuscenes-demo-v1",
        "root": str(args.root.resolve()),
        "version": args.version,
        "frames": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} with {len(records)} real nuScenes frames")


if __name__ == "__main__":
    main()
