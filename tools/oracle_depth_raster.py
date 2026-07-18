#!/usr/bin/env python3
"""Generate a deterministic OpenPCDet-style lidar-to-image depth raster oracle."""

import argparse
import pathlib
import sys

import torch

from export_checkpoint import write_tensors


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def generate(output):
    points = torch.tensor([
        [0, 1.0, 1.0, 2.0, 0.2], [0, 2.0, 1.0, 2.0, 0.3],
        [0, 1.0, 1.0, 2.0, 0.4], [0, -20.0, 1.0, 1.0, 0.5],
        [1, 0.5, 2.0, 4.0, 0.6], [1, 3.0, -1.0, 3.0, 0.7],
    ], dtype=torch.float32)
    batches, cameras, height, width = 2, 2, 8, 12
    lidar_aug = torch.eye(4).repeat(batches, 1, 1)
    lidar_aug[0, :3, 3] = torch.tensor([0.25, -0.5, 0.0])
    lidar_aug[1, 0, 0] = 2.0
    lidar2image = torch.eye(4).repeat(batches, cameras, 1, 1)
    for b in range(batches):
        for c in range(cameras):
            lidar2image[b, c, 0, 0] = 6.0 + c
            lidar2image[b, c, 1, 1] = 5.0 + b
            lidar2image[b, c, 0, 3] = 4.0 + c
            lidar2image[b, c, 1, 3] = 3.0
    image_aug = torch.eye(4).repeat(batches, cameras, 1, 1)
    image_aug[:, 1, 0, 0] = 0.9
    image_aug[:, 1, 1, 1] = 1.1
    image_aug[:, 1, 0, 3] = 0.5
    depth = torch.zeros(batches, cameras, 1, height, width)
    for b in range(batches):
        mask = points[:, 0] == b
        coordinates = points[mask][:, 1:4].clone()
        coordinates -= lidar_aug[b, :3, 3]
        coordinates = torch.inverse(lidar_aug[b, :3, :3]) @ coordinates.T
        coordinates = lidar2image[b, :, :3, :3] @ coordinates
        coordinates += lidar2image[b, :, :3, 3].reshape(-1, 3, 1)
        distance = coordinates[:, 2, :]
        coordinates[:, 2, :] = torch.clamp(coordinates[:, 2, :], 1e-5, 1e5)
        coordinates[:, :2, :] /= coordinates[:, 2:3, :]
        coordinates = image_aug[b, :, :3, :3] @ coordinates
        coordinates += image_aug[b, :, :3, 3].reshape(-1, 3, 1)
        coordinates = coordinates[:, :2, :].transpose(1, 2)[..., [1, 0]]
        on_image = ((coordinates[..., 0] < height) & (coordinates[..., 0] >= 0) &
                    (coordinates[..., 1] < width) & (coordinates[..., 1] >= 0))
        for c in range(cameras):
            pixels = coordinates[c, on_image[c]].long()
            depth[b, c, 0, pixels[:, 0], pixels[:, 1]] = distance[c, on_image[c]]
    tensors = (
        ("depth_raster.points", points), ("depth_raster.lidar_aug", lidar_aug),
        ("depth_raster.lidar2image", lidar2image),
        ("depth_raster.image_aug", image_aug), ("depth_raster.output", depth),
    )
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.output)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_depth_raster: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
