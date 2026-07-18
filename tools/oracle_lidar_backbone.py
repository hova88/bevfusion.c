#!/usr/bin/env python3
"""Generate a dense-emulated real-checkpoint oracle for VoxelResBackBone8x."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def sparse_conv(value, mask, weight, stride=1, padding=0, submanifold=False):
    kernel = weight.permute(4, 3, 0, 1, 2).contiguous()
    value = F.conv3d(value, kernel, stride=stride, padding=padding)
    if submanifold:
        output_mask = mask
    else:
        kd, kh, kw = weight.shape[:3]
        occupancy_kernel = torch.ones(1, 1, kd, kh, kw)
        output_mask = F.conv3d(mask.float(), occupancy_kernel,
                               stride=stride, padding=padding) > 0
    return value * output_mask, output_mask


def batch_norm(value, mask, state, prefix, relu=True):
    value = F.batch_norm(
        value, state[prefix + ".running_mean"], state[prefix + ".running_var"],
        state[prefix + ".weight"], state[prefix + ".bias"],
        training=False, momentum=0.01, eps=1e-3,
    )
    if relu:
        value = F.relu(value)
    return value * mask


def layer(value, mask, state, weight_name, bn_prefix,
          stride=1, padding=0, submanifold=False, relu=True):
    value, mask = sparse_conv(value, mask, state[weight_name],
                              stride, padding, submanifold)
    return batch_norm(value, mask, state, bn_prefix, relu), mask


def residual(value, mask, state, stage, block):
    identity = value
    prefix = f"backbone_3d.conv{stage}.{block}"
    value, _ = layer(value, mask, state, prefix + ".conv1.weight",
                     prefix + ".bn1", padding=1, submanifold=True)
    value, _ = layer(value, mask, state, prefix + ".conv2.weight",
                     prefix + ".bn2", padding=1, submanifold=True, relu=False)
    return F.relu(value + identity) * mask


def generate(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    torch.manual_seed(0x11DA4B)
    coords = torch.tensor([
        [0, 0, 0, 0], [0, 2, 1, 3], [0, 7, 4, 6], [0, 12, 6, 1],
        [0, 20, 2, 5], [0, 30, 7, 7], [0, 38, 5, 2], [0, 40, 3, 4],
    ], dtype=torch.int64)
    features = torch.randn(coords.shape[0], 5)
    value = torch.zeros(1, 5, 41, 8, 8)
    mask = torch.zeros(1, 1, 41, 8, 8, dtype=torch.bool)
    for index, coordinate in enumerate(coords):
        b, z, y, x = coordinate.tolist()
        value[b, :, z, y, x] = features[index]
        mask[b, :, z, y, x] = True
    value, mask = layer(
        value, mask, state, "backbone_3d.conv_input.0.weight",
        "backbone_3d.conv_input.1", padding=1, submanifold=True,
    )
    for stage, channels in enumerate((16, 32, 64, 128), start=1):
        del channels
        if stage > 1:
            padding = (0, 1, 1) if stage == 4 else 1
            value, mask = layer(
                value, mask, state, f"backbone_3d.conv{stage}.0.0.weight",
                f"backbone_3d.conv{stage}.0.1", stride=2, padding=padding,
            )
        for block in range(2):
            value = residual(value, mask, state, stage, block + (1 if stage > 1 else 0))
    value, mask = layer(
        value, mask, state, "backbone_3d.conv_out.0.weight",
        "backbone_3d.conv_out.1", stride=(2, 1, 1), padding=0,
    )
    bev = value.view(1, value.shape[1] * value.shape[2], value.shape[3], value.shape[4])
    tensors = (("lidar_backbone.coords", coords),
               ("lidar_backbone.features", features),
               ("lidar_backbone.output", bev))
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_lidar_backbone: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
