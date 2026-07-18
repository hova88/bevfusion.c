#!/usr/bin/env python3
"""Real-weight compact oracle for Swin -> FPN -> DepthLSS."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors
from oracle_swin_backbone import CHANNELS, DEPTHS, layer_norm, patch_merge, run_block
from oracle_image_fpn import block as fpn_block
from oracle_depth_head import conv_bn_relu


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def generate(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    torch.manual_seed(0xCA4E2A)
    image = torch.randn(2, 3, 32, 48)
    dense_depth = torch.randn(2, 1, 32, 48).abs()

    value = F.conv2d(
        image, state["image_backbone.patch_embed.projection.weight"],
        state["image_backbone.patch_embed.projection.bias"], stride=4,
    ).flatten(2).transpose(1, 2)
    shape = (8, 12)
    value = layer_norm(value, state, "image_backbone.patch_embed.norm")
    swin = []
    for stage, depth in enumerate(DEPTHS):
        for block in range(depth):
            value = run_block(value, shape, state, stage, block)
        if stage > 0:
            normalized = layer_norm(value, state, f"image_backbone.norm{stage}")
            swin.append(normalized.view(2, *shape, CHANNELS[stage]).permute(0, 3, 1, 2).contiguous())
        if stage < 3:
            value, shape = patch_merge(value, shape, state, stage)

    laterals = list(swin)
    for level in (1, 0):
        upper = F.interpolate(laterals[level + 1], size=laterals[level].shape[2:],
                              mode="bilinear", align_corners=False)
        value = torch.cat((laterals[level], upper), dim=1)
        value = fpn_block(value, state, f"neck.lateral_convs.{level}", 0)
        laterals[level] = fpn_block(value, state, f"neck.fpn_convs.{level}", 1)

    value = conv_bn_relu(dense_depth, state, "vtransform.dtransform", 0, 1)
    value = conv_bn_relu(value, state, "vtransform.dtransform", 3, 4,
                         stride=4, padding=2)
    encoded = conv_bn_relu(value, state, "vtransform.dtransform", 6, 7,
                           stride=2, padding=2)
    value = torch.cat((encoded, laterals[0]), dim=1)
    value = conv_bn_relu(value, state, "vtransform.depthnet", 0, 1, padding=1)
    value = conv_bn_relu(value, state, "vtransform.depthnet", 3, 4, padding=1)
    value = F.conv2d(value, state["vtransform.depthnet.6.weight"],
                     state["vtransform.depthnet.6.bias"])
    tensors = (
        ("cuda_camera_full.image", image),
        ("cuda_camera_full.dense_depth", dense_depth),
        ("cuda_camera_full.swin0", swin[0]),
        ("cuda_camera_full.fpn0", laterals[0]),
        ("cuda_camera_full.logits", value[:, :118]),
        ("cuda_camera_full.context", value[:, 118:]),
    )
    write_tensors([named(name, tensor) for name, tensor in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        print(f"oracle_cuda_camera_full: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
