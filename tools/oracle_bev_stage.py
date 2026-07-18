#!/usr/bin/env python3
"""Generate a real-checkpoint oracle for the dense BEV/heatmap stage."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


def named(name, tensor):
    tensor = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), tensor


def batch_norm(value, state, prefix):
    return F.batch_norm(
        value, state[prefix + ".running_mean"], state[prefix + ".running_var"],
        state[prefix + ".weight"], state[prefix + ".bias"],
        training=False, momentum=0.01, eps=1e-3,
    )


def conv_bn_relu(value, state, weight_name, bn_prefix, stride=1, padding=1):
    value = F.conv2d(value, state[weight_name], bias=None, stride=stride, padding=padding)
    return F.relu(batch_norm(value, state, bn_prefix))


def generate(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    torch.manual_seed(0xBEF5A6E)
    value = torch.randn(1, 336, 8, 8)
    input_value = value
    value = conv_bn_relu(value, state, "fuser.conv.0.weight", "fuser.conv.1")
    fuser = value
    block_outputs = []
    upsampled = []
    conv_indices = (1, 4, 7, 10, 13, 16)
    bn_indices = (2, 5, 8, 11, 14, 17)
    for block in range(2):
        for layer, (conv_index, bn_index) in enumerate(zip(conv_indices, bn_indices)):
            stride = 2 if block == 1 and layer == 0 else 1
            value = F.pad(value, (1, 1, 1, 1)) if layer == 0 else value
            value = conv_bn_relu(
                value, state,
                f"backbone_2d.blocks.{block}.{conv_index}.weight",
                f"backbone_2d.blocks.{block}.{bn_index}",
                stride=stride, padding=0 if layer == 0 else 1,
            )
        block_outputs.append(value)
        if block == 0:
            up = F.conv2d(value, state["backbone_2d.deblocks.0.0.weight"])
            up = F.relu(batch_norm(up, state, "backbone_2d.deblocks.0.1"))
        else:
            up = F.conv_transpose2d(
                value, state["backbone_2d.deblocks.1.0.weight"], stride=2
            )
            up = F.relu(batch_norm(up, state, "backbone_2d.deblocks.1.1"))
        upsampled.append(up)
    spatial = torch.cat(upsampled, dim=1)
    shared = F.conv2d(
        spatial, state["dense_head.shared_conv.weight"],
        state["dense_head.shared_conv.bias"], padding=1,
    )
    heatmap_mid = F.conv2d(
        shared, state["dense_head.heatmap_head.0.conv.weight"], padding=1
    )
    heatmap_mid = F.relu(batch_norm(heatmap_mid, state, "dense_head.heatmap_head.0.bn"))
    heatmap = F.conv2d(
        heatmap_mid, state["dense_head.heatmap_head.1.weight"],
        state["dense_head.heatmap_head.1.bias"], padding=1,
    )
    tensors = (
        ("bev_stage.input", input_value), ("bev_stage.fuser", fuser),
        ("bev_stage.block0", block_outputs[0]), ("bev_stage.block1", block_outputs[1]),
        ("bev_stage.up0", upsampled[0]), ("bev_stage.up1", upsampled[1]),
        ("bev_stage.spatial", spatial), ("bev_stage.shared", shared),
        ("bev_stage.heatmap", heatmap),
    )
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_bev_stage: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
