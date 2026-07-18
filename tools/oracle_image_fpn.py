#!/usr/bin/env python3
"""Generate a real-checkpoint oracle for GeneralizedLSSFPN."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def block(value, state, prefix, padding):
    value = F.conv2d(value, state[prefix + ".conv.weight"], padding=padding)
    value = F.batch_norm(
        value, state[prefix + ".bn.running_mean"], state[prefix + ".bn.running_var"],
        state[prefix + ".bn.weight"], state[prefix + ".bn.bias"],
        training=False, momentum=0.1, eps=1e-5,
    )
    return F.relu(value)


def generate(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    torch.manual_seed(0xF90A17)
    inputs = [
        torch.randn(1, 192, 4, 6),
        torch.randn(1, 384, 2, 3),
        torch.randn(1, 768, 1, 2),
    ]
    laterals = list(inputs)
    for level in (1, 0):
        upper = F.interpolate(
            laterals[level + 1], size=laterals[level].shape[2:],
            mode="bilinear", align_corners=False,
        )
        value = torch.cat((laterals[level], upper), dim=1)
        value = block(value, state, f"neck.lateral_convs.{level}", 0)
        laterals[level] = block(value, state, f"neck.fpn_convs.{level}", 1)
    tensors = [(f"image_fpn.input{i}", value) for i, value in enumerate(inputs)]
    tensors.extend((f"image_fpn.output{i}", laterals[i]) for i in range(2))
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_image_fpn: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
