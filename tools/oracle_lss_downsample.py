#!/usr/bin/env python3
"""Generate a real-checkpoint oracle for the post-pool LSS downsample."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def generate(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    torch.manual_seed(0x155D0A)
    input_value = torch.randn(1, 80, 8, 10)
    value = input_value
    for layer, (conv, bn, stride) in enumerate(((0, 1, 1), (3, 4, 2), (6, 7, 1))):
        value = F.conv2d(value, state[f"vtransform.downsample.{conv}.weight"],
                         stride=stride, padding=1)
        prefix = f"vtransform.downsample.{bn}"
        value = F.batch_norm(
            value, state[prefix + ".running_mean"], state[prefix + ".running_var"],
            state[prefix + ".weight"], state[prefix + ".bias"],
            training=False, momentum=0.1, eps=1e-5,
        )
        value = F.relu(value)
    output_value = value.permute(0, 1, 3, 2).contiguous()
    tensors = (("lss_downsample.input", input_value),
               ("lss_downsample.output", output_value))
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_lss_downsample: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
