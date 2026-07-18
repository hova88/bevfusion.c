#!/usr/bin/env python3
"""Generate a real-checkpoint oracle for the DepthLSS depth/context head."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def conv_bn_relu(value, state, base, conv_index, bn_index, stride=1, padding=0):
    value = F.conv2d(
        value, state[f"{base}.{conv_index}.weight"],
        state[f"{base}.{conv_index}.bias"], stride=stride, padding=padding,
    )
    prefix = f"{base}.{bn_index}"
    value = F.batch_norm(
        value, state[prefix + ".running_mean"], state[prefix + ".running_var"],
        state[prefix + ".weight"], state[prefix + ".bias"],
        training=False, momentum=0.1, eps=1e-5,
    )
    return F.relu(value)


def generate(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    torch.manual_seed(0xD3F7A)
    features = torch.randn(1, 256, 2, 3)
    dense_depth = torch.randn(1, 1, 16, 24).abs()
    value = conv_bn_relu(dense_depth, state, "vtransform.dtransform", 0, 1)
    value = conv_bn_relu(value, state, "vtransform.dtransform", 3, 4, stride=4, padding=2)
    encoded_depth = conv_bn_relu(
        value, state, "vtransform.dtransform", 6, 7, stride=2, padding=2
    )
    value = torch.cat((encoded_depth, features), dim=1)
    value = conv_bn_relu(value, state, "vtransform.depthnet", 0, 1, padding=1)
    value = conv_bn_relu(value, state, "vtransform.depthnet", 3, 4, padding=1)
    value = F.conv2d(
        value, state["vtransform.depthnet.6.weight"],
        state["vtransform.depthnet.6.bias"],
    )
    tensors = (
        ("depth_head.features", features),
        ("depth_head.dense_depth", dense_depth),
        ("depth_head.encoded_depth", encoded_depth),
        ("depth_head.logits", value[:, :118]),
        ("depth_head.context", value[:, 118:]),
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
        print(f"oracle_depth_head: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
