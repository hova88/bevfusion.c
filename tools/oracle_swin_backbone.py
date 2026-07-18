#!/usr/bin/env python3
"""Generate a small full Swin-T oracle using the real BEVFusion checkpoint."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


DEPTHS = (2, 2, 6, 2)
CHANNELS = (96, 192, 384, 768)
HEADS = (3, 6, 12, 24)
WINDOW = 7


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def layer_norm(value, state, prefix):
    return F.layer_norm(
        value, (value.shape[-1],), state[prefix + ".weight"],
        state[prefix + ".bias"], 1e-5,
    )


def window_partition(value):
    batches, height, width, channels = value.shape
    return (
        value.view(batches, height // WINDOW, WINDOW, width // WINDOW, WINDOW, channels)
        .permute(0, 1, 3, 2, 4, 5).contiguous()
        .view(-1, WINDOW * WINDOW, channels)
    )


def shifted_attention(value, state, prefix, heads, shift):
    batches, length, channels = value.shape
    # The oracle sizes are carried explicitly by the caller on the tensor.
    height, width = value.hw_shape
    value = value.view(batches, height, width, channels)
    pad_h = (-height) % WINDOW
    pad_w = (-width) % WINDOW
    value = F.pad(value, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    mask = None
    if shift:
        value = torch.roll(value, shifts=(-shift, -shift), dims=(1, 2))
        labels = torch.zeros((1, padded_h, padded_w, 1), dtype=value.dtype)
        h_slices = (slice(0, -WINDOW), slice(-WINDOW, -shift), slice(-shift, None))
        w_slices = (slice(0, -WINDOW), slice(-WINDOW, -shift), slice(-shift, None))
        label = 0
        for ys in h_slices:
            for xs in w_slices:
                labels[:, ys, xs, :] = label
                label += 1
        mask_values = window_partition(labels).view(-1, WINDOW * WINDOW)
        mask = mask_values.unsqueeze(1) - mask_values.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
    windows = window_partition(value)
    qkv = F.linear(windows, state[prefix + ".qkv.weight"], state[prefix + ".qkv.bias"])
    qkv = qkv.view(-1, WINDOW * WINDOW, 3, heads, channels // heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, val = qkv[0], qkv[1], qkv[2]
    attention = (query * ((channels // heads) ** -0.5)) @ key.transpose(-2, -1)
    index = state[prefix + ".relative_position_index"].view(-1)
    bias = state[prefix + ".relative_position_bias_table"][index]
    bias = bias.view(WINDOW * WINDOW, WINDOW * WINDOW, heads).permute(2, 0, 1)
    attention = attention + bias.unsqueeze(0)
    if mask is not None:
        windows_per_image = mask.shape[0]
        attention = attention.view(
            batches, windows_per_image, heads, WINDOW * WINDOW, WINDOW * WINDOW
        ) + mask.unsqueeze(0).unsqueeze(2)
        attention = attention.view(-1, heads, WINDOW * WINDOW, WINDOW * WINDOW)
    attention = attention.softmax(dim=-1)
    windows = (attention @ val).transpose(1, 2).reshape(-1, WINDOW * WINDOW, channels)
    windows = F.linear(windows, state[prefix + ".proj.weight"], state[prefix + ".proj.bias"])
    value = windows.view(
        batches, padded_h // WINDOW, padded_w // WINDOW, WINDOW, WINDOW, channels
    ).permute(0, 1, 3, 2, 4, 5).contiguous().view(batches, padded_h, padded_w, channels)
    if shift:
        value = torch.roll(value, shifts=(shift, shift), dims=(1, 2))
    return value[:, :height, :width, :].contiguous().view(batches, length, channels)


class Tokens(torch.Tensor):
    """Tensor subclass used only to carry the current (H,W) through attention."""


def with_shape(value, shape):
    value.hw_shape = shape
    return value


def run_block(value, shape, state, stage, block):
    prefix = f"image_backbone.stages.{stage}.blocks.{block}"
    identity = value
    normalized = layer_norm(value, state, prefix + ".norm1")
    normalized = with_shape(normalized, shape)
    value = identity + shifted_attention(
        normalized, state, prefix + ".attn.w_msa", HEADS[stage], 3 if block & 1 else 0
    )
    normalized = layer_norm(value, state, prefix + ".norm2")
    hidden = F.linear(
        normalized, state[prefix + ".ffn.layers.0.0.weight"],
        state[prefix + ".ffn.layers.0.0.bias"],
    )
    hidden = F.gelu(hidden)
    hidden = F.linear(
        hidden, state[prefix + ".ffn.layers.1.weight"],
        state[prefix + ".ffn.layers.1.bias"],
    )
    return value + hidden


def patch_merge(value, shape, state, stage):
    height, width = shape
    channels = value.shape[-1]
    batches = value.shape[0]
    value = value.view(batches, height, width, channels).permute(0, 3, 1, 2)
    value = F.pad(value, (0, width & 1, 0, height & 1))
    value = F.unfold(value, kernel_size=2, stride=2).transpose(1, 2)
    prefix = f"image_backbone.stages.{stage}.downsample"
    value = layer_norm(value, state, prefix + ".norm")
    value = F.linear(value, state[prefix + ".reduction.weight"])
    return value, ((height + 1) // 2, (width + 1) // 2)


def generate(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    torch.manual_seed(0x5A17BACC)
    image = torch.randn(1, 3, 32, 48)
    value = F.pad(image, (0, (-image.shape[-1]) % 4, 0, (-image.shape[-2]) % 4))
    value = F.conv2d(
        value, state["image_backbone.patch_embed.projection.weight"],
        state["image_backbone.patch_embed.projection.bias"], stride=4,
    )
    shape = value.shape[-2:]
    value = value.flatten(2).transpose(1, 2)
    value = layer_norm(value, state, "image_backbone.patch_embed.norm")
    outputs = []
    for stage, depth in enumerate(DEPTHS):
        for block in range(depth):
            value = run_block(value, shape, state, stage, block)
        if stage > 0:
            normalized = layer_norm(value, state, f"image_backbone.norm{stage}")
            feature = normalized.view(1, *shape, CHANNELS[stage]).permute(0, 3, 1, 2)
            outputs.append(feature.contiguous())
        if stage < 3:
            value, shape = patch_merge(value, shape, state, stage)
    tensors = [("swin_backbone.input", image)]
    tensors.extend((f"swin_backbone.output{i}", value) for i, value in enumerate(outputs))
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        print(f"oracle_swin_backbone: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
