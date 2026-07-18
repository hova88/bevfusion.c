#!/usr/bin/env python3
"""Real-weight oracle for the chained BEV stage and TransFusion decoder."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors
from oracle_bev_stage import batch_norm, conv_bn_relu
from oracle_transfusion_decoder import attention, named, position_embedding, prediction_head

C = 128


def generate(checkpoint: pathlib.Path, output: pathlib.Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    torch.manual_seed(0xC0DA7A11)
    height = width = 8
    proposals = 20
    fusion = torch.randn(1, 336, height, width)
    value = conv_bn_relu(fusion, state, "fuser.conv.0.weight", "fuser.conv.1")
    upsampled = []
    conv_indices = (1, 4, 7, 10, 13, 16)
    bn_indices = (2, 5, 8, 11, 14, 17)
    for block in range(2):
        for layer, (conv_index, bn_index) in enumerate(zip(conv_indices, bn_indices)):
            stride = 2 if block == 1 and layer == 0 else 1
            value = F.pad(value, (1, 1, 1, 1)) if layer == 0 else value
            value = conv_bn_relu(value, state,
                f"backbone_2d.blocks.{block}.{conv_index}.weight",
                f"backbone_2d.blocks.{block}.{bn_index}", stride=stride,
                padding=0 if layer == 0 else 1)
        if block == 0:
            up = F.conv2d(value, state["backbone_2d.deblocks.0.0.weight"])
            up = F.relu(batch_norm(up, state, "backbone_2d.deblocks.0.1"))
        else:
            up = F.conv_transpose2d(value, state["backbone_2d.deblocks.1.0.weight"], stride=2)
            up = F.relu(batch_norm(up, state, "backbone_2d.deblocks.1.1"))
        upsampled.append(up)
    spatial = torch.cat(upsampled, dim=1)
    shared = F.conv2d(spatial, state["dense_head.shared_conv.weight"],
                      state["dense_head.shared_conv.bias"], padding=1)
    mid = F.conv2d(shared, state["dense_head.heatmap_head.0.conv.weight"], padding=1)
    mid = F.relu(batch_norm(mid, state, "dense_head.heatmap_head.0.bn"))
    dense = F.conv2d(mid, state["dense_head.heatmap_head.1.weight"],
                     state["dense_head.heatmap_head.1.bias"], padding=1)

    heatmap = dense.sigmoid()
    local = torch.zeros_like(heatmap)
    local[:, :, 1:-1, 1:-1] = F.max_pool2d(heatmap, 3, stride=1)
    local[:, 8] = heatmap[:, 8]
    local[:, 9] = heatmap[:, 9]
    suppressed = heatmap * (heatmap == local)
    top = suppressed.flatten(1).argsort(dim=-1, descending=True)[..., :proposals]
    labels, indices = top // (height * width), top % (height * width)
    feature = shared.flatten(2)
    query = feature.gather(2, indices[:, None].expand(-1, C, -1))
    one_hot = F.one_hot(labels, 10).permute(0, 2, 1).float()
    query += F.conv1d(one_hot, state["dense_head.class_encoding.weight"],
                      state["dense_head.class_encoding.bias"])
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    positions = torch.stack((xx + .5, yy + .5), -1).view(1, -1, 2).float()
    query_raw = positions.gather(1, indices[..., None].expand(-1, -1, 2))
    query_pos = position_embedding(query_raw, state, "dense_head.decoder.self_posembed")
    key_pos = position_embedding(positions, state, "dense_head.decoder.cross_posembed")
    tokens, keys = query.transpose(1, 2), feature.transpose(1, 2)
    update = attention(tokens + query_pos, tokens + query_pos, state,
                       "dense_head.decoder.self_attn")
    tokens = F.layer_norm(tokens + update, (C,), state["dense_head.decoder.norm1.weight"],
                          state["dense_head.decoder.norm1.bias"], 1e-5)
    update = attention(tokens + query_pos, keys + key_pos, state,
                       "dense_head.decoder.multihead_attn")
    tokens = F.layer_norm(tokens + update, (C,), state["dense_head.decoder.norm2.weight"],
                          state["dense_head.decoder.norm2.bias"], 1e-5)
    update = F.relu(F.linear(tokens, state["dense_head.decoder.linear1.weight"],
                             state["dense_head.decoder.linear1.bias"]))
    update = F.linear(update, state["dense_head.decoder.linear2.weight"],
                      state["dense_head.decoder.linear2.bias"])
    tokens = F.layer_norm(tokens + update, (C,), state["dense_head.decoder.norm3.weight"],
                          state["dense_head.decoder.norm3.bias"], 1e-5)
    query_bcp = tokens.transpose(1, 2)
    heads = {name: prediction_head(query_bcp, state, name)
             for name in ("center", "height", "dim", "rot", "vel", "heatmap")}
    heads["center"] += query_raw.transpose(1, 2)
    query_scores = suppressed.flatten(2).gather(2, indices[:, None].expand(-1, 10, -1))
    selected = F.one_hot(labels, 10).permute(0, 2, 1)
    class_scores = heads["heatmap"].sigmoid() * query_scores * selected
    final_scores, final_labels = class_scores.max(dim=1)
    metric_center = heads["center"].clone()
    metric_center[:, 0] = metric_center[:, 0] * .6 - 54.0
    metric_center[:, 1] = metric_center[:, 1] * .6 - 54.0
    boxes = torch.cat((metric_center, heads["height"], heads["dim"].exp(),
                       torch.atan2(heads["rot"][:, :1], heads["rot"][:, 1:2]),
                       heads["vel"]), dim=1).transpose(1, 2)
    tensors = [("cuda_tail.input", fusion), ("cuda_tail.spatial", spatial),
               ("cuda_tail.shared", shared), ("cuda_tail.dense_heatmap", dense),
               ("cuda_tail.labels", labels), ("cuda_tail.indices", indices),
               ("cuda_tail.query_scores", query_scores), ("cuda_tail.boxes", boxes),
               ("cuda_tail.scores", final_scores), ("cuda_tail.final_labels", final_labels)]
    tensors += [(f"cuda_tail.{name}", value) for name, value in heads.items()]
    write_tensors([named(name, value) for name, value in tensors], output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_cuda_tail: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
