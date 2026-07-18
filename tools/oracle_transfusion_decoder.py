#!/usr/bin/env python3
"""Generate a real-checkpoint oracle for the complete TransFusion decoder."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors

C = 128
HEADS = 8


def named(name, tensor):
    value = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), value


def batch_norm_1d(value, state, prefix):
    return F.batch_norm(
        value, state[prefix + ".running_mean"], state[prefix + ".running_var"],
        state[prefix + ".weight"], state[prefix + ".bias"],
        training=False, momentum=0.1, eps=1e-5,
    )


def position_embedding(position, state, prefix):
    value = position.transpose(1, 2)
    value = F.conv1d(
        value, state[prefix + ".position_embedding_head.0.weight"],
        state[prefix + ".position_embedding_head.0.bias"],
    )
    value = F.relu(batch_norm_1d(value, state, prefix + ".position_embedding_head.1"))
    return F.conv1d(
        value, state[prefix + ".position_embedding_head.3.weight"],
        state[prefix + ".position_embedding_head.3.bias"],
    ).transpose(1, 2)


def attention(query, key_value, state, prefix):
    weight = state[prefix + ".in_proj_weight"]
    bias = state[prefix + ".in_proj_bias"]
    query_projection = F.linear(query, weight[:C], bias[:C])
    key_projection = F.linear(key_value, weight[C:2 * C], bias[C:2 * C])
    value_projection = F.linear(key_value, weight[2 * C:], bias[2 * C:])
    q = query_projection.view(1, -1, HEADS, C // HEADS).transpose(1, 2)
    k = key_projection.view(1, -1, HEADS, C // HEADS).transpose(1, 2)
    v = value_projection.view(1, -1, HEADS, C // HEADS).transpose(1, 2)
    weights = ((q @ k.transpose(-2, -1)) * ((C // HEADS) ** -0.5)).softmax(dim=-1)
    value = (weights @ v).transpose(1, 2).reshape(1, -1, C)
    return F.linear(value, state[prefix + ".out_proj.weight"],
                    state[prefix + ".out_proj.bias"])


def prediction_head(query_bcp, state, name):
    prefix = f"dense_head.prediction_head.{name}"
    value = F.conv1d(query_bcp, state[prefix + ".0.0.weight"])
    value = F.relu(batch_norm_1d(value, state, prefix + ".0.1"))
    return F.conv1d(value, state[prefix + ".1.weight"], state[prefix + ".1.bias"])


def generate(checkpoint, output):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    torch.manual_seed(0x7A4F510)
    height = width = 5
    proposals = 12
    shared = torch.randn(1, C, height, width)
    dense_heatmap = torch.randn(1, 10, height, width) * 2.0
    heatmap = dense_heatmap.sigmoid()
    local_max = torch.zeros_like(heatmap)
    local_max[:, :, 1:-1, 1:-1] = F.max_pool2d(heatmap, 3, stride=1)
    local_max[:, 8] = heatmap[:, 8]
    local_max[:, 9] = heatmap[:, 9]
    suppressed = heatmap * (heatmap == local_max)
    flattened = suppressed.view(1, -1)
    top = flattened.argsort(dim=-1, descending=True)[..., :proposals]
    labels = top // (height * width)
    indices = top % (height * width)
    feature = shared.flatten(2)
    query = feature.gather(2, indices[:, None].expand(-1, C, -1))
    one_hot = F.one_hot(labels, num_classes=10).permute(0, 2, 1).float()
    query = query + F.conv1d(one_hot, state["dense_head.class_encoding.weight"],
                             state["dense_head.class_encoding.bias"])
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    bev_position = torch.stack((xx + 0.5, yy + 0.5), dim=-1).view(1, -1, 2).float()
    query_position_raw = bev_position.gather(1, indices[..., None].expand(-1, -1, 2))
    query_position = position_embedding(
        query_position_raw, state, "dense_head.decoder.self_posembed"
    )
    key_position = position_embedding(
        bev_position, state, "dense_head.decoder.cross_posembed"
    )
    query_tokens = query.transpose(1, 2)
    key_tokens = feature.transpose(1, 2)
    value = attention(query_tokens + query_position, query_tokens + query_position,
                      state, "dense_head.decoder.self_attn")
    query_tokens = F.layer_norm(
        query_tokens + value, (C,), state["dense_head.decoder.norm1.weight"],
        state["dense_head.decoder.norm1.bias"], 1e-5,
    )
    value = attention(query_tokens + query_position, key_tokens + key_position,
                      state, "dense_head.decoder.multihead_attn")
    query_tokens = F.layer_norm(
        query_tokens + value, (C,), state["dense_head.decoder.norm2.weight"],
        state["dense_head.decoder.norm2.bias"], 1e-5,
    )
    value = F.linear(query_tokens, state["dense_head.decoder.linear1.weight"],
                     state["dense_head.decoder.linear1.bias"])
    value = F.relu(value)
    value = F.linear(value, state["dense_head.decoder.linear2.weight"],
                     state["dense_head.decoder.linear2.bias"])
    query_tokens = F.layer_norm(
        query_tokens + value, (C,), state["dense_head.decoder.norm3.weight"],
        state["dense_head.decoder.norm3.bias"], 1e-5,
    )
    query_bcp = query_tokens.transpose(1, 2)
    heads = {name: prediction_head(query_bcp, state, name)
             for name in ("center", "height", "dim", "rot", "vel", "heatmap")}
    heads["center"] = heads["center"] + query_position_raw.transpose(1, 2)
    query_scores = suppressed.flatten(2).gather(
        2, indices[:, None].expand(-1, 10, -1)
    )
    selected = F.one_hot(labels, num_classes=10).permute(0, 2, 1)
    final_class_scores = heads["heatmap"].sigmoid() * query_scores * selected
    final_scores, final_labels = final_class_scores.max(dim=1)
    metric_center = heads["center"].clone()
    metric_center[:, 0] = metric_center[:, 0] * 0.6 - 54.0
    metric_center[:, 1] = metric_center[:, 1] * 0.6 - 54.0
    boxes = torch.cat((
        metric_center, heads["height"], heads["dim"].exp(),
        torch.atan2(heads["rot"][:, :1], heads["rot"][:, 1:2]),
        heads["vel"],
    ), dim=1).transpose(1, 2)
    tensors = [
        ("transfusion_decoder.shared", shared),
        ("transfusion_decoder.dense_heatmap", dense_heatmap),
        ("transfusion_decoder.labels", labels),
        ("transfusion_decoder.indices", indices),
        ("transfusion_decoder.query", query_bcp),
        ("transfusion_decoder.query_scores", query_scores),
        ("transfusion_decoder.boxes", boxes),
        ("transfusion_decoder.scores", final_scores),
        ("transfusion_decoder.final_labels", final_labels),
    ]
    for name in ("center", "height", "dim", "rot", "vel", "heatmap"):
        tensors.append((f"transfusion_decoder.{name}", heads[name]))
    write_tensors([named(name, value) for name, value in tensors], output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        generate(args.checkpoint, args.output)
    except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_transfusion_decoder: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
