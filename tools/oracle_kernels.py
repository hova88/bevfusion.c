#!/usr/bin/env python3
"""Create deterministic PyTorch operator fixtures for the scalar C backend."""

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F

from export_checkpoint import write_tensors


def named(name, tensor):
    tensor = tensor.detach().cpu().contiguous()
    return name, name.encode("utf-8"), tensor


def swin_shifted_window(query, qkv_weight, qkv_bias, relative_bias,
                        relative_index, projection_weight, projection_bias,
                        heads, window_size, shift_size):
    batches, height, width, channels = query.shape
    pad_right = (window_size - width % window_size) % window_size
    pad_bottom = (window_size - height % window_size) % window_size
    padded = F.pad(query, (0, 0, 0, pad_right, 0, pad_bottom))
    padded_height, padded_width = padded.shape[1:3]
    if shift_size:
        shifted = torch.roll(padded, shifts=(-shift_size, -shift_size), dims=(1, 2))
        image_mask = torch.zeros(1, padded_height, padded_width, 1)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                image_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = image_mask.view(
            1, padded_height // window_size, window_size,
            padded_width // window_size, window_size, 1
        ).permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0)
    else:
        shifted = padded
        attention_mask = None
    windows = shifted.view(
        batches, padded_height // window_size, window_size,
        padded_width // window_size, window_size, channels
    ).permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)
    total_windows, tokens, _ = windows.shape
    qkv = F.linear(windows, qkv_weight, qkv_bias).reshape(
        total_windows, tokens, 3, heads, channels // heads
    ).permute(2, 0, 3, 1, 4)
    q, key, value = qkv[0], qkv[1], qkv[2]
    attention = (q * ((channels // heads) ** -0.5)) @ key.transpose(-2, -1)
    position_bias = relative_bias[relative_index.reshape(-1)].view(
        tokens, tokens, heads
    ).permute(2, 0, 1)
    attention = attention + position_bias.unsqueeze(0)
    if attention_mask is not None:
        window_count = attention_mask.shape[0]
        attention = attention.view(
            total_windows // window_count, window_count, heads, tokens, tokens
        ) + attention_mask.unsqueeze(0).unsqueeze(2)
        attention = attention.view(total_windows, heads, tokens, tokens)
    attention = attention.softmax(dim=-1)
    attended = (attention @ value).transpose(1, 2).reshape(total_windows, tokens, channels)
    attended = F.linear(attended, projection_weight, projection_bias)
    shifted_output = attended.view(
        batches, padded_height // window_size, padded_width // window_size,
        window_size, window_size, channels
    ).permute(0, 1, 3, 2, 4, 5).reshape(batches, padded_height, padded_width, channels)
    if shift_size:
        shifted_output = torch.roll(shifted_output, shifts=(shift_size, shift_size), dims=(1, 2))
    return shifted_output[:, :height, :width].contiguous()


def fixtures():
    torch.manual_seed(0xBEF0510)
    result = []

    conv_input = torch.randn(2, 4, 5, 6)
    conv_weight = torch.randn(6, 2, 3, 3)
    conv_bias = torch.randn(6)
    conv_output = F.conv2d(conv_input, conv_weight, conv_bias, stride=(2, 2),
                           padding=(1, 1), groups=2)
    for name, value in (("conv.input", conv_input), ("conv.weight", conv_weight),
                        ("conv.bias", conv_bias), ("conv.output", conv_output)):
        result.append(named(name, value))

    linear_input = torch.randn(7, 5)
    linear_weight = torch.randn(4, 5)
    linear_bias = torch.randn(4)
    for name, value in (("linear.input", linear_input), ("linear.weight", linear_weight),
                        ("linear.bias", linear_bias),
                        ("linear.output", F.linear(linear_input, linear_weight, linear_bias))):
        result.append(named(name, value))

    bn_input = torch.randn(2, 3, 4, 5)
    bn_scale = torch.randn(3)
    bn_bias = torch.randn(3)
    bn_mean = torch.randn(3)
    bn_variance = torch.rand(3) + 0.2
    bn_output = F.batch_norm(bn_input, bn_mean, bn_variance, bn_scale, bn_bias,
                             training=False, momentum=0.1, eps=1e-3)
    for name, value in (("bn.input", bn_input), ("bn.scale", bn_scale),
                        ("bn.bias", bn_bias), ("bn.mean", bn_mean),
                        ("bn.variance", bn_variance), ("bn.output", bn_output)):
        result.append(named(name, value))

    norm_input = torch.randn(5, 7)
    norm_scale = torch.randn(7)
    norm_bias = torch.randn(7)
    norm_output = F.layer_norm(norm_input, (7,), norm_scale, norm_bias, 1e-5)
    for name, value in (("layer_norm.input", norm_input), ("layer_norm.scale", norm_scale),
                        ("layer_norm.bias", norm_bias), ("layer_norm.output", norm_output)):
        result.append(named(name, value))

    softmax_input = torch.randn(4, 9) * 4
    softmax_input[0, 0] = 100.0
    softmax_input[1, 1] = -100.0
    result.append(named("softmax.input", softmax_input))
    result.append(named("softmax.output", F.softmax(softmax_input, dim=-1)))

    gelu_input = torch.linspace(-8.0, 8.0, 65)
    result.append(named("gelu.input", gelu_input))
    result.append(named("gelu.output", F.gelu(gelu_input, approximate="none")))
    result.append(named("relu.output", F.relu(gelu_input)))

    vfe_points = torch.randn(4, 6, 5)
    vfe_counts = torch.tensor([6, 3, 1, 0], dtype=torch.int64)
    divisor = vfe_counts.clamp(min=1).to(torch.float32).unsqueeze(1)
    mask = torch.arange(6).unsqueeze(0) < vfe_counts.unsqueeze(1)
    vfe_output = (vfe_points * mask.unsqueeze(2)).sum(dim=1) / divisor
    for name, value in (("vfe.points", vfe_points), ("vfe.counts", vfe_counts),
                        ("vfe.output", vfe_output)):
        result.append(named(name, value))

    topk_input = torch.randn(3, 17)
    topk_input += torch.arange(17, dtype=torch.float32) * 1e-4
    topk_values, topk_indices = torch.topk(topk_input, 5, dim=1, largest=True, sorted=True)
    for name, value in (("topk.input", topk_input), ("topk.values", topk_values),
                        ("topk.indices", topk_indices)):
        result.append(named(name, value))

    sparse_coords = torch.tensor([
        [0, 0, 0, 0], [0, 0, 1, 2], [0, 1, 2, 3], [0, 1, 3, 5],
        [0, 2, 0, 4], [0, 2, 4, 1], [0, 3, 2, 0], [0, 3, 4, 5],
    ], dtype=torch.int64)
    sparse_features = torch.randn(8, 3)
    dense = torch.zeros(1, 3, 4, 5, 6)
    mask = torch.zeros(1, 1, 4, 5, 6)
    for index, coord in enumerate(sparse_coords):
        b, z, y, x = coord.tolist()
        dense[b, :, z, y, x] = sparse_features[index]
        mask[b, :, z, y, x] = 1
    subm_weight_oi = torch.randn(4, 3, 3, 3, 3)
    subm_bias = torch.randn(4)
    subm_dense = F.conv3d(dense, subm_weight_oi, subm_bias, padding=1)
    subm_output = torch.stack([subm_dense[b, :, z, y, x] for b, z, y, x in sparse_coords])
    stride_weight_oi = torch.randn(5, 3, 3, 3, 3)
    stride_bias = torch.randn(5)
    stride_dense = F.conv3d(dense, stride_weight_oi, stride_bias, stride=2, padding=1)
    occupancy = F.conv3d(mask, torch.ones(1, 1, 3, 3, 3), stride=2, padding=1) > 0
    stride_coords = occupancy.nonzero(as_tuple=False)[:, [0, 2, 3, 4]]
    stride_output = torch.stack([stride_dense[b, :, z, y, x] for b, z, y, x in stride_coords])
    sparse_items = (
        ("sparse.coords", sparse_coords), ("sparse.features", sparse_features),
        ("sparse.subm.weight", subm_weight_oi.permute(2, 3, 4, 1, 0)),
        ("sparse.subm.bias", subm_bias), ("sparse.subm.output", subm_output),
        ("sparse.stride.weight", stride_weight_oi.permute(2, 3, 4, 1, 0)),
        ("sparse.stride.bias", stride_bias), ("sparse.stride.coords", stride_coords),
        ("sparse.stride.output", stride_output),
    )
    for name, value in sparse_items:
        result.append(named(name, value))

    voxel_points = torch.tensor([
        [-1.8, -1.7, -0.9, 0.1, 0.0], [-1.2, -1.1, -0.2, 0.2, 0.1],
        [-1.1, -1.2, -0.1, 0.3, 0.2], [-1.3, -1.4, -0.3, 0.4, 0.3],
        [0.1, 0.1, 0.1, 0.5, 0.4], [1.9, 1.9, 0.9, 0.6, 0.5],
        [-0.1, 1.1, -0.4, 0.7, 0.6], [1.1, -0.1, 0.3, 0.8, 0.7],
        [0.2, -1.2, 0.4, 0.9, 0.8], [0.3, 1.3, 0.2, 1.0, 0.9],
        [2.0, 0.0, 0.0, 1.1, 1.0], [float("nan"), 0.0, 0.0, 1.2, 1.1],
    ], dtype=torch.float32)
    minimum = (-2.0, -2.0, -1.0)
    voxel_size = (1.0, 1.0, 1.0)
    max_voxels, max_points = 5, 3
    voxel_map = {}
    voxel_rows, voxel_coords, voxel_counts = [], [], []
    voxel_stats = [len(voxel_points), 0, 0, 0, 0, 0]
    for point in voxel_points:
        if not torch.isfinite(point).all():
            voxel_stats[2] += 1
            continue
        xyz = point[:3]
        if not bool(((xyz >= torch.tensor(minimum)) & (xyz < torch.tensor((2.0, 2.0, 1.0)))).all()):
            voxel_stats[3] += 1
            continue
        x, y, z = [int(torch.floor((xyz[i] - minimum[i]) / voxel_size[i])) for i in range(3)]
        key = (z, y, x)
        if key not in voxel_map:
            if len(voxel_rows) == max_voxels:
                voxel_stats[4] += 1
                continue
            voxel_map[key] = len(voxel_rows)
            voxel_rows.append(torch.zeros(max_points, 5))
            voxel_coords.append((0, z, y, x))
            voxel_counts.append(0)
        index = voxel_map[key]
        if voxel_counts[index] == max_points:
            voxel_stats[5] += 1
            continue
        voxel_rows[index][voxel_counts[index]] = point
        voxel_counts[index] += 1
        voxel_stats[1] += 1
    for name, value in (
        ("voxel.points", voxel_points), ("voxel.values", torch.stack(voxel_rows)),
        ("voxel.coords", torch.tensor(voxel_coords, dtype=torch.int64)),
        ("voxel.counts", torch.tensor(voxel_counts, dtype=torch.int64)),
        ("voxel.stats", torch.tensor(voxel_stats, dtype=torch.int64)),
    ):
        result.append(named(name, value))

    batches, cameras, depths, height, width, channels = 2, 2, 4, 3, 5, 3
    depth_values = torch.arange(1, depths + 1, dtype=torch.float32).view(depths, 1, 1).expand(-1, height, width)
    image_x = torch.linspace(0, 19, width).view(1, 1, width).expand(depths, height, width)
    image_y = torch.linspace(0, 11, height).view(1, height, 1).expand(depths, height, width)
    frustum = torch.stack((image_x, image_y, depth_values), dim=-1)
    camera_rotation = torch.empty(batches, cameras, 3, 3)
    camera_translation = torch.empty(batches, cameras, 3)
    intrinsics = torch.empty(batches, cameras, 3, 3)
    post_rotation = torch.empty(batches, cameras, 3, 3)
    post_translation = torch.empty(batches, cameras, 3)
    for batch in range(batches):
        for camera in range(cameras):
            angle = 0.13 * (camera + 1) - 0.04 * batch
            cosine, sine = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            camera_rotation[batch, camera] = torch.tensor([
                [cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]
            ])
            camera_translation[batch, camera] = torch.tensor([
                -2.0 + camera * 3.0, -1.0 + batch * 2.0, 0.2 * camera
            ])
            intrinsics[batch, camera] = torch.tensor([
                [8.0 + camera, 0.0, 9.0], [0.0, 7.0 + batch, 5.0], [0.0, 0.0, 1.0]
            ])
            scale = 0.9 + 0.05 * camera
            post_rotation[batch, camera] = torch.tensor([
                [scale, -0.03, 0.0], [0.02, scale, 0.0], [0.0, 0.0, 1.0]
            ])
            post_translation[batch, camera] = torch.tensor([0.2 * camera, -0.1 * batch, 0.0])
    extra_rotation = torch.empty(batches, 3, 3)
    extra_translation = torch.empty(batches, 3)
    for batch in range(batches):
        angle = -0.07 + 0.11 * batch
        cosine, sine = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        extra_rotation[batch] = torch.tensor([
            [cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]
        ])
        extra_translation[batch] = torch.tensor([0.15 * batch, -0.2 * batch, 0.1])
    points = frustum.view(1, 1, depths, height, width, 3) - post_translation.view(
        batches, cameras, 1, 1, 1, 3
    )
    points = torch.linalg.inv(post_rotation).view(
        batches, cameras, 1, 1, 1, 3, 3
    ).matmul(points.unsqueeze(-1))
    points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), dim=5)
    combine = camera_rotation.matmul(torch.linalg.inv(intrinsics))
    geometry = combine.view(batches, cameras, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    geometry += camera_translation.view(batches, cameras, 1, 1, 1, 3)
    geometry = extra_rotation.view(batches, 1, 1, 1, 1, 3, 3).matmul(
        geometry.unsqueeze(-1)
    ).squeeze(-1)
    geometry += extra_translation.view(batches, 1, 1, 1, 1, 3)
    depth_logits = torch.randn(batches, cameras, depths, height, width)
    context = torch.randn(batches, cameras, channels, height, width)
    probability = depth_logits.softmax(dim=2)
    lifted = probability.unsqueeze(-1) * context.permute(0, 1, 3, 4, 2).unsqueeze(2)
    grid_minimum = torch.tensor([-6.0, -6.0, -3.0])
    grid_step = torch.tensor([2.0, 2.0, 2.0])
    grid_cells = (6, 6, 4)
    grid_coords = ((geometry - grid_minimum) / grid_step).long()
    bev = torch.zeros(batches, channels * grid_cells[2], grid_cells[0], grid_cells[1])
    for batch in range(batches):
        for camera in range(cameras):
            for depth in range(depths):
                for y in range(height):
                    for x in range(width):
                        gx, gy, gz = grid_coords[batch, camera, depth, y, x].tolist()
                        if 0 <= gx < grid_cells[0] and 0 <= gy < grid_cells[1] and 0 <= gz < grid_cells[2]:
                            for channel in range(channels):
                                bev[batch, channel * grid_cells[2] + gz, gx, gy] += \
                                    lifted[batch, camera, depth, y, x, channel]
    lss_items = (
        ("lss.frustum", frustum), ("lss.camera_rotation", camera_rotation),
        ("lss.camera_translation", camera_translation), ("lss.intrinsics", intrinsics),
        ("lss.post_rotation", post_rotation), ("lss.post_translation", post_translation),
        ("lss.extra_rotation", extra_rotation), ("lss.extra_translation", extra_translation),
        ("lss.geometry", geometry), ("lss.depth_logits", depth_logits),
        ("lss.context", context), ("lss.lifted", lifted), ("lss.bev", bev),
    )
    for name, value in lss_items:
        result.append(named(name, value))

    swin_batches, swin_height, swin_width = 2, 5, 7
    swin_channels, swin_heads, swin_window = 8, 2, 3
    swin_input = torch.randn(swin_batches, swin_height, swin_width, swin_channels)
    swin_qkv_weight = torch.randn(3 * swin_channels, swin_channels) * 0.25
    swin_qkv_bias = torch.randn(3 * swin_channels) * 0.1
    swin_projection_weight = torch.randn(swin_channels, swin_channels) * 0.25
    swin_projection_bias = torch.randn(swin_channels) * 0.1
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(swin_window), torch.arange(swin_window), indexing="ij"
    )).flatten(1)
    relative = coordinates[:, :, None] - coordinates[:, None, :]
    relative = relative.permute(1, 2, 0).contiguous()
    relative[:, :, 0] += swin_window - 1
    relative[:, :, 1] += swin_window - 1
    relative[:, :, 0] *= 2 * swin_window - 1
    swin_relative_index = relative.sum(-1).to(torch.int64)
    swin_relative_bias = torch.randn((2 * swin_window - 1) ** 2, swin_heads) * 0.05
    swin_regular = swin_shifted_window(
        swin_input, swin_qkv_weight, swin_qkv_bias, swin_relative_bias,
        swin_relative_index, swin_projection_weight, swin_projection_bias,
        swin_heads, swin_window, 0
    )
    swin_shifted = swin_shifted_window(
        swin_input, swin_qkv_weight, swin_qkv_bias, swin_relative_bias,
        swin_relative_index, swin_projection_weight, swin_projection_bias,
        swin_heads, swin_window, 1
    )
    for name, value in (
        ("swin.input", swin_input), ("swin.qkv_weight", swin_qkv_weight),
        ("swin.qkv_bias", swin_qkv_bias), ("swin.projection_weight", swin_projection_weight),
        ("swin.projection_bias", swin_projection_bias),
        ("swin.relative_index", swin_relative_index),
        ("swin.relative_bias", swin_relative_bias),
        ("swin.regular_output", swin_regular), ("swin.shifted_output", swin_shifted),
    ):
        result.append(named(name, value))

    tf_batches, tf_classes, tf_height, tf_width, tf_proposals = 2, 10, 8, 9, 12
    tf_dense_logits = torch.randn(tf_batches, tf_classes, tf_height, tf_width)
    tf_heatmap = tf_dense_logits.sigmoid()
    tf_local_max = torch.zeros_like(tf_heatmap)
    tf_local_max[:, :, 1:-1, 1:-1] = F.max_pool2d(tf_heatmap, 3, stride=1, padding=0)
    tf_local_max[:, 8] = tf_heatmap[:, 8]
    tf_local_max[:, 9] = tf_heatmap[:, 9]
    tf_suppressed = tf_heatmap * (tf_heatmap == tf_local_max)
    tf_top = tf_suppressed.view(tf_batches, -1).argsort(dim=-1, descending=True)[..., :tf_proposals]
    tf_classes_selected = tf_top // (tf_height * tf_width)
    tf_indices_selected = tf_top % (tf_height * tf_width)
    tf_scores_selected = tf_suppressed.view(tf_batches, -1).gather(1, tf_top)
    tf_query_scores = tf_suppressed.view(tf_batches, tf_classes, -1).gather(
        2, tf_indices_selected[:, None].expand(-1, tf_classes, -1)
    )
    tf_prediction_logits = torch.randn(tf_batches, tf_classes, tf_proposals)
    tf_center = torch.randn(tf_batches, 2, tf_proposals) * 3 + 90
    tf_height_values = torch.randn(tf_batches, 1, tf_proposals)
    tf_dimension_log = torch.randn(tf_batches, 3, tf_proposals) * 0.3
    tf_rotation = torch.randn(tf_batches, 2, tf_proposals)
    tf_velocity = torch.randn(tf_batches, 2, tf_proposals)
    tf_final_scores = torch.empty(tf_batches, tf_proposals)
    for batch in range(tf_batches):
        for proposal in range(tf_proposals):
            label = tf_classes_selected[batch, proposal]
            tf_final_scores[batch, proposal] = tf_prediction_logits[batch, label, proposal].sigmoid() * \
                tf_query_scores[batch, label, proposal]
    tf_boxes = torch.empty(tf_batches, tf_proposals, 9)
    tf_boxes[..., 0] = tf_center[:, 0] * 8 * 0.075 - 54.0
    tf_boxes[..., 1] = tf_center[:, 1] * 8 * 0.075 - 54.0
    tf_boxes[..., 2] = tf_height_values[:, 0]
    tf_boxes[..., 3:6] = tf_dimension_log.permute(0, 2, 1).exp()
    tf_boxes[..., 6] = torch.atan2(tf_rotation[:, 0], tf_rotation[:, 1])
    tf_boxes[..., 7:9] = tf_velocity.permute(0, 2, 1)
    tf_keep = (tf_final_scores > 0.1) & (tf_boxes[..., :3] >= torch.tensor([-61.2, -61.2, -10.0])).all(2) & \
        (tf_boxes[..., :3] <= torch.tensor([61.2, 61.2, 10.0])).all(2)
    for name, value in (
        ("transfusion.dense_logits", tf_dense_logits),
        ("transfusion.suppressed", tf_suppressed),
        ("transfusion.proposal_scores", tf_scores_selected),
        ("transfusion.proposal_classes", tf_classes_selected),
        ("transfusion.proposal_indices", tf_indices_selected),
        ("transfusion.query_scores", tf_query_scores),
        ("transfusion.prediction_logits", tf_prediction_logits),
        ("transfusion.center", tf_center), ("transfusion.height", tf_height_values),
        ("transfusion.dimension_log", tf_dimension_log),
        ("transfusion.rotation", tf_rotation), ("transfusion.velocity", tf_velocity),
        ("transfusion.boxes", tf_boxes), ("transfusion.final_scores", tf_final_scores),
        ("transfusion.keep", tf_keep.to(torch.int64)),
    ):
        result.append(named(name, value))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        write_tensors(fixtures(), args.output)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"oracle_kernels: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
