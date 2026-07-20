#!/usr/bin/env python3
"""Turn the frozen OpenPCDet BEVFusion checkpoint into readable model docs.

The checkpoint is the authority for stored tensor names, shapes, dtypes, and
element counts.  OpenPCDet's pinned config describes architecture semantics;
the activation pipe is the shape contract implemented and tested by this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OPENPCDET_COMMIT = "233f849829b6ac19afb8af8837a0246890908755"
OPENPCDET_ROOT = f"https://github.com/open-mmlab/OpenPCDet/tree/{OPENPCDET_COMMIT}"
CONFIG_URL = f"{OPENPCDET_ROOT}/tools/cfgs/nuscenes_models/bevfusion.yaml"

STAGE_SPECS = (
    ("camera", "Camera · Swin-T", ("image_backbone",), "Window attention and MLP blocks"),
    ("neck", "Camera · FPN", ("neck",), "Three Swin scales become 256-channel features"),
    ("depth", "Camera · depth + LSS", ("vtransform",), "Depth distribution, context, lift-splat and downsample"),
    ("lidar", "LiDAR · sparse CNN", ("backbone_3d",), "MeanVFE followed by VoxelResBackBone8x"),
    ("fusion", "BEV · sensor fusion", ("fuser",), "Camera and LiDAR features meet by metric cell"),
    ("bev", "BEV · dense backbone", ("backbone_2d",), "Two-scale BaseBEVBackbone"),
    ("head", "Detection · TransFusion", ("dense_head",), "Heatmap proposals, attention and metric heads"),
)

# These are activation contracts, not data recovered from the state_dict.
FLOW = (
    ("input", "Sensor input", "RGB + points", "[6, 3, 256, 704] + [P, 5]", "BFI v1 runtime input"),
    ("depth-raster", "Depth raster", "points + calibration", "[6, 1, 256, 704]", "Nearest projected LiDAR depth"),
    ("swin", "Swin-T", "[6, 3, 256, 704]", "[6, 192, 32, 88] · [6, 384, 16, 44] · [6, 768, 8, 22]", "Stages 1–3 from OpenPCDet OUT_INDICES"),
    ("fpn", "Generalized LSS FPN", "three Swin scales", "[6, 256, 32, 88] · [6, 256, 16, 44]", "C runtime consumes the first two outputs"),
    ("depth-head", "DepthLSSTransform head", "image feature + depth raster", "depth [6, 118, 32, 88] · context [6, 80, 32, 88]", "118 bins from [1 m, 60 m) at 0.5 m"),
    ("lss", "Lift-splat", "depth × context + calibration", "[1, 80, 360, 360]", "0.3 m metric BEV cells"),
    ("image-down", "Image BEV downsample", "[1, 80, 360, 360]", "[1, 80, 180, 180]", "Learned stride-2 downsample"),
    ("voxel", "Voxelize + MeanVFE", "[P, 5]", "[V, 5] + coords [V, 4]", "V ≤ configured voxel capacity"),
    ("sparse", "VoxelResBackBone8x", "sparse [V, 5]", "[1, 256, 180, 180]", "HeightCompression at sparse/dense boundary"),
    ("concat", "Metric BEV concat", "camera 80ch + LiDAR 256ch", "[1, 336, 180, 180]", "No learned geometric alignment here"),
    ("bev-backbone", "ConvFuser + BaseBEVBackbone", "[1, 336, 180, 180]", "spatial [1, 512, 180, 180] · shared [1, 128, 180, 180]", "Dense BEV feature extraction"),
    ("proposal", "Heatmap + top-k", "shared [1, 128, 180, 180]", "heatmap [1, 10, 180, 180] · queries [1, 128, 200]", "200 stable proposals"),
    ("decoder", "TransFusion decoder", "queries + shared BEV", "center 2 · height 1 · dim 3 · rot 2 · vel 2 · class 10, each × 200", "One cross-attention decoder layer"),
    ("output", "Metric decode + rotated filter", "raw proposal fields", "≤ 200 × 11 public fields", "Canonical bf_detections ABI"),
)

TENSOR_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked",
                   ".in_proj_weight", ".in_proj_bias", ".weight", ".bias")
CONFIG_MARKERS = (
    "NAME: BevFusion", "NAME: VoxelResBackBone8x", "NAME: SwinTransformer",
    "DEPTHS: [2, 2, 6, 2]", "NAME: GeneralizedLSSFPN",
    "NAME: DepthLSSTransform", "DBOUND: [1.0, 60.0, 0.5]",
    "NAME: ConvFuser", "IN_CHANNEL: 336", "NAME: BaseBEVBackbone",
    "NAME: TransFusionHead", "NUM_PROPOSALS: 200",
)


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    bytes: int


def product(shape: Iterable[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def module_name(tensor_name: str) -> str:
    for suffix in TENSOR_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def stage_for(name: str) -> str:
    prefix = name.split(".", 1)[0]
    for stage_id, _title, prefixes, _copy in STAGE_SPECS:
        if prefix in prefixes:
            return stage_id
    return "other"


def infer_operator(name: str, tensors: list[TensorRecord]) -> str:
    names = {tensor.name for tensor in tensors}
    weight = next((tensor for tensor in tensors
                   if tensor.name.endswith((".weight", "_weight"))), None)
    lowered = name.lower()
    if any(value.endswith(".running_mean") for value in names):
        return "BatchNorm"
    if "relative_position_bias_table" in lowered:
        return "Learned table"
    if "relative_position_index" in lowered or "bev_pos" in lowered:
        return "Buffer"
    if weight is None:
        return "Buffer"
    rank = len(weight.shape)
    if rank == 5:
        return "SparseConv3d"
    if rank == 4:
        return "ConvTranspose2d" if "deblocks" in lowered else "Conv2d"
    if rank == 3:
        return "Conv1d"
    if rank == 2:
        if any(token in lowered for token in ("self_attn", "cross_attn")):
            return "MultiheadAttention"
        return "Linear"
    if rank == 1 and any(token in lowered for token in ("norm", ".ln", ".bn")):
        return "LayerNorm"
    return "Affine"


def load_checkpoint(path: Path) -> list[TensorRecord]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required: pip install -r requirements/export.txt") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain model_state, state_dict, or a state dictionary")
    records: list[TensorRecord] = []
    for name, value in state.items():
        if name == "global_step" or name.endswith(".num_batches_tracked"):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}: expected tensor, got {type(value).__name__}")
        shape = tuple(int(dim) for dim in value.shape)
        records.append(TensorRecord(name, shape, str(value.dtype).removeprefix("torch."),
                                    int(value.numel()), int(value.numel() * value.element_size())))
    if not records:
        raise ValueError("checkpoint contains no tensors")
    return records


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    missing = [marker for marker in CONFIG_MARKERS if marker not in content]
    if missing:
        raise ValueError(f"{path}: not the frozen OpenPCDet BEVFusion config; "
                         f"missing {', '.join(missing)}")


def build_model(records: list[TensorRecord], checkpoint: Path, config: Path) -> dict[str, Any]:
    validate_config(config)
    grouped: dict[str, list[TensorRecord]] = {}
    for tensor in records:
        grouped.setdefault(module_name(tensor.name), []).append(tensor)
    modules = []
    for name, tensors in grouped.items():
        shapes = [list(tensor.shape) for tensor in tensors]
        modules.append({
            "name": name,
            "stage": stage_for(name),
            "operator": infer_operator(name, tensors),
            "tensor_shapes": shapes,
            "tensor_names": [tensor.name for tensor in tensors],
            "dtypes": sorted({tensor.dtype for tensor in tensors}),
            "elements": sum(tensor.elements for tensor in tensors),
            "bytes": sum(tensor.bytes for tensor in tensors),
        })
    stage_order = {spec[0]: index for index, spec in enumerate(STAGE_SPECS)}
    modules.sort(key=lambda module: stage_order.get(module["stage"], len(stage_order)))
    stages = []
    for stage_id, title, prefixes, description in STAGE_SPECS:
        selected = [module for module in modules if module["stage"] == stage_id]
        counter = Counter(module["operator"] for module in selected)
        stages.append({
            "id": stage_id, "title": title, "description": description,
            "prefixes": list(prefixes), "modules": len(selected),
            "tensors": sum(len(module["tensor_names"]) for module in selected),
            "elements": sum(module["elements"] for module in selected),
            "bytes": sum(module["bytes"] for module in selected),
            "operators": dict(sorted(counter.items())),
        })
    total_bytes = sum(tensor.bytes for tensor in records)
    return {
        "schema_version": 1,
        "checkpoint": {
            "file": checkpoint.name,
            "sha256": checkpoint_sha256(checkpoint),
            "file_bytes": checkpoint.stat().st_size,
            "stored_tensor_bytes": total_bytes,
        },
        "sources": {
            "checkpoint": "tensor names, weight shapes, dtypes and stored element counts",
            "openpcdet_config": CONFIG_URL,
            "openpcdet_commit": OPENPCDET_COMMIT,
            "openpcdet_reference_note": "Owner-supplied architecture reference; the checkpoint stores no training commit identity",
            "local_config": str(config.as_posix()),
            "local_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "runtime_contract": "include/*.h, src/runtime.c and full checkpoint oracles",
        },
        "summary": {
            "tensors": len(records), "modules": len(modules),
            "elements": sum(tensor.elements for tensor in records),
            "bytes": total_bytes,
        },
        "flow": [dict(zip(("id", "operator", "input", "output", "note"), row)) for row in FLOW],
        "stages": stages,
        "modules": modules,
    }


def shape_text(shapes: list[list[int]]) -> str:
    return " · ".join("[" + ", ".join(map(str, shape)) + "]" if shape else "[]" for shape in shapes)


def format_count(value: int) -> str:
    return f"{value:,}"


def render_markdown(model: dict[str, Any]) -> str:
    info, summary = model["checkpoint"], model["summary"]
    lines = [
        "# BEVFusion checkpoint model summary", "",
        "> Generated by `tools/visualize_checkpoint.py`; do not edit by hand.", "",
        "This is a state-dictionary summary, not a traced PyTorch graph. Weight tensor facts come",
        "from the checkpoint; operator semantics come from the pinned OpenPCDet config/source;",
        "activation shapes come from the tested BEVFusion.c runtime contract.", "",
        f"- Checkpoint: `{info['file']}`", f"- SHA-256: `{info['sha256']}`",
        f"- OpenPCDet source: [`{model['sources']['openpcdet_commit'][:12]}`]({CONFIG_URL})",
        f"- Stored tensors: **{format_count(summary['tensors'])}**",
        f"- Grouped checkpoint entries: **{format_count(summary['modules'])}**",
        f"- Stored elements: **{format_count(summary['elements'])}**",
        f"- Tensor payload: **{summary['bytes'] / 2**20:.2f} MiB**", "",
        "Regenerate:", "", "```sh",
        "python3 tools/visualize_checkpoint.py /path/to/cbgs_bevfusion.pth", "```", "",
        "## Activation tensor pipe", "",
        "| # | Operator | Input | Output | Contract note |", "|---:|---|---|---|---|",
    ]
    for index, row in enumerate(model["flow"], 1):
        lines.append(f"| {index} | {row['operator']} | `{row['input']}` | `{row['output']}` | {row['note']} |")
    lines += ["", "## Stage summary", "", "| Stage | Modules | Tensors | Elements | Payload | Operator mix |",
              "|---|---:|---:|---:|---:|---|"]
    for stage in model["stages"]:
        mix = ", ".join(f"{name} × {count}" for name, count in stage["operators"].items())
        lines.append(f"| {stage['title']} | {stage['modules']} | {stage['tensors']} | {format_count(stage['elements'])} | {stage['bytes'] / 2**20:.2f} MiB | {mix} |")
    lines += ["", "## Operator and weight tensor inventory", "",
              "Like `torchsummary`, this table is organized by operator module. Shapes are stored",
              "checkpoint tensors—not inferred activation shapes.", "",
              "| # | Stage | Module | Operator | Stored tensor shape(s) | Elements |",
              "|---:|---|---|---|---|---:|"]
    stage_titles = {stage["id"]: stage["title"] for stage in model["stages"]}
    for index, module in enumerate(model["modules"], 1):
        lines.append(f"| {index} | {stage_titles.get(module['stage'], 'Other')} | `{module['name']}` | {module['operator']} | `{shape_text(module['tensor_shapes'])}` | {format_count(module['elements'])} |")
    lines += ["", "## Interpretation limits", "",
              "- A PyTorch `state_dict` does not contain the forward graph or activation shapes.",
              "- BatchNorm running statistics are stored tensors; `num_batches_tracked` and `global_step` are omitted.",
              "- Dropout and stochastic depth are inference identities and therefore have no checkpoint tensor row.",
              "- The checkpoint contains no OpenPCDet commit metadata; the pinned upstream revision is the owner-supplied architecture reference.",
              "- Runtime-only operators such as voxel grouping, lift-splat pooling, concatenation, top-k, decode, and rotated filtering appear in the activation pipe but own no learned weights.", ""]
    return "\n".join(lines)


def write_outputs(model: dict[str, Any], markdown: Path, json_path: Path) -> None:
    markdown.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(model), encoding="utf-8")
    json_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_checkpoint() -> Path:
    root = Path(os.environ.get("NUSCENES_ROOT", "/data/nuscenes"))
    return root / "checkpoints" / "cbgs_bevfusion.pth"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=default_checkpoint())
    parser.add_argument("--config", type=Path, default=root / "cfgs/bevfusion.yaml")
    parser.add_argument("--markdown", type=Path, default=root / "docs/model-summary.md")
    parser.add_argument("--json", dest="json_path", type=Path, default=root / "docs/model-graph.json")
    args = parser.parse_args()
    try:
        records = load_checkpoint(args.checkpoint)
        model = build_model(records, args.checkpoint, args.config)
        write_outputs(model, args.markdown, args.json_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"visualize_checkpoint: {exc}\n")
    print(f"wrote {args.markdown} and {args.json_path}: "
          f"{model['summary']['modules']} modules, {model['summary']['tensors']} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
