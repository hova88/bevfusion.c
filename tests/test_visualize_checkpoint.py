#!/usr/bin/env python3
"""Dependency-free tests for checkpoint summary classification/rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import visualize_checkpoint as visualizer  # noqa: E402


class VisualizerTest(unittest.TestCase):
    def record(self, name: str, shape: tuple[int, ...]) -> visualizer.TensorRecord:
        elements = visualizer.product(shape)
        return visualizer.TensorRecord(name, shape, "float32", elements, elements * 4)

    def test_groups_and_classifies_checkpoint_tensors(self) -> None:
        records = [
            self.record("backbone_3d.conv.weight", (3, 3, 3, 5, 16)),
            self.record("neck.block.0.weight", (256, 192, 1, 1)),
            self.record("neck.block.1.weight", (256,)),
            self.record("neck.block.1.bias", (256,)),
            self.record("neck.block.1.running_mean", (256,)),
            self.record("neck.block.1.running_var", (256,)),
            self.record("dense_head.decoder.multihead_attn.in_proj_weight", (384, 128)),
            self.record("dense_head.decoder.multihead_attn.out_proj.weight", (128, 128)),
        ]
        grouped: dict[str, list[visualizer.TensorRecord]] = {}
        for record in records:
            grouped.setdefault(visualizer.module_name(record.name), []).append(record)
        self.assertEqual(visualizer.infer_operator("backbone_3d.conv", grouped["backbone_3d.conv"]), "SparseConv3d")
        self.assertEqual(visualizer.infer_operator("neck.block.0", grouped["neck.block.0"]), "Conv2d")
        self.assertEqual(visualizer.infer_operator("neck.block.1", grouped["neck.block.1"]), "BatchNorm")
        self.assertEqual(visualizer.infer_operator("dense_head.decoder.multihead_attn", grouped["dense_head.decoder.multihead_attn"]), "MultiheadAttention")
        self.assertEqual(visualizer.infer_operator("dense_head.decoder.multihead_attn.out_proj", grouped["dense_head.decoder.multihead_attn.out_proj"]), "Linear")
        self.assertEqual(visualizer.stage_for("dense_head.decoder.multihead_attn"), "head")

    def test_writes_deterministic_machine_and_human_views(self) -> None:
        records = [self.record("fuser.conv.0.weight", (256, 336, 3, 3))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "cbgs_bevfusion.pth"
            config = root / "bevfusion.yaml"
            checkpoint.write_bytes(b"fixture")
            config.write_text("\n".join(visualizer.CONFIG_MARKERS), encoding="utf-8")
            model = visualizer.build_model(records, checkpoint, config)
            markdown, json_path = root / "summary.md", root / "graph.json"
            visualizer.write_outputs(model, markdown, json_path)
            decoded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(decoded["summary"]["modules"], 1)
            self.assertEqual(decoded["modules"][0]["operator"], "Conv2d")
            self.assertIn("Activation tensor pipe", markdown.read_text(encoding="utf-8"))
            self.assertIn("[256, 336, 3, 3]", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
