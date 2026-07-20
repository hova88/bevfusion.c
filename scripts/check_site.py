#!/usr/bin/env python3
"""Validate the dependency-free GitHub Pages tree and its local links."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.values.append(value)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "docs"
    index = root / "index.html"
    parser = References()
    parser.feed(index.read_text(encoding="utf-8"))
    missing: list[str] = []
    checked = 0
    for value in parser.values:
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path or value.startswith("#"):
            continue
        target = (index.parent / parsed.path).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            missing.append(f"escapes docs/: {value}")
            continue
        checked += 1
        if not target.exists():
            missing.append(value)
    if missing:
        raise SystemExit("broken local site references:\n  " + "\n  ".join(missing))
    graph = json.loads((root / "model-graph.json").read_text(encoding="utf-8"))
    required = {"schema_version", "checkpoint", "sources", "summary", "flow", "stages", "modules"}
    if graph.keys() < required or graph["schema_version"] != 1:
        raise SystemExit("docs/model-graph.json: invalid schema")
    if graph["summary"]["modules"] != len(graph["modules"]):
        raise SystemExit("docs/model-graph.json: module total does not match entries")
    if graph["summary"]["tensors"] != sum(len(module["tensor_names"]) for module in graph["modules"]):
        raise SystemExit("docs/model-graph.json: tensor total does not match entries")
    if len(graph["flow"]) < 10 or len(graph["stages"]) != 7:
        raise SystemExit("docs/model-graph.json: incomplete runtime flow or stage list")
    print(f"docs/index.html: {checked} local references resolve")
    print(f"docs/model-graph.json: {len(graph['flow'])} flow steps, {len(graph['modules'])} entries")


if __name__ == "__main__":
    main()
