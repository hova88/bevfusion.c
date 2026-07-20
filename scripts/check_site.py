#!/usr/bin/env python3
"""Validate the dependency-free GitHub Pages tree and its local links."""

from __future__ import annotations

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
    print(f"docs/index.html: {checked} local references resolve")


if __name__ == "__main__":
    main()
