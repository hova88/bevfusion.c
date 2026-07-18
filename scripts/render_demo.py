#!/usr/bin/env python3
"""Render real /data/nuscenes inference through the C TUI into the README GIF."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CSI = re.compile(r"\x1b\[([0-9;?]*)([@-~])")


def xterm(index: int) -> tuple[int, int, int]:
    base = [
        (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
        (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
        (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    ]
    if index < 16:
        return base[index]
    if index < 232:
        value = index - 16
        levels = [0, 95, 135, 175, 215, 255]
        return levels[value // 36], levels[(value // 6) % 6], levels[value % 6]
    gray = 8 + (index - 232) * 10
    return gray, gray, gray


def terminal_cells(data: str, columns: int, rows: int):
    cells = [[(" ", 250, 233, 0) for _ in range(columns)] for _ in range(rows)]
    x = y = 0
    fg, bg, mode = 250, 233, 0
    cursor = 0
    while cursor < len(data):
        if data[cursor] == "\x1b":
            match = CSI.match(data, cursor)
            if not match:
                cursor += 1
                continue
            params, command = match.groups()
            values = [int(value) if value else 0 for value in params.split(";")]
            if command == "H":
                x = y = 0
            elif command == "m":
                i = 0
                while i < len(values):
                    value = values[i]
                    if value == 0:
                        fg, bg, mode = 250, 233, 0
                    elif value in (1, 2):
                        mode = value
                    elif value == 38 and i + 2 < len(values) and values[i + 1] == 5:
                        fg = values[i + 2]
                        i += 2
                    elif value == 48 and i + 2 < len(values) and values[i + 1] == 5:
                        bg = values[i + 2]
                        i += 2
                    i += 1
            cursor = match.end()
            continue
        char = data[cursor]
        cursor += 1
        if char == "\r":
            x = 0
        elif char == "\n":
            y += 1
            x = 0
        elif 0 <= x < columns and 0 <= y < rows:
            cells[y][x] = (char, fg, bg, mode)
            x += 1
    return cells


def blend(foreground, background, amount: float):
    return tuple(round(background[i] + (foreground[i] - background[i]) * amount)
                 for i in range(3))


def render_frame(data: bytes, columns: int, rows: int, font, braille_font):
    text = data.decode("utf-8")
    cells = terminal_cells(text, columns, rows)
    cell_width = round(font.getlength("M"))
    cell_height = 19
    image = Image.new("RGB", (columns * cell_width, rows * cell_height), xterm(233))
    draw = ImageDraw.Draw(image)
    for y, row in enumerate(cells):
        for x, (char, foreground, background, mode) in enumerate(row):
            bg = xterm(background)
            fg = xterm(foreground)
            if mode == 2:
                fg = blend(fg, bg, 0.55)
            elif mode == 1:
                fg = blend(fg, (255, 255, 255), 0.18)
            left, top = x * cell_width, y * cell_height
            draw.rectangle((left, top, left + cell_width, top + cell_height), fill=bg)
            if char != " ":
                selected_font = braille_font if "\u2800" <= char <= "\u28ff" else font
                draw.text((left, top - 1), char, font=selected_font, fill=fg)
    return image


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    binary = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "build/bevfusion_cuda"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "docs/bevfusion-demo.gif"
    model = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        "/data/nuscenes/bevfusion-demo/bevfusion.bfw")
    demo_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(
        "/data/nuscenes/bevfusion-demo")
    if not binary.exists():
        raise SystemExit(f"missing {binary}; run make")
    if not model.exists():
        raise SystemExit(f"missing {model}; run make model")
    paths = sorted(demo_dir.glob("frame-*.bfi"))
    if not paths:
        raise SystemExit(f"no demo frames under {demo_dir}; run make demo-data")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    braille_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    columns, rows = 100, 30
    frames = []
    for index, path in enumerate(paths):
        ansi = subprocess.check_output(
            [str(binary), "render-cuda", str(model), str(index), str(len(paths)),
             str(columns), str(rows), *(str(item) for item in paths[:index + 1])],
            cwd=root,
        )
        frames.append(render_frame(ansi, columns, rows, font, braille_font))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
        optimize=False,
        disposal=1,
    )
    print(f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
