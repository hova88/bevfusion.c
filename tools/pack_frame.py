#!/usr/bin/env python3
"""Pack preprocessed BEVFusion inputs from an NPZ into canonical BFI v1."""

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

HEADER_BYTES = 128
FIELDS = (
    ("camera_images", (6, 3, 256, 704)),
    ("points", (None, 5)),
    ("camera_intrinsics", (6, 4, 4)),
    ("camera_to_lidar", (6, 4, 4)),
    ("image_augmentation", (6, 4, 4)),
    ("lidar_augmentation", (4, 4)),
    ("lidar_to_image", (6, 4, 4)),
)


def canonical(name: str, value: np.ndarray, shape: tuple) -> np.ndarray:
    array = np.asarray(value)
    if len(shape) != array.ndim or any(want is not None and got != want
                                      for got, want in zip(array.shape, shape)):
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    array = np.ascontiguousarray(array, dtype="<f4")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite input")
    return array


def write_bfi(arrays: list[np.ndarray], output: Path) -> None:
    offsets, cursor, payload_parts = [], HEADER_BYTES, []
    for array in arrays:
        raw = array.tobytes(order="C")
        offsets.append(cursor)
        cursor += len(raw)
        payload_parts.append(raw)
    payload = b"".join(payload_parts)
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<4sII", header, 0, b"BFI1", 1, HEADER_BYTES)
    struct.pack_into("<QQII", header, 16, cursor, arrays[1].shape[0],
                     zlib.crc32(payload), 0)
    struct.pack_into("<7Q", header, 40, *offsets)
    output.write_bytes(header + payload)
    print(f"wrote {output}: {cursor / (1024 * 1024):.2f} MiB, "
          f"{arrays[1].shape[0]} points, CRC32 {zlib.crc32(payload):08x}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("output_bfi", type=Path)
    args = parser.parse_args()
    with np.load(args.input_npz, allow_pickle=False) as source:
        arrays = [canonical(name, source[name], shape) for name, shape in FIELDS]
    if arrays[1].shape[0] == 0:
        raise ValueError("points: at least one point is required")
    write_bfi(arrays, args.output_bfi)


if __name__ == "__main__":
    main()
