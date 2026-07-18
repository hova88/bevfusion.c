#!/usr/bin/env python3
"""Export the frozen BEVFusion checkpoint into the bounds-checkable BFW ABI."""

import argparse
import binascii
import pathlib
import struct
import sys

import torch

MAGIC = b"BFW0001\0"
VERSION = 1
HEADER = struct.Struct("<8sIIQQQQII8s")
ENTRY = struct.Struct("<96sII8IQQII")
ALIGNMENT = 64
DTYPES = {torch.float32: 1, torch.int64: 2}


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def load_tensors(path: pathlib.Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dictionary")
    result = []
    for name, value in state.items():
        if name == "global_step" or name.endswith(".num_batches_tracked"):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}: expected tensor, got {type(value).__name__}")
        if value.dtype not in DTYPES:
            raise TypeError(f"{name}: unsupported dtype {value.dtype}")
        value = value.detach().cpu().contiguous()
        if value.ndim > 8 or any(dim <= 0 or dim > 0xFFFFFFFF for dim in value.shape):
            raise ValueError(f"{name}: unsupported shape {tuple(value.shape)}")
        encoded = name.encode("utf-8")
        if len(encoded) >= 96:
            raise ValueError(f"tensor name exceeds 95 UTF-8 bytes: {name}")
        result.append((name, encoded, value))
    if not result:
        raise ValueError("checkpoint contains no runtime tensors")
    return result


def write_tensors(tensors, destination: pathlib.Path) -> None:
    directory_offset = HEADER.size
    data_offset = align(directory_offset + len(tensors) * ENTRY.size)
    entries = []
    payloads = []
    cursor = data_offset
    for name, encoded, tensor in tensors:
        raw = tensor.numpy().tobytes(order="C")
        cursor = align(cursor)
        dims = list(tensor.shape) + [0] * (8 - tensor.ndim)
        crc = binascii.crc32(raw) & 0xFFFFFFFF
        entries.append(ENTRY.pack(encoded, DTYPES[tensor.dtype], tensor.ndim, *dims,
                                  cursor, len(raw), crc, 0))
        payloads.append((cursor, raw))
        cursor += len(raw)
    directory = b"".join(entries)
    directory_crc = binascii.crc32(directory) & 0xFFFFFFFF
    header = HEADER.pack(MAGIC, VERSION, HEADER.size, len(tensors), directory_offset,
                         data_offset, cursor, directory_crc, 0, b"\0" * 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(directory)
        stream.write(b"\0" * (data_offset - stream.tell()))
        for offset, raw in payloads:
            stream.write(b"\0" * (offset - stream.tell()))
            stream.write(raw)
        stream.flush()
    temporary.replace(destination)
    mib = cursor / (1024 * 1024)
    print(f"exported {len(tensors)} tensors to {destination} ({mib:.2f} MiB)")


def export(source: pathlib.Path, destination: pathlib.Path) -> None:
    write_tensors(load_tensors(source), destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        export(args.checkpoint, args.output)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"export_checkpoint: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
