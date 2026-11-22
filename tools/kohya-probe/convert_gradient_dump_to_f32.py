"""Convert one kohya probe step's LoRA gradients to raw F32 files.

The raw format is intentionally simple so the Rust optimizer replay path can
load it without depending on PyTorch serialization.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def convert(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    converted = 0
    for metadata_path in sorted(source.glob("gradient_*.json")):
        tensor_path = metadata_path.with_suffix(".pt")
        if not tensor_path.exists():
            continue
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(tensor):
            raise TypeError(f"{tensor_path} does not contain a tensor")
        tensor = tensor.detach().float().contiguous()
        values = tensor.reshape(-1).tolist()
        stem = metadata_path.stem
        (destination / f"{stem}.shape.json").write_text(
            json.dumps(list(tensor.shape)), encoding="utf-8"
        )
        (destination / f"{stem}.f32").write_bytes(
            struct.pack(f"<{len(values)}f", *values)
        )
        converted += 1
    if converted == 0:
        raise SystemExit(f"no gradient dumps found in {source}")
    print(f"converted_gradients={converted} destination={destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
