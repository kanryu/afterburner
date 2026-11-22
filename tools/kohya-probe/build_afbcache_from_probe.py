"""Build a one-sample AFBCACHE fixture from an existing Kohya probe dump."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().float().contiguous()
    values = tensor.reshape(-1).tolist()
    return (
        u64(tensor.ndim)
        + b"".join(u64(int(dim)) for dim in tensor.shape)
        + u64(len(values))
        + struct.pack(f"<{len(values)}f", *values)
    )


def bytes_field(value: bytes) -> bytes:
    return u64(len(value)) + value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load = lambda name: torch.load(args.probe_dir / f"{name}.pt", map_location="cpu", weights_only=True)
    latent = load("latent").float().unsqueeze(2)
    text = load("text_embedding").float()
    t5_ids = torch.load(args.adapter_dir / "t5_input_ids.pt", map_location="cpu", weights_only=True)
    t5_ids = t5_ids.long().reshape(-1).tolist()
    t5_mask = torch.load(args.adapter_dir / "t5_attention_mask.pt", map_location="cpu", weights_only=True)
    t5_mask = t5_mask.float().reshape(1, 1, 1, -1)
    qwen_mask = torch.load(
        args.adapter_dir / "qwen_attention_mask.pt", map_location="cpu", weights_only=True
    ) if (args.adapter_dir / "qwen_attention_mask.pt").is_file() else torch.load(
        args.adapter_dir / "source_attention_mask.pt", map_location="cpu", weights_only=True
    )
    qwen_mask = qwen_mask.float().reshape(1, 1, 1, -1)

    output = bytearray(b"AFBCACHE")
    output += struct.pack("<I", 3)
    output += u64(1)
    output += bytes_field(b"kohya-probe/step")
    output += u64(0)
    output += tensor_bytes(text)
    output += b"\x01" + tensor_bytes(qwen_mask)
    output += b"\x01" + u64(len(t5_ids))
    output += b"".join(struct.pack("<I", int(value)) for value in t5_ids)
    output += tensor_bytes(t5_mask)
    output += tensor_bytes(latent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"cache_path={args.output}")
    print(f"text_shape={list(text.shape)}")
    print(f"latent_shape={list(latent.shape)}")
    print(f"adapter_tokens={len(t5_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
