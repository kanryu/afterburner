"""Compare one AFBCACHE sample with a Kohya probe dump.

The probe tensors are saved as CPU float32.  AFBCACHE stores the same
interchange representation as little-endian float32, so this reports both
bit-exact equality and numerical error instead of comparing only shapes.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def tensor(data: bytes, offset: int) -> tuple[np.ndarray, tuple[int, ...], int]:
    rank, offset = u64(data, offset)
    shape = []
    for _ in range(rank):
        value, offset = u64(data, offset)
        shape.append(value)
    count, offset = u64(data, offset)
    end = offset + count * 4
    values = np.frombuffer(data, dtype="<f4", count=count, offset=offset).copy()
    return values.reshape(tuple(shape)), tuple(shape), end


def bytes_field(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = u64(data, offset)
    end = offset + length
    return data[offset:end], end


def read_sample(path: Path, wanted_index: int = 0) -> tuple[str, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if data[:8] != b"AFBCACHE":
        raise ValueError("invalid AFBCACHE magic")
    version = struct.unpack_from("<I", data, 8)[0]
    if version not in (2, 3):
        raise ValueError(f"unsupported AFBCACHE version: {version}")
    offset = 12
    count, offset = u64(data, offset)
    if count == 0 or wanted_index >= count:
        raise ValueError("AFBCACHE contains no samples")
    for index in range(count):
        sample_id, offset = bytes_field(data, offset)
        offset += 8  # repeat index
        text, _, offset = tensor(data, offset)
        has_text_mask = data[offset]
        offset += 1
        if has_text_mask:
            _, _, offset = tensor(data, offset)
        if version >= 3:
            has_adapter = data[offset]
            offset += 1
            if has_adapter:
                token_count, offset = u64(data, offset)
                offset += token_count * 4
                _, _, offset = tensor(data, offset)
        latent, _, offset = tensor(data, offset)
        if index == wanted_index:
            return sample_id.decode("utf-8"), text, latent
    raise ValueError("sample index not found")


def compare(name: str, actual: np.ndarray, expected: torch.Tensor) -> None:
    reference = expected.detach().cpu().float().numpy()
    if name == "latent" and reference.ndim == 4:
        reference = reference[:, :, None, :, :]
    if name == "latent" and reference.ndim == 3:
        reference = reference[None, :, None, :, :]
    if name == "text" and reference.ndim == 2:
        reference = reference[None, :, :]
    if actual.shape != reference.shape:
        print(f"{name}: shape_mismatch cache={actual.shape} probe={reference.shape}")
        return
    delta = np.abs(actual - reference)
    exact = np.count_nonzero(actual.view(np.uint32) == reference.view(np.uint32))
    print(
        f"{name}: shape={actual.shape} exact={exact}/{actual.size} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} l2={np.linalg.norm(delta):.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    sample_id, text, latent = read_sample(args.cache, args.sample_index)
    print(f"cache_sample_id={sample_id}")
    compare("text", text, torch.load(args.probe_dir / "text_embedding.pt", map_location="cpu", weights_only=True))
    compare("latent", latent, torch.load(args.probe_dir / "latent.pt", map_location="cpu", weights_only=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
