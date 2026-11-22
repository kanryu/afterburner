"""Compare effective LoRA updates without requiring numpy or safetensors.

Usage:
    python scripts/compare_lora_delta.py python.safetensors afterburner.safetensors

The comparison is performed on Delta-W = (alpha / rank) * up @ down.  The
factor matrices themselves are not unique, so their element-wise equality is
not a valid functional equivalence test.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path


def load(path: str):
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        tensors = {}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            stream.seek(data_start + info["data_offsets"][0])
            raw = stream.read(info["data_offsets"][1] - info["data_offsets"][0])
            dtype = info["dtype"]
            if dtype == "BF16":
                values = [
                    struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw[i : i + 2])[0] << 16))[0]
                    for i in range(0, len(raw), 2)
                ]
            elif dtype == "F16":
                values = [struct.unpack("<e", raw[i : i + 2])[0] for i in range(0, len(raw), 2)]
            elif dtype == "F32":
                values = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
            else:
                raise ValueError(f"{path}: unsupported dtype {dtype}")
            tensors[key] = (info["shape"], values)
        return header.get("__metadata__", {}), tensors


def effective_stats(up, down, alpha: float):
    out_dim, rank = up[0]
    down_rank, in_dim = down[0]
    if rank != down_rank:
        raise ValueError(f"rank mismatch: {rank} vs {down_rank}")
    scale = alpha / rank
    # A full dense Delta-W would require billions of Python-level multiply
    # operations for Anima's 8192x2048 MLP matrices. Use a deterministic
    # 64x64 sample of the effective matrix; this is sufficient to detect the
    # order-of-magnitude and distribution mismatch without numpy.
    row_count = min(out_dim, 64)
    col_count = min(in_dim, 64)
    row_indices = [(index * out_dim) // row_count for index in range(row_count)]
    col_indices = [(index * in_dim) // col_count for index in range(col_count)]
    values = []
    for row in row_indices:
        for col in col_indices:
            value = sum(up[1][row * rank + k] * down[1][k * in_dim + col] for k in range(rank))
            values.append(value * scale)
    return out_dim, in_dim, sum(abs(value) for value in values) / len(values), math.sqrt(
        sum(value * value for value in values) / len(values)
    ), max(abs(value) for value in values)


def target_keys(tensors):
    return {key[: -len(".lora_up.weight")] for key in tensors if key.endswith(".lora_up.weight")}


def inspect(path: str):
    metadata, tensors = load(path)
    alpha = float(metadata.get("ss_network_alpha", "0"))
    result = {}
    for target in target_keys(tensors):
        up_key = target + ".lora_up.weight"
        down_key = target + ".lora_down.weight"
        alpha_key = target + ".alpha"
        local_alpha = alpha
        if alpha_key in tensors and tensors[alpha_key][1]:
            local_alpha = tensors[alpha_key][1][0]
        result[target] = effective_stats(tensors[up_key], tensors[down_key], local_alpha)
    return metadata, result


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_lora_delta.py PYTHON.safetensors AFTERBURNER.safetensors")
    python_path, afterburner_path = sys.argv[1:]
    python_meta, python = inspect(python_path)
    afterburner_meta, afterburner = inspect(afterburner_path)
    common = sorted(set(python) & set(afterburner))
    print(f"python={Path(python_path).name}")
    print(f"afterburner={Path(afterburner_path).name}")
    print(f"targets python={len(python)} afterburner={len(afterburner)} common={len(common)}")
    print(f"alpha python={python_meta.get('ss_network_alpha')} afterburner={afterburner_meta.get('ss_network_alpha')}")
    python_norm = math.sqrt(sum(python[key][3] ** 2 for key in common) / len(common))
    afterburner_norm = math.sqrt(sum(afterburner[key][3] ** 2 for key in common) / len(common))
    print(f"delta_sample_rms python={python_norm:.9e} afterburner={afterburner_norm:.9e}")
    ratios = []
    for key in common:
        left = python[key][3]
        right = afterburner[key][3]
        ratios.append(right / left if left else float("inf"))
    finite = [value for value in ratios if math.isfinite(value)]
    print(f"delta_norm_ratio mean={sum(finite) / len(finite):.9e} min={min(finite):.9e} max={max(finite):.9e}")
    print("largest_delta_targets_python")
    for key in sorted(common, key=lambda item: python[item][2], reverse=True)[:10]:
        print(f"  {key}: python_mean_abs={python[key][2]:.9e} afterburner_mean_abs={afterburner[key][2]:.9e} python_rms={python[key][3]:.9e} afterburner_rms={afterburner[key][3]:.9e}")


if __name__ == "__main__":
    main()
