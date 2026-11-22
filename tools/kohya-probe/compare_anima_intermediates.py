"""Compare Python probe intermediates with an Afterburner training dump."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def load_rust(directory: Path, name: str) -> torch.Tensor:
    shape = json.loads((directory / f"{name}.shape.json").read_text(encoding="utf-8"))
    raw = (directory / f"{name}.f32").read_bytes()
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def canonical(name: str, value: torch.Tensor) -> torch.Tensor:
    # Python keeps image tokens as [B, T, H, W, D], while Afterburner uses
    # [B, T*H*W, D].  The timestep path similarly keeps a singleton T axis
    # on the Python side.  Normalize only these layout differences; do not
    # silently transpose arbitrary tensors.
    if name == "python_x_embedder" and value.ndim == 5:
        b, t, h, w, d = value.shape
        return value.reshape(b, t * h * w, d)
    if name in {"python_time_condition", "python_timestep_embedding"}:
        if value.ndim == 3 and value.shape[1] == 1:
            return value[:, 0, :]
    return value


def canonical_actual(name: str, reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # The Rust replay/cache layout is [H=136, W=104], whereas the Python
    # probe's original latent is [H=104, W=136].  Consequently Rust token
    # grids are [68, 52] and Python grids are [52, 68].  Restore Python's
    # spatial order before flattening the comparison.
    if name in {
        "python_x_embedder",
        "python_patch_tokens",
        "python_block_00",
        "python_block_01",
        "python_block_27",
        "python_final_layer",
    } and reference.ndim == 5 and value.ndim == 3 and value.shape[1] == 68 * 52:
        b, _, d = value.shape
        _, _, ref_h, ref_w, _ = reference.shape
        grid = value.reshape(b, 68, 52, d)
        if (ref_h, ref_w) == (52, 68):
            grid = grid.transpose(1, 2)
        return grid.reshape(b, ref_h * ref_w, d)
    return value


def stats(reference: torch.Tensor, actual: torch.Tensor, name: str) -> dict[str, object]:
    reference = canonical(name, reference).float()
    actual = canonical_actual(name, reference, actual).float()
    reference_shape = list(reference.shape)
    actual_shape = list(actual.shape)
    reference = reference.reshape(-1)
    actual = actual.reshape(-1)
    if reference.numel() != actual.numel():
        return {
            "reference_shape": reference_shape,
            "actual_shape": actual_shape,
            "reference_numel": reference.numel(),
            "actual_numel": actual.numel(),
        }
    delta = actual - reference
    return {
        "reference_shape": reference_shape,
        "actual_shape": actual_shape,
        "mse": float(delta.square().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "reference_l2": float(reference.square().sum().sqrt()),
        "actual_l2": float(actual.square().sum().sqrt()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_step", type=Path)
    parser.add_argument("rust_step", type=Path)
    args = parser.parse_args()
    mapping = {
        "python_x_embedder": "rust_input_projection",
        "python_patch_tokens": "rust_patch_tokens",
        "python_block00_self_attn": "rust_block00_self_attn",
        "python_block00_cross_attn": "rust_block00_cross_attn",
        "python_block00_mlp": "rust_block00_mlp",
        "python_timestep_embedding": "rust_timestep_embedding",
        "python_time_condition": "rust_time_condition",
        "python_time_condition_norm": "rust_time_condition",
        "python_block_00": "rust_block_00",
        "python_block_01": "rust_block_01",
        "python_block_27": "rust_block_27",
        "python_final_layer": "rust_output_projection",
    }
    result = {}
    for python_name, rust_name in mapping.items():
        python_path = args.python_step / f"{python_name}.pt"
        if not python_path.exists():
            result[python_name] = "missing_python_dump"
            continue
        if not (args.rust_step / f"{rust_name}.shape.json").exists():
            result[python_name] = "missing_rust_dump"
            continue
        result[python_name] = stats(
            torch.load(python_path, map_location="cpu", weights_only=True),
            load_rust(args.rust_step, rust_name),
            python_name,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
