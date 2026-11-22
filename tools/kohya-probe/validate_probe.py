"""Validate one Kohya Anima probe step and an optional checkpoint.

This is intentionally independent of Afterburner's Rust runtime.  It checks
the reference objective and tensor contract before using the dump as a
cross-implementation fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    probe = args.probe_dir
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        "latent",
        "noise",
        "noisy_latent",
        "sigma",
        "timestep",
        "text_embedding",
        "target",
        "prediction",
        "loss_mse",
    ]
    tensors: dict[str, torch.Tensor] = {}
    for name in required:
        path = probe / f"{name}.pt"
        if not path.is_file():
            errors.append(f"missing {path.name}")
            continue
        tensors[name] = load_tensor(path).float()
        if not torch.isfinite(tensors[name]).all():
            errors.append(f"non-finite values in {path.name}")

    if {"latent", "noise", "noisy_latent", "sigma", "target", "prediction"} <= tensors.keys():
        latent = tensors["latent"]
        noise = tensors["noise"]
        noisy = tensors["noisy_latent"].squeeze(2)
        sigma = float(tensors["sigma"].item())
        target = tensors["target"]
        prediction = tensors["prediction"]
        noisy_error = max_abs(noisy, (1.0 - sigma) * latent + sigma * noise)
        target_error = max_abs(target, noise - latent)
        loss = float((prediction - target).square().mean())
        dumped_loss = float(tensors["loss_mse"].item())
        print(f"latent_shape={tuple(latent.shape)}")
        print(f"text_embedding_shape={tuple(tensors['text_embedding'].shape)}")
        print(f"sigma={sigma:.9f}")
        print(f"noisy_latent_formula_max_abs={noisy_error:.9g}")
        print(f"flow_target_formula_max_abs={target_error:.9g}")
        print(f"loss_recomputed={loss:.9g}")
        print(f"loss_dumped={dumped_loss:.9g}")
        # The cached BF16 latent/noise values are dumped as float32 after
        # conversion, so reconstruction can differ by a few BF16 ULPs.
        if noisy_error > 3e-3:
            errors.append(f"noisy latent formula mismatch: {noisy_error}")
        # The probe stores BF16-derived tensors as float32; one BF16 ULP can
        # produce a small absolute residual in the reconstructed target.
        if target_error > 3e-3:
            errors.append(f"flow target mismatch: {target_error}")
        if abs(loss - dumped_loss) > 1e-5:
            errors.append("dumped MSE does not match prediction and target")

    before = {p.name[len("lora_before_") : -3]: p for p in probe.glob("lora_before_*.pt")}
    after = {p.name[len("lora_after_") : -3]: p for p in probe.glob("lora_after_*.pt")}
    gradients = {p.name[len("gradient_") : -3]: p for p in probe.glob("gradient_*.pt")}
    common = set(before) & set(after) & set(gradients)
    print(f"lora_before_tensors={len(before)}")
    print(f"lora_after_tensors={len(after)}")
    print(f"gradient_tensors={len(gradients)}")
    print(f"lora_matching_tensors={len(common)}")
    if len(before) != 560 or len(after) != 560 or len(gradients) != 560:
        errors.append("expected 560 LoRA tensors in the probe")

    changed_probe = 0
    nonzero_gradients = 0
    for key in common:
        b = load_tensor(probe / f"lora_before_{key}.pt").float()
        a = load_tensor(probe / f"lora_after_{key}.pt").float()
        g = load_tensor(probe / f"gradient_{key}.pt").float()
        changed_probe += max_abs(a, b) > 0.0
        nonzero_gradients += float(g.abs().max()) > 0.0
    print(f"probe_changed_tensors={changed_probe}")
    print(f"nonzero_gradient_tensors={nonzero_gradients}")
    if changed_probe == 0:
        warnings.append("probe lora_after tensors did not capture an optimizer update")

    if args.checkpoint:
        from safetensors.torch import load_file

        checkpoint = load_file(str(args.checkpoint), device="cpu")
        changed_checkpoint = 0
        checked_checkpoint = 0
        for key, value in checkpoint.items():
            if not key.endswith((".lora_down.weight", ".lora_up.weight")):
                continue
            probe_key = key.replace(".", "_")
            before_path = probe / f"lora_before_{probe_key}.pt"
            if not before_path.is_file():
                errors.append(f"checkpoint tensor has no probe baseline: {key}")
                continue
            checked_checkpoint += 1
            if max_abs(value, load_tensor(before_path)) > 0.0:
                changed_checkpoint += 1
        print(f"checkpoint_lora_tensors={checked_checkpoint}")
        print(f"checkpoint_changed_vs_probe_before={changed_checkpoint}")
        if checked_checkpoint and changed_checkpoint == 0:
            errors.append("checkpoint contains no LoRA update")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print("VALIDATION=PASS" if not errors else "VALIDATION=FAIL")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
