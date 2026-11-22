"""Export Kohya probe noise/sigma tensors for exact Afterburner replay.

Run with the ComfyUI Python interpreter:
  python export_probe_replay.py --probe-dir ...\\afterburner-probe --output-dir replay
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(args.probe_dir.glob("step-*/noise.pt"))
    if not steps:
        raise SystemExit(f"no step-*/noise.pt files found under {args.probe_dir}")
    for noise_path in steps:
        step_dir = noise_path.parent
        step_name = step_dir.name
        noise = torch.load(noise_path, map_location="cpu", weights_only=True).detach().float().contiguous()
        sigma = torch.load(step_dir / "sigma.pt", map_location="cpu", weights_only=True).detach().float().reshape(-1)
        if sigma.numel() != 1:
            raise SystemExit(f"{step_dir}: sigma must contain one value")
        noise_path_out = args.output_dir / f"{step_name}.noise.f32"
        sigma_path_out = args.output_dir / f"{step_name}.sigma.f32"
        shape_path_out = args.output_dir / f"{step_name}.shape.json"
        noise.numpy().tofile(noise_path_out)
        sigma.numpy().tofile(sigma_path_out)
        shape_path_out.write_text(json.dumps(list(noise.shape)), encoding="utf-8")
        print(f"{step_name}: shape={list(noise.shape)} sigma={float(sigma[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
