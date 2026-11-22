"""Convert kohya Anima probe dumps into Afterburner's replay-input format."""

import argparse
import json
import struct
from pathlib import Path

import torch


def convert(source: Path, destination: Path, steps: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    converted = 0
    for step in range(1, steps + 1):
        source_step = source / f"step-{step:06d}"
        noise_path = source_step / "noise.pt"
        sigma_path = source_step / "sigma.json"
        if not sigma_path.exists():
            sigma_path = source_step / "sigmas.json"
        if not noise_path.exists() or not sigma_path.exists():
            break

        noise = torch.load(noise_path, map_location="cpu", weights_only=True).detach().float()
        # Preserve the probe's exact spatial layout and value order.  The
        # current Python Anima trainer emits [B,C,H,W] with H=136,W=104;
        # replay must not silently transpose it to match an older cache
        # convention, otherwise the same latent is spatially scrambled.
        noise = noise.contiguous()
        sigma_stats = json.loads(sigma_path.read_text(encoding="utf-8"))
        sigma = float(sigma_stats["mean"])
        values = noise.reshape(-1).tolist()
        stem = f"step-{step:06d}"
        (destination / f"{stem}.shape.json").write_text(
            json.dumps(list(noise.shape)), encoding="utf-8"
        )
        (destination / f"{stem}.sigma.f32").write_bytes(struct.pack("<f", sigma))
        (destination / f"{stem}.noise.f32").write_bytes(
            struct.pack(f"<{len(values)}f", *values)
        )
        latent_path = source_step / "latent.pt"
        if latent_path.exists():
            latent = torch.load(latent_path, map_location="cpu", weights_only=True).detach().float()
            latent = latent.contiguous()
            latent_values = latent.reshape(-1).tolist()
            (destination / f"{stem}.latent.shape.json").write_text(
                json.dumps(list(latent.shape)), encoding="utf-8"
            )
            (destination / f"{stem}.latent.f32").write_bytes(
                struct.pack(f"<{len(latent_values)}f", *latent_values)
            )
        converted += 1

    if converted == 0:
        raise SystemExit(f"no usable step dumps found in {source}")
    print(f"converted_steps={converted} destination={destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    convert(args.source, args.destination, args.steps)


if __name__ == "__main__":
    main()
