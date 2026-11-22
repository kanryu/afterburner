"""Detailed inspection of the Kohya Anima preprocessing dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def stats(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().cpu()
    result: dict[str, object] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
    }
    if value.numel() and value.is_floating_point():
        flat = value.float().reshape(-1)
        result.update(
            minimum=float(flat.min()),
            maximum=float(flat.max()),
            mean=float(flat.mean()),
            std=float(flat.std(unbiased=False)),
            l2=float(torch.linalg.vector_norm(flat)),
            nonzero=int(torch.count_nonzero(flat)),
            finite=bool(torch.isfinite(flat).all()),
        )
    return result


def load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def inspect_sample(directory: Path) -> dict[str, object]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    latent = load(directory / "latent.pt").float()
    text = load(directory / "text_embedding.pt").float()
    qwen_mask = load(directory / "qwen_attention_mask.pt").float()
    t5_ids = load(directory / "t5_input_ids.pt").long()
    t5_mask = load(directory / "t5_attention_mask.pt").float()

    # Python stores Anima latents as [C,H,W].  Report per-channel values in
    # that native form, without adding Afterburner's batch/time axes.
    channel_stats = []
    for channel in latent.reshape(latent.shape[0], -1):
        channel_stats.append(
            {
                "mean": float(channel.mean()),
                "std": float(channel.std(unbiased=False)),
                "min": float(channel.min()),
                "max": float(channel.max()),
                "l2": float(torch.linalg.vector_norm(channel)),
            }
        )

    qwen_valid = int(torch.count_nonzero(qwen_mask > 0.5))
    t5_valid = int(torch.count_nonzero(t5_mask > 0.5))
    text_zero = text.abs() == 0
    qwen_padding_zero = None
    if qwen_mask.ndim >= 1 and text.ndim >= 2:
        mask = qwen_mask.reshape(-1).bool()
        padded = text[~mask, :] if text.ndim == 2 else text[:, ~mask, :]
        qwen_padding_zero = bool((padded.abs() == 0).all()) if (~mask).any() else True

    return {
        "manifest": manifest,
        "latent": stats(latent),
        "latent_per_channel": channel_stats,
        "text_embedding": stats(text),
        "text_zero_elements": int(torch.count_nonzero(text_zero)),
        "qwen_mask": stats(qwen_mask),
        "qwen_valid_tokens": qwen_valid,
        "qwen_padding_zero": qwen_padding_zero,
        "t5_mask": stats(t5_mask),
        "t5_valid_tokens": t5_valid,
        "t5_token_count": int(t5_ids.numel()),
        "t5_first_16_ids": t5_ids.reshape(-1)[:16].tolist(),
        "t5_last_16_ids": t5_ids.reshape(-1)[-16:].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = [inspect_sample(path) for path in sorted(args.dump.glob("sample-*")) if (path / "manifest.json").is_file()]
    report = {"sample_count": len(samples), "samples": samples}
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
