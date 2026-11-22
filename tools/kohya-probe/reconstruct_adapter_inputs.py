"""Reconstruct Anima's cached Adapter token inputs from an existing caption.

This post-processes an existing Kohya probe; it does not run model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", type=Path, required=True)
    parser.add_argument("--qwen3", type=Path, required=True)
    parser.add_argument("--t5-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
    sys.path.insert(0, r"C:\local\gnu\ComfyUI\kohya_ss\sd-scripts")
    from library.strategy_anima import AnimaTokenizeStrategy

    caption = args.caption.read_text(encoding="utf-8").strip()
    strategy = AnimaTokenizeStrategy(
        qwen3_path=str(args.qwen3),
        t5_tokenizer_path=str(args.t5_tokenizer),
        qwen3_max_length=512,
        t5_max_length=512,
    )
    qwen_ids, qwen_mask, t5_ids, t5_mask = strategy.tokenize(caption)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(qwen_ids, args.output / "qwen_input_ids.pt")
    torch.save(qwen_mask, args.output / "source_attention_mask.pt")
    torch.save(t5_ids, args.output / "t5_input_ids.pt")
    torch.save(t5_mask, args.output / "t5_attention_mask.pt")
    report = {
        "caption_chars": len(caption),
        "qwen_input_ids_shape": list(qwen_ids.shape),
        "qwen_attention_mask_shape": list(qwen_mask.shape),
        "qwen_valid_tokens": int(qwen_mask.sum()),
        "t5_input_ids_shape": list(t5_ids.shape),
        "t5_attention_mask_shape": list(t5_mask.shape),
        "t5_valid_tokens": int(t5_mask.sum()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
