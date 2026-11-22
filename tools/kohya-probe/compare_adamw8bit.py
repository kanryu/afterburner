"""Compare the reference AdamW and bitsandbytes AdamW8bit updates.

This is a small numerical probe, not a training run. It applies identical
parameters and gradients for a few steps and reports update error plus the
optimizer-state storage used by each implementation.
"""

from __future__ import annotations

import argparse
import json

import torch
from bitsandbytes.optim import AdamW8bit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--parameter-numel", type=int, default=11_468_800)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA is required for bitsandbytes AdamW8bit")

    device = torch.device(args.device)
    torch.manual_seed(1234)
    initial = torch.randn(args.parameter_numel, device=device, dtype=torch.float32)
    gradients = [torch.randn_like(initial) for _ in range(args.steps)]
    reference_param = torch.nn.Parameter(initial.clone())
    quantized_param = torch.nn.Parameter(initial.clone())
    reference = torch.optim.AdamW(
        [reference_param], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    quantized = AdamW8bit(
        [quantized_param], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    for gradient in gradients:
        reference.zero_grad(set_to_none=True)
        quantized.zero_grad(set_to_none=True)
        reference_param.grad = gradient.clone()
        quantized_param.grad = gradient.clone()
        reference.step()
        quantized.step()

    difference = (reference_param.detach() - quantized_param.detach()).abs()

    def state_bytes(optimizer):
        total = 0
        dtypes = {}
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
                    dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + 1
        return total, dtypes

    reference_bytes, reference_dtypes = state_bytes(reference)
    quantized_bytes, quantized_dtypes = state_bytes(quantized)
    result = {
        "steps": args.steps,
        "parameter_numel": initial.numel(),
        "max_abs_update_difference": float(difference.max().item()),
        "mean_abs_update_difference": float(difference.mean().item()),
        "reference_state_bytes": reference_bytes,
        "reference_state_dtypes": reference_dtypes,
        "adamw8bit_state_bytes": quantized_bytes,
        "adamw8bit_state_dtypes": quantized_dtypes,
    }
    if quantized_bytes >= reference_bytes:
        raise AssertionError("AdamW8bit did not reduce optimizer-state storage")
    if float(difference.mean().item()) >= 1e-5:
        raise AssertionError("AdamW8bit mean update error exceeded 1e-5")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
