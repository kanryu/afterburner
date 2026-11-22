"""Dump Qwen Image VAE encoder stages for Rust/Python parity checks."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kohya-root", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resized-width", type=int, default=832)
    parser.add_argument("--resized-height", type=int, default=1109)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-top", type=int, default=10)
    parser.add_argument("--crop-width", type=int, default=832)
    parser.add_argument("--crop-height", type=int, default=1088)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.kohya_root) / "sd-scripts"))
    from library import qwen_image_autoencoder_kl

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image = np.array(Image.open(args.image).convert("RGB"))
    resized = cv2.resize(
        image,
        (args.resized_width, args.resized_height),
        interpolation=cv2.INTER_AREA,
    )
    cropped = resized[
        args.crop_top : args.crop_top + args.crop_height,
        args.crop_left : args.crop_left + args.crop_width,
    ]
    Image.fromarray(cropped).save(output / "python-cropped.png")
    pixels = torch.from_numpy(cropped.copy()).permute(2, 0, 1).unsqueeze(0)
    pixels = pixels.to(dtype=torch.float16) / 127.5 - 1.0
    # Keep a raw, channel-major copy of the exact tensor entering the VAE.
    # This avoids conflating image-codec differences with VAE differences.
    pixels.float().numpy().tofile(output / "python-input-nchw-f32le.bin")
    (output / "python-input.json").write_text(json.dumps({
        "shape": list(pixels.shape),
        "dtype": str(pixels.dtype),
        "resized_shape_hwc": list(resized.shape),
        "crop": [args.crop_left, args.crop_top, args.crop_width, args.crop_height],
        "min": float(pixels.float().min()),
        "max": float(pixels.float().max()),
        "mean": float(pixels.float().mean()),
        "l2": float(torch.linalg.vector_norm(pixels.float().reshape(-1))),
    }, indent=2), encoding="utf-8")

    device = torch.device(args.device)
    vae = qwen_image_autoencoder_kl.load_vae(args.vae, device=device, disable_mmap=True)
    vae.to(device=device, dtype=torch.float16).eval()
    stages = set()
    stage_calls = {}

    def hook(name):
        def capture(_module, _inputs, value):
            if isinstance(value, tuple):
                value = value[0]
            if not isinstance(value, torch.Tensor):
                return
            call_index = stage_calls.get(name, 0) + 1
            stage_calls[name] = call_index
            file_stem = f"{name}__call_{call_index:02d}"
            print(f"[dump] stage={file_stem} shape={list(value.shape)} dtype={value.dtype}", flush=True)
            value = value.detach().cpu()
            stages.add(file_stem)
            path = output / f"{file_stem}.pt"
            torch.save(value, path)
            flat = value.float().reshape(-1)
            (output / f"{file_stem}.json").write_text(json.dumps({
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(flat.min()),
                "max": float(flat.max()),
                "mean": float(flat.mean()),
                "l2": float(torch.linalg.vector_norm(flat)),
            }, indent=2), encoding="utf-8")
            print(f"[dump] saved={file_stem}", flush=True)
        return capture

    # Dump the complete encoder hierarchy.  The Rust implementation exposes
    # residual/downsample boundaries explicitly, while the Python model keeps
    # many of them nested in ModuleList/Sequential containers.  Comparing only
    # named_children() therefore hides the first layer where the two paths
    # diverge.
    for name, module in vae.encoder.named_modules():
        if not name:
            continue
        safe_name = name.replace(".", "__")
        module.register_forward_hook(hook(safe_name))
    # The encoder hook ends at encoder.conv_out.  The training latent is
    # produced by the top-level quant_conv and then takes the first z_dim
    # channels as the posterior mean, so capture those boundaries explicitly.
    vae.quant_conv.register_forward_hook(hook("quant_conv"))
    print(f"[dump] encode_begin device={device} stages={len(list(vae.encoder.named_modules())) - 1}", flush=True)
    try:
        with torch.no_grad():
            encoded = vae.encode(pixels.to(device).unsqueeze(2), return_dict=False)[0]
            posterior_mode = encoded.mode()
            _dump = posterior_mode.detach().cpu()
            _dump.float().reshape(-1).numpy().tofile(output / "posterior-mode-f32le.bin")
            (output / "posterior-mode.json").write_text(json.dumps({
                "shape": list(_dump.shape),
                "dtype": str(_dump.dtype),
                "min": float(_dump.float().min()),
                "max": float(_dump.float().max()),
                "mean": float(_dump.float().mean()),
                "l2": float(torch.linalg.vector_norm(_dump.float().reshape(-1))),
            }, indent=2), encoding="utf-8")
            latents_mean = torch.tensor(vae.latents_mean, device=device, dtype=posterior_mode.dtype).view(1, vae.z_dim, 1, 1, 1)
            latents_std = (1.0 / torch.tensor(vae.latents_std, device=device, dtype=posterior_mode.dtype)).view(1, vae.z_dim, 1, 1, 1)
            normalized = (posterior_mode - latents_mean) * latents_std
            normalized = normalized.detach().cpu()
            normalized.float().reshape(-1).numpy().tofile(output / "normalized-latent-f32le.bin")
            (output / "normalized-latent.json").write_text(json.dumps({
                "shape": list(normalized.shape),
                "dtype": str(normalized.dtype),
                "min": float(normalized.float().min()),
                "max": float(normalized.float().max()),
                "mean": float(normalized.float().mean()),
                "l2": float(torch.linalg.vector_norm(normalized.float().reshape(-1))),
            }, indent=2), encoding="utf-8")
    except Exception as error:
        print(f"[dump] encode_failed={type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    print(f"dumped_python_vae_stages={len(stages)} output={output}")


if __name__ == "__main__":
    main()
