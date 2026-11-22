import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import torch


def stats(value):
    if isinstance(value, (tuple, list)):
        value = next(item for item in value if isinstance(item, torch.Tensor))
    value = value.detach().float()
    result = {
        "shape": list(value.shape),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "l2": float(torch.linalg.vector_norm(value)),
    }
    if value.ndim == 4 and value.shape[1] <= 4:
        result["channel_means"] = [float(item) for item in value.mean(dim=(0, 2, 3))]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dump-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sd_scripts = root.parent.parent / "local" / "gnu" / "ComfyUI" / "kohya_ss" / "sd-scripts"
    # The Windows workspace path is not the Python installation root, so use
    # the installed Kohya path supplied by the caller's environment instead.
    kohya_root = Path(r"C:\local\gnu\ComfyUI\kohya_ss\sd-scripts")
    sys.path.insert(0, str(kohya_root))
    from library import anima_train_utils
    # Import the leaf image helpers directly. Importing `library.dataset` also
    # pulls optional dataset-format dependencies such as `imagesize`, which
    # are irrelevant to this VAE-only probe and may not be installed in the
    # lightweight Kohya runtime.
    from library.utils import IMAGE_TRANSFORMS, load_image, resize_image

    device = torch.device("cuda")
    vae_args = SimpleNamespace(
        vae=args.vae,
        qwen_image_vae_2d=False,
        vae_chunk_size=None,
        vae_disable_cache=False,
    )
    vae = anima_train_utils.load_qwen_image_vae(vae_args, device=device)
    vae = vae.to(dtype=torch.float16)

    image = load_image(args.image)
    resized = resize_image(image, image.shape[1], image.shape[0], 832, 1109)
    resized = resized[10:10 + 1088, 0:832]
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(resized).save(Path(args.dump_dir) / "python-cropped.png")
    pixels = IMAGE_TRANSFORMS(resized).unsqueeze(0).to(device=device, dtype=torch.float16)

    names = [
        "encoder.conv_in",
        "encoder.down_blocks.0",
        "encoder.down_blocks.1",
        "encoder.down_blocks.2",
        "encoder.down_blocks.3",
        "encoder.down_blocks.4",
        "encoder.down_blocks.5",
        "encoder.down_blocks.6",
        "encoder.down_blocks.7",
        "encoder.down_blocks.8",
        "encoder.down_blocks.9",
        "encoder.down_blocks.10",
        "encoder.mid_block.resnets.0",
        "encoder.mid_block.attentions.0",
        "encoder.mid_block.resnets.1",
        "encoder.mid_block.resnets.1.norm1",
        "encoder.mid_block.resnets.1.conv1",
        "encoder.mid_block.resnets.1.norm2",
        "encoder.mid_block.resnets.1.conv2",
        "encoder.mid_block",
        "encoder.norm_out",
        "encoder.conv_out",
        "quant_conv",
    ]
    modules = dict(vae.named_modules())
    outputs = {"input": stats(pixels)}
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.asarray(pixels.detach().float().cpu()).astype("<f4").tofile(dump_dir / "input.f32le")
        (dump_dir / "input.shape").write_text(",".join(str(x) for x in pixels.shape), encoding="ascii")
    hooks = []
    for name in names:
        module = modules[name]
        def capture(_m, _i, output, n=name):
            outputs[n] = stats(output)
            if args.dump_dir:
                value = output[0] if isinstance(output, (tuple, list)) else output
                import numpy as np

                value = value.detach().float().cpu()
                np.asarray(value).astype("<f4").tofile(Path(args.dump_dir) / f"{n.replace('.', '_')}.f32le")
                (Path(args.dump_dir) / f"{n.replace('.', '_')}.shape").write_text(
                    ",".join(str(x) for x in value.shape), encoding="ascii"
                )

        hooks.append(module.register_forward_hook(capture))

    with torch.no_grad():
        vae.encode_pixels_to_latents(pixels)
    for hook in hooks:
        hook.remove()

    Path(args.output).write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
