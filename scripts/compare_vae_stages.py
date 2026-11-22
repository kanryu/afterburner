"""Compare Python VAE stage dumps with Afterburner f32le stage dumps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


PAIRS = {
    "input": "input",
    "conv_in": "conv_in",
    # First residual block: compare the internal operations before looking at
    # the accumulated error at the block output.
    "down_blocks__0__norm1": "residual_0_norm1",
    "down_blocks__0__nonlinearity": "residual_0_nonlinearity",
    "down_blocks__0__conv1": "residual_0_conv1",
    "down_blocks__0__norm2": "residual_0_norm2",
    "down_blocks__0__conv2": "residual_0_conv2",
    "down_blocks__0": "residual_0",
    "down_blocks__1__norm1": "residual_1_norm1",
    "down_blocks__1__conv1": "residual_1_conv1",
    "down_blocks__1__norm2": "residual_1_norm2",
    "down_blocks__1__conv2": "residual_1_conv2",
    "down_blocks__1": "residual_1",
    # The Python resampler exposes both the padded tensor and the strided
    # convolution.  These are especially important for odd spatial sizes.
    "down_blocks__2__resample__0": "downsample_0_padded",
    "down_blocks__2__resample__1": "downsample_0_conv",
    "down_blocks__2": "downsample_0",
    "down_blocks__3": "residual_3",
    "down_blocks__4": "residual_4",
    "down_blocks__5__resample__0": "downsample_1_padded",
    "down_blocks__5__resample__1": "downsample_1_conv",
    "down_blocks__5": "downsample_1",
    "down_blocks__6": "residual_6",
    "down_blocks__7": "residual_7",
    "down_blocks__8__resample__0": "downsample_2_padded",
    "down_blocks__8__resample__1": "downsample_2_conv",
    "down_blocks__8": "downsample_2",
    "down_blocks__9": "residual_9",
    "down_blocks__10": "residual_10",
    "mid_block": "middle",
    "conv_out": "output_conv",
}


def load_rust(root: Path, name: str) -> np.ndarray:
    shape = tuple(int(value) for value in (root / f"{name}.shape").read_text().split(","))
    return np.fromfile(root / f"{name}.f32le", dtype="<f4").reshape(shape)


def load_python(root: Path, name: str) -> np.ndarray:
    value = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True)
    return value.detach().float().numpy()


def load_f32le(root: Path, name: str) -> np.ndarray:
    return np.fromfile(root / name, dtype="<f4")


def compare_arrays(label: str, py: np.ndarray, rust: np.ndarray) -> None:
    if py.shape != rust.shape:
        print(f"{label},SHAPE_MISMATCH:{py.shape}!={rust.shape}")
        return
    diff = rust - py
    py_flat = py.reshape(-1)
    rust_flat = rust.reshape(-1)
    corr = np.corrcoef(py_flat, rust_flat)[0, 1]
    print(
        f"{label},{list(py.shape)},"
        f"{np.max(np.abs(diff)):.8g},{np.sqrt(np.mean(diff * diff)):.8g},"
        f"{np.mean(np.abs(diff)):.8g},{corr:.8g},"
        f"{np.mean(py):.8g},{np.mean(rust):.8g},"
        f"{np.linalg.norm(py):.8g},{np.linalg.norm(rust):.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    args = parser.parse_args()

    print("stage,shape,max_abs,rmse,mean_abs,corr,python_mean,rust_mean,python_l2,rust_l2")
    for python_name, rust_name in PAIRS.items():
        py_path = args.python / f"{python_name}.pt"
        rust_path = args.rust / f"{rust_name}.f32le"
        if not py_path.exists() or not rust_path.exists():
            continue
        py = load_python(args.python, python_name)
        rust = load_rust(args.rust, rust_name)
        if py.ndim == 5 and py.shape[2] == 1:
            py = py[:, :, 0]
        if py.shape != rust.shape:
            print(f"{python_name}->{rust_name},SHAPE_MISMATCH:{py.shape}!={rust.shape}")
            continue
        compare_arrays(f"{python_name}->{rust_name}", py, rust)

    # These files are emitted by dump_python_vae_stages.py and the Rust VAE
    # trace. They are the decisive boundaries after encoder.conv_out.
    extra_pairs = (
        ("posterior-mode-f32le.bin", "quant_conv.f32le", "posterior_mode->quant_conv"),
    )
    for python_name, rust_name, label in extra_pairs:
        py_path = args.python / python_name
        rust_path = args.rust / rust_name
        if not py_path.exists() or not rust_path.exists():
            continue
        py = load_f32le(args.python, python_name)
        rust = load_f32le(args.rust, rust_name)
        shape_path = args.rust / "quant_conv.shape"
        if shape_path.exists():
            shape = tuple(int(value) for value in shape_path.read_text().split(","))
            py = py.reshape(shape)
            rust = rust.reshape(shape)
        compare_arrays(label, py, rust)

    normalized = args.python / "normalized-latent-f32le.bin"
    quant = args.rust / "quant_conv.f32le"
    shape_path = args.rust / "quant_conv.shape"
    if normalized.exists() and quant.exists() and shape_path.exists():
        shape = tuple(int(value) for value in shape_path.read_text().split(","))
        py = load_f32le(args.python, "normalized-latent-f32le.bin").reshape(shape)
        raw = load_f32le(args.rust, "quant_conv.f32le").reshape(shape)
        # Match normalize_qwen_image_latent_with_precision(fp16=true). The
        # constants are the fixed Qwen Image VAE values used by Anima.
        mean = np.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653,
                         -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632,
                         -0.1922, -0.9497, 0.2503, -0.2921], dtype=np.float16)
        std = np.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708,
                        2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
                        1.6382, 1.1253, 2.8251, 1.9160], dtype=np.float16)
        raw16 = raw.astype(np.float16)
        centered = (raw16 - mean.reshape(1, 16, 1, 1, 1)).astype(np.float16)
        inv_std = (1.0 / std).astype(np.float16)
        rust = (centered * inv_std.reshape(1, 16, 1, 1, 1)).astype(np.float32)
        compare_arrays("normalized_latent_python->rust", py, rust)


if __name__ == "__main__":
    main()
