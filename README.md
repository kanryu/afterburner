# Afterburner

Afterburner is a Rust-based AI execution engine and command-line toolkit for
training and exporting LoRA adapters for image-generation models. It uses Burn
for diffusion-transformer training and Candle for text-encoder and VAE
preprocessing.

## GPU build

On Windows, use the MSVC Rust toolchain from a Visual Studio Developer Command
Prompt. The CUDA feature enables both Burn CUDA training and Candle CUDA
preprocessing:

```powershell
cargo check -p afterburner-cli --features cuda --offline
```

The CLI selects the preprocessing device with `--device` and the GPU ordinal
with `--gpu-index`:

```powershell
cargo run -p afterburner-cli --features cuda --offline -- `
  --device cuda --gpu-index 0 `
  --inspect-vae-stem C:\\models\\qwen_image_vae.safetensors
```

Without `--features cuda`, training uses the CPU NdArray backend. If Candle
cannot initialize the requested CUDA device, preprocessing reports the error
and falls back to CPU.

## Training pipeline

The intended workflow is:

1. Encode captions and images into an Afterburner cache.
2. Train the Anima DiT from the cache with Burn.
3. Export LoRA factors as SafeTensors.

Training can be configured with a JSON file or command-line options. The
`--flow-matching`, `--weighting-scheme`, and `--max-grad-norm` options expose
the main Anima training controls.

The JSON configuration accepts the following training fields (all are
optional):

```json
{
  "train_cache": "dataset.afbcache",
  "base_model": "anima-aesthetic-v1.1.safetensors",
  "output_lora": "afterburner-lora.safetensors",
  "steps": 1000,
  "learning_rate": 0.0001,
  "seed": 42,
  "lora_rank": 16,
  "lora_alpha": 16,
  "flow_matching": true,
  "device": "cuda",
  "gpu_index": 0,
  "weighting_scheme": "cosmap",
  "max_grad_norm": 1.0,
  "checkpoint_every": 250,
  "checkpoint_prefix": "checkpoints/anima"
}
```

Use `--config path/to/training.json` to load it. Explicit command-line values
take precedence over values from the JSON file. The supported weighting
schemes are `none`, `sigma_sqrt`, and `cosmap`; `checkpoint_prefix` is used
with `checkpoint_every` to periodically write LoRA-only SafeTensors files.
