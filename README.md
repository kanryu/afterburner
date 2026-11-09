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
and falls back to CPU. If `--device cuda` is used without the feature, the
CLI also reports that CUDA is disabled and uses CPU explicitly.

CUDA is therefore opt-in for both the Burn training backend and Candle
preprocessing:

```powershell
# Default CPU build
cargo build -p afterburner-cli

# Optional CUDA build
cargo build -p afterburner-cli --features cuda
```

CUDA training can optionally use the CUDA backend's BF16 element type:

```powershell
cargo check -p afterburner-cli --features cuda,bf16,checkpointing --offline
```

BF16 reduces the storage size of model weights and activations, but it can
change numerical behavior. Validate loss stability and exported LoRA quality
against the default FP32 path before using it for production runs.

For training with long spatial sequences, enable Burn's balanced checkpointing
strategy in addition to CUDA. Memory-bound operations are then eligible for
forward recomputation during backward, reducing activation retention at the
cost of extra compute:

```powershell
cargo run -p afterburner-cli --features cuda,checkpointing -- `
  --device cuda --gpu-index 0 --train-cache dataset.afbcache `
  --base-model anima.safetensors --output-lora afterburner-lora.safetensors `
  --steps 10 --max-resolution 512
```

Attention score matrices are computed in 512-query-token chunks. This limits
each individual score allocation to `512 x context_length` rather than the
full spatial `sequence_length x context_length` matrix. Checkpointing is
recommended with this mode for training; chunking alone limits allocation
peaks but does not guarantee that all backward intermediates are discarded.

## Training pipeline

The intended workflow is:

1. Encode captions and images into an Afterburner cache.
2. Train the Anima DiT from the cache with Burn.
3. Export LoRA factors as SafeTensors.

For preprocessing, `--tokenizer` and `--t5-tokenizer` are optional. When they
are omitted, the CLI searches beside the Qwen text-encoder model, its
`tokenizer` subdirectory, the model directory's parent, and the directory
specified by `AFTERBURNER_TOKENIZER_DIR`. Explicit paths always take priority.

Use `--max-resolution` to bound the longest image edge before VAE encoding.
This is recommended for full-attention GPU training because the DiT attention
memory grows quadratically with the number of spatial tokens. For example,
`--max-resolution 512` converts a 1024-pixel-class dataset into a practical
smoke-test size.

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
