# Afterburner

Afterburner is a Rust-based AI execution engine and command-line toolkit for
training and exporting LoRA adapters for image-generation models. It uses Burn
for diffusion-transformer training and Candle for text-encoder and VAE
preprocessing.

## CubeCL memory tracing

The repository includes `cubecl.toml` with compact CubeCL memory logging
enabled. CUDA runs report new memory-page events to stderr, while Afterburner
step markers identify the training phase. For a raw allocation trace, copy
`cubecl-trace.toml` to `cubecl.toml`, run the workload with
`AFTERBURNER_MEMORY_TRACE=1`, and summarize the result with:

```powershell
.\tools\filter-cubecl-memory-log.ps1 `
  -InputPath .\training.log `
  -OutputPath .\training-memory-summary.txt
```

## GPU build

On Windows, use the MSVC Rust toolchain from a Visual Studio Developer Command
Prompt. The `cuda` feature enables Burn CUDA training. Candle CUDA
preprocessing is separate and requires `candle-cuda`:

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

Without `--features cuda`, training uses the CPU NdArray backend. If
`--device cuda` is used without `candle-cuda`, preprocessing falls back to CPU.
If Candle cannot initialize the requested CUDA device, preprocessing reports the error
and falls back to CPU. If `--device cuda` is used without the feature, the
CLI also reports that CUDA is disabled and uses CPU explicitly.

Burn CUDA training and Candle CUDA preprocessing are independent opt-in
features:

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

For CUDA runs where allocator fragmentation must be minimized, the optional
`fixed-memory` feature selects CubeCL's exclusive-page pools:

```powershell
cargo build -p afterburner-cli --release --features cuda,bf16,fixed-memory --offline
```

This reduces reserved-pool overhead, but it cannot reclaim tensors that remain
live in the training graph. A stable `in_use` value across optimizer updates is
still required.

BF16 reduces the storage size of model weights and activations, but it can
change numerical behavior. Validate loss stability and exported LoRA quality
against the default FP32 path before using it for production runs.

Vulkan training is available through Burn's built-in `vulkan` feature. It uses
the Burn WGPU backend with the Vulkan/SPIR-V path and does not require a
direct `burn-wgpu` dependency in Afterburner:

```powershell
cargo check -p afterburner-cli --features vulkan,checkpointing --offline
cargo run -p afterburner-cli --features vulkan,checkpointing -- `
  --device vulkan --train-cache dataset.afbcache `
  --base-model anima.safetensors --output-lora afterburner-lora.safetensors `
  --epochs 1
```

When `--device vulkan` is selected, Burn training uses Vulkan while Candle
preprocessing remains on CPU because Candle's preprocessing path currently
supports CPU and CUDA devices.

The CLI interprets `--epochs` as the number of dataset epochs, matching the
kohya_ss workflow. One epoch visits every expanded cache sample once, so a
dataset cache with 270 samples and `--epochs 10` performs 2,700 optimizer
updates. `--steps` remains accepted as a backwards-compatible alias. The
effective update count is printed as `training_steps`.
For a short backend smoke test, `--max-optimization-steps 1` caps the actual
optimizer updates without changing the meaning of `--epochs` for normal runs.

The default optimizer policy is `warmup_cosine`: the learning rate warms up
for 10% of the optimization steps, then decays cosinely to 10% of the peak.
Use `--lr-warmup-steps` and `--lr-min` to override those values. For a
deliberately fixed rate, use `--lr-schedule constant`.

LoRA SafeTensors storage can be selected independently from the training
backend with `--lora-dtype f32` or `--lora-dtype bf16`.
The default follows the compiled training backend: `bf16` builds default to
BF16 output and other builds default to F32. Pre-quantized base tensors such
as INT8/INT4 remain in their original representation and are never widened
or rewritten just to match the LoRA dtype.
The CLI default is `lora_rank=8` and `lora_alpha=16`, matching the practical
Anima training baseline. Use `--lora-rank` explicitly for larger adapters.

For training with long spatial sequences, enable Burn's balanced checkpointing
strategy in addition to CUDA. Memory-bound operations are then eligible for
forward recomputation during backward, reducing activation retention at the
cost of extra compute:

This is not equivalent to kohya_ss's explicit per-DiT-block checkpointing.
Burn 0.19 classifies the dominant matrix multiplications as compute-bound, so
the current feature does not discard and recompute an entire Anima block.
Treat memory preflight and attention chunking as safety measures until a
block-level checkpoint operation or backend-native memory-efficient attention
kernel is available.

Afterburner now uses a narrowly scoped local `burn-autodiff` fork. Its float
MatMul operation is marked memory-bound and receives a retro-forward, allowing
Burn's balanced checkpointing strategy to recompute MatMul inputs during
backward. This reduces retained MatMul intermediates without detaching the
gradient path. Full DiT-block checkpointing remains a separate follow-up.

```powershell
cargo run -p afterburner-cli --features cuda,checkpointing -- `
  --device cuda --gpu-index 0 --train-cache dataset.afbcache `
  --base-model anima.safetensors --output-lora afterburner-lora.safetensors `
  --steps 10 --max-resolution 512
```

Attention score matrices are computed in 128-query-token chunks by default. This limits
each individual score allocation to `chunk_size x context_length` rather than the
full spatial `sequence_length x context_length` matrix. Checkpointing is
recommended with this mode for training; chunking alone limits allocation
peaks but does not guarantee that all backward intermediates are discarded.
For a smaller GPU, override the chunk size with `AFTERBURNER_ATTENTION_CHUNK`,
for example `AFTERBURNER_ATTENTION_CHUNK=64`. Smaller chunks reduce peak
temporary memory at the cost of more kernel launches.

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
  "epochs": 10,
  "learning_rate": 0.0001,
  "seed": 42,
  "lora_rank": 8,
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

When preprocessing without `--output-cache`, Afterburner stores the AFBCACHE
under `target/afterburner-model-cache` using a key derived from the diffusion
model, text encoder, and VAE file names. Use `--model-cache-dir` to change the
directory. The same three file names load the existing cache; changing any one
of them produces a new cache key and rebuilds the cache.

Dataset roots may contain repeat-count subfolders such as
`30_Elvira - Atelier Sophie 2` or `5_other-character`. The numeric prefix is
applied independently to that folder, so multiple folders can use different
repeat counts. The cache loader also compares the expanded sample layout and
rebuilds the model-keyed cache when dataset folders or repeat counts change.
### Memory preflight

Training can be guarded before the base model is initialized:

```text
afterburner-cli --train-cache ... --base-model ... --output-lora ... \
  --steps 1 --memory-budget-gb 16
```

The CLI prints a conservative estimate based on the cache shape, attention
chunk size, backend dtype, base model file, and current 28-block autograd
retention. If the estimate exceeds the budget, it exits before allocating
model tensors. This is a safety gate, not an exact driver-level VRAM report.
The base model is also held in CubeCL's persistent allocation pool on CUDA;
transient activations remain dynamic until a backend-level workspace allocator
is available.

CUDA runs call CubeCL memory cleanup after every optimizer step. Set
`AFTERBURNER_MEMORY_LOG_EVERY=1` to print `bytes_in_use` and
`bytes_reserved` for every step; the default is every 10 steps. A rising
`bytes_reserved` with stable `bytes_in_use` is allocator pooling, while a
rising `bytes_in_use` indicates live tensors or an unreleased graph.
