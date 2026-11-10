use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct TrainingConfig {
    train_cache: Option<String>,
    base_model: Option<String>,
    output_lora: Option<String>,
    steps: Option<usize>,
    learning_rate: Option<f64>,
    lr_schedule: Option<String>,
    lr_warmup_steps: Option<usize>,
    lr_min: Option<f64>,
    seed: Option<u64>,
    lora_rank: Option<usize>,
    lora_alpha: Option<usize>,
    flow_matching: Option<bool>,
    device: Option<String>,
    gpu_index: Option<usize>,
    weighting_scheme: Option<String>,
    max_grad_norm: Option<f32>,
    checkpoint_every: Option<usize>,
    checkpoint_prefix: Option<String>,
    model_cache_dir: Option<String>,
    memory_budget_gb: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct MemoryPreflight {
    base_model_bytes: u64,
    cache_bytes: usize,
    estimated_training_bytes: u64,
    attention_chunk: usize,
    token_count: usize,
    dtype_bytes: usize,
    autodiff_buffer_factor: usize,
}

fn estimate_training_memory(
    base_model: &str,
    cache: &afterburner::InMemoryTrainingCache,
    dtype_bytes: usize,
) -> Result<MemoryPreflight, String> {
    let base_model_bytes = std::fs::metadata(base_model)
        .map_err(|error| format!("failed to stat base model {base_model}: {error}"))?
        .len();
    let sample = cache
        .get(0)
        .ok_or_else(|| "training cache is empty".to_owned())?;
    let latent_shape = &sample.image_latent.shape;
    if latent_shape.len() < 4 {
        return Err(format!(
            "cached latent must have at least 4 dimensions, got {:?}",
            latent_shape
        ));
    }
    // Anima patchifies the latent spatial plane into 2x2 patches and adds the
    // four latent channels from each patch to the input channel dimension.
    let token_count = latent_shape[2].saturating_mul(latent_shape[3]);
    let attention_chunk = std::env::var("AFTERBURNER_ATTENTION_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128);
    let text_tokens = sample.text_embedding.shape.get(1).copied().unwrap_or(512);
    let bytes = dtype_bytes as u128;
    let tokens = token_count as u128;
    let text = text_tokens as u128;
    let heads = 16_u128;
    let hidden = 2048_u128;
    let mlp = 8192_u128;
    const AUTODIFF_BUFFER_FACTOR: usize = 4;
    // This is deliberately conservative: it includes the 28-block autograd
    // activation retention currently used by Burn, plus the largest chunked
    // attention score/value workspace. It is a preflight guard, not a claim
    // about exact driver allocation accounting.
    let block_activation = tokens
        .saturating_mul(hidden)
        .saturating_mul(12)
        .saturating_add(tokens.saturating_mul(mlp).saturating_mul(2))
        .saturating_mul(bytes);
    let attention_workspace = heads
        .saturating_mul(attention_chunk as u128)
        .saturating_mul(tokens.max(text))
        .saturating_mul(4)
        .saturating_mul(bytes);
    let estimated_training_bytes = (base_model_bytes as u128)
        .saturating_add(
            block_activation
                .saturating_mul(28)
                .saturating_mul(AUTODIFF_BUFFER_FACTOR as u128),
        )
        .saturating_add(
            attention_workspace
                .saturating_mul(28)
                .saturating_mul(AUTODIFF_BUFFER_FACTOR as u128),
        )
        .min(u64::MAX as u128) as u64;
    Ok(MemoryPreflight {
        base_model_bytes,
        cache_bytes: cache.total_tensor_bytes(),
        estimated_training_bytes,
        attention_chunk,
        token_count,
        dtype_bytes,
        autodiff_buffer_factor: AUTODIFF_BUFFER_FACTOR,
    })
}

fn print_memory_preflight(preflight: MemoryPreflight) {
    println!("memory_preflight=conservative_estimate");
    println!("memory_base_model_bytes={}", preflight.base_model_bytes);
    println!("memory_cache_host_bytes={}", preflight.cache_bytes);
    println!("memory_token_count={}", preflight.token_count);
    println!("memory_attention_chunk={}", preflight.attention_chunk);
    println!("memory_dtype_bytes={}", preflight.dtype_bytes);
    println!(
        "memory_autodiff_buffer_factor={}",
        preflight.autodiff_buffer_factor
    );
    println!(
        "memory_estimated_training_bytes={}",
        preflight.estimated_training_bytes
    );
}

fn candle_device(name: &str, index: usize) -> candle_core::Device {
    if name.eq_ignore_ascii_case("cuda") {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = index;
            eprintln!(
                "CUDA was requested, but the `cuda` feature is disabled; falling back to CPU."
            );
            eprintln!("Rebuild with `--features cuda` to enable CUDA.");
            return candle_core::Device::Cpu;
        }

        #[cfg(feature = "cuda")]
        {
            return match candle_core::Device::new_cuda(index) {
                Ok(device) => device,
                Err(error) => {
                    eprintln!("CUDA is unavailable for Candle; falling back to CPU: {error}");
                    candle_core::Device::Cpu
                }
            };
        }
    }
    if name.eq_ignore_ascii_case("vulkan") {
        eprintln!("Vulkan is available for Burn training, but Candle preprocessing uses CPU.");
    }
    candle_core::Device::Cpu
}

fn resolve_tokenizer_path(
    explicit: Option<&str>,
    text_encoder_path: &str,
    relative_candidates: &[&str],
    label: &str,
) -> PathBuf {
    if let Some(path) = explicit {
        return PathBuf::from(path);
    }

    let model_path = Path::new(text_encoder_path);
    let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
    let mut search_dirs = vec![model_dir.to_path_buf()];
    search_dirs.push(model_dir.join("tokenizer"));
    if let Some(parent) = model_dir.parent() {
        search_dirs.push(parent.to_path_buf());
        search_dirs.push(parent.join("tokenizer"));
    }
    if let Ok(directory) = std::env::var("AFTERBURNER_TOKENIZER_DIR") {
        search_dirs.insert(0, PathBuf::from(directory));
    }

    for directory in search_dirs {
        for relative in relative_candidates {
            let candidate = directory.join(relative);
            if candidate.is_file() {
                eprintln!("auto-selected {label} tokenizer: {}", candidate.display());
                return candidate;
            }
        }
    }

    eprintln!(
        "unable to find {label} tokenizer near {}",
        model_path.display()
    );
    eprintln!(
        "provide the path with the corresponding CLI option or set AFTERBURNER_TOKENIZER_DIR"
    );
    std::process::exit(2);
}

fn main() {
    const TRAINING_STACK_SIZE: usize = 64 * 1024 * 1024;
    let result = std::thread::Builder::new()
        .name("afterburner-main".to_owned())
        .stack_size(TRAINING_STACK_SIZE)
        .spawn(run)
        .expect("failed to create Afterburner worker thread")
        .join();
    if result.is_err() {
        eprintln!("Afterburner worker thread terminated unexpectedly");
        std::process::exit(1);
    }
}

fn run() {
    let mut model_type = "anima".to_owned();
    let mut inspect_lora = None;
    let mut inspect_lora_factors = None;
    let mut inspect_weights = None;
    let mut inspect_dataset = None;
    let mut inspect_cache = None;
    let mut inspect_text_encoder = None;
    let mut inspect_vae = None;
    let mut inspect_vae_stem = None;
    let mut text_encoder_model = None;
    let mut vae_model = None;
    let mut t5_tokenizer_model = None;
    let mut preprocess_dataset = None;
    let mut train_cache = None;
    let mut base_model = None;
    let mut output_lora = None;
    let mut train_steps = 0usize;
    let mut learning_rate = 1e-4_f64;
    let mut lr_schedule = "warmup_cosine".to_owned();
    let mut lr_warmup_steps = None::<usize>;
    let mut lr_min = None::<f64>;
    let mut max_grad_norm = 0.0_f32;
    let mut seed = 42_u64;
    let mut lora_rank = 16usize;
    let mut lora_alpha = 16usize;
    #[cfg(feature = "bf16")]
    let mut lora_dtype = "bf16".to_owned();
    #[cfg(not(feature = "bf16"))]
    let mut lora_dtype = "f32".to_owned();
    let mut checkpoint_every = 0usize;
    let mut checkpoint_prefix = None;
    let mut weighting_scheme = "none".to_owned();
    let mut flow_matching = false;
    let mut device_name = "cpu".to_owned();
    let mut gpu_index = 0usize;
    let mut max_resolution = 0_u32;
    let mut config_path = None;
    let mut explicit_train_cache = false;
    let mut explicit_base_model = false;
    let mut explicit_output_lora = false;
    let mut explicit_steps = false;
    let mut explicit_learning_rate = false;
    let mut explicit_lr_schedule = false;
    let mut explicit_lr_warmup_steps = false;
    let mut explicit_lr_min = false;
    let mut explicit_seed = false;
    let mut explicit_lora_rank = false;
    let mut explicit_lora_alpha = false;
    let mut explicit_flow_matching = false;
    let mut explicit_device = false;
    let mut explicit_gpu_index = false;
    let mut explicit_weighting_scheme = false;
    let mut explicit_max_grad_norm = false;
    let mut explicit_checkpoint_every = false;
    let mut explicit_checkpoint_prefix = false;
    let mut output_cache = None;
    let mut model_cache_dir = "target/afterburner-model-cache".to_owned();
    let mut memory_budget_gb = None::<f64>;
    let mut explicit_memory_budget = false;
    let mut text_token_ids = vec![1_u32, 1_u32];
    let mut tokenizer_path = None;
    let mut caption = None;
    let mut repeats = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--model-type" {
            if let Some(value) = args.next() {
                model_type = value;
            }
        } else if argument == "--inspect-lora" {
            inspect_lora = args.next();
        } else if argument == "--inspect-lora-factors" {
            inspect_lora_factors = args.next();
        } else if argument == "--inspect-weights" {
            inspect_weights = args.next();
        } else if argument == "--inspect-dataset" {
            inspect_dataset = args.next();
        } else if argument == "--inspect-cache" {
            inspect_cache = args.next();
        } else if argument == "--inspect-text-encoder" {
            inspect_text_encoder = args.next();
        } else if argument == "--text-encoder" {
            text_encoder_model = args.next();
        } else if argument == "--t5-tokenizer" {
            t5_tokenizer_model = args.next();
        } else if argument == "--inspect-vae" {
            inspect_vae = args.next();
        } else if argument == "--inspect-vae-stem" {
            inspect_vae_stem = args.next();
        } else if argument == "--vae" {
            vae_model = args.next();
        } else if argument == "--preprocess-dataset" {
            preprocess_dataset = args.next();
        } else if argument == "--train-cache" {
            train_cache = args.next();
            explicit_train_cache = true;
        } else if argument == "--base-model" {
            base_model = args.next();
            explicit_base_model = true;
        } else if argument == "--output-lora" {
            output_lora = args.next();
            explicit_output_lora = true;
        } else if argument == "--steps" {
            train_steps = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_steps = true;
        } else if argument == "--learning-rate" {
            learning_rate = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1e-4);
            explicit_learning_rate = true;
        } else if argument == "--lr-schedule" {
            if let Some(value) = args.next() {
                lr_schedule = value;
            }
            explicit_lr_schedule = true;
        } else if argument == "--lr-warmup-steps" {
            lr_warmup_steps = args.next().and_then(|value| value.parse().ok());
            explicit_lr_warmup_steps = true;
        } else if argument == "--lr-min" {
            lr_min = args.next().and_then(|value| value.parse().ok());
            explicit_lr_min = true;
        } else if argument == "--max-grad-norm" {
            max_grad_norm = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.0);
            explicit_max_grad_norm = true;
        } else if argument == "--seed" {
            seed = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(42);
            explicit_seed = true;
        } else if argument == "--lora-rank" {
            lora_rank = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16);
            explicit_lora_rank = true;
        } else if argument == "--lora-alpha" {
            lora_alpha = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16);
            explicit_lora_alpha = true;
        } else if argument == "--lora-dtype" {
            if let Some(value) = args.next() {
                lora_dtype = value;
            }
        } else if argument == "--checkpoint-every" {
            checkpoint_every = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_checkpoint_every = true;
        } else if argument == "--checkpoint-prefix" {
            checkpoint_prefix = args.next();
            explicit_checkpoint_prefix = true;
        } else if argument == "--weighting-scheme" {
            if let Some(value) = args.next() {
                weighting_scheme = value;
            }
            explicit_weighting_scheme = true;
        } else if argument == "--device" {
            if let Some(value) = args.next() {
                device_name = value;
            }
            explicit_device = true;
        } else if argument == "--gpu-index" {
            gpu_index = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_gpu_index = true;
        } else if argument == "--max-resolution" {
            max_resolution = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
        } else if argument == "--flow-matching" {
            flow_matching = true;
            explicit_flow_matching = true;
        } else if argument == "--config" {
            config_path = args.next();
        } else if argument == "--output-cache" {
            output_cache = args.next();
        } else if argument == "--model-cache-dir" {
            if let Some(value) = args.next() {
                model_cache_dir = value;
            }
        } else if argument == "--memory-budget-gb" {
            memory_budget_gb = args.next().and_then(|value| value.parse().ok());
            explicit_memory_budget = true;
        } else if argument == "--token-ids" {
            if let Some(value) = args.next() {
                text_token_ids = value
                    .split(',')
                    .filter_map(|item| item.trim().parse::<u32>().ok())
                    .collect();
            }
        } else if argument == "--tokenizer" {
            tokenizer_path = args.next();
        } else if argument == "--caption" {
            caption = args.next();
        } else if argument == "--repeats" {
            repeats = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
        }
    }

    if let Some(path) = config_path {
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            eprintln!("failed to read training config {path}: {error}");
            std::process::exit(2);
        });
        let config: TrainingConfig = serde_json::from_str(&contents).unwrap_or_else(|error| {
            eprintln!("failed to parse training config {path}: {error}");
            std::process::exit(2);
        });
        if !explicit_train_cache {
            train_cache = config.train_cache;
        }
        if !explicit_base_model {
            base_model = config.base_model;
        }
        if !explicit_output_lora {
            output_lora = config.output_lora;
        }
        if !explicit_steps {
            if let Some(value) = config.steps {
                train_steps = value;
            }
        }
        if !explicit_learning_rate {
            if let Some(value) = config.learning_rate {
                learning_rate = value;
            }
        }
        if !explicit_lr_schedule {
            if let Some(value) = config.lr_schedule {
                lr_schedule = value;
            }
        }
        if !explicit_lr_warmup_steps {
            lr_warmup_steps = config.lr_warmup_steps;
        }
        if !explicit_lr_min {
            lr_min = config.lr_min;
        }
        if !explicit_seed {
            if let Some(value) = config.seed {
                seed = value;
            }
        }
        if !explicit_lora_rank {
            if let Some(value) = config.lora_rank {
                lora_rank = value;
            }
        }
        if !explicit_lora_alpha {
            if let Some(value) = config.lora_alpha {
                lora_alpha = value;
            }
        }
        if !explicit_flow_matching {
            if let Some(value) = config.flow_matching {
                flow_matching = value;
            }
        }
        if !explicit_device {
            if let Some(value) = config.device {
                device_name = value;
            }
        }
        if !explicit_gpu_index {
            if let Some(value) = config.gpu_index {
                gpu_index = value;
            }
        }
        if !explicit_weighting_scheme {
            if let Some(value) = config.weighting_scheme {
                weighting_scheme = value;
            }
        }
        if !explicit_max_grad_norm {
            if let Some(value) = config.max_grad_norm {
                max_grad_norm = value;
            }
        }
        if !explicit_checkpoint_every {
            if let Some(value) = config.checkpoint_every {
                checkpoint_every = value;
            }
        }
        if !explicit_checkpoint_prefix {
            checkpoint_prefix = config.checkpoint_prefix;
        }
        if !explicit_memory_budget {
            memory_budget_gb = config.memory_budget_gb;
        }
        if let Some(value) = config.model_cache_dir {
            model_cache_dir = value;
        }
    }

    let model_type = match model_type.parse::<afterburner::ModelType>() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let (revision, diffusion_model, text_encoder, vae) = match model_type {
        afterburner::ModelType::Anima => (
            "anima-aesthetic-v1.1",
            "anima-aesthetic-v1.1.safetensors",
            "qwen_3_06b_base.safetensors",
            "qwen_image_vae.safetensors",
        ),
        afterburner::ModelType::Flux => (
            "flux",
            "flux.safetensors",
            "text-encoder.safetensors",
            "flux2-vae.safetensors",
        ),
        afterburner::ModelType::StableDiffusion => (
            "stable-diffusion",
            "unet.safetensors",
            "clip.safetensors",
            "vae.safetensors",
        ),
    };
    let config =
        afterburner::ModelConfig::new(model_type, revision, diffusion_model, text_encoder, vae);
    let descriptor = afterburner::resolve_model(config).expect("valid model configuration");
    println!("Afterburner CLI {}", afterburner::VERSION);
    println!("model_type={model_type}");
    println!("resolved={descriptor:?}");
    if let Some(path) = inspect_lora {
        let plan = afterburner::inspect_anima_lora_file(path, &afterburner::LoraConfig::default())
            .expect("failed to inspect Anima LoRA targets");
        println!("lora_targets={}", plan.target_count());
        println!("lora_parameters={}", plan.parameter_count());
    }
    if let Some(path) = inspect_lora_factors {
        let plan = afterburner::inspect_anima_lora_factors(path)
            .expect("failed to inspect exported Anima LoRA factors");
        println!("lora_factor_tensors={}", plan.tensor_count());
        println!("lora_factor_parameters={}", plan.parameter_count);
    }
    if let Some(path) = inspect_weights {
        let coverage = afterburner::inspect_anima_weight_coverage(path)
            .expect("failed to inspect Anima weight coverage");
        println!("mapped_weights={}", coverage.mapped_count());
        println!("unmapped_weights={}", coverage.unmapped_count());
        for key in coverage.unmapped {
            println!("unmapped={key}");
        }
    }
    if let Some(path) = inspect_dataset {
        let dataset = afterburner::ImageCaptionDataset::from_dir_with_folder_repeats(path, repeats)
            .expect("failed to inspect image/caption dataset");
        println!("dataset_samples={}", dataset.samples().len());
        println!("dataset_repeats={repeats}");
        for sample in dataset.samples() {
            println!(
                "sample={} repeat={} caption={}",
                sample.image_path.display(),
                sample.repeat_index,
                sample.caption
            );
        }
    }
    if let Some(path) = inspect_cache {
        let cache = afterburner::InMemoryTrainingCache::load(path)
            .expect("failed to load encoded training cache");
        println!("cache_samples={}", cache.len());
        println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
    }
    if let Some(path) = inspect_text_encoder.clone() {
        let device = candle_device(&device_name, gpu_index);
        if let Some(tokenizer_path) = tokenizer_path.clone() {
            let tokenizer = afterburner::QwenTokenizer::from_file(tokenizer_path, 512, 151_643)
                .expect("failed to load Qwen tokenizer");
            let mut encoder = afterburner::CandleQwen3TextEncoder::from_safetensors(
                path,
                afterburner::Qwen3TextConfig::default(),
                candle_core::DType::F32,
                &device,
                tokenizer,
            )
            .expect("failed to load Qwen3 text encoder");
            let caption = caption.as_deref().unwrap_or("a portrait");
            let (embedding, mask) =
                afterburner::TextEncoder::encode_text_with_mask(&mut encoder, caption)
                    .expect("failed to encode caption");
            println!("text_encoder_embedding_shape={:?}", embedding.shape);
            println!(
                "text_encoder_attention_mask_shape={:?}",
                mask.map(|value| value.shape)
            );
        } else {
            let mut encoder = afterburner::CandleQwen3TextEncoder::from_safetensors(
                path,
                afterburner::Qwen3TextConfig::default(),
                candle_core::DType::F32,
                &device,
                |_: &str| Ok::<_, std::convert::Infallible>(text_token_ids.clone()),
            )
            .expect("failed to load Qwen3 text encoder");
            let embedding = encoder
                .encode_token_ids(&text_token_ids)
                .expect("failed to run Qwen3 text encoder");
            println!("text_encoder_embedding_shape={:?}", embedding.shape);
        }
    }
    if let Some(path) = inspect_vae {
        let report =
            afterburner::inspect_qwen_image_vae(path).expect("failed to inspect Qwen Image VAE");
        println!("vae_tensor_count={}", report.tensor_count);
        println!("vae_bfloat16_tensors={}", report.bfloat16_tensors);
        println!(
            "vae_required_keys={}/{}",
            report.required_keys_present, report.required_keys
        );
        for key in report.missing_required_keys {
            println!("vae_missing={key}");
        }
    }
    if let Some(path) = inspect_vae_stem.clone() {
        let device = candle_device(&device_name, gpu_index);
        let stem = afterburner::CandleQwenImageVaeStem::from_safetensors(
            path,
            candle_core::DType::F32,
            &device,
        )
        .expect("failed to load Qwen Image VAE stem");
        let image = image::RgbImage::from_pixel(8, 8, image::Rgb([127, 127, 127]));
        let input =
            afterburner::qwen_image_input(&image, &afterburner::QwenImageVaeConfig::default())
                .expect("failed to prepare Qwen Image VAE input");
        let output = stem.forward(&input).expect("failed to run Qwen VAE stem");
        println!("vae_stem_output_shape={:?}", output.shape);
        let residual = stem
            .forward_first_residual(&input)
            .expect("failed to run Qwen VAE first residual block");
        println!("vae_first_residual_output_shape={:?}", residual.shape);
        let downsample = stem
            .forward_first_downsample(&input)
            .expect("failed to run Qwen VAE first downsample");
        println!("vae_first_downsample_output_shape={:?}", downsample.shape);
        let stack = stem
            .forward_downsample_stack(&input)
            .expect("failed to run Qwen VAE downsample stack");
        println!("vae_downsample_stack_output_shape={:?}", stack.shape);
        let latent = stem
            .forward_latent(&input)
            .expect("failed to run complete Qwen VAE encoder");
        println!("vae_latent_output_shape={:?}", latent.shape);
    }
    if let Some(cache_path) = train_cache {
        let base_model = base_model
            .as_deref()
            .expect("--train-cache requires --base-model");
        let output_lora = output_lora
            .as_deref()
            .expect("--train-cache requires --output-lora");
        if train_steps == 0 {
            eprintln!("--steps must be greater than zero");
            std::process::exit(2);
        }
        eprintln!("[load] loading training cache: {cache_path}");
        let cache = afterburner::InMemoryTrainingCache::load(&cache_path).unwrap_or_else(|error| {
            eprintln!("failed to load encoded training cache {cache_path}: {error}");
            eprintln!("provide the complete .afbcache path; `...` is not a wildcard");
            std::process::exit(2);
        });
        eprintln!("[load] training cache loaded: samples={}", cache.len());
        let training_epochs = train_steps;
        let optimization_steps = training_epochs
            .checked_mul(cache.len())
            .expect("--steps multiplied by cache size overflows");
        let mut lora = afterburner::LoraConfig::default();
        lora.rank = lora_rank;
        lora.alpha = lora_alpha;
        lora.validate().expect("invalid LoRA configuration");
        let output_dtype = match lora_dtype.as_str() {
            "f32" | "float32" => afterburner::LoraOutputDtype::F32,
            "bf16" | "bfloat16" => afterburner::LoraOutputDtype::BF16,
            value => {
                eprintln!("unsupported --lora-dtype: {value} (expected f32 or bf16)");
                std::process::exit(2);
            }
        };
        #[cfg(feature = "bf16")]
        let backend_dtype_bytes = 2usize;
        #[cfg(not(feature = "bf16"))]
        let backend_dtype_bytes = 4usize;
        let memory_preflight = estimate_training_memory(base_model, &cache, backend_dtype_bytes)
            .unwrap_or_else(|error| {
                eprintln!("memory preflight failed: {error}");
                std::process::exit(2);
            });
        print_memory_preflight(memory_preflight);
        if let Some(budget_gb) = memory_budget_gb {
            if !budget_gb.is_finite() || budget_gb <= 0.0 {
                eprintln!("--memory-budget-gb must be a positive finite number");
                std::process::exit(2);
            }
            let budget_bytes = budget_gb * 1024.0 * 1024.0 * 1024.0;
            if memory_preflight.estimated_training_bytes as f64 > budget_bytes {
                eprintln!(
                    "memory preflight rejected run: estimated {:.2} GiB exceeds budget {:.2} GiB; no model tensors were allocated",
                    memory_preflight.estimated_training_bytes as f64 / 1024.0_f64.powi(3),
                    budget_gb
                );
                std::process::exit(3);
            }
            eprintln!(
                "memory preflight accepted: estimated {:.2} GiB <= budget {:.2} GiB",
                memory_preflight.estimated_training_bytes as f64 / 1024.0_f64.powi(3),
                budget_gb
            );
        }
        #[cfg(all(feature = "cuda", feature = "bf16"))]
        type BaseBackend = burn_cuda::Cuda<half::bf16>;
        #[cfg(all(feature = "cuda", not(feature = "bf16")))]
        type BaseBackend = burn_cuda::Cuda;
        #[cfg(all(not(feature = "cuda"), feature = "vulkan", feature = "bf16"))]
        type BaseBackend = burn::backend::Vulkan<half::bf16>;
        #[cfg(all(not(feature = "cuda"), feature = "vulkan", not(feature = "bf16")))]
        type BaseBackend = burn::backend::Vulkan;
        #[cfg(all(not(feature = "cuda"), not(feature = "vulkan")))]
        type BaseBackend = burn_ndarray::NdArray<f32>;
        #[cfg(feature = "checkpointing")]
        type TrainBackend = burn_autodiff::Autodiff<
            BaseBackend,
            burn_autodiff::checkpoint::strategy::BalancedCheckpointing,
        >;
        #[cfg(not(feature = "checkpointing"))]
        type TrainBackend = burn_autodiff::Autodiff<BaseBackend>;
        #[cfg(feature = "cuda")]
        eprintln!("[load] initializing Burn CUDA device");
        #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
        eprintln!("[load] initializing Burn Vulkan device");
        let device = {
            #[cfg(feature = "cuda")]
            {
                burn_cuda::CudaDevice::new(gpu_index)
            }
            #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
            {
                let device = burn::backend::wgpu::WgpuDevice::default();
                burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
                    &device,
                    Default::default(),
                );
                device
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "vulkan")))]
            {
                <TrainBackend as burn::tensor::backend::Backend>::Device::default()
            }
        };
        #[cfg(feature = "cuda")]
        eprintln!("[load] Burn CUDA device initialized");
        #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
        eprintln!("[load] Burn Vulkan device initialized");
        #[cfg(feature = "cuda")]
        use cubecl_core::Runtime;
        #[cfg(feature = "cuda")]
        let memory_client = <cubecl_cuda::CudaRuntime as Runtime>::client(&device);
        eprintln!("loading Anima base model: {base_model}");
        #[cfg(feature = "cuda")]
        let (model, load_report) = {
            // Keep the static base model in CubeCL's persistent pool. Training
            // activations remain dynamic, but the multi-gigabyte base weights
            // cannot fragment or be recycled into transient work buffers.
            memory_client.memory_persistent_allocation((), |_| {
                afterburner::load_anima_model::<TrainBackend>(base_model, &lora, &device)
                    .expect("failed to load Anima base model")
            })
        };
        #[cfg(not(feature = "cuda"))]
        let (model, load_report) =
            afterburner::load_anima_model::<TrainBackend>(base_model, &lora, &device)
                .expect("failed to load Anima base model");
        println!("base_model_loaded={}", load_report.loaded_count());
        println!("base_model_skipped={}", load_report.skipped_count());
        #[cfg(feature = "cuda")]
        {
            let usage = memory_client.memory_usage();
            eprintln!(
                "[memory] after base load: in_use={} reserved={} allocations={}",
                usage.bytes_in_use, usage.bytes_reserved, usage.number_allocs
            );
        }
        let schedule = if flow_matching {
            afterburner::DiffusionSchedule::flow_matching(1000, 1.0)
                .expect("invalid flow-matching schedule")
        } else {
            afterburner::DiffusionSchedule::linear(1000, 0.0001, 0.02)
                .expect("invalid diffusion schedule")
        };
        let target = if flow_matching {
            afterburner::TrainingTarget::FlowMatching
        } else {
            afterburner::TrainingTarget::Epsilon
        };
        let loss_weighting = match weighting_scheme.as_str() {
            "none" => afterburner::LossWeighting::None,
            "sigma_sqrt" => afterburner::LossWeighting::SigmaSqrt,
            "cosmap" => afterburner::LossWeighting::Cosmap,
            value => {
                eprintln!("unsupported --weighting-scheme: {value}");
                std::process::exit(2);
            }
        };
        let mut optimizer_config = burn_optim::AdamConfig::new();
        if max_grad_norm > 0.0 {
            optimizer_config = optimizer_config.with_grad_clipping(Some(
                burn_optim::grad_clipping::GradientClippingConfig::Norm(max_grad_norm),
            ));
        }
        let mut optimizer =
            optimizer_config.init::<TrainBackend, afterburner::AnimaDiT<TrainBackend>>();
        let mut logger = afterburner::StderrLogger::stderr();
        let checkpoint_lora = lora.clone();
        let mut checkpoint = checkpoint_prefix.map(|prefix| {
            move |step: usize,
                  model: &afterburner::AnimaDiT<TrainBackend>|
                  -> Result<(), afterburner::TrainingError> {
                let lora_path = format!("{prefix}-{step:08}.safetensors");
                afterburner::export_anima_lora_with_config_and_dtype(
                    &lora_path,
                    model,
                    &checkpoint_lora,
                    output_dtype,
                )
                .map(|_| ())
                .map_err(|error| afterburner::TrainingError::Checkpoint(error.to_string()))
            }
        });
        let checkpoint = checkpoint.as_mut().map(
            |callback| -> &mut dyn FnMut(
                usize,
                &afterburner::AnimaDiT<TrainBackend>,
            ) -> Result<(), afterburner::TrainingError> { callback },
        );
        let learning_rate_policy = match lr_schedule.as_str() {
            "constant" => afterburner::LearningRatePolicy::Constant { learning_rate },
            "warmup_cosine" | "cosine" => {
                let warmup_steps = lr_warmup_steps.unwrap_or((optimization_steps / 10).max(1));
                let min_learning_rate = lr_min.unwrap_or(learning_rate * 0.1);
                if min_learning_rate < 0.0 || min_learning_rate > learning_rate {
                    eprintln!("--lr-min must be between 0 and --learning-rate");
                    std::process::exit(2);
                }
                afterburner::LearningRatePolicy::WarmupCosine {
                    peak_learning_rate: learning_rate,
                    min_learning_rate,
                    warmup_steps,
                    total_steps: optimization_steps,
                }
            }
            value => {
                eprintln!(
                    "unsupported --lr-schedule: {value} (expected warmup_cosine or constant)"
                );
                std::process::exit(2);
            }
        };
        println!("lr_schedule={lr_schedule}");
        #[cfg(feature = "cuda")]
        let memory_log_every = std::env::var("AFTERBURNER_MEMORY_LOG_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(10);
        #[cfg(feature = "cuda")]
        let mut memory_hook = |step: usize| {
            memory_client.memory_cleanup();
            if step % memory_log_every == 0 || step == 1 {
                let usage = memory_client.memory_usage();
                eprintln!(
                    "[memory] step={step}: in_use={} reserved={} allocations={}",
                    usage.bytes_in_use, usage.bytes_reserved, usage.number_allocs
                );
            }
        };
        #[cfg(feature = "cuda")]
        let memory_hook: Option<&mut dyn FnMut(usize)> = Some(&mut memory_hook);
        #[cfg(not(feature = "cuda"))]
        let memory_hook: Option<&mut dyn FnMut(usize)> = None;
        let (model, report) =
            afterburner::train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy_and_step_hook(
                &mut optimizer,
                model,
                &cache,
                &schedule,
                optimization_steps,
                learning_rate_policy,
                seed,
                target,
                lora.clone(),
                &mut logger,
                (checkpoint_every > 0).then_some(checkpoint_every),
                checkpoint,
                loss_weighting,
                memory_hook,
            )
            .expect("cached training failed");
        let factor_count = afterburner::export_anima_lora_with_config_and_dtype(
            output_lora,
            &model,
            &lora,
            output_dtype,
        )
        .expect("failed to export LoRA factors");
        println!("training_steps={}", report.steps);
        println!("training_epochs={training_epochs}");
        println!("samples_per_epoch={}", cache.len());
        println!("lora_dtype={lora_dtype}");
        println!("last_loss={:?}", report.last_loss);
        println!("lora_factor_tensors={factor_count}");
        println!("lora_path={output_lora}");
    }
    if let Some(dataset_path) = preprocess_dataset {
        let text_path = text_encoder_model
            .as_deref()
            .expect("--preprocess-dataset requires --text-encoder");
        let vae_path = vae_model
            .as_deref()
            .expect("--preprocess-dataset requires --vae");
        let tokenizer_path = resolve_tokenizer_path(
            tokenizer_path.as_deref(),
            text_path,
            &[
                "tokenizer.json",
                "qwen_tokenizer/tokenizer.json",
                "llama_tokenizer/tokenizer.json",
            ],
            "Qwen",
        );
        let t5_tokenizer_path = resolve_tokenizer_path(
            t5_tokenizer_model.as_deref(),
            text_path,
            &["t5_tokenizer/tokenizer.json", "tokenizer.json"],
            "T5",
        );
        let dataset =
            afterburner::ImageCaptionDataset::from_dir_with_folder_repeats(&dataset_path, repeats)
                .expect("failed to load preprocessing dataset");
        let cache_path = if let Some(path) = output_cache {
            std::path::PathBuf::from(path)
        } else {
            let diffusion_path = base_model.as_deref().unwrap_or(diffusion_model);
            let key = afterburner::ModelCacheKey::from_paths(
                model_type,
                diffusion_path,
                text_path,
                vae_path,
            )
            .expect("model cache key requires file names");
            let path = key.path_in(&model_cache_dir);
            if path.is_file() {
                let cache = afterburner::InMemoryTrainingCache::load(&path)
                    .expect("failed to load model-keyed encoded cache");
                cache
                    .validate_anima_shapes()
                    .expect("cached model data does not match the Anima tensor contract");
                let matches_dataset = cache.len() == dataset.samples().len()
                    && cache.iter().zip(dataset.samples()).all(|(cached, sample)| {
                        cached.sample_id == sample.image_path.to_string_lossy()
                            && cached.repeat_index == sample.repeat_index
                    });
                if matches_dataset {
                    println!("cache_hit=true");
                    println!("cache_key={}", key.file_name());
                    println!("preprocessed_samples={}", cache.len());
                    println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
                    println!("cache_path={}", path.display());
                    return;
                }
                println!("cache_hit=false");
                println!("cache_reason=dataset_layout_changed");
            }
            std::fs::create_dir_all(&model_cache_dir)
                .expect("failed to create model cache directory");
            println!("cache_hit=false");
            println!("cache_key={}", key.file_name());
            path
        };
        let device = candle_device(&device_name, gpu_index);
        let tokenizer = afterburner::QwenTokenizer::from_file(&tokenizer_path, 512, 151_643)
            .expect("failed to load Qwen tokenizer");
        let text_encoder = afterburner::CandleQwen3TextEncoder::from_safetensors(
            text_path,
            afterburner::Qwen3TextConfig::default(),
            candle_core::DType::F32,
            &device,
            tokenizer,
        )
        .expect("failed to load Qwen3 text encoder");
        let vae = afterburner::CandleQwenImageVaeStem::from_safetensors(
            vae_path,
            candle_core::DType::F32,
            &device,
        )
        .expect("failed to load Qwen Image VAE");
        let vae = afterburner::QwenImageVaePreprocessor::new(
            vae,
            afterburner::QwenImageVaeConfig::default(),
        )
        .expect("invalid Qwen Image VAE configuration")
        .with_max_dimension((max_resolution > 0).then_some(max_resolution));
        let t5_tokenizer = afterburner::T5Tokenizer::from_file(&t5_tokenizer_path, 512, 0)
            .expect("failed to load T5 tokenizer");
        let text_encoder = afterburner::AnimaTextEncoder::new(text_encoder, t5_tokenizer);
        let mut preprocessor = afterburner::CompositePreprocessor::new(text_encoder, vae);
        let mut logger = afterburner::StderrLogger::stderr();
        let cache =
            afterburner::preprocess_dataset_with_logger(&dataset, &mut preprocessor, &mut logger)
                .expect("failed to preprocess dataset");
        cache
            .validate_anima_shapes()
            .expect("preprocessed cache does not match the Anima tensor contract");
        cache
            .save(&cache_path)
            .expect("failed to save encoded cache");
        println!("preprocessed_samples={}", cache.len());
        println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
        println!("cache_path={}", cache_path.display());
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use burn::{backend::Vulkan, tensor::Tensor};

    #[test]
    fn test_vulkan_smoke() {
        type Backend = Vulkan;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
            &device,
            Default::default(),
        );
        let a = Tensor::<Backend, 2>::ones([32, 32], &device);
        let b = Tensor::<Backend, 2>::ones([32, 32], &device);
        let values = a
            .matmul(b)
            .into_data()
            .to_vec::<f32>()
            .expect("Vulkan matmul output must be readable");
        assert!((values[0] - 32.0).abs() < 1e-3);
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn test_vulkan_bf16_smoke() {
        type Backend = Vulkan<half::bf16>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
            &device,
            Default::default(),
        );
        let a = Tensor::<Backend, 2>::ones([32, 32], &device);
        let b = Tensor::<Backend, 2>::ones([32, 32], &device);
        let values = a
            .matmul(b)
            .into_data()
            .to_vec::<half::bf16>()
            .expect("Vulkan BF16 matmul output must be readable");
        assert!((values[0].to_f32() - 32.0).abs() < 0.5);
    }
}
