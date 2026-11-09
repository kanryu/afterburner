use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct TrainingConfig {
    train_cache: Option<String>,
    base_model: Option<String>,
    output_lora: Option<String>,
    steps: Option<usize>,
    learning_rate: Option<f64>,
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
    let mut max_grad_norm = 0.0_f32;
    let mut seed = 42_u64;
    let mut lora_rank = 16usize;
    let mut lora_alpha = 16usize;
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
        let dataset = afterburner::ImageCaptionDataset::from_dir(path, repeats)
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
        let cache = afterburner::InMemoryTrainingCache::load(cache_path)
            .expect("failed to load encoded training cache");
        let mut lora = afterburner::LoraConfig::default();
        lora.rank = lora_rank;
        lora.alpha = lora_alpha;
        lora.validate().expect("invalid LoRA configuration");
        #[cfg(feature = "cuda")]
        type BaseBackend = burn_cuda::Cuda;
        #[cfg(not(feature = "cuda"))]
        type BaseBackend = burn_ndarray::NdArray<f32>;
        #[cfg(feature = "checkpointing")]
        type TrainBackend = burn_autodiff::Autodiff<
            BaseBackend,
            burn_autodiff::checkpoint::strategy::BalancedCheckpointing,
        >;
        #[cfg(not(feature = "checkpointing"))]
        type TrainBackend = burn_autodiff::Autodiff<BaseBackend>;
        let device = {
            #[cfg(feature = "cuda")]
            {
                burn_cuda::CudaDevice::new(gpu_index)
            }
            #[cfg(not(feature = "cuda"))]
            {
                <TrainBackend as burn::tensor::backend::Backend>::Device::default()
            }
        };
        eprintln!("loading Anima base model: {base_model}");
        let (model, load_report) =
            afterburner::load_anima_model::<TrainBackend>(base_model, &lora, &device)
                .expect("failed to load Anima base model");
        println!("base_model_loaded={}", load_report.loaded_count());
        println!("base_model_skipped={}", load_report.skipped_count());
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
                afterburner::export_anima_lora_with_config(&lora_path, model, &checkpoint_lora)
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
        let (model, report) =
            afterburner::train_cached_with_options_and_logger_and_checkpoint_with_weighting(
                &mut optimizer,
                model,
                &cache,
                &schedule,
                train_steps,
                learning_rate,
                seed,
                target,
                lora.clone(),
                &mut logger,
                (checkpoint_every > 0).then_some(checkpoint_every),
                checkpoint,
                loss_weighting,
            )
            .expect("cached training failed");
        let factor_count = afterburner::export_anima_lora_with_config(output_lora, &model, &lora)
            .expect("failed to export LoRA factors");
        println!("training_steps={}", report.steps);
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
        let cache_path = output_cache
            .as_deref()
            .expect("--preprocess-dataset requires --output-cache");
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
        let dataset = afterburner::ImageCaptionDataset::from_dir(&dataset_path, repeats)
            .expect("failed to load preprocessing dataset");
        let mut logger = afterburner::StderrLogger::stderr();
        let cache =
            afterburner::preprocess_dataset_with_logger(&dataset, &mut preprocessor, &mut logger)
                .expect("failed to preprocess dataset");
        cache
            .validate_anima_shapes()
            .expect("preprocessed cache does not match the Anima tensor contract");
        cache
            .save(cache_path)
            .expect("failed to save encoded cache");
        println!("preprocessed_samples={}", cache.len());
        println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
        println!("cache_path={cache_path}");
    }
}
