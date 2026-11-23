//! Burn training-step glue for cached diffusion examples.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::tensor::{
    ElementConversion, Int, Tensor, TensorData, TensorPrimitive,
    backend::AutodiffBackend,
};
use burn_optim::{GradientsParams, Optimizer};

use crate::{
    cache::InMemoryTrainingCache,
    diffusion::{
        DiffusionSchedule, LossWeighting, TimestepSampling, TrainingExample, TrainingTarget,
    },
    dit::{AnimaAttentionLayer, AnimaDiT, AnimaTensorError, BlockCheckpointBackend},
    logging::{NullLogger, TrainingLogger},
    lora::LoraConfig,
    vae::{VaeError, VaeTensor},
};

static TRAIN_DUMP_STEP: AtomicUsize = AtomicUsize::new(0);
fn dump_training_tensor<B: burn::tensor::backend::Backend, const D: usize>(
    step: usize,
    name: &str,
    tensor: &Tensor<B, D>,
) {
    let Some(directory) = std::env::var_os("AFTERBURNER_TRAIN_DUMP_DIR") else {
        return;
    };
    let max_steps = std::env::var("AFTERBURNER_TRAIN_DUMP_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    if step == 0 || step > max_steps {
        return;
    }
    let directory = std::path::PathBuf::from(directory).join(format!("step-{step:06}"));
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("[training-dump] create directory failed: {error}");
        return;
    }
    let data = tensor.to_data().convert::<f32>();
    // Training loss and the FP32 residual path may produce F32 tensors even
    // when the compiled backend's native element is F16/BF16.  The dump is a
    // diagnostic boundary and must read the tensor's actual storage dtype,
    // not the backend's default FloatElem.
    let values = match data.to_vec::<f32>() {
        Ok(values) => values
            .into_iter()
            .map(|value| value.elem::<f32>())
            .collect::<Vec<_>>(),
        Err(error) => {
            eprintln!("[training-dump] {name} read failed: {error:?}");
            return;
        }
    };
    let safe_name = name.replace(['/', '.'], "_");
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Err(error) = std::fs::write(directory.join(format!("{safe_name}.f32")), bytes) {
        eprintln!("[training-dump] {name} data write failed: {error}");
        return;
    }
    let shape = data.shape.as_slice().to_vec();
    let finite = values.iter().filter(|value| value.is_finite()).count();
    let nonfinite = values.len().saturating_sub(finite);
    let finite_values = values.iter().copied().filter(|value| value.is_finite());
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    for value in finite_values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        sum += f64::from(value);
        sum_abs += f64::from(value.abs());
        sum_squared += f64::from(value) * f64::from(value);
    }
    let count = finite.max(1) as f64;
    let shape_json = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    let metadata = format!(
        "{{\n  \"name\": \"{safe_name}\",\n  \"shape\": [{shape_json}],\n  \"dtype\": \"f32\",\n  \"numel\": {},\n  \"finite\": {finite},\n  \"nonfinite\": {nonfinite},\n  \"min\": {},\n  \"max\": {},\n  \"mean\": {},\n  \"mean_abs\": {},\n  \"l2\": {}\n}}\n",
        values.len(),
        if finite == 0 { "null".into() } else { minimum.to_string() },
        if finite == 0 { "null".into() } else { maximum.to_string() },
        if finite == 0 { "null".into() } else { (sum / count).to_string() },
        if finite == 0 { "null".into() } else { (sum_abs / count).to_string() },
        if finite == 0 { "null".into() } else { sum_squared.sqrt().to_string() },
    );
    if let Err(error) = std::fs::write(directory.join(format!("{safe_name}.json")), metadata) {
        eprintln!("[training-dump] {name} metadata write failed: {error}");
    }
}

fn dump_replay_input(step: usize, sigma: f32, noise: &VaeTensor, latent: &VaeTensor) {
    let Some(directory) = std::env::var_os("AFTERBURNER_TRAIN_DUMP_DIR") else {
        return;
    };
    let max_steps = std::env::var("AFTERBURNER_TRAIN_DUMP_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    if step == 0 || step > max_steps {
        return;
    }
    let directory = std::path::PathBuf::from(directory).join(format!("step-{step:06}"));
    if let Err(error) = std::fs::create_dir_all(&directory) {
        eprintln!("[training-dump] replay directory failed: {error}");
        return;
    }
    let write_tensor = |name: &str, tensor: &VaeTensor| {
        let mut bytes = Vec::with_capacity(tensor.data.len() * std::mem::size_of::<f32>());
        for value in &tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if let Err(error) = std::fs::write(directory.join(format!("{name}.f32")), bytes) {
            eprintln!("[training-dump] {name} replay write failed: {error}");
        }
        let shape = tensor
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        if let Err(error) = std::fs::write(
            directory.join(format!("{name}.shape.json")),
            format!("[{shape}]"),
        ) {
            eprintln!("[training-dump] {name} replay shape write failed: {error}");
        }
    };
    if let Err(error) = std::fs::write(
        directory.join("sigma.f32"),
        sigma.to_le_bytes(),
    ) {
        eprintln!("[training-dump] sigma replay write failed: {error}");
    }
    write_tensor("noise", noise);
    write_tensor("latent", latent);
}

fn load_gradient_replay<B, A>(
    model: &AnimaDiT<B, A>,
    lora: &LoraConfig,
    directory: &std::path::Path,
    device: &<B::InnerBackend as burn::tensor::backend::Backend>::Device,
) -> Result<GradientsParams, TrainingError>
where
    B: AutodiffBackend,
    A: AnimaAttentionLayer<B>,
{
    let step = TRAIN_DUMP_STEP.load(Ordering::Relaxed).max(1);
    let step_directory = directory.join(format!("step-{step:06}"));
    let directory = if step_directory.is_dir() {
        step_directory
    } else {
        directory.to_owned()
    };
    let mut collector = LoraGradientCollector {
        target_modules: lora.target_modules.clone(),
        path: Vec::new(),
        parameters: Vec::new(),
    };
    model.visit(&mut collector);
    let mut gradients = GradientsParams::new();
    for (name, id) in collector.parameters {
        let stem = directory.join(&name);
        let data_path = stem.with_extension("f32");
        let shape_path = stem.with_extension("shape.json");
        let bytes = std::fs::read(&data_path).map_err(|error| {
            TrainingError::Metrics(format!(
                "gradient replay read failed for {}: {error}",
                data_path.display()
            ))
        })?;
        if bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err(TrainingError::Metrics(format!(
                "gradient replay has invalid byte length: {}",
                data_path.display()
            )));
        }
        let shape_text = std::fs::read_to_string(&shape_path).map_err(|error| {
            TrainingError::Metrics(format!(
                "gradient replay shape read failed for {}: {error}",
                shape_path.display()
            ))
        })?;
        let shape = shape_text
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter(|value| !value.trim().is_empty())
            .map(|value| {
                value.trim().parse::<usize>().map_err(|error| {
                    TrainingError::Metrics(format!(
                        "gradient replay shape parse failed for {}: {error}",
                        shape_path.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if shape.len() != 2 {
            return Err(TrainingError::Metrics(format!(
                "gradient replay expects rank-2 LoRA tensors, got {:?} for {}",
                shape,
                data_path.display()
            )));
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        if values.len() != shape[0] * shape[1] {
            return Err(TrainingError::Metrics(format!(
                "gradient replay shape/value mismatch for {}: {:?} vs {} values",
                data_path.display(),
                shape,
                values.len()
            )));
        }
        let tensor = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(values, [shape[0], shape[1]]),
            device,
        );
        gradients.register::<B::InnerBackend, 2>(id, tensor);
    }
    println!(
        "[gradient-replay] loaded {} LoRA gradients from {}",
        gradients.len(),
        directory.display()
    );
    Ok(gradients)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingError {
    Vae(String),
    Model(AnimaTensorError),
    EmptyCache,
    Metrics(String),
    Checkpoint(String),
    ShapeMismatch {
        predicted: Vec<usize>,
        target: Vec<usize>,
    },
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vae(message) => write!(f, "training tensor conversion failed: {message}"),
            Self::Model(error) => write!(f, "training model input failed: {error}"),
            Self::EmptyCache => write!(f, "training cache is empty"),
            Self::Metrics(message) => write!(f, "training metrics error: {message}"),
            Self::Checkpoint(message) => write!(f, "checkpoint error: {message}"),
            Self::ShapeMismatch { predicted, target } => {
                write!(
                    f,
                    "prediction/target shape mismatch: {predicted:?} vs {target:?}"
                )
            }
        }
    }
}

impl std::error::Error for TrainingError {}

impl From<VaeError> for TrainingError {
    fn from(error: VaeError) -> Self {
        Self::Vae(error.to_string())
    }
}

pub trait MixedLossBackend: AutodiffBackend {
    fn mixed_l2_loss(prediction: Tensor<Self, 5>, target: Tensor<Self, 5>) -> Tensor<Self, 1>;
}

impl<Inner, C> MixedLossBackend for burn_autodiff::Autodiff<Inner, C>
where
    Inner: burn::tensor::backend::Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn mixed_l2_loss(prediction: Tensor<Self, 5>, target: Tensor<Self, 5>) -> Tensor<Self, 1> {
        let prediction = match prediction.into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized prediction is unsupported"),
        };
        let target = match target.into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized target is unsupported"),
        };
        let loss = burn_autodiff::ops::fused_l2_loss::<Inner, C>(prediction, target);
        Tensor::from_primitive(TensorPrimitive::Float(loss))
    }
}

/// Computes one cached-example diffusion loss.
///
/// The returned scalar remains attached to Burn's autodiff graph, so callers
/// can invoke `loss.backward()` and pass the resulting gradients to an
/// optimizer. Text encoders and VAEs are not involved in this function.
pub fn training_loss<B, A>(
    model: &AnimaDiT<B, A>,
    example: &TrainingExample,
) -> Result<Tensor<B, 1>, TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
{
    training_loss_with_weighting(model, example, LossWeighting::None)
}

fn loss_weight_for_sigma(weighting: LossWeighting, sigma: Option<f32>) -> f32 {
    match weighting {
        LossWeighting::None => 1.0,
        LossWeighting::SigmaSqrt => sigma.map(|value| value.max(1e-6).powi(-2)).unwrap_or(1.0),
        LossWeighting::Cosmap => sigma
            .map(|value| {
                let denominator = 1.0 - 2.0 * value + 2.0 * value * value;
                2.0 / (std::f32::consts::PI * denominator.max(1e-6))
            })
            .unwrap_or(1.0),
    }
}

/// Computes the DIT loss with the weighting schemes used by kohya_ss Anima.
pub fn training_loss_with_weighting<B, A>(
    model: &AnimaDiT<B, A>,
    example: &TrainingExample,
    weighting: LossWeighting,
) -> Result<Tensor<B, 1>, TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
{
    let dump_step = TRAIN_DUMP_STEP.fetch_add(1, Ordering::Relaxed) + 1;
    let device = model.input_projection.weight.val().device();
    // Lifetime: step input. These cached host tensors are uploaded once per
    // step and must die with the step's autodiff graph; the allocator page is
    // expected to be reused by the next upload.
    let latent = example.noisy_latent.to_burn::<5, B>(&device)?;
    let text = example.text_embedding.to_burn::<3, B>(&device)?;
    let text_mask = example
        .text_attention_mask
        .as_ref()
        .map(|mask| {
            let additive = mask.map_values(|value| if value > 0.5 { 0.0 } else { -1.0e9 });
            additive.to_burn::<4, B>(&device)
        })
        .transpose()?;
    // Lifetime: step input, reusable after backward.
    let timestep = example.timestep_features.to_burn::<2, B>(&device)?;
    dump_training_tensor(dump_step, "noisy_latent", &latent);
    dump_training_tensor(dump_step, "text_embedding", &text);
    dump_training_tensor(dump_step, "timestep_features", &timestep);
    let prediction = if let Some(input_ids) = &example.adapter_input_ids {
        let adapter_ids = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(
                input_ids
                    .iter()
                    .map(|value| *value as i32)
                    .collect::<Vec<_>>(),
                [input_ids.len()],
            ),
            &device,
        )
        .reshape([1, input_ids.len()]);
        let adapter_mask = example
            .adapter_attention_mask
            .as_ref()
            .map(|mask| {
                let additive = mask.map_values(|value| if value > 0.5 { 0.0 } else { -1.0e9 });
                additive.to_burn::<4, B>(&device)
            })
            .transpose()?;
        model
            .forward_latent_with_conditioning(
                latent,
                text,
                timestep,
                None,
                text_mask,
                adapter_ids,
                adapter_mask,
            )
            .map_err(TrainingError::Model)?
    } else {
        model
            .forward_latent_with_text_mask(latent, text, timestep, None, text_mask)
            .map_err(TrainingError::Model)?
    };
    // Lifetime: step target, retained until loss reduction/backward completes.
    let target = example.target.to_burn::<5, B>(&device)?;
    dump_training_tensor(dump_step, "target", &target);
    dump_training_tensor(dump_step, "prediction", &prediction);
    if prediction.dims() != target.dims() {
        return Err(TrainingError::ShapeMismatch {
            predicted: prediction.dims().to_vec(),
            target: target.dims().to_vec(),
        });
    }
    let loss = B::mixed_l2_loss(prediction, target);
    dump_training_tensor(dump_step, "loss", &loss);
    let weight = loss_weight_for_sigma(weighting, example.sigma);
    Ok(loss.mul_scalar(weight))
}

/// Applies one optimizer update to the LoRA-oriented model.
///
/// `LoraLinear` detaches its base weight and bias during forward, while the
/// LoRA factors remain attached. Consequently the standard Burn optimizer
/// receives gradients for the adapters without updating the loaded base
/// linear weights.
pub fn optimizer_step<B, A, O>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    loss: Tensor<B, 1>,
    learning_rate: f64,
) -> Result<AnimaDiT<B, A>, TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
{
    optimizer_step_with_config(
        optimizer,
        model,
        loss,
        learning_rate,
        &LoraConfig::default(),
    )
}

/// Applies an optimizer update using only configured DiT LoRA factor IDs.
pub fn optimizer_step_with_config<B, A, O>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    loss: Tensor<B, 1>,
    learning_rate: f64,
    lora: &LoraConfig,
) -> Result<AnimaDiT<B, A>, TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
{
    let debug_phases = std::env::var_os("AFTERBURNER_DEBUG_TRAINING_PHASES").is_some();
    let compact_progress = std::env::var_os("AFTERBURNER_COMPACT_PROGRESS").is_some()
        || std::env::var_os("AFTERBURNER_TRACE").is_none();
    if !compact_progress {
        eprintln!("[progress] phase=backward_begin");
    }
    if debug_phases {
        eprintln!("[training-phase] backward_begin");
    }
    // Keep the full autodiff gradient container alive only while extracting
    // the configured LoRA gradients. Retaining it through `optimizer.step`
    // keeps non-trainable graph buffers reachable across step boundaries on
    // CUDA and causes a nearly constant VRAM increase on every update.
    let gradients = {
        // Do not shrink the loss here.  This is not AMP loss scaling: the
        // previous implementation multiplied the loss by 1/128 but never
        // unscaled the extracted gradients before AdamW.  In BF16 that
        // quantized many small LoRA updates away and made the adapter much
        // weaker than the reference trainer.  Attention and LoRA backward
        // now accumulate in F32, so the true loss must be propagated.
        let mut all_gradients = loss.backward();
        if !compact_progress {
            eprintln!("[progress] phase=backward_complete");
        }
        if debug_phases {
            eprintln!("[training-phase] backward_complete");
        }
        let ids = trainable_lora_ids(&model, lora);
        if std::env::var_os("AFTERBURNER_DEBUG_LORA_GRADIENT").is_some() {
            let id = model.blocks[0]
                .self_attn
                .standard()
                .q_proj
                .lora_up
                .clone()
                .consume()
                .0;
            eprintln!(
                "[lora-gradient] selected_ids={} q_proj_selected={} q_proj_backward={} mlp1_backward={} output_backward={} ",
                ids.len(),
                ids.contains(&id),
                model.blocks[0]
                    .self_attn
                    .standard()
                    .q_proj
                    .lora_up
                    .val()
                    .grad(&all_gradients)
                    .is_some(),
                model.blocks[0]
                    .mlp_layer1
                    .lora_up
                    .val()
                    .grad(&all_gradients)
                    .is_some(),
                model
                    .output_projection
                    .lora_up
                    .val()
                    .grad(&all_gradients)
                    .is_some()
            );
        }
        let raw_gradients = GradientsParams::from_params(&mut all_gradients, &model, &ids);
        if std::env::var_os("AFTERBURNER_TRAIN_DUMP_GRADIENTS").is_some() {
            dump_lora_gradients::<B, A>(
                &model,
                &raw_gradients,
                &lora,
                TRAIN_DUMP_STEP.load(Ordering::Relaxed).max(1),
            );
        }
        // Gradient replay is a diagnostic mode, not part of normal training.
        // A stale environment variable from a previous Python-vs-Rust
        // comparison must never replace the gradients produced by the
        // current backward pass. Require an explicit opt-in in addition to
        // the directory so ordinary runs remain self-contained.
        let gradient_replay_enabled = std::env::var_os("AFTERBURNER_ENABLE_GRADIENT_REPLAY").is_some();
        if !gradient_replay_enabled && std::env::var_os("AFTERBURNER_GRADIENT_REPLAY_DIR").is_some() {
            eprintln!("[gradient-replay] ignored; set AFTERBURNER_ENABLE_GRADIENT_REPLAY=1 to opt in");
        }
        let gradients = if gradient_replay_enabled {
            if let Some(directory) = std::env::var_os("AFTERBURNER_GRADIENT_REPLAY_DIR") {
            load_gradient_replay::<B, A>(
                &model,
                lora,
                std::path::Path::new(&directory),
                &model.input_projection.weight.val().device(),
            )?
            } else {
                return Err(TrainingError::Metrics(
                    "AFTERBURNER_ENABLE_GRADIENT_REPLAY requires AFTERBURNER_GRADIENT_REPLAY_DIR".into(),
                ));
            }
        } else {
            raw_gradients
        };
        let gradients = gradients.clip_by_global_norm::<B::InnerBackend>(1.0);
        if std::env::var_os("AFTERBURNER_DEBUG_LORA_GRADIENT").is_some() {
            for block_index in [0_usize, 13, 26] {
                let id = model.blocks[block_index]
                    .mlp_layer1
                    .lora_up
                    .clone()
                    .consume()
                    .0;
                match gradients
                    .get::<B::InnerBackend, 2>(id)
                    .map(|tensor| tensor.to_data().to_vec::<B::FloatElem>())
                {
                    Some(Ok(values)) => {
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0.0_f64;
                        let mut sum_squared = 0.0_f64;
                        for value in &values {
                            let value = value.elem::<f32>();
                            min = min.min(value);
                            max = max.max(value);
                            sum += f64::from(value);
                            sum_squared += f64::from(value) * f64::from(value);
                        }
                        let count = values.len().max(1) as f64;
                        eprintln!(
                            "[lora-gradient] block={block_index} mlp_layer1.lora_up min={min:.8e} max={max:.8e} mean={:.8e} l2={:.8e}",
                            sum / count,
                            sum_squared.sqrt()
                        );
                    }
                    Some(Err(error)) => eprintln!(
                        "[lora-gradient] block={block_index} mlp_layer1.lora_up read failed: {error:?}"
                    ),
                    None => {
                        eprintln!("[lora-gradient] block={block_index} mlp_layer1.lora_up missing")
                    }
                }
            }
        }
        if !compact_progress {
            eprintln!("[progress] phase=lora_gradient_extract_complete");
        }
        if std::env::var_os("AFTERBURNER_DEBUG_LORA_GRADIENT").is_some() {
            let id = model.blocks[0]
                .self_attn
                .standard()
                .q_proj
                .lora_up
                .clone()
                .consume()
                .0;
            let values = gradients
                .get::<B::InnerBackend, 2>(id)
                .map(|tensor| tensor.to_data().to_vec::<B::FloatElem>());
            match values {
                Some(Ok(values)) => {
                    let (sum, max, nonzero) = values.iter().fold(
                        (0.0_f32, 0.0_f32, 0_usize),
                        |(sum, max, nonzero), value| {
                            let value = value.elem::<f32>().abs();
                            (
                                sum + value,
                                max.max(value),
                                nonzero + usize::from(value != 0.0),
                            )
                        },
                    );
                    let mean = sum / values.len().max(1) as f32;
                    eprintln!(
                        "[lora-gradient] q_proj.lora_up.mean_abs={mean:.6e} max_abs={max:.6e} nonzero={nonzero}/{}",
                        values.len()
                    );
                }
                Some(Err(error)) => {
                    eprintln!("[lora-gradient] failed to read q_proj.lora_up gradient: {error:?}")
                }
                None => eprintln!(
                    "[lora-gradient] missing gradient for blocks.0.self_attn.q_proj.lora_up"
                ),
            }
        }
        // `backward` consumes the graph's server-side steps, but the loss
        // tensor itself still owns the root primitive and node until it is
        // dropped.  Keep it alive through gradient extraction only; retaining
        // it while Adam builds the next parameter/state tensors makes the old
        // graph overlap the new optimizer allocation peak.
        drop(all_gradients);
        gradients
    };
    if gradients.is_empty() {
        return Err(TrainingError::Metrics(
            "no LoRA gradients reached the optimizer".into(),
        ));
    }
    if !compact_progress {
        eprintln!("[progress] phase=optimizer_begin");
    }
    if debug_phases {
        eprintln!("[training-phase] optimizer_begin");
    }
    let model = optimizer.step(learning_rate, model, gradients);
    if !compact_progress {
        eprintln!("[progress] phase=optimizer_complete");
    }
    if debug_phases {
        eprintln!("[training-phase] optimizer_complete");
    }
    Ok(model)
}

struct LoraIdCollector {
    target_modules: Vec<String>,
    ids: Vec<ParamId>,
    path: Vec<String>,
}

/// Convert Burn's reflected module path to the logical LoRA target name.
///
/// CUDA uses `AnimaAttentionCuda`, which contains the common attention
/// implementation under a `standard` field.  The LoRA configuration is
/// intentionally backend-independent (`self_attn.q_proj`, etc.), so that
/// implementation detail must not participate in target matching.
fn lora_module_name(path: &[String]) -> Option<String> {
    let factor = path.last()?;
    if factor != "lora_down" && factor != "lora_up" {
        return None;
    }
    if path.len() < 3 || path[0] != "blocks" {
        return None;
    }

    let mut segments: Vec<&str> = path[2..path.len() - 1]
        .iter()
        .map(String::as_str)
        .filter(|segment| *segment != "standard")
        .collect();
    if segments.len() == 1 {
        segments[0] = match segments[0] {
            "mlp_layer1" => "mlp.layer1",
            "mlp_layer2" => "mlp.layer2",
            value => value,
        };
    }
    Some(segments.join("."))
}

impl<B: burn::tensor::backend::Backend> ModuleVisitor<B> for LoraIdCollector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let path = &self.path;
        if let Some(module) = lora_module_name(path) {
            if !self.target_modules.iter().any(|target| target == &module) {
                return;
            }
            // Preserve Burn's canonical parameter identity. Cloning and
            // consuming the Param is unnecessary here and obscures the fact
            // that GradientsParams filters by the exact ParamId.
            self.ids.push(param.id);
        }
    }

    #[allow(dead_code)]
    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        _tensor: &Tensor<B, D>,
    ) {
        if let Some(module) = lora_module_name(path) {
            if !self.target_modules.iter().any(|target| target == &module) {
                return;
            }
            self.ids.push(id);
        }
    }
}

#[cfg(test)]
mod lora_id_collector_tests {
    use super::lora_module_name;

    fn path(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    #[test]
    fn normalizes_cuda_attention_wrapper() {
        assert_eq!(
            lora_module_name(&path(&[
                "blocks",
                "0",
                "self_attn",
                "standard",
                "q_proj",
                "lora_up",
            ])),
            Some("self_attn.q_proj".to_owned())
        );
    }

    #[test]
    fn preserves_mlp_target_names() {
        assert_eq!(
            lora_module_name(&path(&["blocks", "0", "mlp_layer1", "lora_down"])),
            Some("mlp.layer1".to_owned())
        );
    }
}

fn trainable_lora_ids<B, A>(model: &AnimaDiT<B, A>, lora: &LoraConfig) -> Vec<ParamId>
where
    B: burn::tensor::backend::Backend,
    A: AnimaAttentionLayer<B>,
{
    let mut collector = LoraIdCollector {
        target_modules: lora.target_modules.clone(),
        ids: Vec::new(),
        path: Vec::new(),
    };
    model.visit(&mut collector);
    collector.ids
}

struct LoraGradientCollector {
    target_modules: Vec<String>,
    path: Vec<String>,
    parameters: Vec<(String, ParamId)>,
}

fn lora_gradient_name(path: &[String]) -> Option<String> {
    let block = path.iter().position(|part| part == "blocks")?;
    let mut parts = path[block..]
        .iter()
        .filter(|part| part.as_str() != "standard")
        .cloned()
        .collect::<Vec<_>>();
    let factor = parts.last()?.clone();
    if factor != "lora_up" && factor != "lora_down" {
        return None;
    }
    parts.push("weight".to_owned());
    Some(format!("gradient_lora_unet_{}", parts.join("_")))
}

impl<B: burn::tensor::backend::Backend> ModuleVisitor<B> for LoraGradientCollector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(module) = lora_module_name(&self.path) else {
            return;
        };
        if self.target_modules.iter().any(|target| target == &module) {
            if let Some(name) = lora_gradient_name(&self.path) {
                self.parameters.push((name, param.id));
            }
        }
    }
}

fn dump_lora_gradients<B, A>(
    model: &AnimaDiT<B, A>,
    gradients: &GradientsParams,
    lora: &LoraConfig,
    step: usize,
) where
    B: AutodiffBackend,
    A: AnimaAttentionLayer<B>,
{
    let mut collector = LoraGradientCollector {
        target_modules: lora.target_modules.clone(),
        path: Vec::new(),
        parameters: Vec::new(),
    };
    model.visit(&mut collector);
    eprintln!(
        "[gradient-dump] collected_lora_parameters={}",
        collector.parameters.len()
    );
    for (name, id) in collector.parameters {
        if let Some(gradient) = gradients.get::<B::InnerBackend, 2>(id) {
            dump_training_tensor(step, &name, &gradient);
        } else {
            eprintln!("[gradient-dump] missing_gradient={name}");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingReport {
    pub steps: usize,
    pub last_loss: Option<f32>,
}

/// Exact stochastic inputs exported from the Python probe.
///
/// This is intentionally separate from the preprocessed cache: the cache
/// contains the clean latent, while this replay record contains the sampled
/// sigma and noise used by a particular optimization step.
#[derive(Debug, Clone)]
pub struct ReplayTrainingInput {
    pub sigma: f32,
    pub noise: crate::vae::VaeTensor,
    /// Optional clean latent override from the Python probe. When present,
    /// the replay step bypasses the cached image latent so model comparisons
    /// can isolate the DiT from VAE/cache preprocessing differences.
    pub latent: Option<crate::vae::VaeTensor>,
}

/// Learning-rate policy applied once per optimizer update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRatePolicy {
    /// Keep the configured learning rate unchanged.
    Constant { learning_rate: f64 },
    /// Linearly warm up, then cosine-decay to the minimum learning rate.
    WarmupCosine {
        peak_learning_rate: f64,
        min_learning_rate: f64,
        warmup_steps: usize,
        total_steps: usize,
    },
}

impl LearningRatePolicy {
    pub fn learning_rate(self, step: usize) -> f64 {
        match self {
            Self::Constant { learning_rate } => learning_rate,
            Self::WarmupCosine {
                peak_learning_rate,
                min_learning_rate,
                warmup_steps,
                total_steps,
            } => {
                let total_steps = total_steps.max(1);
                let warmup_steps = warmup_steps.min(total_steps);
                if warmup_steps > 0 && step < warmup_steps {
                    return peak_learning_rate
                        * (step.saturating_add(1) as f64 / warmup_steps as f64);
                }
                let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
                let decay_step = step.saturating_sub(warmup_steps).min(decay_steps);
                let progress = decay_step as f64 / decay_steps as f64;
                min_learning_rate
                    + 0.5
                        * (peak_learning_rate - min_learning_rate)
                        * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }
    }
}

/// Runs a deterministic cache-only LoRA training loop.
pub fn train_cached<B, A, O>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
{
    train_cached_with_target(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate,
        seed,
        TrainingTarget::Epsilon,
    )
}

/// Cache-only training loop with an explicit diffusion prediction target.
pub fn train_cached_with_target<B, A, O>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
{
    train_cached_with_options(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate,
        seed,
        target_kind,
        LoraConfig::default(),
    )
}

/// Cache-only training loop with explicit target and LoRA configuration.
pub fn train_cached_with_options<B, A, O>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
{
    let mut logger = NullLogger;
    train_cached_with_options_and_logger(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate,
        seed,
        target_kind,
        lora,
        &mut logger,
    )
}

/// Cache-only LoRA training with kohya_ss-inspired preparation and step logs.
pub fn train_cached_with_options_and_logger<B, A, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
    L: TrainingLogger,
{
    train_cached_with_options_and_logger_and_checkpoint(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate,
        seed,
        target_kind,
        lora,
        logger,
        None,
        None,
    )
}

/// Cache-only LoRA training with optional periodic model checkpoints.
///
/// The callback receives the current step and updated model after the
/// optimizer update. It can export LoRA factors or another model snapshot.
pub fn train_cached_with_options_and_logger_and_checkpoint<B, A, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B, A>) -> Result<(), TrainingError>>,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
    L: TrainingLogger,
{
    train_cached_with_options_and_logger_and_checkpoint_with_weighting(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate,
        seed,
        target_kind,
        lora,
        logger,
        checkpoint_every,
        checkpoint,
        LossWeighting::None,
    )
}

/// Checkpoint-capable cache training with explicit Anima loss weighting.
pub fn train_cached_with_options_and_logger_and_checkpoint_with_weighting<B, A, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B, A>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
    L: TrainingLogger,
{
    train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        LearningRatePolicy::Constant { learning_rate },
        seed,
        target_kind,
        lora,
        logger,
        checkpoint_every,
        checkpoint,
        loss_weighting,
    )
}

/// Checkpoint-capable cache training with an adaptive learning-rate policy.
pub fn train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy<B, A, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate_policy: LearningRatePolicy,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B, A>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
    L: TrainingLogger,
{
    train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy_and_step_hook(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate_policy,
        seed,
        target_kind,
        lora,
        logger,
        checkpoint_every,
        checkpoint,
        loss_weighting,
        None,
    )
}

/// Checkpoint-capable cache training with a per-step resource hook.
pub fn train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy_and_step_hook<
    B,
    A,
    O,
    L,
>(
    optimizer: &mut O,
    mut model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate_policy: LearningRatePolicy,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    mut checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B, A>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
    mut after_step: Option<&mut dyn FnMut(usize, &'static str)>,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
    L: TrainingLogger,
{
    train_cached_with_replay(
        optimizer,
        model,
        cache,
        schedule,
        steps,
        learning_rate_policy,
        seed,
        target_kind,
        TimestepSampling::Sigmoid {
            mean: 0.0,
            std: 1.0,
        },
        lora,
        logger,
        checkpoint_every,
        checkpoint,
        loss_weighting,
        after_step,
        None,
    )
}

/// Training entry point with optional Python stochastic-input replay.
pub fn train_cached_with_replay<B, A, O, L>(
    optimizer: &mut O,
    mut model: AnimaDiT<B, A>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate_policy: LearningRatePolicy,
    seed: u64,
    target_kind: TrainingTarget,
    timestep_sampling: TimestepSampling,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    mut checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B, A>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
    mut after_step: Option<&mut dyn FnMut(usize, &'static str)>,
    replay: Option<&[ReplayTrainingInput]>,
) -> Result<(AnimaDiT<B, A>, TrainingReport), TrainingError>
where
    B: AutodiffBackend + MixedLossBackend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B> + burn::module::AutodiffModule<B>,
    A::InnerModule: burn::module::ModuleDisplay,
    O: Optimizer<AnimaDiT<B, A>, B>,
    L: TrainingLogger,
{
    if cache.is_empty() {
        return Err(TrainingError::EmptyCache);
    }
    cache.validate_anima_shapes()?;
    let device = model.input_projection.weight.val().device();
    logger.info("Network train mode: UNet-only (text encoder not trained; cached inputs only).");
    logger.info(&format!(
        "running training / 学習開始\n  num train images * repeats: {}\n  batch size per device: 1\n  total optimization steps: {}",
        cache.len(), steps
    ));
    let mut rng = seed;
    let mut last_loss = None;
    let mut loss_window = VecDeque::with_capacity(20);
    let debug_param_ids = std::env::var_os("AFTERBURNER_DEBUG_PARAM_IDS").is_some();
    let memory_trace = std::env::var_os("AFTERBURNER_MEMORY_TRACE").is_some();
    let timing_trace = std::env::var_os("AFTERBURNER_TIMING").is_some();
    // Synchronizing and sweeping CubeCL's allocator every step is useful for
    // leak diagnostics but stalls the GPU. Keep the safe legacy default while
    // allowing performance runs to sweep less often (0 means final step only).
    let memory_cleanup_every = std::env::var("AFTERBURNER_MEMORY_CLEANUP_EVERY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let retain_memory_pools = std::env::var_os("AFTERBURNER_RETAIN_MEMORY_POOLS").is_some();
    let mut previous_param_ids = None::<Vec<ParamId>>;
    for step in 0..steps {
        let step_started = std::time::Instant::now();
        let sample = cache
            .get(step % cache.len())
            .expect("cache was checked non-empty");
        let replay_input = replay.and_then(|inputs| inputs.get(step));
        let replay_sample = replay_input.and_then(|input| input.latent.as_ref()).map(|latent| {
            let mut sample = sample.clone();
            sample.image_latent = latent.clone();
            sample
        });
        let sample = replay_sample.as_ref().unwrap_or(sample);
        let flow_sigma = if let Some(input) = replay_input {
            Some(input.sigma)
        } else if target_kind == TrainingTarget::FlowMatching {
            Some(
                schedule
                    .sample_sigma(
                        random_unit(&mut rng),
                        random_unit(&mut rng),
                        timestep_sampling,
                    )
                    .map_err(|error| TrainingError::Vae(error.to_string()))?,
            )
        } else {
            None
        };
        let noise = if let Some(input) = replay_input {
            if input.noise.shape != sample.image_latent.shape {
                return Err(TrainingError::Vae(format!(
                    "replay noise shape {:?} does not match cache shape {:?}",
                    input.noise.shape, sample.image_latent.shape
                )));
            }
            input.noise.clone()
        } else {
            gaussian_like(&sample.image_latent.shape, &mut rng)?
        };
        if let Some(sigma) = flow_sigma {
            dump_replay_input(step + 1, sigma, &noise, &sample.image_latent);
        }
        let example = if let Some(sigma) = flow_sigma {
            // Keep the shifted continuous sigma for both the noisy latent and
            // timestep embedding, matching kohya_ss's sigma sampling path.
            TrainingExample::from_cache_with_sigma(sample, sigma, schedule.steps(), noise)
        } else {
            let timestep = next_random(&mut rng) % schedule.steps();
            TrainingExample::from_cache_with_target(sample, schedule, timestep, noise, target_kind)
        }
        .map_err(|error| TrainingError::Vae(error.to_string()))?;
        if std::env::var_os("AFTERBURNER_TRAIN_DUMP_DIR").is_some() {
            let clean_latent = sample.image_latent.to_burn::<5, B>(&device)?;
            dump_training_tensor(step + 1, "clean_latent", &clean_latent);
        }
        if std::env::var_os("AFTERBURNER_COMPACT_PROGRESS").is_none() {
            logger.info(&format!(
                "[progress] step={}/{} phase=forward_begin",
                step + 1,
                steps
            ));
        }
        if memory_trace {
            logger.info(&format!(
                "[memory-phase] step={} phase=forward_begin",
                step + 1
            ));
        }
        if std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some() {
            logger.info(&format!(
                "[debug] sample={} latent={:?} text={:?} timestep={:?} adapter_ids={}",
                step % cache.len(),
                example.noisy_latent.shape,
                example.text_embedding.shape,
                example.timestep_features.shape,
                example.adapter_input_ids.is_some()
            ));
        }
        if std::env::var_os("AFTERBURNER_DEBUG_TRAINING_INPUTS").is_some() {
            let noisy = value_stats(&example.noisy_latent.data);
            let target_stats = value_stats(&example.target.data);
            let time_stats = value_stats(&example.timestep_features.data);
            let input_source = if replay_input.is_some() {
                "python-replay"
            } else {
                "rust-sampled"
            };
            logger.info(&format!(
                "[training-input] step={} source={} sigma={:?} timestep={} noisy_latent(min={:.8e},max={:.8e},mean_abs={:.8e},l2={:.8e}) target(min={:.8e},max={:.8e},mean_abs={:.8e},l2={:.8e}) timestep_features(min={:.8e},max={:.8e},mean_abs={:.8e},l2={:.8e})",
                step + 1,
                input_source,
                example.sigma,
                example.timestep,
                noisy.0,
                noisy.1,
                noisy.2,
                noisy.3,
                target_stats.0,
                target_stats.1,
                target_stats.2,
                target_stats.3,
                time_stats.0,
                time_stats.1,
                time_stats.2,
                time_stats.3,
            ));
        }
        // Lifetime: autodiff graph root for this step. It survives through
        // backward/optimizer, then must be dropped before the next sample.
        let loss = training_loss_with_weighting(&model, &example, loss_weighting)?;
        if timing_trace {
            logger.info(&format!(
                "[timing] step={} forward_loss={:.3}s",
                step + 1,
                step_started.elapsed().as_secs_f64()
            ));
        }
        if memory_trace {
            logger.info(&format!(
                "[memory-phase] step={} phase=forward_loss_complete",
                step + 1
            ));
        }
        if std::env::var_os("AFTERBURNER_COMPACT_PROGRESS").is_none() {
            logger.info(&format!(
                "[progress] step={}/{} phase=forward_loss_complete elapsed={:.3}s",
                step + 1,
                steps,
                step_started.elapsed().as_secs_f64()
            ));
        }
        if let Some(hook) = after_step.as_mut() {
            hook(step + 1, "forward_loss_complete");
        }
        if std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some() {
            logger.info("[debug] loss forward complete");
        }
        let effective_learning_rate = learning_rate_policy.learning_rate(step);
        if debug_param_ids {
            let ids = trainable_lora_ids(&model, &lora);
            let changed = previous_param_ids
                .as_ref()
                .map(|previous| {
                    previous.len() != ids.len()
                        || previous.iter().zip(&ids).any(|(old, new)| old != new)
                })
                .unwrap_or(false);
            let first = ids.first().map(ParamId::val);
            let last = ids.last().map(ParamId::val);
            logger.info(&format!(
                "[param-id] step={} count={} changed={} first={first:?} last={last:?}",
                step + 1,
                ids.len(),
                changed
            ));
            previous_param_ids = Some(ids);
        }
        let optimizer_started = std::time::Instant::now();
        if std::env::var_os("AFTERBURNER_COMPACT_PROGRESS").is_none() {
            logger.info(&format!(
                "[progress] step={}/{} phase=backward_optimizer_begin",
                step + 1,
                steps
            ));
        }
        if memory_trace {
            logger.info(&format!(
                "[memory-phase] step={} phase=backward_optimizer_begin",
                step + 1
            ));
        }
        if std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some() {
            logger.info("[debug] backward and optimizer started");
        }
        let loss_handle = loss.clone();
        model = optimizer_step_with_config(optimizer, model, loss, effective_learning_rate, &lora)?;
        let loss_value = loss_handle
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| TrainingError::Metrics(format!("{error:?}")))?
            .first()
            .copied()
            .unwrap_or_default();
        last_loss = Some(loss_value);
        loss_window.push_back(loss_value);
        if loss_window.len() > 20 {
            loss_window.pop_front();
        }
        let loss_average = loss_window.iter().sum::<f32>() / loss_window.len() as f32;
        logger.info(&format!(
            "steps: {}/{} ({:.0}%), loss={loss_value:.6}, loss_avg20={loss_average:.6}, lr={effective_learning_rate:.3e}",
            step + 1,
            steps,
            ((step + 1) as f32 / steps.max(1) as f32) * 100.0
        ));
        if std::env::var_os("AFTERBURNER_COMPACT_PROGRESS").is_none() {
            logger.info(&format!(
                "[progress] step={}/{} phase=backward_optimizer_complete elapsed={:.3}s",
                step + 1,
                steps,
                optimizer_started.elapsed().as_secs_f64()
            ));
        }
        if let Some(hook) = after_step.as_mut() {
            hook(step + 1, "backward_optimizer_complete");
        }
        if timing_trace {
            logger.info(&format!(
                "[timing] step={} backward_optimizer={:.3}s total_before_cleanup={:.3}s",
                step + 1,
                optimizer_started.elapsed().as_secs_f64(),
                step_started.elapsed().as_secs_f64()
            ));
        }
        if std::env::var_os("AFTERBURNER_DEBUG_LORA_UPDATE").is_some() {
            let values = model.blocks[0]
                .self_attn
                .standard()
                .q_proj
                .lora_up
                .val()
                .to_data()
                .to_vec::<B::FloatElem>()
                .map_err(|error| TrainingError::Metrics(format!("{error:?}")))?;
            let (sum, max) = values.iter().fold((0.0_f32, 0.0_f32), |(sum, max), value| {
                let value = value.elem::<f32>().abs();
                (sum + value, max.max(value))
            });
            let mean = sum / values.len().max(1) as f32;
            logger.info(&format!(
                "[lora-update] step={} q_proj.lora_up.mean_abs={mean:.6e} max_abs={max:.6e}",
                step + 1
            ));
        }
        // Match the step-boundary graph cleanup used by the reference PyTorch
        // trainer. This makes dead tensors eligible for reuse. In retained-pool
        // mode, do not destroy the CUDA pages: they are the next step's
        // overwrite/reuse workspace, not a leak.
        drop(example);
        let cleanup_due = match memory_cleanup_every {
            Some(0) => step + 1 == steps,
            Some(every) => (step + 1) % every == 0 || step + 1 == steps,
            None => step + 1 == steps,
        };
        // Ensure the old graph is detached every step. Delaying this along
        // with allocator sweeping makes the graph itself accumulate between
        // sweeps, producing a large allocation burst at the interval.
        burn_autodiff::cleanup_dead_nodes();
        if cleanup_due {
            // Synchronization is deliberately less frequent than graph
            // cleanup. A device-wide sync on every step serializes the CUDA
            // queue and leaves the GPU idle while the host waits.
            B::sync(&device);
            if !retain_memory_pools
                && std::env::var_os("AFTERBURNER_DISABLE_MEMORY_CLEANUP").is_none()
            {
                B::memory_cleanup(&device);
            }
        }
        if memory_trace {
            logger.info(&format!(
                "[memory-phase] step={} phase=step_cleanup_complete",
                step + 1
            ));
        }
        if let Some(hook) = after_step.as_mut() {
            hook(step + 1, "step_cleanup_complete");
        }
        if std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some() {
            logger.info(&format!(
                "[debug] optimizer step complete in {:.3}s",
                optimizer_started.elapsed().as_secs_f64()
            ));
        }
        if let (Some(interval), Some(callback)) = (checkpoint_every, checkpoint.as_mut()) {
            if interval > 0 && (step + 1) % interval == 0 {
                callback(step + 1, &model)?;
            }
        }
    }
    logger.info(&format!(
        "training completed: steps={}, last_loss={:?}",
        steps, last_loss
    ));
    Ok((model, TrainingReport { steps, last_loss }))
}

fn next_random(state: &mut u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as usize
}

fn value_stats(values: &[f32]) -> (f32, f32, f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut abs_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    for &value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        abs_sum += f64::from(value.abs());
        squared_sum += f64::from(value) * f64::from(value);
    }
    let count = values.len().max(1) as f64;
    (
        minimum,
        maximum,
        (abs_sum / count) as f32,
        squared_sum.sqrt() as f32,
    )
}

fn random_unit(state: &mut u64) -> f32 {
    (next_random(state) as f32 + 1.0) / (u32::MAX as f32 + 2.0)
}

fn gaussian_like(shape: &[usize], state: &mut u64) -> Result<crate::vae::VaeTensor, TrainingError> {
    let count = shape.iter().product();
    let mut data = Vec::with_capacity(count);
    while data.len() < count {
        let u1 = (next_random(state) as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let u2 = (next_random(state) as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * u2;
        data.push(radius * angle.cos());
        if data.len() < count {
            data.push(radius * angle.sin());
        }
    }
    crate::vae::VaeTensor::new(shape.to_vec(), data).map_err(TrainingError::from)
}

#[cfg(test)]
mod tests {
    use super::LearningRatePolicy;
    use super::loss_weight_for_sigma;
    use burn::tensor::{Tensor, TensorData};
    use burn_autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};
    use burn_ndarray::NdArray;

    use crate::diffusion::{LossWeighting, timestep_embedding};

    #[test]
    fn timestep_features_are_finite_and_wide_enough() {
        let features = timestep_embedding(10, 100, 2048);
        assert_eq!(features.shape, vec![1, 2048]);
        assert!(features.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn kohya_cosine_schedule_matches_warmup_ratio_reference() {
        let policy = LearningRatePolicy::WarmupCosine {
            peak_learning_rate: 1.0e-4,
            min_learning_rate: 0.0,
            warmup_steps: 30,
            total_steps: 300,
        };
        assert!((policy.learning_rate(0) - (1.0e-4 / 30.0)).abs() < 1e-12);
        assert!((policy.learning_rate(29) - 1.0e-4).abs() < 1e-12);
        assert!((policy.learning_rate(30) - 1.0e-4).abs() < 1e-12);
        assert!(policy.learning_rate(150) < 1.0e-4);
        // The final optimizer update is the last point before the cosine
        // endpoint; the scheduler reaches the exact minimum on the next
        // scheduler tick, not during step 299 itself.
        assert!(policy.learning_rate(299) > 0.0);
        assert!(policy.learning_rate(299) < 1.0e-7);
    }

    #[test]
    fn kohya_ninety_step_schedule_leaves_warmup_after_step_nine() {
        let policy = LearningRatePolicy::WarmupCosine {
            peak_learning_rate: 1.0e-4,
            min_learning_rate: 0.0,
            warmup_steps: 9,
            total_steps: 90,
        };
        assert!((policy.learning_rate(0) - 1.0e-4 / 9.0).abs() < 1.0e-12);
        assert!((policy.learning_rate(8) - 1.0e-4).abs() < 1.0e-12);
        assert!((policy.learning_rate(9) - 1.0e-4).abs() < 1.0e-12);
        assert!(policy.learning_rate(10) < policy.learning_rate(9));
    }

    #[test]
    fn memory_bound_matmul_recomputes_without_changing_gradients() {
        type Backend = Autodiff<NdArray<f32>, BalancedCheckpointing>;
        let device = Default::default();
        let lhs = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        )
        .require_grad();
        let rhs = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![5.0_f32, 6.0, 7.0, 8.0], [2, 2]),
            &device,
        )
        .require_grad();
        let loss = lhs.clone().matmul(rhs.clone()).sum();
        let gradients = loss.backward();
        let lhs_grad = lhs
            .grad(&gradients)
            .expect("missing lhs gradient")
            .to_data()
            .to_vec::<f32>()
            .expect("invalid lhs gradient data");
        let rhs_grad = rhs
            .grad(&gradients)
            .expect("missing rhs gradient")
            .to_data()
            .to_vec::<f32>()
            .expect("invalid rhs gradient data");
        assert_eq!(lhs_grad, vec![11.0, 15.0, 11.0, 15.0]);
        assert_eq!(rhs_grad, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn mixed_l2_loss_preserves_prediction_gradient() {
        type Backend = Autodiff<NdArray<f32>, BalancedCheckpointing>;
        let device = Default::default();
        let prediction = Tensor::<Backend, 5>::ones([1, 1, 1, 2, 2], &device).require_grad();
        let target = Tensor::<Backend, 5>::zeros([1, 1, 1, 2, 2], &device);
        let loss = <Backend as super::MixedLossBackend>::mixed_l2_loss(
            prediction.clone(),
            target,
        );
        let gradients = loss.backward();
        let gradient = prediction
            .grad(&gradients)
            .expect("mixed L2 loss must return a prediction gradient")
            .into_data()
            .to_vec::<f32>()
            .expect("invalid mixed L2 gradient data");
        assert!(gradient.iter().all(|value| (*value - 0.5).abs() < 1e-6));
    }

    #[test]
    fn kohya_probe_step_three_mse_prefix_matches_fp32_reduction() {
        // Values are the first four BF16-rounded elements from the existing
        // Kohya step-000003 prediction and target dumps.
        let prediction = [
            -0.96484375_f32,
            0.68701171875,
            0.177978515625,
            -0.007617950439453125,
        ];
        let target = [
            -0.8974609375_f32,
            0.7080078125,
            0.241455078125,
            -0.08056640625,
        ];
        let mse = prediction
            .iter()
            .zip(target)
            .map(|(predicted, expected)| (predicted - expected).powi(2))
            .sum::<f32>()
            / prediction.len() as f32;
        assert!((mse - 0.0035830076).abs() < 1e-7);
    }

    #[test]
    fn kohya_anima_loss_weighting_matches_reference_formulas() {
        let sigma = 0.40445560216903687_f32;
        assert_eq!(loss_weight_for_sigma(LossWeighting::None, Some(sigma)), 1.0);
        assert!(
            (loss_weight_for_sigma(LossWeighting::SigmaSqrt, Some(sigma)) - sigma.powi(-2)).abs()
                < 1e-6
        );
        let denominator = 1.0 - 2.0 * sigma + 2.0 * sigma * sigma;
        let expected_cosmap = 2.0 / (std::f32::consts::PI * denominator);
        assert!(
            (loss_weight_for_sigma(LossWeighting::Cosmap, Some(sigma)) - expected_cosmap).abs()
                < 1e-6
        );
    }
}
