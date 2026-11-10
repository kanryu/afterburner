//! Burn training-step glue for cached diffusion examples.

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData, backend::AutodiffBackend};
use burn_optim::{GradientsParams, Optimizer};

use crate::{
    cache::InMemoryTrainingCache,
    diffusion::{
        DiffusionSchedule, LossWeighting, TimestepSampling, TrainingExample, TrainingTarget,
    },
    dit::{AnimaDiT, AnimaTensorError},
    logging::{NullLogger, TrainingLogger},
    lora::LoraConfig,
    vae::VaeError,
};

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

/// Computes one cached-example diffusion loss.
///
/// The returned scalar remains attached to Burn's autodiff graph, so callers
/// can invoke `loss.backward()` and pass the resulting gradients to an
/// optimizer. Text encoders and VAEs are not involved in this function.
pub fn training_loss<B: AutodiffBackend>(
    model: &AnimaDiT<B>,
    example: &TrainingExample,
) -> Result<Tensor<B, 1>, TrainingError> {
    training_loss_with_weighting(model, example, LossWeighting::None)
}

/// Computes the DIT loss with the weighting schemes used by kohya_ss Anima.
pub fn training_loss_with_weighting<B: AutodiffBackend>(
    model: &AnimaDiT<B>,
    example: &TrainingExample,
    weighting: LossWeighting,
) -> Result<Tensor<B, 1>, TrainingError> {
    let device = model.input_projection.weight.val().device();
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
    let timestep = example.timestep_features.to_burn::<2, B>(&device)?;
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
    let target = example.target.to_burn::<5, B>(&device)?;
    if prediction.dims() != target.dims() {
        return Err(TrainingError::ShapeMismatch {
            predicted: prediction.dims().to_vec(),
            target: target.dims().to_vec(),
        });
    }
    let loss = prediction.sub(target).square().mean();
    let weight = match weighting {
        LossWeighting::None => 1.0,
        LossWeighting::SigmaSqrt => example
            .sigma
            .map(|sigma| sigma.max(1e-6).powi(-2))
            .unwrap_or(1.0),
        LossWeighting::Cosmap => example
            .sigma
            .map(|sigma| {
                let denominator = 1.0 - 2.0 * sigma + 2.0 * sigma * sigma;
                2.0 / (std::f32::consts::PI * denominator.max(1e-6))
            })
            .unwrap_or(1.0),
    };
    Ok(loss.mul_scalar(weight))
}

/// Applies one optimizer update to the LoRA-oriented model.
///
/// `LoraLinear` detaches its base weight and bias during forward, while the
/// LoRA factors remain attached. Consequently the standard Burn optimizer
/// receives gradients for the adapters without updating the loaded base
/// linear weights.
pub fn optimizer_step<B, O>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    loss: Tensor<B, 1>,
    learning_rate: f64,
) -> AnimaDiT<B>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn optimizer_step_with_config<B, O>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    loss: Tensor<B, 1>,
    learning_rate: f64,
    lora: &LoraConfig,
) -> AnimaDiT<B>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
{
    // Keep the full autodiff gradient container alive only while extracting
    // the configured LoRA gradients. Retaining it through `optimizer.step`
    // keeps non-trainable graph buffers reachable across step boundaries on
    // CUDA and causes a nearly constant VRAM increase on every update.
    let gradients = {
        let mut all_gradients = loss.backward();
        let ids = trainable_lora_ids(&model, lora);
        GradientsParams::from_params(&mut all_gradients, &model, &ids)
    };
    optimizer.step(learning_rate, model, gradients)
}

struct LoraIdCollector {
    target_modules: Vec<String>,
    ids: Vec<ParamId>,
    path: Vec<String>,
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
        let Some(factor) = path.last() else {
            return;
        };
        if factor != "lora_down" && factor != "lora_up" {
            return;
        }
        if path.len() < 3 || path[0] != "blocks" {
            return;
        }
        let module = match path[2..path.len() - 1].join(".").as_str() {
            "mlp_layer1" => "mlp.layer1".to_owned(),
            "mlp_layer2" => "mlp.layer2".to_owned(),
            value => value.to_owned(),
        };
        if self.target_modules.iter().any(|target| target == &module) {
            self.ids.push(param.clone().consume().0);
        }
    }

    #[allow(dead_code)]
    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        _tensor: &Tensor<B, D>,
    ) {
        let Some(factor) = path.last() else {
            return;
        };
        if factor != "lora_down" && factor != "lora_up" {
            return;
        }
        if path.len() < 3 || path[0] != "blocks" {
            return;
        }
        let module = match path[2..path.len() - 1].join(".").as_str() {
            "mlp_layer1" => "mlp.layer1".to_owned(),
            "mlp_layer2" => "mlp.layer2".to_owned(),
            value => value.to_owned(),
        };
        if self.target_modules.iter().any(|target| target == &module) {
            self.ids.push(id);
        }
    }
}

fn trainable_lora_ids<B: burn::tensor::backend::Backend>(
    model: &AnimaDiT<B>,
    lora: &LoraConfig,
) -> Vec<ParamId> {
    let mut collector = LoraIdCollector {
        target_modules: lora.target_modules.clone(),
        ids: Vec::new(),
        path: Vec::new(),
    };
    model.visit(&mut collector);
    collector.ids
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingReport {
    pub steps: usize,
    pub last_loss: Option<f32>,
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
pub fn train_cached<B, O>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn train_cached_with_target<B, O>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn train_cached_with_options<B, O>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn train_cached_with_options_and_logger<B, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn train_cached_with_options_and_logger_and_checkpoint<B, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B>) -> Result<(), TrainingError>>,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn train_cached_with_options_and_logger_and_checkpoint_with_weighting<B, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate: f64,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
pub fn train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy<B, O, L>(
    optimizer: &mut O,
    model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate_policy: LearningRatePolicy,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
    O,
    L,
>(
    optimizer: &mut O,
    mut model: AnimaDiT<B>,
    cache: &InMemoryTrainingCache,
    schedule: &DiffusionSchedule,
    steps: usize,
    learning_rate_policy: LearningRatePolicy,
    seed: u64,
    target_kind: TrainingTarget,
    lora: LoraConfig,
    logger: &mut L,
    checkpoint_every: Option<usize>,
    mut checkpoint: Option<&mut dyn FnMut(usize, &AnimaDiT<B>) -> Result<(), TrainingError>>,
    loss_weighting: LossWeighting,
    mut after_step: Option<&mut dyn FnMut(usize)>,
) -> Result<(AnimaDiT<B>, TrainingReport), TrainingError>
where
    B: AutodiffBackend,
    O: Optimizer<AnimaDiT<B>, B>,
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
    let debug_param_ids = std::env::var_os("AFTERBURNER_DEBUG_PARAM_IDS").is_some();
    let memory_trace = std::env::var_os("AFTERBURNER_MEMORY_TRACE").is_some();
    let mut previous_param_ids = None::<Vec<ParamId>>;
    for step in 0..steps {
        let sample = cache
            .get(step % cache.len())
            .expect("cache was checked non-empty");
        let flow_sigma = if target_kind == TrainingTarget::FlowMatching {
            Some(
                schedule
                    .sample_sigma(
                        random_unit(&mut rng),
                        random_unit(&mut rng),
                        TimestepSampling::Sigmoid {
                            mean: 0.0,
                            std: 1.0,
                        },
                    )
                    .map_err(|error| TrainingError::Vae(error.to_string()))?,
            )
        } else {
            None
        };
        let noise = gaussian_like(&sample.image_latent.shape, &mut rng)?;
        let example = if let Some(sigma) = flow_sigma {
            TrainingExample::from_cache_with_sigma(sample, sigma, schedule.steps(), noise)
        } else {
            let timestep = next_random(&mut rng) % schedule.steps();
            TrainingExample::from_cache_with_target(sample, schedule, timestep, noise, target_kind)
        }
        .map_err(|error| TrainingError::Vae(error.to_string()))?;
        if memory_trace {
            logger.info(&format!("[memory-phase] step={} phase=forward_begin", step + 1));
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
        let loss = training_loss_with_weighting(&model, &example, loss_weighting)?;
        if memory_trace {
            logger.info(&format!("[memory-phase] step={} phase=forward_loss_complete", step + 1));
        }
        if std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some() {
            logger.info("[debug] loss forward complete");
        }
        let values = loss
            .to_data()
            .to_vec::<<B as burn::tensor::backend::Backend>::FloatElem>()
            .map_err(|error| TrainingError::Metrics(format!("{error:?}")))?;
        last_loss = values.first().map(|value| value.elem::<f32>());
        let loss_value = last_loss.unwrap_or_default();
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
        logger.info(&format!(
            "steps: {}/{} ({:.0}%), avr_loss={loss_value:.3}, lr={effective_learning_rate:.3e}",
            step + 1,
            steps,
            ((step + 1) as f32 / steps.max(1) as f32) * 100.0
        ));
        let optimizer_started = std::time::Instant::now();
        if memory_trace {
            logger.info(&format!("[memory-phase] step={} phase=backward_optimizer_begin", step + 1));
        }
        if std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some() {
            logger.info("[debug] backward and optimizer started");
        }
        model = optimizer_step_with_config(optimizer, model, loss, effective_learning_rate, &lora);
        // Match the step-boundary cleanup used by the reference PyTorch
        // trainer. This gives the backend a chance to release allocator-managed
        // buffers that are no longer reachable after backward. It cannot force
        // the allocator to return pages that still have live tensor handles.
        drop(example);
        // Ensure asynchronous CUDA work has completed before asking the
        // backend allocator to reclaim transient buffers. Without the sync,
        // the allocator can report one step's activation workspace as still
        // live and appear to grow monotonically.
        B::sync(&device);
        if std::env::var_os("AFTERBURNER_DISABLE_MEMORY_CLEANUP").is_none() {
            B::memory_cleanup(&device);
        }
        if memory_trace {
            logger.info(&format!("[memory-phase] step={} phase=step_cleanup_complete", step + 1));
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
        if let Some(hook) = after_step.as_mut() {
            hook(step + 1);
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
    use burn::tensor::{Tensor, TensorData};
    use burn_autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};
    use burn_ndarray::NdArray;

    use crate::diffusion::timestep_embedding;

    #[test]
    fn timestep_features_are_finite_and_wide_enough() {
        let features = timestep_embedding(10, 100, 2048);
        assert_eq!(features.shape, vec![1, 2048]);
        assert!(features.data.iter().all(|value| value.is_finite()));
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
}
