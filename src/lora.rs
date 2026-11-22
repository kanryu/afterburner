//! LoRA target selection and adapter sizing.
//!
//! The computation graph will be attached to Burn modules after the Anima
//! DiT blocks are implemented. This module first makes target selection
//! explicit and reproducible from SafeTensors keys.

use crate::anima::AnimaConfig;
use burn::{
    module::{Initializer, Module, Param},
    tensor::{DType, ElementConversion, Tensor, TensorData, TensorPrimitive, backend::Backend, module::linear},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTargetKind {
    Linear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: usize,
    pub target_modules: Vec<String>,
    pub train_bias: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16,
            target_modules: AnimaConfig::default()
                .lora_targets()
                .into_iter()
                .map(str::to_owned)
                .collect(),
            train_bias: false,
        }
    }
}

impl LoraConfig {
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.rank == 0 {
            return Err(LoraError::InvalidConfig(
                "rank must be greater than zero".into(),
            ));
        }
        if self.alpha == 0 {
            return Err(LoraError::InvalidConfig(
                "alpha must be greater than zero".into(),
            ));
        }
        if self.target_modules.is_empty() {
            return Err(LoraError::InvalidConfig(
                "at least one target module is required".into(),
            ));
        }
        Ok(())
    }

    pub fn scale(&self) -> f32 {
        self.alpha as f32 / self.rank as f32
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    InvalidConfig(String),
    InvalidWeightShape { key: String, shape: Vec<usize> },
}

impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid LoRA configuration: {message}"),
            Self::InvalidWeightShape { key, shape } => {
                write!(
                    f,
                    "LoRA target {key} must be a rank-2 weight, got {shape:?}"
                )
            }
        }
    }
}

impl std::error::Error for LoraError {}

/// A selected rank-2 weight and the adapter dimensions that wrap it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraTarget {
    pub key: String,
    pub kind: LoraTargetKind,
    pub output_dim: usize,
    pub input_dim: usize,
    pub rank: usize,
    pub alpha: usize,
}

/// Configuration for a trainable LoRA-wrapped Linear layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraLinearConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    pub alpha: usize,
    pub bias: bool,
}

impl LoraLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<LoraLinear<B>, LoraError> {
        if self.input_dim == 0 || self.output_dim == 0 || self.rank == 0 {
            return Err(LoraError::InvalidConfig(
                "input_dim, output_dim, and rank must be greater than zero".into(),
            ));
        }
        if self.alpha == 0 {
            return Err(LoraError::InvalidConfig(
                "alpha must be greater than zero".into(),
            ));
        }
        let lora_initializer = Initializer::KaimingUniform {
            gain: 1.0 / 3.0_f64.sqrt(),
            fan_out_only: false,
        };
        // Base weights are populated immediately from SafeTensors by the
        // model loader. Avoid launching random-generation kernels for the
        // multi-billion-parameter base model on GPU backends; only the
        // trainable LoRA branch needs a random initialization here.
        let base_initializer = Initializer::Zeros;
        let bias = self.bias.then(|| {
            Initializer::Zeros
                .init([self.output_dim], device)
                .map(|tensor| tensor.detach())
        });
        let down_seed = lora_initializer
            .init_with::<B, 2, _>(
                [self.rank, self.input_dim],
                Some(self.input_dim),
                Some(self.rank),
                device,
            );
        // Python kohya keeps trainable LoRA parameters in FP32 when
        // mixed_precision=fp16 and full_bf16/full_fp16 are disabled.  A plain
        // BF16->F32 cast would create a non-leaf graph tensor, so materialize
        // a fresh F32 TensorData leaf instead.
        let down_data = down_seed.val().into_data();
        let down_values = down_data
            .to_vec::<B::FloatElem>()
            .map_err(|error| LoraError::InvalidConfig(format!("LoRA init dtype conversion failed: {error:?}")))?
            .into_iter()
            .map(|value| value.elem::<f32>())
            .collect::<Vec<_>>();
        // Lifetime: trainable parameter and optimizer state; persistent for
        // the entire run, never step-local workspace.
        let lora_down = Tensor::<B, 2>::from_data(
            TensorData::new(down_values, [self.rank, self.input_dim]),
            device,
        );
        let lora_down = Tensor::<B, 2>::cast(lora_down, lora_parameter_dtype());
        // Lifetime: trainable parameter and optimizer state; persistent for
        // the entire run, never step-local workspace.
        let lora_up = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.0_f32; self.output_dim * self.rank], [self.output_dim, self.rank]),
            device,
        );
        let lora_up = Tensor::<B, 2>::cast(lora_up, lora_parameter_dtype());
        Ok(LoraLinear {
            weight: base_initializer
                .init_with(
                    [self.output_dim, self.input_dim],
                    Some(self.input_dim),
                    Some(self.output_dim),
                    device,
                )
                .map(|tensor| tensor.detach()),
            bias,
            lora_down: Param::from_tensor(lora_down),
            lora_up: Param::from_tensor(lora_up),
            rank: self.rank,
            alpha: self.alpha,
        })
    }
}

fn lora_parameter_dtype() -> DType {
    if std::env::var_os("AFTERBURNER_LORA_BF16").is_some() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Burn-native Linear + LoRA adapter.
///
/// The base weight is stored as `[output, input]`, matching the inspected
/// Anima SafeTensors layout. The adapter computes `W x + (alpha / rank) B A x`.
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub lora_down: Param<Tensor<B, 2>>,
    pub lora_up: Param<Tensor<B, 2>>,
    pub rank: usize,
    pub alpha: usize,
}

impl<B: Backend> LoraLinear<B> {
    pub fn forward_low_precision<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Use the selected backend precision. The previous BF16 literal made
        // an FP16 reference run silently execute all projections in BF16.
        let input_dtype = input.dtype();
        let input = input.cast(input_dtype);
        let base = linear(
            input.clone(),
            self.weight.val().detach().cast(input_dtype).transpose(),
            self.bias
                .as_ref()
                .map(|bias| bias.val().detach().cast(input_dtype)),
        );
        let delta = linear(
            linear(
                input,
                self.lora_down.val().cast(input_dtype).transpose(),
                None,
            ),
            self.lora_up.val().cast(input_dtype).transpose(),
            None,
        )
        .mul_scalar(self.alpha as f32 / self.rank as f32);
        base + delta
    }

    pub fn forward_sequence_chunked(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let chunk = std::env::var("AFTERBURNER_LINEAR_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        let Some(chunk) = chunk else {
            return self.forward(input);
        };
        let [batch, sequence, features] = input.dims();
        if sequence <= chunk {
            return self.forward(input);
        }
        let mut outputs = Vec::with_capacity(sequence.div_ceil(chunk));
        for start in (0..sequence).step_by(chunk) {
            let end = (start + chunk).min(sequence);
            outputs.push(self.forward(input.clone().slice([
                0..batch,
                start..end,
                0..features,
            ])));
        }
        Tensor::cat(outputs, 1)
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Parameters are stored in FP32 to match kohya_ss. Always perform
        // the actual projection in the activation dtype, including the
        // non-autocast/sequence-chunked path; otherwise an FP32 base weight
        // is paired with a BF16/F16 input and Burn rejects the matmul.
        self.forward_low_precision(input)
    }
}

#[cfg(feature = "cuda")]
impl<Inner, C> LoraLinear<burn_autodiff::Autodiff<Inner, C>>
where
    Inner: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn cast_fused_operands(
        &self,
        input_dtype: DType,
    ) -> (
        burn_autodiff::AutodiffTensor<Inner>,
        burn_autodiff::AutodiffTensor<Inner>,
        burn_autodiff::AutodiffTensor<Inner>,
    ) {
        let weight = match self.weight.val().detach().cast(input_dtype).into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized LoRA weights are unsupported"),
        };
        let down = match self.lora_down.val().cast(input_dtype).into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized LoRA factors are unsupported"),
        };
        let up = match self.lora_up.val().cast(input_dtype).into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized LoRA factors are unsupported"),
        };
        (weight, down, up)
    }

    fn forward_fused_operands<const D: usize>(
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, D>,
        weight: burn_autodiff::AutodiffTensor<Inner>,
        down: burn_autodiff::AutodiffTensor<Inner>,
        up: burn_autodiff::AutodiffTensor<Inner>,
        scale: Inner::FloatElem,
        recomputable: bool,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, D> {
        let input = match input.into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized LoRA input is unsupported"),
        };
        let output = if recomputable {
            burn_autodiff::ops::fused_lora_linear_recomputable::<Inner, C>(
                input, weight, down, up, scale,
            )
        } else {
            burn_autodiff::ops::fused_lora_linear::<Inner, C>(input, weight, down, up, scale)
        };
        Tensor::from_primitive(TensorPrimitive::Float(output))
    }

    /// Splits long `[batch, sequence, features]` projections before creating
    /// the fused graph node. This is primarily needed by the F32 CUDA build:
    /// Python keeps the residual stream in F32, and one unchunked projection
    /// can exceed CubeCL's single-buffer limit.
    pub fn forward_fused_sequence_chunked(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        let chunk = std::env::var("AFTERBURNER_LINEAR_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        let Some(chunk) = chunk else {
            return self.forward_fused(input);
        };
        let [batch, sequence, features] = input.dims();
        if sequence <= chunk {
            return self.forward_fused(input);
        }
        let input_dtype = input.dtype();
        let (weight, down, up) = self.cast_fused_operands(input_dtype);
        let scale = (self.alpha as f32 / self.rank as f32).elem();
        let mut outputs = Vec::with_capacity(sequence.div_ceil(chunk));
        for start in (0..sequence).step_by(chunk) {
            let end = (start + chunk).min(sequence);
            outputs.push(Self::forward_fused_operands(
                input.clone().slice([0..batch, start..end, 0..features]),
                weight.clone(), down.clone(), up.clone(), scale, false,
            ));
        }
        Tensor::cat(outputs, 1)
    }

    /// Recomputable counterpart used by the fused CUDA attention path.
    pub fn forward_fused_recomputable_sequence_chunked(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        let chunk = std::env::var("AFTERBURNER_LINEAR_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        let Some(chunk) = chunk else {
            return self.forward_fused_recomputable(input);
        };
        let [batch, sequence, features] = input.dims();
        if sequence <= chunk {
            return self.forward_fused_recomputable(input);
        }
        let input_dtype = input.dtype();
        let (weight, down, up) = self.cast_fused_operands(input_dtype);
        let scale = (self.alpha as f32 / self.rank as f32).elem();
        let mut outputs = Vec::with_capacity(sequence.div_ceil(chunk));
        for start in (0..sequence).step_by(chunk) {
            let end = (start + chunk).min(sequence);
            outputs.push(Self::forward_fused_operands(
                input.clone().slice([0..batch, start..end, 0..features]),
                weight.clone(), down.clone(), up.clone(), scale, true,
            ));
        }
        Tensor::cat(outputs, 1)
    }

    /// CUDA training path that keeps the base projection and LoRA update in a
    /// single Autodiff operation. Large attention projections use the
    /// recomputable variant below; ordinary projections retain the compact
    /// fused node to avoid keeping a copy of every projection input alive.
    pub fn forward_fused<const D: usize>(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, D>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, D> {
        // Preserve the AutodiffTensor nodes. Calling `.tensor()` here unwraps
        // the inner backend primitive and silently cuts the graph before the
        // fused operation can register its parents.
        // Keep the trainable factors stored in FP32, but present the fused
        // projection with one activation dtype. Burn's fusion layer rejects
        // mixed F16/F32 operands before the custom LoRA op gets a chance to
        // cast them itself. The cast remains in the graph, so its gradient is
        // propagated back into the FP32 parameter.
        let input_dtype = input.dtype();
        let (weight, down, up) = self.cast_fused_operands(input_dtype);
        Self::forward_fused_operands(
            input,
            weight,
            down,
            up,
            (self.alpha as f32 / self.rank as f32).elem(),
            false,
        )
    }

    /// CUDA path for large attention projections. The projection output is
    /// recomputed by the custom autodiff node during backward instead of
    /// being retained for every transformer block.
    pub fn forward_fused_recomputable<const D: usize>(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, D>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, D> {
        let input_dtype = input.dtype();
        let (weight, down, up) = self.cast_fused_operands(input_dtype);
        Self::forward_fused_operands(
            input,
            weight,
            down,
            up,
            (self.alpha as f32 / self.rank as f32).elem(),
            true,
        )
    }
}

impl LoraTarget {
    pub fn parameter_count(&self) -> usize {
        self.rank * (self.input_dim + self.output_dim)
    }

    pub fn scale(&self) -> f32 {
        self.alpha as f32 / self.rank as f32
    }
}

/// Selects Anima DiT Linear weights from a SafeTensors key and shape list.
pub fn select_anima_target(
    key: &str,
    shape: &[usize],
    config: &LoraConfig,
) -> Result<Option<LoraTarget>, LoraError> {
    config.validate()?;
    let Some(key) = normalize_anima_lora_key(key) else {
        return Ok(None);
    };
    if !key.starts_with("blocks.") || key.contains("llm_adapter") {
        return Ok(None);
    }
    let module = key
        .strip_prefix("blocks.")
        .and_then(|value| value.split_once('.'))
        .map(|(_, value)| value);
    let Some(module) = module else {
        return Ok(None);
    };
    if !config.target_modules.iter().any(|target| target == module) {
        return Ok(None);
    }
    let [output_dim, input_dim] = shape else {
        return Err(LoraError::InvalidWeightShape {
            key: key.to_owned(),
            shape: shape.to_vec(),
        });
    };
    Ok(Some(LoraTarget {
        key: key.to_owned(),
        kind: LoraTargetKind::Linear,
        output_dim: *output_dim,
        input_dim: *input_dim,
        rank: config.rank,
        alpha: config.alpha,
    }))
}

/// Normalizes native, ComfyUI, and LoRA-factor key prefixes to the native
/// Anima module path used by target selection and loader dispatch.
pub fn normalize_anima_lora_key(key: &str) -> Option<String> {
    let key = key
        .strip_suffix(".lora_down.weight")
        .or_else(|| key.strip_suffix(".lora_up.weight"))
        .unwrap_or_else(|| key.strip_suffix(".weight").unwrap_or(key));
    if let Some(key) = key
        .strip_prefix("model.diffusion_model.")
        .or_else(|| key.strip_prefix("net."))
    {
        return Some(key.to_owned());
    }
    let key = key.strip_prefix("lora_unet_")?;
    if let Some(rest) = key.strip_prefix("blocks_") {
        let (index, component) = rest.split_once('_')?;
        return Some(format!(
            "blocks.{index}.{}",
            flatten_anima_lora_component(component)?
        ));
    }
    if let Some(rest) = key.strip_prefix("llm_adapter_blocks_") {
        let (index, component) = rest.split_once('_')?;
        return Some(format!(
            "llm_adapter.blocks.{index}.{}",
            flatten_anima_lora_component(component)?
        ));
    }
    None
}

fn flatten_anima_lora_component(component: &str) -> Option<&'static str> {
    match component {
        "self_attn_q_proj" => Some("self_attn.q_proj"),
        "self_attn_k_proj" => Some("self_attn.k_proj"),
        "self_attn_v_proj" => Some("self_attn.v_proj"),
        "self_attn_output_proj" => Some("self_attn.output_proj"),
        "cross_attn_q_proj" => Some("cross_attn.q_proj"),
        "cross_attn_k_proj" => Some("cross_attn.k_proj"),
        "cross_attn_v_proj" => Some("cross_attn.v_proj"),
        "cross_attn_output_proj" => Some("cross_attn.output_proj"),
        "mlp_layer1" => Some("mlp.layer1"),
        "mlp_layer2" => Some("mlp.layer2"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn_autodiff::{
        Autodiff,
        checkpoint::strategy::{BalancedCheckpointing, NoCheckpointing},
    };
    use burn_optim::{AdamConfig, GradientsParams, Optimizer};

    #[test]
    fn selects_anima_self_attention_target() {
        let target = select_anima_target(
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            &[2048, 2048],
            &LoraConfig::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(target.rank, 16);
        assert_eq!(target.parameter_count(), 65_536);
        assert_eq!(target.scale(), 1.0);
    }

    #[test]
    fn selects_anima_target_from_native_net_checkpoint_key() {
        let target = select_anima_target(
            "net.blocks.0.mlp.layer1.weight",
            &[8192, 2048],
            &LoraConfig::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!((target.output_dim, target.input_dim), (8192, 2048));
    }

    #[test]
    fn selects_comfyui_lora_factor_key() {
        let target = select_anima_target(
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
            &[1, 2048],
            &LoraConfig::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(target.key, "blocks.0.self_attn.q_proj");
    }

    #[test]
    fn selects_cross_attention_dimensions() {
        let target = select_anima_target(
            "model.diffusion_model.blocks.0.cross_attn.k_proj.weight",
            &[2048, 1024],
            &LoraConfig::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!((target.output_dim, target.input_dim), (2048, 1024));
    }

    #[test]
    fn excludes_llm_adapter_from_dit_targets() {
        let target = select_anima_target(
            "model.diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight",
            &[1024, 1024],
            &LoraConfig::default(),
        )
        .unwrap();
        assert!(target.is_none());
    }

    #[test]
    fn rejects_non_matrix_target() {
        let error = select_anima_target(
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            &[2048],
            &LoraConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, LoraError::InvalidWeightShape { .. }));
    }

    #[test]
    fn burn_lora_linear_has_zero_initial_delta() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = Default::default();
        let layer = LoraLinearConfig {
            input_dim: 4,
            output_dim: 3,
            rank: 2,
            alpha: 2,
            bias: false,
        }
        .init::<Backend>(&device)
        .unwrap();
        let input = Tensor::<Backend, 2>::ones([2, 4], &device);
        let output = layer.forward(input);
        assert_eq!(output.dims(), [2, 3]);
        assert!(
            layer
                .lora_up
                .val()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn burn_lora_linear_initial_gradients_match_probe_pattern() {
        type Backend = Autodiff<burn_ndarray::NdArray<f32>>;
        let device = Default::default();
        let config = LoraLinearConfig {
            input_dim: 4,
            output_dim: 3,
            rank: 2,
            alpha: 2,
            bias: false,
        };
        let mut layer = config.init::<Backend>(&device).unwrap();
        layer.weight = Param::from_tensor(Tensor::<Backend, 2>::ones([3, 4], &device));
        let loss = layer
            .forward(Tensor::<Backend, 2>::ones([2, 4], &device))
            .square()
            .mean();
        let gradients = loss.backward();
        let down = layer
            .lora_down
            .val()
            .grad(&gradients)
            .expect("missing LoRA down gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let up = layer
            .lora_up
            .val()
            .grad(&gradients)
            .expect("missing LoRA up gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(down.iter().all(|value| value.abs() < 1e-7));
        assert!(up.iter().any(|value| value.abs() > 1e-7));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_lora_linear_preserves_factor_gradients() {
        type Backend = Autodiff<burn_ndarray::NdArray<f32>, NoCheckpointing>;
        let device = Default::default();
        let mut layer = LoraLinearConfig {
            input_dim: 4,
            output_dim: 3,
            rank: 2,
            alpha: 2,
            bias: false,
        }
        .init::<Backend>(&device)
        .unwrap();
        layer.weight = Param::from_tensor(Tensor::<Backend, 2>::ones([3, 4], &device));
        let loss = layer
            .forward_fused(Tensor::<Backend, 2>::ones([2, 4], &device))
            .square()
            .mean();
        let gradients = loss.backward();
        let up = layer
            .lora_up
            .val()
            .grad(&gradients)
            .expect("fused LoRA must return an up-factor gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            up.iter()
                .any(|value| value.is_finite() && value.abs() > 1e-7)
        );
    }

    #[test]
    fn burn_lora_linear_updates_zero_initialized_up_factor() {
        type Backend = Autodiff<burn_ndarray::NdArray<f32>>;
        let device = Default::default();
        let layer = LoraLinearConfig {
            input_dim: 4,
            output_dim: 3,
            rank: 2,
            alpha: 2,
            bias: false,
        }
        .init::<Backend>(&device)
        .unwrap();
        let mut layer = layer;
        layer.weight = Param::from_tensor(Tensor::<Backend, 2>::ones([3, 4], &device));
        let input = Tensor::<Backend, 2>::ones([2, 4], &device);
        let loss = layer.forward(input).square().mean();
        let mut optimizer = AdamConfig::new().init::<Backend, LoraLinear<Backend>>();
        let grads = GradientsParams::from_grads(loss.backward(), &layer);
        let updated = optimizer.step(1.0e-2, layer, grads);
        let up = updated.lora_up.val().into_data().to_vec::<f32>().unwrap();
        assert!(up.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn burn_lora_linear_updates_zero_initialized_up_factor_with_checkpointing() {
        type Backend = Autodiff<burn_ndarray::NdArray<f32>, BalancedCheckpointing>;
        let device = Default::default();
        let layer = LoraLinearConfig {
            input_dim: 4,
            output_dim: 3,
            rank: 2,
            alpha: 2,
            bias: false,
        }
        .init::<Backend>(&device)
        .unwrap();
        let mut layer = layer;
        layer.weight = Param::from_tensor(Tensor::<Backend, 2>::ones([3, 4], &device));
        let input = Tensor::<Backend, 2>::ones([2, 4], &device);
        let loss = layer.forward(input).square().mean();
        let mut optimizer = AdamConfig::new().init::<Backend, LoraLinear<Backend>>();
        let grads = GradientsParams::from_grads(loss.backward(), &layer);
        let updated = optimizer.step(1.0e-2, layer, grads);
        let up = updated.lora_up.val().into_data().to_vec::<f32>().unwrap();
        assert!(up.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn rank_eight_anima_probe_has_560_factors_and_expected_shapes() {
        let config = LoraConfig {
            rank: 8,
            alpha: 8,
            ..LoraConfig::default()
        };
        let modules = AnimaConfig::default().lora_targets();
        let mut factor_count = 0;
        let mut parameter_count = 0;
        for block in 0..28 {
            for module in modules {
                let (output_dim, input_dim) = match module {
                    "self_attn.q_proj"
                    | "self_attn.k_proj"
                    | "self_attn.v_proj"
                    | "self_attn.output_proj" => (2048, 2048),
                    "cross_attn.q_proj" | "cross_attn.output_proj" => (2048, 2048),
                    "cross_attn.k_proj" | "cross_attn.v_proj" => (2048, 1024),
                    "mlp.layer1" => (8192, 2048),
                    "mlp.layer2" => (2048, 8192),
                    other => panic!("unexpected Anima LoRA target {other}"),
                };
                let key = format!("model.diffusion_model.blocks.{block}.{module}.weight");
                let target = select_anima_target(&key, &[output_dim, input_dim], &config)
                    .unwrap()
                    .expect("target should be selected");
                assert_eq!(target.rank, 8);
                assert_eq!(target.parameter_count(), 8 * (output_dim + input_dim));
                factor_count += 2;
                parameter_count += target.parameter_count();
            }
        }
        assert_eq!(factor_count, 560);
        assert_eq!(parameter_count, 11_468_800);
    }
}
