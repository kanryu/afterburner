//! LoRA target selection and adapter sizing.
//!
//! The computation graph will be attached to Burn modules after the Anima
//! DiT blocks are implemented. This module first makes target selection
//! explicit and reproducible from SafeTensors keys.

use crate::anima::AnimaConfig;
use burn::{
    module::{Initializer, Module, Param},
    tensor::{Tensor, backend::Backend, module::linear},
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
        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / 3.0_f64.sqrt(),
            fan_out_only: false,
        };
        let bias = self
            .bias
            .then(|| Initializer::Zeros.init([self.output_dim], device));
        Ok(LoraLinear {
            weight: initializer.init_with(
                [self.output_dim, self.input_dim],
                Some(self.input_dim),
                Some(self.output_dim),
                device,
            ),
            bias,
            lora_down: initializer.init_with(
                [self.rank, self.input_dim],
                Some(self.input_dim),
                Some(self.rank),
                device,
            ),
            lora_up: Initializer::Zeros.init([self.output_dim, self.rank], device),
            rank: self.rank,
            alpha: self.alpha,
        })
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
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let base = linear(
            input.clone(),
            self.weight.val().detach().transpose(),
            self.bias.as_ref().map(|bias| bias.val().detach()),
        );
        let delta = linear(
            linear(input, self.lora_down.val().transpose(), None),
            self.lora_up.val().transpose(),
            None,
        )
        .mul_scalar(self.alpha as f32 / self.rank as f32);
        base + delta
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
    if !key.contains("model.diffusion_model.blocks.")
        || key.contains("llm_adapter")
        || !key.ends_with(".weight")
    {
        return Ok(None);
    }
    let module = key
        .strip_prefix("model.diffusion_model.blocks.")
        .and_then(|value| value.split_once('.'))
        .and_then(|(_, value)| value.strip_suffix(".weight"));
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
