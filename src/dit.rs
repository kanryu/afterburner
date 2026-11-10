//! Burn implementation skeleton for the Anima diffusion transformer.
//!
//! The attention and MLP paths are functional. Weight loading for the
//! individual modules and Anima's AdaLN/timestep conditioning are kept as
//! separate steps because their exact ordering must be verified against the
//! reference engine.

use burn::{
    module::{Initializer, Module, Param},
    nn::{Embedding, EmbeddingConfig},
    tensor::{
        DType, Int, Tensor, TensorData, activation::gelu, activation::silu, activation::softmax,
        backend::Backend,
    },
};

use crate::{
    anima::AnimaConfig,
    lora::{LoraConfig, LoraLinear, LoraLinearConfig},
};

/// Upper bound for the query dimension of one attention score matrix.
///
/// With checkpointing enabled, the per-chunk operations can be recomputed
/// during backward instead of retaining every intermediate activation.
const ATTENTION_QUERY_CHUNK_SIZE: usize = 128;

fn attention_query_chunk_size() -> usize {
    std::env::var("AFTERBURNER_ATTENTION_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(ATTENTION_QUERY_CHUNK_SIZE)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimaDiTConfigError {
    Invalid(String),
    Lora(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimaTensorError {
    InvalidRank { expected: usize, actual: usize },
    InvalidShape { message: String, actual: Vec<usize> },
}

impl std::fmt::Display for AnimaTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRank { expected, actual } => {
                write!(f, "expected rank {expected} tensor, got rank {actual}")
            }
            Self::InvalidShape { message, actual } => write!(f, "{message}: {actual:?}"),
        }
    }
}

impl std::error::Error for AnimaTensorError {}

/// Converts a VAE latent `[B, C, T, H, W]` into Anima image tokens.
///
/// Anima uses a spatial 2x2 patch and one additional mask channel, yielding
/// `17 * 2 * 2 = 68` input features for its default 16-channel latent.
pub fn patchify_latent<B: Backend>(
    latent: Tensor<B, 5>,
    padding_mask: Option<Tensor<B, 4>>,
) -> Result<Tensor<B, 3>, AnimaTensorError> {
    let [batch, channels, frames, height, width] = latent.dims();
    if frames != 1 || height % 2 != 0 || width % 2 != 0 {
        return Err(AnimaTensorError::InvalidShape {
            message: "latent must have one frame and even spatial dimensions".into(),
            actual: vec![batch, channels, frames, height, width],
        });
    }
    let device = latent.device();
    let mask = match padding_mask {
        Some(mask) => {
            let dims = mask.dims();
            if dims != [batch, 1, height, width] {
                return Err(AnimaTensorError::InvalidShape {
                    message: "padding mask must be [B, 1, H, W]".into(),
                    actual: dims.to_vec(),
                });
            }
            mask
        }
        None => Tensor::zeros([batch, 1, height, width], &device),
    };
    let latent = latent.reshape([batch, channels, height, width]);
    let input = Tensor::cat(vec![latent, mask], 1);
    let input_channels = channels + 1;
    Ok(input
        .reshape([batch, input_channels, height / 2, 2, width / 2, 2])
        .swap_dims(1, 2)
        .swap_dims(2, 4)
        .swap_dims(3, 4)
        .reshape([batch, (height / 2) * (width / 2), input_channels * 4]))
}

/// Converts Anima output tokens back into a latent `[B, C, 1, H, W]` tensor.
pub fn unpatchify_latent<B: Backend>(
    tokens: Tensor<B, 3>,
    output_channels: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<B, 5>, AnimaTensorError> {
    let [batch, token_count, feature_count] = tokens.dims();
    if height % 2 != 0 || width % 2 != 0 || token_count != height / 2 * (width / 2) {
        return Err(AnimaTensorError::InvalidShape {
            message: "token count does not match the requested even spatial shape".into(),
            actual: vec![batch, token_count, feature_count],
        });
    }
    if feature_count != output_channels * 4 {
        return Err(AnimaTensorError::InvalidShape {
            message: "output feature count must equal channels * 2 * 2".into(),
            actual: vec![batch, token_count, feature_count],
        });
    }
    Ok(tokens
        .reshape([batch, height / 2, width / 2, output_channels, 2, 2])
        .swap_dims(3, 4)
        .swap_dims(2, 4)
        .swap_dims(1, 2)
        .reshape([batch, output_channels, 1, height, width]))
}

impl std::fmt::Display for AnimaDiTConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid Anima DiT configuration: {message}"),
            Self::Lora(message) => write!(f, "invalid Anima DiT LoRA configuration: {message}"),
        }
    }
}

impl std::error::Error for AnimaDiTConfigError {}

/// Dimensions required by one Anima DiT block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimaDiTBlockConfig {
    pub hidden_size: usize,
    pub context_size: usize,
    pub attention_head_count: usize,
    pub attention_head_size: usize,
    pub mlp_hidden_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: usize,
}

impl AnimaDiTBlockConfig {
    pub fn validate(&self) -> Result<(), AnimaDiTConfigError> {
        if self.hidden_size != self.attention_head_count * self.attention_head_size {
            return Err(AnimaDiTConfigError::Invalid(
                "hidden size must equal head count multiplied by head size".into(),
            ));
        }
        if self.context_size == 0 || self.mlp_hidden_size == 0 {
            return Err(AnimaDiTConfigError::Invalid(
                "context and MLP dimensions must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    fn linear(
        &self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<LoraLinearConfig, AnimaDiTConfigError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(AnimaDiTConfigError::Invalid(
                "linear dimensions must be greater than zero".into(),
            ));
        }
        Ok(LoraLinearConfig {
            input_dim,
            output_dim,
            rank: self.lora_rank,
            alpha: self.lora_alpha,
            bias: false,
        })
    }
}

impl From<&AnimaConfig> for AnimaDiTBlockConfig {
    fn from(config: &AnimaConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            context_size: config.text_hidden_size,
            attention_head_count: config.attention_head_count,
            attention_head_size: config.attention_head_size,
            mlp_hidden_size: config.mlp_hidden_size,
            lora_rank: 16,
            lora_alpha: 16,
        }
    }
}

#[derive(Module, Debug)]
pub struct AdaLnModulation<B: Backend> {
    pub down: LoraLinear<B>,
    pub up: LoraLinear<B>,
    pub hidden_size: usize,
}

#[derive(Module, Debug)]
pub struct AnimaFinalModulation<B: Backend> {
    pub down: LoraLinear<B>,
    pub up: LoraLinear<B>,
    pub hidden_size: usize,
}

impl<B: Backend> AnimaFinalModulation<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            down: config
                .linear(config.hidden_size, 256)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            up: config
                .linear(256, config.hidden_size * 2)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch, _, _] = input.dims();
        let params = self
            .up
            .forward(silu(self.down.forward(condition)))
            .add(timestep_embedding.slice([0..batch, 0..self.hidden_size * 2]));
        let shift = params.clone().slice([0..batch, 0..self.hidden_size]);
        let scale = params.slice([0..batch, self.hidden_size..self.hidden_size * 2]);
        let (variance, mean) = input.clone().var_mean_bias(2);
        input
            .sub(mean)
            .div(variance.add_scalar(1e-6).sqrt())
            .mul(scale.add_scalar(1.0).unsqueeze())
            .add(shift.unsqueeze())
    }
}

impl<B: Backend> AdaLnModulation<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            down: config
                .linear(config.hidden_size, 256)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            up: config
                .linear(256, config.hidden_size * 3)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&self, condition: Tensor<B, 2>) -> Tensor<B, 2> {
        self.up.forward(silu(self.down.forward(condition)))
    }

    fn apply(
        &self,
        input: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, _, _] = input.dims();
        let params = self.forward(condition).add(timestep_embedding);
        let shift = params.clone().slice([0..batch, 0..self.hidden_size]);
        let scale = params
            .clone()
            .slice([0..batch, self.hidden_size..self.hidden_size * 2]);
        let gate = params.slice([0..batch, self.hidden_size * 2..self.hidden_size * 3]);
        let (variance, mean) = input.clone().var_mean_bias(2);
        let normalized = input
            .sub(mean)
            .div(variance.add_scalar(1e-5).sqrt())
            .mul(scale.add_scalar(1.0).unsqueeze())
            .add(shift.unsqueeze());
        (normalized, gate.unsqueeze())
    }
}

#[derive(Module, Debug)]
pub struct AnimaAttention<B: Backend> {
    pub q_proj: LoraLinear<B>,
    pub k_proj: LoraLinear<B>,
    pub v_proj: LoraLinear<B>,
    pub output_proj: LoraLinear<B>,
    pub q_norm: AnimaRmsNorm<B>,
    pub k_norm: AnimaRmsNorm<B>,
    pub use_rotary: bool,
    pub head_count: usize,
    pub head_size: usize,
}

impl<B: Backend> AnimaAttention<B> {
    pub(crate) fn init_self(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init(config, config.hidden_size, true, device)
    }

    pub(crate) fn init_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init(config, config.context_size, false, device)
    }

    /// Initializes the LLM-adapter cross attention.
    ///
    /// The adapter uses the same one-dimensional split-half RoPE for both
    /// query and context positions.  This differs from the DiT cross
    /// attention, which intentionally has no rotary embedding.
    pub(crate) fn init_adapter_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init(config, config.context_size, true, device)
    }

    pub(crate) fn init(
        config: &AnimaDiTBlockConfig,
        key_value_size: usize,
        use_rotary: bool,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            q_proj: config
                .linear(config.hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            k_proj: config
                .linear(key_value_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            v_proj: config
                .linear(key_value_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            output_proj: config
                .linear(config.hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            q_norm: AnimaRmsNorm::init(config.attention_head_size, device),
            k_norm: AnimaRmsNorm::init(config.attention_head_size, device),
            use_rotary,
            head_count: config.attention_head_count,
            head_size: config.attention_head_size,
        })
    }

    pub fn forward(&self, input: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_mask(input, context, None)
    }

    pub fn forward_with_mask(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        self.forward_with_mask_and_spatial(input, context, context_mask, None)
    }

    fn forward_with_mask_and_spatial(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3> {
        let [batch, query_length, _] = input.dims();
        let [_, context_length, _] = context.dims();
        let q = self
            .q_proj
            .forward(input.clone())
            .reshape([batch, query_length, self.head_count, self.head_size])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(context.clone())
            .reshape([batch, context_length, self.head_count, self.head_size])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(context)
            .reshape([batch, context_length, self.head_count, self.head_size])
            .swap_dims(1, 2);
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);
        let (q, k) = if self.use_rotary {
            let device = q.device();
            match spatial {
                Some((height, width)) => (
                    apply_rotary_3d(q, height, width, self.head_size, &device),
                    apply_rotary_3d(k, height, width, self.head_size, &device),
                ),
                None => (
                    apply_rotary(q, query_length, self.head_size, &device),
                    apply_rotary(k, context_length, self.head_size, &device),
                ),
            }
        } else {
            (q, k)
        };
        let key = k.swap_dims(2, 3);
        let scale = 1.0 / (self.head_size as f32).sqrt();
        let mut attended_chunks = Vec::new();
        let query_chunk_size = attention_query_chunk_size();
        for query_start in (0..query_length).step_by(query_chunk_size) {
            let query_end = (query_start + query_chunk_size).min(query_length);
            let query = q.clone().slice([
                0..batch,
                0..self.head_count,
                query_start..query_end,
                0..self.head_size,
            ]);
            let mut scores = query.matmul(key.clone()).mul_scalar(scale);
            if let Some(mask) = context_mask.clone() {
                scores = scores.add(mask);
            }
            // Keep attention tensors in the backend dtype, but evaluate the
            // numerically sensitive softmax normalization in F32. This is
            // required by some Vulkan BF16 shader paths.
            // Capture the dtype before consuming `scores`. Cloning here would
            // keep an additional score node alive in the autodiff graph for
            // every query chunk, which is especially expensive for DiT
            // self-attention.
            let scores_dtype = scores.dtype();
            let attended = softmax(scores.cast(DType::F32), 3)
                .cast(scores_dtype)
                .matmul(v.clone());
            attended_chunks.push(attended);
        }
        let attended = Tensor::cat(attended_chunks, 2).swap_dims(1, 2).reshape([
            batch,
            query_length,
            self.head_count * self.head_size,
        ]);
        self.output_proj.forward(attended)
    }
}

fn apply_rotary<B: Backend>(
    input: Tensor<B, 4>,
    sequence_length: usize,
    head_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, heads, _, _] = input.dims();
    let half = head_size / 2;
    let mut cos_values = Vec::with_capacity(sequence_length * half);
    let mut sin_values = Vec::with_capacity(sequence_length * half);
    for position in 0..sequence_length {
        for index in 0..half {
            let frequency = 10000.0_f32.powf(-2.0 * index as f32 / head_size as f32);
            let angle = position as f32 * frequency;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }
    let cos = Tensor::from_data(
        TensorData::new(cos_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
        device,
    );
    let sin = Tensor::from_data(
        TensorData::new(sin_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
        device,
    );
    let first = input
        .clone()
        .slice([0..batch, 0..heads, 0..sequence_length, 0..half]);
    let second = input.slice([0..batch, 0..heads, 0..sequence_length, half..head_size]);
    Tensor::cat(
        vec![
            first
                .clone()
                .mul(cos.clone())
                .sub(second.clone().mul(sin.clone())),
            second.mul(cos).add(first.mul(sin)),
        ],
        3,
    )
}

fn apply_rotary_3d<B: Backend>(
    input: Tensor<B, 4>,
    height: usize,
    width: usize,
    head_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, heads, sequence_length, _] = input.dims();
    assert_eq!(sequence_length, height * width);
    assert_eq!(head_size % 2, 0);
    let dim_h = head_size / 6 * 2;
    let dim_w = dim_h;
    let dim_t = head_size - dim_h - dim_w;
    let h_theta = 10000.0_f32 * 4.0_f32.powf(dim_h as f32 / (dim_h - 2) as f32);
    let w_theta = h_theta;
    let t_theta = 10000.0_f32;
    let mut frequencies = Vec::with_capacity(sequence_length * (head_size / 2));
    for token in 0..sequence_length {
        let row = token / width;
        let column = token % width;
        let mut values = Vec::with_capacity(head_size / 2);
        for index in 0..dim_t / 2 {
            values.push(0.0_f32 / (t_theta.powf((2 * index) as f32 / dim_t as f32)));
        }
        for index in 0..dim_h / 2 {
            values.push(row as f32 / h_theta.powf((2 * index) as f32 / dim_h as f32));
        }
        for index in 0..dim_w / 2 {
            values.push(column as f32 / w_theta.powf((2 * index) as f32 / dim_w as f32));
        }
        frequencies.extend(values);
    }
    let half = head_size / 2;
    let cos_half = Tensor::from_data(
        TensorData::new(
            frequencies
                .iter()
                .map(|value| value.cos())
                .collect::<Vec<_>>(),
            [1, 1, sequence_length, half],
        )
        .convert::<B::FloatElem>(),
        device,
    );
    let sin_half = Tensor::from_data(
        TensorData::new(
            frequencies
                .iter()
                .map(|value| value.sin())
                .collect::<Vec<_>>(),
            [1, 1, sequence_length, half],
        )
        .convert::<B::FloatElem>(),
        device,
    );
    let first = input
        .clone()
        .slice([0..batch, 0..heads, 0..sequence_length, 0..half]);
    let second = input.slice([0..batch, 0..heads, 0..sequence_length, half..head_size]);
    Tensor::cat(
        vec![
            first
                .clone()
                .mul(cos_half.clone())
                .sub(second.clone().mul(sin_half.clone())),
            second.mul(cos_half).add(first.mul(sin_half)),
        ],
        3,
    )
}

#[derive(Module, Debug)]
pub struct AnimaDiTBlock<B: Backend> {
    pub self_attn: AnimaAttention<B>,
    pub cross_attn: AnimaAttention<B>,
    pub mlp_layer1: LoraLinear<B>,
    pub mlp_layer2: LoraLinear<B>,
    pub self_attn_modulation: AdaLnModulation<B>,
    pub cross_attn_modulation: AdaLnModulation<B>,
    pub mlp_modulation: AdaLnModulation<B>,
}

impl<B: Backend> AnimaDiTBlock<B> {
    pub fn init(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        config.validate()?;
        Ok(Self {
            self_attn: AnimaAttention::init_self(config, device)?,
            cross_attn: AnimaAttention::init_cross(config, device)?,
            mlp_layer1: config
                .linear(config.hidden_size, config.mlp_hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            mlp_layer2: config
                .linear(config.mlp_hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            self_attn_modulation: AdaLnModulation::init(config, device)?,
            cross_attn_modulation: AdaLnModulation::init(config, device)?,
            mlp_modulation: AdaLnModulation::init(config, device)?,
        })
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.forward_with_text_mask(input, text, condition, timestep_embedding, None)
    }

    pub fn forward_with_text_mask(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        self.forward_with_text_mask_and_spatial(
            input,
            text,
            condition,
            timestep_embedding,
            text_mask,
            None,
        )
    }

    fn forward_with_text_mask_and_spatial(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3> {
        let (self_input, self_gate) = self.self_attn_modulation.apply(
            input.clone(),
            condition.clone(),
            timestep_embedding.clone(),
        );
        let self_output = self.self_attn.forward_with_mask_and_spatial(
            self_input.clone(),
            self_input,
            None,
            spatial,
        );
        let hidden = input + self_output.mul(self_gate);
        let (cross_input, cross_gate) = self.cross_attn_modulation.apply(
            hidden.clone(),
            condition.clone(),
            timestep_embedding.clone(),
        );
        let cross_output =
            self.cross_attn
                .forward_with_mask_and_spatial(cross_input, text, text_mask, None);
        let hidden = hidden + cross_output.mul(cross_gate);
        let (mlp_input, mlp_gate) =
            self.mlp_modulation
                .apply(hidden.clone(), condition, timestep_embedding);
        let mlp = self
            .mlp_layer2
            .forward(gelu(self.mlp_layer1.forward(mlp_input)));
        hidden + mlp.mul(mlp_gate)
    }
}

/// Projects precomputed timestep features into the Anima conditioning paths.
/// The first output is `[batch, 3 * hidden]` and is added to each block's
/// shift/scale/gate projection. The second output is the RMS-normalized
/// timestep vector consumed by the per-block AdaLN projections.
#[derive(Module, Debug)]
pub struct AnimaTimestepEmbedder<B: Backend> {
    pub linear1: LoraLinear<B>,
    pub linear2: LoraLinear<B>,
    pub norm: AnimaRmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct AnimaRmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub epsilon: f32,
}

impl<B: Backend> AnimaRmsNorm<B> {
    fn init(size: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([size], device),
            epsilon: 1e-6,
        }
    }

    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = input.dtype();
        let input_f32 = input.clone().cast(DType::F32);
        let rms = input_f32
            .clone()
            .square()
            .mean_dim(D - 1)
            .add_scalar(self.epsilon)
            .sqrt();
        input_f32
            .div(rms)
            .mul(self.gamma.val().detach().cast(DType::F32).unsqueeze())
            .cast(dtype)
    }
}

impl<B: Backend> AnimaTimestepEmbedder<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            linear1: config
                .linear(config.hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            linear2: config
                .linear(config.hidden_size, config.hidden_size * 3)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            norm: AnimaRmsNorm::init(config.hidden_size, device),
        })
    }

    pub fn forward(&self, features: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let normalized = self.norm.forward(features.clone());
        let embedding = self.linear2.forward(silu(self.linear1.forward(features)));
        (embedding, normalized)
    }
}

#[derive(Module, Debug)]
pub struct AnimaLlmAdapterBlock<B: Backend> {
    pub norm_self_attn: AnimaRmsNorm<B>,
    pub norm_cross_attn: AnimaRmsNorm<B>,
    pub norm_mlp: AnimaRmsNorm<B>,
    pub self_attn: AnimaAttention<B>,
    pub cross_attn: AnimaAttention<B>,
    pub mlp_layer1: LoraLinear<B>,
    pub mlp_layer2: LoraLinear<B>,
}

impl<B: Backend> AnimaLlmAdapterBlock<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            norm_self_attn: AnimaRmsNorm::init(config.hidden_size, device),
            norm_cross_attn: AnimaRmsNorm::init(config.hidden_size, device),
            norm_mlp: AnimaRmsNorm::init(config.hidden_size, device),
            self_attn: AnimaAttention::init_self(config, device)?,
            cross_attn: AnimaAttention::init_adapter_cross(config, device)?,
            mlp_layer1: LoraLinearConfig {
                bias: true,
                ..config.linear(config.hidden_size, config.mlp_hidden_size)?
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            mlp_layer2: LoraLinearConfig {
                bias: true,
                ..config.linear(config.mlp_hidden_size, config.hidden_size)?
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
        })
    }

    fn forward_with_mask(
        &self,
        input: Tensor<B, 3>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        self.forward_with_masks(input.clone(), input, text_mask.clone(), text_mask)
    }

    fn forward_with_masks(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        target_mask: Option<Tensor<B, 4>>,
        source_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let self_input = self.norm_self_attn.forward(input.clone());
        let self_output =
            self.self_attn
                .forward_with_mask(self_input.clone(), self_input, target_mask);
        let hidden = input + self_output;
        let cross_input = self.norm_cross_attn.forward(hidden.clone());
        let cross_output =
            self.cross_attn
                .forward_with_mask(cross_input.clone(), text, source_mask);
        let hidden = hidden + cross_output;
        let mlp_input = self.norm_mlp.forward(hidden.clone());
        let mlp = self
            .mlp_layer2
            .forward(gelu(self.mlp_layer1.forward(mlp_input)));
        hidden + mlp
    }
}

#[derive(Module, Debug)]
pub struct AnimaLlmAdapter<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub blocks: Vec<AnimaLlmAdapterBlock<B>>,
    pub norm: AnimaRmsNorm<B>,
    pub output_proj: LoraLinear<B>,
}

impl<B: Backend> AnimaLlmAdapter<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        let mut blocks = Vec::with_capacity(6);
        for _ in 0..6 {
            blocks.push(AnimaLlmAdapterBlock::init(config, device)?);
        }
        let output_proj = config.linear(config.hidden_size, config.hidden_size)?;
        let output_proj = LoraLinearConfig {
            bias: true,
            ..output_proj
        };
        Ok(Self {
            token_embedding: EmbeddingConfig::new(32_128, config.hidden_size).init(device),
            blocks,
            norm: AnimaRmsNorm::init(config.hidden_size, device),
            output_proj: output_proj
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
        })
    }

    pub fn forward(&self, text: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_mask(text, None)
    }

    pub fn forward_with_mask(
        &self,
        mut text: Tensor<B, 3>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        for block in &self.blocks {
            text = block.forward_with_mask(text, text_mask.clone());
        }
        self.output_proj.forward(self.norm.forward(text))
    }

    pub fn forward_with_conditioning(
        &self,
        source: Tensor<B, 3>,
        input_ids: Tensor<B, 2, Int>,
        target_mask: Option<Tensor<B, 4>>,
        source_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let target_mask_for_output = target_mask.clone();
        let mut text = self.token_embedding.forward(input_ids);
        for block in &self.blocks {
            text = block.forward_with_masks(
                text,
                source.clone(),
                target_mask.clone(),
                source_mask.clone(),
            );
        }
        let output = self.output_proj.forward(self.norm.forward(text));
        match target_mask_for_output {
            Some(mask) => {
                let [batch, _, _, sequence] = mask.dims();
                let valid = mask
                    .greater_elem(-0.5)
                    .float()
                    .reshape([batch, sequence, 1]);
                output.mul(valid)
            }
            None => output,
        }
    }
}

#[derive(Module, Debug)]
pub struct AnimaDiT<B: Backend> {
    pub input_projection: LoraLinear<B>,
    pub timestep_embedder: AnimaTimestepEmbedder<B>,
    pub llm_adapter: AnimaLlmAdapter<B>,
    pub blocks: Vec<AnimaDiTBlock<B>>,
    pub final_modulation: AnimaFinalModulation<B>,
    pub output_projection: LoraLinear<B>,
    pub input_features: usize,
    pub latent_channels: usize,
}

impl<B: Backend> AnimaDiT<B> {
    pub fn init(
        config: &AnimaConfig,
        lora: &LoraConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        config.validate().map_err(AnimaDiTConfigError::Invalid)?;
        lora.validate()
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?;
        let mut block_config = AnimaDiTBlockConfig::from(config);
        block_config.lora_rank = lora.rank;
        block_config.lora_alpha = lora.alpha;
        let mut blocks = Vec::with_capacity(config.block_count);
        for _ in 0..config.block_count {
            blocks.push(AnimaDiTBlock::init(&block_config, device)?);
        }
        Ok(Self {
            input_projection: LoraLinearConfig {
                input_dim: config.input_channels,
                output_dim: config.hidden_size,
                rank: lora.rank,
                alpha: lora.alpha,
                bias: false,
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            timestep_embedder: AnimaTimestepEmbedder::init(&block_config, device)?,
            llm_adapter: AnimaLlmAdapter::init(
                &AnimaDiTBlockConfig {
                    hidden_size: config.llm_adapter_hidden_size,
                    context_size: config.llm_adapter_hidden_size,
                    attention_head_count: 16,
                    attention_head_size: 64,
                    mlp_hidden_size: 4096,
                    lora_rank: lora.rank,
                    lora_alpha: lora.alpha,
                },
                device,
            )?,
            blocks,
            final_modulation: AnimaFinalModulation::init(&block_config, device)?,
            output_projection: LoraLinearConfig {
                input_dim: config.hidden_size,
                output_dim: config.output_channels,
                rank: lora.rank,
                alpha: lora.alpha,
                bias: false,
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            input_features: config.input_channels,
            latent_channels: config.output_channels / 4,
        })
    }

    /// Forward pass for already patchified latent tokens and cached text embeddings.
    pub fn forward(
        &self,
        latent_tokens: Tensor<B, 3>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.forward_with_text_mask(latent_tokens, text, timestep_features, None)
    }

    pub fn forward_with_text_mask(
        &self,
        latent_tokens: Tensor<B, 3>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let text = self.llm_adapter.forward_with_mask(text, text_mask.clone());
        let (timestep_embedding, time_condition) =
            self.timestep_embedder.forward(timestep_features);
        let mut hidden = self.input_projection.forward(latent_tokens);
        for block in &self.blocks {
            hidden = block.forward_with_text_mask(
                hidden,
                text.clone(),
                time_condition.clone(),
                timestep_embedding.clone(),
                text_mask.clone(),
            );
        }
        let hidden = self
            .final_modulation
            .forward(hidden, time_condition, timestep_embedding);
        self.output_projection.forward(hidden)
    }

    /// Runs the DiT using VAE latent layout instead of pre-patchified tokens.
    pub fn forward_latent(
        &self,
        latent: Tensor<B, 5>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        padding_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 5>, AnimaTensorError> {
        self.forward_latent_with_text_mask(latent, text, timestep_features, padding_mask, None)
    }

    pub fn forward_latent_with_text_mask(
        &self,
        latent: Tensor<B, 5>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        padding_mask: Option<Tensor<B, 4>>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 5>, AnimaTensorError> {
        let [_, _, _, height, width] = latent.dims();
        let tokens = patchify_latent(latent, padding_mask)?;
        if tokens.dims()[2] != self.input_features {
            return Err(AnimaTensorError::InvalidShape {
                message: format!(
                    "Anima patchified latent must have {} input features",
                    self.input_features
                ),
                actual: tokens.dims().to_vec(),
            });
        }
        let predicted = self.forward_latent_tokens_with_spatial(
            tokens,
            text,
            timestep_features,
            text_mask,
            (height / 2, width / 2),
            None,
            None,
        );
        unpatchify_latent(predicted, self.latent_channels, height, width)
    }

    pub fn forward_latent_with_conditioning(
        &self,
        latent: Tensor<B, 5>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        padding_mask: Option<Tensor<B, 4>>,
        source_mask: Option<Tensor<B, 4>>,
        adapter_input_ids: Tensor<B, 2, Int>,
        adapter_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 5>, AnimaTensorError> {
        let [_, _, _, height, width] = latent.dims();
        let tokens = patchify_latent(latent, padding_mask)?;
        if tokens.dims()[2] != self.input_features {
            return Err(AnimaTensorError::InvalidShape {
                message: format!(
                    "Anima patchified latent must have {} input features",
                    self.input_features
                ),
                actual: tokens.dims().to_vec(),
            });
        }
        let predicted = self.forward_latent_tokens_with_spatial(
            tokens,
            text,
            timestep_features,
            source_mask,
            (height / 2, width / 2),
            Some(adapter_input_ids),
            adapter_mask,
        );
        unpatchify_latent(predicted, self.latent_channels, height, width)
    }

    fn forward_latent_tokens_with_spatial(
        &self,
        latent_tokens: Tensor<B, 3>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
        spatial: (usize, usize),
        adapter_input_ids: Option<Tensor<B, 2, Int>>,
        adapter_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let debug = std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some();
        if debug {
            eprintln!(
                "[forward] start latent={:?} text={:?} timestep={:?} conditioned={}",
                latent_tokens.dims(),
                text.dims(),
                timestep_features.dims(),
                adapter_input_ids.is_some()
            );
        }
        let conditioned = adapter_input_ids.is_some();
        let text = match adapter_input_ids {
            Some(input_ids) => self.llm_adapter.forward_with_conditioning(
                text,
                input_ids,
                adapter_mask,
                text_mask.clone(),
            ),
            None => self.llm_adapter.forward_with_mask(text, text_mask.clone()),
        };
        if debug {
            eprintln!("[forward] text conditioning complete: {:?}", text.dims());
        }
        let (timestep_embedding, time_condition) =
            self.timestep_embedder.forward(timestep_features);
        if debug {
            eprintln!(
                "[forward] timestep complete: embedding={:?} condition={:?}",
                timestep_embedding.dims(),
                time_condition.dims()
            );
        }
        let mut hidden = self.input_projection.forward(latent_tokens);
        if debug {
            eprintln!("[forward] input projection complete: {:?}", hidden.dims());
        }
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_with_text_mask_and_spatial(
                hidden,
                text.clone(),
                time_condition.clone(),
                timestep_embedding.clone(),
                if conditioned { None } else { text_mask.clone() },
                Some(spatial),
            );
            if debug {
                eprintln!("[forward] block {} complete: {:?}", index, hidden.dims());
            }
        }
        let hidden = self
            .final_modulation
            .forward(hidden, time_condition, timestep_embedding);
        if debug {
            eprintln!("[forward] final modulation complete: {:?}", hidden.dims());
        }
        self.output_projection.forward(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_forward_preserves_token_shape() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let block = AnimaDiTBlock::init(&config, &device).unwrap();
        let input = Tensor::<Backend, 3>::ones([1, 3, 8], &device);
        let text = Tensor::<Backend, 3>::ones([1, 5, 8], &device);
        let condition = Tensor::<Backend, 2>::ones([1, 8], &device);
        let timestep_embedding = Tensor::<Backend, 2>::ones([1, 24], &device);
        let output = block.forward(input, text, condition, timestep_embedding);
        assert_eq!(output.dims(), [1, 3, 8]);
    }

    #[test]
    fn attention_chunking_preserves_long_sequence_shape() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let attention = AnimaAttention::<Backend>::init_self(&config, &device).unwrap();
        let input = Tensor::<Backend, 3>::ones([1, ATTENTION_QUERY_CHUNK_SIZE + 1, 8], &device);
        let output = attention.forward(input.clone(), input);
        assert_eq!(output.dims(), [1, ATTENTION_QUERY_CHUNK_SIZE + 1, 8]);
    }

    #[test]
    fn timestep_embedder_matches_cosmos_condition_shapes() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let embedder = AnimaTimestepEmbedder::<Backend>::init(&config, &device).unwrap();
        let (embedding, normalized) = embedder.forward(Tensor::ones([2, 8], &device));
        assert_eq!(embedding.dims(), [2, 24]);
        assert_eq!(normalized.dims(), [2, 8]);
    }

    #[test]
    fn patchify_and_unpatchify_preserve_image_layout() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let latent = Tensor::<Backend, 5>::ones([1, 2, 1, 4, 4], &device);
        let tokens = patchify_latent(latent, None).unwrap();
        assert_eq!(tokens.dims(), [1, 4, 12]);
        let restored = unpatchify_latent(tokens, 3, 4, 4).unwrap();
        assert_eq!(restored.dims(), [1, 3, 1, 4, 4]);
    }

    #[test]
    fn dit_propagates_lora_rank_to_each_block() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let mut config = crate::anima::AnimaConfig::default();
        config.block_count = 1;
        let lora = crate::lora::LoraConfig {
            rank: 1,
            alpha: 1,
            ..Default::default()
        };
        let model = AnimaDiT::<Backend>::init(&config, &lora, &device).unwrap();
        assert_eq!(
            model.blocks[0].self_attn.q_proj.lora_down.val().dims(),
            [1, 2048]
        );
        assert_eq!(
            model.blocks[0].self_attn.q_proj.lora_up.val().dims(),
            [2048, 1]
        );
    }
}
