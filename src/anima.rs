//! Model specifications for the first Anima implementation target.
//!
//! These values are derived from the inspected SafeTensors headers. This
//! module intentionally contains architecture and key-mapping information,
//! not the full Burn computation graph yet.

/// Structural configuration of the Anima diffusion transformer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimaConfig {
    pub block_count: usize,
    pub hidden_size: usize,
    pub text_hidden_size: usize,
    pub attention_head_count: usize,
    pub attention_head_size: usize,
    pub mlp_hidden_size: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub llm_adapter_block_count: usize,
    pub llm_adapter_hidden_size: usize,
    pub text_vocab_size: usize,
}

impl Default for AnimaConfig {
    fn default() -> Self {
        Self {
            block_count: 28,
            hidden_size: 2048,
            text_hidden_size: 1024,
            attention_head_count: 16,
            attention_head_size: 128,
            mlp_hidden_size: 8192,
            input_channels: 68,
            output_channels: 64,
            llm_adapter_block_count: 6,
            llm_adapter_hidden_size: 1024,
            text_vocab_size: 151_936,
        }
    }
}

impl AnimaConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size != self.attention_head_count * self.attention_head_size {
            return Err("hidden_size must equal heads multiplied by head size".into());
        }
        if self.block_count == 0 || self.llm_adapter_block_count == 0 {
            return Err("Anima must contain at least one block".into());
        }
        Ok(())
    }

    /// LoRA targets for the first implementation milestone.
    pub fn lora_targets(&self) -> [&'static str; 10] {
        [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.output_proj",
            "cross_attn.q_proj",
            "cross_attn.k_proj",
            "cross_attn.v_proj",
            "cross_attn.output_proj",
            "mlp.layer1",
            "mlp.layer2",
        ]
    }
}

/// Classification of weighted modules in an Anima DiT block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimaWeightKind {
    SelfAttentionQ,
    SelfAttentionK,
    SelfAttentionV,
    SelfAttentionOutput,
    CrossAttentionQ,
    CrossAttentionK,
    CrossAttentionV,
    CrossAttentionOutput,
    MlpLayer1,
    MlpLayer2,
    AdaLnModulation,
    TimestepEmbedding,
    InputProjection,
    OutputProjection,
    LlmAdapter,
}

/// Maps a SafeTensors key to the first-stage Anima implementation role.
pub fn classify_anima_key(key: &str) -> Option<AnimaWeightKind> {
    if key.contains("llm_adapter") {
        Some(AnimaWeightKind::LlmAdapter)
    } else if key.ends_with("self_attn.q_proj.weight") {
        Some(AnimaWeightKind::SelfAttentionQ)
    } else if key.ends_with("self_attn.k_proj.weight") {
        Some(AnimaWeightKind::SelfAttentionK)
    } else if key.ends_with("self_attn.v_proj.weight") {
        Some(AnimaWeightKind::SelfAttentionV)
    } else if key.ends_with("self_attn.output_proj.weight") {
        Some(AnimaWeightKind::SelfAttentionOutput)
    } else if key.ends_with("cross_attn.q_proj.weight") {
        Some(AnimaWeightKind::CrossAttentionQ)
    } else if key.ends_with("cross_attn.k_proj.weight") {
        Some(AnimaWeightKind::CrossAttentionK)
    } else if key.ends_with("cross_attn.v_proj.weight") {
        Some(AnimaWeightKind::CrossAttentionV)
    } else if key.ends_with("cross_attn.output_proj.weight") {
        Some(AnimaWeightKind::CrossAttentionOutput)
    } else if key.ends_with("mlp.layer1.weight") {
        Some(AnimaWeightKind::MlpLayer1)
    } else if key.ends_with("mlp.layer2.weight") {
        Some(AnimaWeightKind::MlpLayer2)
    } else if key.contains("adaln_modulation") {
        Some(AnimaWeightKind::AdaLnModulation)
    } else if key.contains("t_embedder") || key.contains("t_embedding_norm") {
        Some(AnimaWeightKind::TimestepEmbedding)
    } else if key.contains("x_embedder") {
        Some(AnimaWeightKind::InputProjection)
    } else if key.contains("final_layer") {
        Some(AnimaWeightKind::OutputProjection)
    } else {
        None
    }
}

/// Shape contract of the Qwen Image VAE found beside the Anima weights.
///
/// The 3D convolution kernels mean the Burn implementation must use
/// `[batch, channel, time, height, width]` tensors. For still images, time is
/// represented by a singleton dimension rather than dropping the axis.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageVaeConfig {
    pub input_channels: usize,
    pub latent_channels: usize,
    pub output_channels: usize,
    pub uses_temporal_axis: bool,
    pub encoder_moments_channels: usize,
    pub spatial_downsample: usize,
    pub latent_mean: [f32; 16],
    pub latent_std: [f32; 16],
}

impl Default for QwenImageVaeConfig {
    fn default() -> Self {
        Self {
            input_channels: 3,
            latent_channels: 16,
            output_channels: 3,
            uses_temporal_axis: true,
            encoder_moments_channels: 32,
            spatial_downsample: 8,
            latent_mean: [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            latent_std: [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
            ],
        }
    }
}

impl QwenImageVaeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_channels != 3
            || self.output_channels != 3
            || self.latent_channels != 16
            || self.encoder_moments_channels != 32
            || self.spatial_downsample == 0
        {
            return Err("unsupported Qwen Image VAE channel or scale configuration".into());
        }
        if self
            .latent_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("Qwen Image VAE latent_std must be finite and positive".into());
        }
        Ok(())
    }

    pub fn normalize_latent(&self, values: &mut [f32], channel_count: usize) -> Result<(), String> {
        if channel_count != self.latent_channels || values.len() % channel_count != 0 {
            return Err("latent data does not match the Qwen Image VAE channel count".into());
        }
        let channel_plane = values.len() / channel_count;
        for (index, value) in values.iter_mut().enumerate() {
            // VaeTensor stores channel-major NCHW-compatible data.  Python's
            // `(latent - mean) * (1 / std)` broadcasts over the channel axis;
            // using index % channel_count would incorrectly treat the buffer
            // as interleaved and corrupt every spatial position after the
            // first channel.
            let channel = index / channel_plane;
            *value = (*value - self.latent_mean[channel]) / self.latent_std[channel];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_anima_config_matches_inspected_model() {
        let config = AnimaConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.block_count, 28);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.lora_targets().len(), 10);
    }

    #[test]
    fn classifies_anima_attention_keys() {
        assert_eq!(
            classify_anima_key("model.diffusion_model.blocks.0.self_attn.q_proj.weight"),
            Some(AnimaWeightKind::SelfAttentionQ)
        );
        assert_eq!(
            classify_anima_key("model.diffusion_model.blocks.27.cross_attn.k_proj.weight"),
            Some(AnimaWeightKind::CrossAttentionK)
        );
        assert_eq!(
            classify_anima_key(
                "model.diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight"
            ),
            Some(AnimaWeightKind::LlmAdapter)
        );
    }

    #[test]
    fn qwen_image_vae_keeps_time_axis() {
        let config = QwenImageVaeConfig::default();
        assert!(config.uses_temporal_axis);
        assert_eq!(config.latent_channels * 2, config.encoder_moments_channels);
        assert_eq!(config.spatial_downsample, 8);
        assert!(config.validate().is_ok());
    }
}
