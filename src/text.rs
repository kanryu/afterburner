//! Candle Qwen3 text-encoder adapter used during cache construction.
//!
//! Tokenization is intentionally injected. Anima deployments can use the
//! matching Qwen tokenizer without making the core engine depend on a
//! particular tokenizer file format.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use tokenizers::Tokenizer;

use crate::{
    cache::AdapterConditioning,
    preprocess::{PreprocessError, TextEncoder, TextEncoding},
    vae::VaeTensor,
};

/// Tokenization boundary kept separate from the Qwen3 model execution.
pub trait CaptionTokenizer {
    fn encode(&mut self, caption: &str) -> Result<TokenizedCaption, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedCaption {
    pub token_ids: Vec<u32>,
    pub attention_mask: Option<Vec<f32>>,
}

impl<F, E> CaptionTokenizer for F
where
    F: FnMut(&str) -> Result<Vec<u32>, E>,
    E: std::fmt::Display,
{
    fn encode(&mut self, caption: &str) -> Result<TokenizedCaption, String> {
        self(caption)
            .map(|token_ids| TokenizedCaption {
                token_ids,
                attention_mask: None,
            })
            .map_err(|error| error.to_string())
    }
}

/// Qwen tokenizer.json adapter with deterministic truncation and padding.
pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
    pad_token_id: u32,
}

/// T5 tokenizer used as the target sequence of Anima's LLM adapter.
///
/// The T5 tokens are not passed through a separate T5 encoder. They are
/// embedded by the Anima LLM adapter itself.
pub struct T5Tokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
    pad_token_id: u32,
}

impl T5Tokenizer {
    pub fn from_file(
        path: impl AsRef<Path>,
        max_length: usize,
        pad_token_id: u32,
    ) -> Result<Self, String> {
        if max_length == 0 {
            return Err("T5 tokenizer max_length must be greater than zero".into());
        }
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| format!("failed to load T5 tokenizer {}: {error}", path.display()))?;
        Ok(Self {
            tokenizer,
            max_length,
            pad_token_id,
        })
    }

    pub fn encode_caption(&self, caption: &str) -> Result<TokenizedCaption, String> {
        let encoding = self
            .tokenizer
            .encode(caption, true)
            .map_err(|error| format!("T5 tokenization failed: {error}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(self.max_length);
        let token_count = ids.len();
        ids.resize(self.max_length, self.pad_token_id);
        let mut attention_mask = vec![0.0; self.max_length];
        attention_mask[..token_count].fill(1.0);
        Ok(TokenizedCaption {
            token_ids: ids,
            attention_mask: Some(attention_mask),
        })
    }
}

impl CaptionTokenizer for T5Tokenizer {
    fn encode(&mut self, caption: &str) -> Result<TokenizedCaption, String> {
        self.encode_caption(caption)
    }
}

impl QwenTokenizer {
    pub fn from_file(
        path: impl AsRef<Path>,
        max_length: usize,
        pad_token_id: u32,
    ) -> Result<Self, String> {
        if max_length == 0 {
            return Err("tokenizer max_length must be greater than zero".into());
        }
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| format!("failed to load tokenizer {}: {error}", path.display()))?;
        Ok(Self {
            tokenizer,
            max_length,
            pad_token_id,
        })
    }

    pub fn encode_caption(&self, caption: &str) -> Result<TokenizedCaption, String> {
        let encoding = self
            .tokenizer
            .encode(caption, true)
            .map_err(|error| format!("tokenization failed: {error}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(self.max_length);
        let token_count = ids.len();
        ids.resize(self.max_length, self.pad_token_id);
        let mut attention_mask = vec![0.0; self.max_length];
        attention_mask[..token_count].fill(1.0);
        Ok(TokenizedCaption {
            token_ids: ids,
            attention_mask: Some(attention_mask),
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl CaptionTokenizer for QwenTokenizer {
    fn encode(&mut self, caption: &str) -> Result<TokenizedCaption, String> {
        self.encode_caption(caption)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl Default for Qwen3TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 1_024,
            intermediate_size: 3_072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            head_dim: 128,
            num_key_value_heads: 8,
            max_position_embeddings: 32_768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
        }
    }
}

impl Qwen3TextConfig {
    pub fn validate(&self) -> Result<(), PreprocessError> {
        if self.hidden_size == 0 || self.num_hidden_layers == 0 || self.head_dim == 0 {
            return Err(PreprocessError::Encoder(
                "Qwen3 dimensions must be greater than zero".into(),
            ));
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(PreprocessError::Encoder(
                "Qwen3 attention heads must have a valid GQA ratio".into(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(PreprocessError::Encoder(
                "Qwen3 hidden_size must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    fn candle_config(&self) -> qwen3::Config {
        qwen3::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            attention_bias: false,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: None,
            max_window_layers: 0,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        }
    }
}

/// A Qwen3 encoder backed by Candle and a SafeTensors checkpoint.
pub struct CandleQwen3TextEncoder<T: CaptionTokenizer> {
    model: qwen3::Model,
    tokenizer: T,
    device: Device,
    max_tokens: usize,
}

impl<T: CaptionTokenizer> CandleQwen3TextEncoder<T> {
    /// Loads an Anima/Qwen3 SafeTensors checkpoint.
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: Qwen3TextConfig,
        dtype: DType,
        device: &Device,
        tokenizer: T,
    ) -> Result<Self, PreprocessError> {
        config.validate()?;
        let path = path.as_ref();
        if !path.is_file() {
            return Err(PreprocessError::Encoder(format!(
                "Qwen3 checkpoint does not exist: {}",
                path.display()
            )));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        let model = qwen3::Model::new(&config.candle_config(), vb)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            max_tokens: config.max_position_embeddings,
        })
    }

    /// Runs Qwen3 for already-tokenized input and returns `[batch, seq, hidden]`.
    pub fn encode_token_ids(&mut self, token_ids: &[u32]) -> Result<VaeTensor, PreprocessError> {
        if token_ids.is_empty() {
            return Err(PreprocessError::Encoder("token sequence is empty".into()));
        }
        if token_ids.len() > self.max_tokens {
            return Err(PreprocessError::Encoder(format!(
                "token sequence length {} exceeds configured maximum {}",
                token_ids.len(),
                self.max_tokens
            )));
        }
        let input = Tensor::from_slice(token_ids, (1, token_ids.len()), &self.device)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        let output = self
            .model
            .forward(&input, 0)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        VaeTensor::from_candle(&output).map_err(|error| PreprocessError::Encoder(error.to_string()))
    }

    fn encode_tokenized(
        &mut self,
        caption: TokenizedCaption,
    ) -> Result<(VaeTensor, Option<VaeTensor>), PreprocessError> {
        let embedding = self.encode_token_ids(&caption.token_ids)?;
        let mask = caption.attention_mask.map(|values| VaeTensor {
            shape: vec![1, 1, 1, values.len()],
            data: values,
        });
        Ok((embedding, mask))
    }
}

impl<T: CaptionTokenizer> TextEncoder for CandleQwen3TextEncoder<T> {
    fn encode_text(&mut self, caption: &str) -> Result<VaeTensor, PreprocessError> {
        let tokenized = self
            .tokenizer
            .encode(caption)
            .map_err(|error| PreprocessError::Encoder(format!("tokenization failed: {error}")))?;
        Ok(self.encode_tokenized(tokenized)?.0)
    }

    fn encode_text_with_mask(
        &mut self,
        caption: &str,
    ) -> Result<(VaeTensor, Option<VaeTensor>), PreprocessError> {
        let tokenized = self
            .tokenizer
            .encode(caption)
            .map_err(|error| PreprocessError::Encoder(format!("tokenization failed: {error}")))?;
        self.encode_tokenized(tokenized)
    }
}

/// Combines the frozen Qwen3 encoder with the T5 tokenizer required by the
/// Anima LLM adapter. T5 tokens are cached as IDs; no T5 model is executed.
pub struct AnimaTextEncoder<Q: CaptionTokenizer, T: CaptionTokenizer> {
    pub qwen3: CandleQwen3TextEncoder<Q>,
    pub t5_tokenizer: T,
}

impl<Q: CaptionTokenizer, T: CaptionTokenizer> AnimaTextEncoder<Q, T> {
    pub fn new(qwen3: CandleQwen3TextEncoder<Q>, t5_tokenizer: T) -> Self {
        Self {
            qwen3,
            t5_tokenizer,
        }
    }
}

impl<Q: CaptionTokenizer, T: CaptionTokenizer> TextEncoder for AnimaTextEncoder<Q, T> {
    fn encode_text(&mut self, caption: &str) -> Result<VaeTensor, PreprocessError> {
        self.qwen3.encode_text(caption)
    }

    fn encode_text_with_mask(
        &mut self,
        caption: &str,
    ) -> Result<(VaeTensor, Option<VaeTensor>), PreprocessError> {
        self.qwen3.encode_text_with_mask(caption)
    }

    fn encode_text_with_conditioning(
        &mut self,
        caption: &str,
    ) -> Result<TextEncoding, PreprocessError> {
        let (embedding, attention_mask) = self.qwen3.encode_text_with_mask(caption)?;
        let target = self.t5_tokenizer.encode(caption).map_err(|error| {
            PreprocessError::Encoder(format!("T5 tokenization failed: {error}"))
        })?;
        let target_mask = target.attention_mask.clone().map(|values| VaeTensor {
            shape: vec![1, 1, 1, values.len()],
            data: values,
        });
        let adapter_conditioning = target_mask.map(|mask| AdapterConditioning {
            input_ids: target.token_ids,
            attention_mask: mask,
        });
        Ok(TextEncoding {
            embedding,
            attention_mask,
            adapter_conditioning,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_qwen3_config_matches_anima_encoder_shape() {
        let config = Qwen3TextConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_hidden_layers, 28);
    }

    #[test]
    fn rejects_invalid_gqa_configuration() {
        let mut config = Qwen3TextConfig::default();
        config.num_key_value_heads = 3;
        assert!(config.validate().is_err());
    }
}
