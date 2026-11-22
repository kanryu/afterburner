//! Candle Qwen3 text-encoder adapter used during cache construction.
//!
//! Tokenization is intentionally injected. Anima deployments can use the
//! matching Qwen tokenizer without making the core engine depend on a
//! particular tokenizer file format.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use tokenizers::{
    SplitDelimiterBehavior, Tokenizer,
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BPE,
    normalizers::unicode::NFC,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

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
        let mut attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        ids.truncate(self.max_length);
        attention_mask.truncate(self.max_length);
        ids.resize(self.max_length, self.pad_token_id);
        attention_mask.resize(self.max_length, 0.0);
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
        // Qwen3's Transformers tokenizer is the Qwen2 slow tokenizer.  When
        // its directory contains vocab.json and merges.txt, loading the
        // tokenizer.json directly can produce a different tokenization (in
        // particular, it may add a BOS token and expand a short caption to
        // hundreds of byte-level tokens).  Reconstruct the BPE exactly from
        // the files used by Transformers.
        let tokenizer = match path.parent() {
            Some(directory) => {
                let vocab = directory.join("vocab.json");
                let merges = directory.join("merges.txt");
                if vocab.is_file() && merges.is_file() {
                    let vocab_string = vocab.to_string_lossy();
                    let merges_string = merges.to_string_lossy();
                    let bpe = BPE::from_file(&vocab_string, &merges_string)
                        .byte_fallback(false)
                        .build()
                        .map_err(|error| {
                            format!(
                                "failed to build Qwen BPE from {} and {}: {error}",
                                vocab.display(),
                                merges.display()
                            )
                        })?;
                    let qwen_pretokenize_regex = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
                    let split = Split::new(
                        SplitPattern::Regex(qwen_pretokenize_regex.to_owned()),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )
                    .map_err(|error| format!("failed to build Qwen pre-tokenizer: {error}"))?;
                    let byte_level = ByteLevel::default()
                        .add_prefix_space(false)
                        .use_regex(false);
                    let mut tokenizer = Tokenizer::new(bpe);
                    tokenizer.with_normalizer(Some(NFC));
                    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
                        PreTokenizerWrapper::Split(split),
                        PreTokenizerWrapper::ByteLevel(byte_level),
                    ])));
                    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
                    tokenizer
                } else {
                    Tokenizer::from_file(path).map_err(|error| {
                        format!("failed to load tokenizer {}: {error}", path.display())
                    })?
                }
            }
            None => Tokenizer::from_file(path)
                .map_err(|error| format!("failed to load tokenizer {}: {error}", path.display()))?,
        };
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
        let mut attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        ids.truncate(self.max_length);
        attention_mask.truncate(self.max_length);
        ids.resize(self.max_length, self.pad_token_id);
        attention_mask.resize(self.max_length, 0.0);
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
    initial_model: qwen3::Model,
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
            initial_model: model.clone(),
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
        // Each cache sample is an independent sequence. Candle's Qwen3
        // implementation keeps a KV cache for incremental decoding but does
        // not expose a public reset method, so restore the empty model state
        // before encoding the next caption at offset zero.
        self.model = self.initial_model.clone();
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
        let mut embedding = self.encode_token_ids(&caption.token_ids)?;
        // kohya_ss masks padded Qwen positions after the transformer forward:
        // ``prompt_embeds[~qwen3_attn_mask.bool()] = 0``.  Returning the mask
        // without applying it leaves pad-token activations in the cache and
        // changes both the adapter input and its statistics.
        if let Some(mask) = &caption.attention_mask {
            if mask.len() != embedding.shape[1] {
                return Err(PreprocessError::Encoder(format!(
                    "Qwen attention mask length {} does not match embedding sequence length {}",
                    mask.len(),
                    embedding.shape[1]
                )));
            }
            let hidden = embedding.shape[2];
            for (token, &value) in mask.iter().enumerate() {
                if value == 0.0 {
                    let start = token * hidden;
                    let end = start + hidden;
                    embedding.data[start..end].fill(0.0);
                }
            }
        }
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
    use crate::preprocess::TextEncoder;
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

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

    #[test]
    fn tokenizer_contract_pads_and_masks_deterministically() {
        let vocab = [
            ("[UNK]".to_owned(), 0_u32),
            ("Elvira".to_owned(), 1),
            ("Atelier".to_owned(), 2),
            ("Sophie".to_owned(), 3),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let path =
            std::env::temp_dir().join(format!("afterburner-tokenizer-{}.json", std::process::id()));
        tokenizer.save(&path, false).unwrap();

        let qwen = QwenTokenizer::from_file(&path, 5, 99).unwrap();
        let qwen_encoded = qwen.encode_caption("Elvira Atelier Sophie").unwrap();
        assert_eq!(qwen_encoded.token_ids.len(), 5);
        assert_eq!(
            qwen_encoded.attention_mask,
            Some(vec![1.0, 1.0, 1.0, 0.0, 0.0])
        );
        assert_eq!(&qwen_encoded.token_ids[3..], &[99, 99]);

        let t5 = T5Tokenizer::from_file(&path, 2, 0).unwrap();
        let t5_encoded = t5.encode_caption("Elvira Atelier Sophie").unwrap();
        assert_eq!(t5_encoded.token_ids.len(), 2);
        assert_eq!(t5_encoded.attention_mask, Some(vec![1.0, 1.0]));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "loads the full Qwen3 model; set model/tokenizer paths in the test before running"]
    fn qwen3_and_t5_conditioning_matches_anima_cache_contract() {
        let qwen_model = std::env::var_os("AFTERBURNER_QWEN3_PATH")
            .expect("AFTERBURNER_QWEN3_PATH must point to qwen_3_06b_base.safetensors");
        let qwen_tokenizer = std::env::var_os("AFTERBURNER_QWEN_TOKENIZER_PATH")
            .expect("AFTERBURNER_QWEN_TOKENIZER_PATH must point to tokenizer.json");
        let t5_tokenizer = std::env::var_os("AFTERBURNER_T5_TOKENIZER_PATH")
            .expect("AFTERBURNER_T5_TOKENIZER_PATH must point to tokenizer.json");
        let qwen = QwenTokenizer::from_file(qwen_tokenizer, 512, 151_643).unwrap();
        let t5 = T5Tokenizer::from_file(t5_tokenizer, 512, 0).unwrap();
        let device = Device::Cpu;
        let qwen = CandleQwen3TextEncoder::from_safetensors(
            qwen_model,
            Qwen3TextConfig::default(),
            DType::F32,
            &device,
            qwen,
        )
        .unwrap();
        let mut encoder = AnimaTextEncoder::new(qwen, t5);
        let encoding = encoder
            .encode_text_with_conditioning("Elvira, Atelier Sophie 2")
            .unwrap();
        assert_eq!(encoding.embedding.shape, vec![1, 512, 1024]);
        assert_eq!(encoding.attention_mask.unwrap().shape, vec![1, 1, 1, 512]);
        let adapter = encoding.adapter_conditioning.unwrap();
        assert_eq!(adapter.input_ids.len(), 512);
        assert_eq!(adapter.attention_mask.shape, vec![1, 1, 1, 512]);
        assert!(
            adapter
                .attention_mask
                .data
                .iter()
                .any(|value| *value == 1.0)
        );
    }
}
