//! Afterburner AI execution engine.
//!
//! The initial crate intentionally keeps the public API small. Burn is the
//! target runtime, while Candle is used during the migration period as an
//! implementation reference and compatibility oracle.

pub mod anima;
pub mod cache;
pub mod dataset;
pub mod diffusion;
pub mod dit;
pub mod logging;
pub mod lora;
pub mod model;
pub mod preprocess;
pub mod qwen_vae;
pub mod text;
pub mod training;
pub mod vae;
pub mod weights;

pub use anima::{AnimaConfig, AnimaWeightKind, QwenImageVaeConfig};
pub use cache::{AdapterConditioning, CacheError, CachedTrainingSample, InMemoryTrainingCache};
pub use dataset::{DatasetError, ImageCaptionDataset, TrainingSample};
pub use diffusion::{
    DiffusionError, DiffusionSchedule, LossWeighting, TrainingExample, TrainingTarget,
};
pub use dit::{
    AnimaDiT, AnimaDiTBlock, AnimaDiTBlockConfig, AnimaDiTConfigError, AnimaLlmAdapter,
    AnimaLlmAdapterBlock, AnimaRmsNorm, AnimaTensorError, AnimaTimestepEmbedder, patchify_latent,
    unpatchify_latent,
};
pub use logging::{KohyaLogger, NullLogger, StderrLogger, TrainingLogger, log_preparation};
pub use lora::{LoraConfig, LoraError, LoraLinear, LoraLinearConfig, LoraTarget, LoraTargetKind};
pub use model::{
    ModelCacheKey, ModelConfig, ModelDescriptor, ModelError, ModelType, resolve_model,
};
pub use preprocess::{
    CompositePreprocessor, ImageVae, PreprocessError, TextEncoder, TextEncoding,
    TrainingPreprocessor, preprocess_dataset, preprocess_dataset_with_logger,
};
pub use qwen_vae::CandleQwenImageVaeStem;
pub use text::{
    AnimaTextEncoder, CandleQwen3TextEncoder, CaptionTokenizer, Qwen3TextConfig, QwenTokenizer,
    T5Tokenizer, TokenizedCaption,
};
pub use training::{
    LearningRatePolicy, TrainingError, TrainingReport, optimizer_step, optimizer_step_with_config,
    train_cached, train_cached_with_options, train_cached_with_options_and_logger,
    train_cached_with_options_and_logger_and_checkpoint,
    train_cached_with_options_and_logger_and_checkpoint_with_weighting,
    train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy,
    train_cached_with_options_and_logger_and_checkpoint_with_weighting_and_policy_and_step_hook,
    train_cached_with_target, training_loss, training_loss_with_weighting,
};
pub use vae::{
    LatentScaling, QwenImageVaeEncoder, QwenImageVaePreprocessor, VaeConfig, VaeError, VaePipeline,
    VaeTensor, normalize_qwen_image_latent, qwen_image_input,
};
pub use weights::{
    AnimaLoadReport, AnimaLoraFactorPlan, AnimaLoraPlan, AnimaWeightCoverage, AnimaWeightSlot,
    LoraOutputDtype, QwenImageVaeCoverage, WeightError, export_anima_lora,
    export_anima_lora_with_config, export_anima_lora_with_config_and_dtype,
    inspect_anima_lora_factors, inspect_anima_lora_file, inspect_anima_weight_coverage,
    inspect_qwen_image_vae, load_anima_lora_factors, load_anima_lora_linear, load_anima_model,
    load_anima_weights, map_anima_weight_key,
};

/// Returns the current Afterburner package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the version of the Afterburner runtime through the C ABI.
///
/// The exported symbol is intentionally small for now. It provides a stable
/// smoke-test entry point while the engine API is being developed.
#[unsafe(no_mangle)]
pub extern "C" fn afterburner_version() -> *const std::ffi::c_char {
    c"0.1.0".as_ptr()
}
