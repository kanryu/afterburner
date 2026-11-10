//! Model-type registry and configuration dispatch.

use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::anima::AnimaConfig;

/// Supported diffusion model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Anima,
    Flux,
    StableDiffusion,
}

impl ModelType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anima => "anima",
            Self::Flux => "flux",
            Self::StableDiffusion => "stable-diffusion",
        }
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ModelType {
    type Err = ModelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "anima" => Ok(Self::Anima),
            "flux" => Ok(Self::Flux),
            "stable-diffusion" | "stable_diffusion" | "sd" => Ok(Self::StableDiffusion),
            _ => Err(ModelError::UnknownType(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    UnknownType(String),
    InvalidConfig(String),
    MissingFileName(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownType(value) => write!(f, "unknown model type: {value}"),
            Self::InvalidConfig(message) => write!(f, "invalid model configuration: {message}"),
            Self::MissingFileName(path) => write!(f, "model path has no file name: {path}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Stable cache identity for a model component set.
///
/// The identity intentionally uses file names rather than absolute paths so
/// that the same model bundle can share a cache across installations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCacheKey {
    pub model_type: ModelType,
    pub diffusion_model: String,
    pub text_encoder: String,
    pub vae: String,
}

impl ModelCacheKey {
    pub fn from_paths(
        model_type: ModelType,
        diffusion_model: impl AsRef<Path>,
        text_encoder: impl AsRef<Path>,
        vae: impl AsRef<Path>,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            model_type,
            diffusion_model: file_name(diffusion_model.as_ref())?,
            text_encoder: file_name(text_encoder.as_ref())?,
            vae: file_name(vae.as_ref())?,
        })
    }

    /// Returns a filesystem-safe, deterministic cache file name.
    pub fn file_name(&self) -> String {
        format!(
            "{}--{}--{}--{}.afbcache",
            self.model_type,
            cache_component(&self.diffusion_model),
            cache_component(&self.text_encoder),
            cache_component(&self.vae),
        )
    }

    pub fn path_in(&self, directory: impl AsRef<Path>) -> PathBuf {
        directory.as_ref().join(self.file_name())
    }
}

fn file_name(path: &Path) -> Result<String, ModelError> {
    path.file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ModelError::MissingFileName(path.display().to_string()))
}

fn cache_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// File-level configuration shared by model implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub revision: String,
    pub diffusion_model: String,
    pub text_encoder: String,
    pub vae: String,
}

impl ModelConfig {
    pub fn new(
        model_type: ModelType,
        revision: impl Into<String>,
        diffusion_model: impl Into<String>,
        text_encoder: impl Into<String>,
        vae: impl Into<String>,
    ) -> Self {
        Self {
            model_type,
            revision: revision.into(),
            diffusion_model: diffusion_model.into(),
            text_encoder: text_encoder.into(),
            vae: vae.into(),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        for (name, value) in [
            ("revision", &self.revision),
            ("diffusion_model", &self.diffusion_model),
            ("text_encoder", &self.text_encoder),
            ("vae", &self.vae),
        ] {
            if value.trim().is_empty() {
                return Err(ModelError::InvalidConfig(format!(
                    "{name} must not be empty"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelDescriptor {
    Anima {
        config: AnimaConfig,
        files: ModelConfig,
    },
    Flux {
        files: ModelConfig,
    },
    StableDiffusion {
        files: ModelConfig,
    },
}

/// Resolves the generic model configuration to a model-specific descriptor.
pub fn resolve_model(config: ModelConfig) -> Result<ModelDescriptor, ModelError> {
    config.validate()?;
    Ok(match config.model_type {
        ModelType::Anima => {
            let anima = AnimaConfig::default();
            anima.validate().map_err(ModelError::InvalidConfig)?;
            ModelDescriptor::Anima {
                config: anima,
                files: config,
            }
        }
        ModelType::Flux => ModelDescriptor::Flux { files: config },
        ModelType::StableDiffusion => ModelDescriptor::StableDiffusion { files: config },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model_type: ModelType) -> ModelConfig {
        ModelConfig::new(
            model_type,
            "test",
            "dit.safetensors",
            "text.safetensors",
            "vae.safetensors",
        )
    }

    #[test]
    fn parses_model_type_keys() {
        assert_eq!("anima".parse::<ModelType>().unwrap(), ModelType::Anima);
        assert_eq!(ModelType::Anima.to_string(), "anima");
        assert_eq!(
            "sd".parse::<ModelType>().unwrap(),
            ModelType::StableDiffusion
        );
    }

    #[test]
    fn resolves_anima_to_anima_descriptor() {
        assert!(matches!(
            resolve_model(config(ModelType::Anima)).unwrap(),
            ModelDescriptor::Anima { .. }
        ));
    }

    #[test]
    fn rejects_missing_component_path() {
        let invalid = ModelConfig::new(
            ModelType::Anima,
            "test",
            "",
            "text.safetensors",
            "vae.safetensors",
        );
        assert!(matches!(
            resolve_model(invalid),
            Err(ModelError::InvalidConfig(_))
        ));
    }

    #[test]
    fn model_cache_key_uses_component_file_names() {
        let key = ModelCacheKey::from_paths(
            ModelType::Anima,
            r"C:\models\anima.safetensors",
            r"D:\encoders\qwen.safetensors",
            r"D:\vae\image_vae.safetensors",
        )
        .unwrap();
        assert_eq!(key.diffusion_model, "anima.safetensors");
        assert_eq!(
            key.file_name(),
            "anima--anima.safetensors--qwen.safetensors--image_vae.safetensors.afbcache"
        );
        assert_eq!(
            key.path_in("target/cache"),
            PathBuf::from(
                "target/cache/anima--anima.safetensors--qwen.safetensors--image_vae.safetensors.afbcache"
            )
        );
    }
}
