//! Model-type registry and configuration dispatch.

use std::{fmt, str::FromStr};

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
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownType(value) => write!(f, "unknown model type: {value}"),
            Self::InvalidConfig(message) => write!(f, "invalid model configuration: {message}"),
        }
    }
}

impl std::error::Error for ModelError {}

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
}
