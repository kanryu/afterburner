//! VAE boundary types shared by the Burn runtime and the Candle oracle.
//!
//! The model layers themselves will be added incrementally. This module keeps
//! the data contract and Stable Diffusion latent math in one place so that a
//! Candle implementation and a Burn implementation cannot silently disagree
//! about scaling or tensor layout.

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::anima::QwenImageVaeConfig;
use crate::preprocess::{ImageVae, PreprocessError};

/// Errors returned while preparing VAE inputs and outputs.
#[derive(Debug, Clone, PartialEq)]
pub enum VaeError {
    /// A VAE configuration or scaling factor is invalid.
    InvalidConfig(String),
    /// A tensor shape does not match the expected rank or element count.
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Candle failed to materialize or read a tensor.
    Candle(String),
}

impl std::fmt::Display for VaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid VAE configuration: {message}"),
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "VAE shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::Candle(message) => write!(f, "Candle tensor error: {message}"),
        }
    }
}

impl std::error::Error for VaeError {}

/// Stable Diffusion VAE latent scaling parameters.
///
/// Encoding applies `latent * scaling_factor + shift_factor`; decoding applies
/// the inverse operation. SD 1.x and SD 2.x normally use a zero shift factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentScaling {
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl LatentScaling {
    pub const SD15: Self = Self {
        scaling_factor: 0.18215,
        shift_factor: 0.0,
    };

    pub fn validate(self) -> Result<Self, VaeError> {
        if !self.scaling_factor.is_finite() || self.scaling_factor == 0.0 {
            return Err(VaeError::InvalidConfig(
                "scaling_factor must be finite and non-zero".into(),
            ));
        }
        if !self.shift_factor.is_finite() {
            return Err(VaeError::InvalidConfig(
                "shift_factor must be finite".into(),
            ));
        }
        Ok(self)
    }

    pub fn encode(self, value: f32) -> f32 {
        value * self.scaling_factor + self.shift_factor
    }

    pub fn decode(self, value: f32) -> f32 {
        (value - self.shift_factor) / self.scaling_factor
    }
}

/// Configuration shared by the VAE encoder and decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub sample_size: usize,
    pub scaling: LatentScaling,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            sample_size: 512,
            scaling: LatentScaling::SD15,
        }
    }
}

impl VaeConfig {
    pub fn validate(&self) -> Result<(), VaeError> {
        if self.in_channels == 0 || self.out_channels == 0 || self.latent_channels == 0 {
            return Err(VaeError::InvalidConfig(
                "channel counts must be greater than zero".into(),
            ));
        }
        if self.sample_size == 0 {
            return Err(VaeError::InvalidConfig(
                "sample_size must be greater than zero".into(),
            ));
        }
        self.scaling.validate().map(|_| ())
    }
}

/// A CPU-side interchange representation used at the Candle/Burn boundary.
///
/// It is deliberately explicit about shape and uses NCHW-compatible flat data.
/// The representation is for transfer and comparison; model execution should
/// happen on Burn tensors once the Burn VAE layers are in place.
#[derive(Debug, Clone, PartialEq)]
pub struct VaeTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl VaeTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, VaeError> {
        let expected = shape.iter().product();
        if expected != data.len() {
            return Err(VaeError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![data.len()],
            });
        }
        Ok(Self { shape, data })
    }

    /// Copies a Candle tensor into the neutral interchange representation.
    pub fn from_candle(tensor: &candle_core::Tensor) -> Result<Self, VaeError> {
        let shape = tensor.dims().to_vec();
        let data = tensor
            .flatten_all()
            .map_err(|error| VaeError::Candle(error.to_string()))?
            .to_vec1::<f32>()
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        Self::new(shape, data)
    }

    /// Copies the neutral representation into a Candle tensor.
    pub fn to_candle(&self, device: &candle_core::Device) -> Result<candle_core::Tensor, VaeError> {
        candle_core::Tensor::from_vec(self.data.clone(), self.shape.as_slice(), device)
            .map_err(|error| VaeError::Candle(error.to_string()))
    }

    /// Copies the neutral representation into a Burn tensor on the requested backend.
    pub fn to_burn<const D: usize, B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, VaeError> {
        let shape: [usize; D] =
            self.shape
                .clone()
                .try_into()
                .map_err(|_| VaeError::ShapeMismatch {
                    expected: vec![D],
                    actual: vec![self.shape.len()],
                })?;
        Ok(Tensor::from_data(
            TensorData::new(self.data.clone(), shape).convert::<B::FloatElem>(),
            device,
        ))
    }

    pub fn map_values(&self, function: impl Fn(f32) -> f32) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(function).collect(),
        }
    }
}

/// Converts an RGB image to the Qwen Image VAE's still-image input layout:
/// `[batch, channels, time, height, width]`, with values in `[-1, 1]`.
pub fn qwen_image_input(
    image: &image::RgbImage,
    config: &QwenImageVaeConfig,
) -> Result<VaeTensor, VaeError> {
    config.validate().map_err(VaeError::InvalidConfig)?;
    let width = image.width() as usize;
    let height = image.height() as usize;
    if width == 0
        || height == 0
        || width % config.spatial_downsample != 0
        || height % config.spatial_downsample != 0
    {
        return Err(VaeError::InvalidConfig(format!(
            "Qwen Image VAE input must have non-zero dimensions divisible by {}, got {}x{}",
            config.spatial_downsample, width, height
        )));
    }
    let mut data = Vec::with_capacity(width * height * config.input_channels);
    for pixel in image.pixels() {
        data.push(pixel[0] as f32 / 127.5 - 1.0);
        data.push(pixel[1] as f32 / 127.5 - 1.0);
        data.push(pixel[2] as f32 / 127.5 - 1.0);
    }
    VaeTensor::new(vec![1, config.input_channels, 1, height, width], data)
}

/// Applies the per-channel Qwen Image VAE latent normalization used before
/// passing image latents to Anima.
pub fn normalize_qwen_image_latent(
    latent: &VaeTensor,
    config: &QwenImageVaeConfig,
) -> Result<VaeTensor, VaeError> {
    config.validate().map_err(VaeError::InvalidConfig)?;
    if latent.shape.len() != 5 || latent.shape[1] != config.latent_channels {
        return Err(VaeError::ShapeMismatch {
            expected: vec![1, config.latent_channels, 1, 0, 0],
            actual: latent.shape.clone(),
        });
    }
    let mut normalized = latent.clone();
    config
        .normalize_latent(&mut normalized.data, latent.shape[1])
        .map_err(VaeError::InvalidConfig)?;
    Ok(normalized)
}

/// Runtime boundary for a Qwen Image VAE encoder implementation.
pub trait QwenImageVaeEncoder {
    fn encode_qwen_input(&mut self, input: &VaeTensor) -> Result<VaeTensor, VaeError>;
}

/// Adapts a Qwen Image VAE encoder to the generic image preprocessing API.
pub struct QwenImageVaePreprocessor<E> {
    pub encoder: E,
    pub config: QwenImageVaeConfig,
    max_dimension: Option<u32>,
}

impl<E> QwenImageVaePreprocessor<E> {
    pub fn new(encoder: E, config: QwenImageVaeConfig) -> Result<Self, VaeError> {
        config.validate().map_err(VaeError::InvalidConfig)?;
        Ok(Self {
            encoder,
            config,
            max_dimension: None,
        })
    }

    /// Limits the longest image edge before VAE encoding. This keeps the
    /// resulting DiT token count bounded for full-attention backends.
    pub fn with_max_dimension(mut self, max_dimension: Option<u32>) -> Self {
        self.max_dimension = max_dimension.filter(|value| *value > 0);
        self
    }
}

impl<E: QwenImageVaeEncoder> ImageVae for QwenImageVaePreprocessor<E> {
    fn encode_image(&mut self, image: &image::RgbImage) -> Result<VaeTensor, PreprocessError> {
        let image =
            align_qwen_image_dimensions(image, self.config.spatial_downsample, self.max_dimension);
        let input = qwen_image_input(&image, &self.config)
            .map_err(|error| PreprocessError::Image(error.to_string()))?;
        let latent = self
            .encoder
            .encode_qwen_input(&input)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        normalize_qwen_image_latent(&latent, &self.config)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))
    }
}

fn align_qwen_image_dimensions(
    image: &image::RgbImage,
    multiple: usize,
    max_dimension: Option<u32>,
) -> image::RgbImage {
    let (width, height) = match max_dimension {
        Some(max_dimension) if image.width().max(image.height()) > max_dimension => {
            let scale = max_dimension as f32 / image.width().max(image.height()) as f32;
            (
                ((image.width() as f32 * scale).round() as u32).max(1),
                ((image.height() as f32 * scale).round() as u32).max(1),
            )
        }
        _ => (image.width(), image.height()),
    };
    let multiple = multiple as u32;
    let aligned_width = (width / multiple).max(1) * multiple;
    let aligned_height = (height / multiple).max(1) * multiple;
    if image.width() == aligned_width && image.height() == aligned_height {
        return image.clone();
    }
    image::imageops::resize(
        image,
        aligned_width,
        aligned_height,
        image::imageops::FilterType::Lanczos3,
    )
}

/// Coordinates the data contract between the VAE encoder/decoder and the two
/// runtimes used during the migration period.
#[derive(Debug, Clone, PartialEq)]
pub struct VaePipeline {
    pub config: VaeConfig,
}

impl VaePipeline {
    pub fn new(config: VaeConfig) -> Result<Self, VaeError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Converts a raw VAE encoder sample into diffusion-model latents.
    pub fn encode_latent(&self, sample: &VaeTensor) -> VaeTensor {
        sample.map_values(|value| self.config.scaling.encode(value))
    }

    /// Converts diffusion-model latents into the input expected by the VAE decoder.
    pub fn decode_latent(&self, latent: &VaeTensor) -> VaeTensor {
        latent.map_values(|value| self.config.scaling.decode(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_scaling_round_trips() {
        let scaling = LatentScaling {
            scaling_factor: 0.18215,
            shift_factor: 0.5,
        };
        let value = -1.25;
        let encoded = scaling.encode(value);
        assert!((scaling.decode(encoded) - value).abs() < 1e-6);
    }

    #[test]
    fn candle_and_neutral_tensor_round_trip() {
        let device = candle_core::Device::Cpu;
        let candle =
            candle_core::Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device)
                .unwrap();
        let neutral = VaeTensor::from_candle(&candle).unwrap();
        let restored = neutral.to_candle(&device).unwrap();
        assert_eq!(neutral.shape, vec![1, 1, 2, 2]);
        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn rejects_inconsistent_tensor_data() {
        let error = VaeTensor::new(vec![1, 2, 2], vec![0.0; 3]).unwrap_err();
        assert!(matches!(error, VaeError::ShapeMismatch { .. }));
    }

    #[test]
    fn prepares_qwen_image_input_in_ncthw_layout() {
        let config = QwenImageVaeConfig::default();
        let image = image::RgbImage::from_pixel(8, 16, image::Rgb([255, 127, 0]));
        let tensor = qwen_image_input(&image, &config).unwrap();
        assert_eq!(tensor.shape, vec![1, 3, 1, 16, 8]);
        assert!((tensor.data[0] - 1.0).abs() < 1e-6);
        assert!(tensor.data[1].abs() < 0.01);
        assert!((tensor.data[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_qwen_image_dimensions_not_divisible_by_eight() {
        let image = image::RgbImage::from_pixel(7, 8, image::Rgb([0, 0, 0]));
        assert!(qwen_image_input(&image, &QwenImageVaeConfig::default()).is_err());
    }

    struct IdentityEncoder;

    impl QwenImageVaeEncoder for IdentityEncoder {
        fn encode_qwen_input(&mut self, _input: &VaeTensor) -> Result<VaeTensor, VaeError> {
            VaeTensor::new(vec![1, 16, 1, 1, 1], vec![0.0; 16])
        }
    }

    #[test]
    fn qwen_vae_preprocessor_adapts_encoder_to_image_api() {
        let mut preprocessor =
            QwenImageVaePreprocessor::new(IdentityEncoder, QwenImageVaeConfig::default()).unwrap();
        let image = image::RgbImage::from_pixel(8, 8, image::Rgb([0, 0, 0]));
        let latent = ImageVae::encode_image(&mut preprocessor, &image).unwrap();
        assert_eq!(latent.shape, vec![1, 16, 1, 1, 1]);
    }

    #[test]
    fn qwen_vae_preprocessor_aligns_non_multiple_image_dimensions() {
        let mut preprocessor =
            QwenImageVaePreprocessor::new(IdentityEncoder, QwenImageVaeConfig::default()).unwrap();
        let image = image::RgbImage::from_pixel(1024, 1365, image::Rgb([0, 0, 0]));
        let latent = ImageVae::encode_image(&mut preprocessor, &image).unwrap();
        assert_eq!(latent.shape, vec![1, 16, 1, 1, 1]);
    }

    #[test]
    fn qwen_vae_preprocessor_can_limit_longest_image_edge() {
        let mut preprocessor =
            QwenImageVaePreprocessor::new(IdentityEncoder, QwenImageVaeConfig::default())
                .unwrap()
                .with_max_dimension(Some(512));
        let image = image::RgbImage::from_pixel(1024, 1365, image::Rgb([0, 0, 0]));
        let latent = ImageVae::encode_image(&mut preprocessor, &image).unwrap();
        assert_eq!(latent.shape, vec![1, 16, 1, 1, 1]);
    }
}
