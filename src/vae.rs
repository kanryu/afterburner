//! VAE boundary types shared by the Burn runtime and the Candle oracle.
//!
//! The model layers themselves will be added incrementally. This module keeps
//! the data contract and Stable Diffusion latent math in one place so that a
//! Candle implementation and a Burn implementation cannot silently disagree
//! about scaling or tensor layout.

use std::sync::atomic::{AtomicBool, Ordering};

use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::anima::QwenImageVaeConfig;
use crate::preprocess::{ImageVae, PreprocessError};

static IMAGE_STAGE_DUMP_USED: AtomicBool = AtomicBool::new(false);

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
        let tensor = if tensor.dtype() == candle_core::DType::F32 {
            tensor.clone()
        } else {
            tensor
                .to_dtype(candle_core::DType::F32)
                .map_err(|error| VaeError::Candle(error.to_string()))?
        };
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
    // Candle expects NCHW.  Python's ``ToTensor`` used by kohya_ss performs
    // the same HWC -> CHW conversion before normalization; pushing RGB
    // triplets here would silently feed an interleaved HWC buffer as CHW.
    let mut data = Vec::with_capacity(width * height * config.input_channels);
    for channel in 0..config.input_channels {
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x as u32, y as u32);
                data.push(pixel[channel] as f32 / 127.5 - 1.0);
            }
        }
    }
    VaeTensor::new(vec![1, config.input_channels, 1, height, width], data)
}

/// Applies the per-channel Qwen Image VAE latent normalization used before
/// passing image latents to Anima.
pub fn normalize_qwen_image_latent(
    latent: &VaeTensor,
    config: &QwenImageVaeConfig,
) -> Result<VaeTensor, VaeError> {
    normalize_qwen_image_latent_with_precision(latent, config, false)
}

/// Normalizes a Qwen latent with the same arithmetic boundary as kohya_ss.
///
/// `encode_pixels_to_latents` casts the posterior mode to the VAE dtype before
/// applying the channel mean/std transform.  In the normal Anima path that
/// dtype is fp16, so doing the transform in f32 changes the cached values and
/// makes a Rust cache diverge from Python even when the VAE convolution is
/// otherwise identical.
pub fn normalize_qwen_image_latent_with_precision(
    latent: &VaeTensor,
    config: &QwenImageVaeConfig,
    fp16: bool,
) -> Result<VaeTensor, VaeError> {
    config.validate().map_err(VaeError::InvalidConfig)?;
    if latent.shape.len() != 5 || latent.shape[1] != config.latent_channels {
        return Err(VaeError::ShapeMismatch {
            expected: vec![1, config.latent_channels, 1, 0, 0],
            actual: latent.shape.clone(),
        });
    }
    let mut normalized = latent.clone();
    let channel_count = latent.shape[1];
    let channel_plane = latent.shape[2..].iter().product::<usize>();
    if channel_plane == 0 {
        return Err(VaeError::InvalidConfig(
            "latent data has an empty spatial or temporal dimension".into(),
        ));
    }
    for (index, value) in normalized.data.iter_mut().enumerate() {
        // VaeTensor follows NCHW-compatible channel-major storage. The
        // channel is therefore determined by the plane index, not by
        // `index % channel_count` (which would assume interleaved channels).
        let channel = (index / channel_plane) % channel_count;
        if fp16 {
            let raw = half::f16::from_f32(*value).to_f32();
            // Match Python's `(latents - mean) * (1 / std)` fp16 sequence,
            // including the rounding after each elementary operation.
            let mean = half::f16::from_f32(config.latent_mean[channel]).to_f32();
            let inv_std = half::f16::from_f32(
                1.0 / config.latent_std[channel],
            )
            .to_f32();
            let centered = half::f16::from_f32(raw - mean).to_f32();
            *value = half::f16::from_f32(centered * inv_std).to_f32();
        } else {
            *value = (*value - config.latent_mean[channel]) / config.latent_std[channel];
        }
    }
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
    target_dimensions: Option<(u32, u32)>,
    kohya_buckets: Option<KohyaBucketConfig>,
    fp16_latent_math: bool,
}

/// Kohya_ss bucket preprocessing settings used by the Anima trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KohyaBucketConfig {
    pub max_resolution: u32,
    pub min_bucket_resolution: u32,
    pub max_bucket_resolution: u32,
    pub resolution_step: u32,
}

impl<E> QwenImageVaePreprocessor<E> {
    pub fn new(encoder: E, config: QwenImageVaeConfig) -> Result<Self, VaeError> {
        config.validate().map_err(VaeError::InvalidConfig)?;
        Ok(Self {
            encoder,
            config,
            max_dimension: None,
            target_dimensions: None,
            kohya_buckets: None,
            fp16_latent_math: false,
        })
    }

    /// Limits the longest image edge before VAE encoding. This keeps the
    /// resulting DiT token count bounded for full-attention backends.
    pub fn with_max_dimension(mut self, max_dimension: Option<u32>) -> Self {
        self.max_dimension = max_dimension.filter(|value| *value > 0);
        self
    }

    /// Forces the Kohya-compatible bucket dimensions as `(width, height)`.
    pub fn with_target_dimensions(mut self, dimensions: Option<(u32, u32)>) -> Self {
        self.target_dimensions = dimensions.filter(|(width, height)| *width > 0 && *height > 0);
        self
    }

    /// Uses Kohya_ss's `bucket_no_upscale` preprocessing semantics.
    pub fn with_kohya_buckets(
        mut self,
        max_resolution: Option<u32>,
        min_bucket_resolution: u32,
        max_bucket_resolution: u32,
        resolution_step: u32,
    ) -> Self {
        self.kohya_buckets = max_resolution
            .filter(|value| *value > 0)
            .filter(|_| min_bucket_resolution > 0 && max_bucket_resolution > 0)
            .filter(|_| resolution_step > 0)
            .map(|max_resolution| KohyaBucketConfig {
                max_resolution,
                min_bucket_resolution,
                max_bucket_resolution,
                resolution_step,
            });
        self
    }

    /// Reproduce the fp16 arithmetic used by kohya_ss after VAE encoding.
    pub fn with_fp16_latent_math(mut self, enabled: bool) -> Self {
        self.fp16_latent_math = enabled;
        self
    }
}

impl<E: QwenImageVaeEncoder> ImageVae for QwenImageVaePreprocessor<E> {
    fn encode_image(&mut self, image: &image::RgbImage) -> Result<VaeTensor, PreprocessError> {
        let image = align_qwen_image_dimensions(
            image,
            self.config.spatial_downsample,
            self.max_dimension,
            self.target_dimensions,
            self.kohya_buckets,
        );
        if let Some(directory) = std::env::var_os("AFTERBURNER_IMAGE_STAGE_DUMP_DIR")
            && !IMAGE_STAGE_DUMP_USED.swap(true, Ordering::Relaxed)
        {
            let directory = std::path::PathBuf::from(directory);
            if let Err(error) = std::fs::create_dir_all(&directory) {
                eprintln!("[image-stage] dump directory creation failed: {error}");
            } else if let Err(error) = image.save(directory.join("rust-cropped.png")) {
                eprintln!("[image-stage] cropped image dump failed: {error}");
            } else {
                match qwen_image_input(&image, &self.config) {
                    Ok(input) => {
                        let input_path = directory.join("rust-input-ncthw-f32le.bin");
                        if let Err(error) = write_f32le(&input_path, &input.data) {
                            eprintln!("[image-stage] input dump failed: {error}");
                        }
                        let min = input.data.iter().copied().fold(f32::INFINITY, f32::min);
                        let max = input.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mean = input.data.iter().sum::<f32>() / input.data.len() as f32;
                        let l2 = input.data.iter().map(|value| value * value).sum::<f32>().sqrt();
                        let shape = input
                            .shape
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(", ");
                        let metadata = format!(
                            "{{\n  \"shape\": [{shape}],\n  \"dtype\": \"f32\",\n  \"min\": {min:.9e},\n  \"max\": {max:.9e},\n  \"mean\": {mean:.9e},\n  \"l2\": {l2:.9e}\n}}\n"
                        );
                        if let Err(error) = std::fs::write(
                            directory.join("rust-input.json"),
                            metadata,
                        ) {
                            eprintln!("[image-stage] input metadata dump failed: {error}");
                        }
                    }
                    Err(error) => eprintln!("[image-stage] input conversion failed: {error}"),
                }
            }
        }
        let input = qwen_image_input(&image, &self.config)
            .map_err(|error| PreprocessError::Image(error.to_string()))?;
        let latent = self
            .encoder
            .encode_qwen_input(&input)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        normalize_qwen_image_latent_with_precision(&latent, &self.config, self.fp16_latent_math)
            .map_err(|error| PreprocessError::Encoder(error.to_string()))
    }
}

fn write_f32le(path: &std::path::Path, values: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn align_qwen_image_dimensions(
    image: &image::RgbImage,
    multiple: usize,
    max_dimension: Option<u32>,
    target_dimensions: Option<(u32, u32)>,
    kohya_buckets: Option<KohyaBucketConfig>,
) -> image::RgbImage {
    if target_dimensions.is_none() {
        if let Some(bucket_config) = kohya_buckets {
            return align_kohya_bucket_dimensions(image, bucket_config);
        }
    }
    let (width, height) = match target_dimensions {
        Some((width, height)) => (width, height),
        None => match max_dimension {
            Some(max_dimension) if image.width().max(image.height()) > max_dimension => {
                let scale = max_dimension as f32 / image.width().max(image.height()) as f32;
                (
                    ((image.width() as f32 * scale).round() as u32).max(1),
                    ((image.height() as f32 * scale).round() as u32).max(1),
                )
            }
            _ => (image.width(), image.height()),
        },
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

fn align_kohya_bucket_dimensions(
    image: &image::RgbImage,
    config: KohyaBucketConfig,
) -> image::RgbImage {
    let width = image.width() as f32;
    let height = image.height() as f32;
    let aspect_ratio = width / height;
    let max_area = config.max_resolution as u64 * config.max_resolution as u64;
    let (resized_width, resized_height) =
        if (image.width() as u64 * image.height() as u64) > max_area {
            let resized_width = (max_area as f32 * aspect_ratio).sqrt();
            let resized_height = max_area as f32 / resized_width;
            let step = config.resolution_step as f32;
            let width_rounded = (resized_width / step).floor() * step;
            let height_from_width = (width_rounded / aspect_ratio).round();
            let height_rounded = (resized_height / step).floor() * step;
            let width_from_height = (height_rounded * aspect_ratio).round();
            if ((width_rounded / height_from_width) - aspect_ratio).abs()
                <= ((width_from_height / height_rounded) - aspect_ratio).abs()
            {
                (width_rounded as u32, height_from_width as u32)
            } else {
                (width_from_height as u32, height_rounded as u32)
            }
        } else {
            (image.width(), image.height())
        };
    let step = config.resolution_step;
    let bucket_width = (resized_width / step * step)
        .clamp(config.min_bucket_resolution, config.max_bucket_resolution);
    let bucket_height = (resized_height / step * step)
        .clamp(config.min_bucket_resolution, config.max_bucket_resolution);
    let output_width = resized_width.max(bucket_width);
    let output_height = resized_height.max(bucket_height);
    let resized = if output_width < image.width() && output_height < image.height() {
        // kohya_ss defaults to OpenCV INTER_AREA for downsampling.
        resize_area_rgb(image, output_width, output_height)
    } else {
        image::imageops::resize(image, output_width, output_height, image::imageops::FilterType::Lanczos3)
    };
    let left = resized.width().saturating_sub(bucket_width) / 2;
    let top = resized.height().saturating_sub(bucket_height) / 2;
    image::imageops::crop_imm(&resized, left, top, bucket_width, bucket_height).to_image()
}

fn resize_area_rgb(image: &image::RgbImage, target_width: u32, target_height: u32) -> image::RgbImage {
    let source_width = image.width() as usize;
    let source_height = image.height() as usize;
    let target_width = target_width as usize;
    let target_height = target_height as usize;
    let mut output = image::RgbImage::new(target_width as u32, target_height as u32);
    let scale_x = source_width as f64 / target_width as f64;
    let scale_y = source_height as f64 / target_height as f64;
    for oy in 0..target_height {
        let y0 = oy as f64 * scale_y;
        let y1 = (oy + 1) as f64 * scale_y;
        let sy0 = y0.floor() as usize;
        let sy1 = (y1.ceil() as usize).min(source_height);
        for ox in 0..target_width {
            let x0 = ox as f64 * scale_x;
            let x1 = (ox + 1) as f64 * scale_x;
            let sx0 = x0.floor() as usize;
            let sx1 = (x1.ceil() as usize).min(source_width);
            let mut sums = [0.0_f64; 3];
            let mut weight_sum = 0.0_f64;
            for sy in sy0..sy1 {
                let wy = (y1.min((sy + 1) as f64) - y0.max(sy as f64)).max(0.0);
                for sx in sx0..sx1 {
                    let wx = (x1.min((sx + 1) as f64) - x0.max(sx as f64)).max(0.0);
                    let weight = wx * wy;
                    let pixel = image.get_pixel(sx as u32, sy as u32);
                    for channel in 0..3 {
                        sums[channel] += pixel[channel] as f64 * weight;
                    }
                    weight_sum += weight;
                }
            }
            let mut pixel = [0_u8; 3];
            for channel in 0..3 {
                pixel[channel] = (sums[channel] / weight_sum).round().clamp(0.0, 255.0) as u8;
            }
            output.put_pixel(ox as u32, oy as u32, image::Rgb(pixel));
        }
    }
    output
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
        let plane = 16 * 8;
        assert!(tensor.data[plane].abs() < 0.01);
        assert!((tensor.data[2 * plane] + 1.0).abs() < 1e-6);
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
        let config = QwenImageVaeConfig::default();
        assert!((latent.data[0] + config.latent_mean[0] / config.latent_std[0]).abs() < 1e-6);
    }

    #[test]
    fn qwen_latent_normalization_respects_channel_major_layout() {
        let mut config = QwenImageVaeConfig::default();
        config.latent_mean = [0.0; 16];
        config.latent_std = [1.0; 16];
        config.latent_std[1] = 2.0;
        let latent = VaeTensor::new(vec![1, 16, 1, 1, 2], {
            let mut data = vec![0.0; 32];
            data[0..2].copy_from_slice(&[1.0, 2.0]);
            data[2..4].copy_from_slice(&[3.0, 4.0]);
            data
        })
        .unwrap();
        let normalized = normalize_qwen_image_latent(&latent, &config).unwrap();
        assert_eq!(&normalized.data[0..2], &[1.0, 2.0]);
        assert_eq!(&normalized.data[2..4], &[1.5, 2.0]);
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

    #[test]
    fn qwen_preprocessor_can_use_kohya_bucket_dimensions() {
        let image = image::RgbImage::from_pixel(1024, 1365, image::Rgb([0, 0, 0]));
        let aligned = align_qwen_image_dimensions(
            &image,
            8,
            Some(1024),
            None,
            Some(KohyaBucketConfig {
                max_resolution: 1024,
                min_bucket_resolution: 256,
                max_bucket_resolution: 2048,
                resolution_step: 64,
            }),
        );
        assert_eq!(aligned.dimensions(), (832, 1088));
    }
}
