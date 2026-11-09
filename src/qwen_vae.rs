//! Incremental Candle implementation of the Qwen Image VAE encoder.
//!
//! Still-image training uses a singleton temporal axis. For a causal 3D
//! convolution with T=1, the final temporal kernel slice is the only slice
//! that contributes, so the first migration step can execute it as a 2D
//! convolution while preserving the checkpoint layout.

use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, ops::softmax_last_dim};

use crate::{
    QwenImageVaeConfig,
    preprocess::PreprocessError,
    vae::{QwenImageVaeEncoder, VaeError, VaeTensor, normalize_qwen_image_latent},
};

#[derive(Clone, Debug)]
struct CausalConv2d {
    conv: Conv2d,
}

#[derive(Clone, Debug)]
struct QwenRmsNorm2d {
    gamma: Tensor,
    scale: f64,
}

impl QwenRmsNorm2d {
    fn load(vb: &VarBuilder, key: &str, channels: usize) -> Result<Self, String> {
        let gamma = vb
            .pp(key)
            .get_unchecked("gamma")
            .map_err(|error| error.to_string())?;
        if gamma.elem_count() != channels {
            return Err(format!(
                "{key}.gamma has {} elements, expected {channels}",
                gamma.elem_count()
            ));
        }
        let gamma = gamma
            .reshape((1, channels, 1, 1))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            gamma,
            scale: (channels as f64).sqrt(),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let norm = input
            .sqr()
            .map_err(|error| error.to_string())?
            .sum(1)
            .map_err(|error| error.to_string())?
            .sqrt()
            .map_err(|error| error.to_string())?
            .unsqueeze(1)
            .map_err(|error| error.to_string())?;
        input
            .broadcast_div(&norm)
            .and_then(|value| value.affine(self.scale, 0.0))
            .and_then(|value| value.broadcast_mul(&self.gamma))
            .map_err(|error| error.to_string())
    }
}

/// A Qwen Image residual block for the singleton-temporal image path.
#[derive(Clone, Debug)]
pub struct CandleQwenImageResidualBlock2d {
    norm1: QwenRmsNorm2d,
    conv1: CausalConv2d,
    norm2: QwenRmsNorm2d,
    conv2: CausalConv2d,
    shortcut: Option<CausalConv2d>,
}

impl CandleQwenImageResidualBlock2d {
    pub fn load(
        vb: &VarBuilder,
        key: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, String> {
        let scope = vb.pp(key);
        let shortcut = if scope.contains_tensor("shortcut.weight") {
            Some(CausalConv2d::load(
                vb,
                &format!("{key}.shortcut"),
                input_channels,
                output_channels,
                1,
                1,
                1,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: QwenRmsNorm2d::load(vb, &format!("{key}.residual.0"), input_channels)?,
            conv1: CausalConv2d::load(
                vb,
                &format!("{key}.residual.2"),
                input_channels,
                output_channels,
                3,
                3,
                1,
            )?,
            norm2: QwenRmsNorm2d::load(vb, &format!("{key}.residual.3"), output_channels)?,
            conv2: CausalConv2d::load(
                vb,
                &format!("{key}.residual.6"),
                output_channels,
                output_channels,
                3,
                3,
                1,
            )?,
            shortcut,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let shortcut = match &self.shortcut {
            Some(layer) => layer.forward(input)?,
            None => input.clone(),
        };
        let hidden = self
            .norm1
            .forward(input)?
            .silu()
            .map_err(|error| error.to_string())?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self
            .norm2
            .forward(&hidden)?
            .silu()
            .map_err(|error| error.to_string())?;
        let hidden = self.conv2.forward(&hidden)?;
        hidden.add(&shortcut).map_err(|error| error.to_string())
    }
}

impl CausalConv2d {
    fn load(
        vb: &VarBuilder,
        key: &str,
        input_channels: usize,
        output_channels: usize,
        temporal_kernel: usize,
        spatial_kernel: usize,
        stride: usize,
    ) -> Result<Self, String> {
        let scope = vb.pp(key);
        let weight = scope
            .get(
                (
                    output_channels,
                    input_channels,
                    temporal_kernel,
                    spatial_kernel,
                    spatial_kernel,
                ),
                "weight",
            )
            .map_err(|error| error.to_string())?;
        let weight = weight
            .narrow(2, temporal_kernel.saturating_sub(1), 1)
            .map_err(|error| error.to_string())?
            .squeeze(2)
            .map_err(|error| error.to_string())?;
        let bias = scope
            .get(output_channels, "bias")
            .map_err(|error| error.to_string())?;
        Ok(Self {
            conv: Conv2d::new(
                weight,
                Some(bias),
                Conv2dConfig {
                    padding: spatial_kernel / 2,
                    stride,
                    ..Default::default()
                },
            ),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        self.conv.forward(input).map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug)]
struct SpatialConv2d {
    conv: Conv2d,
}

impl SpatialConv2d {
    fn load(
        vb: &VarBuilder,
        key: &str,
        input_channels: usize,
        output_channels: usize,
    ) -> Result<Self, String> {
        let scope = vb.pp(key);
        let weight = scope
            .get((output_channels, input_channels, 3, 3), "weight")
            .map_err(|error| error.to_string())?;
        let bias = scope
            .get(output_channels, "bias")
            .map_err(|error| error.to_string())?;
        Ok(Self {
            conv: Conv2d::new(
                weight,
                Some(bias),
                Conv2dConfig {
                    padding: 0,
                    stride: 2,
                    ..Default::default()
                },
            ),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let input = input
            .pad_with_zeros(2, 0, 1)
            .and_then(|value| value.pad_with_zeros(3, 0, 1))
            .map_err(|error| error.to_string())?;
        self.conv.forward(&input).map_err(|error| error.to_string())
    }
}

/// Spatial downsample used by the still-image encoder path.
#[derive(Clone, Debug)]
pub struct CandleQwenImageDownsample2d {
    spatial: SpatialConv2d,
    temporal: Option<CausalConv2d>,
}

#[derive(Clone, Debug)]
struct QwenImageAttention2d {
    norm: QwenRmsNorm2d,
    to_qkv: Conv2d,
    proj: Conv2d,
    channels: usize,
}

impl QwenImageAttention2d {
    fn load(vb: &VarBuilder, key: &str, channels: usize) -> Result<Self, String> {
        let scope = vb.pp(key);
        let qkv_weight = scope
            .get((channels * 3, channels, 1, 1), "to_qkv.weight")
            .map_err(|error| error.to_string())?;
        let qkv_bias = scope
            .get(channels * 3, "to_qkv.bias")
            .map_err(|error| error.to_string())?;
        let proj_weight = scope
            .get((channels, channels, 1, 1), "proj.weight")
            .map_err(|error| error.to_string())?;
        let proj_bias = scope
            .get(channels, "proj.bias")
            .map_err(|error| error.to_string())?;
        Ok(Self {
            norm: QwenRmsNorm2d::load(vb, &format!("{key}.norm"), channels)?,
            to_qkv: Conv2d::new(qkv_weight, Some(qkv_bias), Conv2dConfig::default()),
            proj: Conv2d::new(proj_weight, Some(proj_bias), Conv2dConfig::default()),
            channels,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let (batch, _, height, width) = input.dims4().map_err(|error| error.to_string())?;
        let normalized = self.norm.forward(input)?;
        let qkv = self
            .to_qkv
            .forward(&normalized)
            .map_err(|error| error.to_string())?;
        let tokens = height * width;
        let qkv = qkv
            .reshape((batch, self.channels * 3, tokens))
            .map_err(|error| error.to_string())?;
        let query = qkv
            .narrow(1, 0, self.channels)
            .map_err(|error| error.to_string())?;
        let key = qkv
            .narrow(1, self.channels, self.channels)
            .map_err(|error| error.to_string())?;
        let value = qkv
            .narrow(1, self.channels * 2, self.channels)
            .map_err(|error| error.to_string())?;
        let scores = query
            .transpose(1, 2)
            .map_err(|error| error.to_string())?
            .matmul(&key)
            .map_err(|error| error.to_string())?
            .affine(1.0 / (self.channels as f64).sqrt(), 0.0)
            .map_err(|error| error.to_string())?;
        let attended = softmax_last_dim(&scores)
            .map_err(|error| error.to_string())?
            .matmul(&value.transpose(1, 2).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?
            .transpose(1, 2)
            .map_err(|error| error.to_string())?
            .reshape((batch, self.channels, height, width))
            .map_err(|error| error.to_string())?;
        let projected = self
            .proj
            .forward(&attended)
            .map_err(|error| error.to_string())?;
        projected.add(input).map_err(|error| error.to_string())
    }
}

impl CandleQwenImageDownsample2d {
    pub fn load(
        vb: &VarBuilder,
        key: &str,
        input_channels: usize,
        output_channels: usize,
        temporal: bool,
    ) -> Result<Self, String> {
        let spatial = SpatialConv2d::load(
            vb,
            &format!("{key}.resample.1"),
            input_channels,
            output_channels,
        )?;
        let temporal = if temporal {
            Some(CausalConv2d::load(
                vb,
                &format!("{key}.time_conv"),
                output_channels,
                output_channels,
                3,
                1,
                1,
            )?)
        } else {
            None
        };
        Ok(Self { spatial, temporal })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let output = self.spatial.forward(input)?;
        match &self.temporal {
            Some(layer) => layer.forward(&output),
            None => Ok(output),
        }
    }
}

/// The first executable unit of the Candle Qwen Image VAE encoder.
pub struct CandleQwenImageVaeStem {
    conv_in: CausalConv2d,
    first_residual: CandleQwenImageResidualBlock2d,
    second_residual: CandleQwenImageResidualBlock2d,
    first_downsample: CandleQwenImageDownsample2d,
    second_stage_residual1: CandleQwenImageResidualBlock2d,
    second_stage_residual2: CandleQwenImageResidualBlock2d,
    second_downsample: CandleQwenImageDownsample2d,
    third_stage_residual1: CandleQwenImageResidualBlock2d,
    third_stage_residual2: CandleQwenImageResidualBlock2d,
    third_downsample: CandleQwenImageDownsample2d,
    final_residual1: CandleQwenImageResidualBlock2d,
    final_residual2: CandleQwenImageResidualBlock2d,
    middle_residual1: CandleQwenImageResidualBlock2d,
    middle_attention: QwenImageAttention2d,
    middle_residual2: CandleQwenImageResidualBlock2d,
    output_norm: QwenRmsNorm2d,
    output_conv: CausalConv2d,
    quant_conv: CausalConv2d,
    device: Device,
}

impl CandleQwenImageVaeStem {
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, PreprocessError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(PreprocessError::Encoder(format!(
                "Qwen Image VAE checkpoint does not exist: {}",
                path.display()
            )));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
            .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
        let conv_in = CausalConv2d::load(&vb, "encoder.conv1", 3, 96, 3, 3, 1)
            .map_err(PreprocessError::Encoder)?;
        let first_residual =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.0", 96, 96)
                .map_err(PreprocessError::Encoder)?;
        let second_residual =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.1", 96, 96)
                .map_err(PreprocessError::Encoder)?;
        let first_downsample =
            CandleQwenImageDownsample2d::load(&vb, "encoder.downsamples.2", 96, 96, false)
                .map_err(PreprocessError::Encoder)?;
        let second_stage_residual1 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.3", 96, 192)
                .map_err(PreprocessError::Encoder)?;
        let second_stage_residual2 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.4", 192, 192)
                .map_err(PreprocessError::Encoder)?;
        let second_downsample =
            CandleQwenImageDownsample2d::load(&vb, "encoder.downsamples.5", 192, 192, true)
                .map_err(PreprocessError::Encoder)?;
        let third_stage_residual1 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.6", 192, 384)
                .map_err(PreprocessError::Encoder)?;
        let third_stage_residual2 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.7", 384, 384)
                .map_err(PreprocessError::Encoder)?;
        let third_downsample =
            CandleQwenImageDownsample2d::load(&vb, "encoder.downsamples.8", 384, 384, true)
                .map_err(PreprocessError::Encoder)?;
        let final_residual1 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.9", 384, 384)
                .map_err(PreprocessError::Encoder)?;
        let final_residual2 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.downsamples.10", 384, 384)
                .map_err(PreprocessError::Encoder)?;
        let middle_residual1 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.middle.0", 384, 384)
                .map_err(PreprocessError::Encoder)?;
        let middle_attention = QwenImageAttention2d::load(&vb, "encoder.middle.1", 384)
            .map_err(PreprocessError::Encoder)?;
        let middle_residual2 =
            CandleQwenImageResidualBlock2d::load(&vb, "encoder.middle.2", 384, 384)
                .map_err(PreprocessError::Encoder)?;
        let output_norm =
            QwenRmsNorm2d::load(&vb, "encoder.head.0", 384).map_err(PreprocessError::Encoder)?;
        let output_conv = CausalConv2d::load(&vb, "encoder.head.2", 384, 32, 3, 3, 1)
            .map_err(PreprocessError::Encoder)?;
        let quant_conv =
            CausalConv2d::load(&vb, "conv1", 32, 32, 1, 1, 1).map_err(PreprocessError::Encoder)?;
        Ok(Self {
            conv_in,
            first_residual,
            second_residual,
            first_downsample,
            second_stage_residual1,
            second_stage_residual2,
            second_downsample,
            third_stage_residual1,
            third_stage_residual2,
            third_downsample,
            final_residual1,
            final_residual2,
            middle_residual1,
            middle_attention,
            middle_residual2,
            output_norm,
            output_conv,
            quant_conv,
            device: device.clone(),
        })
    }

    pub fn forward(&self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        if input.shape.len() != 5
            || input.shape[0] != 1
            || input.shape[1] != 3
            || input.shape[2] != 1
        {
            return Err(VaeError::ShapeMismatch {
                expected: vec![1, 3, 1, 0, 0],
                actual: input.shape.clone(),
            });
        }
        let input = input
            .to_candle(&self.device)?
            .squeeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        let output = self.conv_in.forward(&input).map_err(VaeError::Candle)?;
        let output = output
            .unsqueeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        VaeTensor::from_candle(&output)
    }

    pub fn forward_first_residual(&self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        if input.shape.len() != 5
            || input.shape[0] != 1
            || input.shape[1] != 3
            || input.shape[2] != 1
        {
            return Err(VaeError::ShapeMismatch {
                expected: vec![1, 3, 1, 0, 0],
                actual: input.shape.clone(),
            });
        }
        let input = input
            .to_candle(&self.device)?
            .squeeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        let output = self.conv_in.forward(&input).map_err(VaeError::Candle)?;
        let output = self
            .first_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = output
            .unsqueeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        VaeTensor::from_candle(&output)
    }

    pub fn forward_first_downsample(&self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        if input.shape.len() != 5
            || input.shape[0] != 1
            || input.shape[1] != 3
            || input.shape[2] != 1
        {
            return Err(VaeError::ShapeMismatch {
                expected: vec![1, 3, 1, 0, 0],
                actual: input.shape.clone(),
            });
        }
        let input = input
            .to_candle(&self.device)?
            .squeeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        let output = self.conv_in.forward(&input).map_err(VaeError::Candle)?;
        let output = self
            .first_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .second_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .first_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?
            .unsqueeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        VaeTensor::from_candle(&output)
    }

    pub fn forward_downsample_stack(&self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        if input.shape.len() != 5
            || input.shape[0] != 1
            || input.shape[1] != 3
            || input.shape[2] != 1
        {
            return Err(VaeError::ShapeMismatch {
                expected: vec![1, 3, 1, 0, 0],
                actual: input.shape.clone(),
            });
        }
        let input = input
            .to_candle(&self.device)?
            .squeeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        let output = self.conv_in.forward(&input).map_err(VaeError::Candle)?;
        let output = self
            .first_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .second_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .first_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .second_stage_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .second_stage_residual2
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .second_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .third_stage_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .third_stage_residual2
            .forward(&output)
            .map_err(VaeError::Candle)?;
        let output = self
            .third_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?
            .unsqueeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        VaeTensor::from_candle(&output)
    }

    /// Runs the complete deterministic Qwen Image VAE encoder and returns the
    /// first 16 channels of `quant_conv`, i.e. the Gaussian mean used as the
    /// still-image latent. The singleton temporal axis is preserved.
    pub fn forward_latent(&self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        if input.shape.len() != 5
            || input.shape[0] != 1
            || input.shape[1] != 3
            || input.shape[2] != 1
        {
            return Err(VaeError::ShapeMismatch {
                expected: vec![1, 3, 1, 0, 0],
                actual: input.shape.clone(),
            });
        }
        let input = input
            .to_candle(&self.device)?
            .squeeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        let mut output = self.conv_in.forward(&input).map_err(VaeError::Candle)?;
        output = self
            .first_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .second_residual
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .first_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .second_stage_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .second_stage_residual2
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .second_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .third_stage_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .third_stage_residual2
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .third_downsample
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .final_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .final_residual2
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .middle_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .middle_attention
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .middle_residual2
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self
            .output_norm
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = output
            .silu()
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        output = self
            .output_conv
            .forward(&output)
            .map_err(VaeError::Candle)?;
        output = self.quant_conv.forward(&output).map_err(VaeError::Candle)?;
        output = output
            .narrow(1, 0, 16)
            .map_err(|error| VaeError::Candle(error.to_string()))?
            .unsqueeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        VaeTensor::from_candle(&output)
    }
}

impl QwenImageVaeEncoder for CandleQwenImageVaeStem {
    fn encode_qwen_input(&mut self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        let latent = self.forward_latent(input)?;
        normalize_qwen_image_latent(&latent, &QwenImageVaeConfig::default())
    }
}
