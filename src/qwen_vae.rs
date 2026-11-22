//! Incremental Candle implementation of the Qwen Image VAE encoder.
//!
//! Still-image training uses a singleton temporal axis. For a causal 3D
//! convolution with T=1, the final temporal kernel slice is the only slice
//! that contributes, so the first migration step can execute it as a 2D
//! convolution while preserving the checkpoint layout.

use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, ops::softmax_last_dim};

use crate::{
    preprocess::PreprocessError,
    vae::{QwenImageVaeEncoder, VaeError, VaeTensor},
};

static VAE_STAGE_TRACE_USED: AtomicBool = AtomicBool::new(false);

fn trace_vae_stage(name: &str, tensor: &Tensor) {
    if std::env::var_os("AFTERBURNER_VAE_STAGE_TRACE").is_none() {
        return;
    }
    let tensor = match tensor.to_dtype(DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => {
            eprintln!("[vae-stage] {name} dtype conversion failed: {error}");
            return;
        }
    };
    let values = match tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
    {
        Ok(values) => values,
        Err(error) => {
            eprintln!("[vae-stage] {name} readback failed: {error}");
            return;
        }
    };
    if let Some(directory) = std::env::var_os("AFTERBURNER_VAE_STAGE_DUMP_DIR") {
        let directory = std::path::PathBuf::from(directory);
        if let Err(error) = std::fs::create_dir_all(&directory) {
            eprintln!("[vae-stage] dump directory creation failed: {error}");
        } else {
            let data_path = directory.join(format!("{name}.f32le"));
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
            for value in &values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            if let Err(error) = std::fs::write(&data_path, bytes) {
                eprintln!("[vae-stage] dump write failed for {name}: {error}");
            }
            let shape = tensor
                .dims()
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            if let Err(error) = std::fs::write(directory.join(format!("{name}.shape")), shape) {
                eprintln!("[vae-stage] shape dump failed for {name}: {error}");
            }
        }
    }
    if values.is_empty() {
        eprintln!("[vae-stage] {name} shape={:?} empty", tensor.dims());
        return;
    }
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let l2 = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    eprintln!(
        "[vae-stage] {name} shape={:?} min={minimum:.8e} max={maximum:.8e} mean={mean:.8e} l2={l2:.8e}",
        tensor.dims()
    );
    if name == "input" && tensor.dims().len() == 4 && tensor.dims()[1] <= 4 {
        let channels = tensor.dims()[1];
        let plane = tensor.dims()[2] * tensor.dims()[3];
        let means = (0..channels)
            .map(|channel| {
                let start = channel * plane;
                values[start..start + plane].iter().sum::<f32>() / plane as f32
            })
            .collect::<Vec<_>>();
        eprintln!("[vae-stage] input_channel_means={means:?}");
    }
}

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
        // Match kohya_ss's QwenImageRMS_norm exactly: PyTorch receives the
        // VAE activation in fp16 and keeps this normalize/scale/multiply
        // chain in fp16. Promoting the reduction to f32 changes the rounded
        // activation before every following residual block.
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
        self.forward_traced(input, None)
    }

    fn forward_traced(&self, input: &Tensor, trace_prefix: Option<&str>) -> Result<Tensor, String> {
        let shortcut = match &self.shortcut {
            Some(layer) => layer.forward(input)?,
            None => input.clone(),
        };
        let hidden = self.norm1.forward(input)?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(&format!("{prefix}_norm1"), &hidden);
        }
        let hidden = hidden.silu().map_err(|error| error.to_string())?;
        let hidden = self.conv1.forward(&hidden)?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(&format!("{prefix}_conv1"), &hidden);
        }
        let hidden = self.norm2.forward(&hidden)?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(&format!("{prefix}_norm2"), &hidden);
        }
        let hidden = hidden.silu().map_err(|error| error.to_string())?;
        let hidden = self.conv2.forward(&hidden)?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(&format!("{prefix}_conv2"), &hidden);
        }
        let output = hidden.add(&shortcut).map_err(|error| error.to_string())?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(prefix, &output);
        }
        Ok(output)
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
            .map_err(|error| error.to_string())?
            // The temporal slice is a non-contiguous view. CUDA Candle's
            // Im2Col path has a separate strided-kernel branch; materialize
            // the ordinary contiguous [out, in, height, width] layout so its
            // kernel interpretation matches PyTorch/cuDNN.
            .contiguous()
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
        self.forward_traced(input, None)
    }

    fn forward_traced(&self, input: &Tensor, trace_prefix: Option<&str>) -> Result<Tensor, String> {
        let input = input
            .pad_with_zeros(2, 0, 1)
            .and_then(|value| value.pad_with_zeros(3, 0, 1))
            .map_err(|error| error.to_string())?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(&format!("{prefix}_padded"), &input);
        }
        let output = self.conv.forward(&input).map_err(|error| error.to_string())?;
        if let Some(prefix) = trace_prefix {
            trace_vae_stage(&format!("{prefix}_conv"), &output);
        }
        Ok(output)
    }
}

/// Spatial downsample used by the still-image encoder path.
#[derive(Clone, Debug)]
pub struct CandleQwenImageDownsample2d {
    spatial: SpatialConv2d,
    _temporal: Option<CausalConv2d>,
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
        let attended = attended
            .to_dtype(normalized.dtype())
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
        Ok(Self {
            spatial,
            _temporal: temporal,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        // Kohya's ordinary still-image encode path calls the VAE without a
        // feature cache. In that path `QwenImageResample` performs only the
        // spatial resample; its temporal convolution is used exclusively by
        // the cached/chunked video path. Applying it here changes every
        // latent even though the input has a singleton time axis.
        self.spatial.forward(input)
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
    dtype: DType,
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
        if std::env::var_os("AFTERBURNER_VAE_STAGE_DUMP_DIR").is_some() {
            trace_vae_stage("conv_in_weight", conv_in.conv.weight());
            if let Some(bias) = conv_in.conv.bias() {
                trace_vae_stage("conv_in_bias", bias);
            }
        }
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
            dtype,
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
            .to_dtype(self.dtype)
            .map_err(|error| VaeError::Candle(error.to_string()))?
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
            .to_dtype(self.dtype)
            .map_err(|error| VaeError::Candle(error.to_string()))?
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
        // Kohya's IMAGE_TRANSFORMS produces an fp16 tensor before the VAE
        // convolution when preprocessing runs in fp16. Match that boundary
        // here instead of feeding the VAE an f32 image tensor.
        let input = input
            .to_candle(&self.device)?
            .to_dtype(self.dtype)
            .map_err(|error| VaeError::Candle(error.to_string()))?
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
        // Kohya's IMAGE_TRANSFORMS produces an fp16 tensor before the VAE
        // convolution when preprocessing runs in fp16. Match that boundary
        // here instead of feeding the VAE an f32 image tensor.
        let input = input
            .to_candle(&self.device)?
            .to_dtype(self.dtype)
            .map_err(|error| VaeError::Candle(error.to_string()))?
            .squeeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        let trace = std::env::var_os("AFTERBURNER_VAE_STAGE_TRACE").is_some()
            && !VAE_STAGE_TRACE_USED.swap(true, Ordering::Relaxed);
        if trace {
            trace_vae_stage("input", &input);
        }
        let mut output = self.conv_in.forward(&input).map_err(VaeError::Candle)?;
        if trace {
            trace_vae_stage("conv_in", &output);
        }
        output = self
            .first_residual
            .forward_traced(&output, trace.then_some("residual_0"))
            .map_err(VaeError::Candle)?;
        output = self
            .second_residual
            .forward_traced(&output, trace.then_some("residual_1"))
            .map_err(VaeError::Candle)?;
        output = self
            .first_downsample
            .spatial
            .forward_traced(&output, trace.then_some("downsample_0"))
            .map_err(VaeError::Candle)?;

        output = self
            .second_stage_residual1
            .forward_traced(&output, trace.then_some("residual_3"))
            .map_err(VaeError::Candle)?;
        output = self
            .second_stage_residual2
            .forward_traced(&output, trace.then_some("residual_4"))
            .map_err(VaeError::Candle)?;
        output = self
            .second_downsample
            .spatial
            .forward_traced(&output, trace.then_some("downsample_1"))
            .map_err(VaeError::Candle)?;
        output = self
            .third_stage_residual1
            .forward_traced(&output, trace.then_some("residual_6"))
            .map_err(VaeError::Candle)?;
        output = self
            .third_stage_residual2
            .forward_traced(&output, trace.then_some("residual_7"))
            .map_err(VaeError::Candle)?;
        output = self
            .third_downsample
            .spatial
            .forward_traced(&output, trace.then_some("downsample_2"))
            .map_err(VaeError::Candle)?;
        output = self
            .final_residual1
            .forward_traced(&output, trace.then_some("residual_9"))
            .map_err(VaeError::Candle)?;
        output = self
            .final_residual2
            .forward_traced(&output, trace.then_some("residual_10"))
            .map_err(VaeError::Candle)?;
        output = self
            .middle_residual1
            .forward(&output)
            .map_err(VaeError::Candle)?;
        if trace {
            trace_vae_stage("middle_residual_0", &output);
        }
        output = self
            .middle_attention
            .forward(&output)
            .map_err(VaeError::Candle)?;
        if trace {
            trace_vae_stage("middle_attention", &output);
        }
        output = self
            .middle_residual2
            .forward_traced(&output, Some("middle_residual_1"))
            .map_err(VaeError::Candle)?;
        if trace {
            trace_vae_stage("middle_residual_1", &output);
            trace_vae_stage("middle", &output);
        }
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
        if trace {
            trace_vae_stage("output_conv", &output);
        }
        output = self.quant_conv.forward(&output).map_err(VaeError::Candle)?;
        output = output
            .narrow(1, 0, 16)
            .map_err(|error| VaeError::Candle(error.to_string()))?
            .unsqueeze(2)
            .map_err(|error| VaeError::Candle(error.to_string()))?;
        if trace {
            trace_vae_stage("quant_conv", &output);
        }
        VaeTensor::from_candle(&output)
    }
}

impl QwenImageVaeEncoder for CandleQwenImageVaeStem {
    fn encode_qwen_input(&mut self, input: &VaeTensor) -> Result<VaeTensor, VaeError> {
        // Return the raw posterior mean. The generic Qwen preprocessor owns
        // the single latent mean/std normalization step, matching Kohya's
        // `encode_pixels_to_latents` contract.
        self.forward_latent(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_rms_norm_normalizes_channels_per_pixel() {
        let device = Device::Cpu;
        let gamma = Tensor::ones((1, 2, 1, 1), DType::F32, &device).unwrap();
        let norm = QwenRmsNorm2d {
            gamma,
            scale: (2.0_f64).sqrt(),
        };
        let input = Tensor::new(&[1.0_f32, 0.0, 0.0, 2.0], &device)
            .unwrap()
            .reshape((1, 2, 1, 2))
            .unwrap();
        let output = norm.forward(&input).unwrap();
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!(values[1].abs() < 1e-6);
        assert!(values[2].abs() < 1e-6);
        assert!((values[3] - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    #[ignore = "loads the full Qwen Image VAE; set AFTERBURNER_QWEN_VAE_PATH to run"]
    fn qwen_vae_safetensors_produces_anima_latent_shape() {
        let path = std::env::var_os("AFTERBURNER_QWEN_VAE_PATH")
            .expect("AFTERBURNER_QWEN_VAE_PATH must point to qwen_image_vae.safetensors");
        let device = Device::Cpu;
        let mut encoder = CandleQwenImageVaeStem::from_safetensors(path, DType::F32, &device)
            .expect("failed to load Qwen Image VAE");
        let input = VaeTensor::new(vec![1, 3, 1, 64, 64], vec![0.0; 3 * 64 * 64]).unwrap();
        let latent = encoder
            .encode_qwen_input(&input)
            .expect("Qwen VAE encoding failed");
        assert_eq!(latent.shape[0..3], [1, 16, 1]);
        assert_eq!(latent.shape[3..], [8, 8]);
        assert!(latent.data.iter().all(|value| value.is_finite()));
    }
}
