//! SafeTensors weight loading for model-specific adapters.

use std::{borrow::Cow, collections::HashMap, fs::File, path::Path};

use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::{ElementConversion, Tensor, TensorData, backend::Backend},
};
use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::{SafeTensors, tensor::Dtype};

use crate::{
    anima::AnimaConfig,
    dit::AnimaDiT,
    lora::{LoraConfig, LoraLinear, LoraLinearConfig, LoraTarget, select_anima_target},
};

#[derive(Debug)]
pub enum WeightError {
    Io(std::io::Error),
    SafeTensor(String),
    MissingTensor(String),
    UnsupportedDtype(String),
    InvalidShape {
        key: String,
        shape: Vec<usize>,
    },
    ShapeMismatch {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NotAnimaLoraTarget(String),
    Lora(String),
    Export(String),
    InvalidRank {
        key: String,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenImageVaeCoverage {
    pub tensor_count: usize,
    pub bfloat16_tensors: usize,
    pub required_keys_present: usize,
    pub required_keys: usize,
    pub missing_required_keys: Vec<String>,
}

/// Inspects the ComfyUI-format Qwen Image VAE checkpoint without loading its
/// 250 MB payload into memory.
pub fn inspect_qwen_image_vae(path: impl AsRef<Path>) -> Result<QwenImageVaeCoverage, WeightError> {
    let file = File::open(path).map_err(WeightError::Io)?;
    let mapped = unsafe { MmapOptions::new().map(&file).map_err(WeightError::Io)? };
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let required = [
        "conv1.weight",
        "conv2.weight",
        "decoder.conv1.weight",
        "decoder.head.2.weight",
        "encoder.conv1.weight",
        "encoder.head.2.weight",
        "conv1.weight",
        "conv2.weight",
    ];
    let missing_required_keys = required
        .iter()
        .filter(|key| tensors.tensor(key).is_err())
        .map(|key| (*key).to_owned())
        .collect::<Vec<_>>();
    let bfloat16_tensors = tensors
        .names()
        .iter()
        .filter_map(|name| tensors.tensor(name).ok())
        .filter(|tensor| tensor.dtype() == Dtype::BF16)
        .count();
    Ok(QwenImageVaeCoverage {
        tensor_count: tensors.names().len(),
        bfloat16_tensors,
        required_keys_present: required.len() - missing_required_keys.len(),
        required_keys: required.len(),
        missing_required_keys,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimaWeightSlot {
    InputProjection,
    OutputProjection,
    FinalModulation { component: String },
    TimestepLinear1,
    TimestepLinear2,
    TimestepNorm,
    LlmAdapterGlobal { component: String },
    Block { index: usize, component: String },
    LlmAdapter { index: usize, component: String },
}

pub fn map_anima_weight_key(key: &str) -> Option<AnimaWeightSlot> {
    let key = key.strip_prefix("model.diffusion_model.")?;
    if key == "x_embedder.proj.1.weight" {
        return Some(AnimaWeightSlot::InputProjection);
    }
    if key == "final_layer.linear.weight" {
        return Some(AnimaWeightSlot::OutputProjection);
    }
    if let Some(component) = key.strip_prefix("final_layer.adaln_modulation.") {
        if component.ends_with(".weight") {
            return Some(AnimaWeightSlot::FinalModulation {
                component: linear_component_name(component),
            });
        }
    }
    if key == "t_embedder.1.linear_1.weight" {
        return Some(AnimaWeightSlot::TimestepLinear1);
    }
    if key == "t_embedder.1.linear_2.weight" {
        return Some(AnimaWeightSlot::TimestepLinear2);
    }
    if key == "t_embedding_norm.weight" {
        return Some(AnimaWeightSlot::TimestepNorm);
    }
    if let Some(component) = key.strip_prefix("llm_adapter.") {
        if !component.starts_with("blocks.") {
            return Some(AnimaWeightSlot::LlmAdapterGlobal {
                component: linear_component_name(component),
            });
        }
    }
    if let Some(rest) = key.strip_prefix("blocks.") {
        let (index, component) = rest.split_once('.')?;
        let index = index.parse().ok()?;
        if component.ends_with(".weight") || component.ends_with(".bias") {
            return Some(AnimaWeightSlot::Block {
                index,
                component: linear_component_name(component),
            });
        }
    }
    if let Some(rest) = key.strip_prefix("llm_adapter.blocks.") {
        let (index, component) = rest.split_once('.')?;
        let index = index.parse().ok()?;
        if component.ends_with(".weight") || component.ends_with(".bias") {
            return Some(AnimaWeightSlot::LlmAdapter {
                index,
                component: linear_component_name(component),
            });
        }
    }
    None
}

fn linear_component_name(component: &str) -> String {
    component
        .strip_suffix(".weight")
        .unwrap_or(component)
        .to_owned()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimaWeightCoverage {
    pub mapped: Vec<(String, AnimaWeightSlot)>,
    pub unmapped: Vec<String>,
}

impl AnimaWeightCoverage {
    pub fn mapped_count(&self) -> usize {
        self.mapped.len()
    }

    pub fn unmapped_count(&self) -> usize {
        self.unmapped.len()
    }
}

pub fn inspect_anima_weight_coverage(
    path: impl AsRef<Path>,
) -> Result<AnimaWeightCoverage, WeightError> {
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let mut coverage = AnimaWeightCoverage {
        mapped: Vec::new(),
        unmapped: Vec::new(),
    };
    for (key, _) in tensors.iter() {
        match map_anima_weight_key(key) {
            Some(slot) => coverage.mapped.push((key.to_owned(), slot)),
            None => coverage.unmapped.push(key.to_owned()),
        }
    }
    coverage.mapped.sort_by(|left, right| left.0.cmp(&right.0));
    coverage.unmapped.sort();
    Ok(coverage)
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AnimaLoadReport {
    pub loaded: Vec<String>,
    pub skipped: Vec<String>,
}

impl AnimaLoadReport {
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn skipped_count(&self) -> usize {
        self.skipped.len()
    }
}

/// Loads all currently supported Anima parameters into an initialized Burn model.
pub fn load_anima_weights<B: Backend>(
    path: impl AsRef<Path>,
    model: &mut AnimaDiT<B>,
) -> Result<AnimaLoadReport, WeightError> {
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let device = model.input_projection.weight.val().device();
    let show_progress = std::env::var_os("AFTERBURNER_LOAD_PROGRESS").is_some();
    let mut report = AnimaLoadReport::default();

    for (index, (key, view)) in tensors.iter().enumerate() {
        if show_progress && index % 25 == 0 {
            eprintln!("loading base tensor {index}: {key}");
        }
        let Some(slot) = map_anima_weight_key(key) else {
            report.skipped.push(key.to_owned());
            continue;
        };
        let loaded = match slot {
            AnimaWeightSlot::InputProjection => {
                load_linear_weight(&mut model.input_projection, view, key, &device)?
            }
            AnimaWeightSlot::OutputProjection => {
                load_linear_weight(&mut model.output_projection, view, key, &device)?
            }
            AnimaWeightSlot::FinalModulation { component } => {
                let target = if component == "1" {
                    &mut model.final_modulation.down
                } else if component == "2" {
                    &mut model.final_modulation.up
                } else {
                    report.skipped.push(key.to_owned());
                    continue;
                };
                load_linear_weight(target, view, key, &device)?
            }
            AnimaWeightSlot::Block { index, component } => {
                let Some(block) = model.blocks.get_mut(index) else {
                    report.skipped.push(key.to_owned());
                    continue;
                };
                let target = match component.as_str() {
                    "self_attn.q_proj" => Some(&mut block.self_attn.q_proj),
                    "self_attn.k_proj" => Some(&mut block.self_attn.k_proj),
                    "self_attn.v_proj" => Some(&mut block.self_attn.v_proj),
                    "self_attn.output_proj" => Some(&mut block.self_attn.output_proj),
                    "cross_attn.q_proj" => Some(&mut block.cross_attn.q_proj),
                    "cross_attn.k_proj" => Some(&mut block.cross_attn.k_proj),
                    "cross_attn.v_proj" => Some(&mut block.cross_attn.v_proj),
                    "cross_attn.output_proj" => Some(&mut block.cross_attn.output_proj),
                    "mlp.layer1" => Some(&mut block.mlp_layer1),
                    "mlp.layer2" => Some(&mut block.mlp_layer2),
                    "adaln_modulation_self_attn.1" => Some(&mut block.self_attn_modulation.down),
                    "adaln_modulation_self_attn.2" => Some(&mut block.self_attn_modulation.up),
                    "adaln_modulation_cross_attn.1" => Some(&mut block.cross_attn_modulation.down),
                    "adaln_modulation_cross_attn.2" => Some(&mut block.cross_attn_modulation.up),
                    "adaln_modulation_mlp.1" => Some(&mut block.mlp_modulation.down),
                    "adaln_modulation_mlp.2" => Some(&mut block.mlp_modulation.up),
                    _ => None,
                };
                let Some(target) = target else {
                    report.skipped.push(key.to_owned());
                    continue;
                };
                load_linear_weight(target, view, key, &device)?
            }
            AnimaWeightSlot::LlmAdapter { index, component } => {
                let Some(block) = model.llm_adapter.blocks.get_mut(index) else {
                    report.skipped.push(key.to_owned());
                    continue;
                };
                match component.as_str() {
                    "self_attn.q_proj" => {
                        load_linear_weight(&mut block.self_attn.q_proj, view, key, &device)?
                    }
                    "self_attn.k_proj" => {
                        load_linear_weight(&mut block.self_attn.k_proj, view, key, &device)?
                    }
                    "self_attn.v_proj" => {
                        load_linear_weight(&mut block.self_attn.v_proj, view, key, &device)?
                    }
                    "self_attn.o_proj" => {
                        load_linear_weight(&mut block.self_attn.output_proj, view, key, &device)?
                    }
                    "cross_attn.q_proj" => {
                        load_linear_weight(&mut block.cross_attn.q_proj, view, key, &device)?
                    }
                    "cross_attn.k_proj" => {
                        load_linear_weight(&mut block.cross_attn.k_proj, view, key, &device)?
                    }
                    "cross_attn.v_proj" => {
                        load_linear_weight(&mut block.cross_attn.v_proj, view, key, &device)?
                    }
                    "cross_attn.o_proj" => {
                        load_linear_weight(&mut block.cross_attn.output_proj, view, key, &device)?
                    }
                    "mlp.0" => load_linear_weight(&mut block.mlp_layer1, view, key, &device)?,
                    "mlp.0.bias" => load_linear_bias(&mut block.mlp_layer1, view, key, &device)?,
                    "mlp.2" => load_linear_weight(&mut block.mlp_layer2, view, key, &device)?,
                    "mlp.2.bias" => load_linear_bias(&mut block.mlp_layer2, view, key, &device)?,
                    "norm_self_attn" => {
                        load_rms_norm_weight(&mut block.norm_self_attn, view, key, &device)?
                    }
                    "norm_cross_attn" => {
                        load_rms_norm_weight(&mut block.norm_cross_attn, view, key, &device)?
                    }
                    "norm_mlp" => load_rms_norm_weight(&mut block.norm_mlp, view, key, &device)?,
                    _ => {
                        report.skipped.push(key.to_owned());
                        continue;
                    }
                }
            }
            AnimaWeightSlot::LlmAdapterGlobal { component } => match component.as_str() {
                "embed" => load_embedding_weight(
                    &mut model.llm_adapter.token_embedding,
                    view,
                    key,
                    &device,
                )?,
                "norm" => load_rms_norm_weight(&mut model.llm_adapter.norm, view, key, &device)?,
                "out_proj" => {
                    load_linear_weight(&mut model.llm_adapter.output_proj, view, key, &device)?
                }
                "out_proj.bias" => {
                    load_linear_bias(&mut model.llm_adapter.output_proj, view, key, &device)?
                }
                _ => {
                    report.skipped.push(key.to_owned());
                    continue;
                }
            },
            AnimaWeightSlot::TimestepLinear1 => {
                load_linear_weight(&mut model.timestep_embedder.linear1, view, key, &device)?
            }
            AnimaWeightSlot::TimestepLinear2 => {
                load_linear_weight(&mut model.timestep_embedder.linear2, view, key, &device)?
            }
            AnimaWeightSlot::TimestepNorm => {
                load_rms_norm_weight(&mut model.timestep_embedder.norm, view, key, &device)?
            }
        };
        if loaded {
            report.loaded.push(key.to_owned());
        }
    }
    Ok(report)
}

/// Initializes the default Anima architecture and loads its base checkpoint.
pub fn load_anima_model<B: Backend>(
    path: impl AsRef<Path>,
    lora: &LoraConfig,
    device: &B::Device,
) -> Result<(AnimaDiT<B>, AnimaLoadReport), WeightError> {
    let mut model = AnimaDiT::init(&AnimaConfig::default(), lora, device)
        .map_err(|error| WeightError::Lora(error.to_string()))?;
    let report = load_anima_weights(path, &mut model)?;
    Ok((model, report))
}

/// Loads an Afterburner-exported LoRA file into an initialized Anima model.
pub fn load_anima_lora_factors<B: Backend>(
    path: impl AsRef<Path>,
    model: &mut AnimaDiT<B>,
) -> Result<usize, WeightError> {
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let device = model.input_projection.weight.val().device();
    let mut loaded = 0;
    for (key, view) in tensors.iter() {
        let Some((target_name, factor)) = key
            .strip_prefix("model.diffusion_model.")
            .and_then(|key| key.strip_suffix(".weight"))
            .and_then(|key| key.rsplit_once(".lora_"))
        else {
            continue;
        };
        let Some(target) = lora_target_mut(model, target_name) else {
            continue;
        };
        if factor != "down" && factor != "up" {
            continue;
        }
        let expected = if factor == "down" {
            target.lora_down.val().dims()
        } else {
            target.lora_up.val().dims()
        };
        ensure_shape(key, view.shape(), &expected)?;
        let tensor = load_tensor::<B, 2>(view, key, &device)?;
        if factor == "down" {
            target.lora_down = Param::from_tensor(tensor);
        } else {
            target.lora_up = Param::from_tensor(tensor);
        }
        loaded += 1;
    }
    Ok(loaded)
}

fn lora_target_mut<'a, B: Backend>(
    model: &'a mut AnimaDiT<B>,
    name: &str,
) -> Option<&'a mut LoraLinear<B>> {
    if name == "input_projection" {
        return Some(&mut model.input_projection);
    }
    if name == "output_projection" {
        return Some(&mut model.output_projection);
    }
    if name == "timestep_embedder.linear1" {
        return Some(&mut model.timestep_embedder.linear1);
    }
    if name == "timestep_embedder.linear2" {
        return Some(&mut model.timestep_embedder.linear2);
    }
    if name == "final_modulation.down" {
        return Some(&mut model.final_modulation.down);
    }
    if name == "final_modulation.up" {
        return Some(&mut model.final_modulation.up);
    }
    if name == "llm_adapter.output_proj" {
        return Some(&mut model.llm_adapter.output_proj);
    }
    let (kind, rest) = name.split_once('.')?;
    if kind == "blocks" {
        let (index, component) = rest.split_once('.')?;
        return block_lora_target_mut(
            model.blocks.get_mut(index.parse::<usize>().ok()?)?,
            component,
        );
    }
    if kind == "llm_adapter" {
        let rest = rest.strip_prefix("blocks.")?;
        let (index, component) = rest.split_once('.')?;
        return adapter_lora_target_mut(
            model
                .llm_adapter
                .blocks
                .get_mut(index.parse::<usize>().ok()?)?,
            component,
        );
    }
    None
}

fn block_lora_target_mut<'a, B: Backend>(
    block: &'a mut crate::dit::AnimaDiTBlock<B>,
    component: &str,
) -> Option<&'a mut LoraLinear<B>> {
    match component {
        "self_attn.q_proj" => Some(&mut block.self_attn.q_proj),
        "self_attn.k_proj" => Some(&mut block.self_attn.k_proj),
        "self_attn.v_proj" => Some(&mut block.self_attn.v_proj),
        "self_attn.output_proj" => Some(&mut block.self_attn.output_proj),
        "cross_attn.q_proj" => Some(&mut block.cross_attn.q_proj),
        "cross_attn.k_proj" => Some(&mut block.cross_attn.k_proj),
        "cross_attn.v_proj" => Some(&mut block.cross_attn.v_proj),
        "cross_attn.output_proj" => Some(&mut block.cross_attn.output_proj),
        "mlp_layer1" => Some(&mut block.mlp_layer1),
        "mlp_layer2" => Some(&mut block.mlp_layer2),
        "self_attn_modulation.down" => Some(&mut block.self_attn_modulation.down),
        "self_attn_modulation.up" => Some(&mut block.self_attn_modulation.up),
        "cross_attn_modulation.down" => Some(&mut block.cross_attn_modulation.down),
        "cross_attn_modulation.up" => Some(&mut block.cross_attn_modulation.up),
        "mlp_modulation.down" => Some(&mut block.mlp_modulation.down),
        "mlp_modulation.up" => Some(&mut block.mlp_modulation.up),
        _ => None,
    }
}

fn adapter_lora_target_mut<'a, B: Backend>(
    block: &'a mut crate::dit::AnimaLlmAdapterBlock<B>,
    component: &str,
) -> Option<&'a mut LoraLinear<B>> {
    match component {
        "self_attn.q_proj" => Some(&mut block.self_attn.q_proj),
        "self_attn.k_proj" => Some(&mut block.self_attn.k_proj),
        "self_attn.v_proj" => Some(&mut block.self_attn.v_proj),
        "self_attn.output_proj" => Some(&mut block.self_attn.output_proj),
        "cross_attn.q_proj" => Some(&mut block.cross_attn.q_proj),
        "cross_attn.k_proj" => Some(&mut block.cross_attn.k_proj),
        "cross_attn.v_proj" => Some(&mut block.cross_attn.v_proj),
        "cross_attn.output_proj" => Some(&mut block.cross_attn.output_proj),
        "mlp_layer1" => Some(&mut block.mlp_layer1),
        "mlp_layer2" => Some(&mut block.mlp_layer2),
        _ => None,
    }
}

fn load_linear_weight<B: Backend>(
    target: &mut LoraLinear<B>,
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<bool, WeightError> {
    ensure_shape(key, view.shape(), &target.weight.val().dims())?;
    target.weight = Param::from_tensor(load_tensor::<B, 2>(view, key, device)?);
    Ok(true)
}

fn load_linear_bias<B: Backend>(
    target: &mut LoraLinear<B>,
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<bool, WeightError> {
    if target.bias.is_none() {
        return Err(WeightError::MissingTensor(format!(
            "{key}: target layer has no bias"
        )));
    }
    ensure_shape(
        key,
        view.shape(),
        &target.bias.as_ref().unwrap().val().dims(),
    )?;
    target.bias = Some(Param::from_tensor(load_tensor::<B, 1>(view, key, device)?));
    Ok(true)
}

fn load_rms_norm_weight<B: Backend>(
    target: &mut crate::dit::AnimaRmsNorm<B>,
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<bool, WeightError> {
    ensure_shape(key, view.shape(), &target.gamma.val().dims())?;
    target.gamma = Param::from_tensor(load_tensor::<B, 1>(view, key, device)?);
    Ok(true)
}

fn load_embedding_weight<B: Backend>(
    target: &mut burn::nn::Embedding<B>,
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<bool, WeightError> {
    ensure_shape(key, view.shape(), &target.weight.val().dims())?;
    target.weight = Param::from_tensor(load_tensor::<B, 2>(view, key, device)?);
    Ok(true)
}

fn ensure_shape(key: &str, actual: &[usize], expected: &[usize]) -> Result<(), WeightError> {
    if actual == expected {
        Ok(())
    } else {
        Err(WeightError::ShapeMismatch {
            key: key.to_owned(),
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        })
    }
}

fn load_tensor<B: Backend, const D: usize>(
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>, WeightError> {
    let shape: [usize; D] =
        view.shape()
            .to_vec()
            .try_into()
            .map_err(|_| WeightError::InvalidRank {
                key: key.to_owned(),
                expected: D,
                actual: view.shape().len(),
            })?;
    let values = tensor_values(view.dtype(), view.data())?;
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "weight I/O error: {error}"),
            Self::SafeTensor(error) => write!(f, "SafeTensors error: {error}"),
            Self::MissingTensor(key) => write!(f, "tensor not found: {key}"),
            Self::UnsupportedDtype(dtype) => write!(f, "unsupported tensor dtype: {dtype}"),
            Self::InvalidShape { key, shape } => {
                write!(f, "tensor {key} must be rank 2, got {shape:?}")
            }
            Self::ShapeMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "tensor {key} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::NotAnimaLoraTarget(key) => write!(f, "not an Anima LoRA target: {key}"),
            Self::Lora(error) => write!(f, "LoRA initialization error: {error}"),
            Self::Export(error) => write!(f, "LoRA export error: {error}"),
            Self::InvalidRank {
                key,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "tensor {key} rank mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for WeightError {}

impl From<std::io::Error> for WeightError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
struct OwnedTensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::tensor::View for OwnedTensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

struct LoraCollector<B: Backend> {
    tensors: Vec<(String, OwnedTensorView)>,
    error: Option<String>,
    target_modules: Vec<String>,
    path: Vec<String>,
    marker: std::marker::PhantomData<B>,
}

impl<B: Backend> LoraCollector<B> {
    fn new(lora: &LoraConfig) -> Self {
        Self {
            tensors: Vec::new(),
            error: None,
            target_modules: lora.target_modules.clone(),
            path: Vec::new(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for LoraCollector<B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let path = &self.path;
        if D != 2 {
            return;
        }
        let Some(name) = path.last() else {
            return;
        };
        if name != "lora_down" && name != "lora_up" {
            return;
        }
        if path.len() < 3 || path[0] != "blocks" {
            return;
        }
        let module = match path[2..path.len() - 1].join(".").as_str() {
            "mlp_layer1" => "mlp.layer1".to_owned(),
            "mlp_layer2" => "mlp.layer2".to_owned(),
            value => value.to_owned(),
        };
        if !self.target_modules.iter().any(|target| target == &module) {
            return;
        }
        let key = format!(
            "model.diffusion_model.{}.weight",
            path.iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(".")
        );
        let tensor = param.val();
        let data = match tensor.to_data().to_vec::<B::FloatElem>() {
            Ok(values) => values
                .iter()
                .map(|value| value.elem::<f32>())
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            Err(error) => {
                self.error = Some(format!("{error:?}"));
                return;
            }
        };
        self.tensors.push((
            key,
            OwnedTensorView {
                dtype: Dtype::F32,
                shape: tensor.dims().to_vec(),
                data,
            },
        ));
    }
}

/// Exports only LoRA factors from an initialized Anima Burn model.
pub fn export_anima_lora<B: Backend>(
    path: impl AsRef<Path>,
    model: &AnimaDiT<B>,
) -> Result<usize, WeightError> {
    export_anima_lora_with_config(path, model, &LoraConfig::default())
}

/// Exports only configured DiT LoRA targets from an initialized model.
pub fn export_anima_lora_with_config<B: Backend>(
    path: impl AsRef<Path>,
    model: &AnimaDiT<B>,
    lora: &LoraConfig,
) -> Result<usize, WeightError> {
    lora.validate()
        .map_err(|error| WeightError::Export(error.to_string()))?;
    let mut collector = LoraCollector::<B>::new(lora);
    model.visit(&mut collector);
    if let Some(error) = collector.error {
        return Err(WeightError::Export(error));
    }
    if collector.tensors.is_empty() {
        return Err(WeightError::Export("model contains no LoRA factors".into()));
    }
    let count = collector.tensors.len();
    let mut metadata = HashMap::new();
    metadata.insert("format".into(), "afterburner-anima-lora-v1".into());
    metadata.insert("base_model".into(), "anima".into());
    safetensors::serialize_to_file(collector.tensors, Some(metadata), path.as_ref())
        .map_err(|error| WeightError::Export(error.to_string()))?;
    Ok(count)
}

/// The complete set of LoRA targets found in an Anima checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimaLoraPlan {
    pub targets: Vec<LoraTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AnimaLoraFactorPlan {
    pub keys: Vec<String>,
    pub parameter_count: usize,
}

impl AnimaLoraFactorPlan {
    pub fn tensor_count(&self) -> usize {
        self.keys.len()
    }
}

/// Inspects an Afterburner-exported LoRA SafeTensors file.
pub fn inspect_anima_lora_factors(
    path: impl AsRef<Path>,
) -> Result<AnimaLoraFactorPlan, WeightError> {
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let mut plan = AnimaLoraFactorPlan::default();
    for (key, view) in tensors.iter() {
        if !key.ends_with(".lora_down.weight") && !key.ends_with(".lora_up.weight") {
            continue;
        }
        if view.shape().len() != 2 {
            return Err(WeightError::InvalidRank {
                key: key.to_owned(),
                expected: 2,
                actual: view.shape().len(),
            });
        }
        plan.parameter_count += view.shape().iter().product::<usize>();
        plan.keys.push(key.to_owned());
    }
    plan.keys.sort();
    Ok(plan)
}

impl AnimaLoraPlan {
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub fn parameter_count(&self) -> usize {
        self.targets.iter().map(LoraTarget::parameter_count).sum()
    }
}

/// Scans an Anima SafeTensors file and builds a deterministic LoRA plan.
///
/// Only metadata is read during this operation; weight values are not copied.
pub fn inspect_anima_lora_file(
    path: impl AsRef<Path>,
    lora: &LoraConfig,
) -> Result<AnimaLoraPlan, WeightError> {
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let mut targets = Vec::new();
    for (key, view) in tensors.iter() {
        if let Some(target) = select_anima_target(key, view.shape(), lora)
            .map_err(|error| WeightError::Lora(error.to_string()))?
        {
            targets.push(target);
        }
    }
    targets.sort_by(|left, right| left.key.cmp(&right.key));
    Ok(AnimaLoraPlan { targets })
}

/// Loads one Anima Linear weight and wraps it in a Burn LoRA layer.
pub fn load_anima_lora_linear<B: Backend>(
    path: impl AsRef<Path>,
    key: &str,
    lora: &LoraConfig,
    device: &B::Device,
) -> Result<LoraLinear<B>, WeightError> {
    let config = AnimaConfig::default();
    if !config
        .lora_targets()
        .iter()
        .any(|target| key.ends_with(&format!("{target}.weight")))
    {
        return Err(WeightError::NotAnimaLoraTarget(key.to_owned()));
    }

    let file = File::open(path)?;
    // SafeTensors keeps zero-copy views into the mapped file. The map remains
    // alive until the selected tensor has been converted into a Burn tensor.
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let view = tensors
        .tensor(key)
        .map_err(|_| WeightError::MissingTensor(key.to_owned()))?;
    let shape = view.shape().to_vec();
    let [output_dim, input_dim] = shape.as_slice() else {
        return Err(WeightError::InvalidShape {
            key: key.to_owned(),
            shape,
        });
    };
    let values = tensor_values(view.dtype(), view.data())?;
    let mut layer = LoraLinearConfig {
        input_dim: *input_dim,
        output_dim: *output_dim,
        rank: lora.rank,
        alpha: lora.alpha,
        bias: false,
    }
    .init::<B>(device)
    .map_err(|error| WeightError::Lora(error.to_string()))?;
    let base = Tensor::from_data(TensorData::new(values, [*output_dim, *input_dim]), device);
    layer.weight = Param::from_tensor(base);
    Ok(layer)
}

fn tensor_values(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>, WeightError> {
    match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|bytes| Ok(f32::from_le_bytes(bytes.try_into().unwrap())))
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|bytes| {
                Ok(bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            })
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|bytes| Ok(f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32()))
            .collect(),
        other => Err(WeightError::UnsupportedDtype(format!("{other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_bfloat16_values() {
        let one = 1.0_f32.to_bits();
        let bytes = (one >> 16) as u16;
        let bytes = bytes.to_le_bytes();
        assert_eq!(tensor_values(Dtype::BF16, &bytes).unwrap(), vec![1.0]);
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let error = tensor_values(Dtype::I32, &[0; 4]).unwrap_err();
        assert!(matches!(error, WeightError::UnsupportedDtype(_)));
    }

    #[test]
    fn keeps_adapter_bias_components_distinct() {
        assert_eq!(
            map_anima_weight_key("model.diffusion_model.llm_adapter.blocks.0.mlp.0.bias"),
            Some(AnimaWeightSlot::LlmAdapter {
                index: 0,
                component: "mlp.0.bias".to_owned(),
            })
        );
        assert_eq!(
            map_anima_weight_key("model.diffusion_model.llm_adapter.out_proj.weight"),
            Some(AnimaWeightSlot::LlmAdapterGlobal {
                component: "out_proj".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_weight_shape_mismatch_before_tensor_creation() {
        let error = ensure_shape("layer.weight", &[3, 4], &[2, 4]).unwrap_err();
        assert!(matches!(error, WeightError::ShapeMismatch { .. }));
    }

    #[test]
    fn collects_lora_factors_through_burn_visitor_stack() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let layer = LoraLinearConfig {
            input_dim: 4,
            output_dim: 8,
            rank: 1,
            alpha: 1,
            bias: false,
        }
        .init::<Backend>(&device)
        .unwrap();
        let mut collector = LoraCollector::<Backend>::new(&LoraConfig::default());
        collector.enter_module("blocks", "AnimaDiT");
        collector.enter_module("0", "Vec");
        collector.enter_module("self_attn", "AnimaAttention");
        collector.enter_module("q_proj", "LoraLinear");
        collector.enter_module("lora_down", "Param");
        collector.visit_float(&layer.lora_down);
        assert_eq!(collector.tensors.len(), 1);
    }
}
