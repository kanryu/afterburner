//! SafeTensors weight loading for model-specific adapters.

use std::{borrow::Cow, collections::HashMap, fs::File, path::Path};

use burn::{
    module::{Module, ModuleDisplay, ModuleVisitor, Param, ParamId},
    tensor::{DType, ElementConversion, Tensor, TensorData, backend::Backend},
};
use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::{SafeTensors, tensor::Dtype};

use crate::{
    anima::AnimaConfig,
    dit::{
        AnimaAttentionLayer, AnimaAttentionStandard, AnimaDiT, AnimaDiTBlock, AnimaLlmAdapterBlock,
        BlockCheckpointBackend,
    },
    lora::{
        LoraConfig, LoraLinear, LoraLinearConfig, LoraTarget, normalize_anima_lora_key,
        select_anima_target,
    },
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
    // Kohya/ComfyUI checkpoints use model.diffusion_model.*, while the
    // native Anima checkpoint used by the current workflow stores the same
    // module tree below net.*. Normalize both forms before dispatching.
    let key = key
        .strip_prefix("model.diffusion_model.")
        .or_else(|| key.strip_prefix("net."))?;
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
pub fn load_anima_weights<B, A>(
    path: impl AsRef<Path>,
    model: &mut AnimaDiT<B, A>,
) -> Result<AnimaLoadReport, WeightError>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let device = model.input_projection.weight.lazy_device();
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
                    "self_attn.q_proj" => Some(&mut block.self_attn.standard_mut().q_proj),
                    "self_attn.k_proj" => Some(&mut block.self_attn.standard_mut().k_proj),
                    "self_attn.v_proj" => Some(&mut block.self_attn.standard_mut().v_proj),
                    "self_attn.output_proj" => {
                        Some(&mut block.self_attn.standard_mut().output_proj)
                    }
                    "cross_attn.q_proj" => Some(&mut block.cross_attn.standard_mut().q_proj),
                    "cross_attn.k_proj" => Some(&mut block.cross_attn.standard_mut().k_proj),
                    "cross_attn.v_proj" => Some(&mut block.cross_attn.standard_mut().v_proj),
                    "cross_attn.output_proj" => {
                        Some(&mut block.cross_attn.standard_mut().output_proj)
                    }
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
                if let Some(target) = target {
                    load_linear_weight(target, view, key, &device)?
                } else {
                    match component.as_str() {
                        "self_attn.q_norm" => load_rms_norm_weight(
                            &mut block.self_attn.standard_mut().q_norm,
                            view,
                            key,
                            &device,
                        )?,
                        "self_attn.k_norm" => load_rms_norm_weight(
                            &mut block.self_attn.standard_mut().k_norm,
                            view,
                            key,
                            &device,
                        )?,
                        "cross_attn.q_norm" => load_rms_norm_weight(
                            &mut block.cross_attn.standard_mut().q_norm,
                            view,
                            key,
                            &device,
                        )?,
                        "cross_attn.k_norm" => load_rms_norm_weight(
                            &mut block.cross_attn.standard_mut().k_norm,
                            view,
                            key,
                            &device,
                        )?,
                        _ => {
                            report.skipped.push(key.to_owned());
                            continue;
                        }
                    }
                }
            }
            AnimaWeightSlot::LlmAdapter { index, component } => {
                let Some(block) = model.llm_adapter.blocks.get_mut(index) else {
                    report.skipped.push(key.to_owned());
                    continue;
                };
                match component.as_str() {
                    "self_attn.q_proj" => load_linear_weight(
                        &mut block.self_attn.standard_mut().q_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "self_attn.k_proj" => load_linear_weight(
                        &mut block.self_attn.standard_mut().k_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "self_attn.v_proj" => load_linear_weight(
                        &mut block.self_attn.standard_mut().v_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "self_attn.o_proj" => load_linear_weight(
                        &mut block.self_attn.standard_mut().output_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "cross_attn.q_proj" => load_linear_weight(
                        &mut block.cross_attn.standard_mut().q_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "cross_attn.k_proj" => load_linear_weight(
                        &mut block.cross_attn.standard_mut().k_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "cross_attn.v_proj" => load_linear_weight(
                        &mut block.cross_attn.standard_mut().v_proj,
                        view,
                        key,
                        &device,
                    )?,
                    "self_attn.q_norm" => load_rms_norm_weight(
                        &mut block.self_attn.standard_mut().q_norm,
                        view,
                        key,
                        &device,
                    )?,
                    "self_attn.k_norm" => load_rms_norm_weight(
                        &mut block.self_attn.standard_mut().k_norm,
                        view,
                        key,
                        &device,
                    )?,
                    "cross_attn.q_norm" => load_rms_norm_weight(
                        &mut block.cross_attn.standard_mut().q_norm,
                        view,
                        key,
                        &device,
                    )?,
                    "cross_attn.k_norm" => load_rms_norm_weight(
                        &mut block.cross_attn.standard_mut().k_norm,
                        view,
                        key,
                        &device,
                    )?,
                    "cross_attn.o_proj" => load_linear_weight(
                        &mut block.cross_attn.standard_mut().output_proj,
                        view,
                        key,
                        &device,
                    )?,
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
pub fn load_anima_model<B: Backend + BlockCheckpointBackend>(
    path: impl AsRef<Path>,
    lora: &LoraConfig,
    device: &B::Device,
) -> Result<(AnimaDiT<B>, AnimaLoadReport), WeightError> {
    load_anima_model_with_attention::<B, AnimaAttentionStandard<B>>(path, lora, device)
}

/// Initializes Anima with an explicitly selected attention implementation.
pub fn load_anima_model_with_attention<B, A>(
    path: impl AsRef<Path>,
    lora: &LoraConfig,
    device: &B::Device,
) -> Result<(AnimaDiT<B, A>, AnimaLoadReport), WeightError>
where
    B: Backend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B>,
{
    let started = std::time::Instant::now();
    eprintln!("[load] AnimaDiT initialization started");
    let mut model = AnimaDiT::<B, A>::init(&AnimaConfig::default(), lora, device)
        .map_err(|error| WeightError::Lora(error.to_string()))?;
    eprintln!(
        "[load] AnimaDiT initialization finished in {:.3}s; loading SafeTensors",
        started.elapsed().as_secs_f64()
    );
    let weights_started = std::time::Instant::now();
    let report = load_anima_weights(path, &mut model)?;
    eprintln!(
        "[load] SafeTensors loading finished in {:.3}s (loaded={}, skipped={})",
        weights_started.elapsed().as_secs_f64(),
        report.loaded_count(),
        report.skipped_count()
    );
    Ok((model, report))
}

/// Loads an Afterburner-exported LoRA file into an initialized Anima model.
pub fn load_anima_lora_factors<B, A>(
    path: impl AsRef<Path>,
    model: &mut AnimaDiT<B, A>,
) -> Result<usize, WeightError>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
    let file = File::open(path)?;
    let mapped = unsafe { MmapOptions::new().map(&file) }.map_err(WeightError::Io)?;
    let tensors = SafeTensors::deserialize(&mapped)
        .map_err(|error| WeightError::SafeTensor(error.to_string()))?;
    let device = model.input_projection.weight.lazy_device();
    let mut loaded = 0;
    for (key, view) in tensors.iter() {
        let Some((target_name, factor)) = normalize_lora_factor_key(key) else {
            continue;
        };
        let Some(target) = lora_target_mut(model, &target_name) else {
            continue;
        };
        if factor != "down" && factor != "up" {
            continue;
        }
        let expected = if factor == "down" {
            target.lora_down.lazy_shape().dims
        } else {
            target.lora_up.lazy_shape().dims
        };
        ensure_shape(key, view.shape(), &expected)?;
        // Python kohya keeps LoRA parameters in FP32 unless full_fp16 or
        // full_bf16 is explicitly requested. The model activation dtype is
        // selected independently at forward time, so do not let the CUDA
        // backend's default dtype silently quantize these trainable factors.
        // Materialize a fresh FP32 leaf; the forward path casts only the
        // operands used by the low-precision projection.
        let tensor = load_tensor::<B, 2>(view, key, &device)?.cast(DType::F32);
        if factor == "down" {
            target.lora_down = Param::from_tensor(tensor);
        } else {
            target.lora_up = Param::from_tensor(tensor);
        }
        loaded += 1;
    }
    Ok(loaded)
}

fn normalize_lora_factor_key(key: &str) -> Option<(String, String)> {
    let key = key.strip_suffix(".weight")?;
    let (target, factor) = key.rsplit_once(".lora_")?;
    if factor != "down" && factor != "up" {
        return None;
    }
    if let Some(target) = target
        .strip_prefix("diffusion_model.")
        .or_else(|| target.strip_prefix("model.diffusion_model."))
    {
        let target = match target {
            name if name.ends_with(".mlp.layer1") => {
                format!("{}mlp_layer1", &name[..name.len() - ".mlp.layer1".len()])
            }
            name if name.ends_with(".mlp.layer2") => {
                format!("{}mlp_layer2", &name[..name.len() - ".mlp.layer2".len()])
            }
            name if name.ends_with(".mlp.0") => {
                format!("{}mlp_layer1", &name[..name.len() - ".mlp.0".len()])
            }
            name if name.ends_with(".mlp.2") => {
                format!("{}mlp_layer2", &name[..name.len() - ".mlp.2".len()])
            }
            name => name.to_owned(),
        };
        return Some((target, factor.to_owned()));
    }
    let target = target.strip_prefix("lora_unet_")?;
    let target = if let Some(rest) = target.strip_prefix("blocks_") {
        let (index, component) = rest.split_once('_')?;
        format!("blocks.{index}.{}", flatten_lora_component(component)?)
    } else if let Some(rest) = target.strip_prefix("llm_adapter_blocks_") {
        let (index, component) = rest.split_once('_')?;
        format!(
            "llm_adapter.blocks.{index}.{}",
            flatten_lora_component(component)?
        )
    } else {
        return None;
    };
    Some((target, factor.to_owned()))
}

fn flatten_lora_component(component: &str) -> Option<&'static str> {
    match component {
        "self_attn_q_proj" => Some("self_attn.q_proj"),
        "self_attn_k_proj" => Some("self_attn.k_proj"),
        "self_attn_v_proj" => Some("self_attn.v_proj"),
        "self_attn_output_proj" => Some("self_attn.output_proj"),
        "cross_attn_q_proj" => Some("cross_attn.q_proj"),
        "cross_attn_k_proj" => Some("cross_attn.k_proj"),
        "cross_attn_v_proj" => Some("cross_attn.v_proj"),
        "cross_attn_output_proj" => Some("cross_attn.output_proj"),
        "mlp_layer1" => Some("mlp_layer1"),
        "mlp_layer2" => Some("mlp_layer2"),
        _ => None,
    }
}

fn lora_target_mut<'a, B, A>(
    model: &'a mut AnimaDiT<B, A>,
    name: &str,
) -> Option<&'a mut LoraLinear<B>>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
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

fn block_lora_target_mut<'a, B, A>(
    block: &'a mut AnimaDiTBlock<B, A>,
    component: &str,
) -> Option<&'a mut LoraLinear<B>>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
    match component {
        "self_attn.q_proj" => Some(&mut block.self_attn.standard_mut().q_proj),
        "self_attn.k_proj" => Some(&mut block.self_attn.standard_mut().k_proj),
        "self_attn.v_proj" => Some(&mut block.self_attn.standard_mut().v_proj),
        "self_attn.output_proj" => Some(&mut block.self_attn.standard_mut().output_proj),
        "cross_attn.q_proj" => Some(&mut block.cross_attn.standard_mut().q_proj),
        "cross_attn.k_proj" => Some(&mut block.cross_attn.standard_mut().k_proj),
        "cross_attn.v_proj" => Some(&mut block.cross_attn.standard_mut().v_proj),
        "cross_attn.output_proj" => Some(&mut block.cross_attn.standard_mut().output_proj),
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

fn adapter_lora_target_mut<'a, B, A>(
    block: &'a mut AnimaLlmAdapterBlock<B, A>,
    component: &str,
) -> Option<&'a mut LoraLinear<B>>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
    match component {
        "self_attn.q_proj" => Some(&mut block.self_attn.standard_mut().q_proj),
        "self_attn.k_proj" => Some(&mut block.self_attn.standard_mut().k_proj),
        "self_attn.v_proj" => Some(&mut block.self_attn.standard_mut().v_proj),
        "self_attn.output_proj" => Some(&mut block.self_attn.standard_mut().output_proj),
        "cross_attn.q_proj" => Some(&mut block.cross_attn.standard_mut().q_proj),
        "cross_attn.k_proj" => Some(&mut block.cross_attn.standard_mut().k_proj),
        "cross_attn.v_proj" => Some(&mut block.cross_attn.standard_mut().v_proj),
        "cross_attn.output_proj" => Some(&mut block.cross_attn.standard_mut().output_proj),
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
    ensure_shape(key, view.shape(), &target.weight.lazy_shape().dims)?;
    // Base weights are frozen during LoRA training.  Do not create a
    // require-grad Autodiff leaf and detach it afterwards: that transient
    // leaf can keep the initialization allocation alive on CUDA.  Install a
    // detached tensor directly as a non-trainable parameter.
    let tensor = load_tensor::<B, 2>(view, key, device)?.cast(base_weight_dtype()).detach();
    target.weight = Param::initialized(ParamId::new(), tensor);
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
        &target.bias.as_ref().unwrap().lazy_shape().dims,
    )?;
    let tensor = load_tensor::<B, 1>(view, key, device)?.cast(base_weight_dtype()).detach();
    target.bias = Some(Param::initialized(ParamId::new(), tensor));
    Ok(true)
}

fn load_rms_norm_weight<B: Backend>(
    target: &mut crate::dit::AnimaRmsNorm<B>,
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<bool, WeightError> {
    ensure_shape(key, view.shape(), &target.gamma.lazy_shape().dims)?;
    let tensor = load_tensor::<B, 1>(view, key, device)?.cast(base_weight_dtype()).detach();
    target.gamma = Param::initialized(ParamId::new(), tensor);
    Ok(true)
}

fn load_embedding_weight<B: Backend>(
    target: &mut burn::nn::Embedding<B>,
    view: safetensors::tensor::TensorView<'_>,
    key: &str,
    device: &B::Device,
) -> Result<bool, WeightError> {
    ensure_shape(key, view.shape(), &target.weight.lazy_shape().dims)?;
    let tensor = load_tensor::<B, 2>(view, key, device)?.cast(base_weight_dtype()).detach();
    target.weight = Param::initialized(ParamId::new(), tensor);
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

fn base_weight_dtype() -> DType {
    // Base parameters are frozen for LoRA training and never need an FP32
    // master copy.  Keeping the two-billion-parameter base in F32 doubles
    // its resident VRAM and causes every fused projection to cast a much
    // larger tensor.  BF16 is the training representation used by the CUDA
    // path; retain the environment switch as an explicit compatibility knob.
    if std::env::var_os("AFTERBURNER_BASE_F32").is_some() {
        DType::F32
    } else {
        DType::BF16
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

/// Storage dtype for exported LoRA factor tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraOutputDtype {
    F32,
    F16,
    BF16,
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
    alpha: f32,
    output_dtype: LoraOutputDtype,
    path: Vec<String>,
    marker: std::marker::PhantomData<B>,
}

impl<B: Backend> LoraCollector<B> {
    fn new(lora: &LoraConfig, output_dtype: LoraOutputDtype) -> Self {
        Self {
            tensors: Vec::new(),
            error: None,
            target_modules: lora.target_modules.clone(),
            alpha: lora.alpha as f32,
            output_dtype,
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
        // The CUDA attention implementation wraps the portable attention
        // module in an additional `standard` field.  That implementation
        // detail must not change the logical LoRA target name used by
        // kohya_ss/ComfyUI.  Normalize both the portable path
        // (`blocks.0.self_attn.q_proj`) and the CUDA path
        // (`blocks.0.self_attn.standard.q_proj`) to the same target.
        let logical_path = path[2..path.len() - 1]
            .iter()
            .filter(|component| component.as_str() != "standard")
            .cloned()
            .collect::<Vec<_>>()
            .join(".");
        let internal_module = match logical_path.as_str() {
            "mlp_layer1" => "mlp.layer1".to_owned(),
            "mlp_layer2" => "mlp.layer2".to_owned(),
            value => value.replace(".standard.", "."),
        };
        if !self
            .target_modules
            .iter()
            .any(|target| target == &internal_module)
        {
            return;
        }
        // ComfyUI's Anima GPT2FeedForward exposes the projections as
        // mlp.layer1 and mlp.layer2, matching the native checkpoint names.
        let module = internal_module;
        // Match kohya_ss/networks.lora_anima, which is the native Anima LoRA
        // format recognized by ComfyUI. Dots in the target path are flattened
        // to underscores under the lora_unet_ prefix.
        let target_key = format!("{}.{}.{}", path[0], path[1], module).replace('.', "_");
        let key_base = format!("lora_unet_{target_key}");
        let key = format!("{key_base}.{name}.weight");
        if name == "lora_down" {
            let (alpha_dtype, alpha_data) = match self.output_dtype {
                LoraOutputDtype::F32 => (Dtype::F32, self.alpha.to_le_bytes().to_vec()),
                // kohya_ss emits alpha in the same storage dtype as the LoRA
                // factors.  Keeping BF16 alpha avoids a dtype-only mismatch
                // with otherwise identical Anima adapters in ComfyUI.
                LoraOutputDtype::BF16 => (
                    Dtype::BF16,
                    bf16::from_f32(self.alpha).to_le_bytes().to_vec(),
                ),
                LoraOutputDtype::F16 => (
                    Dtype::F16,
                    f16::from_f32(self.alpha).to_le_bytes().to_vec(),
                ),
            };
            self.tensors.push((
                format!("{key_base}.alpha"),
                OwnedTensorView {
                    dtype: alpha_dtype,
                    shape: Vec::new(),
                    data: alpha_data,
                },
            ));
        }
        let tensor = param.val();
        let data = match tensor.to_data().to_vec::<B::FloatElem>() {
            Ok(values) => values
                .iter()
                .map(|value| value.elem::<f32>())
                .flat_map(|value| match self.output_dtype {
                    LoraOutputDtype::F32 => value.to_le_bytes().to_vec(),
                    LoraOutputDtype::BF16 => bf16::from_f32(value).to_le_bytes().to_vec(),
                    LoraOutputDtype::F16 => f16::from_f32(value).to_le_bytes().to_vec(),
                })
                .collect(),
            Err(error) => {
                self.error = Some(format!("{error:?}"));
                return;
            }
        };
        self.tensors.push((
            key,
            OwnedTensorView {
                dtype: match self.output_dtype {
                    LoraOutputDtype::F32 => Dtype::F32,
                    LoraOutputDtype::BF16 => Dtype::BF16,
                    LoraOutputDtype::F16 => Dtype::F16,
                },
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
    export_anima_lora_with_config_and_dtype(path, model, lora, LoraOutputDtype::F32)
}

/// Exports configured LoRA targets with an explicit SafeTensors storage dtype.
/// The output dtype is independent of the Burn training backend dtype.
pub fn export_anima_lora_with_config_and_dtype<B: Backend>(
    path: impl AsRef<Path>,
    model: &AnimaDiT<B>,
    lora: &LoraConfig,
    output_dtype: LoraOutputDtype,
) -> Result<usize, WeightError> {
    export_anima_lora_with_config_and_dtype_with_attention(path, model, lora, output_dtype)
}

/// Generic counterpart used when the runtime selects a specialized
/// attention implementation, such as the CUDA fused boundary.
pub fn export_anima_lora_with_config_and_dtype_with_attention<B, A>(
    path: impl AsRef<Path>,
    model: &AnimaDiT<B, A>,
    lora: &LoraConfig,
    output_dtype: LoraOutputDtype,
) -> Result<usize, WeightError>
where
    B: Backend,
    A: Module<B> + ModuleDisplay,
{
    lora.validate()
        .map_err(|error| WeightError::Export(error.to_string()))?;
    let mut collector = LoraCollector::<B>::new(lora, output_dtype);
    model.visit(&mut collector);
    if let Some(error) = collector.error {
        return Err(WeightError::Export(error));
    }
    if collector.tensors.is_empty() {
        return Err(WeightError::Export("model contains no LoRA factors".into()));
    }
    let count = collector
        .tensors
        .iter()
        .filter(|(key, _)| key.ends_with(".lora_down.weight") || key.ends_with(".lora_up.weight"))
        .count();
    let mut metadata = HashMap::new();
    metadata.insert("format".into(), "afterburner-anima-lora-v1".into());
    metadata.insert("base_model".into(), "anima".into());
    // Emit the standard Anima/kohya_ss model-spec fields as well.  ComfyUI
    // can load the tensor keys without them, but the fields make the adapter
    // unambiguous to loaders that select the network implementation and
    // scaling from SafeTensors metadata.
    metadata.insert("modelspec.architecture".into(), "anima-preview/lora".into());
    metadata.insert("modelspec.sai_model_spec".into(), "1.0.1".into());
    metadata.insert("ss_base_model_version".into(), "anima".into());
    metadata.insert("ss_network_module".into(), "networks.lora_anima".into());
    metadata.insert("ss_network_dim".into(), lora.rank.to_string());
    metadata.insert("ss_network_alpha".into(), lora.alpha.to_string());
    metadata.insert("ss_unet_lr".into(), "0.0001".into());
    metadata.insert(
        "dtype".into(),
        match output_dtype {
            LoraOutputDtype::F32 => "F32".into(),
            LoraOutputDtype::BF16 => "BF16".into(),
            LoraOutputDtype::F16 => "F16".into(),
        },
    );
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
    pub f32_tensors: usize,
    pub bf16_tensors: usize,
    pub f16_tensors: usize,
    pub storage_bytes: usize,
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
        match view.dtype() {
            Dtype::F32 => plan.f32_tensors += 1,
            Dtype::BF16 => plan.bf16_tensors += 1,
            Dtype::F16 => plan.f16_tensors += 1,
            _ => {}
        }
        plan.storage_bytes += view.data().len();
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
    let mut factors: HashMap<String, (Option<Vec<usize>>, Option<Vec<usize>>)> = HashMap::new();
    for (key, view) in tensors.iter() {
        if let Some(target) = normalize_anima_lora_key(key)
            && (key.ends_with(".lora_down.weight") || key.ends_with(".lora_up.weight"))
        {
            if view.shape().len() != 2 {
                return Err(WeightError::InvalidRank {
                    key: key.to_owned(),
                    expected: 2,
                    actual: view.shape().len(),
                });
            }
            let entry = factors.entry(target).or_default();
            if key.ends_with(".lora_down.weight") {
                entry.0 = Some(view.shape().to_vec());
            } else {
                entry.1 = Some(view.shape().to_vec());
            }
            continue;
        }
        if let Some(target) = select_anima_target(key, view.shape(), lora)
            .map_err(|error| WeightError::Lora(error.to_string()))?
        {
            targets.push(target);
        }
    }
    for (key, (down, up)) in factors {
        let Some(down) = down else {
            return Err(WeightError::MissingTensor(format!(
                "{key}.lora_down.weight"
            )));
        };
        let Some(up) = up else {
            return Err(WeightError::MissingTensor(format!("{key}.lora_up.weight")));
        };
        if down[0] != up[1] {
            return Err(WeightError::ShapeMismatch {
                key,
                expected: vec![down[0], down[1]],
                actual: up,
            });
        }
        targets.push(LoraTarget {
            key,
            kind: crate::lora::LoraTargetKind::Linear,
            output_dim: up[0],
            input_dim: down[1],
            rank: down[0],
            alpha: lora.alpha,
        });
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
    let base = Tensor::<B, 2>::from_data(
        TensorData::new(values, [*output_dim, *input_dim]),
        device,
    );
    let base = Tensor::<B, 2>::cast(base, DType::F32);
    layer.weight = Param::from_tensor(base).map(|tensor| tensor.detach());
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
    fn maps_rms_norm_weights_for_dit_and_llm_adapter_blocks() {
        let keys = [
            (
                "model.diffusion_model.blocks.0.self_attn.q_norm.weight",
                AnimaWeightSlot::Block {
                    index: 0,
                    component: "self_attn.q_norm".to_owned(),
                },
            ),
            (
                "model.diffusion_model.blocks.27.cross_attn.k_norm.weight",
                AnimaWeightSlot::Block {
                    index: 27,
                    component: "cross_attn.k_norm".to_owned(),
                },
            ),
            (
                "model.diffusion_model.llm_adapter.blocks.0.self_attn.q_norm.weight",
                AnimaWeightSlot::LlmAdapter {
                    index: 0,
                    component: "self_attn.q_norm".to_owned(),
                },
            ),
            (
                "net.llm_adapter.blocks.5.cross_attn.k_norm.weight",
                AnimaWeightSlot::LlmAdapter {
                    index: 5,
                    component: "cross_attn.k_norm".to_owned(),
                },
            ),
        ];
        for (key, expected) in keys {
            assert_eq!(map_anima_weight_key(key), Some(expected), "key={key}");
        }
    }

    #[test]
    fn normalizes_comfyui_lora_unet_factor_keys() {
        assert_eq!(
            normalize_lora_factor_key("lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"),
            Some(("blocks.0.self_attn.q_proj".to_owned(), "down".to_owned()))
        );
        assert_eq!(
            normalize_lora_factor_key(
                "lora_unet_llm_adapter_blocks_5_cross_attn_output_proj.lora_up.weight"
            ),
            Some((
                "llm_adapter.blocks.5.cross_attn.output_proj".to_owned(),
                "up".to_owned()
            ))
        );
        assert_eq!(
            normalize_lora_factor_key("lora_unet_blocks_0_unknown.lora_up.weight"),
            None
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
        let mut collector =
            LoraCollector::<Backend>::new(&LoraConfig::default(), LoraOutputDtype::F32);
        collector.enter_module("blocks", "AnimaDiT");
        collector.enter_module("0", "Vec");
        collector.enter_module("self_attn", "AnimaAttention");
        collector.enter_module("q_proj", "LoraLinear");
        collector.enter_module("lora_down", "Param");
        collector.visit_float(&layer.lora_down);
        assert_eq!(collector.tensors.len(), 2);
    }

    #[test]
    fn inspects_lora_safetensors_factors_and_ignores_metadata() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-lora-roundtrip-{}.safetensors",
            std::process::id()
        ));
        let down = vec![0.1_f32, 0.2, 0.3, 0.4];
        let up = vec![0.5_f32, 0.6, 0.7, 0.8];
        let down_bytes: Vec<u8> = down.iter().flat_map(|value| value.to_le_bytes()).collect();
        let up_bytes: Vec<u8> = up.iter().flat_map(|value| value.to_le_bytes()).collect();
        let tensors = vec![
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![1, 4], &down_bytes).unwrap(),
            ),
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![4, 1], &up_bytes).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        let plan = inspect_anima_lora_factors(&path).unwrap();
        assert_eq!(plan.tensor_count(), 2);
        assert_eq!(plan.parameter_count, 8);
        assert_eq!(plan.f32_tensors, 2);
        assert_eq!(plan.bf16_tensors, 0);
        assert_eq!(plan.storage_bytes, 32);
        assert!(plan.keys.iter().all(|key| key.ends_with(".weight")));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn inspects_comfyui_lora_factor_pair_as_one_target() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-lora-plan-{}.safetensors",
            std::process::id()
        ));
        let down_bytes = vec![0_u8; 2 * 4 * 4];
        let up_bytes = vec![0_u8; 8 * 2 * 4];
        let tensors = vec![
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 4], &down_bytes).unwrap(),
            ),
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![8, 2], &up_bytes).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();

        let mut config = LoraConfig::default();
        config.rank = 2;
        config.alpha = 2;
        let plan = inspect_anima_lora_file(&path, &config).unwrap();
        assert_eq!(plan.target_count(), 1);
        assert_eq!(plan.parameter_count(), 24);
        assert_eq!(plan.targets[0].key, "blocks.0.self_attn.q_proj");
        assert_eq!(plan.targets[0].rank, 2);
        assert_eq!(
            (plan.targets[0].output_dim, plan.targets[0].input_dim),
            (8, 4)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_incomplete_comfyui_lora_factor_pair() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-lora-incomplete-{}.safetensors",
            std::process::id()
        ));
        let data = vec![0_u8; 2 * 4 * 4];
        let tensors = vec![(
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight".to_owned(),
            safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 4], &data).unwrap(),
        )];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        let error = inspect_anima_lora_file(&path, &LoraConfig::default()).unwrap_err();
        assert!(
            matches!(error, WeightError::MissingTensor(key) if key.ends_with(".lora_up.weight"))
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_comfyui_lora_factor_rank_mismatch() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-lora-rank-mismatch-{}.safetensors",
            std::process::id()
        ));
        let down = vec![0_u8; 2 * 4 * 4];
        let up = vec![0_u8; 8 * 3 * 4];
        let tensors = vec![
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 4], &down).unwrap(),
            ),
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![8, 3], &up).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        let error = inspect_anima_lora_file(&path, &LoraConfig::default()).unwrap_err();
        assert!(matches!(error, WeightError::ShapeMismatch { .. }));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reports_bfloat16_lora_storage_size() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-lora-bf16-stats-{}.safetensors",
            std::process::id()
        ));
        let down = vec![0_u8; 2 * 4 * 2];
        let up = vec![0_u8; 8 * 2 * 2];
        let tensors = vec![
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::BF16, vec![2, 4], &down).unwrap(),
            ),
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::BF16, vec![8, 2], &up).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        let plan = inspect_anima_lora_factors(&path).unwrap();
        assert_eq!(plan.tensor_count(), 2);
        assert_eq!(plan.parameter_count, 24);
        assert_eq!(plan.bf16_tensors, 2);
        assert_eq!(plan.f32_tensors, 0);
        assert_eq!(plan.storage_bytes, 48);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_comfyui_lora_factors_into_small_anima_model() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaConfig {
            block_count: 1,
            hidden_size: 16,
            text_hidden_size: 16,
            attention_head_count: 1,
            attention_head_size: 16,
            mlp_hidden_size: 32,
            input_channels: 4,
            output_channels: 4,
            llm_adapter_block_count: 1,
            llm_adapter_hidden_size: 1024,
            text_vocab_size: 32,
        };
        let mut lora = LoraConfig::default();
        lora.rank = 1;
        lora.alpha = 1;
        let mut model: AnimaDiT<Backend> = AnimaDiT::init(&config, &lora, &device).unwrap();
        let path = std::env::temp_dir().join(format!(
            "afterburner-lora-load-{}.safetensors",
            std::process::id()
        ));
        let down = vec![1.0_f32; 16];
        let up = vec![2.0_f32; 16];
        let down_bytes: Vec<u8> = down.iter().flat_map(|value| value.to_le_bytes()).collect();
        let up_bytes: Vec<u8> = up.iter().flat_map(|value| value.to_le_bytes()).collect();
        let tensors = vec![
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![1, 16], &down_bytes).unwrap(),
            ),
            (
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight".to_owned(),
                safetensors::tensor::TensorView::new(Dtype::F32, vec![16, 1], &up_bytes).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, None, &path).unwrap();
        assert_eq!(load_anima_lora_factors(&path, &mut model).unwrap(), 2);
        let down_values = model.blocks[0]
            .self_attn
            .q_proj
            .lora_down
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(down_values, down);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "reads the full Anima checkpoint; set AFTERBURNER_ANIMA_PATH to run"]
    fn anima_checkpoint_key_coverage_matches_reference_counts() {
        let path = std::env::var_os("AFTERBURNER_ANIMA_PATH")
            .expect("AFTERBURNER_ANIMA_PATH must point to anima_baseV10.safetensors");
        let coverage = inspect_anima_weight_coverage(path).unwrap();
        // Key classification recognizes every tensor in the reference
        // checkpoint, including the per-block q_norm/k_norm parameters.
        assert_eq!(coverage.mapped_count(), 685);
        assert_eq!(coverage.unmapped_count(), 0);
        assert!(
            coverage
                .mapped
                .iter()
                .any(|(key, _)| key.ends_with("blocks.0.self_attn.q_proj.weight"))
        );
        assert!(
            coverage
                .mapped
                .iter()
                .any(|(key, _)| key.contains("llm_adapter.out_proj.weight"))
        );
        let norm_count = coverage
            .mapped
            .iter()
            .filter(|(_, slot)| {
                matches!(
                    slot,
                    AnimaWeightSlot::Block { component, .. }
                        | AnimaWeightSlot::LlmAdapter { component, .. }
                        if component.ends_with("_norm")
                )
            })
            .count();
        assert_eq!(norm_count, 136);
    }

    #[test]
    #[ignore = "reads a reference Kohya_ss LoRA; set AFTERBURNER_REFERENCE_LORA_PATH to run"]
    fn kohya_reference_lora_plan_matches_anima_rank_eight_counts() {
        let path = std::env::var_os("AFTERBURNER_REFERENCE_LORA_PATH")
            .expect("AFTERBURNER_REFERENCE_LORA_PATH must point to a Kohya_ss LoRA");
        let plan = inspect_anima_lora_file(path, &LoraConfig::default()).unwrap();
        assert_eq!(plan.target_count(), 280);
        assert_eq!(plan.parameter_count(), 11_468_800);
        assert!(plan.targets.iter().all(|target| target.rank == 8));
    }
}
