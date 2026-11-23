//! Burn implementation skeleton for the Anima diffusion transformer.
//!
//! The attention and MLP paths are functional. Weight loading for the
//! individual modules and Anima's AdaLN/timestep conditioning are kept as
//! separate steps because their exact ordering must be verified against the
//! reference engine.

use burn::{
    module::{Initializer, Module, ModuleDisplay, Param, ParamId},
    nn::{Embedding, EmbeddingConfig},
    tensor::{
        ElementConversion, FloatDType, Int, Tensor, TensorData, TensorPrimitive, activation::gelu,
        activation::silu, backend::Backend,
    },
};
use burn_autodiff::{
    checkpoint_block::{register_dynamic_checkpoint, GradBridge},
    Autodiff, AutodiffTensor, NodeRef, Requirement,
};
use burn_autodiff::checkpoint::strategy::{CheckpointStrategy, NoCheckpointing};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    anima::AnimaConfig,
    lora::{LoraConfig, LoraLinear, LoraLinearConfig},
};

/// Upper bound for the query dimension of one attention score matrix.
///
/// With checkpointing enabled, the per-chunk operations can be recomputed
/// during backward instead of retaining every intermediate activation.
const ATTENTION_QUERY_CHUNK_SIZE: usize = 256;
static ATTENTION_CHUNK_SIZE: AtomicUsize = AtomicUsize::new(ATTENTION_QUERY_CHUNK_SIZE);
static FORWARD_DUMP_STEP: AtomicUsize = AtomicUsize::new(0);
static DIT_BLOCK_FORWARD_INDEX: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static DEBUG_ATTENTION_GRAPH_CALLS: AtomicUsize = AtomicUsize::new(0);

pub trait BlockCheckpointBackend: Backend {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3>;
}

// Block-scoped checkpoint bridge.  The outer graph stores only the block
// output and its tracked parents; backward reconstructs one isolated local
// graph and returns its VJP to the outer graph.
pub type LocalB<B> = Autodiff<B, NoCheckpointing>;

pub struct ParamSnap<B: Backend, const D: usize> {
    param_id: ParamId,
    outer_node: NodeRef,
    primitive: B::FloatTensorPrimitive,
}

impl<B: Backend, const D: usize> ParamSnap<B, D> {
    fn capture<C: CheckpointStrategy>(param: &Param<Tensor<Autodiff<B, C>, D>>, parents: &mut Vec<NodeRef>) -> Self {
        let ad: AutodiffTensor<B> = match param.val().into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized parameters are unsupported"),
        };
        parents.push(ad.node.clone());
        Self { param_id: param.id, outer_node: ad.node, primitive: ad.primitive }
    }

    fn into_local(self, bridges: &mut Vec<GradBridge<B>>) -> Param<Tensor<LocalB<B>, D>> {
        let mut local = AutodiffTensor::new(self.primitive);
        if !matches!(self.outer_node.requirement, Requirement::None) {
            local = local.require_grad();
            bridges.push(GradBridge { outer_node_id: self.outer_node.id, local_tensor: local.clone() });
        }
        Param::initialized(self.param_id, Tensor::from_primitive(TensorPrimitive::Float(local)))
    }

    fn inner<C: CheckpointStrategy>(param: &Param<Tensor<Autodiff<B, C>, D>>) -> Param<Tensor<B, D>> {
        Param::initialized(param.id, param.val().inner())
    }
}

pub struct AnimaRmsNormSnap<B: Backend> { gamma: ParamSnap<B, 1>, epsilon: f32 }

impl<B: Backend> AnimaRmsNormSnap<B> {
    fn capture<C: CheckpointStrategy>(norm: &AnimaRmsNorm<Autodiff<B, C>>, parents: &mut Vec<NodeRef>) -> Self {
        Self { gamma: ParamSnap::capture(&norm.gamma, parents), epsilon: norm.epsilon }
    }
    fn into_local(self, bridges: &mut Vec<GradBridge<B>>) -> AnimaRmsNorm<LocalB<B>> {
        AnimaRmsNorm { gamma: self.gamma.into_local(bridges), epsilon: self.epsilon }
    }
    fn inner<C: CheckpointStrategy>(norm: &AnimaRmsNorm<Autodiff<B, C>>) -> AnimaRmsNorm<B> {
        AnimaRmsNorm { gamma: ParamSnap::inner(&norm.gamma), epsilon: norm.epsilon }
    }
}

pub struct LoraLinearSnap<B: Backend> {
    weight: ParamSnap<B, 2>, bias: Option<ParamSnap<B, 1>>,
    lora_down: ParamSnap<B, 2>, lora_up: ParamSnap<B, 2>, rank: usize, alpha: usize,
}

impl<B: Backend> LoraLinearSnap<B> {
    fn capture<C: CheckpointStrategy>(layer: &LoraLinear<Autodiff<B, C>>, parents: &mut Vec<NodeRef>) -> Self {
        Self {
            weight: ParamSnap::capture(&layer.weight, parents),
            bias: layer.bias.as_ref().map(|v| ParamSnap::capture(v, parents)),
            lora_down: ParamSnap::capture(&layer.lora_down, parents),
            lora_up: ParamSnap::capture(&layer.lora_up, parents), rank: layer.rank, alpha: layer.alpha,
        }
    }
    fn into_local(self, bridges: &mut Vec<GradBridge<B>>) -> LoraLinear<LocalB<B>> {
        LoraLinear { weight: self.weight.into_local(bridges), bias: self.bias.map(|v| v.into_local(bridges)),
            lora_down: self.lora_down.into_local(bridges), lora_up: self.lora_up.into_local(bridges), rank: self.rank, alpha: self.alpha }
    }
    fn inner<C: CheckpointStrategy>(layer: &LoraLinear<Autodiff<B, C>>) -> LoraLinear<B> {
        LoraLinear { weight: ParamSnap::inner(&layer.weight), bias: layer.bias.as_ref().map(ParamSnap::inner),
            lora_down: ParamSnap::inner(&layer.lora_down), lora_up: ParamSnap::inner(&layer.lora_up), rank: layer.rank, alpha: layer.alpha }
    }
}

pub struct AdaLnModulationSnap<B: Backend> { down: LoraLinearSnap<B>, up: LoraLinearSnap<B>, hidden_size: usize }

impl<B: Backend> AdaLnModulationSnap<B> {
    fn capture<C: CheckpointStrategy>(m: &AdaLnModulation<Autodiff<B, C>>, parents: &mut Vec<NodeRef>) -> Self {
        Self { down: LoraLinearSnap::capture(&m.down, parents), up: LoraLinearSnap::capture(&m.up, parents), hidden_size: m.hidden_size }
    }
    fn into_local(self, bridges: &mut Vec<GradBridge<B>>) -> AdaLnModulation<LocalB<B>> {
        AdaLnModulation { down: self.down.into_local(bridges), up: self.up.into_local(bridges), hidden_size: self.hidden_size }
    }
    fn inner<C: CheckpointStrategy>(m: &AdaLnModulation<Autodiff<B, C>>) -> AdaLnModulation<B> {
        AdaLnModulation { down: LoraLinearSnap::inner(&m.down), up: LoraLinearSnap::inner(&m.up), hidden_size: m.hidden_size }
    }
}

pub trait AnimaAttnCheckpoint<B: Backend, C>: Sized {
    type InnerAttn: AnimaAttentionLayer<B>;
    type LocalAttn: AnimaAttentionLayer<LocalB<B>>;
    type Snapshot: Send + 'static;
    fn inner(&self) -> Self::InnerAttn;
    fn capture(&self, parents: &mut Vec<NodeRef>) -> Self::Snapshot;
    fn restore(snap: Self::Snapshot, bridges: &mut Vec<GradBridge<B>>) -> Self::LocalAttn;
}

pub struct AnimaAttentionStandardSnap<B: Backend> {
    q_proj: LoraLinearSnap<B>, k_proj: LoraLinearSnap<B>, v_proj: LoraLinearSnap<B>, output_proj: LoraLinearSnap<B>,
    q_norm: AnimaRmsNormSnap<B>, k_norm: AnimaRmsNormSnap<B>, head_count: usize, head_size: usize, use_rotary: bool,
}

impl<B: Backend, C: CheckpointStrategy> AnimaAttnCheckpoint<B, C> for AnimaAttentionStandard<Autodiff<B, C>> {
    type InnerAttn = AnimaAttentionStandard<B>;
    type LocalAttn = AnimaAttentionStandard<LocalB<B>>;
    type Snapshot = AnimaAttentionStandardSnap<B>;
    fn inner(&self) -> Self::InnerAttn { AnimaAttentionStandard { q_proj: LoraLinearSnap::inner(&self.q_proj), k_proj: LoraLinearSnap::inner(&self.k_proj), v_proj: LoraLinearSnap::inner(&self.v_proj), output_proj: LoraLinearSnap::inner(&self.output_proj), q_norm: AnimaRmsNormSnap::inner(&self.q_norm), k_norm: AnimaRmsNormSnap::inner(&self.k_norm), head_count: self.head_count, head_size: self.head_size, use_rotary: self.use_rotary } }
    fn capture(&self, p: &mut Vec<NodeRef>) -> Self::Snapshot { AnimaAttentionStandardSnap { q_proj: LoraLinearSnap::capture(&self.q_proj,p), k_proj: LoraLinearSnap::capture(&self.k_proj,p), v_proj: LoraLinearSnap::capture(&self.v_proj,p), output_proj: LoraLinearSnap::capture(&self.output_proj,p), q_norm: AnimaRmsNormSnap::capture(&self.q_norm,p), k_norm: AnimaRmsNormSnap::capture(&self.k_norm,p), head_count:self.head_count, head_size:self.head_size, use_rotary:self.use_rotary } }
    fn restore(s: Self::Snapshot, b: &mut Vec<GradBridge<B>>) -> Self::LocalAttn { AnimaAttentionStandard { q_proj:s.q_proj.into_local(b), k_proj:s.k_proj.into_local(b), v_proj:s.v_proj.into_local(b), output_proj:s.output_proj.into_local(b), q_norm:s.q_norm.into_local(b), k_norm:s.k_norm.into_local(b), head_count:s.head_count, head_size:s.head_size, use_rotary:s.use_rotary } }
}

#[cfg(feature = "cuda")]
impl<B: Backend + BlockCheckpointBackend, C: CheckpointStrategy> AnimaAttnCheckpoint<B, C> for AnimaAttentionCuda<Autodiff<B, C>> {
    type InnerAttn = AnimaAttentionStandard<B>;
    type LocalAttn = AnimaAttentionCuda<LocalB<B>>;
    type Snapshot = AnimaAttentionStandardSnap<B>;
    fn inner(&self) -> Self::InnerAttn { self.standard.inner() }
    fn capture(&self, p: &mut Vec<NodeRef>) -> Self::Snapshot { self.standard.capture(p) }
    fn restore(s: Self::Snapshot, b: &mut Vec<GradBridge<B>>) -> Self::LocalAttn {
        AnimaAttentionCuda { standard: AnimaAttentionStandard::<LocalB<B>>::restore(s, b) }
    }
}

pub struct AnimaDiTBlockInputs<B: Backend> {
    pub input: Tensor<B,3>, pub text: Tensor<B,3>, pub condition: Tensor<B,2>, pub timestep_embedding: Tensor<B,2>,
    pub text_mask: Option<Tensor<B,4>>, pub spatial: Option<(usize,usize)>,
}

fn take_autodiff_primitive<B: Backend, C: CheckpointStrategy, const D: usize>(
    tensor: Tensor<Autodiff<B, C>, D>,
) -> AutodiffTensor<B> {
    match tensor.into_primitive() {
        TensorPrimitive::Float(value) => value,
        TensorPrimitive::QFloat(_) => panic!("quantized input unsupported"),
    }
}

struct AnimaDiTBlockInputSnapshot<B: Backend> {
    input: (NodeRef,B::FloatTensorPrimitive), text: (NodeRef,B::FloatTensorPrimitive), condition: (NodeRef,B::FloatTensorPrimitive),
    timestep_embedding: (NodeRef,B::FloatTensorPrimitive), text_mask: Option<(NodeRef,B::FloatTensorPrimitive)>, spatial: Option<(usize,usize)>,
}

impl<B: Backend, C: CheckpointStrategy> AnimaDiTBlockInputs<Autodiff<B,C>> {
    fn split(self) -> (AnimaDiTBlockInputs<B>, AnimaDiTBlockInputSnapshot<B>, Vec<NodeRef>) {
        let i=take_autodiff_primitive(self.input); let t=take_autodiff_primitive(self.text); let c=take_autodiff_primitive(self.condition); let e=take_autodiff_primitive(self.timestep_embedding);
        let mut p=vec![i.node.clone(),t.node.clone(),c.node.clone(),e.node.clone()];
        let (mask,mask_snap)=self.text_mask.map(|m| { let m=take_autodiff_primitive(m); p.push(m.node.clone()); (Some(Tensor::from_primitive(TensorPrimitive::Float(m.primitive.clone()))),Some((m.node,m.primitive))) }).unwrap_or((None,None));
        let inner=AnimaDiTBlockInputs { input:Tensor::from_primitive(TensorPrimitive::Float(i.primitive.clone())), text:Tensor::from_primitive(TensorPrimitive::Float(t.primitive.clone())), condition:Tensor::from_primitive(TensorPrimitive::Float(c.primitive.clone())), timestep_embedding:Tensor::from_primitive(TensorPrimitive::Float(e.primitive.clone())), text_mask:mask, spatial:self.spatial };
        let snap=AnimaDiTBlockInputSnapshot { input:(i.node,i.primitive), text:(t.node,t.primitive), condition:(c.node,c.primitive), timestep_embedding:(e.node,e.primitive), text_mask:mask_snap, spatial:self.spatial };
        (inner,snap,p)
    }
}

impl<B: Backend> AnimaDiTBlockInputSnapshot<B> {
    fn into_local(self, b:&mut Vec<GradBridge<B>>) -> AnimaDiTBlockInputs<LocalB<B>> {
        let mut leaf=|(n,p):(NodeRef,B::FloatTensorPrimitive)| { let mut v=AutodiffTensor::new(p); if !matches!(n.requirement,Requirement::None) { v=v.require_grad(); b.push(GradBridge{outer_node_id:n.id,local_tensor:v.clone()}); } v };
        AnimaDiTBlockInputs { input:Tensor::from_primitive(TensorPrimitive::Float(leaf(self.input))), text:Tensor::from_primitive(TensorPrimitive::Float(leaf(self.text))), condition:Tensor::from_primitive(TensorPrimitive::Float(leaf(self.condition))), timestep_embedding:Tensor::from_primitive(TensorPrimitive::Float(leaf(self.timestep_embedding))), text_mask:self.text_mask.map(|v|Tensor::from_primitive(TensorPrimitive::Float(leaf(v)))), spatial:self.spatial }
    }
}

struct AnimaDiTBlockSnapshot<B: Backend, S> { self_attn:S, cross_attn:S, mlp_layer1:LoraLinearSnap<B>, mlp_layer2:LoraLinearSnap<B>, self_attn_modulation:AdaLnModulationSnap<B>, cross_attn_modulation:AdaLnModulationSnap<B>, mlp_modulation:AdaLnModulationSnap<B> }

impl<B,C,A> AnimaDiTBlock<Autodiff<B,C>,A>
where B: Backend + BlockCheckpointBackend, C: CheckpointStrategy,
    A: AnimaAttnCheckpoint<B,C> + Module<Autodiff<B,C>>,
    A::InnerAttn: AnimaAttentionLayer<B>, A::LocalAttn: AnimaAttentionLayer<LocalB<B>>,
{
    fn checkpoint_inner(&self) -> AnimaDiTBlock<B,A::InnerAttn> { AnimaDiTBlock { self_attn:self.self_attn.inner(), cross_attn:self.cross_attn.inner(), mlp_layer1:LoraLinearSnap::inner(&self.mlp_layer1), mlp_layer2:LoraLinearSnap::inner(&self.mlp_layer2), self_attn_modulation:AdaLnModulationSnap::inner(&self.self_attn_modulation), cross_attn_modulation:AdaLnModulationSnap::inner(&self.cross_attn_modulation), mlp_modulation:AdaLnModulationSnap::inner(&self.mlp_modulation) } }
    fn capture_checkpoint(&self,p:&mut Vec<NodeRef>)->AnimaDiTBlockSnapshot<B,A::Snapshot>{ AnimaDiTBlockSnapshot { self_attn:self.self_attn.capture(p), cross_attn:self.cross_attn.capture(p), mlp_layer1:LoraLinearSnap::capture(&self.mlp_layer1,p), mlp_layer2:LoraLinearSnap::capture(&self.mlp_layer2,p), self_attn_modulation:AdaLnModulationSnap::capture(&self.self_attn_modulation,p), cross_attn_modulation:AdaLnModulationSnap::capture(&self.cross_attn_modulation,p), mlp_modulation:AdaLnModulationSnap::capture(&self.mlp_modulation,p) } }
    pub(crate) fn forward_checkpointed(&self, inputs:AnimaDiTBlockInputs<Autodiff<B,C>>)->Tensor<Autodiff<B,C>,3>{
        let (inner_in,input_snap,mut parents)=inputs.split(); let snap=self.capture_checkpoint(&mut parents); let block=self.checkpoint_inner();
        let raw=block.forward_with_text_mask_and_spatial(inner_in.input,inner_in.text,inner_in.condition,inner_in.timestep_embedding,inner_in.text_mask,inner_in.spatial).into_primitive();
        let raw=match raw { TensorPrimitive::Float(v)=>v, TensorPrimitive::QFloat(_)=>panic!("quantized output unsupported") };
        let out=register_dynamic_checkpoint::<B,3>(raw,&parents,Box::new(move||{ let mut bridges=Vec::new(); let i=input_snap.into_local(&mut bridges); let block=AnimaDiTBlock::<LocalB<B>,A::LocalAttn>{ self_attn:A::restore(snap.self_attn,&mut bridges), cross_attn:A::restore(snap.cross_attn,&mut bridges), mlp_layer1:snap.mlp_layer1.into_local(&mut bridges), mlp_layer2:snap.mlp_layer2.into_local(&mut bridges), self_attn_modulation:snap.self_attn_modulation.into_local(&mut bridges), cross_attn_modulation:snap.cross_attn_modulation.into_local(&mut bridges), mlp_modulation:snap.mlp_modulation.into_local(&mut bridges) }; (block.forward_with_text_mask_and_spatial(i.input,i.text,i.condition,i.timestep_embedding,i.text_mask,i.spatial),bridges) }));
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }

    pub(crate) fn forward_segment_checkpointed(
        blocks: &[&Self],
        inputs: AnimaDiTBlockInputs<Autodiff<B, C>>,
    ) -> Tensor<Autodiff<B, C>, 3> {
        assert!(!blocks.is_empty());
        let (inner_in, input_snap, mut parents) = inputs.split();
        let snapshots = blocks
            .iter()
            .map(|block| block.capture_checkpoint(&mut parents))
            .collect::<Vec<_>>();

        let mut hidden = inner_in.input;
        for block in blocks {
            let inner = block.checkpoint_inner();
            hidden = inner.forward_with_text_mask_and_spatial(
                hidden,
                inner_in.text.clone(),
                inner_in.condition.clone(),
                inner_in.timestep_embedding.clone(),
                inner_in.text_mask.clone(),
                inner_in.spatial,
            );
        }
        let raw = match hidden.into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized output unsupported"),
        };

        let out = register_dynamic_checkpoint::<B, 3>(
            raw,
            &parents,
            Box::new(move || {
                let mut bridges = Vec::new();
                let local_input = input_snap.into_local(&mut bridges);
                let mut hidden = local_input.input;
                for snapshot in snapshots {
                    let block = AnimaDiTBlock::<LocalB<B>, A::LocalAttn> {
                        self_attn: A::restore(snapshot.self_attn, &mut bridges),
                        cross_attn: A::restore(snapshot.cross_attn, &mut bridges),
                        mlp_layer1: snapshot.mlp_layer1.into_local(&mut bridges),
                        mlp_layer2: snapshot.mlp_layer2.into_local(&mut bridges),
                        self_attn_modulation: snapshot.self_attn_modulation.into_local(&mut bridges),
                        cross_attn_modulation: snapshot.cross_attn_modulation.into_local(&mut bridges),
                        mlp_modulation: snapshot.mlp_modulation.into_local(&mut bridges),
                    };
                    hidden = block.forward_with_text_mask_and_spatial(
                        hidden,
                        local_input.text.clone(),
                        local_input.condition.clone(),
                        local_input.timestep_embedding.clone(),
                        local_input.text_mask.clone(),
                        local_input.spatial,
                    );
                }
                (hidden, bridges)
            }),
        );
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}

/// Keep checkpointing at the same granularity as Python's Anima trainer.
///
/// Python wraps the complete DiT block with `torch.utils.checkpoint`; it does
/// not checkpoint individual residuals, projections, attention inputs, or MLP
/// intermediates.  In particular, do not reintroduce an operator-level
/// recomputation switch here: it creates a different graph and can turn one
/// block replay into thousands of nested Burn autodiff replays.
fn checkpoint_internal<B: BlockCheckpointBackend>(tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    tensor
}

/// Mirrors Anima's Python `use_fp32` switch.  The reference model keeps the
/// residual/AdaLN path in FP32 when the autocast activation dtype is FP16;
/// this is a model-level rule, not an incidental cast at one operator.
fn python_fp16_residual_path<B: Backend>(tensor: &Tensor<B, 3>) -> bool {
    std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some()
        && tensor.dtype() == FloatDType::F16.into()
}

fn python_fp16_boundary<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> bool {
    std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some()
        && matches!(tensor.dtype(), burn::tensor::DType::F16 | burn::tensor::DType::F32)
}

fn dump_forward_tensor<B: Backend, const D: usize>(step: usize, name: &str, tensor: &Tensor<B, D>) {
    let Some(root) = std::env::var_os("AFTERBURNER_TRAIN_DUMP_DIR") else {
        return;
    };
    let max_steps = std::env::var("AFTERBURNER_TRAIN_DUMP_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    if step == 0 || step > max_steps {
        return;
    }
    let directory = std::path::PathBuf::from(root).join(format!("step-{step:06}"));
    if std::fs::create_dir_all(&directory).is_err() {
        return;
    }
    // Checkpointing recomputes forward nodes during backward.  When this
    // diagnostic switch is enabled, preserve the first forward value instead
    // of silently replacing it with a recomputation snapshot.
    let first_forward_only = std::env::var_os("AFTERBURNER_TRAIN_DUMP_FIRST_FORWARD").is_some();
    let data_path = directory.join(format!("{name}.f32"));
    let shape_path = directory.join(format!("{name}.shape.json"));
    if first_forward_only && data_path.exists() {
        return;
    }
    let data = tensor.to_data();
    let Ok(values) = data.to_vec::<B::FloatElem>() else {
        return;
    };
    let values = values.into_iter().map(|value| value.elem::<f32>()).collect::<Vec<_>>();
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let _ = std::fs::write(data_path, bytes);
    let _ = std::fs::write(
        shape_path,
        format!("{:?}", data.shape.as_slice()).into_bytes(),
    );
}

impl BlockCheckpointBackend for burn_ndarray::NdArray<f32> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

#[cfg(feature = "cuda")]
impl BlockCheckpointBackend for burn_cuda::Cuda<f32> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

#[cfg(feature = "cuda")]
impl BlockCheckpointBackend for burn_cuda::Cuda<half::bf16> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

#[cfg(feature = "cuda")]
impl BlockCheckpointBackend for burn_cuda::Cuda<half::f16> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

#[cfg(feature = "vulkan")]
impl BlockCheckpointBackend for burn::backend::Vulkan<f32> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

#[cfg(feature = "vulkan")]
impl BlockCheckpointBackend for burn::backend::Vulkan<half::bf16> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

#[cfg(feature = "vulkan")]
impl BlockCheckpointBackend for burn::backend::Vulkan<half::f16> {
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tensor
    }
}

impl<Inner, C> BlockCheckpointBackend for burn_autodiff::Autodiff<Inner, C>
where
    Inner: Backend + BlockCheckpointBackend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn checkpoint_block(tensor: Tensor<Self, 3>) -> Tensor<Self, 3> {
        let primitive = match tensor.into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized checkpoint tensors are unsupported"),
        };
        let primitive = burn_autodiff::ops::checkpoint_boundary::<Inner, C>(primitive);
        Tensor::from_primitive(TensorPrimitive::Float(primitive))
    }
}

pub(crate) fn attention_query_chunk_size() -> usize {
    ATTENTION_CHUNK_SIZE.load(Ordering::Relaxed)
}

/// Sets the query chunk size used by all Anima attention modules.
///
/// The CLI configures this before constructing the model. Keeping the value
/// in the engine, rather than reading an environment variable inside each
/// attention call, makes the runtime setting explicit and reproducible.
pub fn set_attention_chunk_size(size: usize) -> Result<(), AnimaDiTConfigError> {
    if size == 0 {
        return Err(AnimaDiTConfigError::Invalid(
            "attention chunk size must be greater than zero".to_owned(),
        ));
    }
    ATTENTION_CHUNK_SIZE.store(size, Ordering::Relaxed);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimaDiTConfigError {
    Invalid(String),
    Lora(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimaTensorError {
    InvalidRank { expected: usize, actual: usize },
    InvalidShape { message: String, actual: Vec<usize> },
}

impl std::fmt::Display for AnimaTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRank { expected, actual } => {
                write!(f, "expected rank {expected} tensor, got rank {actual}")
            }
            Self::InvalidShape { message, actual } => write!(f, "{message}: {actual:?}"),
        }
    }
}

impl std::error::Error for AnimaTensorError {}

/// Converts a VAE latent `[B, C, T, H, W]` into Anima image tokens.
///
/// Anima uses a spatial 2x2 patch and one additional mask channel, yielding
/// `17 * 2 * 2 = 68` input features for its default 16-channel latent.
pub fn patchify_latent<B: Backend>(
    latent: Tensor<B, 5>,
    padding_mask: Option<Tensor<B, 4>>,
) -> Result<Tensor<B, 3>, AnimaTensorError> {
    let [batch, channels, frames, height, width] = latent.dims();
    if frames != 1 || height % 2 != 0 || width % 2 != 0 {
        return Err(AnimaTensorError::InvalidShape {
            message: "latent must have one frame and even spatial dimensions".into(),
            actual: vec![batch, channels, frames, height, width],
        });
    }
    let device = latent.device();
    let mask = match padding_mask {
        Some(mask) => {
            let dims = mask.dims();
            if dims != [batch, 1, height, width] {
                return Err(AnimaTensorError::InvalidShape {
                    message: "padding mask must be [B, 1, H, W]".into(),
                    actual: dims.to_vec(),
                });
            }
            mask
        }
        None => Tensor::zeros([batch, 1, height, width], &device),
    };
    let latent = latent.reshape([batch, channels, height, width]);
    let input = Tensor::cat(vec![latent, mask], 1);
    let input_channels = channels + 1;
    Ok(input
        .reshape([batch, input_channels, height / 2, 2, width / 2, 2])
        .swap_dims(1, 2)
        .swap_dims(2, 4)
        .swap_dims(3, 4)
        .reshape([batch, (height / 2) * (width / 2), input_channels * 4]))
}

/// Converts Anima output tokens back into a latent `[B, C, 1, H, W]` tensor.
pub fn unpatchify_latent<B: Backend>(
    tokens: Tensor<B, 3>,
    output_channels: usize,
    height: usize,
    width: usize,
) -> Result<Tensor<B, 5>, AnimaTensorError> {
    let [batch, token_count, feature_count] = tokens.dims();
    if height % 2 != 0 || width % 2 != 0 || token_count != height / 2 * (width / 2) {
        return Err(AnimaTensorError::InvalidShape {
            message: "token count does not match the requested even spatial shape".into(),
            actual: vec![batch, token_count, feature_count],
        });
    }
    if feature_count != output_channels * 4 {
        return Err(AnimaTensorError::InvalidShape {
            message: "output feature count must equal channels * 2 * 2".into(),
            actual: vec![batch, token_count, feature_count],
        });
    }
    // Python's `rearrange` emits output features in `[p_h, p_w, t, C]`
    // order.  For the image case (t=1), reshape that as
    // `[H2, W2, p_h, p_w, C]` and permute to `[C, H2, p_h, W2, p_w]`.
    // The previous implementation treated the feature order as
    // `[C, p_h, p_w]`, scrambling channels and spatial positions in every
    // predicted latent while preserving the overall shape and magnitude.
    Ok(tokens
        .reshape([batch, height / 2, width / 2, 2, 2, output_channels])
        .swap_dims(1, 5)
        .swap_dims(2, 5)
        .swap_dims(4, 5)
        .reshape([batch, output_channels, 1, height, width]))
}

impl std::fmt::Display for AnimaDiTConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid Anima DiT configuration: {message}"),
            Self::Lora(message) => write!(f, "invalid Anima DiT LoRA configuration: {message}"),
        }
    }
}

impl std::error::Error for AnimaDiTConfigError {}

/// Dimensions required by one Anima DiT block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimaDiTBlockConfig {
    pub hidden_size: usize,
    pub context_size: usize,
    pub attention_head_count: usize,
    pub attention_head_size: usize,
    pub mlp_hidden_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: usize,
}

impl AnimaDiTBlockConfig {
    pub fn validate(&self) -> Result<(), AnimaDiTConfigError> {
        if self.hidden_size != self.attention_head_count * self.attention_head_size {
            return Err(AnimaDiTConfigError::Invalid(
                "hidden size must equal head count multiplied by head size".into(),
            ));
        }
        if self.context_size == 0 || self.mlp_hidden_size == 0 {
            return Err(AnimaDiTConfigError::Invalid(
                "context and MLP dimensions must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    fn linear(
        &self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<LoraLinearConfig, AnimaDiTConfigError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(AnimaDiTConfigError::Invalid(
                "linear dimensions must be greater than zero".into(),
            ));
        }
        Ok(LoraLinearConfig {
            input_dim,
            output_dim,
            rank: self.lora_rank,
            alpha: self.lora_alpha,
            bias: false,
        })
    }
}

impl From<&AnimaConfig> for AnimaDiTBlockConfig {
    fn from(config: &AnimaConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            context_size: config.text_hidden_size,
            attention_head_count: config.attention_head_count,
            attention_head_size: config.attention_head_size,
            mlp_hidden_size: config.mlp_hidden_size,
            lora_rank: 16,
            lora_alpha: 16,
        }
    }
}

#[derive(Module, Debug)]
pub struct AdaLnModulation<B: Backend> {
    pub down: LoraLinear<B>,
    pub up: LoraLinear<B>,
    pub hidden_size: usize,
}

#[derive(Module, Debug)]
pub struct AnimaFinalModulation<B: Backend> {
    pub down: LoraLinear<B>,
    pub up: LoraLinear<B>,
    pub hidden_size: usize,
}

impl<B: Backend> AnimaFinalModulation<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            down: config
                .linear(config.hidden_size, 256)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            up: config
                .linear(256, config.hidden_size * 2)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.forward_with_linear(input, condition, timestep_embedding, |layer, input| {
            layer.forward(input)
        })
    }

    fn forward_with_linear<F>(
        &self,
        input: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
        linear: F,
    ) -> Tensor<B, 3>
    where
        F: Fn(&LoraLinear<B>, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let [batch, _, _] = input.dims();
        let params = linear(&self.up, silu(linear(&self.down, condition)))
            .add(timestep_embedding.slice([0..batch, 0..self.hidden_size * 2]));
        let shift = params.clone().slice([0..batch, 0..self.hidden_size]);
        let scale = params.slice([0..batch, self.hidden_size..self.hidden_size * 2]);
        let (variance, mean) = input.clone().var_mean_bias(2);
        input
            .sub(mean)
            .div(variance.add_scalar(1e-6).sqrt())
            .mul(scale.add_scalar(1.0).unsqueeze())
            .add(shift.unsqueeze())
    }
}

impl<B: Backend> AdaLnModulation<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            down: config
                .linear(config.hidden_size, 256)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            up: config
                .linear(256, config.hidden_size * 3)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&self, condition: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_with_linear(condition, |layer, input| layer.forward(input))
    }

    fn forward_with_linear<F>(&self, condition: Tensor<B, 2>, linear: F) -> Tensor<B, 2>
    where
        F: Fn(&LoraLinear<B>, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        linear(&self.up, silu(linear(&self.down, condition)))
    }

    fn apply(
        &self,
        input: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.apply_with_linear(input, condition, timestep_embedding, |layer, input| {
            layer.forward(input)
        })
    }

    fn apply_with_linear<F>(
        &self,
        input: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
        linear: F,
    ) -> (Tensor<B, 3>, Tensor<B, 3>)
    where
        F: Fn(&LoraLinear<B>, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let [batch, _, _] = input.dims();
        let params = self
            .forward_with_linear(condition, linear)
            .add(timestep_embedding);
        let shift = params.clone().slice([0..batch, 0..self.hidden_size]);
        let scale = params
            .clone()
            .slice([0..batch, self.hidden_size..self.hidden_size * 2]);
        let gate = params.slice([0..batch, self.hidden_size * 2..self.hidden_size * 3]);
        let (variance, mean) = input.clone().var_mean_bias(2);
        let normalized = input
            .sub(mean)
            // Python Cosmos/Anima uses LayerNorm(eps=1e-6) for every
            // block AdaLN normalization.
            .div(variance.add_scalar(1e-6).sqrt())
            .mul(scale.add_scalar(1.0).unsqueeze())
            .add(shift.unsqueeze());
        (normalized, gate.unsqueeze())
    }
}

/// Reference Burn Tensor implementation of Anima attention.
///
/// This remains the portable path for ndarray and Vulkan. CUDA training will
/// use a separate implementation once the fused Autodiff operation is wired
/// in, so the portable implementation no longer defines the CUDA memory
/// behavior by accident.
#[derive(Module, Debug)]
pub struct AnimaAttentionStandard<B: Backend> {
    pub q_proj: LoraLinear<B>,
    pub k_proj: LoraLinear<B>,
    pub v_proj: LoraLinear<B>,
    pub output_proj: LoraLinear<B>,
    pub q_norm: AnimaRmsNorm<B>,
    pub k_norm: AnimaRmsNorm<B>,
    pub use_rotary: bool,
    pub head_count: usize,
    pub head_size: usize,
}

/// Attention implementation contract used by Anima blocks.
///
/// The default implementation is the portable Burn path. CUDA training can
/// substitute `AnimaAttentionCuda` without changing the surrounding block
/// layout or weight-loader contracts.
pub trait AnimaAttentionLayer<B: Backend>: Module<B> + ModuleDisplay + core::fmt::Debug {
    fn standard(&self) -> &AnimaAttentionStandard<B>;
    fn standard_mut(&mut self) -> &mut AnimaAttentionStandard<B>;

    /// Applies a LoRA linear layer using the backend-specific training path.
    /// CUDA autodiff overrides this with the fused primitive so long sequence
    /// projections do not build three independent MatMul graph branches.
    fn forward_lora_linear(&self, layer: &LoraLinear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        layer.forward(input)
    }

    /// Recomputable sequence projection used for Anima's very wide MLP.
    /// Keeping these intermediates for all 28 blocks defeats block
    /// checkpointing even when attention is chunked.
    fn forward_lora_linear_recomputable(
        &self,
        layer: &LoraLinear<B>,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        layer.forward_sequence_chunked(input)
    }

    fn forward_lora_linear_2d(&self, layer: &LoraLinear<B>, input: Tensor<B, 2>) -> Tensor<B, 2> {
        layer.forward(input)
    }

    fn forward_block(
        block: &AnimaDiTBlock<B, Self>, input: Tensor<B, 3>, text: Tensor<B, 3>,
        condition: Tensor<B, 2>, timestep_embedding: Tensor<B, 2>, text_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3>
    where
        Self: Sized,
        B: BlockCheckpointBackend,
    {
        block.forward_with_text_mask_and_spatial(input, text, condition, timestep_embedding, text_mask, spatial)
    }

    /// Forward a checkpoint segment. CUDA autodiff overrides this so several
    /// adjacent DiT blocks share one local recomputation/backward graph.
    fn forward_block_segment(
        blocks: &[&AnimaDiTBlock<B, Self>], input: Tensor<B, 3>, text: Tensor<B, 3>,
        condition: Tensor<B, 2>, timestep_embedding: Tensor<B, 2>, text_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3>
    where
        Self: Sized,
        B: BlockCheckpointBackend,
    {
        let mut hidden = input;
        for block in blocks {
            hidden = Self::forward_block(
                block, hidden, text.clone(), condition.clone(), timestep_embedding.clone(),
                text_mask.clone(), spatial,
            );
        }
        hidden
    }

    fn init_self(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError>
    where
        Self: Sized;

    fn init_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError>
    where
        Self: Sized;

    fn init_adapter_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError>
    where
        Self: Sized;

    fn forward_with_mask(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3>;

    fn forward_with_mask_and_spatial(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3>;
}

impl<B: Backend> AnimaAttentionStandard<B> {
    pub(crate) fn init_self(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init(config, config.hidden_size, true, device)
    }

    pub(crate) fn init_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init(config, config.context_size, false, device)
    }

    /// Initializes the LLM-adapter cross attention.
    ///
    /// The adapter uses the same one-dimensional split-half RoPE for both
    /// query and context positions.  This differs from the DiT cross
    /// attention, which intentionally has no rotary embedding.
    pub(crate) fn init_adapter_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init(config, config.context_size, true, device)
    }

    pub(crate) fn init(
        config: &AnimaDiTBlockConfig,
        key_value_size: usize,
        use_rotary: bool,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            q_proj: config
                .linear(config.hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            k_proj: config
                .linear(key_value_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            v_proj: config
                .linear(key_value_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            output_proj: config
                .linear(config.hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            q_norm: AnimaRmsNorm::init(config.attention_head_size, device),
            k_norm: AnimaRmsNorm::init(config.attention_head_size, device),
            use_rotary,
            head_count: config.attention_head_count,
            head_size: config.attention_head_size,
        })
    }

    pub fn forward(&self, input: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_mask(input, context, None)
    }

    pub fn forward_with_mask(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        self.forward_with_mask_and_spatial(input, context, context_mask, None)
    }

    fn forward_with_mask_and_spatial(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3> {
        let [batch, query_length, _] = input.dims();
        let [_, context_length, _] = context.dims();
        let input_dtype = input.dtype();
        let mixed_precision = std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some();
        // Lifetime: forward-only projection result. Kept until all query tiles
        // finish; then it is step-local reusable workspace.
        let q = (if mixed_precision {
            self.q_proj.forward_low_precision(input.clone())
        } else {
            self.q_proj.forward_sequence_chunked(input.clone())
        })
            .reshape([batch, query_length, self.head_count, self.head_size])
            .swap_dims(1, 2);
        // Lifetime: forward-only projection result. Kept for all query tiles;
        // it must not become a persistent allocator page across steps.
        let k = (if mixed_precision {
            self.k_proj.forward_low_precision(context.clone())
        } else {
            self.k_proj.forward_sequence_chunked(context.clone())
        })
            .reshape([batch, context_length, self.head_count, self.head_size])
            .swap_dims(1, 2);
        // Lifetime: forward-only projection result. Kept for all key tiles;
        // it must not become a persistent allocator page across steps.
        let v = (if mixed_precision {
            self.v_proj.forward_low_precision(context)
        } else {
            self.v_proj.forward_sequence_chunked(context)
        })
            .reshape([batch, context_length, self.head_count, self.head_size])
            .swap_dims(1, 2);
        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);
        let (q, k) = if self.use_rotary {
            let device = q.device();
            match spatial {
                Some((height, width)) => {
                    apply_rotary_3d_pair(q, k, height, width, self.head_size, &device)
                }
                None => apply_rotary_pair(q, k, query_length, context_length, self.head_size, &device),
            }
        } else {
            (q, k)
        };
        let scale = 1.0 / (self.head_size as f32).sqrt();
        let mut attended_chunks = Vec::new();
        let query_chunk_size = attention_query_chunk_size();
        for query_start in (0..query_length).step_by(query_chunk_size) {
            let query_end = (query_start + query_chunk_size).min(query_length);
            let query = q.clone().slice([
                0..batch,
                0..self.head_count,
                query_start..query_end,
                0..self.head_size,
            ]);
            // Match PyTorch SDPA's memory behavior by tiling the key/value
            // sequence as well as the query sequence. Use wider key tiles
            // than query tiles: this keeps the score workspace bounded while
            // reducing the number of Autodiff matmul nodes. Online softmax
            // keeps only one [query_chunk, key_chunk] score tile and updates
            // the row maximum, denominator, and weighted value sum in one pass.
            // This avoids the full [query_chunk, key_length] attention matrix
            // without paying for a second QK matmul over every key tile.
            let key_chunk_size = query_chunk_size
                .saturating_mul(4)
                .min(context_length)
                .max(1);
            // Lifetime: one query tile. These accumulators are intended to be
            // reused as step-local attention workspace.
            let mut row_max: Option<Tensor<B, 4>> = None;
            let mut denominator: Option<Tensor<B, 4>> = None;
            let mut numerator: Option<Tensor<B, 4>> = None;
            for key_start in (0..context_length).step_by(key_chunk_size) {
                let key_end = (key_start + key_chunk_size).min(context_length);
                let key = k
                    .clone()
                    .slice([
                        0..batch,
                        0..self.head_count,
                        key_start..key_end,
                        0..self.head_size,
                    ])
                    .swap_dims(2, 3);
                let value = v.clone().slice([
                    0..batch,
                    0..self.head_count,
                    key_start..key_end,
                    0..self.head_size,
                ]);
                // Lifetime: one Q/K tile; dead after local softmax accumulation.
                let mut scores = query.clone().matmul(key).mul_scalar(scale);
                if let Some(mask) = context_mask.clone() {
                    let mask_dims = mask.dims();
                    scores = scores.add(mask.slice([
                        0..batch,
                        0..mask_dims[1],
                        0..mask_dims[2],
                        key_start..key_end,
                    ]));
                }
                let local_max = scores.clone().max_dim(3);
                let local_weights = scores.clone().sub(local_max.clone()).exp();
                let local_denominator = local_weights.clone().sum_dim(3);
                let local_numerator = local_weights.matmul(value);
                match (row_max, denominator, numerator) {
                    (Some(previous_max), Some(previous_denominator), Some(previous_numerator)) => {
                        let next_max = previous_max.clone().max_pair(local_max.clone());
                        let previous_scale = previous_max.sub(next_max.clone()).exp();
                        let local_scale = local_max.sub(next_max.clone()).exp();
                        row_max = Some(next_max);
                        denominator = Some(
                            previous_denominator
                                .mul(previous_scale.clone())
                                .add(local_denominator.mul(local_scale.clone())),
                        );
                        numerator = Some(
                            previous_numerator
                                .mul(previous_scale)
                                .add(local_numerator.mul(local_scale)),
                        );
                    }
                    (None, None, None) => {
                        row_max = Some(local_max);
                        denominator = Some(local_denominator);
                        numerator = Some(local_numerator);
                    }
                    _ => unreachable!("attention online-softmax state must advance together"),
                }
            }
            let attended = numerator
                .expect("attention context must contain at least one token")
                .div(denominator.expect("attention context must contain at least one token"));
            attended_chunks.push(attended);
        }
        // Lifetime: attention-call output. Concatenation is a materializing
        // boundary; later optimization can replace it with a preallocated
        // output workspace when Burn exposes write/slice operations.
        let attended = Tensor::cat(attended_chunks, 2).swap_dims(1, 2).reshape([
            batch,
            query_length,
            self.head_count * self.head_size,
        ]);
        let output = if mixed_precision {
            self.output_proj.forward_low_precision(attended)
        } else {
            self.output_proj.forward(attended)
        };
        if mixed_precision {
            output.cast(input_dtype)
        } else {
            output
        }
    }
}

impl<B: Backend> AnimaAttentionLayer<B> for AnimaAttentionStandard<B> {
    fn standard(&self) -> &AnimaAttentionStandard<B> {
        self
    }

    fn standard_mut(&mut self) -> &mut AnimaAttentionStandard<B> {
        self
    }

    fn init_self(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init_self(config, device)
    }

    fn init_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init_cross(config, device)
    }

    fn init_adapter_cross(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Self::init_adapter_cross(config, device)
    }

    fn forward_with_mask(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        AnimaAttentionStandard::forward_with_mask(self, input, context, context_mask)
    }

    fn forward_with_mask_and_spatial(
        &self,
        input: Tensor<B, 3>,
        context: Tensor<B, 3>,
        context_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3> {
        self.forward_with_mask_and_spatial(input, context, context_mask, spatial)
    }
}

/// Compatibility alias while model containers are migrated to an explicit
/// attention implementation type. New CUDA code must not add behavior here;
/// it belongs in `AnimaAttentionCuda`.
pub type AnimaAttention<B> = AnimaAttentionStandard<B>;

/// CUDA attention implementation boundary.
///
/// The wrapper deliberately keeps the projection and normalization modules in
/// the same layout as the standard implementation so weight loading and LoRA
/// target selection remain identical. The fused Autodiff operation handles
/// both the additive cache mask and the optional image-token RoPE coordinates.
#[cfg(feature = "cuda")]
#[derive(Module, Debug)]
pub struct AnimaAttentionCuda<B: Backend> {
    pub standard: AnimaAttentionStandard<B>,
}

#[cfg(feature = "cuda")]
impl<Inner, C> AnimaAttentionCuda<burn_autodiff::Autodiff<Inner, C>>
where
    Inner: Backend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    pub fn from_standard(
        standard: AnimaAttentionStandard<burn_autodiff::Autodiff<Inner, C>>,
    ) -> Self {
        Self { standard }
    }

    pub fn forward(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        self.forward_fused(input, context, None, None)
    }

    pub fn forward_with_mask(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        self.forward_with_mask_and_spatial(input, context, context_mask, None)
    }

    /// API-compatible entry point used by DiT blocks.
    ///
    /// The fused primitive also receives the spatial RoPE coordinates used by
    /// the image-token self-attention path.
    pub fn forward_with_mask_and_spatial(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        // Keep the short LLM-adapter attention on the fused path.  The
        // reference trainer only needs the memory-safe tiled implementation
        // for the long image-token sequence; expanding the adapter into a
        // per-op Autodiff graph makes the F32 residual path needlessly huge.
        let [_, query_length, _] = input.dims();
        let use_standard = std::env::var_os("AFTERBURNER_STANDARD_ATTENTION").is_some()
            && (spatial.is_some() || query_length > 1024);
        if use_standard {
            return self
                .standard
                .forward_with_mask_and_spatial(input, context, context_mask, spatial);
        }
        self.forward_fused(input, context, context_mask, spatial)
    }

    fn forward_fused(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        let trace = std::env::var_os("AFTERBURNER_TRACE").is_some();
        let mixed_precision = std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some();
        let input_dtype = input.dtype();
        let [batch, query_length, _] = input.dims();
        let [_, context_length, _] = context.dims();
        // The custom primitive performs the query chunking internally while
        // keeping the whole attention operation as one Autodiff node. Do not
        // fall back to the standard per-chunk graph here: that path retains
        // every intermediate chunk until backward and recreates the VRAM
        // explosion seen with 1024px samples.
        let chunk_size = attention_query_chunk_size();
        if trace { eprintln!("[progress] phase=attention_q_projection_begin"); }
        // Lifetime: attention-call projection workspace. Reusable after this
        // forward call; never intentionally persistent across training steps.
        let recompute_linear = std::env::var_os("AFTERBURNER_RECOMPUTE_ALL_LINEAR").is_some();
        let q = if mixed_precision && !recompute_linear {
            self.standard.q_proj.forward_low_precision(input)
        } else {
            self.standard
                .q_proj
                .forward_fused_recomputable_sequence_chunked(input)
        }
            .reshape([
                batch,
                query_length,
                self.standard.head_count,
                self.standard.head_size,
            ])
            .swap_dims(1, 2);
        if trace { eprintln!("[progress] phase=attention_q_projection_complete"); }
        if trace { eprintln!("[progress] phase=attention_k_projection_begin"); }
        // Lifetime: attention-call projection workspace. Reusable after this
        // forward call; never intentionally persistent across training steps.
        let k = if mixed_precision && !recompute_linear {
            self.standard.k_proj.forward_low_precision(context.clone())
        } else {
            self.standard
                .k_proj
                .forward_fused_recomputable_sequence_chunked(context.clone())
        }
            .reshape([
                batch,
                context_length,
                self.standard.head_count,
                self.standard.head_size,
            ])
            .swap_dims(1, 2);
        if trace { eprintln!("[progress] phase=attention_k_projection_complete"); }
        if trace { eprintln!("[progress] phase=attention_v_projection_begin"); }
        // Lifetime: attention-call projection workspace. Reusable after this
        // forward call; never intentionally persistent across training steps.
        let v = if mixed_precision && !recompute_linear {
            self.standard.v_proj.forward_low_precision(context)
        } else {
            self.standard
                .v_proj
                .forward_fused_recomputable_sequence_chunked(context)
        }
            .reshape([
                batch,
                context_length,
                self.standard.head_count,
                self.standard.head_size,
            ])
            .swap_dims(1, 2);
        if trace { eprintln!("[progress] phase=attention_v_projection_complete"); }
        if trace { eprintln!("[progress] phase=attention_qk_norm_begin"); }
        let q = burn_autodiff::ops::rms_norm_manual::<Inner>(
            match q.into_primitive() {
                TensorPrimitive::Float(value) => value,
                TensorPrimitive::QFloat(_) => panic!("quantized q norm input is unsupported"),
            },
            match self.standard.q_norm.gamma.val().detach().into_primitive() {
                TensorPrimitive::Float(value) => value.primitive,
                TensorPrimitive::QFloat(_) => panic!("quantized q norm gamma is unsupported"),
            },
            self.standard.q_norm.epsilon,
        );
        let q = Tensor::<burn_autodiff::Autodiff<Inner, C>, 4>::from_primitive(
            TensorPrimitive::Float(q),
        );
        let k = burn_autodiff::ops::rms_norm_manual::<Inner>(
            match k.into_primitive() {
                TensorPrimitive::Float(value) => value,
                TensorPrimitive::QFloat(_) => panic!("quantized k norm input is unsupported"),
            },
            match self.standard.k_norm.gamma.val().detach().into_primitive() {
                TensorPrimitive::Float(value) => value.primitive,
                TensorPrimitive::QFloat(_) => panic!("quantized k norm gamma is unsupported"),
            },
            self.standard.k_norm.epsilon,
        );
        let k = Tensor::<burn_autodiff::Autodiff<Inner, C>, 4>::from_primitive(
            TensorPrimitive::Float(k),
        );
        let dump_attention = std::env::var_os("AFTERBURNER_TRAIN_DUMP_DIR").is_some()
            && query_length >= 3000;
        let attention_dump_prefix = if dump_attention {
            if self.standard.use_rotary { Some("rust_dit_self") } else { Some("rust_dit_cross") }
        } else {
            None
        };
        if let Some(prefix) = attention_dump_prefix {
            let dump_step = FORWARD_DUMP_STEP.load(Ordering::Relaxed);
            dump_forward_tensor(dump_step, &format!("{prefix}_q_norm"), &q);
            dump_forward_tensor(dump_step, &format!("{prefix}_k_norm"), &k);
            dump_forward_tensor(dump_step, &format!("{prefix}_v"), &v);
        }
        if trace { eprintln!("[progress] phase=attention_qk_norm_complete"); }
        let (q, k) = if self.standard.use_rotary {
            let device = q.device();
            if let Some((height, width)) = spatial {
                apply_rotary_3d_pair(q, k, height, width, self.standard.head_size, &device)
            } else {
                apply_rotary_pair(q, k, query_length, context_length, self.standard.head_size, &device)
            }
        } else {
            (q, k)
        };
        // The DiT block is the only checkpoint boundary, matching Python's
        // torch.utils.checkpoint call. Do not add nested q/k/v boundaries here:
        // they cause Burn to replay the same projection and RoPE graph more
        // than once during one block backward.
        if let Some(prefix) = attention_dump_prefix {
            let dump_step = FORWARD_DUMP_STEP.load(Ordering::Relaxed);
            dump_forward_tensor(dump_step, &format!("{prefix}_q_rotary"), &q);
            dump_forward_tensor(dump_step, &format!("{prefix}_k_rotary"), &k);
        }
        #[cfg(feature = "cuda")]
        if std::env::var_os("AFTERBURNER_DEBUG_LORA_GRADIENT").is_some()
            && DEBUG_ATTENTION_GRAPH_CALLS.fetch_add(1, Ordering::Relaxed) < 3
        {
            let requirement = |tensor: &Tensor<burn_autodiff::Autodiff<Inner, C>, 4>| match tensor
                .clone()
                .into_primitive()
            {
                TensorPrimitive::Float(value) => format!("{:?}", value.node.requirement),
                TensorPrimitive::QFloat(_) => "QFloat".to_owned(),
            };
            eprintln!(
                "[attention-graph] q={} k={} v={}",
                requirement(&q),
                requirement(&k),
                requirement(&v)
            );
        }
        let scale = 1.0 / (self.standard.head_size as f32).sqrt();
        let context_mask = context_mask.map(|mask| match mask.into_primitive() {
            TensorPrimitive::Float(value) => value,
            TensorPrimitive::QFloat(_) => panic!("quantized attention masks are unsupported"),
        });
        if trace { eprintln!("[progress] phase=attention_fused_begin"); }
        let attended = burn_autodiff::ops::fused_attention_manual::<Inner>(
            match q.into_primitive() {
                TensorPrimitive::Float(value) => value,
                TensorPrimitive::QFloat(_) => panic!("quantized attention queries are unsupported"),
            },
            match k.into_primitive() {
                TensorPrimitive::Float(value) => value,
                TensorPrimitive::QFloat(_) => panic!("quantized attention keys are unsupported"),
            },
            match v.into_primitive() {
                TensorPrimitive::Float(value) => value,
                TensorPrimitive::QFloat(_) => panic!("quantized attention values are unsupported"),
            },
            context_mask,
            scale.elem(),
            chunk_size,
        );
        if let Some(prefix) = attention_dump_prefix {
            let dump_step = FORWARD_DUMP_STEP.load(Ordering::Relaxed);
            let attended_tensor = Tensor::<burn_autodiff::Autodiff<Inner, C>, 4>::from_primitive(
                TensorPrimitive::Float(attended.clone()),
            );
            dump_forward_tensor(dump_step, &format!("{prefix}_attended"), &attended_tensor);
        }
        if trace { eprintln!("[progress] phase=attention_fused_complete"); }
        let attended = Tensor::<burn_autodiff::Autodiff<Inner, C>, 4>::from_primitive(
            TensorPrimitive::Float(attended),
        )
        .swap_dims(1, 2)
        .reshape([
            batch,
            query_length,
            self.standard.head_count * self.standard.head_size,
        ]);
        if mixed_precision && !recompute_linear {
            self.standard
                .output_proj
                .forward_low_precision(attended)
                .cast(input_dtype)
        } else {
            self.standard
                .output_proj
                .forward_fused_recomputable_sequence_chunked(attended)
                .cast(input_dtype)
        }
    }
}

#[cfg(feature = "cuda")]
impl<Inner, C> AnimaAttentionLayer<burn_autodiff::Autodiff<Inner, C>>
    for AnimaAttentionCuda<burn_autodiff::Autodiff<Inner, C>>
where
    Inner: Backend + BlockCheckpointBackend,
    C: burn_autodiff::checkpoint::strategy::CheckpointStrategy,
{
    fn forward_block(
        block: &AnimaDiTBlock<burn_autodiff::Autodiff<Inner, C>, Self>,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        text: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        condition: Tensor<burn_autodiff::Autodiff<Inner, C>, 2>,
        timestep_embedding: Tensor<burn_autodiff::Autodiff<Inner, C>, 2>,
        text_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        block.forward_checkpointed(AnimaDiTBlockInputs {
            input, text, condition, timestep_embedding, text_mask, spatial,
        })
    }

    fn forward_block_segment(
        blocks: &[&AnimaDiTBlock<burn_autodiff::Autodiff<Inner, C>, Self>],
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        text: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        condition: Tensor<burn_autodiff::Autodiff<Inner, C>, 2>,
        timestep_embedding: Tensor<burn_autodiff::Autodiff<Inner, C>, 2>,
        text_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        AnimaDiTBlock::forward_segment_checkpointed(
            blocks,
            AnimaDiTBlockInputs { input, text, condition, timestep_embedding, text_mask, spatial },
        )
    }

    fn forward_lora_linear(
        &self,
        layer: &LoraLinear<burn_autodiff::Autodiff<Inner, C>>,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        // Keep long 3-D projections as one GPU-side fused LoRA operation.
        // Expanding the BF16 path into ordinary Burn Linear nodes creates
        // thousands of host-scheduled autodiff steps before the CUDA work is
        // submitted.  The fused operation still casts its operands to the
        // activation dtype, while its base weight remains detached.
        layer.forward_fused_sequence_chunked(input)
    }

    fn forward_lora_linear_recomputable(
        &self,
        layer: &LoraLinear<burn_autodiff::Autodiff<Inner, C>>,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        if std::env::var_os("AFTERBURNER_DISABLE_LINEAR_RECOMPUTE").is_none() {
            layer.forward_fused_recomputable_sequence_chunked(input)
        } else {
            layer.forward_fused_sequence_chunked(input)
        }
    }

    fn forward_lora_linear_2d(
        &self,
        layer: &LoraLinear<burn_autodiff::Autodiff<Inner, C>>,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 2>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 2> {
        // AdaLN/timestep projections are small 2-D tensors.  Keep them on
        // Burn's reference Linear path: the fused CUDA LoRA kernel was
        // producing a forward-scale mismatch here (the first stage where
        // Rust diverged from Python), while the long 3-D projections still
        // use the memory-saving fused path below.
        if std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some() {
            layer.forward_low_precision(input)
        } else {
            layer.forward(input)
        }
    }

    fn standard(&self) -> &AnimaAttentionStandard<burn_autodiff::Autodiff<Inner, C>> {
        &self.standard
    }

    fn standard_mut(&mut self) -> &mut AnimaAttentionStandard<burn_autodiff::Autodiff<Inner, C>> {
        &mut self.standard
    }

    fn init_self(
        config: &AnimaDiTBlockConfig,
        device: &<burn_autodiff::Autodiff<Inner, C> as Backend>::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self::from_standard(AnimaAttentionStandard::init_self(
            config, device,
        )?))
    }

    fn init_cross(
        config: &AnimaDiTBlockConfig,
        device: &<burn_autodiff::Autodiff<Inner, C> as Backend>::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self::from_standard(AnimaAttentionStandard::init_cross(
            config, device,
        )?))
    }

    fn init_adapter_cross(
        config: &AnimaDiTBlockConfig,
        device: &<burn_autodiff::Autodiff<Inner, C> as Backend>::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self::from_standard(
            AnimaAttentionStandard::init_adapter_cross(config, device)?,
        ))
    }

    fn forward_with_mask(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        AnimaAttentionCuda::forward_with_mask(self, input, context, context_mask)
    }

    fn forward_with_mask_and_spatial(
        &self,
        input: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context: Tensor<burn_autodiff::Autodiff<Inner, C>, 3>,
        context_mask: Option<Tensor<burn_autodiff::Autodiff<Inner, C>, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<burn_autodiff::Autodiff<Inner, C>, 3> {
        AnimaAttentionCuda::forward_with_mask_and_spatial(
            self,
            input,
            context,
            context_mask,
            spatial,
        )
    }
}

fn apply_rotary<B: Backend>(
    input: Tensor<B, 4>,
    sequence_length: usize,
    head_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, heads, _, _] = input.dims();
    let input_dtype = input.dtype();
    let half = head_size / 2;
    let mut cos_values = Vec::with_capacity(sequence_length * half);
    let mut sin_values = Vec::with_capacity(sequence_length * half);
    for position in 0..sequence_length {
        for index in 0..half {
            let frequency = 10000.0_f32.powf(-2.0 * index as f32 / head_size as f32);
            let angle = position as f32 * frequency;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }
    let cos = Tensor::<B, 4>::from_data(
        TensorData::new(cos_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
        device,
    )
    .cast(input_dtype);
    let sin = Tensor::<B, 4>::from_data(
        TensorData::new(sin_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
        device,
    )
    .cast(input_dtype);
    let first = input
        .clone()
        .slice([0..batch, 0..heads, 0..sequence_length, 0..half]);
    let second = input.slice([0..batch, 0..heads, 0..sequence_length, half..head_size]);
    Tensor::cat(
        vec![
            first
                .clone()
                .mul(cos.clone())
                .sub(second.clone().mul(sin.clone())),
            second.mul(cos).add(first.mul(sin)),
        ],
        3,
    )
}

fn apply_rotary_pair<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    query_length: usize,
    key_length: usize,
    head_size: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    if query_length == key_length {
        let input_dtype = q.dtype();
        let half = head_size / 2;
        let mut cos_values = Vec::with_capacity(query_length * half);
        let mut sin_values = Vec::with_capacity(query_length * half);
        for position in 0..query_length {
            for index in 0..half {
                let frequency = 10000.0_f32.powf(-2.0 * index as f32 / head_size as f32);
                let angle = position as f32 * frequency;
                cos_values.push(angle.cos());
                sin_values.push(angle.sin());
            }
        }
        let cos = Tensor::<B, 4>::from_data(
            TensorData::new(cos_values, [1, 1, query_length, half]).convert::<B::FloatElem>(),
            device,
        )
        .cast(input_dtype);
        let sin = Tensor::<B, 4>::from_data(
            TensorData::new(sin_values, [1, 1, query_length, half]).convert::<B::FloatElem>(),
            device,
        )
        .cast(input_dtype);
        return (
            apply_rotary_with_tables(q, cos.clone(), sin.clone(), head_size),
            apply_rotary_with_tables(k, cos, sin, head_size),
        );
    }
    (
        apply_rotary(q, query_length, head_size, device),
        apply_rotary(k, key_length, head_size, device),
    )
}

fn apply_rotary_with_tables<B: Backend>(
    input: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    head_size: usize,
) -> Tensor<B, 4> {
    let [batch, heads, sequence_length, _] = input.dims();
    let half = head_size / 2;
    let first = input
        .clone()
        .slice([0..batch, 0..heads, 0..sequence_length, 0..half]);
    let second = input.slice([0..batch, 0..heads, 0..sequence_length, half..head_size]);
    Tensor::cat(
        vec![
            first.clone().mul(cos.clone()).sub(second.clone().mul(sin.clone())),
            second.mul(cos).add(first.mul(sin)),
        ],
        3,
    )
}

fn apply_rotary_3d<B: Backend>(
    input: Tensor<B, 4>,
    height: usize,
    width: usize,
    head_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, heads, sequence_length, _] = input.dims();
    let input_dtype = input.dtype();
    assert_eq!(sequence_length, height * width);
    assert_eq!(head_size % 2, 0);
    let dim_h = head_size / 6 * 2;
    let dim_w = dim_h;
    let dim_t = head_size - dim_h - dim_w;
    // kohya's default extrapolation ratios are 1.0, so the base theta for
    // both spatial axes is exactly 10000.  The previous extra `4^...` factor
    // was not present in the Python VideoRopePosition3DEmb implementation
    // and changed every DiT self-attention position encoding.
    let h_theta = 10000.0_f32;
    let w_theta = 10000.0_f32;
    let t_theta = 10000.0_f32;
    let mut frequencies = Vec::with_capacity(sequence_length * (head_size / 2));
    for token in 0..sequence_length {
        let row = token / width;
        let column = token % width;
        let mut values = Vec::with_capacity(head_size / 2);
        for index in 0..dim_t / 2 {
            values.push(0.0_f32 / (t_theta.powf((2 * index) as f32 / dim_t as f32)));
        }
        for index in 0..dim_h / 2 {
            values.push(row as f32 / h_theta.powf((2 * index) as f32 / dim_h as f32));
        }
        for index in 0..dim_w / 2 {
            values.push(column as f32 / w_theta.powf((2 * index) as f32 / dim_w as f32));
        }
        frequencies.extend(values);
    }
    let half = head_size / 2;
    let cos_half = Tensor::<B, 4>::from_data(
        TensorData::new(
            frequencies
                .iter()
                .map(|value| value.cos())
                .collect::<Vec<_>>(),
            [1, 1, sequence_length, half],
        )
        .convert::<B::FloatElem>(),
        device,
    )
    .cast(input_dtype);
    let sin_half = Tensor::<B, 4>::from_data(
        TensorData::new(
            frequencies
                .iter()
                .map(|value| value.sin())
                .collect::<Vec<_>>(),
            [1, 1, sequence_length, half],
        )
        .convert::<B::FloatElem>(),
        device,
    )
    .cast(input_dtype);
    let first = input
        .clone()
        .slice([0..batch, 0..heads, 0..sequence_length, 0..half]);
    let second = input.slice([0..batch, 0..heads, 0..sequence_length, half..head_size]);
    Tensor::cat(
        vec![
            first
                .clone()
                .mul(cos_half.clone())
                .sub(second.clone().mul(sin_half.clone())),
            second.mul(cos_half).add(first.mul(sin_half)),
        ],
        3,
    )
}

fn apply_rotary_3d_pair<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    height: usize,
    width: usize,
    head_size: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // Self-attention has identical q/k spatial lengths. Build the CPU RoPE
    // frequencies and upload the two tables once, then share them between q
    // and k. Cross-attention with a different length falls back to the
    // existing independent path.
    if q.dims()[2] != k.dims()[2] {
        return (
            apply_rotary_3d(q, height, width, head_size, device),
            apply_rotary_3d(k, height, width, head_size, device),
        );
    }
    let sequence_length = q.dims()[2];
    assert_eq!(sequence_length, height * width);
    assert_eq!(head_size % 2, 0);
    let dim_h = head_size / 6 * 2;
    let dim_w = dim_h;
    let dim_t = head_size - dim_h - dim_w;
    let mut frequencies = Vec::with_capacity(sequence_length * (head_size / 2));
    for token in 0..sequence_length {
        let row = token / width;
        let column = token % width;
        for index in 0..dim_t / 2 {
            frequencies.push(0.0_f32 / 10000.0_f32.powf((2 * index) as f32 / dim_t as f32));
        }
        for index in 0..dim_h / 2 {
            frequencies.push(row as f32 / 10000.0_f32.powf((2 * index) as f32 / dim_h as f32));
        }
        for index in 0..dim_w / 2 {
            frequencies.push(column as f32 / 10000.0_f32.powf((2 * index) as f32 / dim_w as f32));
        }
    }
    let half = head_size / 2;
    let cos_values = frequencies.iter().map(|value| value.cos()).collect::<Vec<_>>();
    let sin_values = frequencies.iter().map(|value| value.sin()).collect::<Vec<_>>();
    let input_dtype = q.dtype();
    let cos = Tensor::<B, 4>::from_data(
        TensorData::new(cos_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
        device,
    ).cast(input_dtype);
    let sin = Tensor::<B, 4>::from_data(
        TensorData::new(sin_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
        device,
    ).cast(input_dtype);
    (
        apply_rotary_with_tables(q, cos.clone(), sin.clone(), head_size),
        apply_rotary_with_tables(k, cos, sin, head_size),
    )
}

#[derive(Module, Debug)]
pub struct AnimaDiTBlock<B: Backend, A = AnimaAttentionStandard<B>>
where
    A: Module<B>,
{
    pub self_attn: A,
    pub cross_attn: A,
    pub mlp_layer1: LoraLinear<B>,
    pub mlp_layer2: LoraLinear<B>,
    pub self_attn_modulation: AdaLnModulation<B>,
    pub cross_attn_modulation: AdaLnModulation<B>,
    pub mlp_modulation: AdaLnModulation<B>,
}

impl<B, A> AnimaDiTBlock<B, A>
where
    B: Backend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B>,
{
    pub fn init(
        config: &AnimaDiTBlockConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        config.validate()?;
        Ok(Self {
            self_attn: A::init_self(config, device)?,
            cross_attn: A::init_cross(config, device)?,
            mlp_layer1: config
                .linear(config.hidden_size, config.mlp_hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            mlp_layer2: config
                .linear(config.mlp_hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            self_attn_modulation: AdaLnModulation::init(config, device)?,
            cross_attn_modulation: AdaLnModulation::init(config, device)?,
            mlp_modulation: AdaLnModulation::init(config, device)?,
        })
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.forward_with_text_mask(input, text, condition, timestep_embedding, None)
    }

    pub fn forward_with_text_mask(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        self.forward_with_text_mask_and_spatial(
            input,
            text,
            condition,
            timestep_embedding,
            text_mask,
            None,
        )
    }

    pub(crate) fn forward_with_text_mask_and_spatial(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        condition: Tensor<B, 2>,
        timestep_embedding: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
        spatial: Option<(usize, usize)>,
    ) -> Tensor<B, 3> {
        let dit_block_index = DIT_BLOCK_FORWARD_INDEX.fetch_add(1, Ordering::Relaxed);
        // Python's Block.forward sets use_fp32 when the residual stream is
        // FP16.  Convert the complete block boundary together so every
        // residual, AdaLN parameter, and normalization uses the same dtype.
        // This prevents the Rust path from silently doing the residual math
        // in FP16 while the reference path does it in FP32.
        let activation_dtype = input.dtype();
        let fp32_residuals = python_fp16_residual_path(&input);
        let fp32_boundary = python_fp16_boundary(&input);
        let input = if fp32_residuals { input.cast(FloatDType::F32) } else { input };
        let text = if fp32_boundary { text.cast(FloatDType::F32) } else { text };
        let condition = if fp32_boundary {
            condition.cast(FloatDType::F32)
        } else {
            condition
        };
        let timestep_embedding = if fp32_boundary {
            timestep_embedding.cast(FloatDType::F32)
        } else {
            timestep_embedding
        };
        let residual_dtype = if fp32_residuals {
            FloatDType::F32.into()
        } else {
            activation_dtype
        };
        let trace = std::env::var_os("AFTERBURNER_TRACE").is_some();
        let dump_step = FORWARD_DUMP_STEP.load(Ordering::Relaxed);
        if trace {
            eprintln!(
                "[dtype] block={} input={:?} text={:?} condition={:?} timestep={:?}",
                dit_block_index,
                input.dtype(),
                text.dtype(),
                condition.dtype(),
                timestep_embedding.dtype()
            );
            eprintln!("[progress] phase=block_self_modulation_begin");
        }
        if dit_block_index == 0 && dump_step > 0 {
            let params = self.self_attn_modulation.forward_with_linear(
                condition.clone(),
                |layer, input| self.self_attn.forward_lora_linear_2d(layer, input),
            );
            dump_forward_tensor(dump_step, "rust_block00_self_params", &params);
        }
        // Lifetime: retain only the block-stage boundary, not the complete
        // modulation graph, across the remaining 27 blocks.
        let (self_input, self_gate) = self.self_attn_modulation.apply_with_linear(
            input.clone(),
            condition.clone(),
            timestep_embedding.clone(),
            |layer, input| self.self_attn.forward_lora_linear_2d(layer, input),
        );
        let self_input = checkpoint_internal(self_input.cast(activation_dtype));
        let self_gate = checkpoint_internal(self_gate);
        if dit_block_index == 0 {
            dump_forward_tensor(dump_step, "rust_block00_self_input", &self_input);
            dump_forward_tensor(dump_step, "rust_block00_self_gate", &self_gate);
        }
        if trace { eprintln!("[progress] phase=block_self_modulation_complete"); }
        let self_input_dtype = self_input.dtype();
        let self_gate_dtype = self_gate.dtype();
        // Lifetime: self-attention workspace is eligible for recomputation
        // as soon as this block advances to the cross-attention stage.
        let self_output = self.self_attn.forward_with_mask_and_spatial(
            self_input.clone(),
            self_input,
            None,
            spatial,
        );
        let self_output = checkpoint_internal(self_output.cast(residual_dtype));
        if trace {
            eprintln!("[dtype] block={} self_input={:?} self_gate={:?} self_output={:?}", dit_block_index, self_input_dtype, self_gate_dtype, self_output.dtype());
        }
        if dit_block_index == 0 {
            dump_forward_tensor(
                FORWARD_DUMP_STEP.load(Ordering::Relaxed),
                "rust_block00_self_attn",
                &self_output,
            );
        }
        if trace { eprintln!("[progress] phase=block_self_attention_complete"); }
        // Lifetime: residual output is the only value needed by the next
        // stage; the multiply and add intermediates must not remain live.
        let hidden = checkpoint_internal(input + self_output.mul(self_gate));
        if dit_block_index == 0 {
            dump_forward_tensor(
                FORWARD_DUMP_STEP.load(Ordering::Relaxed),
                "rust_block00_after_self",
                &hidden,
            );
        }
        if trace { eprintln!("[progress] phase=block_cross_modulation_begin"); }
        if dit_block_index == 0 && dump_step > 0 {
            let params = self.cross_attn_modulation.forward_with_linear(
                condition.clone(),
                |layer, input| self.cross_attn.forward_lora_linear_2d(layer, input),
            );
            dump_forward_tensor(dump_step, "rust_block00_cross_params", &params);
        }
        let (cross_input, cross_gate) = self.cross_attn_modulation.apply_with_linear(
            hidden.clone(),
            condition.clone(),
            timestep_embedding.clone(),
            |layer, input| self.cross_attn.forward_lora_linear_2d(layer, input),
        );
        let cross_input = checkpoint_internal(cross_input.cast(activation_dtype));
        let cross_gate = checkpoint_internal(cross_gate);
        if dit_block_index == 0 {
            dump_forward_tensor(dump_step, "rust_block00_cross_input", &cross_input);
            dump_forward_tensor(dump_step, "rust_block00_cross_gate", &cross_gate);
        }
        if trace { eprintln!("[progress] phase=block_cross_modulation_complete"); }
        // Lifetime: discard cross-attention's Q/K/V and score workspace at
        // the block boundary; backward will reconstruct this stage.
        let cross_output =
            self.cross_attn
                .forward_with_mask_and_spatial(cross_input, text.cast(activation_dtype), text_mask, None);
        let cross_output = checkpoint_internal(cross_output.cast(residual_dtype));
        if dit_block_index == 0 {
            dump_forward_tensor(
                FORWARD_DUMP_STEP.load(Ordering::Relaxed),
                "rust_block00_cross_attn",
                &cross_output,
            );
        }
        if trace { eprintln!("[progress] phase=block_cross_attention_complete"); }
        let hidden = checkpoint_internal(hidden + cross_output.mul(cross_gate));
        if dit_block_index == 0 {
            dump_forward_tensor(
                FORWARD_DUMP_STEP.load(Ordering::Relaxed),
                "rust_block00_after_cross",
                &hidden,
            );
        }
        if trace { eprintln!("[progress] phase=block_mlp_begin"); }
        let (mlp_input, mlp_gate) = self.mlp_modulation.apply_with_linear(
            hidden.clone(),
            condition,
            timestep_embedding,
            |layer, input| self.self_attn.forward_lora_linear_2d(layer, input),
        );
        let mlp_input = checkpoint_internal(mlp_input.cast(activation_dtype));
        let mlp_gate = checkpoint_internal(mlp_gate);
        if dit_block_index == 0 {
            dump_forward_tensor(dump_step, "rust_block00_mlp_input", &mlp_input);
            dump_forward_tensor(dump_step, "rust_block00_mlp_gate", &mlp_gate);
        }
        let mixed_precision = std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some();
        // Lifetime: the wide [tokens, 8192] MLP intermediates are the main
        // per-block workspace.  Mark the completed MLP output immediately so
        // it can be recomputed instead of retained for every DiT block.
        // The fused recomputable Linear path is intentionally opt-in.  A
        // block-level checkpoint already bounds the retained activation set;
        // recomputing both wide MLP projections by default can make the
        // autodiff tape recursively replay the same block and effectively
        // stall backward on long sequences.  Keep the normal path as the
        // correctness/progress baseline and enable the extra replay only for
        // focused memory experiments.
        let mlp = if std::env::var_os("AFTERBURNER_ENABLE_LINEAR_RECOMPUTE").is_some() {
            self.self_attn.forward_lora_linear_recomputable(
                &self.mlp_layer2,
                gelu(self.self_attn.forward_lora_linear_recomputable(
                    &self.mlp_layer1,
                    mlp_input,
                )),
            )
        } else {
            self.self_attn.forward_lora_linear(
                &self.mlp_layer2,
                gelu(self.self_attn.forward_lora_linear(&self.mlp_layer1, mlp_input)),
            )
        };
        let mlp = if mixed_precision {
            mlp.cast(residual_dtype)
        } else {
            mlp
        };
        let mlp = checkpoint_internal(mlp);
        if dit_block_index == 0 {
            dump_forward_tensor(
                FORWARD_DUMP_STEP.load(Ordering::Relaxed),
                "rust_block00_mlp",
                &mlp,
            );
        }
        if trace { eprintln!("[progress] phase=block_mlp_complete"); }
        let hidden = checkpoint_internal(hidden + mlp.mul(mlp_gate));
        if dit_block_index == 0 {
            dump_forward_tensor(
                FORWARD_DUMP_STEP.load(Ordering::Relaxed),
                "rust_block00_after_mlp",
                &hidden,
            );
        }
        hidden
    }
}

/// Projects precomputed timestep features into the Anima conditioning paths.
/// The first output is `[batch, 3 * hidden]` and is added to each block's
/// shift/scale/gate projection. The second output is the RMS-normalized
/// timestep vector consumed by the per-block AdaLN projections.
#[derive(Module, Debug)]
pub struct AnimaTimestepEmbedder<B: Backend> {
    pub linear1: LoraLinear<B>,
    pub linear2: LoraLinear<B>,
    pub norm: AnimaRmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct AnimaRmsNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub epsilon: f32,
}

impl<B: Backend> AnimaRmsNorm<B> {
    fn init(size: usize, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([size], device),
            epsilon: 1e-6,
        }
    }

    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // kohya's RMSNorm explicitly performs the normalization in F32 under
        // autocast, then converts the result back to the input dtype.  Do the
        // same here; normalizing in BF16 (even with algebraic rescaling)
        // changes q/k and every subsequent attention result.
        let input_dtype = input.dtype();
        let feature_count = input.dims()[D - 1] as f32;
        let input_f32 = input.cast(FloatDType::F32);
        // Do not use Burn's mean_dim here.  On the CUDA backend/version used
        // by this project, the reduction path can apply an incorrect scale
        // for the last dimension, which shrinks the normalized vector and
        // changes every AdaLN modulation.  Spell out RMS normalization as
        // sum / feature_count so the Python reference formula is explicit.
        let rms = input_f32
            .clone()
            .square()
            .sum_dim(D - 1)
            .div_scalar(feature_count)
            .add_scalar(self.epsilon)
            .sqrt();
        input_f32
            .div(rms)
            .mul(self.gamma.val().detach().cast(FloatDType::F32).unsqueeze())
            .cast(input_dtype)
    }
}

impl<B: Backend> AnimaTimestepEmbedder<B> {
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            linear1: config
                .linear(config.hidden_size, config.hidden_size)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            linear2: config
                .linear(config.hidden_size, config.hidden_size * 3)?
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            norm: AnimaRmsNorm::init(config.hidden_size, device),
        })
    }

    pub fn forward(&self, features: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.forward_with_linear(features, |layer, input| layer.forward(input))
    }

    fn forward_with_linear<F>(
        &self,
        features: Tensor<B, 2>,
        linear: F,
    ) -> (Tensor<B, 2>, Tensor<B, 2>)
    where
        F: Fn(&LoraLinear<B>, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let normalized = self.norm.forward(features.clone());
        let embedding = linear(&self.linear2, silu(linear(&self.linear1, features)));
        (embedding, normalized)
    }
}

#[derive(Module, Debug)]
pub struct AnimaLlmAdapterBlock<B: Backend, A = AnimaAttentionStandard<B>>
where
    A: Module<B>,
{
    pub norm_self_attn: AnimaRmsNorm<B>,
    pub norm_cross_attn: AnimaRmsNorm<B>,
    pub norm_mlp: AnimaRmsNorm<B>,
    pub self_attn: A,
    pub cross_attn: A,
    pub mlp_layer1: LoraLinear<B>,
    pub mlp_layer2: LoraLinear<B>,
}

impl<B, A> AnimaLlmAdapterBlock<B, A>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        Ok(Self {
            norm_self_attn: AnimaRmsNorm::init(config.hidden_size, device),
            norm_cross_attn: AnimaRmsNorm::init(config.hidden_size, device),
            norm_mlp: AnimaRmsNorm::init(config.hidden_size, device),
            self_attn: A::init_self(config, device)?,
            cross_attn: A::init_adapter_cross(config, device)?,
            mlp_layer1: LoraLinearConfig {
                bias: true,
                ..config.linear(config.hidden_size, config.mlp_hidden_size)?
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            mlp_layer2: LoraLinearConfig {
                bias: true,
                ..config.linear(config.mlp_hidden_size, config.hidden_size)?
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
        })
    }

    fn forward_with_mask(
        &self,
        input: Tensor<B, 3>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        self.forward_with_masks(input.clone(), input, text_mask.clone(), text_mask)
    }

    fn forward_with_masks(
        &self,
        input: Tensor<B, 3>,
        text: Tensor<B, 3>,
        target_mask: Option<Tensor<B, 4>>,
        source_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let self_input = self.norm_self_attn.forward(input.clone());
        let self_output =
            self.self_attn
                .forward_with_mask(self_input.clone(), self_input, target_mask);
        let hidden = input + self_output;
        let cross_input = self.norm_cross_attn.forward(hidden.clone());
        let cross_output =
            self.cross_attn
                .forward_with_mask(cross_input.clone(), text, source_mask);
        let hidden = hidden + cross_output;
        let mlp_input = self.norm_mlp.forward(hidden.clone());
        let mlp = self.self_attn.forward_lora_linear(
            &self.mlp_layer2,
            gelu(
                self.self_attn
                    .forward_lora_linear(&self.mlp_layer1, mlp_input),
            ),
        );
        hidden + mlp
    }
}

#[derive(Module, Debug)]
pub struct AnimaLlmAdapter<B: Backend, A = AnimaAttentionStandard<B>>
where
    A: Module<B>,
{
    pub token_embedding: Embedding<B>,
    pub blocks: Vec<AnimaLlmAdapterBlock<B, A>>,
    pub norm: AnimaRmsNorm<B>,
    pub output_proj: LoraLinear<B>,
}

impl<B, A> AnimaLlmAdapter<B, A>
where
    B: Backend,
    A: AnimaAttentionLayer<B>,
{
    fn init(config: &AnimaDiTBlockConfig, device: &B::Device) -> Result<Self, AnimaDiTConfigError> {
        let mut blocks = Vec::with_capacity(6);
        for _ in 0..6 {
            blocks.push(AnimaLlmAdapterBlock::<B, A>::init(config, device)?);
        }
        let output_proj = config.linear(config.hidden_size, config.hidden_size)?;
        let output_proj = LoraLinearConfig {
            bias: true,
            ..output_proj
        };
        Ok(Self {
            token_embedding: EmbeddingConfig::new(32_128, config.hidden_size).init(device),
            blocks,
            norm: AnimaRmsNorm::init(config.hidden_size, device),
            output_proj: output_proj
                .init::<B>(device)
                .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
        })
    }

    pub fn forward(&self, text: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_mask(text, None)
    }

    pub fn forward_with_mask(
        &self,
        mut text: Tensor<B, 3>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        for block in &self.blocks {
            text = block.forward_with_mask(text, text_mask.clone());
        }
        // Match Python LLMAdapter.forward exactly: out_proj is applied before
        // the final RMSNorm.  Reversing these two operations changes both the
        // conditioning scale and every downstream DiT gradient.
        let projected = self
            .blocks[0]
            .self_attn
            .forward_lora_linear(&self.output_proj, text);
        let projected = projected;
        self.norm.forward(projected)
    }

    pub fn forward_with_conditioning(
        &self,
        source: Tensor<B, 3>,
        input_ids: Tensor<B, 2, Int>,
        target_mask: Option<Tensor<B, 4>>,
        source_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let target_mask_for_output = target_mask.clone();
        let mut text = self.token_embedding.forward(input_ids);
        let dump_step = FORWARD_DUMP_STEP.load(Ordering::Relaxed);
        dump_forward_tensor(dump_step, "rust_adapter_token_embedding", &text);
        for (index, block) in self.blocks.iter().enumerate() {
            text = block.forward_with_masks(
                text,
                source.clone(),
                target_mask.clone(),
                source_mask.clone(),
            );
            dump_forward_tensor(dump_step, &format!("rust_adapter_block_{index:02}"), &text);
        }
        let projected = self
            .blocks[0]
            .self_attn
            .forward_lora_linear(&self.output_proj, text);
        dump_forward_tensor(dump_step, "rust_adapter_out_proj", &projected);
        let output = self.norm.forward(projected);
        dump_forward_tensor(dump_step, "rust_adapter_norm", &output);
        match target_mask_for_output {
            Some(mask) => {
                let [batch, _, _, sequence] = mask.dims();
                let valid = mask
                    .greater_elem(-0.5)
                    .float()
                    .reshape([batch, sequence, 1])
                    .cast(output.dtype());
                output.mul(valid)
            }
            None => output,
        }
    }
}

#[derive(Module, Debug)]
pub struct AnimaDiT<B: Backend, A = AnimaAttentionStandard<B>>
where
    A: Module<B>,
{
    pub input_projection: LoraLinear<B>,
    pub timestep_embedder: AnimaTimestepEmbedder<B>,
    pub llm_adapter: AnimaLlmAdapter<B, A>,
    pub blocks: Vec<AnimaDiTBlock<B, A>>,
    pub final_modulation: AnimaFinalModulation<B>,
    pub output_projection: LoraLinear<B>,
    pub input_features: usize,
    pub latent_channels: usize,
}

impl<B, A> AnimaDiT<B, A>
where
    B: Backend + BlockCheckpointBackend,
    A: AnimaAttentionLayer<B>,
{
    pub fn init(
        config: &AnimaConfig,
        lora: &LoraConfig,
        device: &B::Device,
    ) -> Result<Self, AnimaDiTConfigError> {
        config.validate().map_err(AnimaDiTConfigError::Invalid)?;
        lora.validate()
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?;
        let mut block_config = AnimaDiTBlockConfig::from(config);
        block_config.lora_rank = lora.rank;
        block_config.lora_alpha = lora.alpha;
        let mut blocks = Vec::with_capacity(config.block_count);
        for _ in 0..config.block_count {
            blocks.push(AnimaDiTBlock::<B, A>::init(&block_config, device)?);
        }
        Ok(Self {
            input_projection: LoraLinearConfig {
                input_dim: config.input_channels,
                output_dim: config.hidden_size,
                rank: lora.rank,
                alpha: lora.alpha,
                bias: false,
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            timestep_embedder: AnimaTimestepEmbedder::init(&block_config, device)?,
            llm_adapter: AnimaLlmAdapter::<B, A>::init(
                &AnimaDiTBlockConfig {
                    hidden_size: config.llm_adapter_hidden_size,
                    context_size: config.llm_adapter_hidden_size,
                    attention_head_count: 16,
                    attention_head_size: 64,
                    mlp_hidden_size: 4096,
                    lora_rank: lora.rank,
                    lora_alpha: lora.alpha,
                },
                device,
            )?,
            blocks,
            final_modulation: AnimaFinalModulation::init(&block_config, device)?,
            output_projection: LoraLinearConfig {
                input_dim: config.hidden_size,
                output_dim: config.output_channels,
                rank: lora.rank,
                alpha: lora.alpha,
                bias: false,
            }
            .init::<B>(device)
            .map_err(|error| AnimaDiTConfigError::Lora(error.to_string()))?,
            input_features: config.input_channels,
            latent_channels: config.output_channels / 4,
        })
    }

    /// Forward pass for already patchified latent tokens and cached text embeddings.
    pub fn forward(
        &self,
        latent_tokens: Tensor<B, 3>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.forward_with_text_mask(latent_tokens, text, timestep_features, None)
    }

    pub fn forward_with_text_mask(
        &self,
        latent_tokens: Tensor<B, 3>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let text = self.llm_adapter.forward_with_mask(text, text_mask.clone());
        let (timestep_embedding, time_condition) =
            self.timestep_embedder
                .forward_with_linear(timestep_features, |layer, input| {
                    self.blocks[0]
                        .self_attn
                        .forward_lora_linear_2d(layer, input)
                });
        let mut hidden = self.blocks[0]
            .self_attn
            .forward_lora_linear(&self.input_projection, latent_tokens);
        for block in &self.blocks {
            hidden = block.forward_with_text_mask(
                hidden,
                text.clone(),
                time_condition.clone(),
                timestep_embedding.clone(),
                text_mask.clone(),
            );
        }
        // The Python final layer receives the FP32 residual stream produced
        // by the FP16 `use_fp32` blocks. Keep its conditioning tensors on the
        // same boundary before the final AdaLN arithmetic.
        let final_fp32 = python_fp16_boundary(&hidden);
        let time_condition = if final_fp32 {
            time_condition.cast(FloatDType::F32)
        } else {
            time_condition
        };
        let timestep_embedding = if final_fp32 {
            timestep_embedding.cast(FloatDType::F32)
        } else {
            timestep_embedding
        };
        let hidden = self.final_modulation.forward_with_linear(
            hidden,
            time_condition,
            timestep_embedding,
            |layer, input| {
                self.blocks[0]
                    .self_attn
                    .forward_lora_linear_2d(layer, input)
            },
        );
        self.blocks[0]
            .self_attn
            .forward_lora_linear(&self.output_projection, hidden)
    }

    /// Runs the DiT using VAE latent layout instead of pre-patchified tokens.
    pub fn forward_latent(
        &self,
        latent: Tensor<B, 5>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        padding_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 5>, AnimaTensorError> {
        self.forward_latent_with_text_mask(latent, text, timestep_features, padding_mask, None)
    }

    pub fn forward_latent_with_text_mask(
        &self,
        latent: Tensor<B, 5>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        padding_mask: Option<Tensor<B, 4>>,
        text_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 5>, AnimaTensorError> {
        let [_, _, _, height, width] = latent.dims();
        let tokens = patchify_latent(latent, padding_mask)?;
        if tokens.dims()[2] != self.input_features {
            return Err(AnimaTensorError::InvalidShape {
                message: format!(
                    "Anima patchified latent must have {} input features",
                    self.input_features
                ),
                actual: tokens.dims().to_vec(),
            });
        }
        let predicted = self.forward_latent_tokens_with_spatial(
            tokens,
            text,
            timestep_features,
            text_mask,
            (height / 2, width / 2),
            None,
            None,
        );
        unpatchify_latent(predicted, self.latent_channels, height, width)
    }

    pub fn forward_latent_with_conditioning(
        &self,
        latent: Tensor<B, 5>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        padding_mask: Option<Tensor<B, 4>>,
        source_mask: Option<Tensor<B, 4>>,
        adapter_input_ids: Tensor<B, 2, Int>,
        adapter_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 5>, AnimaTensorError> {
        let [_, _, _, height, width] = latent.dims();
        let tokens = patchify_latent(latent, padding_mask)?;
        if tokens.dims()[2] != self.input_features {
            return Err(AnimaTensorError::InvalidShape {
                message: format!(
                    "Anima patchified latent must have {} input features",
                    self.input_features
                ),
                actual: tokens.dims().to_vec(),
            });
        }
        let predicted = self.forward_latent_tokens_with_spatial(
            tokens,
            text,
            timestep_features,
            source_mask,
            (height / 2, width / 2),
            Some(adapter_input_ids),
            adapter_mask,
        );
        unpatchify_latent(predicted, self.latent_channels, height, width)
    }

    fn forward_latent_tokens_with_spatial(
        &self,
        latent_tokens: Tensor<B, 3>,
        text: Tensor<B, 3>,
        timestep_features: Tensor<B, 2>,
        text_mask: Option<Tensor<B, 4>>,
        spatial: (usize, usize),
        adapter_input_ids: Option<Tensor<B, 2, Int>>,
        adapter_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let forward_dump_step = if std::env::var_os("AFTERBURNER_TRAIN_DUMP_DIR").is_some() {
            FORWARD_DUMP_STEP.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        };
        DIT_BLOCK_FORWARD_INDEX.store(0, Ordering::Relaxed);
        let debug = std::env::var_os("AFTERBURNER_DEBUG_FORWARD").is_some();
        if debug {
            eprintln!(
                "[forward] start latent={:?} text={:?} timestep={:?} conditioned={}",
                latent_tokens.dims(),
                text.dims(),
                timestep_features.dims(),
                adapter_input_ids.is_some()
            );
        }
        dump_forward_tensor(forward_dump_step, "rust_patch_tokens", &latent_tokens);
        let conditioned = adapter_input_ids.is_some();
        let compact_progress = std::env::var_os("AFTERBURNER_COMPACT_PROGRESS").is_some()
            || std::env::var_os("AFTERBURNER_TRACE").is_none();
        if !compact_progress {
            eprintln!("[progress] phase=llm_adapter_forward_begin");
        }
        let text = match adapter_input_ids {
            Some(input_ids) => self.llm_adapter.forward_with_conditioning(
                text,
                input_ids,
                adapter_mask,
                text_mask.clone(),
            ),
            None => self.llm_adapter.forward_with_mask(text, text_mask.clone()),
        };
        dump_forward_tensor(forward_dump_step, "rust_llm_adapter", &text);
        if !compact_progress {
            eprintln!("[progress] phase=llm_adapter_forward_complete");
        }
        let (timestep_embedding, time_condition) =
            self.timestep_embedder
                .forward_with_linear(timestep_features, |layer, input| {
                    self.blocks[0]
                        .self_attn
                        .forward_lora_linear_2d(layer, input)
                });
        dump_forward_tensor(
            forward_dump_step,
            "rust_timestep_embedding",
            &timestep_embedding,
        );
        dump_forward_tensor(forward_dump_step, "rust_time_condition", &time_condition);
        let mut hidden = self.blocks[0]
            .self_attn
            .forward_lora_linear(&self.input_projection, latent_tokens);
        dump_forward_tensor(forward_dump_step, "rust_input_projection", &hidden);
        if !compact_progress {
            eprintln!("[progress] phase=input_projection_complete");
        }
        let segment_size = std::env::var("AFTERBURNER_CHECKPOINT_SEGMENT_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(4)
            .min(self.blocks.len().max(1));
        if !compact_progress {
            eprintln!("[checkpoint] segment_size={} segments={}", segment_size,
                (self.blocks.len() + segment_size - 1) / segment_size);
        }
        let mut start = 0usize;
        while start < self.blocks.len() {
            let end = (start + segment_size).min(self.blocks.len());
            let block_slice = self.blocks[start..end]
                .iter()
                .collect::<Vec<_>>();
            if !compact_progress {
                eprintln!(
                    "[progress] phase=dit_segment_forward_begin blocks={}-{}",
                    start + 1, end
                );
            }
            hidden = A::forward_block_segment(
                &block_slice,
                hidden,
                text.clone(),
                time_condition.clone(),
                timestep_embedding.clone(),
                if conditioned { None } else { text_mask.clone() },
                Some(spatial),
            );
            if start == 0 || end == self.blocks.len() {
                dump_forward_tensor(
                    forward_dump_step,
                    &format!("rust_block_{:02}", end - 1),
                    &hidden,
                );
            }
            if !compact_progress {
                eprintln!(
                    "[progress] phase=dit_segment_forward_complete blocks={}-{}",
                    start + 1, end
                );
            }
            if debug {
                eprintln!("[forward] segment {}..{} complete: {:?}", start, end, hidden.dims());
            }
            start = end;
        }
        let final_fp32 = python_fp16_boundary(&hidden);
        let time_condition = if final_fp32 {
            time_condition.cast(FloatDType::F32)
        } else {
            time_condition
        };
        let timestep_embedding = if final_fp32 {
            timestep_embedding.cast(FloatDType::F32)
        } else {
            timestep_embedding
        };
        let hidden = self.final_modulation.forward_with_linear(
            hidden,
            time_condition,
            timestep_embedding,
            |layer, input| {
                self.blocks[0]
                    .self_attn
                    .forward_lora_linear_2d(layer, input)
            },
        );
        dump_forward_tensor(forward_dump_step, "rust_final_modulation", &hidden);
        let output = self.blocks[0]
            .self_attn
            .forward_lora_linear(&self.output_projection, hidden);
        dump_forward_tensor(forward_dump_step, "rust_output_projection", &output);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::{Autodiff, checkpoint::strategy::BalancedCheckpointing};

    #[test]
    fn block_forward_preserves_token_shape() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let block = AnimaDiTBlock::<Backend>::init(&config, &device).unwrap();
        let input = Tensor::<Backend, 3>::from_data(
            burn::tensor::TensorData::new(
                vec![
                    0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                ],
                [1, 3, 8],
            ),
            &device,
        );
        let text = Tensor::<Backend, 3>::ones([1, 5, 8], &device);
        let condition = Tensor::<Backend, 2>::ones([1, 8], &device);
        let timestep_embedding = Tensor::<Backend, 2>::ones([1, 24], &device);
        let output = block.forward(input, text, condition, timestep_embedding);
        assert_eq!(output.dims(), [1, 3, 8]);
    }

    #[test]
    fn attention_chunking_preserves_long_sequence_shape() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let attention = AnimaAttention::<Backend>::init_self(&config, &device).unwrap();
        let input = Tensor::<Backend, 3>::ones([1, ATTENTION_QUERY_CHUNK_SIZE + 1, 8], &device);
        let output = attention.forward(input.clone(), input);
        assert_eq!(output.dims(), [1, ATTENTION_QUERY_CHUNK_SIZE + 1, 8]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_attention_boundary_propagates_gradients() {
        type Inner = burn_ndarray::NdArray<f32>;
        type Backend = Autodiff<Inner, BalancedCheckpointing>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let standard = AnimaAttentionStandard::<Backend>::init_self(&config, &device).unwrap();
        let attention = AnimaAttentionCuda::from_standard(standard);
        let input = Tensor::<Backend, 3>::ones([1, 3, 8], &device).require_grad();
        let output = attention.forward(input.clone(), input.clone());
        assert_eq!(output.dims(), [1, 3, 8]);
        let gradients = output.sum().backward();
        assert!(input.grad(&gradients).is_some());
    }

    #[test]
    fn timestep_embedder_matches_cosmos_condition_shapes() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let embedder = AnimaTimestepEmbedder::<Backend>::init(&config, &device).unwrap();
        let (embedding, normalized) = embedder.forward(Tensor::ones([2, 8], &device));
        assert_eq!(embedding.dims(), [2, 24]);
        assert_eq!(normalized.dims(), [2, 8]);
    }

    #[test]
    fn patchify_and_unpatchify_preserve_image_layout() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let latent = Tensor::<Backend, 5>::ones([1, 2, 1, 4, 4], &device);
        let tokens = patchify_latent(latent, None).unwrap();
        assert_eq!(tokens.dims(), [1, 4, 12]);
        let restored = unpatchify_latent(tokens, 3, 4, 4).unwrap();
        assert_eq!(restored.dims(), [1, 3, 1, 4, 4]);
    }

    #[test]
    fn dit_propagates_lora_rank_to_each_block() {
        type Backend = burn_ndarray::NdArray<f32>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let mut config = crate::anima::AnimaConfig::default();
        config.block_count = 1;
        let lora = crate::lora::LoraConfig {
            rank: 1,
            alpha: 1,
            ..Default::default()
        };
        let model = AnimaDiT::<Backend>::init(&config, &lora, &device).unwrap();
        assert_eq!(
            model.blocks[0].self_attn.q_proj.lora_down.val().dims(),
            [1, 2048]
        );
        assert_eq!(
            model.blocks[0].self_attn.q_proj.lora_up.val().dims(),
            [2048, 1]
        );
    }

    #[test]
    fn attention_checkpointing_propagates_gradient_to_q_lora() {
        type Backend = Autodiff<burn_ndarray::NdArray<f32>, BalancedCheckpointing>;
        use burn::module::Param;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let mut attention = AnimaAttention::init_self(&config, &device).unwrap();
        let identity = Tensor::<Backend, 2>::eye(8, &device);
        attention.q_proj.weight = Param::from_tensor(identity.clone());
        attention.k_proj.weight = Param::from_tensor(identity.clone());
        attention.v_proj.weight = Param::from_tensor(identity.clone());
        attention.output_proj.weight = Param::from_tensor(identity);
        let input = Tensor::<Backend, 3>::from_data(
            burn::tensor::TensorData::new(
                vec![
                    0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                ],
                [1, 3, 8],
            ),
            &device,
        );
        let output = attention.forward(input.clone(), input);
        let loss = output
            .mul(Tensor::<Backend, 3>::from_data(
                burn::tensor::TensorData::new(
                    vec![
                        0.3_f32, 0.1, 0.5, 0.2, 0.7, 0.4, 0.9, 0.6, 1.1, 0.8, 1.3, 1.0, 1.5, 1.2,
                        1.7, 1.4, 1.9, 1.6, 2.1, 1.8, 2.3, 2.0, 2.5, 2.2,
                    ],
                    [1, 3, 8],
                ),
                &device,
            ))
            .sum();
        let gradients = loss.backward();
        let gradient = attention
            .q_proj
            .lora_up
            .val()
            .grad(&gradients)
            .expect("q projection LoRA gradient should reach the loss");
        let values = gradient.into_data().to_vec::<f32>().unwrap();
        assert!(values.iter().any(|value| value.abs() > 1e-8));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_attention_propagates_gradient_to_q_lora() {
        type Backend = Autodiff<burn_ndarray::NdArray<f32>, BalancedCheckpointing>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = AnimaDiTBlockConfig {
            hidden_size: 8,
            context_size: 8,
            attention_head_count: 2,
            attention_head_size: 4,
            mlp_hidden_size: 16,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let standard = AnimaAttentionStandard::init_self(&config, &device).unwrap();
        let mut attention = AnimaAttentionCuda::from_standard(standard);
        let identity = Tensor::<Backend, 2>::eye(8, &device);
        attention.standard.q_proj.weight = Param::from_tensor(identity.clone());
        attention.standard.k_proj.weight = Param::from_tensor(identity.clone());
        attention.standard.v_proj.weight = Param::from_tensor(identity.clone());
        attention.standard.output_proj.weight = Param::from_tensor(identity);
        let input = Tensor::<Backend, 3>::from_data(
            burn::tensor::TensorData::new(
                vec![
                    0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
                ],
                [1, 3, 8],
            ),
            &device,
        );
        let weights = Tensor::<Backend, 3>::from_data(
            burn::tensor::TensorData::new(
                vec![
                    0.3_f32, 0.1, 0.5, 0.2, 0.7, 0.4, 0.9, 0.6, 1.1, 0.8, 1.3, 1.0, 1.5, 1.2, 1.7,
                    1.4, 1.9, 1.6, 2.1, 1.8, 2.3, 2.0, 2.5, 2.2,
                ],
                [1, 3, 8],
            ),
            &device,
        );
        let loss = attention.forward(input.clone(), input).mul(weights).sum();
        let gradients = loss.backward();
        let gradient = attention
            .standard
            .q_proj
            .lora_up
            .val()
            .grad(&gradients)
            .expect("fused attention must return a Q projection gradient");
        let values = gradient.into_data().to_vec::<f32>().unwrap();
        let _key_values = attention
            .standard
            .k_proj
            .lora_up
            .val()
            .grad(&gradients)
            .expect("fused attention must return a K projection gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let _value_values = attention
            .standard
            .v_proj
            .lora_up
            .val()
            .grad(&gradients)
            .expect("fused attention must return a V projection gradient")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > 1e-8));
    }
}
