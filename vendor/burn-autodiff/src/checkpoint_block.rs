//! Block-scoped activation checkpointing with a single recomputation step.

use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{NodeId, NodeRef, Parent, Requirement, Step},
    tensor::AutodiffTensor,
    Autodiff,
    checkpoint::strategy::NoCheckpointing,
};
use alloc::{boxed::Box, vec::Vec};
use burn_tensor::{backend::Backend, Tensor, TensorPrimitive};
use core::fmt;

/// A local gradient and the outer graph node to which it must be returned.
pub struct GradBridge<B: Backend> {
    /// Node in the outer graph.
    pub outer_node_id: NodeId,
    /// Leaf in the isolated local graph.
    pub local_tensor: AutodiffTensor<B>,
}

/// One graph step representing an entire model block.
pub struct CheckpointBlockStep<B: Backend, const OUT_RANK: usize> {
    output_node: NodeRef,
    parents: Vec<Parent>,
    recompute_fn: Box<
        dyn FnOnce(
                ) -> (Tensor<Autodiff<B, NoCheckpointing>, OUT_RANK>, Vec<GradBridge<B>>)
            + Send,
    >,
}

/// A checkpoint step whose backward is implemented directly on the inner
/// backend. The closure receives the upstream gradient and must register all
/// parent gradients into the outer graph.
pub struct ManualVjpStep<B: Backend, const OUT_RANK: usize> {
    output_node: NodeRef,
    parents: Vec<Parent>,
    backward_fn: Box<dyn FnOnce(B::FloatTensorPrimitive, &mut Gradients) + Send>,
}

impl<B: Backend, const OUT_RANK: usize> fmt::Debug for ManualVjpStep<B, OUT_RANK> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManualVjpStep")
            .field("output_node", &self.output_node.id)
            .field("parents", &self.parents.len())
            .field("out_rank", &OUT_RANK)
            .finish()
    }
}

impl<B: Backend, const OUT_RANK: usize> Step for ManualVjpStep<B, OUT_RANK> {
    fn step(self: Box<Self>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let output_gradient = grads.consume::<B>(&self.output_node);
        (self.backward_fn)(output_gradient, grads);
    }

    fn node(&self) -> NodeId { self.output_node.id }
    fn parents(&self) -> &[Parent] { &self.parents }
    fn depth(&self) -> usize { self.output_node.order }
}

impl<B: Backend, const OUT_RANK: usize> fmt::Debug for CheckpointBlockStep<B, OUT_RANK> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckpointBlockStep")
            .field("output_node", &self.output_node.id)
            .field("parents", &self.parents.len())
            .field("out_rank", &OUT_RANK)
            .finish()
    }
}

impl<B: Backend, const OUT_RANK: usize> Step for CheckpointBlockStep<B, OUT_RANK> {
    fn step(self: Box<Self>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let output_gradient = grads.consume::<B>(&self.output_node);
        let _local_scope = crate::runtime::graph::LocalGraphGuard::enter();
        let (local_output, bridges) = (self.recompute_fn)();
        let local_root = match local_output.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            TensorPrimitive::QFloat(_) => panic!("quantized checkpoint output is unsupported"),
        };
        let mut local_grads =
            crate::runtime::graph::GraphMutexClient::backward_local_with_grad::<B>(
                local_root,
                output_gradient,
            );

        for bridge in bridges {
            if let Some(gradient) = local_grads.remove(&bridge.local_tensor) {
                grads.register::<B>(bridge.outer_node_id, gradient);
            }
        }
    }

    fn node(&self) -> NodeId {
        self.output_node.id
    }

    fn parents(&self) -> &[Parent] {
        &self.parents
    }

    fn depth(&self) -> usize {
        self.output_node.order
    }
}

/// Register one recomputation step for an output with dynamically many parents.
pub fn register_dynamic_checkpoint<B: Backend, const OUT_RANK: usize>(
    output: B::FloatTensorPrimitive,
    parent_nodes: &[NodeRef],
    recompute_fn: Box<
        dyn FnOnce(
                ) -> (Tensor<Autodiff<B, NoCheckpointing>, OUT_RANK>, Vec<GradBridge<B>>)
            + Send,
    >,
) -> AutodiffTensor<B> {
    let parents = parent_nodes
        .iter()
        .filter_map(|node| node.clone_if_require_grad())
        .collect::<Vec<_>>();
    if parents.is_empty() {
        return AutodiffTensor::new(output);
    }

    let result = AutodiffTensor::from_parents(
        output,
        &parents,
        Requirement::Grad,
        crate::graph::ComputingProperty::ComputeBound,
    );
    let step = CheckpointBlockStep {
        output_node: result.node.clone(),
        parents: result.node.parents.clone(),
        recompute_fn,
    };
    result.register_step(step, CheckpointerBuilder::default())
}

/// Registers a backend-only checkpoint step. Unlike
/// `register_dynamic_checkpoint`, this never creates a local Autodiff graph.
pub fn register_manual_checkpoint<B: Backend, const OUT_RANK: usize>(
    output: B::FloatTensorPrimitive,
    parent_nodes: &[NodeRef],
    backward_fn: Box<dyn FnOnce(B::FloatTensorPrimitive, &mut Gradients) + Send>,
) -> AutodiffTensor<B> {
    let parents = parent_nodes
        .iter()
        .filter_map(|node| node.clone_if_require_grad())
        .collect::<Vec<_>>();
    if parents.is_empty() {
        return AutodiffTensor::new(output);
    }
    let result = AutodiffTensor::from_parents(
        output,
        &parents,
        Requirement::Grad,
        crate::graph::ComputingProperty::ComputeBound,
    );
    let step: ManualVjpStep<B, OUT_RANK> = ManualVjpStep {
        output_node: result.node.clone(),
        parents: result.node.parents.clone(),
        backward_fn,
    };
    result.register_step(step, CheckpointerBuilder::default())
}

/// Registers one backend-only manual VJP node.
pub fn register_manual_vjp<B: Backend, const OUT_RANK: usize>(
    output: B::FloatTensorPrimitive,
    parent_nodes: &[NodeRef],
    backward_fn: Box<dyn FnOnce(B::FloatTensorPrimitive, &mut Gradients) + Send>,
) -> AutodiffTensor<B> {
    register_manual_checkpoint::<B, OUT_RANK>(output, parent_nodes, backward_fn)
}
