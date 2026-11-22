//! Explicit activation checkpoint boundaries for model-level recomputation.

use super::{Backward, Ops, OpsKind};
use crate::{
    checkpoint::{
        base::Checkpointer,
        retro_forward::RetroForward,
        state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::NodeId,
    tensor::AutodiffTensor,
};
use burn_tensor::backend::Backend;

#[derive(Clone, Copy, Debug)]
struct CheckpointBoundaryBackward;

#[derive(Clone, Debug)]
struct RetroCheckpointBoundary<B: Backend> {
    parent_id: NodeId,
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> RetroForward for RetroCheckpointBoundary<B> {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let parent = states.get_state::<B::FloatTensorPrimitive>(&self.parent_id);
        states.save(out_node, parent);
    }
}

impl<B: Backend> Backward<B, 1> for CheckpointBoundaryBackward {
    type State = ();

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let gradient = grads.consume::<B>(&ops.node);
        if let Some(parent) = ops.parents[0].as_ref() {
            grads.register::<B>(parent.id, gradient);
        }
    }
}

/// Marks an activation as a computed checkpoint through an identity operation.
pub fn checkpoint_boundary<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>,
) -> AutodiffTensor<B> {
    let prep = CheckpointBoundaryBackward
        .prepare::<C>([input.node.clone()])
        .memory_bound()
        .retro_forward(RetroCheckpointBoundary::<B> {
            parent_id: input.node.id,
            _backend: core::marker::PhantomData,
        })
        .parents([&input]);
    // Register a recomputation action instead of retaining the boundary
    // tensor itself.  The boundary is an identity node, so its retro-forward
    // can recover the value from the parent checkpoint without another GPU
    // allocation.  This is important for long DiT chains where every block
    // input would otherwise remain resident until backward.
    match prep.stateful() {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&input);
            prep.finish((), input.primitive)
        }
        OpsKind::UnTracked(prep) => prep.finish(input.primitive),
    }
}
