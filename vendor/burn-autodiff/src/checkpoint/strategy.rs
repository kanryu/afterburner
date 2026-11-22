use core::fmt::Debug;
use core::sync::atomic::{AtomicUsize, Ordering};

use burn_tensor::backend::Backend;

use crate::{graph::ComputingProperty, tensor::AutodiffTensor};
use alloc::sync::Arc;

use super::{
    builder::{ActionType, CheckpointerBuilder},
    retro_forward::RetroForward,
};

/// Strategy for the amount of checkpointing to do during autodiff
pub trait CheckpointStrategy: Clone + Copy + Debug + Default + Send + Sync + 'static {
    /// May modify the compute property depending on the strategy
    fn compute_property<R: RetroForward>(retro_forward: R) -> ComputingProperty;

    /// Checkpoints parents if necessary in the strategy
    fn checkpoint_parents<'a, B2, A>(
        parents: A,
        builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>;
}

#[derive(Debug)]
/// Error that can happen when trying to checkpoint a tensor.
pub enum CheckpointingError {
    /// When a parent is untracked, we can't easily checkpoint its state, since we don't know the
    /// requirements in advanced.
    UntrackedParent,
}

#[derive(Clone, Copy, Debug, Default)]
/// All operations are considered compute bound, notwithstanding how they are marked
pub struct NoCheckpointing {}

impl CheckpointStrategy for NoCheckpointing {
    /// An operation marked as memory bound is actually compute bound.
    fn compute_property<R: RetroForward>(_retro_forward: R) -> ComputingProperty {
        ComputingProperty::ComputeBound
    }

    /// An operation marked as memory bound is actually compute bound.
    /// It's therefore useless to checkpoint the parents
    fn checkpoint_parents<'a, B2, A>(
        _parents: A,
        _builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        // Nothing to do here
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
/// Operation properties are as they are marked (compute or memory bound)
pub struct BalancedCheckpointing {}

impl CheckpointStrategy for BalancedCheckpointing {
    /// An operation marked as memory bound is memory bound.
    /// When memory bound, an operation needs to save its RetroForward
    fn compute_property<R: RetroForward>(retro_forward: R) -> ComputingProperty {
        ComputingProperty::MemoryBound {
            retro_forward: Arc::new(retro_forward),
        }
    }

    /// An operation marked as memory bound is really memory bound.
    /// Since the operation may not checkpoint its parents but may need them indirectly
    /// if asked to recompute itself, the method needs to know the parent tensors to maybe checkpoint them
    fn checkpoint_parents<'a, B2, A>(
        parents: A,
        builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        let mut can_checkpoint = true;

        for tensor in parents.into_iter() {
            if let crate::graph::Requirement::None = tensor.node.requirement {
                can_checkpoint = false;
            } else {
                builder.checkpoint(tensor, ActionType::Backup);
            }
        }

        if !can_checkpoint {
            *builder = CheckpointerBuilder::default();
            return Err(CheckpointingError::UntrackedParent);
        }

        Ok(())
    }
}

/// Recomputes every operation that has a backward graph during reverse mode.
///
/// Burn's normal balanced strategy honors the operation's `compute_bound`
/// annotation. Large transformer blocks contain several compute-bound module
/// operations, however, so their activations remain live until the whole DiT
/// graph is traversed. This strategy deliberately treats those operations as
/// memory-bound as well. It is intended for long-sequence training where
/// bounded VRAM is more important than recomputation cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct AggressiveCheckpointing {}

impl CheckpointStrategy for AggressiveCheckpointing {
    fn compute_property<R: RetroForward>(retro_forward: R) -> ComputingProperty {
        ComputingProperty::MemoryBound {
            retro_forward: Arc::new(retro_forward),
        }
    }

    fn checkpoint_parents<'a, B2, A>(
        parents: A,
        builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        let mut can_checkpoint = true;
        for tensor in parents.into_iter() {
            if let crate::graph::Requirement::None = tensor.node.requirement {
                can_checkpoint = false;
            } else {
                builder.checkpoint(tensor, ActionType::Backup);
            }
        }
        if !can_checkpoint {
            *builder = CheckpointerBuilder::default();
            return Err(CheckpointingError::UntrackedParent);
        }
        Ok(())
    }
}

static BLOCK_CHECKPOINT_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Guard for the region corresponding to one Python `torch.utils.checkpoint`
/// invocation. Only operations created while the guard is alive are treated
/// as memory-bound; operations outside the DiT block remain compute-bound.
#[derive(Debug)]
pub struct BlockCheckpointGuard;

impl Drop for BlockCheckpointGuard {
    fn drop(&mut self) {
        BLOCK_CHECKPOINT_DEPTH.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Checkpoint strategy matching Python's block-scoped activation checkpointing.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockCheckpointing {}

impl BlockCheckpointing {
    /// Enter one checkpointed model block.
    pub fn enter() -> BlockCheckpointGuard {
        BLOCK_CHECKPOINT_DEPTH.fetch_add(1, Ordering::SeqCst);
        BlockCheckpointGuard
    }

    fn active() -> bool {
        BLOCK_CHECKPOINT_DEPTH.load(Ordering::SeqCst) != 0
            && std::env::var_os("AFTERBURNER_DISABLE_BLOCK_RECOMPUTE").is_none()
    }
}

impl CheckpointStrategy for BlockCheckpointing {
    fn compute_property<R: RetroForward>(retro_forward: R) -> ComputingProperty {
        if Self::active() {
            ComputingProperty::MemoryBound {
                retro_forward: Arc::new(retro_forward),
            }
        } else {
            ComputingProperty::ComputeBound
        }
    }

    fn checkpoint_parents<'a, B2, A>(
        parents: A,
        builder: &mut CheckpointerBuilder,
    ) -> Result<(), CheckpointingError>
    where
        B2: Backend,
        A: IntoIterator<Item = &'a AutodiffTensor<B2>>,
    {
        if Self::active() {
            BalancedCheckpointing::checkpoint_parents(parents, builder)
        } else {
            Ok(())
        }
    }
}
