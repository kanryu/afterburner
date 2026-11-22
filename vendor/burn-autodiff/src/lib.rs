#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Autodiff
//!
//! This autodiff library is a part of the Burn project. It is a standalone crate
//! that can be used to perform automatic differentiation on tensors. It is
//! designed to be used with the Burn Tensor crate, but it can be used with any
//! tensor library that implements the `Backend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Checkpoint module.
pub mod checkpoint;
/// Block-scoped activation checkpointing.
pub mod checkpoint_block;
/// Gradients module.
pub mod grads;
/// Operation module.
pub mod ops;

pub(crate) mod graph;
pub use graph::{NodeRef, Requirement};
// Exported for backend extension
pub use graph::NodeId;
pub(crate) mod tensor;
/// Public primitive handle used by custom backend operations.
pub use tensor::AutodiffTensor;
pub(crate) mod utils;

mod backend;

pub(crate) mod runtime;

pub use backend::*;

/// Reclaim autodiff graph nodes whose tensor handles are no longer alive.
///
/// This is intended for optimizer step boundaries, after the previous model
/// has been consumed and replaced by the updated model.
pub fn cleanup_dead_nodes() {
    runtime::graph::cleanup_dead_nodes();
}

#[cfg(feature = "export_tests")]
mod tests;

/// A facade around for HashMap and HashSet.
/// This avoids elaborate import wrangling having to happen in every module.
mod collections {
    #[cfg(not(feature = "std"))]
    pub use hashbrown::{HashMap, HashSet};
    #[cfg(feature = "std")]
    pub use std::collections::{HashMap, HashSet};
}
