/// Weight decay module for optimizers.
pub mod decay;

/// Momentum module for optimizers.
pub mod momentum;

mod adagrad;
mod adam;
mod adamw;
mod adamw8bit;
mod blockwise_q8;
mod base;
mod grad_accum;
mod grads;
mod rmsprop;
mod sgd;
mod simple;
mod visitor;

pub use adagrad::*;
pub use adam::*;
pub use adamw::*;
pub use adamw8bit::*;
pub use blockwise_q8::*;
pub use base::*;
pub use grad_accum::*;
pub use grads::*;
pub use rmsprop::*;
pub use sgd::*;
pub use simple::*;
