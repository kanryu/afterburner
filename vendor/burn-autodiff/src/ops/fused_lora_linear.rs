//! Fused base-linear plus LoRA adapter operation.

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
use burn_tensor::{FloatDType, Shape, TensorMetadata, backend::Backend};

const LORA_BACKWARD_ROW_CHUNK: usize = 256;

/// Gradients produced by the backend-only LoRA VJP.
pub struct LoraLinearGrads<B: Backend> {
    pub d_input: Option<B::FloatTensorPrimitive>,
    pub d_down: B::FloatTensorPrimitive,
    pub d_up: B::FloatTensorPrimitive,
}

/// Computes the VJP of `x W^T + scale * (x A^T) B^T` without creating an
/// Autodiff graph. This is shared by the normal fused operation and the
/// block-level manual backward path.
#[allow(clippy::too_many_arguments)]
pub fn lora_linear_backward<B: Backend>(
    grad: B::FloatTensorPrimitive,
    input_value: B::FloatTensorPrimitive,
    weight_value: B::FloatTensorPrimitive,
    down_value: B::FloatTensorPrimitive,
    up_value: B::FloatTensorPrimitive,
    scale: B::FloatElem,
    need_dx: bool,
) -> LoraLinearGrads<B> {
    let input_shape = input_value.shape();
    let input_dtype: FloatDType = input_value.dtype().into();
    let down_dtype: FloatDType = down_value.dtype().into();
    let up_dtype: FloatDType = up_value.dtype().into();
    let backward_dtype = if std::env::var_os("AFTERBURNER_LORA_BACKWARD_F32").is_some()
        || matches!(input_dtype, FloatDType::F32)
    {
        FloatDType::F32
    } else {
        input_dtype
    };

    let grad = B::float_cast(grad, backward_dtype);
    let input = B::float_cast(input_value, backward_dtype);
    let weight = B::float_cast(weight_value, backward_dtype);
    let down = B::float_cast(down_value, backward_dtype);
    let up = B::float_cast(up_value, backward_dtype);
    let grad_shape = grad.shape();
    let grad = if grad_shape.num_dims() == 3 {
        B::float_reshape(grad, Shape::from([grad_shape[0] * grad_shape[1], grad_shape[2]]))
    } else {
        grad
    };
    let input = if input_shape.num_dims() == 3 {
        B::float_reshape(input, Shape::from([input_shape[0] * input_shape[1], input_shape[2]]))
    } else {
        input
    };

    let d_input = if need_dx {
        let low_grad = B::float_matmul(grad.clone(), up.clone());
        let value = chunked_input_gradient::<B>(
            grad.clone(), weight, low_grad, down.clone(), scale,
        );
        let value = if input_shape.num_dims() == 3 {
            B::float_reshape(value, input_shape.clone())
        } else {
            value
        };
        Some(B::float_cast(value, input_dtype))
    } else {
        None
    };

    let low = B::float_matmul(input.clone(), B::float_transpose(down));
    let d_up = B::float_cast(
        B::float_mul_scalar(
            B::float_transpose(chunked_factor_gradient_f32::<B>(low, grad.clone())),
            scale,
        ),
        up_dtype,
    );
    let d_down = B::float_cast(
        B::float_mul_scalar(
            chunked_factor_gradient_f32::<B>(B::float_matmul(grad, up), input),
            scale,
        ),
        down_dtype,
    );

    LoraLinearGrads { d_input, d_down, d_up }
}

#[derive(Clone, Debug)]
struct LoraState<B: Backend> {
    input: Option<NodeId>,
    input_value: Option<B::FloatTensorPrimitive>,
    down: Option<NodeId>,
    up: Option<NodeId>,
    weight: Option<NodeId>,
    weight_value: Option<B::FloatTensorPrimitive>,
    down_value: Option<B::FloatTensorPrimitive>,
    up_value: Option<B::FloatTensorPrimitive>,
    input_shape: Shape,
    scale: B::FloatElem,
}

fn forward<B: Backend>(
    input: B::FloatTensorPrimitive,
    weight: B::FloatTensorPrimitive,
    down: B::FloatTensorPrimitive,
    up: B::FloatTensorPrimitive,
    scale: B::FloatElem,
) -> B::FloatTensorPrimitive {
    // Match torch autocast in kohya: the projection is accumulated in the
    // surrounding activation dtype (BF16 for the normal CUDA training path),
    // while the backward matrix products below are promoted to F32.  Doing
    // the forward in F32 changes the model prediction before any optimizer
    // update and makes the Python/Rust step comparison invalid.
    let output_dtype = input.dtype();
    let input = B::float_cast(input, output_dtype.into());
    let weight = B::float_cast(weight, output_dtype.into());
    let down = B::float_cast(down, output_dtype.into());
    let up = B::float_cast(up, output_dtype.into());
    let input_shape = input.shape();
    let output_dim = up.shape()[0];
    let input = if input_shape.num_dims() == 3 {
        let dims = [input_shape[0] * input_shape[1], input_shape[2]];
        B::float_reshape(input, Shape::from(dims))
    } else {
        input
    };
    let base = B::float_matmul(input.clone(), B::float_transpose(weight));
    let low = B::float_matmul(input, B::float_transpose(down));
    let delta = B::float_mul_scalar(
        B::float_matmul(low, B::float_transpose(up)),
        scale,
    );
    let output = B::float_add(base, delta);
    if input_shape.num_dims() == 3 {
        let shape = Shape::from([input_shape[0], input_shape[1], output_dim]);
        B::float_reshape(output, shape)
    } else {
        output
    }
}

/// Backend-only LoRA forward used by block-level manual VJPs.
pub fn lora_linear_forward_inner<B: Backend>(
    input: B::FloatTensorPrimitive,
    weight: B::FloatTensorPrimitive,
    down: B::FloatTensorPrimitive,
    up: B::FloatTensorPrimitive,
    scale: B::FloatElem,
) -> B::FloatTensorPrimitive {
    forward::<B>(input, weight, down, up, scale)
}

fn row_slice<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    start: usize,
    end: usize,
) -> B::FloatTensorPrimitive {
    let shape = tensor.shape();
    B::float_slice(
        tensor,
        &[
            burn_tensor::Slice::new(start as isize, Some(end as isize), 1),
            burn_tensor::Slice::new(0, Some(shape[1] as isize), 1),
        ],
    )
}

fn chunked_input_gradient<B: Backend>(
    grad: B::FloatTensorPrimitive,
    weight: B::FloatTensorPrimitive,
    low_grad: B::FloatTensorPrimitive,
    down: B::FloatTensorPrimitive,
    scale: B::FloatElem,
) -> B::FloatTensorPrimitive {
    let rows = grad.shape()[0];
    let mut chunks = Vec::new();
    for start in (0..rows).step_by(LORA_BACKWARD_ROW_CHUNK) {
        let end = (start + LORA_BACKWARD_ROW_CHUNK).min(rows);
        let grad_chunk = row_slice::<B>(grad.clone(), start, end);
        let low_chunk = row_slice::<B>(low_grad.clone(), start, end);
        let base = B::float_matmul(grad_chunk, weight.clone());
        let adapter = B::float_mul_scalar(
            B::float_matmul(low_chunk, down.clone()),
            scale,
        );
        chunks.push(B::float_add(base, adapter));
    }
    B::float_cat(chunks, 0)
}

fn chunked_factor_gradient<B: Backend>(
    left: B::FloatTensorPrimitive,
    right: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let rows = left.shape()[0];
    let mut accumulated = None;
    for start in (0..rows).step_by(LORA_BACKWARD_ROW_CHUNK) {
        let end = (start + LORA_BACKWARD_ROW_CHUNK).min(rows);
        let partial = B::float_matmul(
            B::float_transpose(row_slice::<B>(left.clone(), start, end)),
            row_slice::<B>(right.clone(), start, end),
        );
        accumulated = Some(match accumulated {
            Some(value) => B::float_add(value, partial),
            None => partial,
        });
    }
    accumulated.expect("fused LoRA backward received an empty row dimension")
}

/// Accumulates a LoRA factor gradient in F32 without materializing full-size
/// F32 activation copies. Only one row chunk is promoted at a time; the
/// accumulator itself is the small rank-by-hidden/output factor.
fn chunked_factor_gradient_f32<B: Backend>(
    left: B::FloatTensorPrimitive,
    right: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let rows = left.shape()[0];
    let mut accumulated = None;
    for start in (0..rows).step_by(LORA_BACKWARD_ROW_CHUNK) {
        let end = (start + LORA_BACKWARD_ROW_CHUNK).min(rows);
        let left_chunk = B::float_cast(
            row_slice::<B>(left.clone(), start, end),
            FloatDType::F32,
        );
        let right_chunk = B::float_cast(
            row_slice::<B>(right.clone(), start, end),
            FloatDType::F32,
        );
        let partial = B::float_matmul(B::float_transpose(left_chunk), right_chunk);
        accumulated = Some(match accumulated {
            Some(value) => B::float_add(value, partial),
            None => partial,
        });
    }
    accumulated.expect("fused LoRA backward received an empty row dimension")
}

#[derive(Clone, Copy, Debug)]
struct FusedLoraBackward;

#[derive(Clone, Debug)]
struct RetroFusedLora<B: Backend> {
    input: NodeId,
    weight: NodeId,
    down: NodeId,
    up: NodeId,
    scale: B::FloatElem,
}

impl<B: Backend> RetroForward for RetroFusedLora<B> {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let output = forward::<B>(
            states.get_state::<B::FloatTensorPrimitive>(&self.input),
            states.get_state::<B::FloatTensorPrimitive>(&self.weight),
            states.get_state::<B::FloatTensorPrimitive>(&self.down),
            states.get_state::<B::FloatTensorPrimitive>(&self.up),
            self.scale,
        );
        states.save(out_node, output);
    }
}

impl<B: Backend> Backward<B, 4> for FusedLoraBackward {
    type State = LoraState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let input_value = state.input_value.unwrap_or_else(|| {
            let input_id = state
                .input
                .expect("fused LoRA input checkpoint is missing");
            checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(input_id)
        });
        let down = state
            .down
            .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id));
        let up = state
            .up
            .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id));
        let down_value = down
            .or(state.down_value)
            .expect("fused LoRA down state is missing");
        let up_value = up
            .or(state.up_value)
            .expect("fused LoRA up state is missing");
        let weight_value = state
            .weight
            .map(|id| checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(id))
            .or(state.weight_value)
            .expect("fused LoRA weight state is missing");
        let scale = state.scale;

        // Training freezes the base projection weight. Use the backend-only
        // VJP for this common path so the same implementation is exercised by
        // the upcoming block-level manual backward.
        if ops.parents[1].is_none() {
            let manual = lora_linear_backward::<B>(
                grad,
                input_value,
                weight_value,
                down_value,
                up_value,
                scale,
                ops.parents[0].is_some(),
            );
            if let Some(node) = ops.parents[0].as_ref() {
                grads.register::<B>(
                    node.id,
                    manual.d_input.expect("LoRA input gradient was not produced"),
                );
            }
            if let Some(node) = ops.parents[2].as_ref() {
                grads.register::<B>(node.id, manual.d_down);
            }
            if let Some(node) = ops.parents[3].as_ref() {
                grads.register::<B>(node.id, manual.d_up);
            }
            return;
        }
        // The output gradient is commonly BF16, while LoRA factors are kept
        // in F32 (matching kohya's autocast behavior).  Keep the destination
        // dtype per parent instead of using the output gradient dtype for all
        // four parents; otherwise the LoRA gradients lose precision before
        // they reach the F32 optimizer state.
        let input_gradient_dtype: FloatDType = input_value.dtype().into();
        let down_gradient_dtype: FloatDType = down_value.dtype().into();
        let up_gradient_dtype: FloatDType = up_value.dtype().into();
        let weight_gradient_dtype: FloatDType = weight_value.dtype().into();
        // The low-VRAM path keeps the backward workspace in the model dtype.
        // Previously every large activation, base weight, LoRA factor, and
        // output gradient was unconditionally duplicated as F32 here.  At
        // Anima's sequence and hidden sizes those copies dominate the CUDA
        // pool peak even when the model and LoRA tensors are BF16.
        //
        // Preserve the old F32 behavior when any participating tensor is F32,
        // or when explicitly requested for numerical diagnostics.  The normal
        // low-VRAM run has all participating tensors in BF16 and therefore
        // uses a half-sized backward workspace.
        let backward_dtype = if std::env::var_os("AFTERBURNER_LORA_BACKWARD_F32").is_some()
            || matches!(input_gradient_dtype, FloatDType::F32)
        {
            FloatDType::F32
        } else {
            input_gradient_dtype
        };
        let grad = B::float_cast(grad, backward_dtype);
        let input_value = B::float_cast(input_value, backward_dtype);
        let down_value = B::float_cast(down_value, backward_dtype);
        let up_value = B::float_cast(up_value, backward_dtype);
        let weight_value = B::float_cast(weight_value, backward_dtype);
        let grad_shape = grad.shape();
        let grad = if grad_shape.num_dims() == 3 {
            B::float_reshape(
                grad,
                Shape::from([grad_shape[0] * grad_shape[1], grad_shape[2]]),
            )
        } else {
            grad
        };
        let input_value = if input_value.shape().num_dims() == 3 {
            let shape = input_value.shape();
            B::float_reshape(
                input_value,
                Shape::from([shape[0] * shape[1], shape[2]]),
            )
        } else {
            input_value
        };

        if let Some(node) = ops.parents[0].as_ref() {
            let low_grad = B::float_matmul(grad.clone(), up_value.clone());
            let input_grad = chunked_input_gradient::<B>(
                grad.clone(),
                weight_value,
                low_grad,
                down_value.clone(),
                scale,
            );
            let input_grad = if state.input_shape.num_dims() == 3 {
                B::float_reshape(input_grad, state.input_shape.clone())
            } else {
                input_grad
            };
            grads.register::<B>(node.id, B::float_cast(input_grad, input_gradient_dtype));
        }
        if let Some(node) = ops.parents[3].as_ref() {
            let low = B::float_matmul(input_value.clone(), B::float_transpose(down_value));
            let grad_up = B::float_transpose(chunked_factor_gradient_f32::<B>(low, grad.clone()));
            grads.register::<B>(
                node.id,
                B::float_cast(B::float_mul_scalar(grad_up, scale), up_gradient_dtype),
            );
        }
        if let Some(node) = ops.parents[2].as_ref() {
            let grad_low = B::float_matmul(grad.clone(), up_value);
            let grad_down = chunked_factor_gradient_f32::<B>(grad_low, input_value.clone());
            grads.register::<B>(
                node.id,
                B::float_cast(B::float_mul_scalar(grad_down, scale), down_gradient_dtype),
            );
        }
        if let Some(node) = ops.parents[1].as_ref() {
            let grad_weight = B::float_matmul(B::float_transpose(grad), input_value);
            grads.register::<B>(node.id, B::float_cast(grad_weight, weight_gradient_dtype));
        }
    }
}

/// Applies a base linear layer and a LoRA update as one autodiff operation.
pub fn fused_lora_linear<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>,
    weight: AutodiffTensor<B>,
    down: AutodiffTensor<B>,
    up: AutodiffTensor<B>,
    scale: B::FloatElem,
) -> AutodiffTensor<B> {
    let tracked = [input.is_tracked(), weight.is_tracked(), down.is_tracked(), up.is_tracked()];
    let mut prep = FusedLoraBackward
        .prepare::<C>([
            input.node.clone(),
            weight.node.clone(),
            down.node.clone(),
            up.node.clone(),
        ])
        .compute_bound()
        .parents([&input, &weight, &down, &up]);
    let input_checkpoint = if up.shape()[0] <= 64 {
        Some(prep.checkpoint_computed(&input))
    } else {
        None
    };
    match prep.stateful() {
        OpsKind::Tracked(mut prep) => {
            let input_id = input_checkpoint.or_else(|| {
                (tracked[0] || tracked[1] || tracked[2] || tracked[3])
                    .then(|| prep.checkpoint(&input))
            });
            prep.finish(
                LoraState {
                    input: input_id,
                    input_value: None,
                    down: None,
                    up: None,
                    weight: None,
                    weight_value: Some(weight.primitive.clone()),
                    down_value: Some(down.primitive.clone()),
                    up_value: Some(up.primitive.clone()),
                    input_shape: input.primitive.shape(),
                    scale,
                },
                forward::<B>(input.primitive, weight.primitive, down.primitive, up.primitive, scale),
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(forward::<B>(
            input.primitive,
            weight.primitive,
            down.primitive,
            up.primitive,
            scale,
        )),
    }
}

/// LoRA linear operation backed by a manual InnerBackend VJP.
pub fn fused_lora_linear_manual<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>,
    weight: AutodiffTensor<B>,
    down: AutodiffTensor<B>,
    up: AutodiffTensor<B>,
    scale: B::FloatElem,
) -> AutodiffTensor<B> {
    if weight.is_tracked() {
        return fused_lora_linear::<B, C>(input, weight, down, up, scale);
    }
    let output = forward::<B>(
        input.primitive.clone(),
        weight.primitive.clone(),
        down.primitive.clone(),
        up.primitive.clone(),
        scale,
    );
    let parents = vec![input.node.clone(), down.node.clone(), up.node.clone()];
    let input_value = input.primitive;
    let weight_value = weight.primitive;
    let down_value = down.primitive;
    let up_value = up.primitive;
    let input_id = input.node.id;
    let down_id = down.node.id;
    let up_id = up.node.id;
    crate::checkpoint_block::register_manual_vjp::<B, 3>(
        output,
        &parents,
        Box::new(move |grad, grads| {
            let result = lora_linear_backward::<B>(
                grad,
                input_value,
                weight_value,
                down_value,
                up_value,
                scale,
                true,
            );
            grads.register::<B>(input_id, result.d_input.expect("manual LoRA input gradient missing"));
            grads.register::<B>(down_id, result.d_down);
            grads.register::<B>(up_id, result.d_up);
        }),
    )
}

/// LoRA linear operation whose output is recomputed when a memory-bound
/// consumer (the fused attention primitive) requests it during backward.
/// This avoids retaining Q/K/V projection outputs for every DiT block.
pub fn fused_lora_linear_recomputable<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>,
    weight: AutodiffTensor<B>,
    down: AutodiffTensor<B>,
    up: AutodiffTensor<B>,
    scale: B::FloatElem,
) -> AutodiffTensor<B> {
    let prep = FusedLoraBackward
        .prepare::<C>([
            input.node.clone(),
            weight.node.clone(),
            down.node.clone(),
            up.node.clone(),
        ])
        .memory_bound()
        .retro_forward(RetroFusedLora::<B> {
            input: input.node.id,
            weight: weight.node.id,
            down: down.node.id,
            up: up.node.id,
            scale,
        })
        .parents([&input, &weight, &down, &up]);
    match prep.stateful() {
        OpsKind::Tracked(mut prep) => {
            let input_id = prep.checkpoint(&input);
            let weight_id = prep.checkpoint(&weight);
            let down_id = prep.checkpoint(&down);
            let up_id = prep.checkpoint(&up);
            prep.finish(
                LoraState {
                    input: Some(input_id),
                    input_value: None,
                    down: Some(down_id),
                    up: Some(up_id),
                    weight: Some(weight_id),
                    weight_value: None,
                    down_value: None,
                    up_value: None,
                    input_shape: input.primitive.shape(),
                    scale,
                },
                forward::<B>(input.primitive, weight.primitive, down.primitive, up.primitive, scale),
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(forward::<B>(
            input.primitive,
            weight.primitive,
            down.primitive,
            up.primitive,
            scale,
        )),
    }
}
