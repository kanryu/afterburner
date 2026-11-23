//! Fused scaled-dot-product attention for the autodiff backend.
//!
//! The forward and backward math is executed by the inner backend.  Only one
//! autodiff graph node is created for the complete Q/K/V operation.  This is
//! intentionally separate from a FlashAttention kernel: it is a graph and
//! allocation reduction layer that can be used by every Burn backend first.

use super::{Backward, Ops, OpsKind};
use crate::{
    checkpoint::{
        base::Checkpointer,
        retro_forward::RetroForward,
        state::BackwardStates,
        strategy::CheckpointStrategy,
    },
    grads::Gradients,
    graph::{NodeId, Requirement},
    tensor::AutodiffTensor,
};
use burn_tensor::{FloatDType, TensorMetadata, backend::Backend};
use std::time::Instant;

#[derive(Clone, Debug)]
struct AttentionState<B: Backend> {
    query_id: Option<NodeId>,
    key_id: Option<NodeId>,
    value_id: Option<NodeId>,
    mask: Option<B::FloatTensorPrimitive>,
    query_chunk_size: usize,
    scale: B::FloatElem,
}

#[derive(Clone, Copy, Debug)]
struct FusedAttentionBackward;

#[derive(Clone, Debug)]
struct RetroFusedAttention<B: Backend> {
    query_id: NodeId,
    key_id: NodeId,
    value_id: NodeId,
    mask: Option<B::FloatTensorPrimitive>,
    query_chunk_size: usize,
    scale: B::FloatElem,
}

impl<B: Backend> RetroForward for RetroFusedAttention<B> {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let query = states.get_state::<B::FloatTensorPrimitive>(&self.query_id);
        let key = states.get_state::<B::FloatTensorPrimitive>(&self.key_id);
        let value = states.get_state::<B::FloatTensorPrimitive>(&self.value_id);
        let output = attention_forward::<B>(
            query,
            key,
            value,
            self.mask.clone(),
            self.scale,
            self.query_chunk_size,
        );
        states.save(out_node, output);
    }
}

fn softmax<B: Backend>(
    query: B::FloatTensorPrimitive,
    key: B::FloatTensorPrimitive,
    value: B::FloatTensorPrimitive,
    mask: Option<B::FloatTensorPrimitive>,
    scale: B::FloatElem,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    // Python autocast keeps Q/K/V in one activation dtype. Some Rust paths
    // reach attention from an FP32 embedding while the context arrives in the
    // backend BF16 dtype, so normalize the operands before QK matmul.
    let attention_dtype = query.dtype();
    let key = B::float_cast(key, attention_dtype.into());
    let value = B::float_cast(value, attention_dtype.into());
    let mut scores = B::float_mul_scalar(
        B::float_matmul(query, B::float_swap_dims(key, 2, 3)),
        scale,
    );
    let score_shape = scores.shape();
    if let Some(mask) = mask {
        // The cached attention mask uses the backend's default dtype, while
        // Python's autocast path adds it after QK has selected the activation
        // dtype. Match the score dtype before the elementwise add.
        let mask = B::float_cast(mask, scores.dtype().into());
        scores = B::float_add(scores, B::float_expand(mask, score_shape.clone()));
    }
    let max = B::float_max_dim(scores.clone(), 3);
    let max = B::float_expand(max, score_shape.clone());
    let weights = B::float_exp(B::float_sub(scores, max));
    let denominator = B::float_sum_dim(weights.clone(), 3);
    let denominator = B::float_expand(denominator, [
        score_shape[0], score_shape[1], score_shape[2], 1,
    ].into());
    // Return the normalized probabilities as well as the output.  The old
    // implementation normalized only the output and returned the raw exp
    // scores to backward.  That made dV/dQ/dK grow with sequence length and
    // eventually overflow BF16 in the early DiT blocks.
    let weights = B::float_div(weights, denominator);
    let output = B::float_matmul(weights.clone(), value);
    (output, weights)
}

fn query_slice<B: Backend>(
    query: B::FloatTensorPrimitive,
    start: usize,
    end: usize,
) -> B::FloatTensorPrimitive {
    let shape = query.shape();
    B::float_slice(
        query,
        &[
            burn_tensor::Slice::new(0, Some(shape[0] as isize), 1),
            burn_tensor::Slice::new(0, Some(shape[1] as isize), 1),
            burn_tensor::Slice::new(start as isize, Some(end as isize), 1),
            burn_tensor::Slice::new(0, Some(shape[3] as isize), 1),
        ],
    )
}

pub fn attention_forward<B: Backend>(
    query: B::FloatTensorPrimitive,
    key: B::FloatTensorPrimitive,
    value: B::FloatTensorPrimitive,
    mask: Option<B::FloatTensorPrimitive>,
    scale: B::FloatElem,
    query_chunk_size: usize,
) -> B::FloatTensorPrimitive {
    let query_length = query.shape()[2];
    let chunk_size = query_chunk_size.max(1).min(query_length.max(1));
    if chunk_size >= query_length {
        return softmax::<B>(query, key, value, mask, scale).0;
    }
    let mut outputs = Vec::new();
    for start in (0..query_length).step_by(chunk_size) {
        let end = (start + chunk_size).min(query_length);
        let q = query_slice::<B>(query.clone(), start, end);
        outputs.push(softmax::<B>(q, key.clone(), value.clone(), mask.clone(), scale).0);
    }
    B::float_cat(outputs, 2)
}

/// Backend-only scaled-dot-product attention VJP.
///
/// This path contains no graph or checkpoint objects and is intended for the
/// block-level manual backward implementation.
pub fn attention_backward<B: Backend>(
    grad_output: B::FloatTensorPrimitive,
    query: B::FloatTensorPrimitive,
    key: B::FloatTensorPrimitive,
    value: B::FloatTensorPrimitive,
    mask: Option<B::FloatTensorPrimitive>,
    scale: B::FloatElem,
    query_chunk_size: usize,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    let query_dtype: FloatDType = query.dtype().into();
    let key_dtype: FloatDType = key.dtype().into();
    let value_dtype: FloatDType = value.dtype().into();
    let backward_dtype = if std::env::var_os("AFTERBURNER_ATTENTION_BACKWARD_F32").is_some() {
        FloatDType::F32
    } else {
        query_dtype
    };
    let key = B::float_cast(key, backward_dtype);
    let value = B::float_cast(value, backward_dtype);
    let mask = mask.map(|mask| B::float_cast(mask, backward_dtype));
    let query_length = query.shape()[2];
    let chunk_size = query_chunk_size.max(1).min(query_length.max(1));
    let mut d_queries = Vec::new();
    let mut d_key = None;
    let mut d_value = None;
    for start in (0..query_length).step_by(chunk_size) {
        let end = (start + chunk_size).min(query_length);
        let q = B::float_cast(query_slice::<B>(query.clone(), start, end), backward_dtype);
        let go = B::float_cast(query_slice::<B>(grad_output.clone(), start, end), backward_dtype);
        let (_, weights) = softmax::<B>(q.clone(), key.clone(), value.clone(), mask.clone(), scale);
        let dv = B::float_matmul(B::float_swap_dims(weights.clone(), 2, 3), go.clone());
        let dw = B::float_matmul(go, B::float_swap_dims(value.clone(), 2, 3));
        let shape = weights.shape();
        let row = B::float_expand(
            B::float_sum_dim(B::float_mul(dw.clone(), weights.clone()), 3),
            [shape[0], shape[1], shape[2], 1].into(),
        );
        let ds = B::float_mul(weights, B::float_sub(dw, row));
        let dq = B::float_mul_scalar(B::float_matmul(ds.clone(), key.clone()), scale);
        let dk = B::float_mul_scalar(
            B::float_matmul(B::float_swap_dims(ds, 2, 3), q),
            scale,
        );
        d_queries.push(dq);
        d_key = Some(match d_key {
            Some(value) => B::float_add(value, dk),
            None => dk,
        });
        d_value = Some(match d_value {
            Some(value) => B::float_add(value, dv),
            None => dv,
        });
    }
    (
        B::float_cast(B::float_cat(d_queries, 2), query_dtype),
        B::float_cast(d_key.expect("attention produced no key gradient"), key_dtype),
        B::float_cast(d_value.expect("attention produced no value gradient"), value_dtype),
    )
}

impl<B: Backend> Backward<B, 3> for FusedAttentionBackward {
    type State = AttentionState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let debug_backward = std::env::var_os("AFTERBURNER_DEBUG_ATTENTION_BACKWARD").is_some();
        let started = Instant::now();
        if debug_backward {
            eprintln!(
                "[fused-attention] backward_begin chunk={} query_node={:?}",
                ops.state.query_chunk_size,
                ops.state.query_id
            );
        }
        let grad_output = grads.consume::<B>(&ops.node);
        let AttentionState {
            query_id,
            key_id,
            value_id,
            mask,
            query_chunk_size,
            scale,
        } = ops.state;
        let query_node = query_id.expect("fused attention query checkpoint is missing");
        let key_node = key_id.expect("fused attention key checkpoint is missing");
        let value_node = value_id.expect("fused attention value checkpoint is missing");
        let query = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(query_node);
        let key = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(key_node);
        let value = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(value_node);
        let (d_query, d_key, d_value) = attention_backward::<B>(
            grad_output,
            query,
            key,
            value,
            mask,
            scale,
            query_chunk_size,
        );
        if let Some(node) = ops.parents[0].as_ref() {
            grads.register::<B>(node.id, d_query);
        }
        if let Some(node) = ops.parents[1].as_ref() {
            grads.register::<B>(node.id, d_key);
        }
        if let Some(node) = ops.parents[2].as_ref() {
            grads.register::<B>(node.id, d_value);
        }
        if debug_backward {
            eprintln!(
                "[fused-attention] backward_complete elapsed_ms={}",
                started.elapsed().as_millis()
            );
        }
        // The direct VJP above is the sole backward implementation.
    }
}

/// Compute scaled-dot-product attention as one autodiff operation.
///
/// Inputs use `[batch, heads, sequence, head_size]`. The optional additive mask
/// is broadcast to the score shape and is treated as a non-trainable input,
/// matching the fixed attention masks used by the cached training pipeline.
pub fn fused_attention<B: Backend, C: CheckpointStrategy>(
    query: AutodiffTensor<B>,
    key: AutodiffTensor<B>,
    value: AutodiffTensor<B>,
    scale: B::FloatElem,
) -> AutodiffTensor<B> {
    fused_attention_with_mask_and_chunk::<B, C>(query, key, value, None, scale, usize::MAX)
}

/// Computes scaled dot-product attention with an optional additive mask.
///
/// The operation is represented by one autodiff node and recomputes its
/// chunked forward during backward, so the score matrix is not retained.
pub fn fused_attention_with_mask<B: Backend, C: CheckpointStrategy>(
    query: AutodiffTensor<B>,
    key: AutodiffTensor<B>,
    value: AutodiffTensor<B>,
    mask: Option<AutodiffTensor<B>>,
    scale: B::FloatElem,
) -> AutodiffTensor<B> {
    fused_attention_with_mask_and_chunk::<B, C>(query, key, value, mask, scale, usize::MAX)
}

/// Computes masked scaled dot-product attention with an explicit query chunk.
///
/// Query chunks bound the temporary score and probability tensors while the
/// custom backward reconstructs each chunk as needed.
pub fn fused_attention_with_mask_and_chunk<B: Backend, C: CheckpointStrategy>(
    query: AutodiffTensor<B>,
    key: AutodiffTensor<B>,
    value: AutodiffTensor<B>,
    mask: Option<AutodiffTensor<B>>,
    scale: B::FloatElem,
    query_chunk_size: usize,
) -> AutodiffTensor<B> {
    let output = attention_forward::<B>(
        query.primitive.clone(),
        key.primitive.clone(),
        value.primitive.clone(),
        mask.as_ref().map(|value| value.primitive.clone()),
        scale,
        query_chunk_size,
    );

    // Checkpointing is performed at the DiT-block boundary.  Making every
    // attention node memory-bound creates nested recomputation inside the
    // block checkpoint and retains a very large autodiff tape.  The Python
    // trainer recomputes a complete block, not each attention chunk, so keep
    // this primitive compute-bound and retain only the Q/K/V values required
    // by its backward operation.
    match FusedAttentionBackward
        .prepare::<C>([query.node.clone(), key.node.clone(), value.node.clone()])
        .compute_bound()
        .parents([&query, &key, &value])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            // Register the inputs through the checkpoint strategy.  For the
            // recomputable projection path this creates a RetroForward action
            // rather than retaining the completed Q/K/V buffer; compute-bound
            // fallback paths still receive the explicit value they require.
            let query_id = prep.checkpoint(&query);
            let key_id = prep.checkpoint(&key);
            let value_id = prep.checkpoint(&value);
            prep.finish(
                AttentionState {
                    query_id: Some(query_id),
                    key_id: Some(key_id),
                    value_id: Some(value_id),
                    mask: mask.as_ref().map(|value| value.primitive.clone()),
                    query_chunk_size,
                    scale,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

/// Attention operation backed by a direct InnerBackend VJP.
///
/// The forward pass creates one outer node, but never builds an inner
/// Autodiff tape. Backward reconstructs the softmax tiles with
/// `attention_backward` and registers gradients directly into the outer
/// graph. This is the same boundary used by the block-level checkpoint path.
pub fn fused_attention_manual<B: Backend>(
    query: AutodiffTensor<B>,
    key: AutodiffTensor<B>,
    value: AutodiffTensor<B>,
    mask: Option<AutodiffTensor<B>>,
    scale: B::FloatElem,
    query_chunk_size: usize,
) -> AutodiffTensor<B> {
    let output = attention_forward::<B>(
        query.primitive.clone(),
        key.primitive.clone(),
        value.primitive.clone(),
        mask.as_ref().map(|m| m.primitive.clone()),
        scale,
        query_chunk_size,
    );
    let query_node = query.node.clone();
    let key_node = key.node.clone();
    let value_node = value.node.clone();
    let query_value = query.primitive;
    let key_value = key.primitive;
    let value_value = value.primitive;
    let mask_value = mask.map(|m| m.primitive);
    let parents = vec![query_node.clone(), key_node.clone(), value_node.clone()];
    crate::checkpoint_block::register_manual_vjp::<B, 4>(
        output,
        &parents,
        Box::new(move |grad, grads| {
            let (d_query, d_key, d_value) = attention_backward::<B>(
                grad,
                query_value,
                key_value,
                value_value,
                mask_value,
                scale,
                query_chunk_size,
            );
            if !matches!(query_node.requirement, Requirement::None) {
                grads.register::<B>(query_node.id, d_query);
            }
            if !matches!(key_node.requirement, Requirement::None) {
                grads.register::<B>(key_node.id, d_key);
            }
            if !matches!(value_node.requirement, Requirement::None) {
                grads.register::<B>(value_node.id, d_value);
            }
        }),
    )
}
