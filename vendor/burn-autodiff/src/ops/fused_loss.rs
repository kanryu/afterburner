//! Mixed-dtype loss operations used by the training path.

use super::{Backward, Ops, OpsKind};
use crate::{
    checkpoint::base::Checkpointer,
    grads::Gradients,
    tensor::AutodiffTensor,
};
use burn_tensor::{ElementConversion, FloatDType, Shape, TensorMetadata, backend::Backend};

#[derive(Clone, Debug)]
struct L2LossState<B: Backend> {
    diff: B::FloatTensorPrimitive,
    input_shape: Shape,
    input_dtype: FloatDType,
    target_dtype: FloatDType,
}

#[derive(Clone, Copy, Debug)]
struct FusedL2LossBackward;

impl<B: Backend> Backward<B, 2> for FusedL2LossBackward {
    type State = L2LossState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let count = state.input_shape.num_elements() as f32;
        let grad = B::float_expand(B::float_cast(grads.consume::<B>(&ops.node), FloatDType::F32), state.input_shape.clone());
        let base = B::float_mul_scalar(
            B::float_mul(state.diff, grad),
            (2.0 / count).elem(),
        );

        if let Some(parent) = ops.parents[0].as_ref() {
            grads.register::<B>(parent.id, B::float_cast(base.clone(), state.input_dtype));
        }
        if let Some(parent) = ops.parents[1].as_ref() {
            grads.register::<B>(parent.id, B::float_cast(B::float_neg(base), state.target_dtype));
        }
    }
}

/// Computes mean squared error in FP32 while preserving gradients to inputs
/// stored in BF16 or another source dtype.
pub fn fused_l2_loss<B: Backend, C: crate::checkpoint::strategy::CheckpointStrategy>(
    input: AutodiffTensor<B>,
    target: AutodiffTensor<B>,
) -> AutodiffTensor<B> {
    let input_dtype = input.primitive.dtype().into();
    let target_dtype = target.primitive.dtype().into();
    let input_shape = input.primitive.shape();
    // Match PyTorch's autocast loss behavior: promote both operands before
    // subtraction.  Subtracting in BF16 first loses the small LoRA residual
    // that drives the update, especially once the prediction gets close to
    // the target.
    let input_f32 = B::float_cast(input.primitive.clone(), FloatDType::F32);
    let target_f32 = B::float_cast(target.primitive.clone(), FloatDType::F32);
    let diff = B::float_sub(input_f32, target_f32);
    let output = B::float_mean(B::float_mul(diff.clone(), diff.clone()));
    let state = L2LossState {
        diff,
        input_shape,
        input_dtype,
        target_dtype,
    };

    match FusedL2LossBackward
        .prepare::<C>([input.node.clone(), target.node.clone()])
        .compute_bound()
        .parents([&input, &target])
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(state, output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}
