//! Backend-only vector-Jacobian products used by the block manual backward.

use crate::{graph::Requirement, tensor::AutodiffTensor};
use burn_tensor::{ElementConversion, FloatDType, Shape, TensorMetadata, backend::Backend};

/// RMSNorm forward computed entirely on the inner backend.
pub fn rms_norm_forward<B: Backend>(
    input: B::FloatTensorPrimitive,
    gamma: B::FloatTensorPrimitive,
    epsilon: f32,
) -> B::FloatTensorPrimitive {
    let input_dtype: FloatDType = input.dtype().into();
    let shape = input.shape();
    let last_dim = shape.num_dims() - 1;
    let feature_count = (shape[last_dim] as f32).elem();
    let x = B::float_cast(input, FloatDType::F32);
    let rms = B::float_sqrt(B::float_add_scalar(
        B::float_div_scalar(B::float_sum_dim(B::float_mul(x.clone(), x.clone()), last_dim), feature_count),
        epsilon.elem(),
    ));
    let gamma = B::float_cast(gamma, FloatDType::F32);
    let mut gamma_shape = vec![1; shape.num_dims()];
    gamma_shape[last_dim] = shape[last_dim];
    let gamma = B::float_expand(B::float_reshape(gamma, Shape::from(gamma_shape)), shape.clone());
    B::float_cast(B::float_div(B::float_mul(x, gamma), B::float_expand(rms, shape)), input_dtype)
}

/// RMSNorm represented by one outer node and a backend-only VJP.
pub fn rms_norm_manual<B: Backend>(
    input: AutodiffTensor<B>,
    gamma: B::FloatTensorPrimitive,
    epsilon: f32,
) -> AutodiffTensor<B> {
    let input_node = input.node.clone();
    let input_value = input.primitive;
    let gamma_value = gamma;
    let output = rms_norm_forward::<B>(input_value.clone(), gamma_value.clone(), epsilon);
    crate::checkpoint_block::register_manual_vjp::<B, 4>(
        output,
        &[input_node.clone()],
        Box::new(move |grad, grads| {
            if !matches!(input_node.requirement, Requirement::None) {
                grads.register::<B>(
                    input_node.id,
                    rms_norm_backward::<B>(grad, input_value, gamma_value, epsilon),
                );
            }
        }),
    )
}

/// Backward pass for the RMSNorm used by Q/K normalization.
pub fn rms_norm_backward<B: Backend>(
    grad_output: B::FloatTensorPrimitive,
    input: B::FloatTensorPrimitive,
    gamma: B::FloatTensorPrimitive,
    epsilon: f32,
) -> B::FloatTensorPrimitive {
    let input_dtype: FloatDType = input.dtype().into();
    let shape = input.shape();
    let last_dim = shape.num_dims() - 1;
    let feature_count = (shape[last_dim] as f32).elem();
    let x = B::float_cast(input, FloatDType::F32);
    let dy = B::float_cast(grad_output, FloatDType::F32);
    let gamma = B::float_cast(gamma, FloatDType::F32);
    let gamma_shape = Shape::from({
        let mut dims = vec![1; shape.num_dims()];
        dims[last_dim] = shape[last_dim];
        dims
    });
    let gamma = B::float_expand(B::float_reshape(gamma, gamma_shape), shape.clone());
    let rms = B::float_sqrt(B::float_add_scalar(
        B::float_div_scalar(B::float_sum_dim(B::float_mul(x.clone(), x.clone()), last_dim), feature_count),
        epsilon.elem(),
    ));
    let rms_expanded = B::float_expand(rms, shape.clone());
    let x_hat = B::float_div(x, rms_expanded.clone());
    let g = B::float_mul(dy, gamma);
    let dot = B::float_expand(
        B::float_div_scalar(B::float_sum_dim(B::float_mul(g.clone(), x_hat.clone()), last_dim), feature_count),
        shape,
    );
    B::float_cast(
        B::float_div(B::float_sub(g, B::float_mul(x_hat, dot)), rms_expanded),
        input_dtype,
    )
}

/// Saved values needed for the input-side AdaLN VJP. AdaLN uses LayerNorm,
/// unlike the RMSNorm used inside attention.
pub struct AdaLnInputCache<B: Backend> {
    pub normalized: B::FloatTensorPrimitive,
    pub inv_std: B::FloatTensorPrimitive,
    pub scale: B::FloatTensorPrimitive,
}

/// Backpropagates through `y = layer_norm(x) * (1 + scale)` and the residual
/// `z = x + sublayer(y) * gate`. Gradients for the frozen conditioning branch
/// are intentionally not produced.
pub fn adaln_input_backward<B: Backend>(
    grad_residual: B::FloatTensorPrimitive,
    grad_normalized: B::FloatTensorPrimitive,
    cache: AdaLnInputCache<B>,
    residual_dtype: FloatDType,
) -> B::FloatTensorPrimitive {
    let shape = grad_residual.shape();
    let hidden = (shape[shape.num_dims() - 1] as f32).elem();
    let dz = B::float_cast(grad_residual, residual_dtype);
    let dy = B::float_cast(grad_normalized, residual_dtype);
    let normalized = B::float_cast(cache.normalized, residual_dtype);
    let inv_std = B::float_cast(cache.inv_std, residual_dtype);
    let scale = B::float_cast(cache.scale, residual_dtype);
    let scale = B::float_expand(B::float_add_scalar(scale, 1.0.elem()), shape.clone());
    let d_norm = B::float_mul(dy, scale);
    let mean = B::float_expand(
        B::float_div_scalar(B::float_sum_dim(d_norm.clone(), 2), hidden),
        shape.clone(),
    );
    let mean_x = B::float_expand(
        B::float_div_scalar(B::float_sum_dim(B::float_mul(d_norm.clone(), normalized.clone()), 2), hidden),
        shape.clone(),
    );
    let centered = B::float_sub(
        B::float_sub(d_norm, mean),
        B::float_mul(normalized, mean_x),
    );
    B::float_add(
        dz,
        B::float_mul(B::float_expand(inv_std, shape), centered),
    )
}

/// Backend-only GELU VJP using the same backend primitive as Burn Autodiff.
pub fn gelu_backward<B: Backend>(
    input: B::FloatTensorPrimitive,
    grad_output: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    B::gelu_backward(input, grad_output)
}
