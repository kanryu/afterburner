//! Direct `InnerBackend` manual VJP for Anima DiT blocks.

use burn::tensor::{
    ElementConversion, FloatDType, Shape, Slice, Tensor, TensorData, TensorMetadata,
    TensorPrimitive, backend::Backend,
};
use burn_autodiff::{
    Autodiff, AutodiffTensor, NodeId, NodeRef,
    checkpoint::strategy::CheckpointStrategy,
    checkpoint_block::register_manual_vjp,
    grads::Gradients,
    ops::{
        attention_backward, attention_forward, lora_linear_backward, lora_linear_forward_inner,
        LoraLinearGrads,
    },
};

use crate::{
    dit::{
        AdaLnModulation, AnimaAttentionCuda, AnimaDiTBlock, AnimaRmsNorm,
        attention_query_chunk_size,
    },
    lora::LoraLinear,
};

#[derive(Clone)]
struct InnerLoraWeights<B: Backend> {
    weight: B::FloatTensorPrimitive,
    down: B::FloatTensorPrimitive,
    up: B::FloatTensorPrimitive,
    down_node: NodeRef,
    up_node: NodeRef,
    scale: B::FloatElem,
}

#[derive(Clone)]
struct InnerAdaLnWeights<B: Backend> {
    down: InnerLoraWeights<B>,
    up: InnerLoraWeights<B>,
    hidden_size: usize,
}

#[derive(Clone)]
struct InnerAttentionWeights<B: Backend> {
    q_proj: InnerLoraWeights<B>,
    k_proj: InnerLoraWeights<B>,
    v_proj: InnerLoraWeights<B>,
    output_proj: InnerLoraWeights<B>,
    q_gamma: B::FloatTensorPrimitive,
    k_gamma: B::FloatTensorPrimitive,
    rms_eps: f32,
    use_rotary: bool,
    head_count: usize,
    head_size: usize,
}

#[derive(Clone)]
struct InnerDiTBlockWeights<B: Backend> {
    self_attn: InnerAttentionWeights<B>,
    cross_attn: InnerAttentionWeights<B>,
    mlp_layer1: InnerLoraWeights<B>,
    mlp_layer2: InnerLoraWeights<B>,
    self_attn_mod: InnerAdaLnWeights<B>,
    cross_attn_mod: InnerAdaLnWeights<B>,
    mlp_mod: InnerAdaLnWeights<B>,
}

struct AdaLnForwardCache<B: Backend> {
    normalized_act: B::FloatTensorPrimitive,
    x_hat: B::FloatTensorPrimitive,
    inv_std: B::FloatTensorPrimitive,
    scale: B::FloatTensorPrimitive,
    gate: B::FloatTensorPrimitive,
}

struct BlockRecomputeCache<B: Backend> {
    self_mod: AdaLnForwardCache<B>,
    q_self_pre: B::FloatTensorPrimitive,
    k_self_pre: B::FloatTensorPrimitive,
    q_self_normed: B::FloatTensorPrimitive,
    k_self_normed: B::FloatTensorPrimitive,
    q_self_rot: B::FloatTensorPrimitive,
    k_self_rot: B::FloatTensorPrimitive,
    v_self: B::FloatTensorPrimitive,
    self_attended_2d: B::FloatTensorPrimitive,
    cross_mod: AdaLnForwardCache<B>,
    q_cross_pre: B::FloatTensorPrimitive,
    k_cross_pre: B::FloatTensorPrimitive,
    q_cross_normed: B::FloatTensorPrimitive,
    k_cross_normed: B::FloatTensorPrimitive,
    v_cross: B::FloatTensorPrimitive,
    cross_attended_2d: B::FloatTensorPrimitive,
    mlp_mod: AdaLnForwardCache<B>,
    mlp_h1: B::FloatTensorPrimitive,
    mlp_h2: B::FloatTensorPrimitive,
}

fn unwrap_autodiff<B: Backend>(tensor: AutodiffTensor<B>) -> (B::FloatTensorPrimitive, NodeRef) {
    (tensor.primitive, tensor.node)
}

fn extract_tensor_inner<B, C, const D: usize>(
    tensor: Tensor<Autodiff<B, C>, D>,
) -> (B::FloatTensorPrimitive, NodeRef)
where
    B: Backend,
    C: CheckpointStrategy,
{
    match tensor.into_primitive() {
        TensorPrimitive::Float(value) => unwrap_autodiff(value),
        TensorPrimitive::QFloat(_) => panic!("quantized tensors are unsupported in DiT VJP"),
    }
}

fn extract_lora_weights<B, C>(layer: &LoraLinear<Autodiff<B, C>>) -> InnerLoraWeights<B>
where
    B: Backend,
    C: CheckpointStrategy,
{
    let (weight, _) = extract_tensor_inner(layer.weight.val());
    let (down, down_node) = extract_tensor_inner(layer.lora_down.val());
    let (up, up_node) = extract_tensor_inner(layer.lora_up.val());
    let scale = (layer.alpha as f32 / layer.rank as f32).elem();
    InnerLoraWeights {
        weight,
        down,
        up,
        down_node,
        up_node,
        scale,
    }
}

fn extract_adaln_weights<B, C>(
    modulation: &AdaLnModulation<Autodiff<B, C>>,
) -> InnerAdaLnWeights<B>
where
    B: Backend,
    C: CheckpointStrategy,
{
    InnerAdaLnWeights {
        down: extract_lora_weights(&modulation.down),
        up: extract_lora_weights(&modulation.up),
        hidden_size: modulation.hidden_size,
    }
}

fn extract_rms_norm<B, C>(norm: &AnimaRmsNorm<Autodiff<B, C>>) -> (B::FloatTensorPrimitive, f32)
where
    B: Backend,
    C: CheckpointStrategy,
{
    let (gamma, _) = extract_tensor_inner(norm.gamma.val());
    (gamma, norm.epsilon)
}

fn extract_attention_weights<B, C>(
    attn: &AnimaAttentionCuda<Autodiff<B, C>>,
) -> InnerAttentionWeights<B>
where
    B: Backend,
    C: CheckpointStrategy,
{
    let std = &attn.standard;
    let (q_gamma, rms_eps) = extract_rms_norm(&std.q_norm);
    let (k_gamma, _) = extract_rms_norm(&std.k_norm);
    InnerAttentionWeights {
        q_proj: extract_lora_weights(&std.q_proj),
        k_proj: extract_lora_weights(&std.k_proj),
        v_proj: extract_lora_weights(&std.v_proj),
        output_proj: extract_lora_weights(&std.output_proj),
        q_gamma,
        k_gamma,
        rms_eps,
        use_rotary: std.use_rotary,
        head_count: std.head_count,
        head_size: std.head_size,
    }
}

fn extract_block_weights<B, C>(
    block: &AnimaDiTBlock<Autodiff<B, C>, AnimaAttentionCuda<Autodiff<B, C>>>,
) -> InnerDiTBlockWeights<B>
where
    B: Backend,
    C: CheckpointStrategy,
{
    InnerDiTBlockWeights {
        self_attn: extract_attention_weights(&block.self_attn),
        cross_attn: extract_attention_weights(&block.cross_attn),
        mlp_layer1: extract_lora_weights(&block.mlp_layer1),
        mlp_layer2: extract_lora_weights(&block.mlp_layer2),
        self_attn_mod: extract_adaln_weights(&block.self_attn_modulation),
        cross_attn_mod: extract_adaln_weights(&block.cross_attn_modulation),
        mlp_mod: extract_adaln_weights(&block.mlp_modulation),
    }
}

fn run_lora_inner<B: Backend>(
    weights: &InnerLoraWeights<B>,
    input: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let dtype: FloatDType = input.dtype().into();
    let w = B::float_cast(weights.weight.clone(), dtype);
    let d = B::float_cast(weights.down.clone(), dtype);
    let u = B::float_cast(weights.up.clone(), dtype);
    lora_linear_forward_inner::<B>(input, w, d, u, weights.scale)
}

fn adaln_forward_cached<B: Backend>(
    weights: &InnerAdaLnWeights<B>,
    input: B::FloatTensorPrimitive,
    condition: B::FloatTensorPrimitive,
    timestep_embedding: B::FloatTensorPrimitive,
    activation_dtype: FloatDType,
) -> AdaLnForwardCache<B> {
    let shape = input.shape();
    let batch = shape[0];
    let hidden_size = weights.hidden_size;
    let res_dtype: FloatDType = input.dtype().into();

    let h1 = run_lora_inner(&weights.down, condition);
    let h1_silu = B::float_mul(
        h1.clone(),
        B::float_recip(B::float_add_scalar(
            B::float_exp(B::float_neg(h1)),
            1.0_f32.elem(),
        )),
    );
    let params = B::float_add(run_lora_inner(&weights.up, h1_silu), timestep_embedding);

    let shift = B::float_slice(
        params.clone(),
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(0, Some(hidden_size as isize), 1),
        ],
    );
    let scale = B::float_slice(
        params.clone(),
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(hidden_size as isize, Some((hidden_size * 2) as isize), 1),
        ],
    );
    let gate = B::float_slice(
        params,
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(
                (hidden_size * 2) as isize,
                Some((hidden_size * 3) as isize),
                1,
            ),
        ],
    );

    let shift_3d = B::float_reshape(shift, Shape::from([batch, 1, hidden_size]));
    let scale_3d = B::float_reshape(scale, Shape::from([batch, 1, hidden_size]));
    let gate_3d = B::float_reshape(gate, Shape::from([batch, 1, hidden_size]));

    let hidden_f32 = hidden_size as f32;
    let mean = B::float_div_scalar(B::float_sum_dim(input.clone(), 2), hidden_f32.elem());
    let mean_exp = B::float_expand(mean, shape.clone());
    let diff = B::float_sub(input, mean_exp);
    let var = B::float_div_scalar(
        B::float_sum_dim(B::float_mul(diff.clone(), diff.clone()), 2),
        hidden_f32.elem(),
    );
    let std = B::float_sqrt(B::float_add_scalar(var, 1e-6_f32.elem()));
    let inv_std = B::float_recip(std);
    let inv_std_exp = B::float_expand(inv_std.clone(), shape.clone());
    let x_hat = B::float_mul(diff, inv_std_exp);

    let scale_plus_one = B::float_expand(B::float_add_scalar(scale_3d.clone(), 1.0_f32.elem()), shape.clone());
    let shift_exp = B::float_expand(shift_3d, shape);
    let normalized = B::float_add(B::float_mul(x_hat.clone(), scale_plus_one), shift_exp);
    let normalized_act = B::float_cast(normalized, activation_dtype);

    AdaLnForwardCache {
        normalized_act,
        x_hat: B::float_cast(x_hat, res_dtype),
        inv_std: B::float_cast(inv_std, res_dtype),
        scale: B::float_cast(scale_3d, res_dtype),
        gate: B::float_cast(gate_3d, res_dtype),
    }
}

fn adaln_input_backward<B: Backend>(
    d_residual_out: B::FloatTensorPrimitive,
    d_sublayer_in_act: B::FloatTensorPrimitive,
    cache: &AdaLnForwardCache<B>,
    residual_dtype: FloatDType,
) -> B::FloatTensorPrimitive {
    let shape = d_residual_out.shape();
    let hidden_f32 = shape[2] as f32;

    let d_z = B::float_cast(d_residual_out, residual_dtype);
    let d_y = B::float_cast(d_sublayer_in_act, residual_dtype);
    let scale_plus_one =
        B::float_expand(B::float_add_scalar(cache.scale.clone(), 1.0_f32.elem()), shape.clone());
    let d_x_hat = B::float_mul(d_y, scale_plus_one);

    let mean_dx = B::float_expand(
        B::float_div_scalar(B::float_sum_dim(d_x_hat.clone(), 2), hidden_f32.elem()),
        shape.clone(),
    );
    let mean_dx_xhat = B::float_expand(
        B::float_div_scalar(
            B::float_sum_dim(B::float_mul(d_x_hat.clone(), cache.x_hat.clone()), 2),
            hidden_f32.elem(),
        ),
        shape.clone(),
    );

    let centered = B::float_sub(
        B::float_sub(d_x_hat, mean_dx),
        B::float_mul(cache.x_hat.clone(), mean_dx_xhat),
    );
    let inv_std_exp = B::float_expand(cache.inv_std.clone(), shape);
    let d_x_adaln = B::float_mul(inv_std_exp, centered);

    B::float_add(d_z, d_x_adaln)
}

fn rms_norm_forward_inner<B: Backend>(
    input: B::FloatTensorPrimitive,
    gamma: B::FloatTensorPrimitive,
    epsilon: f32,
) -> B::FloatTensorPrimitive {
    let input_dtype: FloatDType = input.dtype().into();
    let shape = input.shape();
    let last_dim = shape.num_dims() - 1;
    let feature_count = shape[last_dim] as f32;

    let x_f32 = B::float_cast(input, FloatDType::F32);
    let gamma_f32 = B::float_cast(gamma, FloatDType::F32);
    let mut gamma_dims = vec![1; shape.num_dims()];
    gamma_dims[last_dim] = shape[last_dim];
    let gamma_exp = B::float_expand(B::float_reshape(gamma_f32, Shape::from(gamma_dims)), shape.clone());

    let rms = B::float_sqrt(B::float_add_scalar(
        B::float_div_scalar(
            B::float_sum_dim(B::float_mul(x_f32.clone(), x_f32.clone()), last_dim),
            feature_count.elem(),
        ),
        epsilon.elem(),
    ));
    let rms_exp = B::float_expand(rms, shape);
    let out_f32 = B::float_mul(B::float_div(x_f32, rms_exp), gamma_exp);
    B::float_cast(out_f32, input_dtype)
}

fn rms_norm_backward_inner<B: Backend>(
    grad_output: B::FloatTensorPrimitive,
    input: B::FloatTensorPrimitive,
    gamma: B::FloatTensorPrimitive,
    epsilon: f32,
) -> B::FloatTensorPrimitive {
    let input_dtype: FloatDType = input.dtype().into();
    let shape = input.shape();
    let last_dim = shape.num_dims() - 1;
    let feature_count = shape[last_dim] as f32;

    let x_f32 = B::float_cast(input, FloatDType::F32);
    let dy_f32 = B::float_cast(grad_output, FloatDType::F32);
    let gamma_f32 = B::float_cast(gamma, FloatDType::F32);
    let mut gamma_dims = vec![1; shape.num_dims()];
    gamma_dims[last_dim] = shape[last_dim];
    let gamma_exp = B::float_expand(B::float_reshape(gamma_f32, Shape::from(gamma_dims)), shape.clone());

    let rms = B::float_sqrt(B::float_add_scalar(
        B::float_div_scalar(
            B::float_sum_dim(B::float_mul(x_f32.clone(), x_f32.clone()), last_dim),
            feature_count.elem(),
        ),
        epsilon.elem(),
    ));
    let rms_exp = B::float_expand(rms, shape.clone());
    let x_hat = B::float_div(x_f32, rms_exp.clone());
    let g = B::float_mul(dy_f32, gamma_exp);

    let dot_mean = B::float_expand(
        B::float_div_scalar(
            B::float_sum_dim(B::float_mul(g.clone(), x_hat.clone()), last_dim),
            feature_count.elem(),
        ),
        shape,
    );
    let dx_f32 = B::float_div(B::float_sub(g, B::float_mul(x_hat, dot_mean)), rms_exp);
    B::float_cast(dx_f32, input_dtype)
}

fn build_rotary_3d_tables<B: Backend>(
    height: usize,
    width: usize,
    head_size: usize,
    dtype: FloatDType,
    device: &B::Device,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    let sequence_length = height * width;
    let dim_h = head_size / 6 * 2;
    let dim_w = dim_h;
    let dim_t = head_size - dim_h - dim_w;
    let half = head_size / 2;
    let mut frequencies = Vec::with_capacity(sequence_length * half);
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
    let cos_values = frequencies.iter().map(|v| v.cos()).collect::<Vec<_>>();
    let sin_values = frequencies.iter().map(|v| v.sin()).collect::<Vec<_>>();
    let cos = B::float_cast(
        B::float_from_data(
            TensorData::new(cos_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
            device,
        ),
        dtype,
    );
    let sin = B::float_cast(
        B::float_from_data(
            TensorData::new(sin_values, [1, 1, sequence_length, half]).convert::<B::FloatElem>(),
            device,
        ),
        dtype,
    );
    (cos, sin)
}

fn rotary_forward_inner<B: Backend>(
    input: B::FloatTensorPrimitive,
    cos: B::FloatTensorPrimitive,
    sin: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let shape = input.shape();
    let [batch, heads, seq, head_size] = [shape[0], shape[1], shape[2], shape[3]];
    let half = head_size / 2;
    let first = B::float_slice(
        input.clone(),
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(0, Some(heads as isize), 1),
            Slice::new(0, Some(seq as isize), 1),
            Slice::new(0, Some(half as isize), 1),
        ],
    );
    let second = B::float_slice(
        input,
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(0, Some(heads as isize), 1),
            Slice::new(0, Some(seq as isize), 1),
            Slice::new(half as isize, Some(head_size as isize), 1),
        ],
    );
    let cos_exp = B::float_expand(cos, first.shape());
    let sin_exp = B::float_expand(sin, first.shape());
    let out_first = B::float_sub(
        B::float_mul(first.clone(), cos_exp.clone()),
        B::float_mul(second.clone(), sin_exp.clone()),
    );
    let out_second = B::float_add(
        B::float_mul(second, cos_exp),
        B::float_mul(first, sin_exp),
    );
    B::float_cat(vec![out_first, out_second], 3)
}

fn rotary_backward_inner<B: Backend>(
    grad_output: B::FloatTensorPrimitive,
    cos: B::FloatTensorPrimitive,
    sin: B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let shape = grad_output.shape();
    let [batch, heads, seq, head_size] = [shape[0], shape[1], shape[2], shape[3]];
    let half = head_size / 2;
    let dy_first = B::float_slice(
        grad_output.clone(),
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(0, Some(heads as isize), 1),
            Slice::new(0, Some(seq as isize), 1),
            Slice::new(0, Some(half as isize), 1),
        ],
    );
    let dy_second = B::float_slice(
        grad_output,
        &[
            Slice::new(0, Some(batch as isize), 1),
            Slice::new(0, Some(heads as isize), 1),
            Slice::new(0, Some(seq as isize), 1),
            Slice::new(half as isize, Some(head_size as isize), 1),
        ],
    );
    let cos_exp = B::float_expand(cos, dy_first.shape());
    let sin_exp = B::float_expand(sin, dy_first.shape());
    let dx_first = B::float_add(
        B::float_mul(dy_first.clone(), cos_exp.clone()),
        B::float_mul(dy_second.clone(), sin_exp.clone()),
    );
    let dx_second = B::float_sub(
        B::float_mul(dy_second, cos_exp),
        B::float_mul(dy_first, sin_exp),
    );
    B::float_cat(vec![dx_first, dx_second], 3)
}

fn block_forward_inner<B: Backend>(
    block: &InnerDiTBlockWeights<B>,
    input: B::FloatTensorPrimitive,
    text_act: B::FloatTensorPrimitive,
    condition: B::FloatTensorPrimitive,
    timestep_embedding: B::FloatTensorPrimitive,
    text_mask: Option<B::FloatTensorPrimitive>,
    rope: Option<(B::FloatTensorPrimitive, B::FloatTensorPrimitive)>,
    activation_dtype: FloatDType,
    residual_dtype: FloatDType,
    attention_chunk: usize,
) -> (B::FloatTensorPrimitive, Option<BlockRecomputeCache<B>>) {
    let shape = input.shape();
    let [batch, seq, hidden_size] = [shape[0], shape[1], shape[2]];
    let text_len = text_act.shape()[1];
    let heads = block.self_attn.head_count;
    let head_size = block.self_attn.head_size;
    let attn_scale: B::FloatElem = (1.0 / (head_size as f32).sqrt()).elem();

    // 1. Self-Attention stage
    let self_mod = adaln_forward_cached(
        &block.self_attn_mod,
        input.clone(),
        condition.clone(),
        timestep_embedding.clone(),
        activation_dtype,
    );
    let q_self_pre = B::float_swap_dims(
        B::float_reshape(
            run_lora_inner(&block.self_attn.q_proj, self_mod.normalized_act.clone()),
            Shape::from([batch, seq, heads, head_size]),
        ),
        1,
        2,
    );
    let k_self_pre = B::float_swap_dims(
        B::float_reshape(
            run_lora_inner(&block.self_attn.k_proj, self_mod.normalized_act.clone()),
            Shape::from([batch, seq, heads, head_size]),
        ),
        1,
        2,
    );
    let v_self = B::float_swap_dims(
        B::float_reshape(
            run_lora_inner(&block.self_attn.v_proj, self_mod.normalized_act.clone()),
            Shape::from([batch, seq, heads, head_size]),
        ),
        1,
        2,
    );
    let q_self_normed = rms_norm_forward_inner::<B>(
        q_self_pre.clone(),
        block.self_attn.q_gamma.clone(),
        block.self_attn.rms_eps,
    );
    let k_self_normed = rms_norm_forward_inner::<B>(
        k_self_pre.clone(),
        block.self_attn.k_gamma.clone(),
        block.self_attn.rms_eps,
    );
    let (q_self_rot, k_self_rot) = match &rope {
        Some((cos, sin)) if block.self_attn.use_rotary => (
            rotary_forward_inner::<B>(q_self_normed.clone(), cos.clone(), sin.clone()),
            rotary_forward_inner::<B>(k_self_normed.clone(), cos.clone(), sin.clone()),
        ),
        _ => (q_self_normed.clone(), k_self_normed.clone()),
    };
    let self_attended = attention_forward::<B>(
        q_self_rot.clone(),
        k_self_rot.clone(),
        v_self.clone(),
        None,
        attn_scale,
        attention_chunk,
    );
    let self_attended_2d = B::float_reshape(
        B::float_swap_dims(self_attended, 1, 2),
        Shape::from([batch, seq, hidden_size]),
    );
    let self_out = B::float_cast(
        run_lora_inner(&block.self_attn.output_proj, self_attended_2d.clone()),
        residual_dtype,
    );
    let after_self = B::float_add(
        input,
        B::float_mul(self_out, B::float_expand(self_mod.gate.clone(), shape.clone())),
    );

    // 2. Cross-Attention stage
    let cross_mod = adaln_forward_cached(
        &block.cross_attn_mod,
        after_self.clone(),
        condition.clone(),
        timestep_embedding.clone(),
        activation_dtype,
    );
    let q_cross_pre = B::float_swap_dims(
        B::float_reshape(
            run_lora_inner(&block.cross_attn.q_proj, cross_mod.normalized_act.clone()),
            Shape::from([batch, seq, heads, head_size]),
        ),
        1,
        2,
    );
    let k_cross_pre = B::float_swap_dims(
        B::float_reshape(
            run_lora_inner(&block.cross_attn.k_proj, text_act.clone()),
            Shape::from([batch, text_len, heads, head_size]),
        ),
        1,
        2,
    );
    let v_cross = B::float_swap_dims(
        B::float_reshape(
            run_lora_inner(&block.cross_attn.v_proj, text_act),
            Shape::from([batch, text_len, heads, head_size]),
        ),
        1,
        2,
    );
    let q_cross_normed = rms_norm_forward_inner::<B>(
        q_cross_pre.clone(),
        block.cross_attn.q_gamma.clone(),
        block.cross_attn.rms_eps,
    );
    let k_cross_normed = rms_norm_forward_inner::<B>(
        k_cross_pre.clone(),
        block.cross_attn.k_gamma.clone(),
        block.cross_attn.rms_eps,
    );
    let cross_attended = attention_forward::<B>(
        q_cross_normed.clone(),
        k_cross_normed.clone(),
        v_cross.clone(),
        text_mask,
        attn_scale,
        attention_chunk,
    );
    let cross_attended_2d = B::float_reshape(
        B::float_swap_dims(cross_attended, 1, 2),
        Shape::from([batch, seq, hidden_size]),
    );
    let cross_out = B::float_cast(
        run_lora_inner(&block.cross_attn.output_proj, cross_attended_2d.clone()),
        residual_dtype,
    );
    let after_cross = B::float_add(
        after_self,
        B::float_mul(
            cross_out,
            B::float_expand(cross_mod.gate.clone(), shape.clone()),
        ),
    );

    // 3. MLP stage
    let mlp_mod = adaln_forward_cached(
        &block.mlp_mod,
        after_cross.clone(),
        condition,
        timestep_embedding,
        activation_dtype,
    );
    let mlp_h1 = run_lora_inner(&block.mlp_layer1, mlp_mod.normalized_act.clone());
    let mlp_h2 = B::gelu(mlp_h1.clone());
    let mlp_out = B::float_cast(
        run_lora_inner(&block.mlp_layer2, mlp_h2.clone()),
        residual_dtype,
    );
    let output = B::float_add(
        after_cross,
        B::float_mul(mlp_out, B::float_expand(mlp_mod.gate.clone(), shape)),
    );

    let cache = BlockRecomputeCache {
        self_mod,
        q_self_pre,
        k_self_pre,
        q_self_normed,
        k_self_normed,
        q_self_rot,
        k_self_rot,
        v_self,
        self_attended_2d,
        cross_mod,
        q_cross_pre,
        k_cross_pre,
        q_cross_normed,
        k_cross_normed,
        v_cross,
        cross_attended_2d,
        mlp_mod,
        mlp_h1,
        mlp_h2,
    };
    (output, Some(cache))
}

fn register_lora_grads<B: Backend>(
    weights: &InnerLoraWeights<B>,
    d_down: B::FloatTensorPrimitive,
    d_up: B::FloatTensorPrimitive,
    grads: &mut Gradients,
) {
    grads.register::<B>(weights.down_node.id, d_down);
    grads.register::<B>(weights.up_node.id, d_up);
}

#[allow(clippy::too_many_arguments)]
fn block_backward_inner<B: Backend>(
    block: &InnerDiTBlockWeights<B>,
    cache: BlockRecomputeCache<B>,
    d_output: B::FloatTensorPrimitive,
    text_act: B::FloatTensorPrimitive,
    text_mask: Option<B::FloatTensorPrimitive>,
    rope: Option<(B::FloatTensorPrimitive, B::FloatTensorPrimitive)>,
    activation_dtype: FloatDType,
    residual_dtype: FloatDType,
    attention_chunk: usize,
    need_prev_dx: bool,
    grads: &mut Gradients,
) -> Option<B::FloatTensorPrimitive> {
    let shape = d_output.shape();
    let [batch, seq, hidden_size] = [shape[0], shape[1], shape[2]];
    let heads = block.self_attn.head_count;
    let head_size = block.self_attn.head_size;
    let attn_scale: B::FloatElem = (1.0 / (head_size as f32).sqrt()).elem();

    // --- 1. MLP VJP ---
    let d_mlp_out_act = B::float_cast(
        B::float_mul(
            d_output.clone(),
            B::float_expand(cache.mlp_mod.gate.clone(), shape.clone()),
        ),
        activation_dtype,
    );
    let mlp2_grads = lora_linear_backward::<B>(
        d_mlp_out_act,
        cache.mlp_h2,
        block.mlp_layer2.weight.clone(),
        block.mlp_layer2.down.clone(),
        block.mlp_layer2.up.clone(),
        block.mlp_layer2.scale,
        true,
    );
    register_lora_grads(&block.mlp_layer2, mlp2_grads.d_down, mlp2_grads.d_up, grads);

    let d_mlp_h1 = B::gelu_backward(cache.mlp_h1, mlp2_grads.d_input.unwrap());
    let mlp1_grads = lora_linear_backward::<B>(
        d_mlp_h1,
        cache.mlp_mod.normalized_act.clone(),
        block.mlp_layer1.weight.clone(),
        block.mlp_layer1.down.clone(),
        block.mlp_layer1.up.clone(),
        block.mlp_layer1.scale,
        true,
    );
    register_lora_grads(&block.mlp_layer1, mlp1_grads.d_down, mlp1_grads.d_up, grads);

    let d_after_cross = adaln_input_backward::<B>(
        d_output,
        mlp1_grads.d_input.unwrap(),
        &cache.mlp_mod,
        residual_dtype,
    );

    // --- 2. Cross-Attention VJP ---
    let d_cross_out_act = B::float_cast(
        B::float_mul(
            d_after_cross.clone(),
            B::float_expand(cache.cross_mod.gate.clone(), shape.clone()),
        ),
        activation_dtype,
    );
    let cross_o_grads = lora_linear_backward::<B>(
        d_cross_out_act,
        cache.cross_attended_2d,
        block.cross_attn.output_proj.weight.clone(),
        block.cross_attn.output_proj.down.clone(),
        block.cross_attn.output_proj.up.clone(),
        block.cross_attn.output_proj.scale,
        true,
    );
    register_lora_grads(
        &block.cross_attn.output_proj,
        cross_o_grads.d_down,
        cross_o_grads.d_up,
        grads,
    );

    let d_cross_attended = B::float_swap_dims(
        B::float_reshape(
            cross_o_grads.d_input.unwrap(),
            Shape::from([batch, seq, heads, head_size]),
        ),
        1,
        2,
    );
    let (d_q_cross_normed, d_k_cross_normed, d_v_cross) = attention_backward::<B>(
        d_cross_attended,
        cache.q_cross_normed,
        cache.k_cross_normed.clone(),
        cache.v_cross,
        text_mask,
        attn_scale,
        attention_chunk,
    );
    let d_q_cross_pre = rms_norm_backward_inner::<B>(
        d_q_cross_normed,
        cache.q_cross_pre,
        block.cross_attn.q_gamma.clone(),
        block.cross_attn.rms_eps,
    );
    let d_q_cross_2d = B::float_reshape(
        B::float_swap_dims(d_q_cross_pre, 1, 2),
        Shape::from([batch, seq, hidden_size]),
    );
    let cross_q_grads = lora_linear_backward::<B>(
        d_q_cross_2d,
        cache.cross_mod.normalized_act.clone(),
        block.cross_attn.q_proj.weight.clone(),
        block.cross_attn.q_proj.down.clone(),
        block.cross_attn.q_proj.up.clone(),
        block.cross_attn.q_proj.scale,
        true,
    );
    register_lora_grads(
        &block.cross_attn.q_proj,
        cross_q_grads.d_down,
        cross_q_grads.d_up,
        grads,
    );

    // k_proj and v_proj take frozen `text_act`; `d_text` is not needed, so skip dX.
    let d_k_cross_pre = rms_norm_backward_inner::<B>(
        d_k_cross_normed,
        cache.k_cross_pre,
        block.cross_attn.k_gamma.clone(),
        block.cross_attn.rms_eps,
    );
    let text_len = text_act.shape()[1];
    let d_k_cross_2d = B::float_reshape(
        B::float_swap_dims(d_k_cross_pre, 1, 2),
        Shape::from([batch, text_len, hidden_size]),
    );
    let cross_k_grads = lora_linear_backward::<B>(
        d_k_cross_2d,
        text_act.clone(),
        block.cross_attn.k_proj.weight.clone(),
        block.cross_attn.k_proj.down.clone(),
        block.cross_attn.k_proj.up.clone(),
        block.cross_attn.k_proj.scale,
        false,
    );
    register_lora_grads(
        &block.cross_attn.k_proj,
        cross_k_grads.d_down,
        cross_k_grads.d_up,
        grads,
    );

    let d_v_cross_2d = B::float_reshape(
        B::float_swap_dims(d_v_cross, 1, 2),
        Shape::from([batch, text_len, hidden_size]),
    );
    let cross_v_grads = lora_linear_backward::<B>(
        d_v_cross_2d,
        text_act,
        block.cross_attn.v_proj.weight.clone(),
        block.cross_attn.v_proj.down.clone(),
        block.cross_attn.v_proj.up.clone(),
        block.cross_attn.v_proj.scale,
        false,
    );
    register_lora_grads(
        &block.cross_attn.v_proj,
        cross_v_grads.d_down,
        cross_v_grads.d_up,
        grads,
    );

    let d_after_self = adaln_input_backward::<B>(
        d_after_cross,
        cross_q_grads.d_input.unwrap(),
        &cache.cross_mod,
        residual_dtype,
    );

    // --- 3. Self-Attention VJP ---
    let d_self_out_act = B::float_cast(
        B::float_mul(
            d_after_self.clone(),
            B::float_expand(cache.self_mod.gate.clone(), shape),
        ),
        activation_dtype,
    );
    let self_o_grads = lora_linear_backward::<B>(
        d_self_out_act,
        cache.self_attended_2d,
        block.self_attn.output_proj.weight.clone(),
        block.self_attn.output_proj.down.clone(),
        block.self_attn.output_proj.up.clone(),
        block.self_attn.output_proj.scale,
        true,
    );
    register_lora_grads(
        &block.self_attn.output_proj,
        self_o_grads.d_down,
        self_o_grads.d_up,
        grads,
    );

    let d_self_attended = B::float_swap_dims(
        B::float_reshape(
            self_o_grads.d_input.unwrap(),
            Shape::from([batch, seq, heads, head_size]),
        ),
        1,
        2,
    );
    let (d_q_self_rot, d_k_self_rot, d_v_self) = attention_backward::<B>(
        d_self_attended,
        cache.q_self_rot,
        cache.k_self_rot,
        cache.v_self,
        None,
        attn_scale,
        attention_chunk,
    );
    let (d_q_self_normed, d_k_self_normed) = match &rope {
        Some((cos, sin)) if block.self_attn.use_rotary => (
            rotary_backward_inner::<B>(d_q_self_rot, cos.clone(), sin.clone()),
            rotary_backward_inner::<B>(d_k_self_rot, cos.clone(), sin.clone()),
        ),
        _ => (d_q_self_rot, d_k_self_rot),
    };
    let d_q_self_pre = rms_norm_backward_inner::<B>(
        d_q_self_normed,
        cache.q_self_pre,
        block.self_attn.q_gamma.clone(),
        block.self_attn.rms_eps,
    );
    let d_k_self_pre = rms_norm_backward_inner::<B>(
        d_k_self_normed,
        cache.k_self_pre,
        block.self_attn.k_gamma.clone(),
        block.self_attn.rms_eps,
    );

    let d_q_self_2d = B::float_reshape(
        B::float_swap_dims(d_q_self_pre, 1, 2),
        Shape::from([batch, seq, hidden_size]),
    );
    let d_k_self_2d = B::float_reshape(
        B::float_swap_dims(d_k_self_pre, 1, 2),
        Shape::from([batch, seq, hidden_size]),
    );
    let d_v_self_2d = B::float_reshape(
        B::float_swap_dims(d_v_self, 1, 2),
        Shape::from([batch, seq, hidden_size]),
    );

    let self_q_grads = lora_linear_backward::<B>(
        d_q_self_2d,
        cache.self_mod.normalized_act.clone(),
        block.self_attn.q_proj.weight.clone(),
        block.self_attn.q_proj.down.clone(),
        block.self_attn.q_proj.up.clone(),
        block.self_attn.q_proj.scale,
        need_prev_dx,
    );
    register_lora_grads(
        &block.self_attn.q_proj,
        self_q_grads.d_down,
        self_q_grads.d_up,
        grads,
    );

    let self_k_grads = lora_linear_backward::<B>(
        d_k_self_2d,
        cache.self_mod.normalized_act.clone(),
        block.self_attn.k_proj.weight.clone(),
        block.self_attn.k_proj.down.clone(),
        block.self_attn.k_proj.up.clone(),
        block.self_attn.k_proj.scale,
        need_prev_dx,
    );
    register_lora_grads(
        &block.self_attn.k_proj,
        self_k_grads.d_down,
        self_k_grads.d_up,
        grads,
    );

    let self_v_grads = lora_linear_backward::<B>(
        d_v_self_2d,
        cache.self_mod.normalized_act.clone(),
        block.self_attn.v_proj.weight.clone(),
        block.self_attn.v_proj.down.clone(),
        block.self_attn.v_proj.up.clone(),
        block.self_attn.v_proj.scale,
        need_prev_dx,
    );
    register_lora_grads(
        &block.self_attn.v_proj,
        self_v_grads.d_down,
        self_v_grads.d_up,
        grads,
    );

    if !need_prev_dx {
        return None;
    }

    let d_self_in = B::float_add(
        B::float_add(self_q_grads.d_input.unwrap(), self_k_grads.d_input.unwrap()),
        self_v_grads.d_input.unwrap(),
    );
    Some(adaln_input_backward::<B>(
        d_after_self,
        d_self_in,
        &cache.self_mod,
        residual_dtype,
    ))
}

/// Executes all DiT blocks on `InnerBackend` during forward, retaining only
/// block-entry activations, and runs a single reverse loop manual VJP during
/// backward without constructing any local autodiff graph.
pub fn forward_dit_blocks_manual_vjp<B, C>(
    blocks: &[AnimaDiTBlock<Autodiff<B, C>, AnimaAttentionCuda<Autodiff<B, C>>>],
    hidden: Tensor<Autodiff<B, C>, 3>,
    text: Tensor<Autodiff<B, C>, 3>,
    time_condition: Tensor<Autodiff<B, C>, 2>,
    timestep_embedding: Tensor<Autodiff<B, C>, 2>,
    text_mask: Option<Tensor<Autodiff<B, C>, 4>>,
    spatial: (usize, usize),
) -> Tensor<Autodiff<B, C>, 3>
where
    B: Backend,
    C: CheckpointStrategy,
{
    let activation_dtype: FloatDType = hidden.dtype().into();
    let mixed_precision = std::env::var_os("AFTERBURNER_MIXED_PRECISION").is_some();
    let fp32_residuals = mixed_precision && matches!(activation_dtype, FloatDType::F16);
    let fp32_boundary =
        mixed_precision && matches!(activation_dtype, FloatDType::F16 | FloatDType::F32);
    let residual_dtype = if fp32_residuals {
        FloatDType::F32
    } else {
        activation_dtype
    };

    let (hidden_raw, hidden_node) = extract_tensor_inner(hidden);
    let (text_raw, _) = extract_tensor_inner(text);
    let (cond_raw, _) = extract_tensor_inner(time_condition);
    let (t_emb_raw, _) = extract_tensor_inner(timestep_embedding);
    let mask_raw = text_mask.map(|mask| extract_tensor_inner(mask).0);

    let device = B::float_device(&hidden_raw);
    let mut current_hidden = if fp32_residuals {
        B::float_cast(hidden_raw, FloatDType::F32)
    } else {
        hidden_raw
    };
    let text_boundary = if fp32_boundary {
        B::float_cast(text_raw, FloatDType::F32)
    } else {
        text_raw
    };
    let text_act = B::float_cast(text_boundary, activation_dtype);
    let cond_boundary = if fp32_boundary {
        B::float_cast(cond_raw, FloatDType::F32)
    } else {
        cond_raw
    };
    let t_emb_boundary = if fp32_boundary {
        B::float_cast(t_emb_raw, FloatDType::F32)
    } else {
        t_emb_raw
    };

    let head_size = blocks[0].self_attn.standard.head_size;
    let rope = Some(build_rotary_3d_tables::<B>(
        spatial.0,
        spatial.1,
        head_size,
        activation_dtype,
        &device,
    ));
    let attention_chunk = attention_query_chunk_size();

    let mut block_weights = Vec::with_capacity(blocks.len());
    let mut block_inputs = Vec::with_capacity(blocks.len());
    let mut parent_nodes = Vec::with_capacity(blocks.len() * 20 + 1);
    let input_requires_grad = !hidden_node.requirement.is_none();
    if input_requires_grad {
        parent_nodes.push(hidden_node.clone());
    }

    for block in blocks {
        let w = extract_block_weights(block);
        for lora in [
            &w.self_attn.q_proj,
            &w.self_attn.k_proj,
            &w.self_attn.v_proj,
            &w.self_attn.output_proj,
            &w.cross_attn.q_proj,
            &w.cross_attn.k_proj,
            &w.cross_attn.v_proj,
            &w.cross_attn.output_proj,
            &w.mlp_layer1,
            &w.mlp_layer2,
        ] {
            parent_nodes.push(lora.down_node.clone());
            parent_nodes.push(lora.up_node.clone());
        }
        block_inputs.push(current_hidden.clone());
        let (next_hidden, _) = block_forward_inner::<B>(
            &w,
            current_hidden,
            text_act.clone(),
            cond_boundary.clone(),
            t_emb_boundary.clone(),
            mask_raw.clone(),
            rope.clone(),
            activation_dtype,
            residual_dtype,
            attention_chunk,
        );
        current_hidden = next_hidden;
        block_weights.push(w);
    }

    let input_node_id: Option<NodeId> = input_requires_grad.then_some(hidden_node.id);
    let backward_fn = Box::new(
        move |grad_output: B::FloatTensorPrimitive, grads: &mut Gradients| {
            let mut d_hidden = grad_output;
            for block_idx in (0..block_weights.len()).rev() {
                let need_prev_dx = block_idx > 0 || input_node_id.is_some();
                let w = &block_weights[block_idx];
                let h_in = block_inputs[block_idx].clone();
                let (_, cache) = block_forward_inner::<B>(
                    w,
                    h_in,
                    text_act.clone(),
                    cond_boundary.clone(),
                    t_emb_boundary.clone(),
                    mask_raw.clone(),
                    rope.clone(),
                    activation_dtype,
                    residual_dtype,
                    attention_chunk,
                );
                let cache = cache.expect("block recompute cache must be produced");
                if let Some(prev_d_hidden) = block_backward_inner::<B>(
                    w,
                    cache,
                    d_hidden.clone(),
                    text_act.clone(),
                    mask_raw.clone(),
                    rope.clone(),
                    activation_dtype,
                    residual_dtype,
                    attention_chunk,
                    need_prev_dx,
                    grads,
                ) {
                    d_hidden = prev_d_hidden;
                }
            }
            if let Some(in_id) = input_node_id {
                let dx_final = B::float_cast(d_hidden, activation_dtype);
                grads.register::<B>(in_id, dx_final);
            }
        },
    );

    let output_ad = register_manual_vjp::<B, 3>(current_hidden, &parent_nodes, backward_fn);
    Tensor::from_primitive(TensorPrimitive::Float(output_ad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        anima::AnimaConfig,
        dit::{AnimaDiT, AnimaDiTBlockConfig},
        lora::LoraConfig,
    };
    use burn::module::Param;
    use burn_autodiff::checkpoint::strategy::NoCheckpointing;
    use burn_ndarray::NdArray;
    use burn_optim::GradientsParams;

    type Inner = NdArray<f32>;
    type AdBackend = Autodiff<Inner, NoCheckpointing>;

    fn build_test_block(device: &<AdBackend as Backend>::Device) -> AnimaDiTBlock<AdBackend, AnimaAttentionCuda<AdBackend>> {
        let config = AnimaDiTBlockConfig {
            hidden_size: 12,
            context_size: 12,
            attention_head_count: 2,
            attention_head_size: 6,
            mlp_hidden_size: 24,
            lora_rank: 2,
            lora_alpha: 2,
        };
        let mut block = AnimaDiTBlock::<AdBackend, AnimaAttentionCuda<AdBackend>>::init(&config, device).unwrap();
        let eye12 = Tensor::<AdBackend, 2>::eye(12, device);
        let ones2 = Tensor::<AdBackend, 2>::ones([2, 12], device).mul_scalar(0.05);
        let ones2_out = Tensor::<AdBackend, 2>::ones([12, 2], device).mul_scalar(0.05);

        for proj in [
            &mut block.self_attn.standard.q_proj,
            &mut block.self_attn.standard.k_proj,
            &mut block.self_attn.standard.v_proj,
            &mut block.self_attn.standard.output_proj,
            &mut block.cross_attn.standard.q_proj,
            &mut block.cross_attn.standard.k_proj,
            &mut block.cross_attn.standard.v_proj,
            &mut block.cross_attn.standard.output_proj,
        ] {
            proj.weight = Param::from_tensor(eye12.clone());
            proj.lora_down = Param::from_tensor(ones2.clone());
            proj.lora_up = Param::from_tensor(ones2_out.clone());
        }
        block.mlp_layer1.weight = Param::from_tensor(Tensor::ones([24, 12], device).mul_scalar(0.1));
        block.mlp_layer1.lora_down = Param::from_tensor(ones2.clone());
        block.mlp_layer1.lora_up = Param::from_tensor(Tensor::ones([24, 2], device).mul_scalar(0.05));
        block.mlp_layer2.weight = Param::from_tensor(Tensor::ones([12, 24], device).mul_scalar(0.1));
        block.mlp_layer2.lora_down = Param::from_tensor(Tensor::ones([2, 24], device).mul_scalar(0.05));
        block.mlp_layer2.lora_up = Param::from_tensor(ones2_out);
        block
    }

    #[test]
    fn dit_blocks_manual_vjp_matches_autodiff_gradients() {
        let device = <AdBackend as Backend>::Device::default();
        let mut config = AnimaConfig::default();
        config.block_count = 2;
        config.hidden_size = 12;
        config.text_hidden_size = 12;
        config.attention_head_count = 2;
        config.attention_head_size = 6;
        config.mlp_hidden_size = 24;
        config.input_channels = 4;
        config.output_channels = 4;
        config.llm_adapter_block_count = 1;
        config.llm_adapter_hidden_size = 12;
        let lora = LoraConfig {
            rank: 2,
            alpha: 2,
            ..Default::default()
        };
        let mut model = AnimaDiT::<AdBackend, AnimaAttentionCuda<AdBackend>>::init(&config, &lora, &device).unwrap();
        model.blocks[0] = build_test_block(&device);
        model.blocks[1] = build_test_block(&device);

        let hidden_data: Vec<f32> = (0..48).map(|i| (i as f32 * 0.03) - 0.5).collect();
        let text_data: Vec<f32> = (0..36).map(|i| (i as f32 * 0.02) - 0.3).collect();
        let cond_data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.05) + 0.1).collect();
        let temb_data: Vec<f32> = (0..36).map(|i| (i as f32 * 0.01) - 0.15).collect();
        let grad_weights_data: Vec<f32> = (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let hidden_ref = Tensor::<AdBackend, 3>::from_data(TensorData::new(hidden_data.clone(), [1, 4, 12]), &device).require_grad();
        let text = Tensor::<AdBackend, 3>::from_data(TensorData::new(text_data, [1, 3, 12]), &device);
        let cond = Tensor::<AdBackend, 2>::from_data(TensorData::new(cond_data, [1, 12]), &device);
        let temb = Tensor::<AdBackend, 2>::from_data(TensorData::new(temb_data, [1, 36]), &device);
        let weights = Tensor::<AdBackend, 3>::from_data(TensorData::new(grad_weights_data, [1, 4, 12]), &device);

        // Reference Autodiff forward + backward
        let mut out_ref = hidden_ref.clone();
        for block in &model.blocks {
            out_ref = block.forward_with_text_mask_and_spatial(
                out_ref,
                text.clone(),
                cond.clone(),
                temb.clone(),
                None,
                Some((2, 2)),
            );
        }
        let loss_ref = (out_ref.clone() * weights.clone()).sum();
        let mut grads_ref = loss_ref.backward();

        // Manual VJP forward + backward
        let hidden_vjp = Tensor::<AdBackend, 3>::from_data(TensorData::new(hidden_data, [1, 4, 12]), &device).require_grad();
        let out_vjp = forward_dit_blocks_manual_vjp(
            &model.blocks,
            hidden_vjp.clone(),
            text,
            cond,
            temb,
            None,
            (2, 2),
        );
        let loss_vjp = (out_vjp.clone() * weights).sum();
        let mut grads_vjp = loss_vjp.backward();

        // 1. Forward output parity
        let out_ref_vec = out_ref.into_data().to_vec::<f32>().unwrap();
        let out_vjp_vec = out_vjp.into_data().to_vec::<f32>().unwrap();
        for (a, b) in out_ref_vec.iter().zip(&out_vjp_vec) {
            assert!((a - b).abs() < 1e-4, "forward mismatch: {a} vs {b}");
        }

        // 2. Input d_hidden parity
        let dx_ref = hidden_ref.grad(&grads_ref).unwrap().into_data().to_vec::<f32>().unwrap();
        let dx_vjp = hidden_vjp.grad(&grads_vjp).unwrap().into_data().to_vec::<f32>().unwrap();
        for (a, b) in dx_ref.iter().zip(&dx_vjp) {
            assert!((a - b).abs() < 1e-4, "dx mismatch: {a} vs {b}");
        }

        // 3. All 20 LoRA factor gradients across all blocks parity
        let mut ids = Vec::new();
        for block in &model.blocks {
            for layer in [
                &block.self_attn.standard.q_proj,
                &block.self_attn.standard.k_proj,
                &block.self_attn.standard.v_proj,
                &block.self_attn.standard.output_proj,
                &block.cross_attn.standard.q_proj,
                &block.cross_attn.standard.k_proj,
                &block.cross_attn.standard.v_proj,
                &block.cross_attn.standard.output_proj,
                &block.mlp_layer1,
                &block.mlp_layer2,
            ] {
                ids.push(layer.lora_down.id);
                ids.push(layer.lora_up.id);
            }
        }
        let ref_params = GradientsParams::from_params(&mut grads_ref, &model, &ids);
        let vjp_params = GradientsParams::from_params(&mut grads_vjp, &model, &ids);
        for id in ids {
            let g_ref = ref_params.get::<Inner, 2>(id).unwrap().into_data().to_vec::<f32>().unwrap();
            let g_vjp = vjp_params.get::<Inner, 2>(id).unwrap().into_data().to_vec::<f32>().unwrap();
            for (a, b) in g_ref.iter().zip(&g_vjp) {
                assert!((a - b).abs() < 1e-4, "LoRA grad mismatch for {id:?}: {a} vs {b}");
            }
        }
    }
}
