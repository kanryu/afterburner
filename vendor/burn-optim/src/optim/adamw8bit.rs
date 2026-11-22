use burn_core as burn;

use burn::config::Config;
use burn::tensor::{DType, Int, Tensor, TensorData, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};
use burn::{module::AutodiffModule, record::Record};

use super::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::LearningRate;

const BLOCK_SIZE: usize = 4096;

#[derive(Config, Debug)]
pub struct AdamW8bitConfig {
    #[config(default = 0.9)] beta_1: f32,
    #[config(default = 0.999)] beta_2: f32,
    #[config(default = 1e-8)] epsilon: f32,
    #[config(default = 1e-2)] weight_decay: f32,
}

#[derive(Clone)]
pub struct AdamW8bit { beta_1: f32, beta_2: f32, epsilon: f32, weight_decay: f32 }

#[derive(Record, Clone, new)]
pub struct AdamW8bitState<B: Backend, const D: usize> {
    pub moment_1_values: Tensor<B, 1, Int>,
    pub moment_1_scales: Tensor<B, 1>,
    pub moment_2_values: Tensor<B, 1, Int>,
    pub moment_2_scales: Tensor<B, 1>,
    pub master: Tensor<B, D>,
    pub time: usize,
}

fn pack<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> (Tensor<B, 1, Int>, Tensor<B, 1>) {
    let device = tensor.device();
    let data = tensor.into_data();
    let values = data.to_vec::<f32>().expect("AdamW8bit state must be F32");
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    let mut scales = Vec::with_capacity(values.len().div_ceil(BLOCK_SIZE));
    for block in values.chunks(BLOCK_SIZE) {
        let max_abs = block.iter().copied().filter(|v| v.is_finite()).map(f32::abs).fold(0.0, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);
        for chunk in block.chunks(4) {
            let mut bytes = [0_u8; 4];
            for (index, value) in chunk.iter().enumerate() {
                let value = if value.is_finite() { *value } else { 0.0 };
                bytes[index] = ((value / scale).round().clamp(-127.0, 127.0) as i8) as u8;
            }
            packed.push(i32::from_le_bytes(bytes));
        }
    }
    (
        Tensor::from_data(TensorData::new(packed, [values.len().div_ceil(4)]), &device),
        // bitsandbytes keeps block scales in FP32 even when the trainable
        // model uses BF16.  Keeping this tensor explicitly FP32 avoids
        // quantizing the quantizer and losing small LoRA updates.
        Tensor::<B, 1>::from_data(
            TensorData::new(scales, [values.len().div_ceil(BLOCK_SIZE)]),
            &device,
        )
        .cast(DType::F32),
    )
}

fn unpack<B: Backend, const D: usize>(values: Tensor<B, 1, Int>, scales: Tensor<B, 1>, shape: [usize; D]) -> Tensor<B, D> {
    let device = values.device();
    let words = values.into_data().to_vec::<i32>().expect("invalid packed state");
    let scales = scales
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("invalid scale state");
    let length: usize = shape.iter().product();
    let mut output = Vec::with_capacity(length);
    for (word_index, word) in words.iter().enumerate() {
        let scale = scales.get((word_index * 4) / BLOCK_SIZE).copied().unwrap_or(1.0);
        for byte in word.to_le_bytes() {
            if output.len() == length { break; }
            output.push(i8::from_le_bytes([byte]) as f32 * scale);
        }
    }
    Tensor::from_data(TensorData::new(output, shape), &device)
}

impl<B: Backend> SimpleOptimizer<B> for AdamW8bit {
    type State<const D: usize> = AdamW8bitState<B, D>;

    fn step<const D: usize>(&self, lr: LearningRate, tensor: Tensor<B, D>, grad: Tensor<B, D>, state: Option<Self::State<D>>) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let dtype = tensor.dtype();
        let finite = |value: Tensor<B, D>| value.clone().mask_fill(value.is_finite().bool_not(), 0.0).cast(DType::F32);
        let grad = finite(grad.cast(DType::F32));
        let shape = tensor.dims();
        let (master, moment_1, moment_2, time) = match state {
            Some(state) => {
                let m1 = unpack(state.moment_1_values, state.moment_1_scales, shape);
                let m2 = unpack(state.moment_2_values, state.moment_2_scales, shape);
                (finite(state.master), finite(m1).mul_scalar(self.beta_1).add(grad.clone().mul_scalar(1.0 - self.beta_1)), finite(m2).mul_scalar(self.beta_2).add(grad.powi_scalar(2).mul_scalar(1.0 - self.beta_2)), state.time + 1)
            }
            None => (tensor.cast(DType::F32), grad.clone().mul_scalar(1.0 - self.beta_1), grad.powi_scalar(2).mul_scalar(1.0 - self.beta_2), 1),
        };
        let time = time as i32;
        let update = moment_1.clone().div_scalar(1.0 - self.beta_1.powi(time)).div(moment_2.clone().div_scalar(1.0 - self.beta_2.powi(time)).sqrt().add_scalar(self.epsilon));
        let master = finite(master.mul_scalar(1.0 - lr * self.weight_decay as f64).sub(update.mul_scalar(lr)));
        let (m1_values, m1_scales) = pack(moment_1);
        let (m2_values, m2_scales) = pack(moment_2);
        (master.clone().cast(dtype), Some(AdamW8bitState::new(m1_values, m1_scales, m2_values, m2_scales, master, time as usize)))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.moment_1_values = state.moment_1_values.to_device(device);
        state.moment_1_scales = state.moment_1_scales.to_device(device);
        state.moment_2_values = state.moment_2_values.to_device(device);
        state.moment_2_scales = state.moment_2_scales.to_device(device);
        state.master = state.master.to_device(device);
        state
    }
}

impl AdamW8bitConfig {
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> OptimizerAdaptor<AdamW8bit, M, B> {
        OptimizerAdaptor::from(AdamW8bit { beta_1: self.beta_1, beta_2: self.beta_2, epsilon: self.epsilon, weight_decay: self.weight_decay })
    }
}
