//! Host-side blockwise symmetric INT8 quantization used by AdamW8bit.
//!
//! This is deliberately independent of Burn's QTensor CUDA kernels. The
//! bitsandbytes AdamW8bit state uses signed INT8 values with one FP32 scale
//! per block; keeping this representation explicit lets the optimizer test
//! quantization without making model tensors quantized.

pub const BLOCK_SIZE: usize = 4096;

#[derive(Clone, Debug, PartialEq)]
pub struct BlockwiseQ8 {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub len: usize,
}

impl BlockwiseQ8 {
    pub fn zeros(len: usize) -> Self {
        Self::quantize(&vec![0.0; len])
    }

    pub fn quantize(input: &[f32]) -> Self {
        let mut values = Vec::with_capacity(input.len());
        let mut scales = Vec::with_capacity(input.len().div_ceil(BLOCK_SIZE));

        for block in input.chunks(BLOCK_SIZE) {
            let max_abs = block
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .map(f32::abs)
                .fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            values.extend(block.iter().map(|value| {
                if !value.is_finite() {
                    0
                } else {
                    (value / scale).round().clamp(-127.0, 127.0) as i8
                }
            }));
        }

        Self {
            values,
            scales,
            len: input.len(),
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.values
            .chunks(BLOCK_SIZE)
            .enumerate()
            .flat_map(|(block_index, block)| {
                let scale = self.scales.get(block_index).copied().unwrap_or(1.0);
                block.iter().map(move |value| f32::from(*value) * scale)
            })
            .take(self.len)
            .collect()
    }

    pub fn ema_update(&mut self, beta: f32, input: &[f32]) {
        assert_eq!(self.len, input.len(), "blockwise state shape mismatch");
        let mut values = self.dequantize();
        for (state, value) in values.iter_mut().zip(input.iter().copied()) {
            let value = if value.is_finite() { value } else { 0.0 };
            *state = *state * beta + value * (1.0 - beta);
        }
        *self = Self::quantize(&values);
    }
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_SIZE, BlockwiseQ8};

    #[test]
    fn round_trip_is_bounded_by_half_a_quantization_step() {
        let input = (0..(BLOCK_SIZE + 3))
            .map(|index| (index as f32 - 2000.0) / 113.0)
            .collect::<Vec<_>>();
        let quantized = BlockwiseQ8::quantize(&input);
        let output = quantized.dequantize();
        assert_eq!(output.len(), input.len());
        for (index, (source, restored)) in input.iter().zip(output.iter()).enumerate() {
            let block_index = index / BLOCK_SIZE;
            let tolerance = quantized.scales[block_index] * 0.51;
            assert!((source - restored).abs() <= tolerance + 1e-6);
        }
    }

    #[test]
    fn non_finite_values_are_sanitized() {
        let quantized = BlockwiseQ8::quantize(&[f32::NAN, f32::INFINITY, -2.0]);
        assert_eq!(quantized.dequantize()[0], 0.0);
        assert_eq!(quantized.dequantize()[1], 0.0);
        assert!(quantized.dequantize()[2].is_finite());
    }

    #[test]
    fn quantized_ema_state_remains_finite() {
        let mut state = BlockwiseQ8::zeros(BLOCK_SIZE + 1);
        state.ema_update(0.9, &vec![1.0; BLOCK_SIZE + 1]);
        state.ema_update(0.9, &[f32::NAN; BLOCK_SIZE + 1]);
        assert!(state.dequantize().iter().all(|value| value.is_finite()));
        assert!(state.dequantize().iter().all(|value| *value >= 0.0));
    }
}
