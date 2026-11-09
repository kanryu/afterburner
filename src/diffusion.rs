//! DiT/UNet training input construction from cached latents.
//!
//! This module deliberately does not depend on an image, caption, VAE, or
//! text encoder. Its input is the cache produced during preprocessing. The
//! resulting example is ready to be converted to Burn tensors and passed to a
//! trainable diffusion model.

use crate::{cache::CachedTrainingSample, vae::VaeTensor};

/// The target predicted by the diffusion model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingTarget {
    /// Predict the Gaussian noise added to the clean latent.
    Epsilon,
    /// Predict the velocity target used by flow-matching style models.
    VPrediction,
    /// Flow-matching target: the direction from the clean latent to noise.
    FlowMatching,
}

/// Per-example loss weighting used by Anima's flow-matching trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossWeighting {
    None,
    SigmaSqrt,
    Cosmap,
}

/// Timestep distributions used by flow-matching training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimestepSampling {
    Uniform,
    Sigmoid { mean: f32, std: f32 },
}

/// Errors raised while creating a diffusion training example.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffusionError {
    InvalidSchedule(String),
    InvalidTimestep {
        timestep: usize,
        steps: usize,
    },
    ShapeMismatch {
        latent: Vec<usize>,
        noise: Vec<usize>,
    },
}

impl std::fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSchedule(message) => write!(f, "invalid diffusion schedule: {message}"),
            Self::InvalidTimestep { timestep, steps } => {
                write!(
                    f,
                    "timestep {timestep} is outside schedule with {steps} steps"
                )
            }
            Self::ShapeMismatch { latent, noise } => {
                write!(f, "latent/noise shape mismatch: {latent:?} vs {noise:?}")
            }
        }
    }
}

impl std::error::Error for DiffusionError {}

/// A simple DDPM-style schedule represented by cumulative alpha products.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionSchedule {
    alpha_cumprod: Vec<f32>,
    kind: ScheduleKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScheduleKind {
    Ddpm,
    FlowMatching { shift: f32 },
}

impl DiffusionSchedule {
    /// Builds a linear-beta schedule with `steps` entries.
    pub fn linear(steps: usize, beta_start: f32, beta_end: f32) -> Result<Self, DiffusionError> {
        if steps == 0 {
            return Err(DiffusionError::InvalidSchedule(
                "steps must be greater than zero".into(),
            ));
        }
        if !beta_start.is_finite()
            || !beta_end.is_finite()
            || beta_start <= 0.0
            || beta_end <= beta_start
            || beta_end >= 1.0
        {
            return Err(DiffusionError::InvalidSchedule(
                "beta values must satisfy 0 < beta_start < beta_end < 1".into(),
            ));
        }

        let mut alpha_cumprod = Vec::with_capacity(steps);
        let mut cumulative = 1.0;
        for index in 0..steps {
            let ratio = if steps == 1 {
                0.0
            } else {
                index as f32 / (steps - 1) as f32
            };
            let beta = beta_start + ratio * (beta_end - beta_start);
            cumulative *= 1.0 - beta;
            alpha_cumprod.push(cumulative);
        }
        Ok(Self {
            alpha_cumprod,
            kind: ScheduleKind::Ddpm,
        })
    }

    /// Builds the linear flow-matching schedule used by Anima-style DiT
    /// training. `shift=1` is the identity mapping; other positive shifts
    /// move the uniform sigma samples toward the requested end of the path.
    pub fn flow_matching(steps: usize, shift: f32) -> Result<Self, DiffusionError> {
        if steps == 0 || !shift.is_finite() || shift <= 0.0 {
            return Err(DiffusionError::InvalidSchedule(
                "flow-matching steps must be non-zero and shift must be finite and positive".into(),
            ));
        }
        Ok(Self {
            alpha_cumprod: vec![0.0; steps],
            kind: ScheduleKind::FlowMatching { shift },
        })
    }

    pub fn steps(&self) -> usize {
        self.alpha_cumprod.len()
    }

    /// Adds a supplied noise tensor to a cached clean latent.
    pub fn add_noise(
        &self,
        latent: &VaeTensor,
        noise: &VaeTensor,
        timestep: usize,
    ) -> Result<VaeTensor, DiffusionError> {
        let (alpha, sigma) = self.coefficients(timestep)?;
        if latent.shape != noise.shape {
            return Err(DiffusionError::ShapeMismatch {
                latent: latent.shape.clone(),
                noise: noise.shape.clone(),
            });
        }
        let data = latent
            .data
            .iter()
            .zip(&noise.data)
            .map(|(clean, noise)| alpha * clean + sigma * noise)
            .collect();
        Ok(VaeTensor {
            shape: latent.shape.clone(),
            data,
        })
    }

    pub fn coefficients(&self, timestep: usize) -> Result<(f32, f32), DiffusionError> {
        if let ScheduleKind::FlowMatching { shift } = self.kind {
            if timestep >= self.steps() {
                return Err(DiffusionError::InvalidTimestep {
                    timestep,
                    steps: self.steps(),
                });
            }
            let normalized = if self.steps() <= 1 {
                0.0
            } else {
                timestep as f32 / (self.steps() - 1) as f32
            };
            let sigma = shift * normalized / (1.0 + (shift - 1.0) * normalized);
            return Ok((1.0 - sigma, sigma));
        }
        let Some(&alpha_cumprod) = self.alpha_cumprod.get(timestep) else {
            return Err(DiffusionError::InvalidTimestep {
                timestep,
                steps: self.steps(),
            });
        };
        Ok((alpha_cumprod.sqrt(), (1.0 - alpha_cumprod).sqrt()))
    }

    /// Converts two uniform random values into a schedule timestep.
    ///
    /// The sigmoid mode uses a logit-normal sample, matching the sampling
    /// parameters used by the Anima kohya_ss configuration.
    pub fn sample_timestep(
        &self,
        first_uniform: f32,
        second_uniform: f32,
        sampling: TimestepSampling,
    ) -> Result<usize, DiffusionError> {
        if self.steps() == 0 {
            return Err(DiffusionError::InvalidSchedule("schedule is empty".into()));
        }
        let value = self.sample_sigma(first_uniform, second_uniform, sampling)?;
        Ok(((value * self.steps() as f32).floor() as usize).min(self.steps() - 1))
    }

    /// Samples a continuous sigma in `[0, 1]`. Sigmoid sampling intentionally
    /// does not apply `discrete_flow_shift`, matching kohya_ss.
    pub fn sample_sigma(
        &self,
        first_uniform: f32,
        second_uniform: f32,
        sampling: TimestepSampling,
    ) -> Result<f32, DiffusionError> {
        let first = first_uniform.clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let second = second_uniform.clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let value = match sampling {
            TimestepSampling::Uniform => first,
            TimestepSampling::Sigmoid { mean, std } => {
                if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                    return Err(DiffusionError::InvalidSchedule(
                        "sigmoid timestep mean must be finite and std must be positive".into(),
                    ));
                }
                let radius = (-2.0 * first.ln()).sqrt();
                let normal = radius * (2.0 * std::f32::consts::PI * second).cos();
                1.0 / (1.0 + -(mean + std * normal).exp())
            }
        };
        Ok(value.clamp(0.0, 1.0))
    }
}

/// All inputs required by one DiT/UNet training step.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingExample {
    pub sample_id: String,
    pub timestep: usize,
    pub noisy_latent: VaeTensor,
    pub text_embedding: VaeTensor,
    pub text_attention_mask: Option<VaeTensor>,
    pub adapter_input_ids: Option<Vec<u32>>,
    pub adapter_attention_mask: Option<VaeTensor>,
    pub target: VaeTensor,
    /// Sinusoidal timestep features consumed by the DiT time embedder.
    pub timestep_features: VaeTensor,
    pub target_kind: TrainingTarget,
    /// Continuous flow sigma, when the example was sampled from flow matching.
    pub sigma: Option<f32>,
}

impl TrainingExample {
    /// Constructs an epsilon-prediction example from a cached sample.
    pub fn from_cache(
        sample: &CachedTrainingSample,
        schedule: &DiffusionSchedule,
        timestep: usize,
        noise: VaeTensor,
    ) -> Result<Self, DiffusionError> {
        Self::from_cache_with_target(sample, schedule, timestep, noise, TrainingTarget::Epsilon)
    }

    /// Constructs an example for either epsilon or velocity prediction.
    pub fn from_cache_with_target(
        sample: &CachedTrainingSample,
        schedule: &DiffusionSchedule,
        timestep: usize,
        noise: VaeTensor,
        target_kind: TrainingTarget,
    ) -> Result<Self, DiffusionError> {
        let noisy_latent = schedule.add_noise(&sample.image_latent, &noise, timestep)?;
        let target = match target_kind {
            TrainingTarget::Epsilon => noise,
            TrainingTarget::VPrediction => {
                let (alpha, sigma) = schedule.coefficients(timestep)?;
                VaeTensor {
                    shape: sample.image_latent.shape.clone(),
                    data: sample
                        .image_latent
                        .data
                        .iter()
                        .zip(&noise.data)
                        .map(|(clean, noise)| alpha * noise - sigma * clean)
                        .collect(),
                }
            }
            TrainingTarget::FlowMatching => VaeTensor {
                shape: sample.image_latent.shape.clone(),
                data: noise
                    .data
                    .iter()
                    .zip(&sample.image_latent.data)
                    .map(|(noise, clean)| noise - clean)
                    .collect(),
            },
        };
        Ok(Self {
            sample_id: sample.sample_id.clone(),
            timestep,
            noisy_latent,
            text_embedding: sample.text_embedding.clone(),
            text_attention_mask: sample.text_attention_mask.clone(),
            adapter_input_ids: sample
                .adapter_conditioning
                .as_ref()
                .map(|value| value.input_ids.clone()),
            adapter_attention_mask: sample
                .adapter_conditioning
                .as_ref()
                .map(|value| value.attention_mask.clone()),
            target,
            timestep_features: timestep_embedding(timestep, schedule.steps(), 2048),
            target_kind,
            sigma: None,
        })
    }

    /// Constructs a flow-matching example from a continuous sigma sample.
    pub fn from_cache_with_sigma(
        sample: &CachedTrainingSample,
        sigma: f32,
        steps: usize,
        noise: VaeTensor,
    ) -> Result<Self, DiffusionError> {
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) || steps == 0 {
            return Err(DiffusionError::InvalidSchedule(
                "sigma must be finite in [0, 1] and steps must be non-zero".into(),
            ));
        }
        if sample.image_latent.shape != noise.shape {
            return Err(DiffusionError::ShapeMismatch {
                latent: sample.image_latent.shape.clone(),
                noise: noise.shape.clone(),
            });
        }
        let noisy_latent = VaeTensor {
            shape: sample.image_latent.shape.clone(),
            data: sample
                .image_latent
                .data
                .iter()
                .zip(&noise.data)
                .map(|(clean, noise)| (1.0 - sigma) * clean + sigma * noise)
                .collect(),
        };
        let target = VaeTensor {
            shape: sample.image_latent.shape.clone(),
            data: noise
                .data
                .iter()
                .zip(&sample.image_latent.data)
                .map(|(noise, clean)| noise - clean)
                .collect(),
        };
        let timestep = (sigma * steps as f32).floor() as usize;
        Ok(Self {
            sample_id: sample.sample_id.clone(),
            timestep,
            noisy_latent,
            text_embedding: sample.text_embedding.clone(),
            text_attention_mask: sample.text_attention_mask.clone(),
            adapter_input_ids: sample
                .adapter_conditioning
                .as_ref()
                .map(|value| value.input_ids.clone()),
            adapter_attention_mask: sample
                .adapter_conditioning
                .as_ref()
                .map(|value| value.attention_mask.clone()),
            target,
            timestep_features: timestep_embedding_value(sigma, 2048),
            target_kind: TrainingTarget::FlowMatching,
            sigma: Some(sigma),
        })
    }
}

/// Builds the 2048-wide sinusoidal features expected by Anima's timestep MLP.
pub fn timestep_embedding(timestep: usize, steps: usize, dimension: usize) -> VaeTensor {
    timestep_embedding_value(timestep as f32 / steps.max(1) as f32, dimension)
}

fn timestep_embedding_value(normalized: f32, dimension: usize) -> VaeTensor {
    let mut values = Vec::with_capacity(dimension);
    for index in 0..dimension / 2 {
        let frequency = (-std::f32::consts::LN_10 * 4.0 * index as f32 / dimension as f32).exp();
        let angle = normalized * frequency;
        values.push(angle.sin());
        values.push(angle.cos());
    }
    if dimension % 2 != 0 {
        values.push(0.0);
    }
    VaeTensor {
        shape: vec![1, dimension],
        data: values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: Vec<f32>) -> VaeTensor {
        VaeTensor::new(vec![values.len()], values).unwrap()
    }

    #[test]
    fn schedule_adds_noise_without_source_models() {
        let schedule = DiffusionSchedule::linear(10, 0.0001, 0.02).unwrap();
        let latent = tensor(vec![1.0, 2.0]);
        let noise = tensor(vec![0.5, -0.5]);
        let result = schedule.add_noise(&latent, &noise, 3).unwrap();
        assert_eq!(result.shape, vec![2]);
        assert_ne!(result.data, latent.data);
    }

    #[test]
    fn cached_sample_becomes_training_example() {
        let sample = CachedTrainingSample::new(
            "image-01",
            0,
            tensor(vec![0.1, 0.2]),
            tensor(vec![1.0, 2.0]),
        );
        let schedule = DiffusionSchedule::linear(10, 0.0001, 0.02).unwrap();
        let example =
            TrainingExample::from_cache(&sample, &schedule, 0, tensor(vec![0.5, -0.5])).unwrap();
        assert_eq!(example.sample_id, "image-01");
        assert_eq!(example.target_kind, TrainingTarget::Epsilon);
        assert_eq!(example.text_embedding.data, vec![0.1, 0.2]);
        assert_eq!(example.timestep_features.shape, vec![1, 2048]);
    }

    #[test]
    fn rejects_wrong_noise_shape() {
        let schedule = DiffusionSchedule::linear(10, 0.0001, 0.02).unwrap();
        let error = schedule
            .add_noise(&tensor(vec![1.0, 2.0]), &tensor(vec![1.0]), 0)
            .unwrap_err();
        assert!(matches!(error, DiffusionError::ShapeMismatch { .. }));
    }

    #[test]
    fn builds_velocity_target_from_cached_latent() {
        let sample = CachedTrainingSample::new(
            "image-01",
            0,
            tensor(vec![0.1, 0.2]),
            tensor(vec![1.0, 2.0]),
        );
        let schedule = DiffusionSchedule::linear(10, 0.0001, 0.02).unwrap();
        let example = TrainingExample::from_cache_with_target(
            &sample,
            &schedule,
            3,
            tensor(vec![0.5, -0.5]),
            TrainingTarget::VPrediction,
        )
        .unwrap();
        assert_eq!(example.target_kind, TrainingTarget::VPrediction);
        assert_ne!(example.target.data, vec![0.5, -0.5]);
    }

    #[test]
    fn flow_matching_schedule_builds_direction_target() {
        let schedule = DiffusionSchedule::flow_matching(10, 1.0).unwrap();
        let sample =
            CachedTrainingSample::new("sample", 0, tensor(vec![0.0]), tensor(vec![0.25, -0.5]));
        let example = TrainingExample::from_cache_with_target(
            &sample,
            &schedule,
            5,
            tensor(vec![1.0, 0.5]),
            TrainingTarget::FlowMatching,
        )
        .unwrap();
        assert!((example.noisy_latent.data[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((example.noisy_latent.data[1] - 1.0 / 18.0).abs() < 1e-6);
        assert_eq!(example.target.data, vec![0.75, 1.0]);
        assert_eq!(example.target_kind, TrainingTarget::FlowMatching);
    }

    #[test]
    fn sigmoid_timestep_sampling_stays_within_schedule() {
        let schedule = DiffusionSchedule::flow_matching(1000, 1.0).unwrap();
        let timestep = schedule
            .sample_timestep(
                0.25,
                0.75,
                TimestepSampling::Sigmoid {
                    mean: 0.0,
                    std: 1.0,
                },
            )
            .unwrap();
        assert!(timestep < 1000);
        assert!(
            schedule
                .sample_timestep(
                    0.5,
                    0.5,
                    TimestepSampling::Sigmoid {
                        mean: 0.0,
                        std: 0.0
                    }
                )
                .is_err()
        );
    }
}
