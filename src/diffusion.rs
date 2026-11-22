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
    /// Kohya's discrete `sigma` sampler: draw a logit-normal quantile,
    /// select the corresponding descending scheduler entry, then apply the
    /// flow shift. This is intentionally distinct from the continuous
    /// sigmoid sampler because the scheduler reverses the timestep axis.
    Sigma {
        mean: f32,
        std: f32,
    },
    Sigmoid {
        mean: f32,
        std: f32,
    },
    /// Logit-normal sigma sampling used by kohya_ss Anima training.
    LogitNormal {
        mean: f32,
        std: f32,
    },
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

    /// Samples a continuous sigma in `[0, 1]`. Flow-matching schedules apply
    /// their shift after sampling, matching kohya_ss's sigma path.
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
            TimestepSampling::Sigma { mean, std } => {
                if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                    return Err(DiffusionError::InvalidSchedule(
                        "sigma timestep mean must be finite and std must be positive".into(),
                    ));
                }
                let radius = (-2.0 * first.ln()).sqrt();
                let normal = radius * (2.0 * std::f32::consts::PI * second).cos();
                // sd3_train_utils builds timesteps in descending order and
                // indexes them with floor(u * 1000). The equivalent
                // continuous value is therefore 1 - sigmoid(normal).
                let quantile = 1.0 / (1.0 + (-(mean + std * normal)).exp());
                1.0 - quantile
            }
            TimestepSampling::Sigmoid { mean, std } => {
                if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                    return Err(DiffusionError::InvalidSchedule(
                        "sigmoid timestep mean must be finite and std must be positive".into(),
                    ));
                }
                let radius = (-2.0 * first.ln()).sqrt();
                let normal = radius * (2.0 * std::f32::consts::PI * second).cos();
                // Logistic sigmoid: 1 / (1 + exp(-x)). Keep the negation
                // inside exp; `1 + -exp(x)` collapses most samples to the
                // lower clamp boundary and silently trains at sigma=0.
                1.0 / (1.0 + (-(mean + std * normal)).exp())
            }
            TimestepSampling::LogitNormal { mean, std } => {
                if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                    return Err(DiffusionError::InvalidSchedule(
                        "logit-normal timestep mean must be finite and std must be positive".into(),
                    ));
                }
                let radius = (-2.0 * first.ln()).sqrt();
                let normal = radius * (2.0 * std::f32::consts::PI * second).cos();
                1.0 / (1.0 + (-(mean + std * normal)).exp())
            }
        };
        let value = match self.kind {
            ScheduleKind::FlowMatching { shift } => shift * value / (1.0 + (shift - 1.0) * value),
            ScheduleKind::Ddpm => value,
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
            // kohya_ss samples `timesteps = sigmas * num_train_timesteps`,
            // casts that tensor to the mixed-precision input dtype, and only
            // then builds the sinusoidal embedding. Reproduce that rounding
            // before creating the feature vector instead of embedding the
            // unrounded continuous sigma directly.
            timestep_features: timestep_embedding_from_sampled_sigma(sigma, steps, 2048),
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
    let half_dimension = dimension / 2;
    for index in 0..half_dimension {
        // Match kohya_ss Anima's TimestepEmbedder exactly:
        // exponent = -log(10000) * arange(half_dim) / half_dim,
        // followed by concat(cos(embedding), sin(embedding)).
        let frequency =
            (-std::f32::consts::LN_10 * 4.0 * index as f32 / half_dimension as f32).exp();
        let angle = normalized * frequency;
        values.push(angle.cos());
    }
    for index in 0..half_dimension {
        let frequency =
            (-std::f32::consts::LN_10 * 4.0 * index as f32 / half_dimension as f32).exp();
        let angle = normalized * frequency;
        values.push(angle.sin());
    }
    if dimension % 2 != 0 {
        values.push(0.0);
    }
    VaeTensor {
        shape: vec![1, dimension],
        data: values,
    }
}

/// Builds Anima timestep features from kohya_ss's continuous sigma sampler.
///
/// The Python trainer first forms `sigma * steps`, stores it in the selected
/// mixed-precision dtype (FP16 for the reference run), divides by `steps` in
/// the timestep embedder, and computes the sinusoidal features from that
/// rounded value. The explicit FP16 round here keeps the feature input on the
/// same side of the quantization boundary as the reference implementation.
fn timestep_embedding_from_sampled_sigma(sigma: f32, steps: usize, dimension: usize) -> VaeTensor {
    let sampled_timestep = half::f16::from_f32(sigma * steps as f32).to_f32();
    timestep_embedding_value(sampled_timestep / steps.max(1) as f32, dimension)
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
    fn timestep_features_match_anima_cosine_then_sine_layout() {
        let features = timestep_embedding_value(0.25, 4);
        let expected = vec![
            0.25_f32.cos(),
            (0.25_f32 * 0.01).cos(),
            0.25_f32.sin(),
            (0.25_f32 * 0.01).sin(),
        ];
        for (actual, expected) in features.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn flow_cache_uses_normalized_timestep_features() {
        let sample = CachedTrainingSample::new(
            "image-01",
            0,
            tensor(vec![0.1, 0.2]),
            tensor(vec![1.0, 2.0]),
        );
        let example =
            TrainingExample::from_cache_with_sigma(&sample, 0.5, 1000, tensor(vec![0.3, -0.4]))
                .unwrap();
        let rounded_timestep = half::f16::from_f32(0.5 * 1000.0).to_f32() / 1000.0;
        let expected = timestep_embedding_value(rounded_timestep, 2048);
        assert_eq!(example.timestep, 500);
        assert_eq!(example.timestep_features.data, expected.data);
    }

    #[test]
    fn flow_cache_matches_kohya_probe_step_three_reference_values() {
        // First four values copied from the existing Kohya probe dump at
        // outputs/afterburner-probe/step-000003. The probe tensors are BF16,
        // so a small tolerance is intentional when comparing reconstructed
        // FP32 arithmetic.
        let sample = CachedTrainingSample::new(
            "kohya-step3",
            0,
            tensor(vec![0.0, 0.0, 0.0, 0.0]),
            tensor(vec![
                0.320068359375,
                0.433349609375,
                0.401123046875,
                0.44970703125,
            ]),
        );
        let sigma = 0.40445560216903687_f32;
        let noise = tensor(vec![-0.57763671875, 1.1416015625, 0.642578125, 0.369140625]);
        let example = TrainingExample::from_cache_with_sigma(&sample, sigma, 1000, noise).unwrap();
        let expected_noisy = vec![
            -0.042999267578125_f32,
            0.7197265625,
            0.498779296875,
            0.417236328125,
        ];
        let expected_target = vec![
            -0.8974609375_f32,
            0.7080078125,
            0.241455078125,
            -0.08056640625,
        ];
        for (actual, expected) in example.noisy_latent.data.iter().zip(expected_noisy) {
            assert!((actual - expected).abs() < 2e-3);
        }
        for (actual, expected) in example.target.data.iter().zip(expected_target) {
            assert!((actual - expected).abs() < 2e-3);
        }
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

    #[test]
    fn sigmoid_timestep_sampling_uses_logistic_sign_correctly() {
        let schedule = DiffusionSchedule::flow_matching(1000, 1.0).unwrap();
        let sigma = schedule
            .sample_sigma(
                0.5682303265,
                0.2254634289,
                TimestepSampling::Sigmoid {
                    mean: 0.0,
                    std: 1.0,
                },
            )
            .unwrap();
        assert!((sigma - 0.5407264).abs() < 1e-5, "sigma={sigma}");
    }

    #[test]
    fn sigma_sampling_matches_descending_kohya_scheduler_axis() {
        let schedule = DiffusionSchedule::flow_matching(1000, 3.0).unwrap();
        let sigma = schedule
            .sample_sigma(
                0.5,
                0.5,
                TimestepSampling::Sigma {
                    mean: 0.0,
                    std: 1.0,
                },
            )
            .unwrap();
        // Box-Muller at (0.5, 0.5) produces a non-zero normal variate because
        // cos(2*pi*0.5) = -1. The descending scheduler and shift=3 mapping
        // therefore produce the following deterministic reference value.
        assert!((sigma - 0.9068716).abs() < 1e-6, "sigma={sigma}");
    }
}
