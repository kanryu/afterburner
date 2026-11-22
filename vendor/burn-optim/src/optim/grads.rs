use burn_core as burn;

#[cfg(feature = "collective")]
use burn_collective::{CollectiveError, PeerId, ReduceOperation, all_reduce};

use burn::{
    Tensor,
    tensor::{
        backend::{AutodiffBackend, Backend},
        container::TensorContainer,
        ElementConversion, FloatDType, TensorPrimitive,
    },
};

use burn::module::{AutodiffModule, ParamId};

use super::visitor::{GradientsParamsChangeDevice, GradientsParamsConverter};

/// Data type that contains gradients for parameters.
#[derive(Default, Debug)]
pub struct GradientsParams {
    container: TensorContainer<ParamId>,
}

impl GradientsParams {
    /// Creates a new [GradientsParams](GradientsParams).
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    ///
    /// Note: This consumes the gradients. See ['from_module'] to extract gradients only for
    ///  a specific module.
    pub fn from_grads<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    pub fn from_module<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](AutodiffModule) and given parameters.
    pub fn from_params<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
        params: &[ParamId],
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor =
            GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, Some(params.to_vec()));
        module.visit(&mut visitor);
        grads_params
    }

    /// Get the gradients for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// You should use [remove](GradientsParams::remove) if you want to get the gradients
    /// only one time.
    pub fn get<B, const D: usize>(&self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.get(&id).map(Tensor::from_primitive)
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    pub fn remove<B, const D: usize>(&mut self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.remove(&id).map(Tensor::from_primitive)
    }

    /// Register a gradients tensor for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter id](ParamId), it will be replaced.
    pub fn register<B, const D: usize>(&mut self, id: ParamId, value: Tensor<B, D>)
    where
        B: Backend,
    {
        self.container.register(id, value.into_primitive())
    }

    /// The number of gradients tensors registered.
    pub fn len(&self) -> usize {
        self.container.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clips all registered gradients using one global L2 norm.
    ///
    /// PyTorch/Accelerate applies `max_grad_norm` to the complete parameter
    /// set. Applying the same operation independently to every tensor changes
    /// the optimizer trajectory substantially for LoRA models, which contain
    /// hundreds of factor tensors.
    pub fn clip_by_global_norm<B: Backend>(mut self, max_norm: f32) -> Self {
        if max_norm <= 0.0 || self.container.is_empty() {
            return self;
        }

        let ids = self.container.ids().into_iter().copied().collect::<Vec<_>>();
        let mut values = Vec::with_capacity(ids.len());
        let mut sum_squares = None;

        for id in ids {
            let Some(primitive) = self.container.remove::<B>(&id) else {
                continue;
            };
            let value = match &primitive {
                TensorPrimitive::Float(value) => value.clone(),
                TensorPrimitive::QFloat(value) => B::dequantize(value.clone()),
            };
            let value = B::float_cast(value, FloatDType::F32);
            let local_sum = B::float_sum(B::float_mul(value.clone(), value));
            sum_squares = Some(match sum_squares {
                Some(total) => B::float_add(total, local_sum),
                None => local_sum,
            });
            values.push((id, primitive));
        }

        let Some(sum_squares) = sum_squares else {
            return self;
        };
        let norm = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(sum_squares))
            .sqrt()
            .into_data()
            .to_vec::<f32>()
            .ok()
            .and_then(|values| values.first().copied())
            .unwrap_or(0.0);
        let scale = (max_norm / (norm + 1.0e-6)).min(1.0);

        for (id, primitive) in values {
            let value = match primitive {
                TensorPrimitive::Float(value) => value,
                TensorPrimitive::QFloat(value) => B::dequantize(value),
            };
            self.container.register::<B>(
                id,
                TensorPrimitive::Float(B::float_mul_scalar(value, scale.elem())),
            );
        }
        self
    }

    /// Change the device of each tensor gradients registered for the given [module](AutodiffModule).
    pub fn to_device<B: AutodiffBackend, M: AutodiffModule<B>>(
        mut self,
        device: &B::Device,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M, B>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }

    /// Syncs the gradient params with the other peers in the collective.
    #[cfg(feature = "collective")]
    pub fn all_reduce<B: Backend>(
        mut self,
        peer_id: PeerId,
        op: ReduceOperation,
    ) -> Result<Self, CollectiveError> {
        let mut ids = self
            .container
            .ids()
            .into_iter()
            .copied()
            .collect::<Vec<ParamId>>();
        // This is crucial, since the all-reduce operations need to happen in the same order for the same parameters on all nodes!
        ids.sort();

        for id in ids {
            let Some(grad) = self.container.remove::<B>(&id) else {
                todo!()
            };

            let grad = match grad {
                burn::tensor::TensorPrimitive::Float(grad) => {
                    let grad = all_reduce::<B>(peer_id, grad, op)?;
                    burn::tensor::TensorPrimitive::Float(grad)
                }
                burn::tensor::TensorPrimitive::QFloat(_grad) => {
                    unimplemented!("quantized all-reduce unimplemented")
                }
            };

            self.container.register::<B>(id, grad);
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use burn::module::{Module, list_param_ids};
    use burn::tensor::{Distribution, backend::Backend};
    use burn_nn::{Linear, LinearConfig};

    #[test]
    fn test_convert_grads() {
        let device = Default::default();
        let layer_1 = layer::<TestAutodiffBackend>(&device);
        let mut layer_2 = layer_1.clone();
        layer_2 = layer_2.fork(&device);
        let loss_1 = layer_1.forward(random_tensor(&device));
        let loss_2 = layer_2.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer_1);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer_2);

        let param_ids_1 = list_param_ids(&layer_1);
        let param_ids_2 = list_param_ids(&layer_2);

        assert_eq!(param_ids_1, param_ids_2);
        assert_eq!(grads_1.len(), param_ids_1.len());
        assert_eq!(grads_2.len(), param_ids_2.len());
    }

    fn layer<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
