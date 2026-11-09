//! Dataset-to-cache preprocessing boundary.

use std::collections::HashMap;

use crate::{
    cache::{AdapterConditioning, CachedTrainingSample, InMemoryTrainingCache},
    dataset::{DatasetError, ImageCaptionDataset},
    logging::{NullLogger, TrainingLogger},
    vae::VaeTensor,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreprocessError {
    Dataset(String),
    Image(String),
    Encoder(String),
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset(message) => write!(f, "preprocessing dataset error: {message}"),
            Self::Image(message) => write!(f, "preprocessing image error: {message}"),
            Self::Encoder(message) => write!(f, "preprocessing encoder error: {message}"),
        }
    }
}

impl std::error::Error for PreprocessError {}

impl From<DatasetError> for PreprocessError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error.to_string())
    }
}

/// Supplies text embeddings during preprocessing.
pub trait TextEncoder {
    fn encode_text(&mut self, caption: &str) -> Result<VaeTensor, PreprocessError>;

    fn encode_text_with_mask(
        &mut self,
        caption: &str,
    ) -> Result<(VaeTensor, Option<VaeTensor>), PreprocessError> {
        Ok((self.encode_text(caption)?, None))
    }

    fn encode_text_with_conditioning(
        &mut self,
        caption: &str,
    ) -> Result<TextEncoding, PreprocessError> {
        let (embedding, attention_mask) = self.encode_text_with_mask(caption)?;
        Ok(TextEncoding {
            embedding,
            attention_mask,
            adapter_conditioning: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextEncoding {
    pub embedding: VaeTensor,
    pub attention_mask: Option<VaeTensor>,
    pub adapter_conditioning: Option<AdapterConditioning>,
}

/// Supplies VAE latents during preprocessing.
pub trait ImageVae {
    fn encode_image(&mut self, image: &image::RgbImage) -> Result<VaeTensor, PreprocessError>;
}

/// Combined compatibility boundary for callers that own both models.
pub trait TrainingPreprocessor: TextEncoder + ImageVae {}

impl<T: TextEncoder + ImageVae> TrainingPreprocessor for T {}

/// Composes independently implemented text and image encoders.
pub struct CompositePreprocessor<T, V> {
    pub text_encoder: T,
    pub vae: V,
}

impl<T, V> CompositePreprocessor<T, V> {
    pub fn new(text_encoder: T, vae: V) -> Self {
        Self { text_encoder, vae }
    }
}

impl<T: TextEncoder, V: ImageVae> TextEncoder for CompositePreprocessor<T, V> {
    fn encode_text(&mut self, caption: &str) -> Result<VaeTensor, PreprocessError> {
        self.text_encoder.encode_text(caption)
    }

    fn encode_text_with_mask(
        &mut self,
        caption: &str,
    ) -> Result<(VaeTensor, Option<VaeTensor>), PreprocessError> {
        self.text_encoder.encode_text_with_mask(caption)
    }

    fn encode_text_with_conditioning(
        &mut self,
        caption: &str,
    ) -> Result<TextEncoding, PreprocessError> {
        self.text_encoder.encode_text_with_conditioning(caption)
    }
}

impl<T: TextEncoder, V: ImageVae> ImageVae for CompositePreprocessor<T, V> {
    fn encode_image(&mut self, image: &image::RgbImage) -> Result<VaeTensor, PreprocessError> {
        self.vae.encode_image(image)
    }
}

/// Encodes every image/caption pair into the host-memory training cache.
pub fn preprocess_dataset<P: TrainingPreprocessor>(
    dataset: &ImageCaptionDataset,
    preprocessor: &mut P,
) -> Result<InMemoryTrainingCache, PreprocessError> {
    let mut logger = NullLogger;
    preprocess_dataset_with_logger(dataset, preprocessor, &mut logger)
}

/// Encodes the dataset while emitting kohya_ss-inspired cache progress logs.
pub fn preprocess_dataset_with_logger<P: TrainingPreprocessor, L: TrainingLogger>(
    dataset: &ImageCaptionDataset,
    preprocessor: &mut P,
    logger: &mut L,
) -> Result<InMemoryTrainingCache, PreprocessError> {
    let mut cache = InMemoryTrainingCache::new();
    let mut encoded: HashMap<
        _,
        (
            VaeTensor,
            Option<VaeTensor>,
            Option<AdapterConditioning>,
            VaeTensor,
        ),
    > = HashMap::new();
    logger.info(&format!(
        "[Dataset 0] caching latents with caching strategy. samples={}",
        dataset.samples().len()
    ));
    for sample in dataset.samples() {
        let key = sample.image_path.clone();
        let (text_embedding, text_attention_mask, adapter_conditioning, image_latent) =
            if let Some(encoded) = encoded.get(&key) {
                encoded.clone()
            } else {
                let image = ImageCaptionDataset::load_rgb(sample)
                    .map_err(|error| PreprocessError::Image(error.to_string()))?;
                logger.info(&format!(
                    "caching latents... {}",
                    sample.image_path.display()
                ));
                let text = preprocessor.encode_text_with_conditioning(&sample.caption)?;
                let value = (
                    text.embedding,
                    text.attention_mask,
                    text.adapter_conditioning,
                    preprocessor.encode_image(&image)?,
                );
                encoded.insert(key, value.clone());
                value
            };
        let cached = CachedTrainingSample::new(
            sample.image_path.to_string_lossy(),
            sample.repeat_index,
            text_embedding,
            image_latent,
        );
        let cached = match text_attention_mask {
            Some(mask) => cached.with_text_attention_mask(mask),
            None => cached,
        };
        cache.push(match adapter_conditioning {
            Some(conditioning) => cached.with_adapter_conditioning(conditioning),
            None => cached,
        });
    }
    logger.info(&format!(
        "caching Text Encoder outputs... {} unique samples",
        encoded.len()
    ));
    cache
        .validate()
        .map_err(|error| PreprocessError::Encoder(error.to_string()))?;
    Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    struct FakePreprocessor {
        text_calls: Rc<Cell<usize>>,
        image_calls: Rc<Cell<usize>>,
    }

    impl TextEncoder for FakePreprocessor {
        fn encode_text(&mut self, caption: &str) -> Result<VaeTensor, PreprocessError> {
            self.text_calls.set(self.text_calls.get() + 1);
            Ok(VaeTensor::new(vec![1, 1], vec![caption.len() as f32]).unwrap())
        }
    }

    impl ImageVae for FakePreprocessor {
        fn encode_image(&mut self, image: &image::RgbImage) -> Result<VaeTensor, PreprocessError> {
            self.image_calls.set(self.image_calls.get() + 1);
            Ok(VaeTensor::new(vec![1, 1], vec![image.width() as f32]).unwrap())
        }
    }

    #[test]
    fn preprocessing_populates_cache_without_training_models() {
        let root =
            std::env::temp_dir().join(format!("afterburner-preprocess-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let image = image::RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 0]));
        image.save(root.join("sample.png")).unwrap();
        std::fs::write(root.join("sample.txt"), "red square").unwrap();
        let dataset = ImageCaptionDataset::from_dir(&root, 2).unwrap();
        let text_calls = Rc::new(Cell::new(0));
        let image_calls = Rc::new(Cell::new(0));
        let mut processor = FakePreprocessor {
            text_calls: text_calls.clone(),
            image_calls: image_calls.clone(),
        };
        let cache = preprocess_dataset(&dataset, &mut processor).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(0).unwrap().image_latent.data, vec![2.0]);
        assert_eq!(text_calls.get(), 1);
        assert_eq!(image_calls.get(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
