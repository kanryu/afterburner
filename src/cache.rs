//! Host-memory cache for preprocessed LoRA training samples.
//!
//! The cache is built once from image/caption pairs. Text embeddings are
//! produced by the text encoder and image latents by the VAE. The training
//! loop then consumes only these tensors and can release the text encoder,
//! VAE, source images, and source captions from its working set.

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::vae::{VaeError, VaeTensor};

const CACHE_MAGIC: &[u8; 8] = b"AFBCACHE";
const CACHE_VERSION: u32 = 3;

#[derive(Debug)]
pub enum CacheError {
    Io(std::io::Error),
    InvalidFormat(String),
    Vae(VaeError),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cache I/O error: {error}"),
            Self::InvalidFormat(message) => write!(f, "invalid cache format: {message}"),
            Self::Vae(error) => write!(f, "invalid cached tensor: {error}"),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<std::io::Error> for CacheError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<VaeError> for CacheError {
    fn from(error: VaeError) -> Self {
        Self::Vae(error)
    }
}

/// One fully preprocessed training sample.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedTrainingSample {
    /// Stable identifier for tracing a cached sample back to its source.
    pub sample_id: String,
    /// Repeat index inherited from the expanded image/caption dataset.
    pub repeat_index: usize,
    /// Output of the text encoder, normally shaped `[batch, sequence, hidden]`.
    pub text_embedding: VaeTensor,
    /// Optional `[batch, 1, 1, sequence]` mask for padded text tokens.
    pub text_attention_mask: Option<VaeTensor>,
    /// T5 token IDs and mask consumed by Anima's LLM adapter.
    pub adapter_conditioning: Option<AdapterConditioning>,
    /// Output of VAE encoding after latent scaling, normally shaped
    /// `[batch, latent_channels, height / 8, width / 8]`.
    pub image_latent: VaeTensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterConditioning {
    pub input_ids: Vec<u32>,
    pub attention_mask: VaeTensor,
}

impl CachedTrainingSample {
    pub fn new(
        sample_id: impl Into<String>,
        repeat_index: usize,
        text_embedding: VaeTensor,
        image_latent: VaeTensor,
    ) -> Self {
        Self {
            sample_id: sample_id.into(),
            repeat_index,
            text_embedding,
            text_attention_mask: None,
            adapter_conditioning: None,
            image_latent,
        }
    }

    pub fn with_text_attention_mask(mut self, mask: VaeTensor) -> Self {
        self.text_attention_mask = Some(mask);
        self
    }

    pub fn with_adapter_conditioning(mut self, conditioning: AdapterConditioning) -> Self {
        self.adapter_conditioning = Some(conditioning);
        self
    }

    /// Approximate host-memory usage of the tensor payload.
    pub fn tensor_bytes(&self) -> usize {
        let mask = self
            .text_attention_mask
            .as_ref()
            .map_or(0, |value| value.data.len());
        let adapter = self.adapter_conditioning.as_ref().map_or(0, |value| {
            value.input_ids.len() * size_of::<u32>()
                + value.attention_mask.data.len() * size_of::<f32>()
        });
        (self.text_embedding.data.len() + mask + self.image_latent.data.len()) * size_of::<f32>()
            + adapter
    }
}

/// In-memory cache consumed by the Burn training loop.
#[derive(Debug, Clone, Default)]
pub struct InMemoryTrainingCache {
    samples: Vec<CachedTrainingSample>,
}

impl InMemoryTrainingCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, sample: CachedTrainingSample) {
        self.samples.push(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&CachedTrainingSample> {
        self.samples.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &CachedTrainingSample> {
        self.samples.iter()
    }

    pub fn total_tensor_bytes(&self) -> usize {
        self.samples
            .iter()
            .map(CachedTrainingSample::tensor_bytes)
            .sum()
    }

    /// Validates the cache before handing it to a training loop.
    pub fn validate(&self) -> Result<(), VaeError> {
        for sample in &self.samples {
            if sample.sample_id.trim().is_empty() {
                return Err(VaeError::InvalidConfig(
                    "cached sample_id must not be empty".into(),
                ));
            }
            if sample
                .text_embedding
                .data
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(VaeError::InvalidConfig(format!(
                    "text embedding for {} contains a non-finite value",
                    sample.sample_id
                )));
            }
            if let Some(mask) = &sample.text_attention_mask {
                if mask.shape.len() != 4
                    || mask.shape[0] != 1
                    || mask.shape[1] != 1
                    || mask.shape[2] != 1
                {
                    return Err(VaeError::InvalidConfig(format!(
                        "text attention mask for {} must have shape [1, 1, 1, sequence]",
                        sample.sample_id
                    )));
                }
                if mask
                    .data
                    .iter()
                    .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
                {
                    return Err(VaeError::InvalidConfig(format!(
                        "text attention mask for {} contains invalid values",
                        sample.sample_id
                    )));
                }
            }
            if let Some(conditioning) = &sample.adapter_conditioning {
                if conditioning.input_ids.is_empty()
                    || conditioning.attention_mask.shape.len() != 4
                    || conditioning.attention_mask.shape[0] != 1
                    || conditioning.attention_mask.shape[1] != 1
                    || conditioning.attention_mask.shape[2] != 1
                    || conditioning.attention_mask.shape[3] != conditioning.input_ids.len()
                    || conditioning
                        .attention_mask
                        .data
                        .iter()
                        .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
                {
                    return Err(VaeError::InvalidConfig(format!(
                        "adapter conditioning for {} has invalid token/mask shapes",
                        sample.sample_id
                    )));
                }
            }
            if sample
                .image_latent
                .data
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(VaeError::InvalidConfig(format!(
                    "image latent for {} contains a non-finite value",
                    sample.sample_id
                )));
            }
        }
        Ok(())
    }

    /// Validates the tensor contract consumed by the Anima DiT.
    pub fn validate_anima_shapes(&self) -> Result<(), VaeError> {
        self.validate()?;
        for sample in &self.samples {
            let text = &sample.text_embedding.shape;
            if text.len() != 3 || text[0] != 1 || text[2] != 1024 || text[1] == 0 {
                return Err(VaeError::InvalidConfig(format!(
                    "text embedding for {} must have shape [1, sequence, 1024], got {text:?}",
                    sample.sample_id
                )));
            }
            if let Some(mask) = &sample.text_attention_mask {
                if mask.shape[3] != text[1] {
                    return Err(VaeError::InvalidConfig(format!(
                        "text attention mask for {} has sequence length {}, expected {}",
                        sample.sample_id, mask.shape[3], text[1]
                    )));
                }
            }
            let latent = &sample.image_latent.shape;
            if latent.len() != 5
                || latent[0] != 1
                || latent[1] != 16
                || latent[2] != 1
                || latent[3] == 0
                || latent[4] == 0
                || latent[3] % 2 != 0
                || latent[4] % 2 != 0
            {
                return Err(VaeError::InvalidConfig(format!(
                    "image latent for {} must have shape [1, 16, 1, even H, even W], got {latent:?}",
                    sample.sample_id
                )));
            }
        }
        Ok(())
    }

    /// Persists the fully encoded cache without retaining source images or text.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CacheError> {
        self.validate()?;
        let mut file = File::create(path)?;
        file.write_all(CACHE_MAGIC)?;
        file.write_all(&CACHE_VERSION.to_le_bytes())?;
        write_u64(&mut file, self.samples.len() as u64)?;
        for sample in &self.samples {
            write_bytes(&mut file, sample.sample_id.as_bytes())?;
            write_u64(&mut file, sample.repeat_index as u64)?;
            write_tensor(&mut file, &sample.text_embedding)?;
            match &sample.text_attention_mask {
                Some(mask) => {
                    file.write_all(&[1])?;
                    write_tensor(&mut file, mask)?;
                }
                None => file.write_all(&[0])?,
            }
            match &sample.adapter_conditioning {
                Some(conditioning) => {
                    file.write_all(&[1])?;
                    write_u32_vec(&mut file, &conditioning.input_ids)?;
                    write_tensor(&mut file, &conditioning.attention_mask)?;
                }
                None => file.write_all(&[0])?,
            }
            write_tensor(&mut file, &sample.image_latent)?;
        }
        Ok(())
    }

    /// Restores a cache produced by [`Self::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CacheError> {
        let mut file = File::open(path)?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(CacheError::InvalidFormat("bad magic".into()));
        }
        let version = read_u32(&mut file)?;
        if version != 1 && version != 2 && version != CACHE_VERSION {
            return Err(CacheError::InvalidFormat(format!(
                "unsupported version {version}"
            )));
        }
        let count = bounded_count(read_u64(&mut file)?, "sample count")?;
        let mut cache = Self::new();
        let show_progress = std::env::var_os("AFTERBURNER_CACHE_PROGRESS").is_some();
        for index in 0..count {
            if show_progress && (index == 0 || index % 10 == 0) {
                eprintln!("[cache] loading sample {}/{}", index + 1, count);
            }
            let sample_id = String::from_utf8(read_bytes(&mut file)?)
                .map_err(|_| CacheError::InvalidFormat("sample id is not UTF-8".into()))?;
            let repeat_index = read_u64(&mut file)? as usize;
            let text_embedding = read_tensor(&mut file)?;
            let text_attention_mask = if version >= 2 {
                match read_byte(&mut file)? {
                    0 => None,
                    1 => Some(read_tensor(&mut file)?),
                    value => {
                        return Err(CacheError::InvalidFormat(format!(
                            "invalid text mask flag {value}"
                        )));
                    }
                }
            } else {
                None
            };
            let adapter_conditioning = if version >= 3 {
                match read_byte(&mut file)? {
                    0 => None,
                    1 => Some(AdapterConditioning {
                        input_ids: read_u32_vec(&mut file)?,
                        attention_mask: read_tensor(&mut file)?,
                    }),
                    value => {
                        return Err(CacheError::InvalidFormat(format!(
                            "invalid adapter conditioning flag {value}"
                        )));
                    }
                }
            } else {
                None
            };
            let image_latent = read_tensor(&mut file)?;
            let sample =
                CachedTrainingSample::new(sample_id, repeat_index, text_embedding, image_latent);
            let sample = match text_attention_mask {
                Some(mask) => sample.with_text_attention_mask(mask),
                None => sample,
            };
            cache.push(match adapter_conditioning {
                Some(conditioning) => sample.with_adapter_conditioning(conditioning),
                None => sample,
            });
        }
        if show_progress {
            eprintln!("[cache] validating {} loaded samples", cache.len());
        }
        cache.validate()?;
        if show_progress {
            eprintln!("[cache] validation complete");
        }
        Ok(cache)
    }
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<(), CacheError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(CacheError::from)
}

fn write_u32_vec(writer: &mut impl Write, values: &[u32]) -> Result<(), CacheError> {
    write_u64(writer, values.len() as u64)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_u32_vec(reader: &mut impl Read) -> Result<Vec<u32>, CacheError> {
    let count = bounded_count(read_u64(reader)?, "adapter token count")?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        values.push(u32::from_le_bytes(bytes));
    }
    Ok(values)
}

fn read_u64(reader: &mut impl Read) -> Result<u64, CacheError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, CacheError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_byte(reader: &mut impl Read) -> Result<u8, CacheError> {
    let mut byte = [0; 1];
    reader.read_exact(&mut byte)?;
    Ok(byte[0])
}

fn write_bytes(writer: &mut impl Write, bytes: &[u8]) -> Result<(), CacheError> {
    write_u64(writer, bytes.len() as u64)?;
    writer.write_all(bytes).map_err(CacheError::from)
}

fn read_bytes(reader: &mut impl Read) -> Result<Vec<u8>, CacheError> {
    let length = bounded_count(read_u64(reader)?, "byte length")?;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn write_tensor(writer: &mut impl Write, tensor: &VaeTensor) -> Result<(), CacheError> {
    write_u64(writer, tensor.shape.len() as u64)?;
    for dimension in &tensor.shape {
        write_u64(writer, *dimension as u64)?;
    }
    write_u64(writer, tensor.data.len() as u64)?;
    for value in &tensor.data {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_tensor(reader: &mut impl Read) -> Result<VaeTensor, CacheError> {
    let rank = bounded_count(read_u64(reader)?, "tensor rank")?;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(bounded_count(read_u64(reader)?, "tensor dimension")?);
    }
    let count = bounded_count(read_u64(reader)?, "tensor element count")?;
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        data.push(f32::from_le_bytes(bytes));
    }
    Ok(VaeTensor::new(shape, data)?)
}

fn bounded_count(value: u64, label: &str) -> Result<usize, CacheError> {
    let value = usize::try_from(value)
        .map_err(|_| CacheError::InvalidFormat(format!("{label} is too large")))?;
    if value > 100_000_000 {
        return Err(CacheError::InvalidFormat(format!(
            "{label} exceeds safety limit"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(value: f32) -> VaeTensor {
        VaeTensor::new(vec![1], vec![value]).unwrap()
    }

    #[test]
    fn keeps_encoded_samples_in_host_memory() {
        let mut cache = InMemoryTrainingCache::new();
        cache.push(CachedTrainingSample::new(
            "image-01",
            0,
            tensor(1.0),
            tensor(2.0),
        ));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_tensor_bytes(), 8);
        assert!(cache.validate().is_ok());
    }

    #[test]
    fn rejects_non_finite_encoded_values() {
        let mut cache = InMemoryTrainingCache::new();
        cache.push(CachedTrainingSample::new(
            "image-01",
            0,
            tensor(f32::NAN),
            tensor(2.0),
        ));
        assert!(cache.validate().is_err());
    }

    #[test]
    fn validates_anima_tensor_contract() {
        let mut cache = InMemoryTrainingCache::new();
        cache.push(CachedTrainingSample::new(
            "image-01",
            0,
            VaeTensor::new(vec![1, 2, 1024], vec![0.0; 2 * 1024]).unwrap(),
            VaeTensor::new(vec![1, 16, 1, 2, 2], vec![0.0; 64]).unwrap(),
        ));
        assert!(cache.validate_anima_shapes().is_ok());

        let mut invalid = InMemoryTrainingCache::new();
        invalid.push(CachedTrainingSample::new(
            "image-01",
            0,
            VaeTensor::new(vec![1, 2], vec![0.0; 2]).unwrap(),
            VaeTensor::new(vec![1, 16, 1, 2, 2], vec![0.0; 64]).unwrap(),
        ));
        assert!(invalid.validate_anima_shapes().is_err());
    }

    #[test]
    fn saves_and_loads_encoded_cache() {
        let path = std::env::temp_dir().join(format!("afterburner-cache-{}", std::process::id()));
        let mut cache = InMemoryTrainingCache::new();
        let sample = CachedTrainingSample::new(
            "image-01",
            2,
            VaeTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap(),
            VaeTensor::new(vec![1, 1, 1, 2, 2], vec![0.0; 4]).unwrap(),
        )
        .with_adapter_conditioning(AdapterConditioning {
            input_ids: vec![1, 0],
            attention_mask: VaeTensor::new(vec![1, 1, 1, 2], vec![1.0, 0.0]).unwrap(),
        });
        cache.push(sample);
        cache.save(&path).unwrap();
        let restored = InMemoryTrainingCache::load(&path).unwrap();
        assert_eq!(
            restored.iter().collect::<Vec<_>>(),
            cache.iter().collect::<Vec<_>>()
        );
        std::fs::remove_file(path).unwrap();
    }
}
