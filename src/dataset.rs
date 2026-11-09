//! Image/caption dataset support for LoRA training.
//!
//! A sample consists of an image and a UTF-8 text file with the same stem:
//! `portrait.png` pairs with `portrait.txt`. The repeat count expands the
//! dataset deterministically so the training loop can treat each occurrence
//! as a normal sample.

use std::{
    fs,
    path::{Path, PathBuf},
};

use burn_dataset::Dataset;

/// Errors found while building an image/caption dataset.
#[derive(Debug)]
pub enum DatasetError {
    Io(std::io::Error),
    InvalidRepeatCount,
    MissingCaption {
        image: PathBuf,
        caption: PathBuf,
    },
    EmptyCaption {
        caption: PathBuf,
    },
    InvalidCaption {
        caption: PathBuf,
        source: std::string::FromUtf8Error,
    },
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "dataset I/O error: {error}"),
            Self::InvalidRepeatCount => write!(f, "repeat count must be greater than zero"),
            Self::MissingCaption { image, caption } => {
                write!(f, "missing caption {caption:?} for image {image:?}")
            }
            Self::EmptyCaption { caption } => write!(f, "caption {caption:?} is empty"),
            Self::InvalidCaption { caption, source } => {
                write!(f, "caption {caption:?} is not valid UTF-8: {source}")
            }
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// One expanded training sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingSample {
    pub image_path: PathBuf,
    pub caption_path: PathBuf,
    pub caption: String,
    pub repeat_index: usize,
}

/// A deterministic Burn dataset of image/caption pairs.
#[derive(Debug, Clone, Default)]
pub struct ImageCaptionDataset {
    samples: Vec<TrainingSample>,
}

impl ImageCaptionDataset {
    /// Scans a directory and expands every valid pair `repeat_count` times.
    pub fn from_dir(root: impl AsRef<Path>, repeat_count: usize) -> Result<Self, DatasetError> {
        if repeat_count == 0 {
            return Err(DatasetError::InvalidRepeatCount);
        }

        let mut images = fs::read_dir(root.as_ref())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && is_supported_image(path))
            .collect::<Vec<_>>();
        images.sort();

        let mut samples = Vec::with_capacity(images.len() * repeat_count);
        for image_path in images {
            let caption_path = image_path.with_extension("txt");
            if !caption_path.is_file() {
                return Err(DatasetError::MissingCaption {
                    image: image_path,
                    caption: caption_path,
                });
            }

            let bytes = fs::read(&caption_path)?;
            let caption =
                String::from_utf8(bytes).map_err(|source| DatasetError::InvalidCaption {
                    caption: caption_path.clone(),
                    source,
                })?;
            let caption = caption.trim().to_owned();
            if caption.is_empty() {
                return Err(DatasetError::EmptyCaption {
                    caption: caption_path,
                });
            }

            for repeat_index in 0..repeat_count {
                samples.push(TrainingSample {
                    image_path: image_path.clone(),
                    caption_path: caption_path.clone(),
                    caption: caption.clone(),
                    repeat_index,
                });
            }
        }

        Ok(Self { samples })
    }

    /// Returns the expanded sample list for inspection or a custom sampler.
    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    /// Decodes a sample as an RGB image. Tensor conversion is kept separate so
    /// preprocessing can later be performed directly on the selected backend.
    pub fn load_rgb(sample: &TrainingSample) -> Result<image::RgbImage, image::ImageError> {
        image::open(&sample.image_path).map(|image| image.to_rgb8())
    }
}

impl Dataset<TrainingSample> for ImageCaptionDataset {
    fn get(&self, index: usize) -> Option<TrainingSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn pairs_captions_and_expands_repeats() {
        let root = std::env::temp_dir().join(format!("afterburner-dataset-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let mut caption = fs::File::create(root.join("sample.txt")).unwrap();
        writeln!(caption, "a portrait, studio lighting").unwrap();
        fs::write(root.join("sample.png"), []).unwrap();

        let dataset = ImageCaptionDataset::from_dir(&root, 3).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(2).unwrap().repeat_index, 2);
        assert_eq!(
            dataset.get(0).unwrap().caption,
            "a portrait, studio lighting"
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_caption() {
        let root = std::env::temp_dir().join(format!(
            "afterburner-dataset-missing-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("sample.png"), []).unwrap();

        assert!(matches!(
            ImageCaptionDataset::from_dir(&root, 1),
            Err(DatasetError::MissingCaption { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }
}
