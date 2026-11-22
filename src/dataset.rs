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

        let mut samples = Vec::new();
        append_directory_samples(root.as_ref(), repeat_count, &mut samples)?;
        Ok(Self { samples })
    }

    /// Scans a dataset root and derives repeat counts from immediate folders.
    ///
    /// A folder named `30_character` contributes each image-caption pair 30
    /// times. When no numbered child folders exist, the root folder itself is
    /// checked for the same prefix before falling back to `default_repeat`.
    pub fn from_dir_with_folder_repeats(
        root: impl AsRef<Path>,
        default_repeat: usize,
    ) -> Result<Self, DatasetError> {
        if default_repeat == 0 {
            return Err(DatasetError::InvalidRepeatCount);
        }

        let root = root.as_ref();
        let mut directories = fs::read_dir(root)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        directories.sort();
        let numbered = directories
            .iter()
            .filter_map(|path| parse_repeat_folder_name(path).map(|repeat| (path, repeat)))
            .collect::<Vec<_>>();

        let mut samples = Vec::new();
        if numbered.is_empty() {
            let repeat = parse_repeat_folder_name(root).unwrap_or(default_repeat);
            append_directory_samples(root, repeat, &mut samples)?;
        } else {
            for (directory, repeat) in numbered {
                append_directory_samples(directory, repeat, &mut samples)?;
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

fn append_directory_samples(
    root: &Path,
    repeat_count: usize,
    samples: &mut Vec<TrainingSample>,
) -> Result<(), DatasetError> {
    let mut images = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_supported_image(path))
        .collect::<Vec<_>>();
    images.sort();

    for image_path in images {
        let caption_path = image_path.with_extension("txt");
        if !caption_path.is_file() {
            return Err(DatasetError::MissingCaption {
                image: image_path,
                caption: caption_path,
            });
        }

        let bytes = fs::read(&caption_path)?;
        let caption = String::from_utf8(bytes).map_err(|source| DatasetError::InvalidCaption {
            caption: caption_path.clone(),
            source,
        })?;
        // Kohya's default caption processing uses only the first line when
        // wildcard processing is disabled (the normal training path). Keep
        // the same semantics so multi-line caption files do not change the
        // text encoder input in Afterburner.
        let caption = caption
            .trim()
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();
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
    Ok(())
}

fn parse_repeat_folder_name(path: &Path) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    let (prefix, _) = name.split_once('_')?;
    let repeat = prefix.parse::<usize>().ok()?;
    (repeat > 0).then_some(repeat)
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

    #[test]
    fn uses_first_caption_line_like_kohya_default_processing() {
        let root = std::env::temp_dir().join(format!(
            "afterburner-dataset-first-line-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("sample.png"), b"not an image").unwrap();
        fs::write(root.join("sample.txt"), "first line\nsecond line\n").unwrap();

        let dataset = ImageCaptionDataset::from_dir(&root, 1).unwrap();
        assert_eq!(dataset.get(0).unwrap().caption, "first line");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn expands_repeats_per_numbered_subfolder() {
        let root =
            std::env::temp_dir().join(format!("afterburner-folder-repeats-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("30_Elvira")).unwrap();
        fs::create_dir_all(root.join("2_other")).unwrap();
        for (directory, name) in [("30_Elvira", "elvira"), ("2_other", "other")] {
            let image = root.join(directory).join(format!("{name}.png"));
            fs::write(&image, []).unwrap();
            fs::write(image.with_extension("txt"), name).unwrap();
        }

        let dataset = ImageCaptionDataset::from_dir_with_folder_repeats(&root, 1).unwrap();
        assert_eq!(dataset.len(), 32);
        assert_eq!(
            dataset
                .samples()
                .iter()
                .filter(|sample| sample.caption == "elvira")
                .count(),
            30
        );
        assert_eq!(
            dataset
                .samples()
                .iter()
                .filter(|sample| sample.caption == "other")
                .count(),
            2
        );
        fs::remove_dir_all(root).unwrap();
    }
}
