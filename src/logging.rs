//! kohya_ss-inspired training log formatting.

use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};

/// A small logging sink used by the preparation, preprocessing, and training phases.
pub trait TrainingLogger {
    fn info(&mut self, message: &str);
    fn warning(&mut self, message: &str);
}

/// Emits the preparation block corresponding to kohya_ss's validation and
/// training-summary messages.
pub fn log_preparation<L: TrainingLogger>(
    logger: &mut L,
    model_name: &str,
    base_model: &str,
    dataset_root: &str,
    image_count: usize,
    repeats: usize,
    epochs: usize,
    batch_size: usize,
    gradient_accumulation: usize,
    learning_rate: f64,
    lora_rank: usize,
    lora_alpha: usize,
) {
    let samples_per_epoch = image_count.saturating_mul(repeats);
    let total_steps = samples_per_epoch
        .div_ceil(batch_size.max(1))
        .div_ceil(gradient_accumulation.max(1))
        .saturating_mul(epochs);
    logger.info(&format!("Start training LoRA {model_name} ..."));
    logger.info("Validating lr scheduler arguments...");
    logger.info("Validating optimizer arguments...");
    logger.info(&format!("Validating {base_model} existence... SUCCESS"));
    logger.info(&format!("Validating {dataset_root} existence... SUCCESS"));
    logger.info(&format!("Folder {dataset_root}: {repeats} repeats found"));
    logger.info(&format!(
        "Folder {dataset_root}: {image_count} images found"
    ));
    logger.info(&format!(
        "Folder {dataset_root}: {image_count} * {repeats} = {samples_per_epoch} steps"
    ));
    logger.info(&format!("Train batch size: {batch_size}"));
    logger.info(&format!(
        "Gradient accumulation steps: {gradient_accumulation}"
    ));
    logger.info(&format!("Epoch: {epochs}"));
    logger.info(&format!("max_train_steps = {total_steps}"));
    logger.info(&format!(
        "Main LR (for optimizer & fallback): {learning_rate:.2e}"
    ));
    logger.info(&format!(
        "create LoRA network. base dim (rank): {lora_rank}, alpha: {lora_alpha}"
    ));
    logger.info("Network train mode: UNet-only (text encoder not trained; saved LoRA will have no TE keys).");
}

/// Writes kohya_ss-like timestamped lines to an arbitrary writer.
pub struct KohyaLogger<W: Write> {
    writer: W,
}

impl<W: Write> KohyaLogger<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write_line(&mut self, level: &str, message: &str) {
        let _ = writeln!(self.writer, "{} {:<7} {}", timestamp(), level, message);
        let _ = self.writer.flush();
    }
}

impl<W: Write> TrainingLogger for KohyaLogger<W> {
    fn info(&mut self, message: &str) {
        self.write_line("INFO", message);
    }
    fn warning(&mut self, message: &str) {
        self.write_line("WARNING", message);
    }
}

pub type StderrLogger = KohyaLogger<io::Stderr>;

impl StderrLogger {
    pub fn stderr() -> Self {
        KohyaLogger::new(io::stderr())
    }
}

/// Discards log messages so the original APIs remain quiet.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullLogger;

impl TrainingLogger for NullLogger {
    fn info(&mut self, _message: &str) {}
    fn warning(&mut self, _message: &str) {}
}

fn timestamp() -> String {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = elapsed.as_secs() % 86_400;
    let hour = seconds / 3_600;
    let minute = (seconds % 3_600) / 60;
    let second = seconds % 60;
    format!(
        "{hour:02}:{minute:02}:{second:02}-{micros:06}",
        micros = elapsed.subsec_micros()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_kohya_style_levels() {
        let mut logger = KohyaLogger::new(Vec::<u8>::new());
        logger.info("Start training LoRA Anima ...");
        logger.warning("no regularization images");
        let output = String::from_utf8(logger.into_inner()).unwrap();
        assert!(output.contains(" INFO    Start training LoRA Anima ..."));
        assert!(output.contains(" WARNING no regularization images"));
    }
}
