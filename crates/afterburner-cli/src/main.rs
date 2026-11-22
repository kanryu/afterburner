use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

static SESSION_LOG: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

fn init_session_log(path: &Path) {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open session log file");
    let _ = SESSION_LOG.set(Mutex::new(file));
}

fn append_session_log(message: &str) {
    use std::io::Write;
    if let Some(file) = SESSION_LOG.get()
        && let Ok(mut file) = file.lock()
    {
        let _ = writeln!(file, "{message}");
        let _ = file.flush();
    }
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn trim_cuda_default_pool(min_bytes: usize) -> Result<(), String> {
    use std::ffi::{c_void, CString};

    type CuResult = i32;
    type CuDevice = i32;
    type CuMemoryPool = *mut c_void;
    type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
    type CuGetDefaultPool = unsafe extern "system" fn(*mut CuMemoryPool, CuDevice) -> CuResult;
    type CuTrimTo = unsafe extern "system" fn(CuMemoryPool, usize) -> CuResult;

    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    let module_name = CString::new("nvcuda.dll").map_err(|error| error.to_string())?;
    let module = unsafe { LoadLibraryA(module_name.as_ptr() as *const u8) };
    if module.is_null() {
        return Err("LoadLibraryA(nvcuda.dll) failed".to_owned());
    }
    let result = (|| {
        let symbol = |name: &'static [u8]| -> Result<*mut c_void, String> {
            let pointer = unsafe { GetProcAddress(module, name.as_ptr()) };
            if pointer.is_null() {
                Err(format!("CUDA symbol {} is unavailable", String::from_utf8_lossy(name)))
            } else {
                Ok(pointer)
            }
        };
        let device_get: CuDeviceGet = unsafe { std::mem::transmute(symbol(b"cuDeviceGet\0")?) };
        let pool_get: CuGetDefaultPool = unsafe {
            std::mem::transmute(symbol(b"cuDeviceGetDefaultMemPool\0")?)
        };
        let trim_to: CuTrimTo = unsafe { std::mem::transmute(symbol(b"cuMemPoolTrimTo\0")?) };
        let mut device = 0;
        let mut pool = std::ptr::null_mut();
        let code = unsafe { device_get(&mut device, 0) };
        if code != 0 {
            return Err(format!("cuDeviceGet failed with CUDA error {code}"));
        }
        let code = unsafe { pool_get(&mut pool, device) };
        if code != 0 {
            return Err(format!("cuDeviceGetDefaultMemPool failed with CUDA error {code}"));
        }
        let code = unsafe { trim_to(pool, min_bytes) };
        if code != 0 {
            return Err(format!("cuMemPoolTrimTo failed with CUDA error {code}"));
        }
        Ok(())
    })();
    unsafe { FreeLibrary(module) };
    result
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn cuda_driver_memory() -> Option<(usize, usize)> {
    use std::ffi::{c_void, CString};
    type CuResult = i32;
    type CuMemoryInfo = unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult;
    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }
    let name = CString::new("nvcuda.dll").ok()?;
    let module = unsafe { LoadLibraryA(name.as_ptr() as *const u8) };
    if module.is_null() { return None; }
    let symbol = CString::new("cuMemGetInfo_v2").ok()?;
    let ptr = unsafe { GetProcAddress(module, symbol.as_ptr() as *const u8) };
    if ptr.is_null() {
        unsafe { FreeLibrary(module); }
        return None;
    }
    let get_info: CuMemoryInfo = unsafe { std::mem::transmute(ptr) };
    let mut free = 0usize;
    let mut total = 0usize;
    let code = unsafe { get_info(&mut free, &mut total) };
    unsafe { FreeLibrary(module); }
    (code == 0).then_some((total.saturating_sub(free), free))
}

macro_rules! println {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        std::println!("{message}");
        append_session_log(&message);
    }};
}

macro_rules! eprintln {
    ($($arg:tt)*) => {{
        let message = format!($($arg)*);
        std::eprintln!("{message}");
        append_session_log(&message);
    }};
}

fn load_replay_inputs(
    directory: impl AsRef<Path>,
    _cache: &afterburner::InMemoryTrainingCache,
) -> Result<Vec<afterburner::ReplayTrainingInput>, String> {
    let directory = directory.as_ref();
    let mut result = Vec::new();
    // A one-sample cache can still replay many optimization steps; the probe
    // directory, not the cache length, defines how many records are present.
    for step in 1..=10_000 {
        let stem = format!("step-{step:06}");
        let shape_path = directory.join(format!("{stem}.shape.json"));
        let sigma_path = directory.join(format!("{stem}.sigma.f32"));
        let noise_path = directory.join(format!("{stem}.noise.f32"));
        if !shape_path.exists() || !sigma_path.exists() || !noise_path.exists() {
            break;
        }
        let shape: Vec<usize> = serde_json::from_slice(
            &std::fs::read(&shape_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("{}: {error}", shape_path.display()))?;
        let sigma_bytes = std::fs::read(&sigma_path).map_err(|error| error.to_string())?;
        if sigma_bytes.len() != 4 {
            return Err(format!("{} must contain one f32", sigma_path.display()));
        }
        let sigma = f32::from_le_bytes(sigma_bytes[..4].try_into().unwrap());
        let noise_bytes = std::fs::read(&noise_path).map_err(|error| error.to_string())?;
        if noise_bytes.len() % 4 != 0 {
            return Err(format!("{} has a non-f32 byte length", noise_path.display()));
        }
        let noise_data = noise_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let expected = shape.iter().product::<usize>();
        if noise_data.len() != expected {
            return Err(format!(
                "{} has {} values, expected {}",
                noise_path.display(),
                noise_data.len(),
                expected
            ));
        }
        // Kohya dumps Anima noise as [B,C,H,W], while the cached training
        // contract keeps the singleton video/time axis: [B,C,1,H,W].
        let cache_shape = _cache
            .get(0)
            .map(|sample| sample.image_latent.shape.clone());
        let shape = if shape.len() == 4
            && cache_shape.as_ref().is_some_and(|cached| {
                cached.len() == 5 && cached[2] == 1 && cached[0] == shape[0] && cached[1] == shape[1]
            })
        {
            vec![shape[0], shape[1], 1, shape[2], shape[3]]
        } else {
            shape
        };
        let noise = afterburner::VaeTensor::new(shape, noise_data)
            .map_err(|error| error.to_string())?;
        let latent_shape_path = directory.join(format!("{stem}.latent.shape.json"));
        let latent_path = directory.join(format!("{stem}.latent.f32"));
        let latent = if latent_shape_path.exists() || latent_path.exists() {
            if !latent_shape_path.exists() || !latent_path.exists() {
                return Err(format!(
                    "latent replay requires both {} and {}",
                    latent_shape_path.display(),
                    latent_path.display()
                ));
            }
            let latent_shape: Vec<usize> = serde_json::from_slice(
                &std::fs::read(&latent_shape_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("{}: {error}", latent_shape_path.display()))?;
            let latent_bytes = std::fs::read(&latent_path).map_err(|error| error.to_string())?;
            if latent_bytes.len() % 4 != 0 {
                return Err(format!("{} has a non-f32 byte length", latent_path.display()));
            }
            let latent_data = latent_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            if latent_data.len() != latent_shape.iter().product::<usize>() {
                return Err(format!(
                    "{} has {} values, expected {}",
                    latent_path.display(),
                    latent_data.len(),
                    latent_shape.iter().product::<usize>()
                ));
            }
            let latent_shape = if latent_shape.len() == 4
                && cache_shape.as_ref().is_some_and(|cached| {
                    cached.len() == 5
                        && cached[2] == 1
                        && cached[0] == latent_shape[0]
                        && cached[1] == latent_shape[1]
                }) {
                vec![latent_shape[0], latent_shape[1], 1, latent_shape[2], latent_shape[3]]
            } else {
                latent_shape
            };
            Some(
                afterburner::VaeTensor::new(latent_shape, latent_data)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        result.push(afterburner::ReplayTrainingInput { sigma, noise, latent });
    }
    if result.is_empty() {
        return Err(format!("no replay records found in {}", directory.display()));
    }
    Ok(result)
}

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct TrainingConfig {
    train_cache: Option<String>,
    base_model: Option<String>,
    output_lora: Option<String>,
    epochs: Option<usize>,
    steps: Option<usize>,
    learning_rate: Option<f64>,
    lr_schedule: Option<String>,
    lr_warmup_steps: Option<f64>,
    lr_min: Option<f64>,
    seed: Option<u64>,
    lora_rank: Option<usize>,
    lora_alpha: Option<usize>,
    flow_matching: Option<bool>,
    flow_shift: Option<f32>,
    #[serde(alias = "anima_timestep_sampling")]
    timestep_sampling: Option<String>,
    sigmoid_scale: Option<f32>,
    device: Option<String>,
    gpu_index: Option<usize>,
    weighting_scheme: Option<String>,
    max_grad_norm: Option<f32>,
    checkpoint_every: Option<usize>,
    checkpoint_prefix: Option<String>,
    model_cache_dir: Option<String>,
    memory_budget_gb: Option<f64>,
    attention_chunk: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct MemoryPreflight {
    base_model_bytes: u64,
    cache_bytes: usize,
    estimated_training_bytes: u64,
    attention_chunk: usize,
    token_count: usize,
    dtype_bytes: usize,
    autodiff_buffer_factor: usize,
}

fn estimate_training_memory(
    base_model: &str,
    cache: &afterburner::InMemoryTrainingCache,
    dtype_bytes: usize,
    attention_chunk: usize,
) -> Result<MemoryPreflight, String> {
    let base_model_bytes = std::fs::metadata(base_model)
        .map_err(|error| format!("failed to stat base model {base_model}: {error}"))?
        .len();
    let sample = cache
        .get(0)
        .ok_or_else(|| "training cache is empty".to_owned())?;
    let latent_shape = &sample.image_latent.shape;
    if latent_shape.len() < 5 {
        return Err(format!(
            "cached latent must have at least 4 dimensions, got {:?}",
            latent_shape
        ));
    }
    // Anima patchifies the latent spatial plane into 2x2 patches and adds the
    // four latent channels from each patch to the input channel dimension.
    // Cached latents are [batch, channels, frames, height, width], so the
    // DiT token count is the post-patch spatial area, not shape[2] * shape[3].
    let token_count = latent_shape[3]
        .checked_div(2)
        .unwrap_or(0)
        .saturating_mul(latent_shape[4].checked_div(2).unwrap_or(0));
    let bytes = dtype_bytes as u128;
    let tokens = token_count as u128;
    let heads = 16_u128;
    let hidden = 2048_u128;
    let mlp = 8192_u128;
    // Without checkpointing, Burn must retain a large fraction of the
    // 28-block activation graph. Balanced checkpointing recomputes those
    // activations during backward, so the preflight must not reject the run
    // using the non-checkpointed multiplier.
    #[cfg(feature = "checkpointing")]
    const AUTODIFF_BUFFER_FACTOR: usize = 1;
    #[cfg(not(feature = "checkpointing"))]
    const AUTODIFF_BUFFER_FACTOR: usize = 4;
    // This is deliberately conservative: it includes the 28-block autograd
    // activation retention currently used by Burn, plus the largest tiled
    // attention score/value workspace. Attention now chunks both query and
    // key sequences, so the score workspace is quadratic in the chunk size,
    // not in the full token count.
    let block_activation = tokens
        .saturating_mul(hidden)
        .saturating_mul(12)
        .saturating_add(tokens.saturating_mul(mlp).saturating_mul(2))
        .saturating_mul(bytes);
    let key_chunk = tokens
        .min((attention_chunk as u128).saturating_mul(4))
        .max(1);
    let attention_workspace = heads
        .saturating_mul(attention_chunk as u128)
        .saturating_mul(key_chunk)
        .saturating_mul(4)
        .saturating_mul(bytes);
    let estimated_training_bytes = (base_model_bytes as u128)
        .saturating_add(
            block_activation
                .saturating_mul(28)
                .saturating_mul(AUTODIFF_BUFFER_FACTOR as u128),
        )
        .saturating_add(
            attention_workspace
                .saturating_mul(28)
                .saturating_mul(AUTODIFF_BUFFER_FACTOR as u128),
        )
        .min(u64::MAX as u128) as u64;
    Ok(MemoryPreflight {
        base_model_bytes,
        cache_bytes: cache.total_tensor_bytes(),
        estimated_training_bytes,
        attention_chunk,
        token_count,
        dtype_bytes,
        autodiff_buffer_factor: AUTODIFF_BUFFER_FACTOR,
    })
}

fn print_memory_preflight(preflight: MemoryPreflight) {
    println!("memory_preflight=conservative_estimate");
    println!("memory_base_model_bytes={}", preflight.base_model_bytes);
    println!("memory_cache_host_bytes={}", preflight.cache_bytes);
    println!("memory_token_count={}", preflight.token_count);
    println!("memory_attention_chunk={}", preflight.attention_chunk);
    println!("memory_dtype_bytes={}", preflight.dtype_bytes);
    println!(
        "memory_autodiff_buffer_factor={}",
        preflight.autodiff_buffer_factor
    );
    println!(
        "memory_estimated_training_bytes={}",
        preflight.estimated_training_bytes
    );
}

fn print_cache_tensor_stats(label: &str, tensor: &afterburner::VaeTensor) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    for value in &tensor.data {
        minimum = minimum.min(*value);
        maximum = maximum.max(*value);
        sum += f64::from(*value);
        sum_squared += f64::from(*value) * f64::from(*value);
    }
    let count = tensor.data.len().max(1) as f64;
    println!(
        "cache_first_{label}_stats=min:{:.8e},max:{:.8e},mean:{:.8e},l2:{:.8e}",
        minimum,
        maximum,
        sum / count,
        sum_squared.sqrt()
    );
}

fn candle_device(name: &str, index: usize) -> candle_core::Device {
    if name.eq_ignore_ascii_case("cuda") {
        #[cfg(not(feature = "candle-cuda"))]
        {
            let _ = index;
            eprintln!(
                "Candle CUDA was requested, but the `candle-cuda` feature is disabled; falling back to CPU."
            );
            eprintln!("Rebuild with `--features candle-cuda` to enable Candle CUDA preprocessing.");
            return candle_core::Device::Cpu;
        }

        #[cfg(feature = "candle-cuda")]
        {
            return match candle_core::Device::new_cuda(index) {
                Ok(device) => device,
                Err(error) => {
                    eprintln!("CUDA is unavailable for Candle; falling back to CPU: {error}");
                    candle_core::Device::Cpu
                }
            };
        }
    }
    if name.eq_ignore_ascii_case("vulkan") {
        eprintln!("Vulkan is available for Burn training, but Candle preprocessing uses CPU.");
    }
    candle_core::Device::Cpu
}

fn resolve_tokenizer_path(
    explicit: Option<&str>,
    text_encoder_path: &str,
    relative_candidates: &[&str],
    label: &str,
) -> PathBuf {
    if let Some(path) = explicit {
        return PathBuf::from(path);
    }

    let model_path = Path::new(text_encoder_path);
    let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
    let mut search_dirs = vec![model_dir.to_path_buf()];
    search_dirs.push(model_dir.join("tokenizer"));
    if let Some(parent) = model_dir.parent() {
        search_dirs.push(parent.to_path_buf());
        search_dirs.push(parent.join("tokenizer"));
    }
    // Keep the reference tokenizers in the source distribution so a checkout
    // can preprocess a dataset without relying on a ComfyUI/Kohya install.
    search_dirs.push(PathBuf::from("assets/tokenizers"));
    if let Ok(executable) = std::env::current_exe() {
        let mut directory = executable.parent();
        for _ in 0..3 {
            if let Some(parent) = directory {
                search_dirs.push(parent.join("assets/tokenizers"));
                directory = parent.parent();
            }
        }
    }
    let scoped_environment_variable = match label {
        "Qwen" => Some("AFTERBURNER_QWEN_TOKENIZER_DIR"),
        "T5" => Some("AFTERBURNER_T5_TOKENIZER_DIR"),
        _ => None,
    };
    if let Some(variable) = scoped_environment_variable {
        if let Ok(directory) = std::env::var(variable) {
            search_dirs.insert(0, PathBuf::from(directory));
        }
    }
    if let Ok(directory) = std::env::var("AFTERBURNER_TOKENIZER_DIR") {
        search_dirs.insert(1.min(search_dirs.len()), PathBuf::from(directory));
    }

    for directory in search_dirs {
        for relative in relative_candidates {
            let candidate = directory.join(relative);
            if candidate.is_file() {
                eprintln!("auto-selected {label} tokenizer: {}", candidate.display());
                return candidate;
            }
        }
    }

    eprintln!(
        "unable to find {label} tokenizer near {}",
        model_path.display()
    );
    eprintln!(
        "provide the path with the corresponding CLI option or set the scoped tokenizer directory environment variable"
    );
    std::process::exit(2);
}

fn main() {
    const TRAINING_STACK_SIZE: usize = 64 * 1024 * 1024;
    let result = std::thread::Builder::new()
        .name("afterburner-main".to_owned())
        .stack_size(TRAINING_STACK_SIZE)
        .spawn(run)
        .expect("failed to create Afterburner worker thread")
        .join();
    if let Err(payload) = result {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("non-string panic payload");
        eprintln!("Afterburner worker thread terminated unexpectedly: {message}");
        eprintln!("Set RUST_BACKTRACE=1 to include the Rust stack trace.");
        std::process::exit(1);
    }
}

fn run() {
    let mut model_type = "anima".to_owned();
    let mut inspect_lora = None;
    let mut inspect_lora_factors = None;
    let mut inspect_weights = None;
    let mut inspect_dataset = None;
    let mut inspect_cache = None;
    let mut dump_cache_latent = None::<String>;
    let mut dump_cache_latents = None::<String>;
    let mut inspect_text_encoder = None;
    let mut inspect_vae = None;
    let mut inspect_vae_stem = None;
    let mut text_encoder_model = None;
    let mut vae_model = None;
    let mut t5_tokenizer_model = None;
    let mut preprocess_dataset = None;
    let mut train_cache = None;
    let mut base_model = None;
    let mut initial_lora = None;
    let mut output_lora = None;
    let mut train_steps = 0usize;
    let mut max_optimization_steps = None::<usize>;
    let mut learning_rate = 1e-4_f64;
    let mut lr_schedule = "warmup_cosine".to_owned();
    let mut lr_warmup_steps = None::<f64>;
    let mut lr_min = None::<f64>;
    // Kohya_ss's reference Anima configuration uses max_grad_norm=1.
    let mut max_grad_norm = 1.0_f32;
    // Match kohya_ss's Anima reference configuration. Runs that need a
    // different stochastic sequence can still override this with --seed.
    let mut seed = 0_u64;
    // Match the practical kohya_ss/Anima baseline. Higher ranks remain
    // available through --lora-rank, but rank 8 keeps the default adapter
    // size and optimizer state within the intended 16 GiB class.
    let mut lora_rank = 8usize;
    let mut lora_alpha = 16usize;
    #[cfg(feature = "bf16")]
    let mut lora_dtype = "bf16".to_owned();
    #[cfg(all(not(feature = "bf16"), feature = "f16"))]
    let mut lora_dtype = "f16".to_owned();
    #[cfg(all(not(feature = "bf16"), not(feature = "f16")))]
    let mut lora_dtype = "f32".to_owned();
    let mut checkpoint_every = 0usize;
    let mut checkpoint_prefix = None;
    let mut weighting_scheme = "none".to_owned();
    // Anima is a rectified-flow model. Keep the model-correct objective as
    // the CLI default; the config file can still explicitly set false for
    // legacy epsilon-prediction experiments.
    let mut flow_matching = true;
    let mut flow_shift = 1.0_f32;
    // kohya_ss Anima defaults to the continuous sigmoid sampler.
    let mut timestep_sampling = "sigmoid".to_owned();
    let mut sigmoid_scale = 1.0_f32;
    let mut device_name = "cpu".to_owned();
    let mut gpu_index = 0usize;
    let mut max_resolution = 0_u32;
    let mut bucket_resolution = None::<(u32, u32)>;
    let mut config_path = None;
    let mut explicit_train_cache = false;
    let mut explicit_base_model = false;
    let mut explicit_output_lora = false;
    let mut explicit_steps = false;
    let mut explicit_learning_rate = false;
    let mut explicit_lr_schedule = false;
    let mut explicit_lr_warmup_steps = false;
    let mut explicit_lr_min = false;
    let mut explicit_seed = false;
    let mut explicit_lora_rank = false;
    let mut explicit_lora_alpha = false;
    let mut explicit_flow_matching = false;
    let mut explicit_flow_shift = false;
    let mut explicit_timestep_sampling = false;
    let mut explicit_sigmoid_scale = false;
    let mut explicit_device = false;
    let mut explicit_gpu_index = false;
    let mut explicit_weighting_scheme = false;
    let mut explicit_max_grad_norm = false;
    let mut explicit_checkpoint_every = false;
    let mut explicit_checkpoint_prefix = false;
    let mut output_cache = None;
    let mut replay_input_dir = None::<String>;
    let mut model_cache_dir = "target/afterburner-model-cache".to_owned();
    let mut log_dir = "logs".to_owned();
    let mut memory_budget_gb = None::<f64>;
    let mut explicit_memory_budget = false;
    let mut attention_chunk = 256usize;
    let mut explicit_attention_chunk = false;
    // Compatibility switches modelled after ComfyUI's memory controls.
    // DynamicVRAM module staging is not available in the training backend;
    // smart-memory mode can still avoid pinning static weights in the
    // persistent CubeCL pool.
    let mut disable_dynamic_vram = false;
    let mut disable_smart_memory = false;
    let mut matmul_backward_chunk = 512usize;
    let mut text_token_ids = vec![1_u32, 1_u32];
    let mut tokenizer_path = None;
    let mut preprocess_dtype = "f32".to_owned();
    let mut caption = None;
    let mut repeats = 1usize;
    // Keep normal output to initialization, step progress, and completion.
    // Detailed phase/block tracing is opt-in via --trace.
    let mut compact_progress = true;
    let mut trace_logs = false;
    let mut memory_allocation_trace = false;
    let mut cuda_kernel_dump = None::<String>;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--model-type" {
            if let Some(value) = args.next() {
                model_type = value;
            }
        } else if argument == "--inspect-lora" {
            inspect_lora = args.next();
        } else if argument == "--inspect-lora-factors" {
            inspect_lora_factors = args.next();
        } else if argument == "--inspect-weights" {
            inspect_weights = args.next();
        } else if argument == "--inspect-dataset" {
            inspect_dataset = args.next();
        } else if argument == "--inspect-cache" {
            inspect_cache = args.next();
        } else if argument == "--dump-cache-latent" {
            dump_cache_latent = args.next();
        } else if argument == "--dump-cache-latents" {
            dump_cache_latents = args.next();
        } else if argument == "--inspect-text-encoder" {
            inspect_text_encoder = args.next();
        } else if argument == "--text-encoder" {
            text_encoder_model = args.next();
        } else if argument == "--t5-tokenizer" {
            t5_tokenizer_model = args.next();
        } else if argument == "--inspect-vae" {
            inspect_vae = args.next();
        } else if argument == "--inspect-vae-stem" {
            inspect_vae_stem = args.next();
        } else if argument == "--vae" {
            vae_model = args.next();
        } else if argument == "--preprocess-dataset" {
            preprocess_dataset = args.next();
        } else if argument == "--train-cache" {
            train_cache = args.next();
            explicit_train_cache = true;
        } else if argument == "--base-model" {
            base_model = args.next();
            explicit_base_model = true;
        } else if argument == "--initial-lora" {
            initial_lora = args.next();
        } else if argument == "--output-lora" {
            output_lora = args.next();
            explicit_output_lora = true;
        } else if argument == "--steps" {
            train_steps = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_steps = true;
        } else if argument == "--epochs" {
            train_steps = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_steps = true;
        } else if argument == "--max-optimization-steps" {
            max_optimization_steps = args.next().and_then(|value| value.parse().ok());
        } else if argument == "--learning-rate" {
            learning_rate = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1e-4);
            explicit_learning_rate = true;
        } else if argument == "--lr-schedule" {
            if let Some(value) = args.next() {
                lr_schedule = value;
            }
            explicit_lr_schedule = true;
        } else if argument == "--lr-warmup-steps" {
            lr_warmup_steps = args.next().and_then(|value| value.parse().ok());
            explicit_lr_warmup_steps = true;
        } else if argument == "--lr-min" {
            lr_min = args.next().and_then(|value| value.parse().ok());
            explicit_lr_min = true;
        } else if argument == "--max-grad-norm" {
            max_grad_norm = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.0);
            explicit_max_grad_norm = true;
        } else if argument == "--seed" {
            seed = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(42);
            explicit_seed = true;
        } else if argument == "--lora-rank" {
            lora_rank = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16);
            explicit_lora_rank = true;
        } else if argument == "--lora-alpha" {
            lora_alpha = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16);
            explicit_lora_alpha = true;
        } else if argument == "--lora-dtype" {
            if let Some(value) = args.next() {
                lora_dtype = value;
            }
        } else if argument == "--preprocess-dtype" {
            if let Some(value) = args.next() {
                preprocess_dtype = value;
            }
        } else if argument == "--checkpoint-every" {
            checkpoint_every = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_checkpoint_every = true;
        } else if argument == "--checkpoint-prefix" {
            checkpoint_prefix = args.next();
            explicit_checkpoint_prefix = true;
        } else if argument == "--weighting-scheme" {
            if let Some(value) = args.next() {
                weighting_scheme = value;
            }
            explicit_weighting_scheme = true;
        } else if argument == "--device" {
            if let Some(value) = args.next() {
                device_name = value;
            }
            explicit_device = true;
        } else if argument == "--gpu-index" {
            gpu_index = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_gpu_index = true;
        } else if argument == "--max-resolution" {
            max_resolution = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
        } else if argument == "--bucket-resolution" {
            bucket_resolution = args.next().and_then(|value| {
                let mut parts = value.split(',');
                let width = parts.next()?.parse().ok()?;
                let height = parts.next()?.parse().ok()?;
                Some((width, height))
            });
        } else if argument == "--flow-matching" {
            flow_matching = true;
            explicit_flow_matching = true;
        } else if argument == "--flow-shift" {
            flow_shift = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1.0);
            explicit_flow_shift = true;
        } else if argument == "--timestep-sampling" {
            if let Some(value) = args.next() {
                timestep_sampling = value;
            }
            explicit_timestep_sampling = true;
        } else if argument == "--sigmoid-scale" {
            sigmoid_scale = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1.0);
            explicit_sigmoid_scale = true;
        } else if argument == "--config" {
            config_path = args.next();
        } else if argument == "--output-cache" {
            output_cache = args.next();
        } else if argument == "--replay-input-dir" {
            replay_input_dir = args.next();
        } else if argument == "--model-cache-dir" {
            if let Some(value) = args.next() {
                model_cache_dir = value;
            }
        } else if argument == "--log-dir" {
            if let Some(value) = args.next() {
                log_dir = value;
            }
        } else if argument == "--memory-budget-gb" {
            memory_budget_gb = args.next().and_then(|value| value.parse().ok());
            explicit_memory_budget = true;
        } else if argument == "--attention-chunk" {
            attention_chunk = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            explicit_attention_chunk = true;
        } else if argument == "--disable-dynamic-vram" {
            disable_dynamic_vram = true;
        } else if argument == "--disable-smart-memory" {
            disable_smart_memory = true;
        } else if argument == "--matmul-backward-chunk" {
            matmul_backward_chunk = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(512);
        } else if argument == "--token-ids" {
            if let Some(value) = args.next() {
                text_token_ids = value
                    .split(',')
                    .filter_map(|item| item.trim().parse::<u32>().ok())
                    .collect();
            }
        } else if argument == "--tokenizer" {
            tokenizer_path = args.next();
        } else if argument == "--caption" {
            caption = args.next();
        } else if argument == "--repeats" {
            repeats = args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
        } else if argument == "--compact-progress" {
            compact_progress = true;
        } else if argument == "--trace" {
            trace_logs = true;
            compact_progress = false;
        } else if argument == "--memory-allocation-trace" {
            memory_allocation_trace = true;
        } else if argument == "--cuda-kernel-dump" {
            cuda_kernel_dump = args.next();
        }
    }

    if let Some(path) = config_path {
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            eprintln!("failed to read training config {path}: {error}");
            std::process::exit(2);
        });
        let config: TrainingConfig = serde_json::from_str(&contents).unwrap_or_else(|error| {
            eprintln!("failed to parse training config {path}: {error}");
            std::process::exit(2);
        });
        if !explicit_train_cache {
            train_cache = config.train_cache;
        }
        if !explicit_base_model {
            base_model = config.base_model;
        }
        if !explicit_output_lora {
            output_lora = config.output_lora;
        }
        if !explicit_steps {
            if let Some(value) = config.epochs.or(config.steps) {
                train_steps = value;
            }
        }
        if !explicit_learning_rate {
            if let Some(value) = config.learning_rate {
                learning_rate = value;
            }
        }
        if !explicit_lr_schedule {
            if let Some(value) = config.lr_schedule {
                lr_schedule = value;
            }
        }
        if !explicit_lr_warmup_steps {
            lr_warmup_steps = config.lr_warmup_steps;
        }
        if !explicit_lr_min {
            lr_min = config.lr_min;
        }
        if !explicit_seed {
            if let Some(value) = config.seed {
                seed = value;
            }
        }
        if !explicit_lora_rank {
            if let Some(value) = config.lora_rank {
                lora_rank = value;
            }
        }
        if !explicit_lora_alpha {
            if let Some(value) = config.lora_alpha {
                lora_alpha = value;
            }
        }
        if !explicit_flow_matching {
            if let Some(value) = config.flow_matching {
                flow_matching = value;
            }
        }
        if !explicit_flow_shift {
            if let Some(value) = config.flow_shift {
                flow_shift = value;
            }
        }
        if !explicit_timestep_sampling {
            if let Some(value) = config.timestep_sampling {
                timestep_sampling = value;
            }
        }
        if !explicit_sigmoid_scale {
            if let Some(value) = config.sigmoid_scale {
                sigmoid_scale = value;
            }
        }
        if !explicit_device {
            if let Some(value) = config.device {
                device_name = value;
            }
        }
        if !explicit_gpu_index {
            if let Some(value) = config.gpu_index {
                gpu_index = value;
            }
        }
        if !explicit_weighting_scheme {
            if let Some(value) = config.weighting_scheme {
                weighting_scheme = value;
            }
        }
        if !explicit_max_grad_norm {
            if let Some(value) = config.max_grad_norm {
                max_grad_norm = value;
            }
        }
        if !explicit_checkpoint_every {
            if let Some(value) = config.checkpoint_every {
                checkpoint_every = value;
            }
        }
        if !explicit_checkpoint_prefix {
            checkpoint_prefix = config.checkpoint_prefix;
        }
        if !explicit_memory_budget {
            memory_budget_gb = config.memory_budget_gb;
        }
        if !explicit_attention_chunk {
            if let Some(value) = config.attention_chunk {
                attention_chunk = value;
            }
        }
        if let Some(value) = config.model_cache_dir {
            model_cache_dir = value;
        }
    }

    if trace_logs {
        unsafe { std::env::set_var("AFTERBURNER_TRACE", "1") };
    } else if compact_progress {
        // The model crate intentionally keeps its progress output independent
        // from CLI parsing. Propagate this explicit CLI choice to the model
        // execution path before any training/preprocessing starts.
        unsafe { std::env::set_var("AFTERBURNER_COMPACT_PROGRESS", "1") };
    }
    if memory_allocation_trace || trace_logs {
        unsafe { std::env::set_var("AFTERBURNER_CUBECL_ALLOC_TRACE", "1") };
        println!("memory_allocation_trace=true");
    }

    // ComfyUI's low-VRAM mode keeps model and adapter parameters in the
    // backend's reduced precision and recomputes long projections in small
    // pieces.  Make the two explicit low-VRAM switches select the same
    // behavior for the training path instead of merely changing log text.
    if disable_dynamic_vram || disable_smart_memory {
        unsafe {
            std::env::set_var("AFTERBURNER_BASE_BF16", "1");
            std::env::set_var("AFTERBURNER_LORA_BF16", "1");
            std::env::set_var("AFTERBURNER_MIXED_PRECISION", "1");
            // Do not force per-linear recomputation here.  Python's
            // checkpointing boundary is block-scoped; replaying every large
            // Q/K/V/O and MLP projection outside that region creates a nested
            // backward graph and can grow the host commit without making
            // optimizer progress. The block-scoped strategy is the only
            // recomputation policy.
            std::env::remove_var("AFTERBURNER_MLP_RECOMPUTE");
            std::env::remove_var("AFTERBURNER_RECOMPUTE_ALL_LINEAR");
            std::env::set_var("AFTERBURNER_DISABLE_LINEAR_RECOMPUTE", "1");
            // ComfyUI's disable-smart-memory mode aggressively returns dead
            // pages to the driver. Do not retain CubeCL pages here: retaining
            // them makes Task Manager report the allocator high-water mark as
            // live VRAM and defeats the requested low-memory behavior.
            std::env::remove_var("AFTERBURNER_RETAIN_MEMORY_POOLS");
            std::env::remove_var("AFTERBURNER_DISABLE_MEMORY_CLEANUP");
            if std::env::var_os("AFTERBURNER_LINEAR_CHUNK").is_none() {
                std::env::set_var("AFTERBURNER_LINEAR_CHUNK", "512");
            }
        }
        println!("low_vram_precision=bf16");
        println!("low_vram_linear_chunk=512");
    }

    let model_type = match model_type.parse::<afterburner::ModelType>() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let (revision, diffusion_model, text_encoder, vae) = match model_type {
        afterburner::ModelType::Anima => (
            "anima-aesthetic-v1.1",
            "anima-aesthetic-v1.1.safetensors",
            "qwen_3_06b_base.safetensors",
            "qwen_image_vae.safetensors",
        ),
        afterburner::ModelType::Flux => (
            "flux",
            "flux.safetensors",
            "text-encoder.safetensors",
            "flux2-vae.safetensors",
        ),
        afterburner::ModelType::StableDiffusion => (
            "stable-diffusion",
            "unet.safetensors",
            "clip.safetensors",
            "vae.safetensors",
        ),
    };
    let config =
        afterburner::ModelConfig::new(model_type, revision, diffusion_model, text_encoder, vae);
    let descriptor = afterburner::resolve_model(config).expect("valid model configuration");
    let log_dir_path = PathBuf::from(&log_dir);
    std::fs::create_dir_all(&log_dir_path).unwrap_or_else(|error| {
        eprintln!(
            "failed to create log directory {}: {error}",
            log_dir_path.display()
        );
        std::process::exit(2);
    });
    let log_path = log_dir_path.join(format!(
        "afterburner-{}.log",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    ));
    init_session_log(&log_path);
    if let Some(path) = cuda_kernel_dump.as_deref() {
        // CubeCL reads its global logging configuration before the CUDA
        // runtime/client is initialized.  Full debug mode records generated
        // kernel compilation details and executed-kernel profiling entries.
        unsafe {
            std::env::set_var("CUBECL_DEBUG_LOG", path);
            std::env::set_var("CUBECL_DEBUG_OPTION", "debug-full");
        }
        println!("cuda_kernel_dump={path}");
    }
    if memory_allocation_trace || trace_logs {
        let allocation_log_path = log_path.with_extension("memory.log");
        unsafe {
            std::env::set_var(
                "AFTERBURNER_CUBECL_ALLOC_TRACE_FILE",
                allocation_log_path.to_string_lossy().as_ref(),
            );
        }
        println!("memory_allocation_log={}", allocation_log_path.display());
    }
    println!("Afterburner CLI {}", afterburner::VERSION);
    println!("log_path={}", log_path.display());
    println!("model_type={model_type}");
    println!("resolved={descriptor:?}");
    if let Some(path) = inspect_lora {
        let plan = afterburner::inspect_anima_lora_file(path, &afterburner::LoraConfig::default())
            .expect("failed to inspect Anima LoRA targets");
        println!("lora_targets={}", plan.target_count());
        println!("lora_parameters={}", plan.parameter_count());
    }
    if let Some(path) = inspect_lora_factors {
        let plan = afterburner::inspect_anima_lora_factors(path)
            .expect("failed to inspect exported Anima LoRA factors");
        println!("lora_factor_tensors={}", plan.tensor_count());
        println!("lora_factor_parameters={}", plan.parameter_count);
        println!("lora_factor_f32_tensors={}", plan.f32_tensors);
        println!("lora_factor_bf16_tensors={}", plan.bf16_tensors);
        println!("lora_factor_f16_tensors={}", plan.f16_tensors);
        println!("lora_factor_storage_bytes={}", plan.storage_bytes);
    }
    if let Some(path) = inspect_weights {
        let coverage = afterburner::inspect_anima_weight_coverage(path)
            .expect("failed to inspect Anima weight coverage");
        println!("mapped_weights={}", coverage.mapped_count());
        println!("unmapped_weights={}", coverage.unmapped_count());
        for key in coverage.unmapped {
            println!("unmapped={key}");
        }
    }
    if let Some(path) = inspect_dataset {
        let dataset = afterburner::ImageCaptionDataset::from_dir_with_folder_repeats(path, repeats)
            .expect("failed to inspect image/caption dataset");
        println!("dataset_samples={}", dataset.samples().len());
        println!("dataset_repeats={repeats}");
        for sample in dataset.samples() {
            println!(
                "sample={} repeat={} caption={}",
                sample.image_path.display(),
                sample.repeat_index,
                sample.caption
            );
        }
    }
    if let Some(path) = inspect_cache {
        let cache = afterburner::InMemoryTrainingCache::load(path)
            .expect("failed to load encoded training cache");
        println!("cache_samples={}", cache.len());
        println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
        if let Some(sample) = cache.get(0) {
            println!("cache_first_sample_id={}", sample.sample_id);
            println!("cache_first_repeat_index={}", sample.repeat_index);
            println!("cache_first_text_shape={:?}", sample.text_embedding.shape);
            println!("cache_first_latent_shape={:?}", sample.image_latent.shape);
            println!(
                "cache_first_text_mask_shape={:?}",
                sample.text_attention_mask.as_ref().map(|mask| &mask.shape)
            );
            if let Some(mask) = &sample.text_attention_mask {
                println!(
                    "cache_first_text_mask_valid_tokens={}",
                    mask.data.iter().filter(|value| **value > 0.0).count()
                );
            }
            print_cache_tensor_stats("text", &sample.text_embedding);
            print_cache_tensor_stats("latent", &sample.image_latent);
            if let Some(output) = dump_cache_latent.as_deref() {
                let output = Path::new(output);
                if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent).expect("failed to create cache latent dump directory");
                }
                let mut bytes = Vec::with_capacity(sample.image_latent.data.len() * 4);
                for value in &sample.image_latent.data {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                std::fs::write(output, bytes).expect("failed to write cache latent dump");
                let shape_path = output.with_extension("shape.json");
                std::fs::write(
                    shape_path,
                    serde_json::to_vec_pretty(&sample.image_latent.shape)
                        .expect("failed to serialize cache latent shape"),
                )
                .expect("failed to write cache latent shape");
                println!("cache_first_latent_dump={}", output.display());
            }
            if let Some(directory) = dump_cache_latents.as_deref() {
                let directory = Path::new(directory);
                std::fs::create_dir_all(directory)
                    .expect("failed to create cache latent dump directory");
                for (index, item) in cache.iter().enumerate() {
                    let stem = format!("sample-{index:06}");
                    let data_path = directory.join(format!("{stem}.f32le"));
                    let mut bytes = Vec::with_capacity(item.image_latent.data.len() * 4);
                    for value in &item.image_latent.data {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    std::fs::write(&data_path, bytes)
                        .expect("failed to write cache latent dump");
                    std::fs::write(
                        directory.join(format!("{stem}.shape.json")),
                        serde_json::to_vec(&item.image_latent.shape)
                            .expect("failed to serialize cache latent shape"),
                    )
                    .expect("failed to write cache latent shape");
                    std::fs::write(
                        directory.join(format!("{stem}.manifest.json")),
                        serde_json::to_vec_pretty(&serde_json::json!({
                            "sample_id": item.sample_id,
                            "repeat_index": item.repeat_index,
                            "shape": item.image_latent.shape,
                        }))
                        .expect("failed to serialize cache latent manifest"),
                    )
                    .expect("failed to write cache latent manifest");
                }
                println!("cache_latents_dumped={} directory={}", cache.len(), directory.display());
            }
            if let Some(adapter) = &sample.adapter_conditioning {
                println!(
                    "cache_first_adapter_token_count={}",
                    adapter.input_ids.len()
                );
                println!(
                    "cache_first_adapter_mask_shape={:?}",
                    adapter.attention_mask.shape
                );
                println!(
                    "cache_first_adapter_valid_tokens={}",
                    adapter
                        .attention_mask
                        .data
                        .iter()
                        .filter(|value| **value > 0.0)
                        .count()
                );
            } else {
                println!("cache_first_adapter_token_count=none");
            }
        }
    }
    if let Some(path) = inspect_text_encoder.clone() {
        let device = candle_device(&device_name, gpu_index);
        if let Some(tokenizer_path) = tokenizer_path.clone() {
            let tokenizer = afterburner::QwenTokenizer::from_file(tokenizer_path, 512, 151_643)
                .expect("failed to load Qwen tokenizer");
            let mut encoder = afterburner::CandleQwen3TextEncoder::from_safetensors(
                path,
                afterburner::Qwen3TextConfig::default(),
                candle_core::DType::F32,
                &device,
                tokenizer,
            )
            .expect("failed to load Qwen3 text encoder");
            let caption = caption.as_deref().unwrap_or("a portrait");
            let (embedding, mask) =
                afterburner::TextEncoder::encode_text_with_mask(&mut encoder, caption)
                    .expect("failed to encode caption");
            println!("text_encoder_embedding_shape={:?}", embedding.shape);
            println!(
                "text_encoder_attention_mask_shape={:?}",
                mask.as_ref().map(|value| value.shape.clone())
            );
            if let Some(mask) = mask {
                println!(
                    "text_encoder_attention_mask_valid_tokens={}",
                    mask.data.iter().filter(|value| **value > 0.0).count()
                );
            }
        } else {
            let mut encoder = afterburner::CandleQwen3TextEncoder::from_safetensors(
                path,
                afterburner::Qwen3TextConfig::default(),
                candle_core::DType::F32,
                &device,
                |_: &str| Ok::<_, std::convert::Infallible>(text_token_ids.clone()),
            )
            .expect("failed to load Qwen3 text encoder");
            let embedding = encoder
                .encode_token_ids(&text_token_ids)
                .expect("failed to run Qwen3 text encoder");
            println!("text_encoder_embedding_shape={:?}", embedding.shape);
        }
    }
    if let Some(path) = inspect_vae {
        let report =
            afterburner::inspect_qwen_image_vae(path).expect("failed to inspect Qwen Image VAE");
        println!("vae_tensor_count={}", report.tensor_count);
        println!("vae_bfloat16_tensors={}", report.bfloat16_tensors);
        println!(
            "vae_required_keys={}/{}",
            report.required_keys_present, report.required_keys
        );
        for key in report.missing_required_keys {
            println!("vae_missing={key}");
        }
    }
    if let Some(path) = inspect_vae_stem.clone() {
        let device = candle_device(&device_name, gpu_index);
        let stem = afterburner::CandleQwenImageVaeStem::from_safetensors(
            path,
            candle_core::DType::F32,
            &device,
        )
        .expect("failed to load Qwen Image VAE stem");
        let image = image::RgbImage::from_pixel(8, 8, image::Rgb([127, 127, 127]));
        let input =
            afterburner::qwen_image_input(&image, &afterburner::QwenImageVaeConfig::default())
                .expect("failed to prepare Qwen Image VAE input");
        let output = stem.forward(&input).expect("failed to run Qwen VAE stem");
        println!("vae_stem_output_shape={:?}", output.shape);
        let residual = stem
            .forward_first_residual(&input)
            .expect("failed to run Qwen VAE first residual block");
        println!("vae_first_residual_output_shape={:?}", residual.shape);
        let downsample = stem
            .forward_first_downsample(&input)
            .expect("failed to run Qwen VAE first downsample");
        println!("vae_first_downsample_output_shape={:?}", downsample.shape);
        let stack = stem
            .forward_downsample_stack(&input)
            .expect("failed to run Qwen VAE downsample stack");
        println!("vae_downsample_stack_output_shape={:?}", stack.shape);
        let latent = stem
            .forward_latent(&input)
            .expect("failed to run complete Qwen VAE encoder");
        println!("vae_latent_output_shape={:?}", latent.shape);
    }
    if let Some(cache_path) = train_cache {
        let base_model = base_model
            .as_deref()
            .expect("--train-cache requires --base-model");
        let output_lora = output_lora
            .as_deref()
            .expect("--train-cache requires --output-lora");
        if train_steps == 0 {
            eprintln!("--epochs must be greater than zero");
            std::process::exit(2);
        }
        let cache_load_started = std::time::Instant::now();
        eprintln!("[load] loading training cache: {cache_path}");
        let cache = afterburner::InMemoryTrainingCache::load(&cache_path).unwrap_or_else(|error| {
            eprintln!("failed to load encoded training cache {cache_path}: {error}");
            eprintln!("provide the complete .afbcache path; `...` is not a wildcard");
            std::process::exit(2);
        });
        let replay_inputs = replay_input_dir.as_deref().map(|path| {
            load_replay_inputs(path, &cache).unwrap_or_else(|error| {
                panic!("failed to load Python replay inputs: {error}");
            })
        });
        if let Some(inputs) = &replay_inputs {
            println!("python_replay_inputs={}", inputs.len());
        }
        eprintln!(
            "[load] training cache loaded: samples={} tensor_bytes={} elapsed={:.3}s",
            cache.len(),
            cache.total_tensor_bytes(),
            cache_load_started.elapsed().as_secs_f64()
        );
        let training_epochs = train_steps;
        let full_optimization_steps = training_epochs
            .checked_mul(cache.len())
            .expect("--steps multiplied by cache size overflows");
        let optimization_steps = max_optimization_steps
            .map(|limit| limit.min(full_optimization_steps))
            .unwrap_or(full_optimization_steps);
        if optimization_steps == 0 {
            eprintln!("--max-optimization-steps must be greater than zero");
            std::process::exit(2);
        }
        if optimization_steps != full_optimization_steps {
            eprintln!(
                "optimization_step_limit={optimization_steps} (full_run={full_optimization_steps})"
            );
        }
        // Keep the scheduler horizon tied to the requested training run, not
        // to an optional smoke-test cap.  kohya_ss computes its cosine/warmup
        // schedule from the planned max_train_steps and may save an early
        // checkpoint before that horizon is reached.
        let scheduler_total_steps = full_optimization_steps;
        if scheduler_total_steps != optimization_steps {
            eprintln!(
                "scheduler_total_steps={scheduler_total_steps} (optimization_step_limit={optimization_steps})"
            );
        }
        let mut lora = afterburner::LoraConfig::default();
        lora.rank = lora_rank;
        lora.alpha = lora_alpha;
        lora.validate().expect("invalid LoRA configuration");
        let output_dtype = match lora_dtype.as_str() {
            "f32" | "float32" => afterburner::LoraOutputDtype::F32,
            "bf16" | "bfloat16" => afterburner::LoraOutputDtype::BF16,
            "f16" | "fp16" | "float16" => afterburner::LoraOutputDtype::F16,
            value => {
                eprintln!("unsupported --lora-dtype: {value} (expected f32, f16, or bf16)");
                std::process::exit(2);
            }
        };
        #[cfg(any(feature = "bf16", feature = "f16"))]
        let backend_dtype_bytes = 2usize;
        #[cfg(all(not(feature = "bf16"), not(feature = "f16")))]
        let backend_dtype_bytes = 4usize;
        if attention_chunk == 0 {
            eprintln!("--attention-chunk must be greater than zero");
            std::process::exit(2);
        }
        let requested_attention_chunk = attention_chunk;
        let token_count = cache
            .get(0)
            .map(|sample| {
                sample.image_latent.shape[3]
                    .checked_div(2)
                    .unwrap_or(0)
                    .saturating_mul(sample.image_latent.shape[4].checked_div(2).unwrap_or(0))
            })
            .unwrap_or(0);
        // At 1024px the 3536-token Anima sequence makes a 256/64-query
        // checkpoint/backward workspace exceed the practical 16 GiB class,
        // even though the post-cleanup allocator report looks small. Cap the
        // transient workspace automatically for constrained runs; smaller
        // caches retain the user-selected value.
        if memory_budget_gb.is_some_and(|budget| budget <= 16.0)
            && (disable_dynamic_vram || disable_smart_memory)
            && token_count > 2048
            && attention_chunk > 8
        {
            attention_chunk = 8;
            eprintln!(
                "attention_chunk_auto_cap=requested:{requested_attention_chunk},effective:{attention_chunk},tokens:{token_count},budget_gb:{:.1}",
                memory_budget_gb.unwrap_or(16.0)
            );
        }
        if matmul_backward_chunk == 0 {
            eprintln!("--matmul-backward-chunk must be greater than zero");
            std::process::exit(2);
        }
        // The previous unconditional cap of 16 made every constrained run
        // submit tiny backward matmul tiles, even with an 18-20 GiB budget.
        // Keep the genuinely tight 16 GiB profile safe, but allow the user
        // supplied 32/64 tile on larger cards so CUDA can reach useful
        // occupancy instead of spending the step in kernel-launch overhead.
        if (disable_dynamic_vram || disable_smart_memory) && matmul_backward_chunk > 16 {
            let cap = match memory_budget_gb {
                Some(budget) if budget <= 16.0 => 16,
                Some(budget) if budget <= 20.0 => 64,
                _ => 128,
            };
            if matmul_backward_chunk > cap {
                matmul_backward_chunk = cap;
                println!("matmul_backward_chunk_auto_cap={cap}");
            }
        }
        afterburner::set_attention_chunk_size(attention_chunk)
            .expect("validated attention chunk size");
        burn_autodiff::ops::set_matmul_backward_chunk_size(matmul_backward_chunk);
        println!("attention_chunk={attention_chunk}");
        println!("attention_chunk_requested={requested_attention_chunk}");
        println!("matmul_backward_chunk={matmul_backward_chunk}");
        let memory_preflight =
            estimate_training_memory(base_model, &cache, backend_dtype_bytes, attention_chunk)
                .unwrap_or_else(|error| {
                    eprintln!("memory preflight failed: {error}");
                    std::process::exit(2);
                });
        print_memory_preflight(memory_preflight);
        if let Some(budget_gb) = memory_budget_gb {
            if !budget_gb.is_finite() || budget_gb <= 0.0 {
                eprintln!("--memory-budget-gb must be a positive finite number");
                std::process::exit(2);
            }
            let budget_bytes = budget_gb * 1024.0 * 1024.0 * 1024.0;
            if memory_preflight.estimated_training_bytes as f64 > budget_bytes {
                eprintln!(
                    "memory preflight rejected run: estimated {:.2} GiB exceeds budget {:.2} GiB; no model tensors were allocated",
                    memory_preflight.estimated_training_bytes as f64 / 1024.0_f64.powi(3),
                    budget_gb
                );
                std::process::exit(3);
            }
            eprintln!(
                "memory preflight accepted: estimated {:.2} GiB <= budget {:.2} GiB",
                memory_preflight.estimated_training_bytes as f64 / 1024.0_f64.powi(3),
                budget_gb
            );
        }
        #[cfg(all(feature = "cuda", feature = "bf16"))]
        type BaseBackend = burn_cuda::Cuda<half::bf16>;
        #[cfg(all(feature = "cuda", not(feature = "bf16"), feature = "f16"))]
        type BaseBackend = burn_cuda::Cuda<half::f16>;
        #[cfg(all(feature = "cuda", not(feature = "bf16"), not(feature = "f16")))]
        type BaseBackend = burn_cuda::Cuda;
        #[cfg(all(not(feature = "cuda"), feature = "vulkan", feature = "bf16"))]
        type BaseBackend = burn::backend::Vulkan<half::bf16>;
        #[cfg(all(not(feature = "cuda"), feature = "vulkan", not(feature = "bf16"), feature = "f16"))]
        type BaseBackend = burn::backend::Vulkan<half::f16>;
        #[cfg(all(not(feature = "cuda"), feature = "vulkan", not(feature = "bf16"), not(feature = "f16")))]
        type BaseBackend = burn::backend::Vulkan;
        #[cfg(all(not(feature = "cuda"), not(feature = "vulkan")))]
        type BaseBackend = burn_ndarray::NdArray<f32>;
        #[cfg(feature = "checkpointing")]
        type TrainBackend = burn_autodiff::Autodiff<
            BaseBackend,
            burn_autodiff::checkpoint::strategy::NoCheckpointing,
        >;
        #[cfg(not(feature = "checkpointing"))]
        type TrainBackend = burn_autodiff::Autodiff<BaseBackend>;
        // CUDA training uses the fused Autodiff attention boundary.  The
        // portable backends retain the reference implementation so the CLI
        // keeps the same numerical fallback everywhere else.
        #[cfg(feature = "cuda")]
        type TrainAttention = afterburner::AnimaAttentionCuda<TrainBackend>;
        #[cfg(not(feature = "cuda"))]
        type TrainAttention = afterburner::AnimaAttentionStandard<TrainBackend>;
        type TrainModel = afterburner::AnimaDiT<TrainBackend, TrainAttention>;
        #[cfg(feature = "cuda")]
        eprintln!("[load] initializing Burn CUDA device");
        #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
        eprintln!("[load] initializing Burn Vulkan device");
        let device = {
            #[cfg(feature = "cuda")]
            {
                burn_cuda::CudaDevice::new(gpu_index)
            }
            #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
            {
                let device = burn::backend::wgpu::WgpuDevice::default();
                burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
                    &device,
                    Default::default(),
                );
                device
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "vulkan")))]
            {
                <TrainBackend as burn::tensor::backend::Backend>::Device::default()
            }
        };
        #[cfg(feature = "cuda")]
        eprintln!("[load] Burn CUDA device initialized");
        #[cfg(all(not(feature = "cuda"), feature = "vulkan"))]
        eprintln!("[load] Burn Vulkan device initialized");
        #[cfg(feature = "cuda")]
        use cubecl_core::Runtime;
        #[cfg(feature = "cuda")]
        let memory_client = <cubecl_cuda::CudaRuntime as Runtime>::client(&device);
        eprintln!("loading Anima base model: {base_model}");
        #[cfg(feature = "cuda")]
        let (mut model, load_report) = {
            // Afterburner does not yet have ComfyUI's CPU-offload machinery.
            // Initialize the architecture in the normal allocator first.
            // Only the SafeTensors replacement pass is persistent; otherwise
            // the random initialization weights remain pinned beside the
            // loaded weights in CubeCL's persistent pool.
            eprintln!("[load] AnimaDiT initialization started");
            let init_started = std::time::Instant::now();
            let mut model = afterburner::AnimaDiT::<TrainBackend, TrainAttention>::init(
                &afterburner::AnimaConfig::default(),
                &lora,
                &device,
            )
            .expect("failed to initialize Anima model");
            eprintln!(
                "[load] AnimaDiT initialization finished in {:.3}s; loading SafeTensors",
                init_started.elapsed().as_secs_f64()
            );
            let load_started = std::time::Instant::now();
            let load_report = memory_client.memory_persistent_allocation((), |_| {
                afterburner::load_anima_weights(base_model, &mut model)
                    .expect("failed to load Anima base model")
            });
            eprintln!(
                "[load] SafeTensors loading finished in {:.3}s (loaded={}, skipped={})",
                load_started.elapsed().as_secs_f64(),
                load_report.loaded_count(),
                load_report.skipped_count()
            );
            (model, load_report)
        };
        #[cfg(not(feature = "cuda"))]
        let (mut model, load_report) = afterburner::load_anima_model_with_attention::<
            TrainBackend,
            TrainAttention,
        >(base_model, &lora, &device)
        .expect("failed to load Anima base model");
        // Model initialization creates temporary random parameter tensors
        // before SafeTensors replaces them.  With an Autodiff backend those
        // initialization nodes can otherwise keep the old allocations alive
        // into the first training step, effectively doubling model VRAM.
        burn_autodiff::cleanup_dead_nodes();
        #[cfg(feature = "cuda")]
        memory_client.memory_cleanup();
        println!("base_model_loaded={}", load_report.loaded_count());
        println!("base_model_skipped={}", load_report.skipped_count());
        if let Some(initial_lora_path) = initial_lora.as_deref() {
            eprintln!("[load] loading initial LoRA factors: {initial_lora_path}");
            let loaded = afterburner::load_anima_lora_factors::<TrainBackend, TrainAttention>(
                initial_lora_path,
                &mut model,
            )
            .unwrap_or_else(|error| {
                eprintln!("failed to load initial LoRA factors: {error}");
                std::process::exit(2);
            });
            println!("initial_lora_loaded={loaded}");
        }
        #[cfg(feature = "cuda")]
        {
            let usage = memory_client.memory_usage();
            eprintln!(
                "[memory] after base load: in_use={} reserved={} allocations={}",
                usage.bytes_in_use, usage.bytes_reserved, usage.number_allocs
            );
        }
        let schedule = if flow_matching {
            // Match the kohya_ss Anima reference configuration.
            afterburner::DiffusionSchedule::flow_matching(1000, flow_shift)
                .expect("invalid flow-matching schedule")
        } else {
            afterburner::DiffusionSchedule::linear(1000, 0.0001, 0.02)
                .expect("invalid diffusion schedule")
        };
        let target = if flow_matching {
            afterburner::TrainingTarget::FlowMatching
        } else {
            afterburner::TrainingTarget::Epsilon
        };
        println!(
            "training_target={}",
            if flow_matching {
                "flow_matching"
            } else {
                "epsilon"
            }
        );
        if flow_matching {
            println!("flow_shift={flow_shift}");
            println!("seed={seed}");
        }
        #[cfg(feature = "checkpointing")]
        // The dynamic block checkpoint is not enabled until the Anima block
        // snapshot path is wired into the forward loop. Do not label the
        // current legacy operator-granular strategy as Python-compatible.
        println!("autodiff_checkpointing=burn_dynamic_block_checkpoint");
        #[cfg(not(feature = "checkpointing"))]
        println!("autodiff_checkpointing=disabled");
        let loss_weighting = match weighting_scheme.as_str() {
            "none" => afterburner::LossWeighting::None,
            // kohya_ss uses `logit_normal` for timestep/sigma sampling. It
            // does not apply an additional per-sample loss weight for this
            // setting; the Sigma sampler is selected in the training loop.
            "logit_normal" => afterburner::LossWeighting::None,
            "sigma_sqrt" => afterburner::LossWeighting::SigmaSqrt,
            "cosmap" => afterburner::LossWeighting::Cosmap,
            value => {
                eprintln!("unsupported --weighting-scheme: {value}");
                std::process::exit(2);
            }
        };
        if !sigmoid_scale.is_finite() || sigmoid_scale <= 0.0 {
            eprintln!("--sigmoid-scale must be positive and finite");
            std::process::exit(2);
        }
        let timestep_sampling = match timestep_sampling.as_str() {
            "sigmoid" => afterburner::TimestepSampling::Sigmoid {
                mean: 0.0,
                std: sigmoid_scale,
            },
            "uniform" => afterburner::TimestepSampling::Uniform,
            "sigma" => afterburner::TimestepSampling::Sigma {
                mean: 0.0,
                std: sigmoid_scale,
            },
            value => {
                eprintln!("unsupported --timestep-sampling: {value} (expected sigmoid, uniform, or sigma)");
                std::process::exit(2);
            }
        };
        println!("timestep_sampling={timestep_sampling:?}");
        let optimizer_epsilon = 1e-8_f32;
        println!("optimizer_epsilon={optimizer_epsilon:.1e}");
        println!("optimizer=AdamW8bit");
        println!("optimizer_state=int8_blockwise");
        println!("optimizer_state_quantization=Q8S");
        println!("optimizer_master_parameters=f32");
        println!("optimizer_weight_decay=0");
        println!("loss_scale=1");
        let optimizer_config = burn_optim::AdamW8bitConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(optimizer_epsilon)
            // kohya_ss's AdamW8bit with an empty optimizer_args uses the
            // bitsandbytes default: no decoupled weight decay.
            .with_weight_decay(0.0);
        if max_grad_norm > 0.0 {
            // kohya_ss uses global gradient-norm clipping (`max_grad_norm`),
            // not per-element value clipping.  The latter changes the
            // gradient distribution and therefore cannot reproduce Python
            // training.  Keep the reference policy here; the vendored Burn
            // implementation sanitizes invalid lanes before reduction.
            println!("gradient_clipping=global_norm");
        }
        let mut optimizer = optimizer_config.init::<TrainBackend, TrainModel>();
        let mut logger = afterburner::StderrFileLogger::open(&log_path)
            .expect("failed to open session log file");
        let checkpoint_lora = lora.clone();
        let mut checkpoint = checkpoint_prefix.map(|prefix| {
            move |step: usize, model: &TrainModel| -> Result<(), afterburner::TrainingError> {
                let lora_path = format!("{prefix}-{step:08}.safetensors");
                afterburner::export_anima_lora_with_config_and_dtype_with_attention(
                    &lora_path,
                    model,
                    &checkpoint_lora,
                    output_dtype,
                )
                .map(|_| ())
                .map_err(|error| afterburner::TrainingError::Checkpoint(error.to_string()))
            }
        });
        let checkpoint =
            checkpoint.as_mut().map(
                |callback| -> &mut dyn FnMut(
                    usize,
                    &TrainModel,
                ) -> Result<(), afterburner::TrainingError> { callback },
            );
        let learning_rate_policy = match lr_schedule.as_str() {
            "constant" => afterburner::LearningRatePolicy::Constant { learning_rate },
            "warmup_cosine" | "cosine" => {
                let warmup_steps = lr_warmup_steps
                    .map(|value| {
                        if value > 0.0 && value < 1.0 {
                            (value * scheduler_total_steps as f64).floor() as usize
                        } else {
                            value.max(0.0).floor() as usize
                        }
                    })
                    .unwrap_or((scheduler_total_steps / 10).max(1));
                let min_learning_rate = lr_min.unwrap_or(0.0);
                if min_learning_rate < 0.0 || min_learning_rate > learning_rate {
                    eprintln!("--lr-min must be between 0 and --learning-rate");
                    std::process::exit(2);
                }
                println!(
                    "lr_warmup_steps={} (of {}), lr_decay_begin_step={}, lr_min={:.3e}",
                    warmup_steps,
                    scheduler_total_steps,
                    warmup_steps.saturating_add(1),
                    min_learning_rate
                );
                afterburner::LearningRatePolicy::WarmupCosine {
                    peak_learning_rate: learning_rate,
                    min_learning_rate,
                    warmup_steps,
                    total_steps: scheduler_total_steps,
                }
            }
            value => {
                eprintln!(
                    "unsupported --lr-schedule: {value} (expected warmup_cosine or constant)"
                );
                std::process::exit(2);
            }
        };
        println!("lr_schedule={lr_schedule}");
        println!("disable_dynamic_vram={disable_dynamic_vram}");
        println!("disable_smart_memory={disable_smart_memory}");
        #[cfg(feature = "cuda")]
        println!("cubecl_persistent_memory=static_weights_only");
        if disable_dynamic_vram {
            eprintln!(
                "[memory] DynamicVRAM module staging is not implemented in the training backend; using static model loading"
            );
        }
        if disable_smart_memory {
            eprintln!(
                "[memory] smart-memory CPU offload requested, but unavailable in this backend; keeping static weights persistent and reclaiming transient pages"
            );
        }
        #[cfg(feature = "cuda")]
        let runtime_memory_budget_bytes =
            memory_budget_gb.map(|value| (value * 1024.0_f64.powi(3)).min(u64::MAX as f64) as u64);
        #[cfg(feature = "cuda")]
        let mut previous_memory_in_use = None::<u64>;
        #[cfg(feature = "cuda")]
        let mut phase_memory_step = 0usize;
        #[cfg(feature = "cuda")]
        let mut phase_peak_in_use = 0u64;
        #[cfg(feature = "cuda")]
        let mut phase_peak_reserved = 0u64;
        #[cfg(feature = "cuda")]
        let retain_memory_pools = std::env::var_os("AFTERBURNER_RETAIN_MEMORY_POOLS").is_some();
        #[cfg(feature = "cuda")]
        let allow_trim_overcommit = std::env::var_os("AFTERBURNER_CUDA_TRIM_GB").is_some();
        #[cfg(feature = "cuda")]
        let mut memory_hook = |step: usize, phase: &'static str| {
            if phase == "step_cleanup_complete" && !retain_memory_pools {
                // Drop only free transient work pages; model weights and live
                // optimizer state remain resident for the next step.
                memory_client.memory_cleanup();
            }
            if phase == "step_cleanup_complete"
                && let Ok(gb) = std::env::var("AFTERBURNER_CUDA_TRIM_GB")
                && let Ok(gb) = gb.parse::<f64>()
            {
                let min_bytes = (gb.max(0.0) * 1024.0_f64.powi(3)) as usize;
                if let Err(error) = trim_cuda_default_pool(min_bytes) {
                    eprintln!("[cuda-memory] pool trim failed: {error}");
                } else {
                    println!("[cuda-memory] pool trimmed: min_reserved_gb={gb:.2}");
                }
            }
            let usage = memory_client.memory_usage();
            if matches!(phase, "forward_loss_complete" | "backward_optimizer_complete" | "step_cleanup_complete") {
                let driver = cuda_driver_memory()
                    .map(|(used, free)| format!(" driver_used={used} driver_free={free}"))
                    .unwrap_or_default();
                println!(
                    "[memory-summary] step={step} phase={phase} in_use={} reserved={} allocations={}{}",
                    usage.bytes_in_use,
                    usage.bytes_reserved,
                    usage.number_allocs,
                    driver
                );
            }
            if phase_memory_step != step {
                phase_memory_step = step;
                phase_peak_in_use = 0;
                phase_peak_reserved = 0;
            }
            phase_peak_in_use = phase_peak_in_use.max(usage.bytes_in_use);
            phase_peak_reserved = phase_peak_reserved.max(usage.bytes_reserved);
            if let Some(budget_bytes) = runtime_memory_budget_bytes {
                if usage.bytes_reserved > budget_bytes {
                    if allow_trim_overcommit {
                        eprintln!(
                            "memory budget overcommit at step={step} phase={phase}: reserved={} budget={budget_bytes}; continuing to step cleanup for CUDA pool trim",
                            usage.bytes_reserved
                        );
                    } else {
                        eprintln!(
                            "memory budget exceeded at step={step} phase={phase}: reserved={} budget={budget_bytes}; stopping before further allocations",
                            usage.bytes_reserved
                        );
                        std::process::exit(3);
                    }
                }
            }
            let delta = previous_memory_in_use
                .map(|previous| usage.bytes_in_use.saturating_sub(previous))
                .unwrap_or(0);
            if trace_logs {
                eprintln!(
                    "[memory] step={step} phase={phase}: in_use={} reserved={} delta_in_use={} allocations={} peak_in_use={} peak_reserved={}",
                    usage.bytes_in_use,
                    usage.bytes_reserved,
                    delta,
                    usage.number_allocs,
                    phase_peak_in_use,
                    phase_peak_reserved
                );
            }
            previous_memory_in_use = Some(usage.bytes_in_use);
        };
        #[cfg(feature = "cuda")]
        let memory_hook: Option<&mut dyn FnMut(usize, &'static str)> = Some(&mut memory_hook);
        #[cfg(not(feature = "cuda"))]
        let memory_hook: Option<&mut dyn FnMut(usize, &'static str)> = None;
        let (model, report) =
            afterburner::train_cached_with_replay(
                &mut optimizer,
                model,
                &cache,
                &schedule,
                optimization_steps,
                learning_rate_policy,
                seed,
                target,
                timestep_sampling,
                lora.clone(),
                &mut logger,
                (checkpoint_every > 0).then_some(checkpoint_every),
                checkpoint,
                loss_weighting,
                memory_hook,
                replay_inputs.as_deref(),
            )
            .expect("cached training failed");
        let factor_count = afterburner::export_anima_lora_with_config_and_dtype_with_attention(
            output_lora,
            &model,
            &lora,
            output_dtype,
        )
        .expect("failed to export LoRA factors");
        println!("training_steps={}", report.steps);
        println!("training_epochs={training_epochs}");
        println!("samples_per_epoch={}", cache.len());
        println!(
            "training_dtype={}",
            if cfg!(feature = "bf16") {
                "bf16"
            } else if cfg!(feature = "f16") {
                "f16"
            } else {
                "f32"
            }
        );
        println!("output_dtype={lora_dtype}");
        println!("last_loss={:?}", report.last_loss);
        println!("lora_factor_tensors={factor_count}");
        println!("lora_path={output_lora}");
    }
    if let Some(dataset_path) = preprocess_dataset {
        let text_path = text_encoder_model
            .as_deref()
            .expect("--preprocess-dataset requires --text-encoder");
        let vae_path = vae_model
            .as_deref()
            .expect("--preprocess-dataset requires --vae");
        let tokenizer_path = resolve_tokenizer_path(
            tokenizer_path.as_deref(),
            text_path,
            &[
                "tokenizer.json",
                "qwen_tokenizer/tokenizer.json",
                "llama_tokenizer/tokenizer.json",
                "qwen3_06b/tokenizer.json",
            ],
            "Qwen",
        );
        let t5_tokenizer_path = resolve_tokenizer_path(
            t5_tokenizer_model.as_deref(),
            text_path,
            &[
                "t5_tokenizer/tokenizer.json",
                "t5_old/tokenizer.json",
                "tokenizer.json",
            ],
            "T5",
        );
        println!("qwen_tokenizer={}", tokenizer_path.display());
        println!("t5_tokenizer={}", t5_tokenizer_path.display());
        for (label, path) in [
            ("qwen_tokenizer", &tokenizer_path),
            ("t5_tokenizer", &t5_tokenizer_path),
        ] {
            let metadata = std::fs::metadata(path)
                .unwrap_or_else(|error| panic!("failed to stat {label} {}: {error}", path.display()));
            println!("{label}_bytes={}", metadata.len());
        }
        let dataset =
            afterburner::ImageCaptionDataset::from_dir_with_folder_repeats(&dataset_path, repeats)
                .expect("failed to load preprocessing dataset");
        let cache_path = if let Some(path) = output_cache {
            std::path::PathBuf::from(path)
        } else {
            let diffusion_path = base_model.as_deref().unwrap_or(diffusion_model);
            let key = afterburner::ModelCacheKey::from_paths(
                model_type,
                diffusion_path,
                text_path,
                vae_path,
            )
            .expect("model cache key requires file names");
            let path = key.path_in(&model_cache_dir);
            if path.is_file() {
                let cache = afterburner::InMemoryTrainingCache::load(&path)
                    .expect("failed to load model-keyed encoded cache");
                cache
                    .validate_anima_shapes()
                    .expect("cached model data does not match the Anima tensor contract");
                let matches_dataset = cache.len() == dataset.samples().len()
                    && cache.iter().zip(dataset.samples()).all(|(cached, sample)| {
                        cached.sample_id == sample.image_path.to_string_lossy()
                            && cached.repeat_index == sample.repeat_index
                    });
                if matches_dataset {
                    println!("cache_hit=true");
                    println!("cache_key={}", key.file_name());
                    println!("preprocessed_samples={}", cache.len());
                    println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
                    println!("cache_path={}", path.display());
                    return;
                }
                println!("cache_hit=false");
                println!("cache_reason=dataset_layout_changed");
            }
            std::fs::create_dir_all(&model_cache_dir)
                .expect("failed to create model cache directory");
            println!("cache_hit=false");
            println!("cache_key={}", key.file_name());
            path
        };
        let device = candle_device(&device_name, gpu_index);
        let preprocess_dtype = match preprocess_dtype.as_str() {
            "f32" | "float32" => candle_core::DType::F32,
            "f16" | "float16" | "fp16" => candle_core::DType::F16,
            "bf16" | "bfloat16" => candle_core::DType::BF16,
            value => {
                eprintln!("unsupported --preprocess-dtype: {value} (expected f32, f16, or bf16)");
                std::process::exit(2);
            }
        };
        println!("preprocess_dtype={preprocess_dtype:?}");
        let tokenizer = afterburner::QwenTokenizer::from_file(&tokenizer_path, 512, 151_643)
            .expect("failed to load Qwen tokenizer");
        let model_load_started = std::time::Instant::now();
        eprintln!("[preprocess] loading Qwen text encoder");
        let text_encoder = afterburner::CandleQwen3TextEncoder::from_safetensors(
            text_path,
            afterburner::Qwen3TextConfig::default(),
            preprocess_dtype,
            &device,
            tokenizer,
        )
        .expect("failed to load Qwen3 text encoder");
        eprintln!(
            "[preprocess] Qwen text encoder loaded in {:.3}s",
            model_load_started.elapsed().as_secs_f64()
        );
        let vae_load_started = std::time::Instant::now();
        eprintln!("[preprocess] loading Qwen image VAE");
        let vae = afterburner::CandleQwenImageVaeStem::from_safetensors(
            vae_path,
            preprocess_dtype,
            &device,
        )
        .expect("failed to load Qwen Image VAE");
        eprintln!(
            "[preprocess] Qwen image VAE loaded in {:.3}s",
            vae_load_started.elapsed().as_secs_f64()
        );
        let vae = afterburner::QwenImageVaePreprocessor::new(
            vae,
            afterburner::QwenImageVaeConfig::default(),
        )
        .map(|preprocessor| {
            preprocessor.with_fp16_latent_math(preprocess_dtype == candle_core::DType::F16)
        })
        .expect("invalid Qwen Image VAE configuration")
        .with_max_dimension((max_resolution > 0).then_some(max_resolution))
        .with_kohya_buckets(
            (max_resolution > 0).then_some(max_resolution),
            256,
            2048,
            64,
        )
        .with_target_dimensions(bucket_resolution);
        let t5_tokenizer = afterburner::T5Tokenizer::from_file(&t5_tokenizer_path, 512, 0)
            .expect("failed to load T5 tokenizer");
        let text_encoder = afterburner::AnimaTextEncoder::new(text_encoder, t5_tokenizer);
        let mut preprocessor = afterburner::CompositePreprocessor::new(text_encoder, vae);
        let mut logger = afterburner::StderrFileLogger::open(&log_path)
            .expect("failed to open session log file");
        let preprocess_started = std::time::Instant::now();
        eprintln!(
            "[preprocess] encoding dataset samples={}",
            dataset.samples().len()
        );
        let cache =
            afterburner::preprocess_dataset_with_logger(&dataset, &mut preprocessor, &mut logger)
                .expect("failed to preprocess dataset");
        eprintln!(
            "[preprocess] dataset encoded in {:.3}s",
            preprocess_started.elapsed().as_secs_f64()
        );
        cache
            .validate_anima_shapes()
            .expect("preprocessed cache does not match the Anima tensor contract");
        cache
            .save(&cache_path)
            .expect("failed to save encoded cache");
        println!("preprocessed_samples={}", cache.len());
        println!("cache_tensor_bytes={}", cache.total_tensor_bytes());
        println!("cache_path={}", cache_path.display());
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use burn::{backend::Vulkan, tensor::Tensor};

    #[test]
    fn test_vulkan_smoke() {
        type Backend = Vulkan;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
            &device,
            Default::default(),
        );
        let a = Tensor::<Backend, 2>::ones([32, 32], &device);
        let b = Tensor::<Backend, 2>::ones([32, 32], &device);
        let values = a
            .matmul(b)
            .into_data()
            .to_vec::<f32>()
            .expect("Vulkan matmul output must be readable");
        assert!((values[0] - 32.0).abs() < 1e-3);
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn test_vulkan_bf16_smoke() {
        type Backend = Vulkan<half::bf16>;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
            &device,
            Default::default(),
        );
        let a = Tensor::<Backend, 2>::ones([32, 32], &device);
        let b = Tensor::<Backend, 2>::ones([32, 32], &device);
        let values = a
            .matmul(b)
            .into_data()
            .to_vec::<half::bf16>()
            .expect("Vulkan BF16 matmul output must be readable");
        assert!((values[0].to_f32() - 32.0).abs() < 0.5);
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn cache_with_shape(height: usize, width: usize) -> afterburner::InMemoryTrainingCache {
        let text = afterburner::VaeTensor::new(vec![1, 512, 1024], vec![0.0; 512 * 1024]).unwrap();
        let latent = afterburner::VaeTensor::new(
            vec![1, 16, 1, height, width],
            vec![0.0; 16 * height * width],
        )
        .unwrap();
        let mut cache = afterburner::InMemoryTrainingCache::new();
        cache.push(afterburner::CachedTrainingSample::new(
            "sample", 0, text, latent,
        ));
        cache
    }

    #[test]
    fn memory_preflight_uses_post_patch_token_count() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-cli-memory-preflight-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, vec![0_u8; 1024]).unwrap();
        let cache = cache_with_shape(128, 128);
        let estimate = estimate_training_memory(&path.to_string_lossy(), &cache, 2, 256).unwrap();
        assert_eq!(estimate.token_count, 4096);
        assert_eq!(estimate.attention_chunk, 256);
        assert_eq!(estimate.dtype_bytes, 2);
        assert!(estimate.estimated_training_bytes > estimate.base_model_bytes);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn memory_preflight_grows_with_attention_chunk() {
        let path = std::env::temp_dir().join(format!(
            "afterburner-cli-memory-chunk-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, vec![0_u8; 1024]).unwrap();
        let cache = cache_with_shape(64, 64);
        let small = estimate_training_memory(&path.to_string_lossy(), &cache, 2, 64).unwrap();
        let large = estimate_training_memory(&path.to_string_lossy(), &cache, 2, 256).unwrap();
        assert!(large.estimated_training_bytes > small.estimated_training_bytes);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn explicit_tokenizer_path_takes_precedence() {
        let path = resolve_tokenizer_path(
            Some("C:\\explicit\\tokenizer.json"),
            "C:\\models\\qwen.safetensors",
            &["tokenizer.json"],
            "Qwen",
        );
        assert_eq!(path, PathBuf::from("C:\\explicit\\tokenizer.json"));
    }

    #[test]
    fn tokenizer_search_finds_model_directory_candidate() {
        let root =
            std::env::temp_dir().join(format!("afterburner-cli-tokenizer-{}", std::process::id()));
        let model_dir = root.join("models");
        std::fs::create_dir_all(&model_dir).unwrap();
        let model_path = model_dir.join("qwen.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        std::fs::write(&model_path, []).unwrap();
        std::fs::write(&tokenizer_path, []).unwrap();

        let resolved = resolve_tokenizer_path(
            None,
            model_path.to_str().unwrap(),
            &["tokenizer.json"],
            "Qwen",
        );
        assert_eq!(resolved, tokenizer_path);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn training_config_deserializes_epoch_and_runtime_options() {
        let config: TrainingConfig = serde_json::from_str(
            r#"{
                "train_cache": "cache.afbcache",
                "base_model": "anima.safetensors",
                "output_lora": "elvira.safetensors",
                "epochs": 3,
                "learning_rate": 0.0001,
                "lr_schedule": "warmup_cosine",
                "lora_rank": 8,
                "lora_alpha": 8,
                "device": "cuda",
                "gpu_index": 1,
                "memory_budget_gb": 16.0,
                "attention_chunk": 128
            }"#,
        )
        .unwrap();
        assert_eq!(config.epochs, Some(3));
        assert_eq!(config.steps, None);
        assert_eq!(config.lora_rank, Some(8));
        assert_eq!(config.lora_alpha, Some(8));
        assert_eq!(config.device.as_deref(), Some("cuda"));
        assert_eq!(config.gpu_index, Some(1));
        assert_eq!(config.attention_chunk, Some(128));
    }

    #[test]
    fn training_config_accepts_steps_alias_without_epoch() {
        let config: TrainingConfig = serde_json::from_str(r#"{"steps": 10}"#).unwrap();
        assert_eq!(config.steps, Some(10));
        assert_eq!(config.epochs, None);
    }
}
