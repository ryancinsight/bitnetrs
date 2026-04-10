//! # bitnet — BitNet b1.58 Inference CLI
//!
//! Command-line interface for running BitNet b1.58 inference with CPU, GPU,
//! and NPU support.
//!
//! ## Subcommands
//!
//! - `download` — Download model weights from HuggingFace Hub.
//! - `generate` — Generate text from a prompt (non-interactive).
//! - `chat`     — Interactive multi-turn chat mode.
//!
//! ## Usage Examples
//!
//! ```text
//! # Download the 2B model weights
//! bitnet download --cache-dir ~/.cache/bitnet
//!
//! # Generate text
//! bitnet generate \
//!     --model ~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T/model.safetensors \
//!     --prompt "The capital of France is" \
//!     --max-tokens 64 \
//!     --temperature 0.7
//!
//! # Interactive chat (CPU)
//! bitnet chat \
//!     --model ~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T/model.safetensors \
//!     --device cpu \
//!     --system-prompt "You are a helpful AI assistant."
//!
//! # Interactive chat (GPU)
//! bitnet chat --model /path/to/model.safetensors --device gpu
//! ```
//!
//! ## Environment Variables
//!
//! - `HF_TOKEN`      — HuggingFace access token (required for gated repos).
//! - `RUST_LOG`      — Log level filter (e.g. `bitnet=debug`, `info`).
//! - `BITNET_CACHE`  — Default cache directory (overrides `~/.cache/bitnet`).

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand, ValueEnum};
use tracing::debug;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use bitnet_core::backend::Device;
use bitnet_inference::{ChatPipeline, InferenceEngine, SamplingConfig};
use bitnet_weights::{
    hf_hub::{download_model_from_hf, HF_BASE_URL},
    DEFAULT_CACHE_SUBDIR,
};

// ---------------------------------------------------------------------------
// Root CLI
// ---------------------------------------------------------------------------

/// BitNet b1.58 Inference Engine
///
/// Runs 1.58-bit LLM inference on CPU, GPU (Vulkan/Metal/DX12), and NPU.
///
/// Set RUST_LOG=info for progress output, RUST_LOG=debug for verbose logging.
#[derive(Debug, Parser)]
#[command(
    name = "bitnet",
    version = env!("CARGO_PKG_VERSION"),
    author = "Ryan Clanton <ryanclanton@outlook.com>",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

#[derive(Debug, Subcommand)]
enum Commands {
    /// Download model weights from HuggingFace Hub.
    ///
    /// Downloads the BitNet b1.58 2B-4T packed deployment checkpoint to a local
    /// cache directory.  Skips files that are already cached.
    ///
    /// Example:
    ///   bitnet download
    ///   bitnet download --repo microsoft/bitnet-b1.58-2B-4T --cache-dir /data/models
    Download(DownloadArgs),

    /// Generate text from a prompt (non-interactive).
    ///
    /// Encodes the prompt, runs the forward pass, and prints the generated
    /// continuation to stdout.
    ///
    /// Example:
    ///   bitnet generate --model /path/to/model.safetensors --prompt "Once upon a time"
    Generate(GenerateArgs),

    /// Interactive multi-turn chat mode.
    ///
    /// Reads user messages from stdin line-by-line and prints the model's
    /// responses to stdout.  Type 'exit' or 'quit' (or send EOF) to stop.
    ///
    /// Example:
    ///   bitnet chat --model /path/to/model.safetensors --device gpu
    Chat(ChatArgs),
}

// ---------------------------------------------------------------------------
// Shared device selector
// ---------------------------------------------------------------------------

/// Compute device to use for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeviceArg {
    /// Host CPU (Rayon-parallel).  Default.
    Cpu,
    /// First available GPU via wgpu (Vulkan / Metal / DX12).
    Gpu,
    /// NPU via DirectML (Windows) or falls back to CPU.
    Npu,
}

impl DeviceArg {
    fn into_device(self, threads: Option<usize>, gpu_id: u32) -> Device {
        match self {
            Self::Cpu => Device::Cpu { threads },
            Self::Gpu => Device::Gpu { device_id: gpu_id },
            Self::Npu => Device::Npu { device_id: 0 },
        }
    }
}

// ---------------------------------------------------------------------------
// Download subcommand
// ---------------------------------------------------------------------------

/// Arguments for `bitnet download`.
#[derive(Debug, Parser)]
struct DownloadArgs {
    /// HuggingFace repository ID to download from.
    ///
    /// Defaults to the official BitNet 2B packed deployment checkpoint.
    #[arg(
        long,
        default_value = "microsoft/bitnet-b1.58-2B-4T",
        value_name = "REPO_ID",
        help = "HuggingFace repo ID (default: microsoft/bitnet-b1.58-2B-4T)"
    )]
    repo: String,

    /// Git revision (branch, tag, or commit hash) to download from.
    #[arg(
        long,
        default_value = "main",
        value_name = "REVISION",
        help = "Git revision to download (default: main)"
    )]
    revision: String,

    /// Local directory where downloaded files are cached.
    ///
    /// Files are stored at `<cache-dir>/<repo_owner>__<repo_name>/`.
    /// Defaults to `~/.cache/bitnet` (or `%USERPROFILE%\.cache\bitnet` on Windows),
    /// overridable via the `BITNET_CACHE` environment variable.
    #[arg(
        long,
        value_name = "DIR",
        help = "Local cache directory (default: ~/.cache/bitnet)"
    )]
    cache_dir: Option<PathBuf>,

    /// Additional files to download beyond the default set.
    ///
    /// The default set is: model.safetensors, config.json, tokenizer.json.
    #[arg(
        long,
        value_name = "FILE",
        num_args = 0..,
        help = "Extra filenames to download (in addition to defaults)"
    )]
    extra_files: Vec<String>,
}

// ---------------------------------------------------------------------------
// Generate subcommand
// ---------------------------------------------------------------------------

/// Arguments for `bitnet generate`.
#[derive(Debug, Parser)]
struct GenerateArgs {
    /// Path to the BF16 `.safetensors` model checkpoint.
    #[arg(
        short,
        long,
        value_name = "PATH",
        help = "Path to model.safetensors (packed or BF16 HuggingFace checkpoint)"
    )]
    model: PathBuf,

    /// Text prompt to continue from.
    #[arg(
        short,
        long,
        value_name = "TEXT",
        help = "Input prompt to generate a continuation for"
    )]
    prompt: String,

    /// Compute device to use for inference.
    #[arg(
        short,
        long,
        value_enum,
        default_value = "cpu",
        value_name = "DEVICE",
        help = "Compute device: cpu | gpu | npu"
    )]
    device: DeviceArg,

    /// Number of CPU threads (CPU device only, 0 = all cores).
    #[arg(
        long,
        default_value = "0",
        value_name = "N",
        help = "Number of CPU threads (0 = all available)"
    )]
    threads: usize,

    /// GPU adapter index (GPU device only).
    #[arg(
        long,
        default_value = "0",
        value_name = "ID",
        help = "GPU adapter index (0 = first available)"
    )]
    gpu_id: u32,

    /// Maximum number of new tokens to generate.
    #[arg(
        short = 'n',
        long,
        default_value = "256",
        value_name = "N",
        help = "Maximum number of new tokens to generate"
    )]
    max_tokens: usize,

    /// Sampling temperature τ > 0.
    ///
    /// Lower = more deterministic. `1.0` = unchanged. `0.01` ≈ greedy.
    #[arg(
        long,
        default_value = "0.7",
        value_name = "TEMP",
        help = "Sampling temperature (0 < τ ≤ 2, default 0.7)"
    )]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    ///
    /// Keep the smallest token set whose cumulative probability ≥ p.
    /// `1.0` disables nucleus filtering.
    #[arg(
        long,
        default_value = "0.9",
        value_name = "P",
        help = "Nucleus sampling threshold (0 < p ≤ 1, default 0.9)"
    )]
    top_p: f32,

    /// Top-k vocabulary truncation.  `0` disables top-k.
    #[arg(
        long,
        default_value = "50",
        value_name = "K",
        help = "Top-k vocabulary size (0 = disabled, default 50)"
    )]
    top_k: usize,

    /// Repetition penalty α ≥ 1.0.  `1.0` disables the penalty.
    #[arg(
        long,
        default_value = "1.1",
        value_name = "ALPHA",
        help = "Repetition penalty ≥ 1.0 (default 1.1)"
    )]
    repetition_penalty: f32,

    /// PRNG seed for reproducible sampling.
    #[arg(
        long,
        default_value = "42",
        value_name = "SEED",
        help = "Random seed for reproducible sampling"
    )]
    seed: u64,
}

// ---------------------------------------------------------------------------
// Chat subcommand
// ---------------------------------------------------------------------------

/// Arguments for `bitnet chat`.
#[derive(Debug, Parser)]
struct ChatArgs {
    /// Path to the BF16 `.safetensors` model checkpoint.
    #[arg(
        short,
        long,
        value_name = "PATH",
        help = "Path to model.safetensors (packed or BF16 HuggingFace checkpoint)"
    )]
    model: PathBuf,

    /// Compute device to use for inference.
    #[arg(
        short,
        long,
        value_enum,
        default_value = "cpu",
        value_name = "DEVICE",
        help = "Compute device: cpu | gpu | npu"
    )]
    device: DeviceArg,

    /// Number of CPU threads (CPU device only, 0 = all cores).
    #[arg(
        long,
        default_value = "0",
        value_name = "N",
        help = "Number of CPU threads (0 = all available)"
    )]
    threads: usize,

    /// GPU adapter index (GPU device only).
    #[arg(
        long,
        default_value = "0",
        value_name = "ID",
        help = "GPU adapter index (0 = first available)"
    )]
    gpu_id: u32,

    /// System prompt for the assistant.
    #[arg(
        long,
        default_value = "You are a helpful, honest, and concise AI assistant.",
        value_name = "TEXT",
        help = "System prompt describing the assistant's role"
    )]
    system_prompt: String,

    /// Maximum tokens per response.
    #[arg(
        short = 'n',
        long,
        default_value = "512",
        value_name = "N",
        help = "Maximum new tokens per response"
    )]
    max_tokens: usize,

    /// Sampling temperature.
    #[arg(
        long,
        default_value = "0.7",
        value_name = "TEMP",
        help = "Sampling temperature (default 0.7)"
    )]
    temperature: f32,

    /// Top-p nucleus threshold.
    #[arg(
        long,
        default_value = "0.9",
        value_name = "P",
        help = "Top-p nucleus threshold (default 0.9)"
    )]
    top_p: f32,

    /// Top-k vocabulary size.
    #[arg(
        long,
        default_value = "50",
        value_name = "K",
        help = "Top-k vocabulary size (default 50)"
    )]
    top_k: usize,

    /// Repetition penalty.
    #[arg(
        long,
        default_value = "1.1",
        value_name = "ALPHA",
        help = "Repetition penalty ≥ 1.0 (default 1.1)"
    )]
    repetition_penalty: f32,

    /// Disable the interactive prompt prefix (useful for scripting).
    #[arg(
        long,
        default_value = "false",
        help = "Suppress 'You> ' and 'Assistant> ' prompt labels"
    )]
    no_labels: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise tracing subscriber.
    // Default log level: WARN (quiet by default).
    // Override with RUST_LOG=info or RUST_LOG=bitnet=debug.
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(io::stderr))
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")))
        .init();

    debug!("BitNet CLI starting");

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Download(args) => run_download(args).await,
        Commands::Generate(args) => run_generate(args),
        Commands::Chat(args) => run_chat(args),
    };

    if let Err(ref e) = result {
        // Print the full error chain to stderr.
        eprintln!("\nError: {e:#}");
        std::process::exit(1);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// download handler
// ---------------------------------------------------------------------------

async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let cache_dir = resolve_cache_dir(args.cache_dir.as_deref())?;

    // Default files to download.
    let mut files: Vec<&str> = vec!["model.safetensors", "config.json", "tokenizer.json"];

    // Add any extra files specified on the command line.
    let extra: Vec<String> = args.extra_files.clone();
    let extra_refs: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();
    files.extend_from_slice(&extra_refs);

    eprintln!(
        "Downloading {} file(s) from {}/{} @ {}",
        files.len(),
        HF_BASE_URL,
        args.repo,
        args.revision
    );
    eprintln!("Cache directory: {}", cache_dir.display());
    eprintln!();

    let downloaded = download_model_from_hf(&args.repo, &args.revision, &files, Some(&cache_dir))
        .await
        .with_context(|| {
            format!(
                "Failed to download model from '{}'.\n\
                 If this is a gated repo, set the HF_TOKEN environment variable.",
                args.repo
            )
        })?;

    eprintln!();
    eprintln!("Downloaded {} file(s):", downloaded.len());
    for (filename, path) in &downloaded {
        let size = std::fs::metadata(path)
            .map(|m| format_bytes(m.len()))
            .unwrap_or_else(|_| "?".to_string());
        eprintln!("  {} → {} ({})", filename, path.display(), size);
    }

    // Print the path to use with --model.
    if let Some(st_path) = downloaded.get("model.safetensors") {
        eprintln!();
        eprintln!("To run inference, use:");
        eprintln!(
            "  bitnet generate --model {} --prompt \"Your prompt here\"",
            st_path.display()
        );
        eprintln!("  bitnet chat     --model {}", st_path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// generate handler
// ---------------------------------------------------------------------------

fn run_generate(args: GenerateArgs) -> anyhow::Result<()> {
    // Validate model path.
    if !args.model.exists() {
        return Err(anyhow!(
            "Model file not found: {}\n\
             Run `bitnet download` to fetch the weights first.",
            args.model.display()
        ));
    }

    // Build sampling config.
    let sampling = SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_new_tokens: args.max_tokens,
        seed: args.seed,
    };
    sampling.validate().context("Invalid sampling parameters")?;

    // Resolve device.
    let threads = if args.threads == 0 {
        None
    } else {
        Some(args.threads)
    };
    let device = args.device.into_device(threads, args.gpu_id);

    eprintln!(
        "Loading model from {} on {} …",
        args.model.display(),
        device
    );

    let mut engine = InferenceEngine::new(&args.model, device)
        .context("Failed to initialise inference engine")?;

    eprintln!("Generating …\n");

    // Print the prompt (no trailing newline so continuation follows naturally).
    print!("{}", args.prompt);
    io::stdout().flush().ok();

    let start = std::time::Instant::now();
    let (_continuation, n_tokens) = engine
        .generate_streaming(&args.prompt, &sampling, |token_text| {
            print!("{token_text}");
            io::stdout().flush().ok();
            std::ops::ControlFlow::Continue(())
        })
        .context("Generation failed")?;
    let elapsed = start.elapsed();

    println!(); // newline after streaming output

    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        n_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!("\n[{n_tokens} tokens in {elapsed:.1?} → {tokens_per_sec:.1} tok/s]");

    Ok(())
}

// ---------------------------------------------------------------------------
// chat handler
// ---------------------------------------------------------------------------

fn run_chat(args: ChatArgs) -> anyhow::Result<()> {
    // Validate model path.
    if !args.model.exists() {
        return Err(anyhow!(
            "Model file not found: {}\n\
             Run `bitnet download` to fetch the weights first.",
            args.model.display()
        ));
    }

    // Build sampling config.
    let sampling = SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_new_tokens: args.max_tokens,
        seed: 0, // vary per turn via past_tokens length
    };
    sampling.validate().context("Invalid sampling parameters")?;

    // Resolve device.
    let threads = if args.threads == 0 {
        None
    } else {
        Some(args.threads)
    };
    let device = args.device.into_device(threads, args.gpu_id);

    eprintln!(
        "Loading model from {} on {} …",
        args.model.display(),
        device
    );

    let mut pipeline = ChatPipeline::new(&args.model, device, &args.system_prompt)
        .context("Failed to initialise chat pipeline")?;

    // Welcome message.
    eprintln!("\n╭─ BitNet b1.58 Chat ─────────────────────────────────────────╮");
    eprintln!("│  Type your message and press Enter.                          │");
    eprintln!("│  Commands:  /reset  (clear history)  |  exit / quit (stop)  │");
    eprintln!("╰──────────────────────────────────────────────────────────────╯\n");

    if !args.system_prompt.is_empty() {
        eprintln!("System: {}\n", args.system_prompt);
    }

    let stdin = io::stdin();
    let mut turn = 0usize;

    loop {
        // Print user prompt label.
        if !args.no_labels {
            print!("You> ");
            io::stdout().flush().ok();
        }

        // Read a line from stdin.
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl+D / Ctrl+Z).
                eprintln!("\n[EOF — exiting]");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }

        let input = line.trim();

        // Handle exit commands.
        if matches!(
            input.to_lowercase().as_str(),
            "exit" | "quit" | "/exit" | "/quit"
        ) {
            eprintln!("Goodbye!");
            break;
        }

        // Handle /reset command.
        if input.eq_ignore_ascii_case("/reset") {
            pipeline.reset_conversation();
            eprintln!("[Conversation reset.]\n");
            continue;
        }

        // Skip empty lines.
        if input.is_empty() {
            continue;
        }

        // Generate response.
        if !args.no_labels {
            print!("Assistant> ");
            io::stdout().flush().ok();
        }

        let start = std::time::Instant::now();
        match pipeline.chat_streaming(input, &sampling, |token_text| {
            print!("{token_text}");
            io::stdout().flush().ok();
            std::ops::ControlFlow::Continue(())
        }) {
            Ok((_response, n_tokens)) => {
                println!(); // newline after streaming output
                let elapsed = start.elapsed();
                let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
                    n_tokens as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                debug!(
                    turn,
                    n_tokens,
                    elapsed_ms = elapsed.as_millis(),
                    tokens_per_sec = format!("{tokens_per_sec:.1}"),
                    "Chat turn complete"
                );
                turn += 1;
                println!(); // blank line between turns
            }
            Err(e) => {
                eprintln!("\n[Generation error: {e:#}]");
                eprintln!("[The conversation context has been reset.]\n");
                pipeline.reset_conversation();
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Resolve the cache directory from the CLI argument, BITNET_CACHE env var,
/// or the default `~/.cache/bitnet`.
fn resolve_cache_dir(arg: Option<&std::path::Path>) -> anyhow::Result<PathBuf> {
    if let Some(dir) = arg {
        return Ok(dir.to_path_buf());
    }

    // Check BITNET_CACHE environment variable.
    if let Ok(env_dir) = std::env::var("BITNET_CACHE") {
        if !env_dir.is_empty() {
            return Ok(PathBuf::from(env_dir));
        }
    }

    // Default: $HOME/.cache/bitnet (or %USERPROFILE%\.cache\bitnet on Windows).
    let home = home_dir().ok_or_else(|| {
        anyhow!(
            "Cannot determine home directory. \
             Set the BITNET_CACHE environment variable to a cache path."
        )
    })?;

    Ok(home.join(DEFAULT_CACHE_SUBDIR))
}

/// Attempt to locate the user's home directory.
///
/// Tries `HOME` (Unix) then `USERPROFILE` (Windows).
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Format a byte count as a human-readable string (e.g. `"4.7 GiB"`).
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_gib() {
        let s = format_bytes(5 * 1024 * 1024 * 1024);
        assert!(s.contains("GiB"), "GiB format: {s}");
        assert!(s.contains("5.0"), "5 GiB: {s}");
    }

    #[test]
    fn format_bytes_mib() {
        let s = format_bytes(2 * 1024 * 1024);
        assert!(s.contains("MiB"), "MiB format: {s}");
        assert!(s.contains("2.0"), "2 MiB: {s}");
    }

    #[test]
    fn format_bytes_kib() {
        let s = format_bytes(512 * 1024);
        assert!(s.contains("KiB"), "KiB format: {s}");
    }

    #[test]
    fn format_bytes_small() {
        let s = format_bytes(512);
        assert!(s.contains('B') && !s.contains("KiB"), "bytes format: {s}");
    }

    #[test]
    fn format_bytes_zero() {
        let s = format_bytes(0);
        assert_eq!(s, "0 B");
    }

    #[test]
    fn resolve_cache_dir_with_explicit_arg() {
        let dir = PathBuf::from("/tmp/my_cache");
        let result = resolve_cache_dir(Some(&dir)).unwrap();
        assert_eq!(result, dir);
    }

    #[test]
    fn resolve_cache_dir_from_env() {
        // Temporarily set BITNET_CACHE.
        std::env::set_var("BITNET_CACHE", "/tmp/bitnet_test_cache");
        let result = resolve_cache_dir(None).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/bitnet_test_cache"));
        std::env::remove_var("BITNET_CACHE");
    }

    #[test]
    fn device_arg_cpu_into_device() {
        let d = DeviceArg::Cpu.into_device(None, 0);
        assert_eq!(d, Device::Cpu { threads: None });
    }

    #[test]
    fn device_arg_cpu_with_threads_into_device() {
        let d = DeviceArg::Cpu.into_device(Some(4), 0);
        assert_eq!(d, Device::Cpu { threads: Some(4) });
    }

    #[test]
    fn device_arg_gpu_into_device() {
        let d = DeviceArg::Gpu.into_device(None, 1);
        assert_eq!(d, Device::Gpu { device_id: 1 });
    }

    #[test]
    fn device_arg_npu_into_device() {
        let d = DeviceArg::Npu.into_device(None, 0);
        assert_eq!(d, Device::Npu { device_id: 0 });
    }

    #[test]
    fn sampling_config_greedy_is_valid() {
        SamplingConfig::greedy()
            .validate()
            .expect("greedy must be valid");
    }

    #[test]
    fn sampling_config_chat_defaults_is_valid() {
        SamplingConfig::chat_defaults()
            .validate()
            .expect("chat_defaults must be valid");
    }

    #[test]
    fn cli_parses_download_default_repo() {
        let args = Cli::try_parse_from(["bitnet", "download"]);
        assert!(args.is_ok(), "download with no args must parse");
        if let Commands::Download(d) = args.unwrap().command {
            assert_eq!(d.repo, "microsoft/bitnet-b1.58-2B-4T");
            assert_eq!(d.revision, "main");
        }
    }

    #[test]
    fn cli_parses_download_custom_repo() {
        let args = Cli::try_parse_from([
            "bitnet",
            "download",
            "--repo",
            "custom/model",
            "--revision",
            "v1.0",
            "--cache-dir",
            "/tmp/cache",
        ]);
        assert!(args.is_ok(), "download with custom args must parse");
        if let Commands::Download(d) = args.unwrap().command {
            assert_eq!(d.repo, "custom/model");
            assert_eq!(d.revision, "v1.0");
            assert_eq!(d.cache_dir, Some(PathBuf::from("/tmp/cache")));
        }
    }

    #[test]
    fn cli_parses_generate_required_args() {
        let args = Cli::try_parse_from([
            "bitnet",
            "generate",
            "--model",
            "/path/to/model.safetensors",
            "--prompt",
            "Hello world",
        ]);
        assert!(args.is_ok(), "generate must parse with required args");
        if let Commands::Generate(g) = args.unwrap().command {
            assert_eq!(g.model, PathBuf::from("/path/to/model.safetensors"));
            assert_eq!(g.prompt, "Hello world");
            assert_eq!(g.device, DeviceArg::Cpu);
            assert_eq!(g.max_tokens, 256);
        }
    }

    #[test]
    fn cli_parses_generate_all_args() {
        let args = Cli::try_parse_from([
            "bitnet",
            "generate",
            "--model",
            "/model.safetensors",
            "--prompt",
            "Test prompt",
            "--device",
            "gpu",
            "--gpu-id",
            "1",
            "--max-tokens",
            "128",
            "--temperature",
            "0.5",
            "--top-p",
            "0.8",
            "--top-k",
            "40",
            "--repetition-penalty",
            "1.2",
            "--seed",
            "99",
        ]);
        assert!(
            args.is_ok(),
            "generate with all args must parse: {:?}",
            args.err()
        );
        if let Commands::Generate(g) = args.unwrap().command {
            assert_eq!(g.device, DeviceArg::Gpu);
            assert_eq!(g.gpu_id, 1);
            assert_eq!(g.max_tokens, 128);
            assert!((g.temperature - 0.5).abs() < 1e-6);
            assert!((g.top_p - 0.8).abs() < 1e-6);
            assert_eq!(g.top_k, 40);
            assert!((g.repetition_penalty - 1.2).abs() < 1e-6);
            assert_eq!(g.seed, 99);
        }
    }

    #[test]
    fn cli_parses_chat_required_args() {
        let args = Cli::try_parse_from(["bitnet", "chat", "--model", "/path/to/model.safetensors"]);
        assert!(args.is_ok(), "chat must parse with model arg");
        if let Commands::Chat(c) = args.unwrap().command {
            assert_eq!(c.model, PathBuf::from("/path/to/model.safetensors"));
            assert_eq!(c.device, DeviceArg::Cpu);
            assert!(
                !c.system_prompt.is_empty(),
                "default system prompt must exist"
            );
        }
    }

    #[test]
    fn cli_parses_chat_npu_device() {
        let args = Cli::try_parse_from([
            "bitnet",
            "chat",
            "--model",
            "/model.safetensors",
            "--device",
            "npu",
        ]);
        assert!(args.is_ok());
        if let Commands::Chat(c) = args.unwrap().command {
            assert_eq!(c.device, DeviceArg::Npu);
        }
    }

    #[test]
    fn cli_parses_chat_custom_system_prompt() {
        let args = Cli::try_parse_from([
            "bitnet",
            "chat",
            "--model",
            "/model.safetensors",
            "--system-prompt",
            "You are a pirate.",
        ]);
        assert!(args.is_ok());
        if let Commands::Chat(c) = args.unwrap().command {
            assert_eq!(c.system_prompt, "You are a pirate.");
        }
    }

    #[test]
    fn cli_generate_missing_model_returns_parse_error() {
        let args = Cli::try_parse_from(["bitnet", "generate", "--prompt", "test"]);
        assert!(args.is_err(), "generate without --model must fail to parse");
    }

    #[test]
    fn cli_generate_missing_prompt_returns_parse_error() {
        let args = Cli::try_parse_from(["bitnet", "generate", "--model", "/model.safetensors"]);
        assert!(
            args.is_err(),
            "generate without --prompt must fail to parse"
        );
    }

    #[test]
    fn cli_chat_missing_model_returns_parse_error() {
        let args = Cli::try_parse_from(["bitnet", "chat"]);
        assert!(args.is_err(), "chat without --model must fail to parse");
    }

    #[test]
    fn cli_no_subcommand_returns_parse_error() {
        let args = Cli::try_parse_from(["bitnet"]);
        assert!(args.is_err(), "no subcommand must fail to parse");
    }

    #[test]
    fn cli_unknown_subcommand_returns_parse_error() {
        let args = Cli::try_parse_from(["bitnet", "infer"]);
        assert!(args.is_err(), "unknown subcommand must fail to parse");
    }

    #[test]
    fn cli_version_flag_succeeds() {
        // --version causes early exit with success — captured as a parse error
        // in clap's try_parse_from (exit code 0).
        let result = Cli::try_parse_from(["bitnet", "--version"]);
        // clap exits via DisplayVersion error on --version.
        // We just verify it doesn't panic.
        let _ = result;
    }

    #[test]
    fn cli_help_flag_succeeds() {
        let result = Cli::try_parse_from(["bitnet", "--help"]);
        let _ = result; // clap exits via Help error — must not panic
    }
}
