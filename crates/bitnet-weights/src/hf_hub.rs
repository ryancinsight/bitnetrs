//! HuggingFace Hub model download client for BitNet b1.58 weights.
//!
//! # Overview
//!
//! Downloads model files from HuggingFace Hub using the most reliable
//! transport available on the current platform:
//!
//! - **Windows**: spawns the system `curl.exe` (ships with Windows 10/11,
//!   uses WinHTTP and the Windows certificate store — bypasses Rust TLS
//!   stack issues that produce WSAECONNRESET / OS error 10054 against
//!   HuggingFace's Cloudflare CDN).
//! - **Other platforms**: uses the [`hf_hub`] Rust crate (reqwest + rustls).
//!
//! # Authentication
//!
//! Set `HF_TOKEN` in the environment before calling.  Both the curl and
//! hf_hub paths honour this variable automatically.
//!
//! # Caching
//!
//! Files that already exist and are non-empty in `cache_dir` are returned
//! immediately without any network activity.
//!
//! # Invariants
//!
//! - The returned `HashMap<String, PathBuf>` maps each requested filename to
//!   its absolute local path in `cache_dir/<repo_owner>__<repo_name>/`.
//! - Every returned path points to a file that exists and is readable.
//! - Partial downloads are cleaned up on failure.
//! - The function is idempotent: calling it twice is safe.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing::{debug, info, instrument, warn};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Base URL for HuggingFace Hub.
pub const HF_BASE_URL: &str = "https://huggingface.co";

/// Default git revision to download from.
pub const DEFAULT_REVISION: &str = "main";

// ---------------------------------------------------------------------------
// download_model_from_hf  (public API)
// ---------------------------------------------------------------------------

/// Download a set of files from a HuggingFace Hub repository.
///
/// On **Windows** the function spawns the system `curl.exe`
/// (`C:\Windows\System32\curl.exe`) which uses WinHTTP and the Windows
/// certificate store.  This avoids the WSAECONNRESET (OS error 10054) that
/// Rust TLS stacks sometimes produce against HuggingFace's Cloudflare CDN.
///
/// On **other platforms** the function uses the [`hf_hub`] Rust crate.
///
/// Files already present in `cache_dir` are skipped.
///
/// # Arguments
///
/// - `repo_id`:   Repository in `owner/name` format, e.g.
///   `"microsoft/bitnet-b1.58-2B-4T"`.
/// - `revision`:  Git revision (`"main"` or a commit SHA).
/// - `files`:     Filenames to download, e.g. `&["model.safetensors"]`.
/// - `cache_dir`: Local directory.  Files are saved at
///   `cache_dir/<owner>__<name>/<filename>`.  Pass `None` to use
///   `~/.cache/bitnet`.
///
/// # Environment Variables
///
/// - `HF_TOKEN`: Optional access token for private / gated repositories.
///
/// # Errors
///
/// Returns an error if any file cannot be downloaded after all retries.
#[instrument(
    level = "info",
    skip(files, cache_dir),
    fields(repo = repo_id, revision = revision, n_files = files.len())
)]
pub async fn download_model_from_hf(
    repo_id: &str,
    revision: &str,
    files: &[&str],
    cache_dir: Option<&Path>,
) -> anyhow::Result<HashMap<String, PathBuf>> {
    if repo_id.is_empty() {
        return Err(anyhow!("repo_id must not be empty"));
    }
    if revision.is_empty() {
        return Err(anyhow!("revision must not be empty"));
    }
    if files.is_empty() {
        return Ok(HashMap::new());
    }

    // Resolve local cache directory.
    let repo_cache_name = repo_id.replace('/', "__");
    let base = match cache_dir {
        Some(d) => d.to_path_buf(),
        None => default_cache_dir()?,
    };
    let repo_dir = base.join(&repo_cache_name);
    tokio::fs::create_dir_all(&repo_dir)
        .await
        .with_context(|| format!("Failed to create cache directory: {}", repo_dir.display()))?;

    info!(
        repo = repo_id,
        revision = revision,
        cache_dir = %repo_dir.display(),
        n_files = files.len(),
        "Starting HuggingFace Hub download"
    );

    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    if token.is_some() {
        debug!("HF_TOKEN found — requests will be authenticated");
    } else {
        debug!("No HF_TOKEN set — downloading without authentication");
    }

    let multi = MultiProgress::new();
    let mut result = HashMap::with_capacity(files.len());

    for &filename in files {
        let dest = repo_dir.join(filename);

        // Cache hit: file exists and is non-empty.
        if let Ok(meta) = tokio::fs::metadata(&dest).await {
            if meta.len() > 0 {
                info!(
                    filename,
                    path = %dest.display(),
                    size_bytes = meta.len(),
                    "Cache hit — skipping download"
                );
                result.insert(filename.to_string(), dest);
                continue;
            }
        }

        // Ensure parent directory exists (filename may contain subdirs).
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }

        let url = format!("{HF_BASE_URL}/{repo_id}/resolve/{revision}/{filename}");
        info!(filename, url = %url, "Downloading");

        let pb = create_spinner(&multi, filename);

        let download_result = {
            #[cfg(windows)]
            {
                download_with_curl(&url, &dest, token.as_deref(), filename).await
            }
            #[cfg(not(windows))]
            {
                download_with_hf_hub(repo_id, revision, filename, &dest, token.as_deref()).await
            }
        };

        match download_result {
            Ok(()) => {
                pb.finish_with_message(format!("✓ {filename}"));
                if let Ok(meta) = tokio::fs::metadata(&dest).await {
                    info!(
                        filename,
                        path = %dest.display(),
                        size_bytes = meta.len(),
                        "Download complete"
                    );
                }
                result.insert(filename.to_string(), dest.clone());

                // ── SHA256 integrity verification ─────────────────────────────────────────
                let sha_url = format!("{HF_BASE_URL}/{repo_id}/resolve/{revision}/{filename}");
                match try_download_sha256(&sha_url, token.as_deref()).await {
                    Some(expected_hash) => {
                        let actual_hash = compute_sha256_hex(&dest)
                            .with_context(|| format!("SHA256 computation failed for {filename}"))?;
                        if actual_hash != expected_hash {
                            return Err(anyhow::anyhow!(
                                "SHA256 mismatch for '{filename}':\n  expected: {expected_hash}\n  computed: {actual_hash}\n\
                                 The downloaded file may be corrupted or tampered. \
                                 Delete the cache at '{}' and retry.",
                                dest.display()
                            ));
                        }
                        info!(filename, sha256 = %actual_hash, "SHA256 verified ✓");
                    }
                    None => {
                        warn!(
                            filename,
                            "No SHA256 sidecar (.sha256) available for this file; skipping integrity verification"
                        );
                    }
                }
            }
            Err(e) => {
                pb.abandon_with_message(format!("✗ {filename}: error"));
                // Clean up any partial file.
                let _ = tokio::fs::remove_file(&dest).await;
                return Err(e.context(format!(
                    "Failed to download '{filename}' from '{repo_id}'.\n\
                     If this is a gated repository, set HF_TOKEN in your environment.\n\
                     Windows: run  $env:HF_TOKEN = '<token>'  in PowerShell first."
                )));
            }
        }
    }

    info!(n_files = result.len(), "All downloads complete");
    Ok(result)
}

// ---------------------------------------------------------------------------
// Blocking wrapper
// ---------------------------------------------------------------------------

/// Synchronously download model files from HuggingFace Hub.
///
/// Creates a single-threaded `tokio` runtime and blocks until all files are
/// downloaded.  Equivalent to calling [`download_model_from_hf`] from a
/// non-async context.
pub fn download_model_from_hf_blocking(
    repo_id: &str,
    revision: &str,
    files: &[&str],
    cache_dir: Option<&Path>,
) -> anyhow::Result<HashMap<String, PathBuf>> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create tokio runtime for blocking download")?
        .block_on(download_model_from_hf(repo_id, revision, files, cache_dir))
}

// ---------------------------------------------------------------------------
// Windows: curl-based download
// ---------------------------------------------------------------------------

/// Download `url` to `dest` using the system `curl.exe`.
///
/// Uses `curl --location --retry 5 --retry-all-errors` with WinHTTP under
/// the hood, which avoids WSAECONNRESET (OS 10054) from Rust TLS stacks.
/// `curl.exe` ships with Windows 10 version 1803+ at
/// `C:\Windows\System32\curl.exe`.
#[cfg(windows)]
async fn download_with_curl(
    url: &str,
    dest: &Path,
    token: Option<&str>,
    filename: &str,
) -> anyhow::Result<()> {
    use tokio::process::Command;

    // Prefer the inbox Windows curl to avoid PATH-injected versions that
    // might not have WinHTTP support.
    let curl_paths = [
        r"C:\Windows\System32\curl.exe",
        "curl", // fall back to PATH
    ];

    let curl = curl_paths
        .iter()
        .find(|p| std::path::Path::new(p).exists() || **p == "curl")
        .copied()
        .unwrap_or("curl");

    let tmp = dest.with_extension("tmp");

    let mut cmd = Command::new(curl);
    cmd
        // Follow redirects (HuggingFace uses CDN redirects to Cloudflare R2).
        .arg("--location")
        // Fail with a non-zero exit code on HTTP errors (4xx / 5xx).
        .arg("--fail")
        // Show errors even with --silent.
        .arg("--show-error")
        // Write to a temporary file so we can rename atomically on success.
        .arg("--output")
        .arg(&tmp)
        // Retry on transient failures: 5 attempts with 3 s back-off.
        .arg("--retry")
        .arg("5")
        .arg("--retry-delay")
        .arg("3")
        .arg("--retry-all-errors")
        // Keep going on partial failures when resuming.
        .arg("--continue-at")
        .arg("-")
        // 60 s connect timeout.
        .arg("--connect-timeout")
        .arg("60")
        // 10 min max time for large files (~5 GiB BF16 checkpoint).
        .arg("--max-time")
        .arg("3600")
        // Use WinHTTP (native Windows HTTP stack) for TLS — avoids OpenSSL
        // certificate and TLS fingerprint issues with HF's Cloudflare CDN.
        .arg("--ssl-no-revoke"); // Skip revocation check (common corporate proxy issue)

    // Append Authorization header if a token is provided.
    if let Some(tok) = token {
        cmd.arg("--header")
            .arg(format!("Authorization: Bearer {tok}"));
    }

    cmd.arg(url);

    debug!(curl = curl, url, dest = %dest.display(), "Spawning curl");

    let output = cmd
        .output()
        .await
        .with_context(|| format!("Failed to spawn curl to download '{filename}'"))?;

    if !output.status.success() {
        let _ = tokio::fs::remove_file(&tmp).await;
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow!(
            "curl failed (exit {}) downloading '{filename}'.\n\
             stderr: {stderr}\n\
             stdout: {stdout}\n\
             URL: {url}",
            output.status
        ));
    }

    // Verify we actually got something.
    let tmp_size = tokio::fs::metadata(&tmp)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    if tmp_size == 0 {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(anyhow!(
            "curl downloaded 0 bytes for '{filename}' — the file may not exist in the repo"
        ));
    }

    // Atomic rename to final path.
    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("Failed to rename {} → {}", tmp.display(), dest.display()))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Non-Windows: hf_hub-based download
// ---------------------------------------------------------------------------

/// Download `filename` from `repo_id @ revision` using the `hf_hub` Rust crate.
///
/// `hf_hub` handles authentication, CDN redirects, and caching.  It returns
/// the path to a locally-cached copy, which we then copy to `dest` so that
/// our cache layout is consistent regardless of platform.
#[cfg(not(windows))]
async fn download_with_hf_hub(
    repo_id: &str,
    revision: &str,
    filename: &str,
    dest: &Path,
    token: Option<&str>,
) -> anyhow::Result<()> {
    use hf_hub::api::tokio::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let mut builder = ApiBuilder::new()
        .with_progress(false)
        .with_token(token.map(|t| t.to_string()));

    // Point the hub cache at the parent of dest so files end up in the right place.
    if let Some(parent) = dest.parent().and_then(|p| p.parent()) {
        builder = builder.with_cache_dir(parent.to_path_buf());
    }

    let api = builder
        .build()
        .context("Failed to build hf_hub API client")?;

    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let cached = repo
        .get(filename)
        .await
        .map_err(|e| anyhow!(e))
        .with_context(|| format!("hf_hub failed to download '{filename}' from '{repo_id}'"))?;

    // If the hub put the file somewhere other than dest, copy it.
    if cached != dest {
        tokio::fs::copy(&cached, dest)
            .await
            .with_context(|| format!("Failed to copy {} → {}", cached.display(), dest.display()))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Return the default cache directory (`~/.cache/bitnet`).
fn default_cache_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow!(
                "Cannot determine home directory.\n\
                 Set BITNET_CACHE environment variable to an explicit path."
            )
        })?;
    Ok(home.join(".cache").join("bitnet"))
}

/// Create a spinner progress bar for a file download in progress.
fn create_spinner(multi: &MultiProgress, filename: &str) -> ProgressBar {
    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(format!("Downloading {filename}…"));
    pb.enable_steady_tick(Duration::from_millis(120));
    pb
}

// ---------------------------------------------------------------------------
// SHA256 integrity helpers
// ---------------------------------------------------------------------------

/// Compute SHA-256 of a local file and return it as a lowercase hex string.
///
/// # Invariant
/// The returned string is always exactly 64 lowercase hex characters.
fn compute_sha256_hex(path: &std::path::Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open file for SHA256: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).with_context(|| "SHA256 read error")?;
        if n == 0 {
            break;
        }
        Digest::update(&mut hasher, &buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Attempt to download a `.sha256` sidecar file.
/// Returns `Some(hex_hash)` on success, `None` if unavailable.
async fn try_download_sha256(url: &str, token: Option<&str>) -> Option<String> {
    let sha256_url = format!("{url}.sha256");
    let client = reqwest::Client::builder().build().ok()?;
    let mut req = client.get(&sha256_url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.text()
        .await
        .ok()
        .map(|t| t.trim().to_lowercase().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Constant checks
    // -----------------------------------------------------------------------

    #[test]
    fn hf_base_url_is_https() {
        assert!(HF_BASE_URL.starts_with("https://"), "must be HTTPS");
    }

    #[test]
    fn default_revision_is_main() {
        assert_eq!(DEFAULT_REVISION, "main");
    }

    // -----------------------------------------------------------------------
    // Argument validation — no network required
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn empty_files_returns_empty_map() {
        let tmp = TempDir::new().unwrap();
        let result = download_model_from_hf("test/repo", "main", &[], Some(tmp.path()))
            .await
            .unwrap();
        assert!(result.is_empty(), "empty files list must return empty map");
    }

    #[tokio::test]
    async fn empty_repo_id_returns_error() {
        let tmp = TempDir::new().unwrap();
        let err = download_model_from_hf("", "main", &["f.bin"], Some(tmp.path()))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("repo_id"),
            "error must mention repo_id: {err}"
        );
    }

    #[tokio::test]
    async fn empty_revision_returns_error() {
        let tmp = TempDir::new().unwrap();
        let err = download_model_from_hf("owner/repo", "", &["f.bin"], Some(tmp.path()))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("revision"),
            "error must mention revision: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // Cache hit — no network required
    // -----------------------------------------------------------------------

    /// A file that already exists and is non-empty must be returned immediately
    /// without any network activity.
    #[tokio::test]
    async fn cached_file_is_returned_without_download() {
        let tmp = TempDir::new().unwrap();
        let repo_dir = tmp.path().join("owner__repo");
        tokio::fs::create_dir_all(&repo_dir).await.unwrap();

        let cached = repo_dir.join("config.json");
        tokio::fs::write(&cached, b"{\"model_type\":\"bitnet\"}")
            .await
            .unwrap();

        let result =
            download_model_from_hf("owner/repo", "main", &["config.json"], Some(tmp.path()))
                .await
                .unwrap();

        assert_eq!(result.len(), 1);
        let path = &result["config.json"];
        assert!(
            path.exists(),
            "returned path must exist: {}",
            path.display()
        );
        // Must be the same file we pre-created.
        assert_eq!(path, &cached);
    }

    // -----------------------------------------------------------------------
    // Blocking wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn blocking_empty_files_returns_empty_map() {
        let tmp = TempDir::new().unwrap();
        let result =
            download_model_from_hf_blocking("test/repo", "main", &[], Some(tmp.path())).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn blocking_empty_repo_id_returns_error() {
        let tmp = TempDir::new().unwrap();
        let err =
            download_model_from_hf_blocking("", "main", &["f.bin"], Some(tmp.path())).unwrap_err();
        assert!(err.to_string().contains("repo_id"));
    }

    // -----------------------------------------------------------------------
    // URL format
    // -----------------------------------------------------------------------

    #[test]
    fn hf_url_format_is_correct() {
        let url =
            format!("{HF_BASE_URL}/microsoft/bitnet-b1.58-2B-4T/resolve/main/model.safetensors");
        assert_eq!(
            url,
            "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/model.safetensors"
        );
    }

    #[test]
    fn repo_cache_name_replaces_slash_with_double_underscore() {
        let name = "microsoft/bitnet-b1.58-2B-4T".replace('/', "__");
        assert_eq!(name, "microsoft__bitnet-b1.58-2B-4T");
        assert!(!name.contains('/'));
    }

    // -----------------------------------------------------------------------
    // Windows curl availability (Windows only)
    // -----------------------------------------------------------------------

    #[cfg(windows)]
    #[test]
    fn windows_curl_exists_at_system32() {
        // Windows 10 v1803+ ships curl.exe in System32.
        let path = std::path::Path::new(r"C:\Windows\System32\curl.exe");
        assert!(
            path.exists(),
            "curl.exe must be present at C:\\Windows\\System32\\curl.exe on Windows 10+. \
             Please update Windows or install curl manually."
        );
    }

    // -----------------------------------------------------------------------
    // Network integration test (opt-in via RUN_NETWORK_TESTS=1)
    // -----------------------------------------------------------------------

    /// Downloads config.json from the real BitNet 2B repo to verify e2e.
    /// Skipped by default; set `RUN_NETWORK_TESTS=1` to enable.
    #[tokio::test]
    async fn network_download_config_json() {
        if std::env::var("RUN_NETWORK_TESTS").unwrap_or_default() != "1" {
            return;
        }
        let tmp = TempDir::new().unwrap();
        let result = download_model_from_hf(
            "microsoft/bitnet-b1.58-2B-4T",
            "main",
            &["config.json"],
            Some(tmp.path()),
        )
        .await;

        assert!(
            result.is_ok(),
            "Network download failed: {:?}",
            result.err()
        );
        let files = result.unwrap();
        let path = &files["config.json"];
        assert!(path.exists(), "config.json must exist locally");
        let size = std::fs::metadata(path).unwrap().len();
        assert!(size > 0, "config.json must not be empty");
    }

    // -----------------------------------------------------------------------
    // SHA256 integrity verification
    // -----------------------------------------------------------------------

    #[test]
    fn sha256_of_known_content_is_correct() {
        // SHA256 of the ASCII string "hello\n" is well-known.
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("test.txt");
        std::fs::write(&path, b"hello\n").unwrap();
        let hash = compute_sha256_hex(&path).unwrap();
        assert_eq!(
            hash, "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
            "SHA256 of 'hello\\n' must match well-known reference"
        );
    }

    #[test]
    fn sha256_of_empty_file_is_correct() {
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("empty.bin");
        std::fs::write(&path, b"").unwrap();
        let hash = compute_sha256_hex(&path).unwrap();
        assert_eq!(
            hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA256 of empty file must match RFC standard"
        );
    }

    #[test]
    fn sha256_of_nonexistent_file_returns_error() {
        let path = std::path::Path::new("/nonexistent/path/file.bin");
        let result = compute_sha256_hex(path);
        assert!(
            result.is_err(),
            "SHA256 of nonexistent file must return an error"
        );
    }
}
