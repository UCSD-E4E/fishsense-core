use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};

/// Offline or air-gapped builds: point this at a local copy of the pinned
/// fishial.onnx and nothing is downloaded. The file is still checked against
/// the manifest.
const LOCAL_MODEL_ENV: &str = "FISHSENSE_FISHIAL_MODEL";
const MAX_RETRIES: u32 = 5;
// Only limit how long we wait to establish the connection, not the total
// transfer time — the ONNX model is large and a per-byte timeout would fire
// on slow links.
const CONNECT_TIMEOUT_SECS: u64 = 30;
// Backoff between attempts. Immediate retries are useless against HTTP 429
// (rate limiting) — HuggingFace returns 429 under load and needs a cooldown,
// so we wait `BASE * 2^(attempt-1)` capped at MAX, or the server's
// `Retry-After` when it sends one. MAX also caps a hostile `Retry-After` so a
// bad value can't hang the build.
const BASE_BACKOFF_SECS: u64 = 2;
const MAX_BACKOFF_SECS: u64 = 60;

/// A failed download attempt, carrying enough context to decide whether and
/// how long to wait before retrying.
struct FetchError {
    msg: String,
    /// Server-requested cooldown from a `Retry-After` header, if any.
    retry_after: Option<Duration>,
    /// Whether retrying could plausibly succeed. A 404/403 won't fix itself;
    /// a 429/5xx/transport error might.
    retryable: bool,
}

/// Seconds a failed attempt should wait before the next one.
fn backoff(attempt: u32) -> Duration {
    let secs = BASE_BACKOFF_SECS
        .saturating_mul(1u64 << (attempt - 1))
        .min(MAX_BACKOFF_SECS);
    Duration::from_secs(secs)
}

/// Parse an integer-seconds `Retry-After`, clamped to MAX_BACKOFF_SECS. The
/// HTTP-date form is not parsed (rare from HF); we fall back to backoff there.
fn parse_retry_after(resp: &reqwest::blocking::Response) -> Option<Duration> {
    let secs = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(Duration::from_secs(secs.min(MAX_BACKOFF_SECS)))
}

fn download(client: &reqwest::blocking::Client, url: &str, dest: &Path) -> Result<(), FetchError> {
    let mut response = client.get(url).send().map_err(|e| FetchError {
        msg: e.to_string(),
        retry_after: None,
        retryable: true, // transport/timeout errors are worth another try
    })?;

    let status = response.status();
    if !status.is_success() {
        // Retry only on rate-limit / request-timeout / server errors; a 4xx
        // like 404 (wrong URL) or 403 (auth) will never fix itself.
        let retryable = status.as_u16() == 429
            || status.as_u16() == 408
            || status.is_server_error();
        return Err(FetchError {
            msg: format!("HTTP status {status}"),
            retry_after: parse_retry_after(&response),
            retryable,
        });
    }

    let map = |e: std::io::Error| FetchError {
        msg: e.to_string(),
        retry_after: None,
        retryable: true,
    };
    let mut file = std::fs::File::create(dest).map_err(map)?;
    std::io::copy(&mut response, &mut file).map_err(map)?;
    file.flush().map_err(map)?;
    Ok(())
}

/// What the manifest pins for the embedded FishIAL: the default `fishial`
/// version's server artifact, and where Hugging Face serves that revision.
struct Pinned {
    url: String,
    sha256: String,
    size: u64,
}

fn pinned_fishial(manifest_path: &Path) -> Pinned {
    let text = std::fs::read_to_string(manifest_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", manifest_path.display()));
    let table: toml::Table = text.parse().expect("models.toml is not valid TOML");
    let model = table["model"]
        .as_array()
        .expect("models.toml: no [[model]]")
        .iter()
        .find(|m| {
            m["name"].as_str() == Some("fishial")
                && m.get("default").and_then(|d| d.as_bool()) == Some(true)
        })
        .expect("models.toml: no default fishial version");
    let artifact = model["artifact"]
        .as_array()
        .expect("models.toml: fishial has no artifact")
        .iter()
        .find(|a| {
            a["targets"]
                .as_array()
                .is_some_and(|t| t.iter().any(|t| t.as_str() == Some("server")))
        })
        .expect("models.toml: fishial has no server artifact");
    let hf = &model["origin"]["huggingface"];
    Pinned {
        url: format!(
            "https://huggingface.co/{}/resolve/{}/{}?download=true",
            hf["repo"].as_str().expect("fishial origin repo"),
            hf["revision"].as_str().expect("fishial origin revision"),
            artifact["filename"].as_str().expect("fishial filename"),
        ),
        sha256: artifact["sha256"]
            .as_str()
            .expect("fishial sha256")
            .to_ascii_lowercase(),
        size: artifact["size"].as_integer().expect("fishial size") as u64,
    }
}

/// `Ok` if `path` is exactly the pinned file.
fn check(path: &Path, pinned: &Pinned) -> Result<(), String> {
    let size = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size != pinned.size {
        return Err(format!("expected {} bytes, got {size}", pinned.size));
    }
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    std::io::copy(&mut file, &mut hasher).map_err(|e| e.to_string())?;
    let got: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if got != pinned.sha256 {
        return Err(format!("expected sha256 {}, got {got}", pinned.sha256));
    }
    Ok(())
}

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let pinned = pinned_fishial(&manifest_dir.join("models.toml"));

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=models.toml");
    println!("cargo:rerun-if-env-changed={LOCAL_MODEL_ENV}");

    // A local copy wins, and never touches the network.
    if let Some(local) = std::env::var_os(LOCAL_MODEL_ENV) {
        // Absolute, so the file checked here is the file embedded:
        // `include_bytes!` resolves a relative path from the source file that
        // uses it (src/fish/), not from here. A relative value resolves from
        // the crate directory, which is this script's working directory.
        let local = PathBuf::from(&local).canonicalize().unwrap_or_else(|e| {
            panic!(
                "{LOCAL_MODEL_ENV}={} (relative paths resolve from {}): {e}",
                PathBuf::from(&local).display(),
                manifest_dir.display()
            )
        });
        // Otherwise new bytes written to the same path would skip this check
        // on the next build yet still be picked up by `include_bytes!`.
        println!("cargo:rerun-if-changed={}", local.display());
        if let Err(e) = check(&local, &pinned) {
            panic!(
                "{LOCAL_MODEL_ENV}={} is not the pinned fishial.onnx: {e}",
                local.display()
            );
        }
        println!("cargo:rustc-env=FISHIAL_MODEL_PATH={}", local.display());
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let model_path = out_dir.join("fishial.onnx");

    // A copy cached by an earlier build is reused only if it is still the
    // pinned file. Before the pin, the cache held whatever HF `main` served
    // that day, and a later bump of the pin must not keep embedding old bytes.
    if model_path.exists()
        && let Err(e) = check(&model_path, &pinned)
    {
        eprintln!("build.rs: cached fishial.onnx is stale ({e}); downloading again");
        std::fs::remove_file(&model_path).expect("failed to remove stale fishial.onnx");
    }

    if !model_path.exists() {
        eprintln!("build.rs: downloading {} …", pinned.url);

        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .build()
            .expect("failed to build HTTP client");

        let tmp_path = model_path.with_extension("onnx.tmp");
        let mut last_err: Option<String> = None;
        for attempt in 1..=MAX_RETRIES {
            match download(&client, &pinned.url, &tmp_path) {
                Ok(()) => {
                    // A wrong file is not a transient failure: fail the build
                    // rather than embed it, or retry into the same bytes.
                    if let Err(e) = check(&tmp_path, &pinned) {
                        let _ = std::fs::remove_file(&tmp_path);
                        panic!("downloaded fishial.onnx is not the pinned file: {e}");
                    }
                    std::fs::rename(&tmp_path, &model_path)
                        .expect("failed to move downloaded model into place");
                    eprintln!("build.rs: model saved to {}", model_path.display());
                    last_err = None;
                    break;
                }
                Err(e) => {
                    eprintln!("build.rs: attempt {attempt}/{MAX_RETRIES} failed: {}", e.msg);
                    let _ = std::fs::remove_file(&tmp_path);
                    last_err = Some(e.msg.clone());
                    // Fail fast on errors that a retry can't fix.
                    if !e.retryable {
                        break;
                    }
                    // Wait before the next attempt (honoring Retry-After); no
                    // point sleeping after the final attempt.
                    if attempt < MAX_RETRIES {
                        let delay = e.retry_after.unwrap_or_else(|| backoff(attempt));
                        eprintln!("build.rs: retrying in {}s", delay.as_secs());
                        thread::sleep(delay);
                    }
                }
            }
        }

        if let Some(e) = last_err {
            panic!("failed to download fishial.onnx after {MAX_RETRIES} attempts: {e}");
        }
    }

    // Emit the path so the library can embed it with include_bytes!.
    println!(
        "cargo:rustc-env=FISHIAL_MODEL_PATH={}",
        model_path.display()
    );
}
