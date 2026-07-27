use std::io::Write;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

const MODEL_URL: &str =
    "https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx?download=true";
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

fn download(client: &reqwest::blocking::Client, dest: &std::path::Path) -> Result<(), FetchError> {
    let mut response = client.get(MODEL_URL).send().map_err(|e| FetchError {
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

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let model_path = out_dir.join("fishial.onnx");

    if !model_path.exists() {
        eprintln!("build.rs: downloading fishial.onnx from HuggingFace …");

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
            match download(&client, &tmp_path) {
                Ok(()) => {
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

    // Re-run only if this script itself changes; the cached model in OUT_DIR
    // persists across incremental rebuilds automatically.
    println!("cargo:rerun-if-changed=build.rs");
}
