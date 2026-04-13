use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

const MODEL_URL: &str =
    "https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx?download=true";
const MAX_RETRIES: u32 = 3;
// Only limit how long we wait to establish the connection, not the total
// transfer time — the ONNX model is large and a per-byte timeout would fire
// on slow links.
const CONNECT_TIMEOUT_SECS: u64 = 30;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let model_path = out_dir.join("fishial.onnx");

    if !model_path.exists() {
        eprintln!("build.rs: downloading fishial.onnx from HuggingFace …");

        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .build()
            .expect("failed to build HTTP client");

        let mut last_err: Option<String> = None;
        for attempt in 1..=MAX_RETRIES {
            let tmp_path = model_path.with_extension("onnx.tmp");
            let result = (|| -> Result<(), String> {
                let mut response = client
                    .get(MODEL_URL)
                    .send()
                    .map_err(|e| e.to_string())?
                    .error_for_status()
                    .map_err(|e| e.to_string())?;
                let mut file = std::fs::File::create(&tmp_path)
                    .map_err(|e| e.to_string())?;
                response.copy_to(&mut file).map_err(|e| e.to_string())?;
                file.flush().map_err(|e| e.to_string())?;
                std::fs::rename(&tmp_path, &model_path).map_err(|e| e.to_string())?;
                Ok(())
            })();

            match result {
                Ok(()) => {
                    eprintln!("build.rs: model saved to {}", model_path.display());
                    last_err = None;
                    break;
                }
                Err(e) => {
                    eprintln!("build.rs: attempt {attempt}/{MAX_RETRIES} failed: {e}");
                    let _ = std::fs::remove_file(&tmp_path);
                    last_err = Some(e);
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
