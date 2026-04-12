use std::path::PathBuf;

const MODEL_URL: &str =
    "https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx?download=true";

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let model_path = out_dir.join("fishial.onnx");

    if !model_path.exists() {
        eprintln!("build.rs: downloading fishial.onnx from HuggingFace …");
        let response = reqwest::blocking::get(MODEL_URL)
            .expect("failed to download fishial.onnx model");
        let bytes = response.bytes().expect("failed to read model response bytes");
        std::fs::write(&model_path, &bytes)
            .expect("failed to write fishial.onnx to OUT_DIR");
        eprintln!("build.rs: model saved to {}", model_path.display());
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
