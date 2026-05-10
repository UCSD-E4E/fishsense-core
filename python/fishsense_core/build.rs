//! Stages ORT's dynamic provider libraries (CUDA, TensorRT, …) next to the
//! python package source so maturin bundles them into the wheel.
//!
//! ORT's provider-bridge loads `libonnxruntime_providers_*.so` from the
//! directory of the consuming binary, which for a maturin extension is the
//! directory holding `_native.<abi>.so`. ort-sys's `copy-dylibs` already drops
//! the libs into `target/<profile>/`; we mirror them into the package source
//! dir so they end up alongside `_native.so` in the installed wheel.

use std::path::PathBuf;

fn main() {
    // Linux-only: link a small static library that provides glibc 2.38's
    // C23 `__isoc23_strtol`/`__isoc23_strtoll`/etc. as wrappers around the
    // pre-2.38 `strtol`/`strtoll` exports. The pyke-built ONNX Runtime
    // binaries pulled in by `ort-sys` reference the C23 versions, so
    // without these stubs the cdylib fails to dlopen on any system with
    // glibc < 2.38 — including Ubuntu 22.04 LTS, which is inside the
    // manylinux_2_34 baseline we claim to support. See isoc23_compat.c.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rerun-if-changed=src/isoc23_compat.c");
        cc::Build::new()
            .file("src/isoc23_compat.c")
            .flag_if_supported("-Wno-deprecated-declarations")
            .compile("isoc23_compat");
    }

    // This script stages files in the python package source dir as a
    // side-effect, and that state isn't visible to cargo's normal dependency
    // tracking. Force a re-run on every build by listing a path that doesn't
    // exist — cargo replays cached build-script output otherwise, and the
    // staging work would silently be skipped. The work itself is cheap
    // (a stat + symlink per provider lib).
    println!("cargo:rerun-if-changed=__force-rerun__");

    let pkg_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("fishsense_core");

    // Always sweep any previously-staged provider libs first so a build that
    // drops the cuda feature doesn't ship them in the wheel.
    sweep_staged_provider_libs(&pkg_dir);

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let provider_libs = match find_provider_libs() {
        Some(libs) => libs,
        None => return,
    };

    for src in provider_libs {
        let name = src.file_name().expect("provider lib has no filename");
        let dest = pkg_dir.join(name);

        #[cfg(unix)]
        std::os::unix::fs::symlink(&src, &dest).unwrap_or_else(|e| {
            panic!("symlink {} -> {} failed: {e}", src.display(), dest.display())
        });

        #[cfg(windows)]
        std::fs::copy(&src, &dest)
            .unwrap_or_else(|e| panic!("copy {} -> {} failed: {e}", src.display(), dest.display()));
    }
}

fn sweep_staged_provider_libs(pkg_dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(pkg_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let is_provider = (name.starts_with("libonnxruntime_providers_")
            || name.starts_with("onnxruntime_providers_"))
            && (name.ends_with(".so") || name.ends_with(".dll") || name.ends_with(".dylib"));
        if is_provider {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Locate ORT's provider dylibs under the cargo target dir, if any are present.
/// Returns `None` when this build doesn't pull a CUDA/TensorRT-enabled ORT
/// (e.g. CPU-only builds), so the build script is a no-op.
fn find_provider_libs() -> Option<Vec<PathBuf>> {
    // OUT_DIR is target/<profile>/build/<crate>-<hash>/out — climb to <profile>/.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").ok()?);
    let target_profile_dir = out_dir.ancestors().nth(3)?.to_path_buf();

    let (prefix, ext) = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => ("onnxruntime_providers_", ".dll"),
        Ok("macos") => ("libonnxruntime_providers_", ".dylib"),
        _ => ("libonnxruntime_providers_", ".so"),
    };

    let entries = std::fs::read_dir(&target_profile_dir).ok()?;
    let libs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(ext))
        })
        .collect();

    if libs.is_empty() { None } else { Some(libs) }
}
