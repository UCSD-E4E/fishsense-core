# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a dual-workspace monorepo: a **Cargo workspace** and a **uv workspace**.

```
rust/fishsense-core/      # Pure Rust library — all compute-heavy algorithms
python/fishsense_core/    # PyO3/maturin bindings + Python wrappers
pyproject.toml            # Root uv workspace + fishsense-meta package
```

### Rust module map

```
src/
  errors.rs                        # FishSenseError enum
  world_point_handler.rs           # WorldPointHandler — projects image coords to 3D via K⁻¹
  laser/calibration.rs             # calibrate_laser() — 3D laser origin + orientation
  fish/fish_segmentation.rs        # FishSegmentation — ONNX instance segmentation (FishIAL)
  fish/fish_head_tail_detector.rs  # FishHeadTailDetector — PCA + geometry head/tail; predict_keypoint_depths method
  fish/fish_length_calculator.rs   # FishLengthCalculator — 3D fish length from depth map
  fish/fish_pca.rs                 # estimate_endpoints() — PCA on fish mask
  fish/fish_geometry.rs            # perimeter extraction, polygon splitting, endpoint correction
  fish/fish_plane_fit.rs           # predict_keypoint_depths() — mask-bounded RANSAC plane fit + local-median fallback
  spatial/types.rs                 # ImageCoord, DepthMap newtypes
```

### Python package map

```
python/fishsense_core/
  src/lib.rs                       # PyO3 _native module — register submodules here
  fishsense_core/
    laser.py                       # calibrate_laser() wraps _native.laser.calibrate_laser
    image/image.py                 # Abstract Image base class
    image/raw_image.py             # Raw camera decoding (rawpy + CLAHE + auto-gamma)
    image/rectified_image.py       # cv2.undistort via CameraIntrinsics
```

The Python package exposes Rust functions through `fishsense_core._native`. New Rust functions must be registered in `python/fishsense_core/src/lib.rs` before they are callable from Python.

## Adding a new algorithm

1. Implement in `rust/fishsense-core/src/<module>/`.
2. Add the module to `rust/fishsense-core/src/lib.rs`.
3. If Python access is needed, add a PyO3 wrapper in `python/fishsense_core/src/` and register the submodule in `python/fishsense_core/src/lib.rs`.
4. Add a Python convenience wrapper in `python/fishsense_core/fishsense_core/` that imports from `_native`.

## ONNX model (fish segmentation)

`build.rs` downloads the FishIAL model from HuggingFace at compile time and embeds it with `include_bytes!`. No network access is needed at runtime. The model is a Mask R-CNN variant; score threshold = 0.3, mask threshold = 0.5.

## Build commands

**Rust**
```bash
cargo build            # build all workspace members
cargo test             # run Rust unit tests
# Lint per valid feature set — NOT --all-features. `cuda` (→ ort/load-dynamic)
# and the default (→ ort/download-binaries) are mutually-exclusive ORT linking
# modes, and `coreml`'s provider type is macOS-only; combining them drops the
# execution-provider types (E0433). CI lints these separately per platform.
cargo clippy --all-targets -- -D warnings                                  # default
cargo clippy --all-targets --no-default-features --features cuda -- -D warnings    # Linux
cargo clippy --all-targets --no-default-features --features coreml -- -D warnings  # macOS
```

**Python (uv)** — run from `python/fishsense_core/`
```bash
uv sync --group dev    # install all deps including dev extras
maturin develop        # compile and install the Rust extension into the active venv
uv run pytest          # run Python tests
uv run pytest fishsense_core/path/to/test_file.py::test_name   # run a single test
uv run pylint fishsense_core/**/*.py   # lint
```

`maturin develop` must be run before `import fishsense_core._native` will work.

## CI workflows

| File | Trigger | Purpose |
|---|---|---|
| `.github/workflows/rust.yml` | every push | clippy → build → test |
| `.github/workflows/python.yml` | every push | pylint (3.12) + pytest (3.13, 3.14) |
| `.github/workflows/maturin.yml` | every push | smoke-test wheel build on ubuntu-latest (Linux x86_64 only; output discarded) |
| `.github/workflows/release-please.yml` | push to main; manual `workflow_dispatch` | opens release PRs and, on merge, builds + uploads manylinux_2_34 wheels to the GitHub Release |

## Versioning

Versioning is automated with **release-please** and **conventional commits**:

- `fix:` → patch bump, `feat:` → minor bump, `feat!:` / `BREAKING CHANGE:` → major bump
- `chore:`, `docs:`, `refactor:`, etc. produce no version bump
- Both `rust/fishsense-core` and `python/fishsense_core` are kept on the same version via the `linked-versions` plugin in `release-please-config.json` (with `merge: false`)
- Each version bump produces **two** GitHub Releases: `fishsense-core-v<X.Y.Z>` (rust crate) and `fishsense_core-v<X.Y.Z>` (python package). Mind the hyphen vs. underscore.
- The wheel-upload job in `release-please.yml` reads the `rust/fishsense-core--tag_name` output and attaches wheels to the **hyphen-form** release (`fishsense-core-v<X.Y.Z>`); the underscore-form release stays empty.
- Wheels published: cp312 / cp313 / cp314 manylinux_2_34 x86_64, in two flavors:
  - **CPU** (`fishsense_core-<X.Y.Z>-…`) — the default build; `ort/download-binaries` statically links pyke's CPU ONNX Runtime into `_native.so`.
  - **CUDA** (`fishsense_core-<X.Y.Z>+cu12-…`) — built `--features cuda` → `ort/load-dynamic`; the cdylib links no ORT. The wheel depends on `onnxruntime-gpu==1.24.*` (Microsoft's manylinux_2_27 build, which — unlike pyke's prebuilt ORT-CUDA — loads on glibc < 2.38); `__init__.py`'s `_set_ort_dylib_path()` points `ORT_DYLIB_PATH` at the installed `onnxruntime` package's `libonnxruntime.so`. The CUDA *runtime* (`libcudart`/`libcudnn`/…) still comes from `nvidia-*-cu12` pip packages, which `_preload_nvidia_libs()` loads — install those too. The `+cu12` build also gets two CI-only manifest tweaks (the local version and the `onnxruntime-gpu` dep) `sed`'d into `Cargo.toml`/`pyproject.toml` in `release-please.yml`.
  No Windows, macOS, or aarch64 wheels. No PyPI publish — wheels live only as GitHub Release assets (and PyPI would reject the `+cu12` local version anyway).
- Manual rebuild for an existing tag: trigger `release-please.yml` via `workflow_dispatch` and pass the tag name (e.g. `fishsense-core-v2.1.1`).
