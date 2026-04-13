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
  gpu.rs                           # wGPU device/queue acquisition
  world_point_handler.rs           # WorldPointHandler — projects image coords to 3D via K⁻¹
  laser/calibration.rs             # calibrate_laser() — 3D laser origin + orientation
  fish/fish_segmentation.rs        # FishSegmentation — ONNX instance segmentation (FishIAL)
  fish/fish_head_tail_detector.rs  # FishHeadTailDetector — 3-stage head/tail pipeline
  fish/fish_length_calculator.rs   # FishLengthCalculator — 3D fish length from depth map
  fish/fish_pca.rs                 # estimate_endpoints() — PCA on fish mask
  fish/fish_geometry.rs            # perimeter extraction, polygon splitting, endpoint correction
  spatial/connected_components.rs  # connected_components() — GPU compute via wGPU (WGSL)
  spatial/types.rs                 # ImageCoord, DepthCoord, DepthMap newtypes
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

## GPU compute (spatial)

`spatial/connected_components.rs` uses wGPU with inline WGSL shaders. The async `connected_components(depth_map, epsilon)` function acquires a GPU device via `gpu::get_device_and_queue()`. Tests that exercise this path require a GPU.

## Build commands

**Rust**
```bash
cargo build            # build all workspace members
cargo test             # run Rust unit tests
cargo clippy --all-targets --all-features -- -D warnings   # lint (CI standard)
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
| `.github/workflows/maturin.yml` | every push | maturin wheel build check on Linux, Windows, macOS |
| `.github/workflows/release-please.yml` | push to main | opens a version-bump PR from conventional commits |

## Versioning

Versioning is automated with **release-please** and **conventional commits**:

- `fix:` → patch bump, `feat:` → minor bump, `feat!:` / `BREAKING CHANGE:` → major bump
- `chore:`, `docs:`, `refactor:`, etc. produce no version bump
- Both `rust/fishsense-core` and `python/fishsense_core` are kept on the same version (`linked-versions: true` in `release-please-config.json`)
- Merging the release PR creates a GitHub tag, which triggers the maturin CI workflow to publish wheels to PyPI
