# fishsense-core

Core algorithms for the [FishSense](https://e4e.ucsd.edu/fishsense) project. The library is written in Rust and exposed to Python via PyO3/maturin.

## Features

- **Laser calibration** — estimates 3D laser origin and orientation from observed points
- **Fish segmentation** — ONNX-based instance segmentation (FishIAL / Mask R-CNN) that returns per-fish instance masks
- **Head/tail detection** — three-stage pipeline (PCA → polygon geometry → depth-map snapping) that localises head and tail in image coordinates
- **Spatial processing** — GPU-accelerated connected-components labelling for depth maps via wGPU compute shaders
- **Image utilities** — raw camera decoding, auto-gamma, CLAHE, and distortion correction

## Using from Rust

Add to `Cargo.toml`:

```toml
[dependencies]
fishsense-core = { git = "https://github.com/UCSD-E4E/fishsense-core", tag = "v0.2.0" } # x-release-please-version
```

Key types and functions:

```rust
use fishsense_core::laser::calibrate_laser;
use fishsense_core::fish::{FishSegmentation, FishHeadTailDetector};
use fishsense_core::spatial::connected_components;
```

## Using from Python

Add to `pyproject.toml` (requires the Rust toolchain to build):

```toml
[project]
dependencies = ["fishsense-core"]

[tool.uv.sources]
fishsense-core = { git = "https://github.com/UCSD-E4E/fishsense-core", tag = "v0.2.0", subdirectory = "python/fishsense_core" } # x-release-please-version
```

Python API:

```python
import numpy as np
from fishsense_core.laser import calibrate_laser

# Laser calibration
points = np.array([...], dtype=np.float32)  # shape (N, 3)
origin, orientation = calibrate_laser(points)

# Image loading
from pathlib import Path
from fishsense_core.image.raw_image import RawImage
img = RawImage(Path("photo.ARW"))
data = img.data  # HxWx3 uint8 BGR
```

Rust functions are available directly through `fishsense_core._native` (e.g. `_native.laser.calibrate_laser`). Higher-level Python wrappers live in `fishsense_core.*`.

## Development

**Prerequisites:** Rust toolchain, Python 3.13+, [uv](https://docs.astral.sh/uv/)

**macOS:** The `opencv` crate requires `libclang` at build time. Add the following to your `~/.zshrc` (or `.envrc` if you use direnv):

```bash
export LIBCLANG_PATH="$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib"
export DYLD_LIBRARY_PATH="$LIBCLANG_PATH"
```

**Ubuntu:** Install system dependencies before building:

```bash
sudo apt-get install -y clang libclang-dev libopencv-dev
```

```bash
# Install Python dependencies and build the Rust extension
cd python/fishsense_core
uv sync --group dev
maturin develop

# Run tests
uv run pytest
cargo test

# Lint
uv run pylint fishsense_core/**/*.py
cargo clippy --all-targets --all-features -- -D warnings
```

## Repository structure

```
rust/fishsense-core/src/
  errors.rs                   # FishSenseError enum
  gpu.rs                      # wGPU device/queue acquisition
  laser/calibration.rs        # Laser 3D calibration
  fish/fish_segmentation.rs   # ONNX instance segmentation
  fish/fish_head_tail_detector.rs  # Head/tail localisation pipeline
  fish/fish_pca.rs            # PCA-based endpoint estimation
  fish/fish_geometry.rs       # Polygon perimeter and endpoint correction
  spatial/connected_components.rs  # GPU connected-components (WGSL shader)
  spatial/types.rs            # DepthMap wrapper

python/fishsense_core/
  src/lib.rs                  # PyO3 module registration (_native)
  fishsense_core/
    laser.py                  # calibrate_laser Python wrapper
    image/image.py            # Abstract Image base class
    image/raw_image.py        # Raw camera image decoding
    image/rectified_image.py  # Distortion correction
```

New algorithms should be implemented in `rust/fishsense-core/src/` and registered in `python/fishsense_core/src/lib.rs` if Python access is needed.
