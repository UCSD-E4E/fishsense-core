# fishsense-core

Core algorithms for the [FishSense](https://e4e.ucsd.edu/fishsense) project. The library is written in Rust and exposed to Python via PyO3/maturin.

## Using from Rust

Add to `Cargo.toml`:

```toml
[dependencies]
fishsense-core = { git = "https://github.com/UCSD-E4E/fishsense-core", tag = "v0.1.0" }
```

## Using from Python

Add to `pyproject.toml` (requires the Rust toolchain to build):

```toml
[project]
dependencies = ["fishsense-core"]

[tool.uv.sources]
fishsense-core = { git = "https://github.com/UCSD-E4E/fishsense-core", tag = "v0.1.0", subdirectory = "python/fishsense_core" }
```

## Development

**Prerequisites:** Rust toolchain, Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
# Install Python dependencies and build the Rust extension
cd python/fishsense_core
uv sync --group dev
maturin develop

# Run tests
uv run pytest

# Lint
uv run pylint fishsense_core/**/*.py
cargo clippy --all-targets --all-features -- -D warnings
```

## Repository structure

```
rust/fishsense-core/      # Pure Rust library — algorithms live here
python/fishsense_core/    # PyO3/maturin bindings (fishsense_core._native)
```

New algorithms should be implemented in `rust/fishsense-core` and exposed through the PyO3 bindings in `python/fishsense_core/src/` if Python access is needed.
