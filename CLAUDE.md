# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a dual-workspace monorepo: a **Cargo workspace** and a **uv workspace**.

```
rust/fishsense-core/      # Pure Rust library — compute-heavy algorithms
python/fishsense_core/    # PyO3/maturin bindings that expose Rust to Python
pyproject.toml            # Root uv workspace + fishsense-meta package
```

The Python package exposes Rust functions through `fishsense_core._native`. For example, `python/fishsense_core/fishsense_core/laser.py` imports `_native.laser.calibrate_laser` directly from the compiled extension. New Rust functions must be registered in the PyO3 module before they are callable from Python.

## Build commands

**Rust**
```bash
cargo build            # build all workspace members
cargo test             # run Rust unit tests
cargo clippy --all-targets --all-features -- -D warnings   # lint (CI standard)
```

**Python (uv)**
```bash
uv sync --group dev    # install all deps including dev extras
uv run pytest          # run Python tests
uv run pytest tests/path/to/test_file.py::test_name   # run a single test
uv run pylint python/**/*.py   # lint
```

**Maturin (build the Rust extension for local Python use)**
```bash
cd python/fishsense_core
maturin develop        # compile and install the extension into the active venv
```

Without running `maturin develop` first, `import fishsense_core._native` will fail at runtime.

## CI workflows

| File | Trigger | Purpose |
|---|---|---|
| `.github/workflows/rust.yml` | every push | clippy → build → test |
| `.github/workflows/python.yml` | every push | pylint (3.12) + pytest (3.13, 3.14) |
| `python/fishsense_core/.github/workflows/CI.yml` | push to main/tags, PRs | maturin wheel builds for all platforms; publishes to PyPI on tag push (requires `PYPI_API_TOKEN` secret) |
| `.github/workflows/release-please.yml` | push to main | opens a version-bump PR from conventional commits |

## Versioning

Versioning is automated with **release-please** and **conventional commits**:

- `fix:` → patch bump, `feat:` → minor bump, `feat!:` / `BREAKING CHANGE:` → major bump
- `chore:`, `docs:`, `refactor:`, etc. produce no version bump
- Both `rust/fishsense-core` and `python/fishsense_core` are kept on the same version (`linked-versions: true` in `release-please-config.json`)
- Merging the release PR creates a GitHub tag, which triggers the maturin CI workflow to publish wheels to PyPI
