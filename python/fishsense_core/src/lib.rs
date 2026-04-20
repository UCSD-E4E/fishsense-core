mod fish;
mod laser;

use pyo3::prelude::*;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route Rust `tracing` events into Python's `logging` module:
    //   tracing  →  tracing_log::LogTracer  →  log records  →  pyo3_log  →  Python logging
    // Both inits are idempotent (errors are silently ignored so re-import is safe).
    let _ = tracing_log::LogTracer::init();
    let _ = pyo3_log::try_init();

    let fish_mod = PyModule::new(py, "fish")?;
    fish::register(py, &fish_mod)?;
    m.add_submodule(&fish_mod)?;

    let laser_mod = PyModule::new(py, "laser")?;
    laser::register(py, &laser_mod)?;
    m.add_submodule(&laser_mod)?;
    Ok(())
}