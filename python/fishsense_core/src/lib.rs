mod fish;
mod laser;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route Rust `tracing` events into Python's `logging` module:
    //   tracing  →  tracing_log::LogTracer  →  log records  →  pyo3_log  →  Python logging
    // Both inits are idempotent (errors are silently ignored so re-import is safe).
    let _ = tracing_log::LogTracer::init();
    let _ = pyo3_log::try_init();

    // Register submodules in `sys.modules` so `from fishsense_core._native.<sub>
    // import X` works — PyO3's add_submodule only attaches the child as an
    // attribute, which is not enough for Python's import machinery.
    let sys_modules = py.import("sys")?.getattr("modules")?.cast_into::<PyDict>()?;

    let fish_mod = PyModule::new(py, "fishsense_core._native.fish")?;
    fish::register(py, &fish_mod)?;
    m.add_submodule(&fish_mod)?;
    sys_modules.set_item("fishsense_core._native.fish", &fish_mod)?;

    let laser_mod = PyModule::new(py, "fishsense_core._native.laser")?;
    laser::register(py, &laser_mod)?;
    m.add_submodule(&laser_mod)?;
    sys_modules.set_item("fishsense_core._native.laser", &laser_mod)?;
    Ok(())
}