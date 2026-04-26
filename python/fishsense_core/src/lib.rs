mod fish;
mod laser;
mod world_point;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route Rust `tracing` events into Python's `logging` module:
    //   tracing  →  tracing_log::LogTracer  →  log records  →  pyo3_log  →  Python logging
    // Both inits are idempotent (errors are silently ignored so re-import is safe).
    let _ = tracing_log::LogTracer::init();
    let _ = pyo3_log::try_init();

    // Submodules need two things to be usable from Python:
    //   1. `__name__` set to the full dotted path (so Python's import machinery
    //      treats them as real submodules, not stray objects).
    //   2. An attribute on the parent module keyed by the *short* name, so
    //      `_native.fish` / `_native.laser` work as plain attribute access.
    // `add_submodule` uses `__name__` as the attribute key, which would leave
    // the attribute named `fishsense_core._native.fish` — so we set it ourselves.
    let sys_modules = py.import("sys")?.getattr("modules")?.cast_into::<PyDict>()?;

    let fish_mod = PyModule::new(py, "fishsense_core._native.fish")?;
    fish::register(py, &fish_mod)?;
    m.setattr("fish", &fish_mod)?;
    sys_modules.set_item("fishsense_core._native.fish", &fish_mod)?;

    let laser_mod = PyModule::new(py, "fishsense_core._native.laser")?;
    laser::register(py, &laser_mod)?;
    m.setattr("laser", &laser_mod)?;
    sys_modules.set_item("fishsense_core._native.laser", &laser_mod)?;

    let world_point_mod = PyModule::new(py, "fishsense_core._native.world_point")?;
    world_point::register(py, &world_point_mod)?;
    m.setattr("world_point", &world_point_mod)?;
    sys_modules.set_item("fishsense_core._native.world_point", &world_point_mod)?;
    Ok(())
}