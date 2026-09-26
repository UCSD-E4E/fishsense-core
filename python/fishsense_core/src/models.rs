use pyo3::prelude::*;

/// The model manifest compiled into the Rust crate, verbatim. Python parses it
/// itself (`fishsense_core.models`), so both languages read one file.
#[pyfunction]
fn manifest_toml() -> &'static str {
    fishsense_core::models::MANIFEST_TOML
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(manifest_toml, m)?)?;
    Ok(())
}
