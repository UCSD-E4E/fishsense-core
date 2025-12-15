mod laser;

use pyo3::prelude::*;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let laser_mod = PyModule::new(py, "laser")?;
    laser::register(py, &laser_mod)?;
    m.add_submodule(&laser_mod)?;
    Ok(())
}