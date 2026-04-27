use fishsense_core::laser::calibrate_laser as calibrate_laser_rust;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

type CalibrateLaserResult<'py> = PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)>;

#[pyfunction]
fn calibrate_laser<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
) -> CalibrateLaserResult<'py> {
    let points_rust = points.as_array().to_owned();
    let (laser_origin, laser_orientation) = calibrate_laser_rust(&points_rust)
        .map_err(|e| PyValueError::new_err(format!("calibration failed: {e}")))?;

    Ok((
        laser_origin.into_pyarray(py),
        laser_orientation.into_pyarray(py),
    ))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calibrate_laser, m)?)?;
    Ok(())
}
