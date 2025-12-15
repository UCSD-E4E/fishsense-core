use pyo3::prelude::*;

// A Python module implemented in Rust.
#[pymodule]
mod _native {
    use pyo3::prelude::*;

    #[pymodule]
    mod laser {
        use fishsense_core::laser::calibrate_laser as calibrate_laser_rust;
        use ndarray::Array2;
        use numpy::{IntoPyArray, Ix2, PyArray1, PyReadonlyArrayDyn};
        use pyo3::{exceptions::PyValueError, prelude::*};

        #[pyfunction]
        fn calibrate_laser<'py>(py: Python<'py>, points: PyReadonlyArrayDyn<'py, f64>) ->
            PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
            let points_rust: Array2<f32> = points.as_array()
                                .map(|p| p.to_owned() as f32)
                                .into_dimensionality::<Ix2>()
                                .map_err(|e| PyValueError::new_err(format!("expected a 2D array: {e}")))?;
            let (laser_origin, laser_orientation) = calibrate_laser_rust(&points_rust)
                .map_err(|e| PyValueError::new_err(format!("calibration failed: {e}")))?;

            let laser_origin_py = laser_origin
                    .map(|p| p.to_owned() as f64)
                    .into_pyarray(py);
            let laser_orientation_py = laser_orientation
                    .map(|p| p.to_owned() as f64)
                    .into_pyarray(py);

            Ok((laser_origin_py, laser_orientation_py))
        }
    }
}
