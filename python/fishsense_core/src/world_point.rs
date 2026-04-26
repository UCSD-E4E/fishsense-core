use fishsense_core::world_point_handler::WorldPointHandler as WorldPointHandlerRust;
use ndarray::{Array1, Array2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct WorldPointHandler {
    inner: WorldPointHandlerRust,
}

fn to_array1_f32(arr: PyReadonlyArrayDyn<'_, f64>, name: &str) -> PyResult<Array1<f32>> {
    arr.as_array()
        .map(|v| *v as f32)
        .into_dimensionality::<Ix1>()
        .map_err(|e| PyValueError::new_err(format!("{name} must be 1D: {e}")))
}

#[pymethods]
impl WorldPointHandler {
    #[new]
    fn new(camera_intrinsics_inverted: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Self> {
        let k_inv: Array2<f32> = camera_intrinsics_inverted
            .as_array()
            .map(|v| *v as f32)
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(format!("camera_intrinsics_inverted must be 2D: {e}")))?;
        Ok(Self {
            inner: WorldPointHandlerRust { camera_intrinsics_inverted: k_inv },
        })
    }

    fn project_image_point<'py>(
        &self,
        py: Python<'py>,
        image_point: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pt = to_array1_f32(image_point, "image_point")?;
        let result = self.inner.project_image_point(&pt);
        Ok(result.mapv(|v| v as f64).into_pyarray(py))
    }

    fn compute_world_point_from_depth<'py>(
        &self,
        py: Python<'py>,
        image_point: PyReadonlyArrayDyn<'py, f64>,
        depth: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pt = to_array1_f32(image_point, "image_point")?;
        let result = self.inner.compute_world_point_from_depth(&pt, depth as f32);
        Ok(result.mapv(|v| v as f64).into_pyarray(py))
    }

    fn compute_world_point_from_laser<'py>(
        &self,
        py: Python<'py>,
        laser_origin: PyReadonlyArrayDyn<'py, f64>,
        laser_axis: PyReadonlyArrayDyn<'py, f64>,
        image_point: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let origin = to_array1_f32(laser_origin, "laser_origin")?;
        let axis = to_array1_f32(laser_axis, "laser_axis")?;
        let pt = to_array1_f32(image_point, "image_point")?;
        let result = self.inner.compute_world_point_from_laser(&origin, &axis, &pt);
        Ok(result.mapv(|v| v as f64).into_pyarray(py))
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WorldPointHandler>()?;
    Ok(())
}
