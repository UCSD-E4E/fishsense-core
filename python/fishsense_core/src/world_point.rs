use fishsense_core::world_point_handler::WorldPointHandler as WorldPointHandlerRust;
use ndarray::{Array1, Array2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct WorldPointHandler {
    inner: WorldPointHandlerRust,
}

/// Coerce a 1-D f64 array to `Array1<f32>`, rejecting the wrong length here so the
/// caller gets a `ValueError` instead of a panic from inside ndarray — or, for
/// `image_point`, a silently truncated homogeneous pixel.
fn to_array1_f32(
    arr: PyReadonlyArrayDyn<'_, f64>,
    name: &str,
    expected_len: usize,
) -> PyResult<Array1<f32>> {
    let vec = arr
        .as_array()
        .map(|v| *v as f32)
        .into_dimensionality::<Ix1>()
        .map_err(|e| PyValueError::new_err(format!("{name} must be 1D: {e}")))?;
    if vec.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "{name} must have {expected_len} elements, got {}",
            vec.len()
        )));
    }
    Ok(vec)
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
        if k_inv.dim() != (3, 3) {
            return Err(PyValueError::new_err(format!(
                "camera_intrinsics_inverted must be 3x3, got {:?}",
                k_inv.dim()
            )));
        }
        Ok(Self {
            inner: WorldPointHandlerRust { camera_intrinsics_inverted: k_inv },
        })
    }

    fn project_image_point<'py>(
        &self,
        py: Python<'py>,
        image_point: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pt = to_array1_f32(image_point, "image_point", 2)?;
        let result = self.inner.project_image_point(&pt);
        Ok(result.mapv(|v| v as f64).into_pyarray(py))
    }

    fn compute_world_point_from_depth<'py>(
        &self,
        py: Python<'py>,
        image_point: PyReadonlyArrayDyn<'py, f64>,
        depth: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pt = to_array1_f32(image_point, "image_point", 2)?;
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
        let origin = to_array1_f32(laser_origin, "laser_origin", 3)?;
        let axis = to_array1_f32(laser_axis, "laser_axis", 3)?;
        let pt = to_array1_f32(image_point, "image_point", 2)?;
        let result = self.inner.compute_world_point_from_laser(&origin, &axis, &pt);
        Ok(result.mapv(|v| v as f64).into_pyarray(py))
    }

    /// Returns `(point, residual)` — the triangulated point plus the closest-approach
    /// distance between the camera ray and the laser line. The residual is the signal
    /// that separates a real laser dot from a pixel this calibration cannot explain.
    fn compute_world_point_from_laser_with_residual<'py>(
        &self,
        py: Python<'py>,
        laser_origin: PyReadonlyArrayDyn<'py, f64>,
        laser_axis: PyReadonlyArrayDyn<'py, f64>,
        image_point: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
        let origin = to_array1_f32(laser_origin, "laser_origin", 3)?;
        let axis = to_array1_f32(laser_axis, "laser_axis", 3)?;
        let pt = to_array1_f32(image_point, "image_point", 2)?;
        let result = self
            .inner
            .compute_world_point_from_laser_with_residual(&origin, &axis, &pt);
        Ok((
            result.point.mapv(|v| v as f64).into_pyarray(py),
            result.residual as f64,
        ))
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WorldPointHandler>()?;
    Ok(())
}
