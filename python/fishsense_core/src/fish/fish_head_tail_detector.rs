use fishsense_core::fish::fish_head_tail_detector::FishHeadTailDetector as FishHeadTailDetectorRust;
use fishsense_core::spatial::types::{DepthCoord, DepthMap};
use ndarray::{Array1, Array2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

fn to_depth_coord(arr: PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<DepthCoord> {
    let v: Array1<f32> = arr
        .as_array()
        .to_owned()
        .into_dimensionality::<Ix1>()
        .map_err(|e| PyValueError::new_err(format!("{name} must be a 1D [x, y] array: {e}")))?;
    if v.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have length 2 ([x, y]), got {}",
            v.len()
        )));
    }
    Ok(DepthCoord(v))
}

type HeadTailResult<'py> = PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)>;

#[pyclass]
pub struct FishHeadTailDetector {
    inner: FishHeadTailDetectorRust,
}

#[pymethods]
impl FishHeadTailDetector {
    #[new]
    fn new() -> Self {
        Self {
            inner: FishHeadTailDetectorRust {},
        }
    }

    fn find_head_tail_img<'py>(
        &self,
        py: Python<'py>,
        mask: PyReadonlyArrayDyn<'py, u8>,
    ) -> HeadTailResult<'py> {
        let mask_rust: Array2<u8> = mask
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(format!("expected a 2D (H, W) u8 mask: {e}")))?;

        let coords = self
            .inner
            .find_head_tail_img(&mask_rust)
            .map_err(|e| PyValueError::new_err(format!("find_head_tail_img failed: {e}")))?;

        Ok((coords.head.0.into_pyarray(py), coords.tail.0.into_pyarray(py)))
    }

    fn find_head_tail_depth<'py>(
        &self,
        py: Python<'py>,
        mask: PyReadonlyArrayDyn<'py, u8>,
        depth_map: PyReadonlyArrayDyn<'py, f32>,
    ) -> HeadTailResult<'py> {
        let mask_rust: Array2<u8> = mask
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(format!("expected a 2D (H, W) u8 mask: {e}")))?;
        let depth_rust: Array2<f32> = depth_map
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(format!("expected a 2D (H, W) f32 depth map: {e}")))?;
        let depth_map_rust = DepthMap(depth_rust);

        let coords = pollster::block_on(self.inner.find_head_tail_depth(&mask_rust, &depth_map_rust))
            .map_err(|e| PyValueError::new_err(format!("find_head_tail_depth failed: {e}")))?;

        Ok((coords.head.0.into_pyarray(py), coords.tail.0.into_pyarray(py)))
    }

    fn snap_to_depth_map<'py>(
        &self,
        py: Python<'py>,
        depth_map: PyReadonlyArrayDyn<'py, f32>,
        left_depth_coord: PyReadonlyArrayDyn<'py, f32>,
        right_depth_coord: PyReadonlyArrayDyn<'py, f32>,
    ) -> HeadTailResult<'py> {
        let depth_rust: Array2<f32> = depth_map
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(format!("expected a 2D (H, W) f32 depth map: {e}")))?;
        let depth_map_rust = DepthMap(depth_rust);
        let left = to_depth_coord(left_depth_coord, "left_depth_coord")?;
        let right = to_depth_coord(right_depth_coord, "right_depth_coord")?;

        let snapped = pollster::block_on(
            self.inner.snap_to_depth_map(&depth_map_rust, &left, &right),
        )
        .map_err(|e| PyValueError::new_err(format!("snap_to_depth_map failed: {e}")))?;

        Ok((snapped.left.0.into_pyarray(py), snapped.right.0.into_pyarray(py)))
    }
}
