use fishsense_core::fish::fish_head_tail_detector::FishHeadTailDetector as FishHeadTailDetectorRust;
use fishsense_core::spatial::types::DepthMap;
use ndarray::{Array2, Ix2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

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

        let snapped = pollster::block_on(self.inner.find_head_tail_depth(&mask_rust, &depth_map_rust))
            .map_err(|e| PyValueError::new_err(format!("find_head_tail_depth failed: {e}")))?;

        Ok((snapped.left.0.into_pyarray(py), snapped.right.0.into_pyarray(py)))
    }
}
