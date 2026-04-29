use fishsense_core::fish::fish_head_tail_detector::FishHeadTailDetector as FishHeadTailDetectorRust;
use fishsense_core::spatial::types::{DepthMap, ImageCoord};
use ndarray::{Array1, Array2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

type HeadTailResult<'py> = PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)>;

fn to_image_coord(arr: PyReadonlyArrayDyn<'_, f32>, name: &str) -> PyResult<ImageCoord> {
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
    Ok(ImageCoord(v))
}

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

    /// Predict snout and fork depth via a mask-bounded RANSAC plane
    /// fit in camera 3-D coordinates.
    ///
    /// Args:
    ///     mask: (H, W) uint8 binary fish mask, in RGB grid space.
    ///     depth_map: (H', W') float32 depth map in metres.
    ///     k_inv: 3×3 inverse camera intrinsics, RGB grid space.
    ///     snout_xy, fork_xy: length-2 [x, y] keypoint coordinates,
    ///         in mask (RGB) space.
    ///
    /// Returns:
    ///     (snout_depth_m, fork_depth_m)
    fn predict_keypoint_depths(
        &self,
        mask: PyReadonlyArrayDyn<'_, u8>,
        depth_map: PyReadonlyArrayDyn<'_, f32>,
        k_inv: PyReadonlyArrayDyn<'_, f32>,
        snout_xy: PyReadonlyArrayDyn<'_, f32>,
        fork_xy: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<(f32, f32)> {
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
        let k_inv_rust: Array2<f32> = k_inv
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(format!("expected a 3x3 f32 k_inv: {e}")))?;
        let depth_map_rust = DepthMap(depth_rust);
        let snout = to_image_coord(snout_xy, "snout_xy")?;
        let fork = to_image_coord(fork_xy, "fork_xy")?;

        self.inner
            .predict_keypoint_depths(&depth_map_rust, &mask_rust, &k_inv_rust, &snout, &fork)
            .map_err(|e| PyValueError::new_err(format!("predict_keypoint_depths failed: {e}")))
    }
}
