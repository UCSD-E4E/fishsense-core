use ndarray::{Array3, Ix3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

use fishsense_core::fish::fish_segmentation::FishSegmentation as FishSegmentationRust;

#[pyclass]
pub struct FishSegmentation {
    inner: FishSegmentationRust,
}

#[pymethods]
impl FishSegmentation {
    #[new]
    fn new() -> Self {
        Self {
            inner: FishSegmentationRust::new(),
        }
    }

    fn load_model(&mut self) -> PyResult<()> {
        self.inner
            .load_model()
            .map_err(|e| PyValueError::new_err(format!("load_model failed: {e}")))
    }

    fn inference<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArrayDyn<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let img_rust: Array3<u8> = img
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix3>()
            .map_err(|e| PyValueError::new_err(format!("expected a 3D (H, W, 3) u8 image: {e}")))?;

        let mask = self
            .inner
            .inference(&img_rust)
            .map_err(|e| PyValueError::new_err(format!("inference failed: {e}")))?;

        Ok(mask.into_pyarray(py))
    }

    /// Runs segmentation and returns a single-instance binary mask (0 or 255)
    /// of the largest-area fish detection.  Returns `None` if no detection
    /// passes the filters.  Use this when the downstream consumer expects
    /// one-fish-per-image.
    fn inference_single<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArrayDyn<'py, u8>,
    ) -> PyResult<Option<Bound<'py, PyArray2<u8>>>> {
        let img_rust: Array3<u8> = img
            .as_array()
            .to_owned()
            .into_dimensionality::<Ix3>()
            .map_err(|e| PyValueError::new_err(format!("expected a 3D (H, W, 3) u8 image: {e}")))?;

        let mask = self
            .inner
            .inference_single(&img_rust)
            .map_err(|e| PyValueError::new_err(format!("inference_single failed: {e}")))?;

        Ok(mask.map(|m| m.into_pyarray(py)))
    }
}
