mod fish_segmentation;

use pyo3::prelude::*;

use fish_segmentation::FishSegmentation;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FishSegmentation>()?;
    Ok(())
}
