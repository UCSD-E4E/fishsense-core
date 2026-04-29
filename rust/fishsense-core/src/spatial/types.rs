use ndarray::{Array1, Array2};

/// 2D coordinate `[x, y]` in image-space (the grid of the mask passed
/// to the detector). On iOS this is the camera RGB resolution, e.g.
/// 1920×1440. `x` is a column index, `y` is a row index.
pub struct ImageCoord(pub Array1<f32>);

/// Per-pixel depth in metres. Indexed `[row, col]` = `[y, x]`.
pub struct DepthMap(pub Array2<f32>);
