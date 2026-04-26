use ndarray::{Array1, Array2};

/// 2D coordinate `[x, y]` in image-space (the grid of the mask passed
/// to the detector). On iOS this is the camera RGB resolution, e.g.
/// 1920×1440. `x` is a column index, `y` is a row index.
///
/// Image-space coordinates and depth-index coordinates are NOT
/// interchangeable when the mask and depth-map resolutions differ
/// (the standard FishSense ARKit setup) — convert via
/// `(depth_dim / image_dim)` before indexing the depth map.
pub struct ImageCoord(pub Array1<f32>);

/// 2D coordinate `[x, y]` in depth-map index space (the grid of the
/// `DepthMap`). On iOS this is the LiDAR/ARKit resolution, e.g.
/// 256×192, which is typically coarser than the image grid. `x` is a
/// column index in the depth map, `y` is a row index.
///
/// To convert from `ImageCoord` to `DepthCoord`:
///   `dx = ix * depth_w / image_w`,
///   `dy = iy * depth_h / image_h`.
pub struct DepthCoord(pub Array1<f32>);

/// Per-pixel depth in metres. Indexed `[row, col]` = `[y, x]`.
pub struct DepthMap(pub Array2<f32>);
