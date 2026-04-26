use ndarray::Array1;
use std::ops::Index;
use tracing::{debug, instrument};

use crate::{
    errors::FishSenseError,
    fish::{
        fish_geometry::{
            classify_orientation, compute_scale, correct_head, correct_tail, extract_perimeter,
            perpendicular_bisector, polygon_from_perimeter, split_polygon,
        },
        fish_pca::estimate_endpoints,
    },
    spatial::{
        connected_components::connected_components,
        types::{DepthCoord, DepthMap, ImageCoord},
    },
};
use geo::{Closest, ClosestPoint, Distance, Euclidean, Point};
use ndarray::Array2;

impl Index<usize> for DepthCoord {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.0[i]
    }
}

/// Head/tail endpoints in image (mask) pixel space — the grid of the
/// mask passed to the detector.
pub struct HeadTailCoords {
    pub head: ImageCoord,
    pub tail: ImageCoord,
}

/// Head/tail endpoints snapped to the depth map's connected component,
/// expressed in **depth-index space** (`DepthCoord`). This is the return
/// type of the lower-level `snap_to_depth_map` entry point; callers who
/// just want endpoints to plot on the source image or feed into
/// `FishLengthCalculator` should use `find_head_tail_depth`, which
/// returns `HeadTailCoords` in mask space.
pub struct SnappedDepthMap {
    pub left: DepthCoord,
    pub right: DepthCoord,
}

pub struct FishHeadTailDetector {}

impl FishHeadTailDetector {
    /// Two-stage pipeline: PCA → geometry refinement.
    ///
    /// 1. Estimates raw endpoints with PCA from the binary `mask`.
    /// 2. Classifies and corrects head/tail using polygon geometry.
    ///
    /// Returns `HeadTailCoords { head, tail }` in image (pixel) coordinates.
    /// Call `find_head_tail_depth` if you also need depth-map snapping.
    #[instrument(skip(self, mask), fields(height = mask.dim().0, width = mask.dim().1))]
    pub fn find_head_tail_img(
        &self,
        mask: &Array2<u8>,
    ) -> Result<HeadTailCoords, FishSenseError> {
        // ── Stage 1: PCA ────────────────────────────────────────────────────
        let pca = estimate_endpoints(mask)?;
        let left = pca.left;
        let right = pca.right;
        debug!(
            left_x = left[0], left_y = left[1],
            right_x = right[0], right_y = right[1],
            "PCA endpoints estimated"
        );

        // ── Stage 2: Geometry refinement ───────────────────────────────────
        let perimeter = extract_perimeter(mask);
        if perimeter.len() < 3 {
            return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
                "fish mask perimeter has fewer than 3 points"
            )));
        }

        let classified = classify_orientation(mask, &perimeter, left, right)?;
        let head = classified.head;
        let tail = classified.tail;
        debug!(
            head_x = head[0], head_y = head[1],
            tail_x = tail[0], tail_y = tail[1],
            "head/tail classified from perimeter"
        );

        // Build the two polygon halves for correction.
        let scale = compute_scale(&perimeter, head, tail);
        let (perp_a, perp_b) = perpendicular_bisector(head, tail, scale);
        let poly = polygon_from_perimeter(&perimeter);
        let (half0, half1) = split_polygon(&poly, perp_a, perp_b);

        // Assign head/tail halves based on proximity.
        let head_pt = Point::new(head[0], head[1]);
        let d0: f64 = match half0.exterior().closest_point(&head_pt) {
            Closest::Intersection(q) | Closest::SinglePoint(q) => Euclidean::distance(head_pt, q),
            Closest::Indeterminate => f64::INFINITY,
        };
        let d1: f64 = match half1.exterior().closest_point(&head_pt) {
            Closest::Intersection(q) | Closest::SinglePoint(q) => Euclidean::distance(head_pt, q),
            Closest::Indeterminate => f64::INFINITY,
        };
        let (head_half, tail_half) = if d0 <= d1 {
            (half0, half1)
        } else {
            (half1, half0)
        };

        let head_corrected = correct_head(head, tail, &head_half, scale);
        let tail_corrected = correct_tail(head, tail, &tail_half, scale);
        debug!(
            head_x = head_corrected[0], head_y = head_corrected[1],
            tail_x = tail_corrected[0], tail_y = tail_corrected[1],
            "endpoints corrected via polygon geometry"
        );

        Ok(HeadTailCoords {
            head: ImageCoord(ndarray::array![
                head_corrected[0] as f32,
                head_corrected[1] as f32
            ]),
            tail: ImageCoord(ndarray::array![
                tail_corrected[0] as f32,
                tail_corrected[1] as f32
            ]),
        })
    }

    /// Full three-stage pipeline: PCA → geometry refinement → depth-map snap.
    ///
    /// 1. `find_head_tail_img` produces image-space endpoints on the mask grid.
    /// 2. Those coords are rescaled into the depth grid so `snap_to_depth_map`
    ///    can run nearest-neighbour against depth-grid pixels.
    /// 3. The snapped depth-grid endpoints are rescaled back into mask space.
    ///
    /// Returns `HeadTailCoords` in **mask space** — the same coordinate space
    /// as the input mask, so the result drops directly into anything that
    /// already speaks image coords (overlay rendering, `FishLengthCalculator`,
    /// labelled-keypoint comparisons). Callers that need the depth-grid
    /// indices directly (e.g. to sample depth values from `depth_map`) should
    /// drive `snap_to_depth_map` themselves; that lower-level entry point
    /// keeps a `DepthCoord` in / `DepthCoord` out contract.
    ///
    /// Note: the rescale-to-depth-grid step is required whenever
    /// `mask.dim() != depth_map.0.dim()`. Skipping it (the pre-1.4.1
    /// behaviour) collapsed both endpoints to the depth map's bottom-right
    /// corner whenever the mask was higher-resolution than the depth map —
    /// the standard iOS ARKit configuration (~1920×1440 RGB mask alongside
    /// 256×192 LiDAR depth).
    #[instrument(skip(self, mask, depth_map), fields(height = mask.dim().0, width = mask.dim().1))]
    pub async fn find_head_tail_depth(
        &self,
        mask: &Array2<u8>,
        depth_map: &DepthMap,
    ) -> Result<HeadTailCoords, FishSenseError> {
        let coords = self.find_head_tail_img(mask)?;

        // ── Stage 3: Snap to depth map ──────────────────────────────────────
        // Rescale image-space endpoints into depth-index space, snap on the
        // depth grid, then rescale the snapped points back into mask space so
        // the caller never has to think about the depth grid's resolution.
        let mask_dim = mask.dim();
        let depth_dim = depth_map.0.dim();
        let head_d = scale_image_to_depth(&coords.head, mask_dim, depth_dim);
        let tail_d = scale_image_to_depth(&coords.tail, mask_dim, depth_dim);

        let snapped = self
            .snap_to_depth_map(depth_map, &head_d, &tail_d)
            .await?;

        let head = scale_depth_to_image(&snapped.left, mask_dim, depth_dim);
        let tail = scale_depth_to_image(&snapped.right, mask_dim, depth_dim);
        debug!(
            head_x = head.0[0], head_y = head.0[1],
            tail_x = tail.0[0], tail_y = tail.0[1],
            "endpoints snapped to depth component, rescaled back to mask space"
        );
        Ok(HeadTailCoords { head, tail })
    }

    /// Snaps `left_depth_coord` and `right_depth_coord` to the nearest pixel
    /// of the connected depth component that contains the midpoint between
    /// them.
    ///
    /// Lower-level entry point. Most callers want `find_head_tail_depth`,
    /// which handles the full pipeline and returns mask-space coordinates;
    /// reach for `snap_to_depth_map` when you already have depth-grid coords
    /// or when you need the depth-grid indices directly (e.g. to sample
    /// depth values out of the same `DepthMap` afterwards).
    ///
    /// Mirrors the Python `correct_labels` function in 01_process.ipynb:
    ///   1. Compute the midpoint of the two points.
    ///   2. Run connected-components on the depth map (epsilon = 0.005).
    ///   3. Find the component label at the midpoint.
    ///   4. Snap each point to the nearest pixel in that component (L2).
    ///
    /// Both inputs and outputs are `DepthCoord`s — i.e. coordinates already
    /// expressed on the depth map's index grid (`[x, y]`, where `x` is a
    /// column and `y` is a row of the depth map).
    #[instrument(skip(self, depth_map, left_depth_coord, right_depth_coord))]
    pub async fn snap_to_depth_map(
        &self,
        depth_map: &DepthMap,
        left_depth_coord: &DepthCoord,
        right_depth_coord: &DepthCoord,
    ) -> Result<SnappedDepthMap, FishSenseError> {
        const EPSILON: f32 = 0.005;

        let labels = connected_components(depth_map, EPSILON).await?;

        // Midpoint in [x, y]; clamp to valid index range.
        let (height, width) = labels.dim();
        let mid_x = (((left_depth_coord.0[0] + right_depth_coord.0[0]) / 2.0).round() as usize)
            .min(width.saturating_sub(1));
        let mid_y = (((left_depth_coord.0[1] + right_depth_coord.0[1]) / 2.0).round() as usize)
            .min(height.saturating_sub(1));

        let target_label = labels[[mid_y, mid_x]];
        debug!(mid_x, mid_y, target_label, "midpoint component identified");

        // Collect every pixel in the same component as the midpoint.
        // Convert from (row, col) → [x, y] to match DepthCoord convention.
        let component: Vec<[f32; 2]> = labels
            .indexed_iter()
            .filter(|&(_, &label)| label == target_label)
            .map(|((row, col), _)| [col as f32, row as f32])
            .collect();

        // Find the component pixel nearest to `coord` by squared Euclidean distance.
        let nearest = |coord: &Array1<f32>| -> Array1<f32> {
            let cx = coord[0];
            let cy = coord[1];
            let best = component
                .iter()
                .min_by(|a, b| {
                    let da = (a[0] - cx).powi(2) + (a[1] - cy).powi(2);
                    let db = (b[0] - cx).powi(2) + (b[1] - cy).powi(2);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("component is non-empty");
            ndarray::array![best[0], best[1]]
        };

        Ok(SnappedDepthMap {
            left: DepthCoord(nearest(&left_depth_coord.0)),
            right: DepthCoord(nearest(&right_depth_coord.0)),
        })
    }
}

/// Rescales an image-space `[x, y]` coordinate into depth-index space.
/// `image_dim` and `depth_dim` are `(height, width)` tuples (the
/// `Array2::dim()` convention). When the two grids match this is a
/// no-op.
fn scale_image_to_depth(
    coord: &ImageCoord,
    image_dim: (usize, usize),
    depth_dim: (usize, usize),
) -> DepthCoord {
    let (image_h, image_w) = image_dim;
    let (depth_h, depth_w) = depth_dim;
    let sx = depth_w as f32 / image_w as f32;
    let sy = depth_h as f32 / image_h as f32;
    DepthCoord(ndarray::array![coord.0[0] * sx, coord.0[1] * sy])
}

/// Inverse of `scale_image_to_depth`: rescales a depth-index `[x, y]`
/// coordinate back into image (mask) space.
fn scale_depth_to_image(
    coord: &DepthCoord,
    image_dim: (usize, usize),
    depth_dim: (usize, usize),
) -> ImageCoord {
    let (image_h, image_w) = image_dim;
    let (depth_h, depth_w) = depth_dim;
    let sx = image_w as f32 / depth_w as f32;
    let sy = image_h as f32 / depth_h as f32;
    ImageCoord(ndarray::array![coord.0[0] * sx, coord.0[1] * sy])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    // ── snap_to_depth_map tests (pre-existing) ────────────────────────────

    /// Verifies the core snapping behaviour against the Python `correct_labels`
    /// implementation in 01_process.ipynb.
    #[tokio::test]
    async fn test_snap_to_depth_map_snaps_to_nearest_connected_component() {
        let detector = FishHeadTailDetector {};

        let mut depth_data = Array2::<f32>::zeros((7, 7));
        for row in 2..=4usize {
            for col in 2..=4usize {
                depth_data[[row, col]] = 0.5;
            }
        }
        let depth_map = DepthMap(depth_data);

        let left = DepthCoord(array![1.0_f32, 1.0]);
        let right = DepthCoord(array![5.0_f32, 5.0]);

        let result = detector.snap_to_depth_map(&depth_map, &left, &right).await.unwrap();

        assert_eq!(result.left[0] as i32, 2, "left x should snap to 2");
        assert_eq!(result.left[1] as i32, 2, "left y should snap to 2");
        assert_eq!(result.right[0] as i32, 4, "right x should snap to 4");
        assert_eq!(result.right[1] as i32, 4, "right y should snap to 4");
    }

    #[tokio::test]
    async fn test_snap_to_depth_map_no_change_when_already_on_component() {
        let detector = FishHeadTailDetector {};
        let depth_map = DepthMap(Array2::<f32>::from_elem((5, 5), 0.5));
        let left = DepthCoord(array![0.0_f32, 0.0]);
        let right = DepthCoord(array![4.0_f32, 4.0]);
        let result = detector.snap_to_depth_map(&depth_map, &left, &right).await.unwrap();
        assert_eq!(result.left[0] as i32, 0);
        assert_eq!(result.left[1] as i32, 0);
        assert_eq!(result.right[0] as i32, 4);
        assert_eq!(result.right[1] as i32, 4);
    }

    #[tokio::test]
    async fn test_snap_to_depth_map_coincident_annotations() {
        let detector = FishHeadTailDetector {};
        let mut depth_data = Array2::<f32>::zeros((5, 5));
        depth_data[[2, 2]] = 1.0;
        let depth_map = DepthMap(depth_data);
        let left = DepthCoord(array![2.0_f32, 2.0]);
        let right = DepthCoord(array![2.0_f32, 2.0]);
        let result = detector.snap_to_depth_map(&depth_map, &left, &right).await.unwrap();
        assert_eq!(result.left[0] as i32, 2);
        assert_eq!(result.left[1] as i32, 2);
        assert_eq!(result.right[0] as i32, 2);
        assert_eq!(result.right[1] as i32, 2);
    }

    // ── find_head_tail integration tests ─────────────────────────────────

    /// A synthetic "fish": wide horizontal bar with full-coverage depth map.
    /// After PCA + geometry + snap, head and tail should be near the two ends
    /// of the bar (col ≈ 2 and col ≈ 57). Mask and depth grids match here so
    /// mask-space and depth-index space numerically coincide.
    #[tokio::test]
    async fn test_find_head_tail_horizontal_bar() {
        let detector = FishHeadTailDetector {};

        // Build a horizontal bar mask: rows 8..12, cols 2..58 in a 20×60 image.
        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                mask[[r, c]] = 1;
            }
        }

        // Uniform depth map — everything is one component.
        let depth_map = DepthMap(Array2::<f32>::from_elem((20, 60), 1.0));

        let result = detector.find_head_tail_depth(&mask, &depth_map).await;
        assert!(result.is_ok(), "find_head_tail failed: {:?}", result.err());
        let coords = result.unwrap();

        // head and tail should be near the two extreme columns of the mask.
        let head_col = coords.head.0[0] as usize;
        let tail_col = coords.tail.0[0] as usize;

        let (min_col, max_col) = if head_col < tail_col {
            (head_col, tail_col)
        } else {
            (tail_col, head_col)
        };

        assert!(
            min_col <= 5,
            "one endpoint should be near col 2, got cols {head_col} and {tail_col}"
        );
        assert!(
            max_col >= 55,
            "other endpoint should be near col 57, got cols {head_col} and {tail_col}"
        );
    }

    /// When the corrected points already lie on the depth component they should
    /// not move significantly after snapping.
    #[tokio::test]
    async fn test_find_head_tail_points_stay_in_depth_component() {
        let detector = FishHeadTailDetector {};

        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                mask[[r, c]] = 1;
            }
        }

        // Depth map matches the mask exactly.
        let mut depth_data = Array2::<f32>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                depth_data[[r, c]] = 1.0;
            }
        }
        let depth_map = DepthMap(depth_data);

        let result = detector.find_head_tail_depth(&mask, &depth_map).await;
        assert!(result.is_ok());
        let coords = result.unwrap();

        // Returned points must be inside the mask region (rows 8..12, cols 2..58).
        let head_r = coords.head.0[1] as usize;
        let head_c = coords.head.0[0] as usize;
        let tail_r = coords.tail.0[1] as usize;
        let tail_c = coords.tail.0[0] as usize;

        assert!(
            (8..12).contains(&head_r) && (2..58).contains(&head_c),
            "head should be inside mask, got ({head_r}, {head_c})"
        );
        assert!(
            (8..12).contains(&tail_r) && (2..58).contains(&tail_c),
            "tail should be inside mask, got ({tail_r}, {tail_c})"
        );
    }

    /// Empty mask → `find_head_tail` should return `Err`.
    #[tokio::test]
    async fn test_find_head_tail_empty_mask_returns_err() {
        let detector = FishHeadTailDetector {};
        let mask = Array2::<u8>::zeros((20, 60));
        let depth_map = DepthMap(Array2::<f32>::from_elem((20, 60), 1.0));
        assert!(detector.find_head_tail_depth(&mask, &depth_map).await.is_err());
    }

    // ── find_head_tail_img unit tests ─────────────────────────────────────

    /// `find_head_tail_img` is sync and requires no depth map.
    /// Endpoints of a horizontal bar should be near the two extreme columns.
    #[test]
    fn test_find_head_tail_img_horizontal_bar() {
        let detector = FishHeadTailDetector {};

        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                mask[[r, c]] = 1;
            }
        }

        let result = detector.find_head_tail_img(&mask);
        assert!(result.is_ok(), "find_head_tail_img failed: {:?}", result.err());
        let coords = result.unwrap();

        let head_col = coords.head.0[0] as usize;
        let tail_col = coords.tail.0[0] as usize;
        let (min_col, max_col) = if head_col < tail_col {
            (head_col, tail_col)
        } else {
            (tail_col, head_col)
        };

        assert!(
            min_col <= 5,
            "one img endpoint should be near col 2, got cols {head_col} and {tail_col}"
        );
        assert!(
            max_col >= 55,
            "other img endpoint should be near col 57, got cols {head_col} and {tail_col}"
        );
    }

    /// Endpoints of a vertical bar should be near the two extreme rows.
    #[test]
    fn test_find_head_tail_img_vertical_bar() {
        let detector = FishHeadTailDetector {};

        // Vertical bar: cols 8..12, rows 2..58 in a 60×20 image.
        let mut mask = Array2::<u8>::zeros((60, 20));
        for r in 2..58 {
            for c in 8..12 {
                mask[[r, c]] = 1;
            }
        }

        let result = detector.find_head_tail_img(&mask);
        assert!(result.is_ok(), "find_head_tail_img failed: {:?}", result.err());
        let coords = result.unwrap();

        // y is index 1 in [x, y] ImageCoord.
        let head_row = coords.head.0[1] as usize;
        let tail_row = coords.tail.0[1] as usize;
        let (min_row, max_row) = if head_row < tail_row {
            (head_row, tail_row)
        } else {
            (tail_row, head_row)
        };

        assert!(
            min_row <= 5,
            "one img endpoint should be near row 2, got rows {head_row} and {tail_row}"
        );
        assert!(
            max_row >= 55,
            "other img endpoint should be near row 57, got rows {head_row} and {tail_row}"
        );
    }

    /// Returned image coordinates must lie within the mask bounding box.
    #[test]
    fn test_find_head_tail_img_coords_within_mask_bounding_box() {
        let detector = FishHeadTailDetector {};

        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                mask[[r, c]] = 1;
            }
        }

        let coords = detector.find_head_tail_img(&mask).unwrap();

        for (label, coord) in [("head", &coords.head), ("tail", &coords.tail)] {
            let x = coord.0[0] as usize;
            let y = coord.0[1] as usize;
            assert!(
                x < 60,
                "{label} x={x} is out of image width"
            );
            assert!(
                y < 20,
                "{label} y={y} is out of image height"
            );
        }
    }

    /// Empty mask → `find_head_tail_img` should return `Err` (no depth map needed).
    #[test]
    fn test_find_head_tail_img_empty_mask_returns_err() {
        let detector = FishHeadTailDetector {};
        let mask = Array2::<u8>::zeros((20, 60));
        assert!(detector.find_head_tail_img(&mask).is_err());
    }

    /// Regression: on a real fish mask the detector used to return points that
    /// drifted far from the labelled snout/fork (see
    /// `tests/fixtures/head_tail_regression/`). Assert each returned endpoint
    /// lands within `TOL` of the labelled keypoint, with the correct
    /// head/tail orientation.
    #[test]
    fn test_find_head_tail_img_matches_labeled_fixture() {
        use ndarray_npy::read_npy;
        use std::path::PathBuf;

        const TOL_PX: f32 = 25.0;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/head_tail_regression");
        let mask: Array2<u8> = read_npy(fixture_dir.join("mask.npy"))
            .expect("mask.npy should load");

        // Labelled keypoints (from coords.json, [x, y] pixel coords).
        let snout = [550.9737_f32, 708.1848];
        let fork = [1533.3898_f32, 656.1633];

        let detector = FishHeadTailDetector {};
        let coords = detector
            .find_head_tail_img(&mask)
            .expect("detector should succeed on fixture mask");

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let head = [coords.head.0[0], coords.head.0[1]];
        let tail = [coords.tail.0[0], coords.tail.0[1]];

        let head_to_snout = dist(head, snout);
        let head_to_fork = dist(head, fork);
        assert!(
            head_to_snout < head_to_fork,
            "head should be closer to the snout than the fork: \
             head={head:?} snout={snout:?} fork={fork:?}"
        );

        assert!(
            head_to_snout <= TOL_PX,
            "head {head:?} is {head_to_snout:.1} px from labelled snout {snout:?} (tol {TOL_PX})"
        );
        let tail_to_fork = dist(tail, fork);
        assert!(
            tail_to_fork <= TOL_PX,
            "tail {tail:?} is {tail_to_fork:.1} px from labelled fork {fork:?} (tol {TOL_PX})"
        );
    }

    /// Regression: on a real fish mask that faces RIGHT (snout on the right
    /// end, fork on the left) the detector used to return points drifting off
    /// the head/tail axis — one near the dorsal-fin tip, one nowhere near
    /// either labelled endpoint. Mirrors the companion fixture
    /// `tests/fixtures/head_tail_regression/` (fish facing LEFT). Asserts
    /// each returned endpoint lands within `TOL` of its labelled keypoint,
    /// with the correct head/tail orientation.
    #[test]
    fn test_find_head_tail_img_snout_right_regression() {
        use ndarray_npy::read_npy;
        use std::path::PathBuf;

        const TOL_PX: f32 = 30.0;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/head_tail_snout_right");
        let mask: Array2<u8> = read_npy(fixture_dir.join("mask.npy"))
            .expect("mask.npy should load");

        // Labelled keypoints (from coords.json, [x, y] pixel coords).
        let snout = [1524.54_f32, 667.867];
        let fork = [349.432_f32, 805.949];

        let detector = FishHeadTailDetector {};
        let coords = detector
            .find_head_tail_img(&mask)
            .expect("detector should succeed on fixture mask");

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let head = [coords.head.0[0], coords.head.0[1]];
        let tail = [coords.tail.0[0], coords.tail.0[1]];

        let head_to_snout = dist(head, snout);
        let head_to_fork = dist(head, fork);
        assert!(
            head_to_snout < head_to_fork,
            "head should be closer to the snout than the fork: \
             head={head:?} snout={snout:?} fork={fork:?}"
        );

        assert!(
            head_to_snout <= TOL_PX,
            "head {head:?} is {head_to_snout:.1} px from labelled snout {snout:?} (tol {TOL_PX})"
        );
        let tail_to_fork = dist(tail, fork);
        assert!(
            tail_to_fork <= TOL_PX,
            "tail {tail:?} is {tail_to_fork:.1} px from labelled fork {fork:?} (tol {TOL_PX})"
        );
    }

    /// Regression: on a real fish mask the classifier used to return endpoint
    /// positions close to the labelled snout/fork but with head and tail
    /// *swapped* — the point returned as `head` lands on the fork and vice
    /// versa. Fixture at `tests/fixtures/head_tail_concavity_swap/` (fish
    /// facing LEFT: snout at the left end, fork at the right). Asserts
    /// each returned endpoint is within `TOL` of its labelled keypoint.
    #[test]
    fn test_find_head_tail_img_head_tail_not_swapped() {
        use ndarray_npy::read_npy;
        use std::path::PathBuf;

        const TOL_PX: f32 = 80.0;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/head_tail_concavity_swap");
        let mask: Array2<u8> =
            read_npy(fixture_dir.join("mask.npy")).expect("mask.npy should load");

        // Labelled keypoints (from coords.json, [x, y] pixel coords).
        let snout = [298.7084_f32, 907.3973];
        let fork = [1778.1605_f32, 817.2211];

        let detector = FishHeadTailDetector {};
        let coords = detector
            .find_head_tail_img(&mask)
            .expect("detector should succeed on fixture mask");

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let head = [coords.head.0[0], coords.head.0[1]];
        let tail = [coords.tail.0[0], coords.tail.0[1]];

        let head_to_snout = dist(head, snout);
        let head_to_fork = dist(head, fork);
        assert!(
            head_to_snout < head_to_fork,
            "head/tail swapped: head={head:?} is closer to fork {fork:?} ({head_to_fork:.1} px) \
             than to snout {snout:?} ({head_to_snout:.1} px)"
        );

        let tail_to_fork = dist(tail, fork);
        assert!(
            head_to_snout <= TOL_PX,
            "head {head:?} is {head_to_snout:.1} px from labelled snout {snout:?} (tol {TOL_PX})"
        );
        assert!(
            tail_to_fork <= TOL_PX,
            "tail {tail:?} is {tail_to_fork:.1} px from labelled fork {fork:?} (tol {TOL_PX})"
        );
    }

    /// Regression over a curated subset of the 2026-04-19 bug-report fixture.
    /// These three cases were the motivating examples for switching the
    /// head/tail classifier to the peduncle + hull-area-delta cascade;
    /// they are `likely_swap` failures (PCA endpoints approximately
    /// correct, orientation flipped under the previous classifier).
    ///
    /// Assertions (in-tree, curated cases):
    /// - Orientation: the returned head is strictly closer to the
    ///   labeled snout than to the labeled fork.
    /// - Endpoint proximity: each endpoint lies within
    ///   `max(80, 0.12 * fish_length)` pixels of its label.
    ///
    /// Set `FISHSENSE_BUG_FIXTURE=<path_to_fixture_root>` to sweep the
    /// full 519-case fixture. The sweep covers two disjoint sub-sets:
    ///
    /// - **`likely_swap`**: endpoints are approximately correct, only
    ///   the head/tail *label* is potentially flipped. Assert
    ///   orientation only; ≥25 % pass rate floor.
    /// - **Fork-only `endpoints_wrong`** (filtered as
    ///   `snout_distance_px < 40 && fork_distance_px > 100` in
    ///   `index.json`): head endpoint was already correct under the
    ///   pre-PR detector; this sub-set exists to validate that
    ///   `correct_tail` now reaches the fork notch. Assert fork
    ///   endpoint within `max(80, 0.12 * fish_length)` of label, and
    ///   orientation correct. ≥45 % pass rate floor.
    ///
    /// Floors are set below the current measured pass rates (swap
    /// 35.7 %, fork 51.5 %) so the test catches regressions without
    /// flaking on minor geometry shifts. The residual swap failures
    /// are rockfish-class snout-taper cases where the peduncle min is
    /// interior to the search range; disambiguating them needs a
    /// richer signal than width minima alone and is out of scope for
    /// this PR. The other `endpoints_wrong` sub-sets (snout-only
    /// occlusion, both-wrong mask fragments / PCA-axis tilt) are
    /// also out of scope — upstream mask / PCA failures that
    /// geometry-stage fixes cannot address.
    #[test]
    fn test_find_head_tail_img_bug_report_fixture() {
        use ndarray_npy::read_npy;
        use serde_json::Value;
        use std::path::PathBuf;

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        #[derive(Debug, Clone, Copy)]
        enum Check {
            OrientationOnly,
            StrictEndpoints,
            ForkOnly,
        }

        let run_case = |case_dir: &std::path::Path, check: Check| -> Result<(), String> {
            let mask: Array2<u8> = read_npy(case_dir.join("mask.npy"))
                .map_err(|e| format!("mask.npy load: {e}"))?;
            let coords_raw = std::fs::read_to_string(case_dir.join("coords.json"))
                .map_err(|e| format!("coords.json load: {e}"))?;
            let coords: Value = serde_json::from_str(&coords_raw)
                .map_err(|e| format!("coords.json parse: {e}"))?;
            let snout = [
                coords["expected"]["snout_xy"][0].as_f64().unwrap() as f32,
                coords["expected"]["snout_xy"][1].as_f64().unwrap() as f32,
            ];
            let fork = [
                coords["expected"]["fork_xy"][0].as_f64().unwrap() as f32,
                coords["expected"]["fork_xy"][1].as_f64().unwrap() as f32,
            ];
            let fish_length = dist(snout, fork);
            let endpoint_tol = 80.0_f32.max(0.12 * fish_length);

            let detector = FishHeadTailDetector {};
            let result = detector
                .find_head_tail_img(&mask)
                .map_err(|e| format!("detector err: {e:?}"))?;

            let head = [result.head.0[0], result.head.0[1]];
            let tail = [result.tail.0[0], result.tail.0[1]];
            let head_to_snout = dist(head, snout);
            let head_to_fork = dist(head, fork);
            let tail_to_fork = dist(tail, fork);

            if head_to_snout >= head_to_fork {
                return Err(format!(
                    "orientation: head {head:?} is {head_to_snout:.0} px from snout \
                     vs {head_to_fork:.0} px from fork — should be closer to snout"
                ));
            }
            match check {
                Check::OrientationOnly => {}
                Check::StrictEndpoints => {
                    if head_to_snout > endpoint_tol {
                        return Err(format!(
                            "head {head:?} is {head_to_snout:.0} px from snout \
                             (tol {endpoint_tol:.0} = max(80, 12% of {fish_length:.0} px fish))"
                        ));
                    }
                    if tail_to_fork > endpoint_tol {
                        return Err(format!(
                            "tail {tail:?} is {tail_to_fork:.0} px from fork \
                             (tol {endpoint_tol:.0} = max(80, 12% of {fish_length:.0} px fish))"
                        ));
                    }
                }
                Check::ForkOnly => {
                    if tail_to_fork > endpoint_tol {
                        return Err(format!(
                            "tail {tail:?} is {tail_to_fork:.0} px from fork \
                             (tol {endpoint_tol:.0} = max(80, 12% of {fish_length:.0} px fish))"
                        ));
                    }
                }
            }
            Ok(())
        };

        if let Ok(root) = std::env::var("FISHSENSE_BUG_FIXTURE") {
            let root = PathBuf::from(root);
            let index_raw = std::fs::read_to_string(root.join("index.json"))
                .expect("index.json at fixture root");
            let index: Value = serde_json::from_str(&index_raw).expect("index.json parse");
            let cases = index["cases"].as_array().expect("index.json cases array");

            let swap_cases: Vec<&str> = cases
                .iter()
                .filter(|c| c["failure_mode"].as_str() == Some("likely_swap"))
                .map(|c| c["case"].as_str().unwrap())
                .collect();

            // Fork-only: endpoints_wrong where snout is fine but fork is far.
            let fork_only_cases: Vec<&str> = cases
                .iter()
                .filter(|c| {
                    c["failure_mode"].as_str() == Some("endpoints_wrong")
                        && c["snout_distance_px"].as_f64().unwrap_or(f64::MAX) < 40.0
                        && c["fork_distance_px"].as_f64().unwrap_or(0.0) > 100.0
                })
                .map(|c| c["case"].as_str().unwrap())
                .collect();

            // ── Swap sub-set: orientation only ──
            let mut swap_passes = 0usize;
            let mut swap_failures: Vec<(String, String)> = Vec::new();
            for name in &swap_cases {
                match run_case(&root.join(name), Check::OrientationOnly) {
                    Ok(()) => swap_passes += 1,
                    Err(msg) => swap_failures.push((name.to_string(), msg)),
                }
            }
            let swap_total = swap_cases.len();
            let swap_rate = swap_passes as f64 / swap_total.max(1) as f64;

            // ── Fork-only sub-set: fork endpoint proximity ──
            let mut fork_passes = 0usize;
            let mut fork_failures: Vec<(String, String)> = Vec::new();
            for name in &fork_only_cases {
                match run_case(&root.join(name), Check::ForkOnly) {
                    Ok(()) => fork_passes += 1,
                    Err(msg) => fork_failures.push((name.to_string(), msg)),
                }
            }
            let fork_total = fork_only_cases.len();
            let fork_rate = fork_passes as f64 / fork_total.max(1) as f64;

            println!(
                "[bug_report_fixture] likely_swap orientation pass rate: \
                 {}/{} ({:.1}%)",
                swap_passes,
                swap_total,
                swap_rate * 100.0
            );
            println!(
                "[bug_report_fixture] fork-only endpoints_wrong fork-endpoint pass rate: \
                 {}/{} ({:.1}%)",
                fork_passes,
                fork_total,
                fork_rate * 100.0
            );

            // Floors are well below the measured rates (swap: 5/14 = 35.7%,
            // fork: 17/33 = 51.5%) so minor further regressions don't trip
            // the test; the goal is to catch substantial regressions. The
            // residual swap failures are rockfish-class snout-taper cases
            // where the peduncle min is interior (not at the boundary) —
            // those need a richer signal than width minima alone, out of
            // scope for this PR.
            const MIN_SWAP_RATE: f64 = 0.25;
            const MIN_FORK_RATE: f64 = 0.45;
            assert!(
                swap_rate >= MIN_SWAP_RATE,
                "likely_swap orientation pass rate {:.1}% ({}/{}) below floor {:.0}%. \
                 Sample failures: {:?}",
                swap_rate * 100.0,
                swap_passes,
                swap_total,
                MIN_SWAP_RATE * 100.0,
                swap_failures.iter().take(5).collect::<Vec<_>>()
            );
            assert!(
                fork_rate >= MIN_FORK_RATE,
                "fork-only fork-endpoint pass rate {:.1}% ({}/{}) below floor {:.0}%. \
                 Sample failures: {:?}",
                fork_rate * 100.0,
                fork_passes,
                fork_total,
                MIN_FORK_RATE * 100.0,
                fork_failures.iter().take(5).collect::<Vec<_>>()
            );
        } else {
            let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/bug_report");
            for name in ["case_01", "case_150", "case_273"] {
                run_case(&base.join(name), Check::StrictEndpoints)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    }

    /// When the depth component covers only the centre of the mask, snapping
    /// must pull the endpoints inward relative to the raw image coordinates.
    /// This verifies that `find_head_tail_depth` ≠ `find_head_tail_img` when
    /// the depth coverage is limited. Mask and depth grids are the same size
    /// here so the depth-strip bounds (cols 20..40) are also the bounds in
    /// mask space.
    #[tokio::test]
    async fn test_find_head_tail_depth_snaps_endpoints_toward_depth_component() {
        let detector = FishHeadTailDetector {};

        // Full-width horizontal bar mask.
        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 0..60 {
                mask[[r, c]] = 1;
            }
        }

        // Depth map only covers the centre strip (cols 20..40).
        let mut depth_data = Array2::<f32>::zeros((20, 60));
        for r in 8..12 {
            for c in 20..40 {
                depth_data[[r, c]] = 1.0;
            }
        }
        let depth_map = DepthMap(depth_data);

        // img coords should span nearly the full width (near cols 0 and 59).
        let img = detector.find_head_tail_img(&mask).unwrap();
        let img_head_col = img.head.0[0];
        let img_tail_col = img.tail.0[0];
        let img_span = (img_head_col - img_tail_col).abs();
        assert!(
            img_span >= 50.0,
            "img endpoints should span the full bar, got cols {img_head_col} and {img_tail_col}"
        );

        // After snapping, both endpoints must land inside the centre strip
        // (mask cols 20..40, identical to depth cols here since grids match).
        let coords = detector.find_head_tail_depth(&mask, &depth_map).await.unwrap();
        let head_col = coords.head.0[0] as usize;
        let tail_col = coords.tail.0[0] as usize;
        assert!(
            (20..40).contains(&head_col),
            "head col {head_col} should be inside the strip mapped to mask cols 20..40"
        );
        assert!(
            (20..40).contains(&tail_col),
            "tail col {tail_col} should be inside the strip mapped to mask cols 20..40"
        );
    }

    // ── Regression tests for mismatched mask/depth grids (iOS standard) ──
    //
    // Pre-1.4.1, when the mask grid was finer than the depth grid (e.g.
    // 1920×1440 RGB mask alongside 256×192 ARKit depth), `snap_to_depth_map`
    // clamped the mask-space midpoint into the depth grid's bottom-right
    // corner and the nearest-neighbour search collapsed both endpoints onto
    // a single corner pixel. Pre-1.5.0, the public return type leaked the
    // depth grid's index space onto the caller — overlay points clustered
    // in the upper-left of landscape RGB frames until the iOS bridge added
    // a manual rescale. These tests use a 2× downsample (mask 60×40 /
    // depth 30×20) so a CI runner without a GPU can exercise the CPU
    // `connected_components` fallback quickly, AND they assert that the
    // public return is in mask space (range checks against mask dims, not
    // depth dims).
    //
    /// Mask resolution > depth resolution, depth is uniform (single
    /// component). Endpoints must remain distinct (not collapsed) and lie
    /// inside the mask bar in mask-space coordinates.
    #[tokio::test]
    async fn test_find_head_tail_depth_higher_res_mask_full_coverage() {
        let detector = FishHeadTailDetector {};

        // Mask: 40h × 60w with a horizontal bar (rows 16..24, cols 5..55).
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 16..24 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }

        // Depth: 20h × 30w (2× downsample), uniform → one component.
        let depth_map = DepthMap(Array2::<f32>::from_elem((20, 30), 1.0));

        let coords = detector
            .find_head_tail_depth(&mask, &depth_map)
            .await
            .expect("detector should succeed on mismatched grids");

        let head = (coords.head.0[0], coords.head.0[1]);
        let tail = (coords.tail.0[0], coords.tail.0[1]);

        // Pre-1.5.0, this returned depth-index coords (max ~29 × ~19) — a
        // mask-space range check would have failed. After the v1.5.0 fix,
        // the result is mask-space and must fit inside the mask dims.
        assert!(
            (0.0..60.0).contains(&head.0) && (0.0..40.0).contains(&head.1),
            "head {head:?} must be in mask space (60w × 40h), not depth-index space"
        );
        assert!(
            (0.0..60.0).contains(&tail.0) && (0.0..40.0).contains(&tail.1),
            "tail {tail:?} must be in mask space (60w × 40h), not depth-index space"
        );

        // Pre-1.4.1 the snap stage collapsed both endpoints onto a single
        // corner; assert they remain distinct and span the bar.
        assert_ne!(head, tail, "endpoints collapsed to a single point: {head:?}");
        assert!(
            (head.0 - tail.0).abs() >= 20.0,
            "endpoints should span the bar in mask-x, got head={head:?}, tail={tail:?}"
        );

        // Each endpoint should sit inside the mask bar — small slack at the
        // ends is fine because PCA endpoints sit just inside the bar, and
        // depth-grid quantisation (2× coarser) loses sub-pixel position.
        for (label, mx, my) in [("head", head.0, head.1), ("tail", tail.0, tail.1)] {
            assert!(
                (4.0..=56.0).contains(&mx),
                "{label} mask-x {mx} should be near the bar (cols 5..55)"
            );
            assert!(
                (15.0..=25.0).contains(&my),
                "{label} mask-y {my} should be near the bar (rows 16..24)"
            );
        }
    }

    /// Mask resolution > depth resolution, depth covers only a centre strip.
    /// Snapping must pull both endpoints into the strip; the public return is
    /// in mask space, so the assertions are against the strip's image-space
    /// projection (depth cols 12..18 → mask cols 24..36; depth rows 8..12 →
    /// mask rows 16..24).
    #[tokio::test]
    async fn test_find_head_tail_depth_higher_res_mask_partial_depth() {
        let detector = FishHeadTailDetector {};

        // Mask: 40h × 60w with a horizontal bar (rows 16..24, cols 5..55).
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 16..24 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }

        // Depth: 20h × 30w, only a centre strip (rows 8..12, cols 12..18) is
        // populated. The strip's midpoint maps to depth-coord (15, 10), which
        // matches the post-rescale image-space midpoint for the bar.
        let mut depth_data = Array2::<f32>::zeros((20, 30));
        for r in 8..12 {
            for c in 12..18 {
                depth_data[[r, c]] = 1.0;
            }
        }
        let depth_map = DepthMap(depth_data);

        let coords = detector
            .find_head_tail_depth(&mask, &depth_map)
            .await
            .expect("detector should succeed on mismatched grids");

        let head = (coords.head.0[0], coords.head.0[1]);
        let tail = (coords.tail.0[0], coords.tail.0[1]);
        assert_ne!(head, tail, "endpoints collapsed to a single point: {head:?}");

        // Strip in mask space: cols [12*2, 18*2) = [24, 36); rows [8*2, 12*2)
        // = [16, 24). Endpoints should land inside that mask-space strip.
        for (label, mx, my) in [("head", head.0, head.1), ("tail", tail.0, tail.1)] {
            assert!(
                (24.0..36.0).contains(&mx),
                "{label} mask-x {mx} should be inside the strip (mask cols 24..36) — \
                 returns must be in mask space, not depth-index space"
            );
            assert!(
                (16.0..24.0).contains(&my),
                "{label} mask-y {my} should be inside the strip (mask rows 16..24)"
            );
        }
    }
}
