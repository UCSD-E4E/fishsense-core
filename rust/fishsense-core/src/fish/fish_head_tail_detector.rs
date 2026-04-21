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

pub struct HeadTailCoords {
    pub head: ImageCoord,
    pub tail: ImageCoord,
}

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
    /// Calls `find_head_tail_img` for stages 1–2, then snaps each corrected
    /// point to the nearest pixel of the depth component that contains the
    /// midpoint (see `snap_to_depth_map`).
    ///
    /// Returns `SnappedDepthMap { left: head, right: tail }`.
    #[instrument(skip(self, mask, depth_map), fields(height = mask.dim().0, width = mask.dim().1))]
    pub async fn find_head_tail_depth(
        &self,
        mask: &Array2<u8>,
        depth_map: &DepthMap,
    ) -> Result<SnappedDepthMap, FishSenseError> {
        let coords = self.find_head_tail_img(mask)?;

        // ── Stage 3: Snap to depth map ──────────────────────────────────────
        let snapped = self
            .snap_to_depth_map(depth_map, &coords.head, &coords.tail)
            .await?;

        debug!(
            head_x = snapped.left.0[0], head_y = snapped.left.0[1],
            tail_x = snapped.right.0[0], tail_y = snapped.right.0[1],
            "endpoints snapped to depth component"
        );
        Ok(snapped)
    }

    /// Snaps `left_img_coord` and `right_img_coord` to the nearest pixel of the
    /// connected depth component that contains the midpoint between them.
    ///
    /// Mirrors the Python `correct_labels` function in 01_process.ipynb:
    ///   1. Compute the midpoint of the two annotation points.
    ///   2. Run connected-components on the depth map (epsilon = 0.005).
    ///   3. Find the component label at the midpoint.
    ///   4. Snap each annotation to the nearest pixel in that component (L2).
    ///
    /// Coordinates are in `[x, y]` order; the depth map is indexed `[row, col]`
    /// i.e. `[y, x]`.
    #[instrument(skip(self, depth_map, left_img_coord, right_img_coord))]
    pub async fn snap_to_depth_map(
        &self,
        depth_map: &DepthMap,
        left_img_coord: &ImageCoord,
        right_img_coord: &ImageCoord,
    ) -> Result<SnappedDepthMap, FishSenseError> {
        const EPSILON: f32 = 0.005;

        let labels = connected_components(depth_map, EPSILON).await?;

        // Midpoint in [x, y]; clamp to valid index range.
        let (height, width) = labels.dim();
        let mid_x = (((left_img_coord.0[0] + right_img_coord.0[0]) / 2.0).round() as usize)
            .min(width.saturating_sub(1));
        let mid_y = (((left_img_coord.0[1] + right_img_coord.0[1]) / 2.0).round() as usize)
            .min(height.saturating_sub(1));

        let target_label = labels[[mid_y, mid_x]];
        debug!(mid_x, mid_y, target_label, "midpoint component identified");

        // Collect every pixel in the same component as the midpoint.
        // Convert from (row, col) → [x, y] to match ImageCoord convention.
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
            left: DepthCoord(nearest(&left_img_coord.0)),
            right: DepthCoord(nearest(&right_img_coord.0)),
        })
    }
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

        let left = ImageCoord(array![1.0_f32, 1.0]);
        let right = ImageCoord(array![5.0_f32, 5.0]);

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
        let left = ImageCoord(array![0.0_f32, 0.0]);
        let right = ImageCoord(array![4.0_f32, 4.0]);
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
        let left = ImageCoord(array![2.0_f32, 2.0]);
        let right = ImageCoord(array![2.0_f32, 2.0]);
        let result = detector.snap_to_depth_map(&depth_map, &left, &right).await.unwrap();
        assert_eq!(result.left[0] as i32, 2);
        assert_eq!(result.left[1] as i32, 2);
        assert_eq!(result.right[0] as i32, 2);
        assert_eq!(result.right[1] as i32, 2);
    }

    // ── find_head_tail integration tests ─────────────────────────────────

    /// A synthetic "fish": wide horizontal bar with full-coverage depth map.
    /// After PCA + geometry + snap, head and tail should be near the two ends
    /// of the bar (col ≈ 2 and col ≈ 57).
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
        let snapped = result.unwrap();

        // head (left) and tail (right) should be near the two extreme columns.
        let head_col = snapped.left[0] as usize;
        let tail_col = snapped.right[0] as usize;

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
        let snapped = result.unwrap();

        // Snapped points must be inside the mask region (rows 8..12, cols 2..58).
        let head_r = snapped.left[1] as usize;
        let head_c = snapped.left[0] as usize;
        let tail_r = snapped.right[1] as usize;
        let tail_c = snapped.right[0] as usize;

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
    /// Assertions:
    /// - Orientation: the returned head is strictly closer to the
    ///   labeled snout than to the labeled fork. This is the fix under
    ///   test.
    /// - Endpoint proximity: each endpoint lies within
    ///   `max(80, 0.12 * fish_length)` pixels of its label.
    ///   The scale factor matters for large fish (case_01 is ~1500 px
    ///   long; the labeled snout and the mask's extreme pixel are
    ///   genuinely ~160 px apart even after `correct_head`, which is a
    ///   PCA-endpoint-precision issue, not an orientation issue).
    ///
    /// Set `FISHSENSE_BUG_FIXTURE=<path_to_fixture_root>` to sweep every
    /// `likely_swap` case in the full 519-case fixture; the full-sweep
    /// mode asserts orientation only (endpoint precision is out of
    /// scope for this PR) and requires a ≥90 % pass rate with `case_01`
    /// (worst failure) mandatory.
    #[test]
    fn test_find_head_tail_img_bug_report_fixture() {
        use ndarray_npy::read_npy;
        use serde_json::Value;
        use std::path::PathBuf;

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        // Runs one case. `strict_endpoints = true` also checks endpoint
        // proximity; `false` checks orientation only (used by the full
        // sweep, where the goal is just to detect swap regressions).
        let run_case = |case_dir: &std::path::Path,
                        strict_endpoints: bool|
         -> Result<(), String> {
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
            if strict_endpoints {
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

            let mut passes = 0usize;
            let mut case_01_result: Option<Result<(), String>> = None;
            let mut failures: Vec<(String, String)> = Vec::new();
            for name in &swap_cases {
                let res = run_case(&root.join(name), false);
                if *name == "case_01" {
                    case_01_result = Some(res.clone());
                }
                match res {
                    Ok(()) => passes += 1,
                    Err(msg) => failures.push((name.to_string(), msg)),
                }
            }
            let total = swap_cases.len();
            let rate = passes as f64 / total.max(1) as f64;

            if let Some(Err(msg)) = case_01_result {
                panic!("case_01 (worst likely_swap) must pass orientation; failed: {msg}");
            }

            println!(
                "[bug_report_fixture] likely_swap orientation pass rate: \
                 {}/{} ({:.1}%)",
                passes,
                total,
                rate * 100.0
            );

            const MIN_PASS_RATE: f64 = 0.90;
            assert!(
                rate >= MIN_PASS_RATE,
                "likely_swap orientation pass rate {:.1}% ({}/{}) below floor {:.0}%. \
                 Sample failures: {:?}",
                rate * 100.0,
                passes,
                total,
                MIN_PASS_RATE * 100.0,
                failures.iter().take(5).collect::<Vec<_>>()
            );
        } else {
            let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/bug_report");
            for name in ["case_01", "case_150", "case_273"] {
                run_case(&base.join(name), true).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    }

    /// When the depth component covers only the centre of the mask, snapping
    /// must pull the endpoints inward relative to the raw image coordinates.
    /// This verifies that `find_head_tail_depth` ≠ `find_head_tail_img` when
    /// the depth coverage is limited.
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

        // depth coords must be snapped inside the centre strip (cols 20..40).
        let snapped = detector.find_head_tail_depth(&mask, &depth_map).await.unwrap();
        let snapped_head_col = snapped.left[0] as usize;
        let snapped_tail_col = snapped.right[0] as usize;
        assert!(
            (20..40).contains(&snapped_head_col),
            "snapped head col {snapped_head_col} should be inside depth strip 20..40"
        );
        assert!(
            (20..40).contains(&snapped_tail_col),
            "snapped tail col {snapped_tail_col} should be inside depth strip 20..40"
        );
    }
}
