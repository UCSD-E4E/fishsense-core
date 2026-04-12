use ndarray::Array1;

use crate::{
    errors::FishSenseError,
    fish::{
        fish_geometry::{
            classify_from_perimeter, compute_scale, correct_head, correct_tail,
            extract_perimeter, perpendicular_bisector, polygon_from_perimeter, split_polygon,
        },
        fish_pca::estimate_endpoints,
    },
    spatial::{connected_components::connected_components, types::DepthMap},
};
use geo::{Closest, ClosestPoint, Distance, Euclidean, Point};
use ndarray::Array2;

pub struct ImageCoord(pub Array1<f32>);

pub struct SnappedDepthMap {
    pub left: Array1<f32>,
    pub right: Array1<f32>,
}

pub struct FishHeadTailDetector {}

impl FishHeadTailDetector {
    /// Full three-stage pipeline: PCA → geometry refinement → depth-map snap.
    ///
    /// 1. Estimates raw endpoints with PCA from the binary `mask`.
    /// 2. Classifies and corrects head/tail using polygon geometry.
    /// 3. Snaps each corrected point to the nearest pixel of the depth
    ///    component that contains the midpoint (see `snap_to_depth_map`).
    ///
    /// Returns `SnappedDepthMap { left: head, right: tail }`.
    pub async fn find_head_tail(
        &self,
        mask: &Array2<u8>,
        depth_map: &DepthMap,
    ) -> Result<SnappedDepthMap, FishSenseError> {
        // ── Stage 1: PCA ────────────────────────────────────────────────────
        let pca = estimate_endpoints(mask)?;
        let left = pca.left;
        let right = pca.right;

        // ── Stage 2: Geometry refinement ───────────────────────────────────
        let perimeter = extract_perimeter(mask);
        if perimeter.len() < 3 {
            return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
                "fish mask perimeter has fewer than 3 points"
            )));
        }

        let classified = classify_from_perimeter(&perimeter, left, right)?;
        let head = classified.head;
        let tail = classified.tail;

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

        // ── Stage 3: Snap to depth map ──────────────────────────────────────
        let head_coord = ImageCoord(ndarray::array![
            head_corrected[0] as f32,
            head_corrected[1] as f32
        ]);
        let tail_coord = ImageCoord(ndarray::array![
            tail_corrected[0] as f32,
            tail_corrected[1] as f32
        ]);

        let snapped = self
            .snap_to_depth_map(depth_map, &head_coord, &tail_coord)
            .await?;

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
            left: nearest(&left_img_coord.0),
            right: nearest(&right_img_coord.0),
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

        let result = detector.find_head_tail(&mask, &depth_map).await;
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

        let result = detector.find_head_tail(&mask, &depth_map).await;
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
        assert!(detector.find_head_tail(&mask, &depth_map).await.is_err());
    }
}
