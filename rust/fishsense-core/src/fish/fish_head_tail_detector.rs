use ndarray::Array1;

use crate::{
    errors::FishSenseError,
    spatial::{connected_components::connected_components, types::DepthMap},
};

pub struct ImageCoord(pub Array1<f32>);

pub struct SnappedDepthMap {
    pub left: Array1<f32>,
    pub right: Array1<f32>,
}

pub struct FishHeadTailDetector {}

impl FishHeadTailDetector {
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

    /// Verifies the core snapping behaviour against the Python `correct_labels`
    /// implementation in 01_process.ipynb.
    ///
    /// Setup:
    ///   7×7 depth map — border pixels = 0.0, centre 3×3 block (rows 2–4, cols 2–4) = 0.5.
    ///   Annotations: left=[1,1], right=[5,5] — both land on the zero-depth border.
    ///   The midpoint [3,3] falls inside the 0.5 connected region.
    ///
    /// Python equivalent:
    ///   correct_labels(depth_map, np.array([[1,1],[5,5]]))
    ///   → [[2,2],[4,4]]
    ///
    /// Expected:
    ///   BFS from [3,3] collects the 9-pixel centre block.
    ///   left  snaps to [2,2] — nearest centre-block pixel to [1,1].
    ///   right snaps to [4,4] — nearest centre-block pixel to [5,5].
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

    /// When annotation points already lie on the connected component they should
    /// be returned unchanged (nearest component point is themselves).
    ///
    /// Setup: uniform 5×5 depth map — the whole grid is one component.
    #[tokio::test]
    async fn test_snap_to_depth_map_no_change_when_already_on_component() {
        let detector = FishHeadTailDetector {};

        let depth_map = DepthMap(Array2::<f32>::from_elem((5, 5), 0.5));

        let left = ImageCoord(array![0.0_f32, 0.0]);
        let right = ImageCoord(array![4.0_f32, 4.0]);

        let result = detector.snap_to_depth_map(&depth_map, &left, &right).await.unwrap();

        assert_eq!(result.left[0] as i32, 0, "left x should stay 0");
        assert_eq!(result.left[1] as i32, 0, "left y should stay 0");
        assert_eq!(result.right[0] as i32, 4, "right x should stay 4");
        assert_eq!(result.right[1] as i32, 4, "right y should stay 4");
    }

    /// When both annotations coincide the midpoint is the same pixel and the BFS
    /// component still needs to contain at least that pixel; both outputs snap to
    /// the same nearest component point.
    #[tokio::test]
    async fn test_snap_to_depth_map_coincident_annotations() {
        let detector = FishHeadTailDetector {};

        let mut depth_data = Array2::<f32>::zeros((5, 5));
        // Single connected island at [2,2]
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
}
