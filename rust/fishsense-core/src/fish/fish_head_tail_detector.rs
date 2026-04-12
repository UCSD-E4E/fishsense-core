use ndarray::{Array1, Array2};

use crate::spatial::types::DepthMap;

pub struct ImageCoord(pub Array1<f32>);

pub struct SnappedDepthMap {
    pub left: Array1<f32>,
    pub right: Array1<f32>,
}

pub struct FishHeadTailDetector {}

impl FishHeadTailDetector {
    fn snap_to_depth_map(&self, depth_map: &DepthMap, left_img_coord: &ImageCoord, right_img_coord: &ImageCoord) -> SnappedDepthMap {
        // Implementation for snapping to depth map
        todo!()
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
    #[test]
    fn test_snap_to_depth_map_snaps_to_nearest_connected_component() {
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

        let result = detector.snap_to_depth_map(&depth_map, &left, &right);

        assert_eq!(result.left[0] as i32, 2, "left x should snap to 2");
        assert_eq!(result.left[1] as i32, 2, "left y should snap to 2");
        assert_eq!(result.right[0] as i32, 4, "right x should snap to 4");
        assert_eq!(result.right[1] as i32, 4, "right y should snap to 4");
    }

    /// When annotation points already lie on the connected component they should
    /// be returned unchanged (nearest component point is themselves).
    ///
    /// Setup: uniform 5×5 depth map — the whole grid is one component.
    #[test]
    fn test_snap_to_depth_map_no_change_when_already_on_component() {
        let detector = FishHeadTailDetector {};

        let depth_map = DepthMap(Array2::<f32>::from_elem((5, 5), 0.5));

        let left = ImageCoord(array![0.0_f32, 0.0]);
        let right = ImageCoord(array![4.0_f32, 4.0]);

        let result = detector.snap_to_depth_map(&depth_map, &left, &right);

        assert_eq!(result.left[0] as i32, 0, "left x should stay 0");
        assert_eq!(result.left[1] as i32, 0, "left y should stay 0");
        assert_eq!(result.right[0] as i32, 4, "right x should stay 4");
        assert_eq!(result.right[1] as i32, 4, "right y should stay 4");
    }

    /// When both annotations coincide the midpoint is the same pixel and the BFS
    /// component still needs to contain at least that pixel; both outputs snap to
    /// the same nearest component point.
    #[test]
    fn test_snap_to_depth_map_coincident_annotations() {
        let detector = FishHeadTailDetector {};

        let mut depth_data = Array2::<f32>::zeros((5, 5));
        // Single connected island at [2,2]
        depth_data[[2, 2]] = 1.0;

        let depth_map = DepthMap(depth_data);

        let left = ImageCoord(array![2.0_f32, 2.0]);
        let right = ImageCoord(array![2.0_f32, 2.0]);

        let result = detector.snap_to_depth_map(&depth_map, &left, &right);

        assert_eq!(result.left[0] as i32, 2);
        assert_eq!(result.left[1] as i32, 2);
        assert_eq!(result.right[0] as i32, 2);
        assert_eq!(result.right[1] as i32, 2);
    }
}
