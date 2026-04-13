use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;

use crate::world_point_handler::WorldPointHandler;

pub struct FishLengthCalculator {
    pub world_point_handler: WorldPointHandler,
    pub image_height: usize,
    pub image_width: usize
}

impl FishLengthCalculator {
    pub fn calculate_fish_length(&self, depth_map: &Array2<f32>, left_depth_coord: &Array1<f32>, right_depth_coord: &Array1<f32>) -> f32 {
        let (left_depth, right_depth) = (depth_map[[left_depth_coord[0] as usize, left_depth_coord[1] as usize]], depth_map[[right_depth_coord[0] as usize, right_depth_coord[1] as usize]]);

        let left_3d = self.world_point_handler.compute_world_point_from_depth(left_depth_coord, left_depth);
        let right_3d = self.world_point_handler.compute_world_point_from_depth(right_depth_coord, right_depth);

        (&left_3d - &right_3d).norm_l2()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::world_point_handler::WorldPointHandler;

    use super::FishLengthCalculator;

    #[test]
    fn calculate_fish_length() {
        let f_inv = 0.000_353_109_85_f32;
        let camera_intrinsics_inverted = array![[f_inv, 0f32, 0f32], [0f32, f_inv, 0f32], [0f32, 0f32, 1f32]];

        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted
        };

        let image_height = 3016;
        let image_width = 3987;
        let depth_map = Array2::from_elem((image_height, image_width), 0.535_531_04_f32);
        let fish_length_calcualtor = FishLengthCalculator {
            image_height,
            image_width,
            world_point_handler
        };

        let left = array![889.631_6_f32, 336.585_48_f32];
        let right = array![-355.368_4_f32, 395.585_48_f32];
        let fish_length = fish_length_calcualtor.calculate_fish_length(&depth_map, &left, &right);

        assert_eq!(fish_length, 0.23569532);
    }

    /// Same pixel for both endpoints → length should be zero.
    #[test]
    fn calculate_fish_length_same_point_is_zero() {
        let identity = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted: identity,
        };
        let depth_map = Array2::from_elem((10, 10), 1.0f32);
        let calc = FishLengthCalculator {
            image_height: 10,
            image_width: 10,
            world_point_handler,
        };
        let point = array![5.0f32, 5.0];
        let length = calc.calculate_fish_length(&depth_map, &point, &point);
        assert!(length.abs() < 1e-6, "expected 0.0, got {length}");
    }

    /// With identity intrinsics and unit depth, the 3-D distance equals the
    /// 2-D Euclidean distance between the x-components of the two image coords
    /// (since `norm` uses only the first two components of the 3-D vector and
    /// the y world-coords happen to cancel in this simple case).
    ///
    /// left  = [3, 5], right = [7, 5] → Δx = 4, Δy = 0 → length = 4.
    #[test]
    fn calculate_fish_length_horizontal_separation() {
        let identity = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted: identity,
        };
        let depth_map = Array2::from_elem((10, 10), 1.0f32);
        let calc = FishLengthCalculator {
            image_height: 10,
            image_width: 10,
            world_point_handler,
        };
        // Both points share the same row (y=5) so row index is 5 for both.
        // left col=3 → depth_map[[5,3]]=1, right col=7 → depth_map[[5,7]]=1.
        let left = array![3.0f32, 5.0];
        let right = array![7.0f32, 5.0];
        let length = calc.calculate_fish_length(&depth_map, &left, &right);
        assert!((length - 4.0).abs() < 1e-5, "expected 4.0, got {length}");
    }
}