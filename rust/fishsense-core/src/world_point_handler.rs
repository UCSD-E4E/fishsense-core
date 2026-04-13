use ndarray::{array, Array2};

pub struct WorldPointHandler {
    pub camera_intrinsics_inverted: Array2<f32>
}

impl WorldPointHandler {
    pub fn compute_world_point_from_depth(&self, image_coordinate: &ndarray::Array1<f32>, depth: f32) -> ndarray::Array1<f32> {
        // The camera intrinsics includes the pixel pitch.
        self.camera_intrinsics_inverted.dot(&array![image_coordinate[0], image_coordinate[1], 1f32]) * depth
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::world_point_handler::WorldPointHandler;

    #[test]
    fn compute_world_point_from_depth() {
        let image_point = array![889.63158192f32, 336.58548892f32];
        let depth = 0.5355310460918119f32;
        let camera_intrinsics_inverted = array![[0.00070161547, 0.0, 0.0], [0.0, 0.00070161547, 0.0], [-0.67513853, -0.5045314, 1.0]].t().mapv(|v| v as f32);

        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted
        };

        assert_eq!(world_point_handler.compute_world_point_from_depth(&image_point, depth), array![-0.02729025, -0.14372465, depth]);
    }

    /// Identity intrinsics (K_inv = I): world point should equal [x, y, 1] * depth.
    #[test]
    fn compute_world_point_identity_intrinsics() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let image_point = array![3.0_f32, 4.0];
        let depth = 2.0_f32;
        let result = handler.compute_world_point_from_depth(&image_point, depth);
        assert!((result[0] - 6.0).abs() < 1e-5, "x: expected 6.0, got {}", result[0]);
        assert!((result[1] - 8.0).abs() < 1e-5, "y: expected 8.0, got {}", result[1]);
        assert!((result[2] - 2.0).abs() < 1e-5, "z: expected 2.0, got {}", result[2]);
    }

    /// Zero depth should produce the zero vector.
    #[test]
    fn compute_world_point_zero_depth() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let image_point = array![100.0_f32, 200.0];
        let result = handler.compute_world_point_from_depth(&image_point, 0.0);
        assert_eq!(result, array![0.0_f32, 0.0, 0.0]);
    }

    /// Origin image coordinate with identity intrinsics: result is [0, 0, depth].
    #[test]
    fn compute_world_point_origin_image_coord() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let image_point = array![0.0_f32, 0.0];
        let depth = 5.0_f32;
        let result = handler.compute_world_point_from_depth(&image_point, depth);
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - depth).abs() < 1e-5);
    }
}