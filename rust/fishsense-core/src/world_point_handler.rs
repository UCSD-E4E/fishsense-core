use ndarray::{array, Array1, Array2};

pub struct WorldPointHandler {
    pub camera_intrinsics_inverted: Array2<f32>
}

impl WorldPointHandler {
    /// Project an image-space coordinate into camera-space via K⁻¹ · [x, y, 1].
    /// Result is the un-scaled ray (no depth applied).
    pub fn project_image_point(&self, image_coordinate: &Array1<f32>) -> Array1<f32> {
        self.camera_intrinsics_inverted.dot(&array![image_coordinate[0], image_coordinate[1], 1f32])
    }

    pub fn compute_world_point_from_depth(&self, image_coordinate: &Array1<f32>, depth: f32) -> Array1<f32> {
        // The camera intrinsics includes the pixel pitch.
        self.project_image_point(image_coordinate) * depth
    }

    /// Triangulate the 3D point seen at `image_coordinate` against a known laser line
    /// (defined by `laser_origin` and unit `laser_axis` in camera space).
    /// Uses the least-squares closest-point formulation between the camera ray and the laser line,
    /// matching the convention where the camera looks down -z (hence the sign flip on the projected point).
    pub fn compute_world_point_from_laser(
        &self,
        laser_origin: &Array1<f32>,
        laser_axis: &Array1<f32>,
        image_coordinate: &Array1<f32>,
    ) -> Array1<f32> {
        let projected_point = self.project_image_point(image_coordinate);
        let norm = projected_point.dot(&projected_point).sqrt();
        let final_laser_axis: Array1<f32> = projected_point.mapv(|v| -v / norm);

        let dot_fla_lo = final_laser_axis.dot(laser_origin);
        let dot_la_lo = laser_axis.dot(laser_origin);
        let dot_la_fla = laser_axis.dot(&final_laser_axis);

        let point_constant = (dot_fla_lo - dot_la_lo * dot_la_fla) / (1.0 - dot_la_fla * dot_la_fla);

        final_laser_axis.mapv(|v| v * point_constant)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::world_point_handler::WorldPointHandler;

    #[test]
    fn compute_world_point_from_depth() {
        let image_point = array![889.631_6_f32, 336.585_48_f32];
        let depth = 0.535_531_04_f32;
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

    /// project_image_point with identity K⁻¹: returns [x, y, 1] unscaled.
    #[test]
    fn project_image_point_identity() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let result = handler.project_image_point(&array![3.0_f32, 4.0]);
        assert_eq!(result, array![3.0_f32, 4.0, 1.0]);
    }

    /// Triangulating a camera ray against a laser line that passes through the ray's
    /// expected hit point should recover that point. With identity K⁻¹, image (0,0)
    /// maps to ray direction (0, 0, -1) (after the camera-faces-−z sign flip), so a
    /// laser line passing through (0, 0, -d) along axis (0, 0, 1) should triangulate
    /// to (0, 0, -d).
    #[test]
    fn compute_world_point_from_laser_axial_hit() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let laser_origin = array![0.0_f32, 0.0, -2.0];
        let laser_axis = array![1.0_f32, 0.0, 0.0]; // perpendicular to camera ray
        let image_point = array![0.0_f32, 0.0];
        let result = handler.compute_world_point_from_laser(&laser_origin, &laser_axis, &image_point);
        // closest point on camera ray (0,0,-t) to laser line (s,0,-2) is (0,0,-2)
        assert!((result[0] - 0.0).abs() < 1e-5, "x: {}", result[0]);
        assert!((result[1] - 0.0).abs() < 1e-5, "y: {}", result[1]);
        assert!((result[2] - (-2.0)).abs() < 1e-5, "z: {}", result[2]);
    }
}