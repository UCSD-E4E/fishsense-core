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
    /// (defined by `laser_origin` and `laser_axis` in camera space).
    ///
    /// `laser_axis` is a direction: any non-zero length is accepted and normalised
    /// internally, so the result does not depend on its magnitude. A zero-length axis
    /// has no direction and yields a non-finite point rather than a plausible wrong one.
    ///
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
        let camera_axis: Array1<f32> = projected_point.mapv(|v| -v / norm);

        // The magnitude of a direction carries no meaning, but the closed form below is
        // only valid for a unit axis: normalise rather than silently returning a point
        // scaled by an arbitrary factor. ||laser_axis|| == 0 normalises to NaN, which
        // propagates to the result instead of looking like an ordinary shallow depth.
        let axis_norm = laser_axis.dot(laser_axis).sqrt();
        let unit_laser_axis: Array1<f32> = laser_axis.mapv(|v| v / axis_norm);

        let dot_ca_lo = camera_axis.dot(laser_origin);
        let dot_la_lo = unit_laser_axis.dot(laser_origin);
        let dot_la_ca = unit_laser_axis.dot(&camera_axis);

        let point_constant = (dot_ca_lo - dot_la_lo * dot_la_ca) / (1.0 - dot_la_ca * dot_la_ca);

        camera_axis.mapv(|v| v * point_constant)
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

    /// Regression: `laser_axis` is a direction, so its magnitude must not change the
    /// answer. Before normalisation an axis of length 1.5 put a 1.5 m dot at ~1 mm.
    #[test]
    fn compute_world_point_from_laser_ignores_axis_magnitude() {
        // K = [[2000, 0, 1000], [0, 2000, 750], [0, 0, 1]]
        let k_inv = array![[0.0005_f32, 0.0, -0.5], [0.0, 0.0005, -0.375], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: k_inv };

        // Laser at (0.104, 0, 0) aimed at (0.2, 0.1, 1.5); the dot is at the target.
        let laser_origin = array![0.104_f32, 0.0, 0.0];
        let target = array![0.2_f32, 0.1, 1.5];
        let axis = &target - &laser_origin; // ||axis|| == 1.5064, not 1
        let image_point = array![1_266.666_7_f32, 883.333_3];

        let unit_axis = {
            let n = axis.dot(&axis).sqrt();
            axis.mapv(|v| v / n)
        };

        let from_raw = handler.compute_world_point_from_laser(&laser_origin, &axis, &image_point);
        let from_unit = handler.compute_world_point_from_laser(&laser_origin, &unit_axis, &image_point);

        for i in 0..3 {
            assert!((from_raw[i] - target[i]).abs() < 1e-3, "axis {i}: expected {}, got {}", target[i], from_raw[i]);
            assert!((from_raw[i] - from_unit[i]).abs() < 1e-5, "axis {i}: raw {} != unit {}", from_raw[i], from_unit[i]);
        }

        // ...and scaling a unit axis is likewise a no-op.
        for k in [0.5_f32, 2.0, 10.0] {
            let scaled = handler.compute_world_point_from_laser(&laser_origin, &unit_axis.mapv(|v| v * k), &image_point);
            for i in 0..3 {
                assert!((scaled[i] - from_unit[i]).abs() < 1e-4, "k={k} axis {i}: {} != {}", scaled[i], from_unit[i]);
            }
        }
    }

    /// A zero-length axis has no direction: the answer must be non-finite rather than
    /// an ordinary-looking shallow point.
    #[test]
    fn compute_world_point_from_laser_zero_axis_is_not_finite() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let result = handler.compute_world_point_from_laser(
            &array![0.104_f32, 0.0, 0.0],
            &array![0.0_f32, 0.0, 0.0],
            &array![10.0_f32, 20.0],
        );
        assert!(result.iter().all(|v| !v.is_finite()), "expected non-finite, got {result}");
    }
}
