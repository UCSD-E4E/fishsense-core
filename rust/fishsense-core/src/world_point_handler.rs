use ndarray::{array, Array1, Array2};

/// The outcome of triangulating a camera ray against a laser line.
pub struct LaserTriangulation {
    /// Closest point on the camera ray to the laser line, in camera space.
    pub point: Array1<f32>,
    /// Distance between the camera ray and the laser line at their closest approach,
    /// in the same units as `laser_origin`. Zero when the two genuinely intersect.
    ///
    /// It measures only the component of the inconsistency *across* the laser's
    /// epipolar line — the image line the laser sweeps out. A pixel error *along*
    /// that line moves the answer up and down the laser and leaves the residual at
    /// zero: in a 1.5 m test scene, 100 px along-epipolar keeps the residual under
    /// 1e-7 m while the depth drops to 0.87 m. So a small residual means "consistent
    /// with this laser", not "correct depth" — combine it with a plausible-range
    /// check. The worst case is an all-zero `laser_origin`: a missing or zeroed
    /// extrinsics row has no baseline to triangulate against, so *every* pixel comes
    /// back at the camera centre with residual exactly 0. Only the depth check
    /// catches that. It is also metric, so the same pixel error yields a larger residual
    /// further from the camera; a fixed threshold is implicitly depth-dependent, and
    /// the f32 solve puts a noise floor of ~1e-5 m under a metre-scale scene.
    pub residual: f32,
}

pub struct WorldPointHandler {
    pub camera_intrinsics_inverted: Array2<f32>
}

impl WorldPointHandler {
    /// Project an image-space coordinate into camera-space via K⁻¹ · [x, y, 1].
    /// Result is the un-scaled ray (no depth applied).
    ///
    /// # Panics
    /// If `image_coordinate` is not `[x, y]`, or `camera_intrinsics_inverted` is not
    /// 3×3. A homogeneous `[x, y, w]` pixel is a mistake worth hearing about, not a
    /// third element to drop.
    pub fn project_image_point(&self, image_coordinate: &Array1<f32>) -> Array1<f32> {
        assert_eq!(
            self.camera_intrinsics_inverted.dim(),
            (3, 3),
            "camera_intrinsics_inverted must be 3x3"
        );
        assert_eq!(
            image_coordinate.len(),
            2,
            "image_coordinate must be [x, y]"
        );
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
    /// The returned point is the closest point on the *camera ray* to the laser line;
    /// it always exists, even for a pixel no laser dot could have produced. Callers that
    /// need to tell a clean intersection from a distant closest approach should use
    /// [`Self::compute_world_point_from_laser_with_residual`] instead.
    pub fn compute_world_point_from_laser(
        &self,
        laser_origin: &Array1<f32>,
        laser_axis: &Array1<f32>,
        image_coordinate: &Array1<f32>,
    ) -> Array1<f32> {
        self.compute_world_point_from_laser_with_residual(laser_origin, laser_axis, image_coordinate)
            .point
    }

    /// [`Self::compute_world_point_from_laser`] plus the closest-approach distance the
    /// solve computes on the way — the one number that separates "these rays meet" from
    /// "their closest approach is 3 metres apart". See [`LaserTriangulation::residual`]
    /// for what it does and does not catch.
    ///
    /// Uses the least-squares closest-point formulation between the camera ray and the
    /// laser line, matching the convention where the camera looks down -z (hence the
    /// sign flip on the projected point).
    ///
    /// # Panics
    /// If `laser_origin` or `laser_axis` is not a 3-vector, or `image_coordinate` is
    /// not `[x, y]`.
    pub fn compute_world_point_from_laser_with_residual(
        &self,
        laser_origin: &Array1<f32>,
        laser_axis: &Array1<f32>,
        image_coordinate: &Array1<f32>,
    ) -> LaserTriangulation {
        assert_eq!(laser_origin.len(), 3, "laser_origin must be a 3-vector");
        assert_eq!(laser_axis.len(), 3, "laser_axis must be a 3-vector");

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

        // Zero when the two lines are parallel — there is no unique closest point, so
        // the division yields a non-finite point and residual.
        let denominator = 1.0 - dot_la_ca * dot_la_ca;

        let camera_t = (dot_ca_lo - dot_la_lo * dot_la_ca) / denominator;
        let laser_s = (dot_la_ca * dot_ca_lo - dot_la_lo) / denominator;

        let point = camera_axis.mapv(|v| v * camera_t);
        let closest_point_on_laser = laser_origin + &unit_laser_axis.mapv(|v| v * laser_s);
        let residual = (&point - &closest_point_on_laser)
            .mapv(|v| v * v)
            .sum()
            .sqrt();

        LaserTriangulation { point, residual }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1};

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

    /// Camera ray and laser line that genuinely meet: residual is zero.
    #[test]
    fn compute_world_point_from_laser_residual_zero_when_rays_meet() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        // Camera ray through image (0,0) is (0, 0, -t); the laser line y=0, z=-2 crosses it.
        let result = handler.compute_world_point_from_laser_with_residual(
            &array![1.0_f32, 0.0, -2.0],
            &array![3.0_f32, 0.0, 0.0],
            &array![0.0_f32, 0.0],
        );
        assert!(result.residual.abs() < 1e-5, "residual: {}", result.residual);
        assert!((result.point[2] - (-2.0)).abs() < 1e-5, "z: {}", result.point[2]);
    }

    /// Skew rays: the residual is their closest-approach distance, so a caller can tell
    /// a real intersection from a pixel this laser could not have produced.
    #[test]
    fn compute_world_point_from_laser_residual_is_closest_approach() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        // Same line as above, lifted 1 unit in y: it now passes 1 unit from the camera ray.
        let result = handler.compute_world_point_from_laser_with_residual(
            &array![1.0_f32, 1.0, -2.0],
            &array![1.0_f32, 0.0, 0.0],
            &array![0.0_f32, 0.0],
        );
        assert!((result.residual - 1.0).abs() < 1e-5, "residual: {}", result.residual);
    }

    /// The point returned by the plain call is the point the residual call reports.
    #[test]
    fn compute_world_point_from_laser_agrees_with_residual_variant() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        let origin = array![0.104_f32, 0.02, -1.0];
        let axis = array![0.3_f32, 1.0, 0.4];
        let image_point = array![0.2_f32, -0.35];
        let plain = handler.compute_world_point_from_laser(&origin, &axis, &image_point);
        let detailed = handler.compute_world_point_from_laser_with_residual(&origin, &axis, &image_point);
        assert_eq!(plain, detailed.point);
    }

    /// Laser line parallel to the camera ray: no unique closest point, so the solve is
    /// non-finite rather than confidently wrong.
    #[test]
    fn compute_world_point_from_laser_parallel_lines_are_not_finite() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        // Image (0,0) looks along (0, 0, -1); an axis along z is parallel to it.
        let result = handler.compute_world_point_from_laser_with_residual(
            &array![0.104_f32, 0.0, 0.0],
            &array![0.0_f32, 0.0, 1.0],
            &array![0.0_f32, 0.0],
        );
        assert!(!result.residual.is_finite(), "residual: {}", result.residual);
    }

    /// K = [[2000, 0, 1000], [0, 2000, 750], [0, 0, 1]] — the scene both laser issues
    /// were reported against.
    fn reported_handler() -> WorldPointHandler {
        WorldPointHandler {
            camera_intrinsics_inverted: array![
                [0.0005_f32, 0.0, -0.5],
                [0.0, 0.0005, -0.375],
                [0.0, 0.0, 1.0]
            ],
        }
    }

    /// Distance from `p` to the line through `o` along `a` — an independent check on
    /// the residual that does not reuse anything from the solve.
    fn distance_to_line(p: &Array1<f32>, o: &Array1<f32>, a: &Array1<f32>) -> f32 {
        let norm = a.dot(a).sqrt();
        let unit = a.mapv(|v| v / norm);
        let offset = p - o;
        let along = offset.dot(&unit);
        let across = &offset - &unit.mapv(|v| v * along);
        across.dot(&across).sqrt()
    }

    /// The contract, checked without hand-computed expectations: the answer lies on
    /// the camera ray, the residual is its distance to the laser line, and no other
    /// point on the ray is closer.
    #[test]
    fn compute_world_point_from_laser_is_the_closest_point_on_the_camera_ray() {
        let handler = reported_handler();
        // Deliberately generic: nothing axis-aligned, nothing perpendicular.
        let laser_origin = array![0.104_f32, -0.031, 0.012];
        let laser_axis = array![0.21_f32, -0.44, 1.37];
        let image_point = array![1_180.0_f32, 640.0];

        let result =
            handler.compute_world_point_from_laser_with_residual(&laser_origin, &laser_axis, &image_point);

        // ...on the camera ray: the point is parallel to K⁻¹·[x, y, 1].
        let ray = handler.project_image_point(&image_point);
        let cos = result.point.dot(&ray)
            / (result.point.dot(&result.point).sqrt() * ray.dot(&ray).sqrt());
        assert!(cos.abs() > 1.0 - 1e-5, "point is off the camera ray: cos = {cos}");

        // ...residual really is the distance from that point to the laser line.
        let independent = distance_to_line(&result.point, &laser_origin, &laser_axis);
        assert!(
            (result.residual - independent).abs() < 1e-5,
            "residual {} != distance to line {independent}",
            result.residual
        );

        // ...and it is the minimum: sliding along the ray in either direction is worse.
        let ray_unit = {
            let n = ray.dot(&ray).sqrt();
            ray.mapv(|v| v / n)
        };
        for step in [-0.05_f32, 0.05] {
            let moved = &result.point + &ray_unit.mapv(|v| v * step);
            let worse = distance_to_line(&moved, &laser_origin, &laser_axis);
            assert!(worse > result.residual, "step {step} gave {worse} <= {}", result.residual);
        }
    }

    /// A laser line is a line, not a ray: negating the axis describes the same line and
    /// must give the same answer. Guards the `-(â·o)(â·d)` sign pair in the solve.
    #[test]
    fn compute_world_point_from_laser_is_invariant_to_axis_sign() {
        let handler = reported_handler();
        let laser_origin = array![0.104_f32, -0.031, 0.012];
        let laser_axis = array![0.21_f32, -0.44, 1.37];
        let image_point = array![1_180.0_f32, 640.0];

        let forward =
            handler.compute_world_point_from_laser_with_residual(&laser_origin, &laser_axis, &image_point);
        let backward = handler.compute_world_point_from_laser_with_residual(
            &laser_origin,
            &laser_axis.mapv(|v| -v),
            &image_point,
        );

        for i in 0..3 {
            assert!((forward.point[i] - backward.point[i]).abs() < 1e-6, "axis {i}: {} vs {}", forward.point[i], backward.point[i]);
        }
        assert!((forward.residual - backward.residual).abs() < 1e-6);
    }

    /// The residual's blind spot, pinned so the documented caveat cannot drift: it only
    /// sees error *across* the laser's epipolar line. Here that line is v = 883.33, so
    /// error in u leaves the residual at zero while wrecking the depth, and error in v
    /// is what the residual actually reports.
    #[test]
    fn compute_world_point_from_laser_residual_is_blind_along_the_epipolar_line() {
        let handler = reported_handler();
        let laser_origin = array![0.104_f32, 0.0, 0.0];
        let laser_axis = array![0.096_f32, 0.1, 1.5]; // aimed at (0.2, 0.1, 1.5)

        let along = handler.compute_world_point_from_laser_with_residual(
            &laser_origin,
            &laser_axis,
            &array![1_366.666_7_f32, 883.333_3], // +100 px along the epipolar line
        );
        assert!(along.residual < 1e-4, "along-epipolar residual: {}", along.residual);
        assert!(
            (along.point[2] - 1.5).abs() > 0.5,
            "expected a badly wrong depth, got {}",
            along.point[2]
        );

        let across = handler.compute_world_point_from_laser_with_residual(
            &laser_origin,
            &laser_axis,
            &array![1_266.666_7_f32, 893.333_3], // +10 px across it
        );
        assert!(across.residual > 5e-3, "across-epipolar residual: {}", across.residual);
        // ...even though this one barely moved the depth.
        assert!((across.point[2] - 1.5).abs() < 0.05, "depth: {}", across.point[2]);
    }

    /// The other half of the caveat: a laser line through the camera centre meets every
    /// camera ray there exactly, so residual 0 can still mean a useless point.
    #[test]
    fn compute_world_point_from_laser_residual_is_zero_at_the_camera_centre() {
        let handler = reported_handler();
        let result = handler.compute_world_point_from_laser_with_residual(
            &array![0.104_f32, 0.0, 0.0],
            &array![1.0_f32, 0.0, 0.0], // parallel to the image plane, through the centre
            &array![1_200.0_f32, 800.0],
        );
        assert!(result.residual < 1e-5, "residual: {}", result.residual);
        assert!(
            result.point.dot(&result.point).sqrt() < 1e-5,
            "expected the camera centre, got {}",
            result.point
        );
    }

    /// A homogeneous `[x, y, w]` pixel used to lose its `w` silently, which is the
    /// same class of quiet wrong answer as the non-unit `laser_axis` bug.
    #[test]
    #[should_panic(expected = "image_coordinate must be [x, y]")]
    fn project_image_point_rejects_a_homogeneous_pixel() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        handler.project_image_point(&array![6.0_f32, 8.0, 2.0]);
    }

    /// Mis-shaped extrinsics used to panic deep inside ndarray's dot product.
    #[test]
    #[should_panic(expected = "laser_origin must be a 3-vector")]
    fn compute_world_point_from_laser_rejects_a_2d_laser_origin() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        handler.compute_world_point_from_laser(
            &array![0.104_f32, 0.0],
            &array![1.0_f32, 0.0, 0.0],
            &array![0.0_f32, 0.0],
        );
    }

    #[test]
    #[should_panic(expected = "laser_axis must be a 3-vector")]
    fn compute_world_point_from_laser_rejects_a_4d_laser_axis() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let handler = WorldPointHandler { camera_intrinsics_inverted: identity };
        handler.compute_world_point_from_laser(
            &array![0.104_f32, 0.0, 0.0],
            &array![1.0_f32, 0.0, 0.0, 0.0],
            &array![0.0_f32, 0.0],
        );
    }

    #[test]
    #[should_panic(expected = "camera_intrinsics_inverted must be 3x3")]
    fn project_image_point_rejects_non_3x3_intrinsics() {
        let handler = WorldPointHandler {
            camera_intrinsics_inverted: array![[1.0_f32, 0.0], [0.0, 1.0]],
        };
        handler.project_image_point(&array![3.0_f32, 4.0]);
    }

    /// The residual's worst failure: with no baseline (an all-zero `laser_origin`,
    /// i.e. a missing or zeroed extrinsics row) every pixel triangulates to the camera
    /// centre and reports a *perfect* residual. Nothing here is wrong — the geometry
    /// really is degenerate — but it is why callers must keep the depth check.
    #[test]
    fn compute_world_point_from_laser_endorses_a_zeroed_calibration() {
        let handler = reported_handler();
        let zeroed_origin = array![0.0_f32, 0.0, 0.0];
        let laser_axis = array![0.096_f32, 0.1, 1.5];

        for image_point in [
            array![1_266.666_7_f32, 883.333_3],
            array![400.0_f32, 200.0],
            array![1_900.0_f32, 1_400.0],
        ] {
            let result = handler.compute_world_point_from_laser_with_residual(
                &zeroed_origin,
                &laser_axis,
                &image_point,
            );
            assert!(result.residual < 1e-6, "residual: {}", result.residual);
            assert!(
                result.point.dot(&result.point).sqrt() < 1e-6,
                "expected the camera centre, got {}",
                result.point
            );
        }
    }

    /// A detector that hands back a NaN pixel, or a calibration row that parses to NaN,
    /// must not come back with a finite-looking answer.
    #[test]
    fn compute_world_point_from_laser_propagates_non_finite_inputs() {
        let handler = reported_handler();
        let cases = [
            (array![0.104_f32, 0.0, 0.0], array![f32::NAN, 883.333_3_f32]),
            (array![f32::NAN, 0.0, 0.0], array![1_266.666_7_f32, 883.333_3]),
        ];
        for (laser_origin, image_point) in cases {
            let result = handler.compute_world_point_from_laser_with_residual(
                &laser_origin,
                &array![0.096_f32, 0.1, 1.5],
                &image_point,
            );
            assert!(result.point.iter().all(|v| !v.is_finite()), "point: {}", result.point);
            assert!(!result.residual.is_finite(), "residual: {}", result.residual);
        }
    }

    /// The residual is metric, so the same pixel error costs proportionally more
    /// further away — which is why a fixed threshold is implicitly depth-dependent.
    /// Doubling the range doubles the residual for a fixed 10 px transverse error.
    #[test]
    fn compute_world_point_from_laser_residual_scales_with_range() {
        let handler = reported_handler();
        let laser_origin = array![0.104_f32, 0.0, 0.0];
        // Every one of these targets sits on the same camera ray, so the pixel is
        // unchanged and only the range varies.
        let on_axis = array![1_266.666_7_f32, 883.333_3];
        let offset = array![1_266.666_7_f32, 893.333_3]; // +10 px across the epipolar line

        let mut previous = 0.0_f32;
        for depth in [0.75_f32, 1.5, 3.0, 6.0] {
            let target = array![0.133_333_34_f32 * depth, 0.066_666_67 * depth, depth];
            let laser_axis = &target - &laser_origin;

            let clean = handler
                .compute_world_point_from_laser_with_residual(&laser_origin, &laser_axis, &on_axis)
                .residual;
            assert!(clean < 1e-4, "clean residual at {depth} m: {clean}");

            let residual = handler
                .compute_world_point_from_laser_with_residual(&laser_origin, &laser_axis, &offset)
                .residual;
            if previous > 0.0 {
                let ratio = residual / previous;
                assert!((1.8..=2.2).contains(&ratio), "at {depth} m the ratio was {ratio}");
            }
            previous = residual;
        }
    }
}
