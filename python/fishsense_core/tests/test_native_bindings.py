"""Smoke tests for the Rust-backed ``_native`` bindings.

These do not exercise the algorithms deeply — the Rust crate already has
unit tests for that. They verify that the PyO3 glue correctly:
  * registers each submodule under its short attribute name,
  * marshals numpy arrays across the FFI boundary,
  * returns results with the expected shape and dtype.
"""
# pylint: disable=import-error
import numpy as np
import pytest

from fishsense_core.fish import FishHeadTailDetector, FishSegmentation
from fishsense_core.laser import calibrate_laser
from fishsense_core.world_point import WorldPointHandler


# ---------------------------------------------------------------------------
# calibrate_laser
# ---------------------------------------------------------------------------

class TestCalibrateLaser:
    def test_vertical_line_returns_origin_and_z_axis(self):
        """Two points along the z-axis → origin at (0,0,0), orientation (0,0,1)."""
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32)
        origin, orientation = calibrate_laser(points)

        np.testing.assert_allclose(origin, [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(orientation, [0.0, 0.0, 1.0], atol=1e-5)

    def test_orientation_is_unit_vector(self):
        points = np.array([
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [3.0, 3.0, 3.0],
        ], dtype=np.float32)
        _, orientation = calibrate_laser(points)
        assert orientation.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(orientation), 1.0, atol=1e-5)

    def test_accepts_float32_input(self):
        """Regression: the documented contract is float32 — the binding must accept it."""
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32)
        origin, orientation = calibrate_laser(points)
        assert origin.shape == (3,)
        assert orientation.shape == (3,)
        assert np.issubdtype(origin.dtype, np.floating)
        assert np.issubdtype(orientation.dtype, np.floating)


# ---------------------------------------------------------------------------
# WorldPointHandler
# ---------------------------------------------------------------------------

class TestWorldPointHandler:
    """The Python wrapper coerces all array inputs to float64 before the
    native call, so callers don't need to think about dtype. The native
    binding rejects non-float64 arrays at the PyO3 boundary; without the
    coercion stage13 in fishsense-lite-mono crashed on every dive whose
    laser_label coordinates came back from the SDK as ints.
    """

    @staticmethod
    def _identity():
        return WorldPointHandler(np.eye(3))

    def test_project_image_point_int_input(self):
        """Regression: int-dtype image_point must not raise.

        Also pins the wrapper's output dtype to float64 — the native binding
        returns f64 today and downstream callers shouldn't have to defend
        against a silent precision change.
        """
        result = self._identity().project_image_point(np.array([100, 200]))
        assert result.shape == (3,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [100.0, 200.0, 1.0])

    def test_project_image_point_float64_input(self):
        result = self._identity().project_image_point(np.array([3.0, 4.0]))
        np.testing.assert_allclose(result, [3.0, 4.0, 1.0])

    def test_project_image_point_float32_input(self):
        result = self._identity().project_image_point(np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(result, [3.0, 4.0, 1.0])

    def test_project_image_point_python_list_input(self):
        """np.asarray accepts plain lists too — wrapper should handle that."""
        result = self._identity().project_image_point([3, 4])
        np.testing.assert_allclose(result, [3.0, 4.0, 1.0])

    def test_compute_world_point_from_depth_int_inputs(self):
        result = self._identity().compute_world_point_from_depth(np.array([3, 4]), 2)
        np.testing.assert_allclose(result, [6.0, 8.0, 2.0])

    def test_compute_world_point_from_laser_int_inputs(self):
        """All three array args (origin, axis, image_point) must accept ints."""
        result = self._identity().compute_world_point_from_laser(
            np.array([0, 0, -2]),  # laser_origin
            np.array([1, 0, 0]),  # laser_axis (perpendicular to camera ray)
            np.array([0, 0]),  # image_point
        )
        # closest point on camera ray (0,0,-t) to laser line (s,0,-2) is (0,0,-2)
        np.testing.assert_allclose(result, [0.0, 0.0, -2.0], atol=1e-5)

    def test_constructor_int_intrinsics(self):
        """K_inv passed as ints (e.g. np.eye default int dtype after astype) must work."""
        h = WorldPointHandler(np.eye(3, dtype=np.int64))
        assert h is not None

    def test_constructor_python_list_of_lists(self):
        """K_inv passed as a plain list-of-lists must work — np.asarray handles it."""
        h = WorldPointHandler([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = h.project_image_point([3, 4])
        np.testing.assert_allclose(result, [3.0, 4.0, 1.0])

    # -- input shape ---------------------------------------------------------

    @pytest.mark.parametrize(
        "bad_pixel",
        [
            np.array([1200.0, 800.0, 1.0]),  # homogeneous — the w used to be dropped
            np.array([1200.0]),
            np.array([[1200.0, 800.0]]),  # 2-D
        ],
    )
    def test_image_point_must_be_xy(self, bad_pixel):
        """A mis-shaped pixel is a ValueError, not a silent truncation or a panic."""
        with pytest.raises(ValueError):
            self._identity().project_image_point(bad_pixel)

    @pytest.mark.parametrize("bad_origin", [np.array([0.1, 0.0]), np.array([0.1, 0.0, 0.0, 0.0])])
    def test_laser_origin_must_be_a_3_vector(self, bad_origin):
        """Used to panic inside ndarray's dot product (PanicException in Python)."""
        with pytest.raises(ValueError):
            self._identity().compute_world_point_from_laser(
                bad_origin, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0])
            )

    @pytest.mark.parametrize("bad_axis", [np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])])
    def test_laser_axis_must_be_a_3_vector(self, bad_axis):
        with pytest.raises(ValueError):
            self._identity().compute_world_point_from_laser_with_residual(
                np.array([0.0, 0.0, -2.0]), bad_axis, np.array([0.0, 0.0])
            )

    @pytest.mark.parametrize("bad_k_inv", [np.eye(2), np.eye(4), np.ones(9)])
    def test_intrinsics_must_be_3x3(self, bad_k_inv):
        with pytest.raises(ValueError):
            WorldPointHandler(bad_k_inv)

    # -- laser triangulation ------------------------------------------------

    @staticmethod
    def _laser_scene():
        """A camera, a laser at (0.104, 0, 0), and a dot it puts at (0.2, 0.1, 1.5)."""
        k = np.array([[2000.0, 0.0, 1000.0], [0.0, 2000.0, 750.0], [0.0, 0.0, 1.0]])
        handler = WorldPointHandler(np.linalg.inv(k))
        origin = np.array([0.104, 0.0, 0.0])
        target = np.array([0.2, 0.1, 1.5])
        pixel = (k @ target)[:2] / (k @ target)[2]
        return handler, origin, target, pixel

    def test_compute_world_point_from_laser_ignores_axis_magnitude(self):
        """Regression: a non-unit axis used to put a 1.5 m dot at ~1 mm.

        ``laser_axis`` is a direction — ``target - origin`` (norm 1.5064) must
        triangulate to the same point as the unit vector calibration returns.
        """
        handler, origin, target, pixel = self._laser_scene()
        axis = target - origin
        assert not np.isclose(np.linalg.norm(axis), 1.0)  # the whole point

        raw = handler.compute_world_point_from_laser(origin, axis, pixel)
        unit = handler.compute_world_point_from_laser(
            origin, axis / np.linalg.norm(axis), pixel
        )
        np.testing.assert_allclose(raw, target, atol=1e-3)
        np.testing.assert_allclose(raw, unit, atol=1e-5)

    @pytest.mark.parametrize("scale", [0.5, 2.0, 4.0, 10.0])
    def test_compute_world_point_from_laser_scale_invariant(self, scale):
        """Scaling a unit axis is a no-op; it used to flip the depth's sign."""
        handler, origin, target, pixel = self._laser_scene()
        unit = (target - origin) / np.linalg.norm(target - origin)
        np.testing.assert_allclose(
            handler.compute_world_point_from_laser(origin, unit * scale, pixel),
            handler.compute_world_point_from_laser(origin, unit, pixel),
            atol=1e-4,
        )

    def test_compute_world_point_from_laser_rejects_zero_axis(self):
        """A zero axis points nowhere — it must not answer with a ~1 cm depth."""
        handler, origin, _, pixel = self._laser_scene()
        with pytest.raises(ValueError):
            handler.compute_world_point_from_laser(origin, np.zeros(3), pixel)

    def test_compute_world_point_from_laser_rejects_non_finite_axis(self):
        """NaN in the axis is a broken calibration row, not a direction."""
        handler, origin, _, pixel = self._laser_scene()
        with pytest.raises(ValueError):
            handler.compute_world_point_from_laser(
                origin, np.array([np.nan, 0.0, 1.0]), pixel
            )

    def test_residual_variant_agrees_with_plain_call(self):
        """The companion returns the same point, plus a plain float residual."""
        handler, origin, target, pixel = self._laser_scene()
        axis = target - origin
        point, residual = handler.compute_world_point_from_laser_with_residual(
            origin, axis, pixel
        )
        np.testing.assert_array_equal(
            point, handler.compute_world_point_from_laser(origin, axis, pixel)
        )
        assert isinstance(residual, float)

    def test_residual_is_small_for_a_consistent_dot(self):
        """The dot really is on the laser line, so the rays meet."""
        handler, origin, target, pixel = self._laser_scene()
        point, residual = handler.compute_world_point_from_laser_with_residual(
            origin, target - origin, pixel
        )
        np.testing.assert_allclose(point, target, atol=1e-3)
        assert residual < 1e-3

    def test_residual_is_the_closest_approach_distance(self):
        """Identity K⁻¹: image (0,0) looks along -z; a laser line 1 unit off in y
        passes exactly 1 unit from that ray."""
        handler = self._identity()
        point, residual = handler.compute_world_point_from_laser_with_residual(
            np.array([1.0, 1.0, -2.0]),  # laser_origin, lifted 1 in y
            np.array([1.0, 0.0, 0.0]),  # laser_axis
            np.array([0.0, 0.0]),  # image_point
        )
        np.testing.assert_allclose(point, [0.0, 0.0, -2.0], atol=1e-5)
        np.testing.assert_allclose(residual, 1.0, atol=1e-5)

    def test_residual_is_zero_when_the_rays_meet(self):
        """Laser line y=0, z=-2 crosses the camera ray through image (0,0)."""
        _, residual = self._identity().compute_world_point_from_laser_with_residual(
            np.array([1.0, 0.0, -2.0]), np.array([3.0, 0.0, 0.0]), np.array([0.0, 0.0])
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)

    def test_residual_variant_int_inputs(self):
        """Same float64 coercion contract as the rest of the wrapper."""
        point, residual = self._identity().compute_world_point_from_laser_with_residual(
            np.array([0, 0, -2]), np.array([1, 0, 0]), np.array([0, 0])
        )
        np.testing.assert_allclose(point, [0.0, 0.0, -2.0], atol=1e-5)
        assert residual == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("bad_axis", [np.zeros(3), np.array([np.nan, 0.0, 1.0])])
    def test_residual_variant_rejects_degenerate_axis(self, bad_axis):
        """Validation must not be limited to the plain call."""
        handler, origin, _, pixel = self._laser_scene()
        with pytest.raises(ValueError):
            handler.compute_world_point_from_laser_with_residual(origin, bad_axis, pixel)

    def test_residual_variant_output_types(self):
        """Pins the float64/(3,)/float contract the rest of the wrapper promises."""
        handler, origin, target, pixel = self._laser_scene()
        point, residual = handler.compute_world_point_from_laser_with_residual(
            origin, target - origin, pixel
        )
        assert point.shape == (3,)
        assert point.dtype == np.float64
        assert isinstance(residual, float)
        # ...and the plain call keeps the same dtype contract
        plain = handler.compute_world_point_from_laser(origin, target - origin, pixel)
        assert plain.dtype == np.float64

    def test_residual_is_blind_to_error_along_the_epipolar_line(self):
        """The documented blind spot, pinned where callers will rely on it.

        The laser sweeps out the image line v = 883.33 here, so moving the dot
        100 px in u stays on that line: the residual sees nothing while the
        depth collapses from 1.5 m to ~0.87 m. A residual check alone cannot
        catch a dot mislabelled along the laser.
        """
        handler, origin, target, pixel = self._laser_scene()
        point, residual = handler.compute_world_point_from_laser_with_residual(
            origin, target - origin, pixel + np.array([100.0, 0.0])
        )
        assert residual < 1e-4
        assert abs(point[2] - 1.5) > 0.5

    def test_residual_flags_error_across_the_epipolar_line(self):
        """The half it does catch: 10 px of transverse error is millimetres of residual."""
        handler, origin, target, pixel = self._laser_scene()
        clean = handler.compute_world_point_from_laser_with_residual(
            origin, target - origin, pixel
        )[1]
        offset = handler.compute_world_point_from_laser_with_residual(
            origin, target - origin, pixel + np.array([0.0, 10.0])
        )[1]
        assert clean < 1e-4
        assert offset > 5e-3

    def test_a_small_residual_can_still_mean_a_useless_point(self):
        """Issue 2's example: a dot on the wrong side of the principal point.

        The rays nearly meet — a few centimetres — but behind the camera, so the
        residual must be paired with a positive-depth check, not trusted alone.
        """
        handler, _, _, _ = self._laser_scene()
        point, residual = handler.compute_world_point_from_laser_with_residual(
            np.array([0.104, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([800.0, 700.0])
        )
        assert point[2] < 0
        assert residual < 0.1

    def test_a_zeroed_calibration_row_reports_a_perfect_residual(self):
        """The residual's worst failure mode, pinned where callers will meet it.

        An all-zero ``laser_origin`` has no baseline, so every pixel triangulates
        to the camera centre with residual 0 — a residual threshold alone would
        pass a whole dive of garbage. The depth check is what catches it.
        """
        handler, _, target, pixel = self._laser_scene()
        for image_point in (pixel, np.array([400.0, 200.0]), np.array([1900.0, 1400.0])):
            point, residual = handler.compute_world_point_from_laser_with_residual(
                np.zeros(3), target, image_point
            )
            assert residual < 1e-6
            np.testing.assert_allclose(point, [0.0, 0.0, 0.0], atol=1e-6)
            assert not point[2] > 0  # ...but the depth check rejects it

    @pytest.mark.parametrize(
        "origin,pixel_x",
        [(np.array([0.104, 0.0, 0.0]), np.nan), (np.array([np.nan, 0.0, 0.0]), 1266.6667)],
    )
    def test_non_finite_inputs_propagate(self, origin, pixel_x):
        """A NaN from a failed detector must not come back as a finite-looking point."""
        handler, _, target, _ = self._laser_scene()
        point, residual = handler.compute_world_point_from_laser_with_residual(
            origin, target - np.array([0.104, 0.0, 0.0]), np.array([pixel_x, 883.3333])
        )
        assert not np.isfinite(point).any()
        assert not np.isfinite(residual)

    def test_accepts_non_contiguous_and_fortran_order_inputs(self):
        """Sliced views and F-order arrays are ordinary caller inputs; the numpy
        marshalling must handle their strides, not just packed C-order buffers."""
        k = np.array([[2000.0, 0.0, 1000.0], [0.0, 2000.0, 750.0], [0.0, 0.0, 1.0]])
        origin = np.array([0.104, 0.0, 0.0])
        axis = np.array([0.096, 0.1, 1.5])
        pixel = np.array([1266.6667, 883.3333])

        packed = WorldPointHandler(np.linalg.inv(k)).compute_world_point_from_laser(
            origin, axis, pixel
        )
        strided = WorldPointHandler(
            np.asfortranarray(np.linalg.inv(k))
        ).compute_world_point_from_laser(
            np.array([0.104, 9.0, 0.0, 9.0, 0.0, 9.0])[::2],  # non-contiguous view
            axis,
            np.array([1266.6667, 0.0, 883.3333])[::2],
        )
        np.testing.assert_allclose(strided, packed, atol=1e-6)

    def test_triangulation_round_trips_a_calibrated_laser(self):
        """End-to-end over the seam the bug lived on: calibrate → project → triangulate.

        ``calibrate_laser`` returns some point on the fitted line plus a unit
        direction; neither which point nor which way it faces may affect the
        answer, and the recovered 3D point must be the one that was projected.
        """
        k = np.array([[2000.0, 0.0, 1000.0], [0.0, 2000.0, 750.0], [0.0, 0.0, 1.0]])
        handler = WorldPointHandler(np.linalg.inv(k))

        base = np.array([0.104, 0.0, 0.0])
        direction = np.array([0.096, 0.1, 1.5])
        direction = direction / np.linalg.norm(direction)
        on_line = np.array([base + s * direction for s in (1.0, 1.4, 1.8)])

        origin, orientation = calibrate_laser(on_line.astype(np.float32))

        expected = base + 1.4 * direction
        pixel = (k @ expected)[:2] / (k @ expected)[2]
        point, residual = handler.compute_world_point_from_laser_with_residual(
            origin, orientation, pixel
        )

        np.testing.assert_allclose(point, expected, atol=1e-3)
        assert residual < 1e-3

    def test_parallel_laser_and_camera_ray_are_not_finite(self):
        """No unique closest point — the caller must be able to see that."""
        point, residual = self._identity().compute_world_point_from_laser_with_residual(
            np.array([0.104, 0.0, 0.0]),  # laser offset from the camera centre
            np.array([0.0, 0.0, 1.0]),  # ...aimed parallel to the camera ray
            np.array([0.0, 0.0]),
        )
        assert not np.isfinite(point).any()
        assert not np.isfinite(residual)


# ---------------------------------------------------------------------------
# FishHeadTailDetector
# ---------------------------------------------------------------------------

def _horizontal_bar_mask() -> np.ndarray:
    mask = np.zeros((20, 60), dtype=np.uint8)
    mask[8:12, 2:58] = 1
    return mask


class TestFishHeadTailDetector:
    def test_find_head_tail_img_horizontal_bar(self):
        """Endpoints of a horizontal bar should land near the bar's two ends."""
        detector = FishHeadTailDetector()
        head, tail = detector.find_head_tail_img(_horizontal_bar_mask())

        assert head.shape == (2,) and head.dtype == np.float32
        assert tail.shape == (2,) and tail.dtype == np.float32

        xs = sorted([float(head[0]), float(tail[0])])
        assert xs[0] <= 5, f"one endpoint should sit near col 2, got {xs}"
        assert xs[1] >= 55, f"other endpoint should sit near col 57, got {xs}"

    def test_find_head_tail_img_empty_mask_raises(self):
        detector = FishHeadTailDetector()
        with pytest.raises(ValueError):
            detector.find_head_tail_img(np.zeros((20, 60), dtype=np.uint8))

    def test_find_head_tail_img_rejects_non_2d_mask(self):
        detector = FishHeadTailDetector()
        with pytest.raises(ValueError):
            detector.find_head_tail_img(np.zeros((5, 5, 3), dtype=np.uint8))

    def test_predict_keypoint_depths_uniform_plane(self):
        """Frontoparallel scene → both keypoints recover the scene depth."""
        detector = FishHeadTailDetector()
        depth_value = np.float32(0.5)
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:30, 5:55] = 1
        depth = np.full((40, 60), depth_value, dtype=np.float32)
        k_inv = np.eye(3, dtype=np.float32)
        snout = np.array([10.0, 20.0], dtype=np.float32)
        fork = np.array([50.0, 20.0], dtype=np.float32)

        snout_d, fork_d = detector.predict_keypoint_depths(mask, depth, k_inv, snout, fork)

        assert isinstance(snout_d, float) and isinstance(fork_d, float)
        np.testing.assert_allclose(snout_d, depth_value, atol=1e-3)
        np.testing.assert_allclose(fork_d, depth_value, atol=1e-3)

    def test_predict_keypoint_depths_rejects_wrong_length_keypoint(self):
        """Snout/fork must be length-2 [x, y] arrays."""
        detector = FishHeadTailDetector()
        mask = np.ones((4, 4), dtype=np.uint8)
        depth = np.full((4, 4), 1.0, dtype=np.float32)
        k_inv = np.eye(3, dtype=np.float32)
        snout = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        fork = np.array([2.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError):
            detector.predict_keypoint_depths(mask, depth, k_inv, snout, fork)

    def test_predict_keypoint_depths_rejects_non_3x3_k_inv(self):
        """K_inv must be 3×3 — bad shape surfaces as ValueError, not panic."""
        detector = FishHeadTailDetector()
        mask = np.ones((4, 4), dtype=np.uint8)
        depth = np.full((4, 4), 1.0, dtype=np.float32)
        bad_k = np.zeros((2, 3), dtype=np.float32)
        snout = np.array([1.0, 1.0], dtype=np.float32)
        fork = np.array([2.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError):
            detector.predict_keypoint_depths(mask, depth, bad_k, snout, fork)

    def test_predict_keypoint_depths_is_deterministic(self):
        """Seeded RANSAC → identical depths across repeated calls."""
        detector = FishHeadTailDetector()
        rng = np.random.default_rng(0)
        depth = (0.5 + 0.001 * rng.standard_normal((40, 60))).astype(np.float32)
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:30, 5:55] = 1
        k_inv = np.eye(3, dtype=np.float32)
        snout = np.array([10.0, 20.0], dtype=np.float32)
        fork = np.array([50.0, 20.0], dtype=np.float32)

        a = detector.predict_keypoint_depths(mask, depth, k_inv, snout, fork)
        b = detector.predict_keypoint_depths(mask, depth, k_inv, snout, fork)
        assert a == b


# ---------------------------------------------------------------------------
# FishSegmentation
# ---------------------------------------------------------------------------

class TestFishSegmentation:
    """Instantiation-only smoke test.

    `inference` loads a real ONNX model and runs it, which is too heavy for a
    smoke test; full model behaviour is covered by Rust integration tests.
    """

    def test_construct(self):
        seg = FishSegmentation()
        assert seg is not None

    def test_active_provider_none_before_load(self):
        """`active_provider` must be queryable before `load_model` and return None."""
        seg = FishSegmentation()
        assert seg.active_provider() is None

    def test_active_provider_set_after_load(self):
        """After `load_model`, `active_provider` returns one of the known
        labels. CI runners are typically CPU-only, but we don't assert "CPU"
        here so the same test passes in a CUDA-enabled environment."""
        seg = FishSegmentation()
        seg.load_model()
        provider = seg.active_provider()
        assert provider in {"CPU", "CUDA", "CoreML"}, provider
