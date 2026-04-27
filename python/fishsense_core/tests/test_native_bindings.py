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

    def test_snap_to_depth_map_snaps_to_midpoint_component(self):
        """Endpoints land in the connected depth component containing the midpoint."""
        detector = FishHeadTailDetector()
        # Two flat depth regions split at column 2; the midpoint at (2, 2) sits
        # in the right (depth 1.0) component, so both endpoints snap there.
        depth = np.full((5, 5), 1.0, dtype=np.float32)
        depth[:, :2] = 5.0
        left = np.array([0.0, 2.0], dtype=np.float32)
        right = np.array([4.0, 2.0], dtype=np.float32)

        snapped_left, snapped_right = detector.snap_to_depth_map(depth, left, right)

        assert snapped_left.shape == (2,) and snapped_left.dtype == np.float32
        assert snapped_right.shape == (2,) and snapped_right.dtype == np.float32
        # left was outside the right component → snaps onto its boundary at x=2.
        np.testing.assert_allclose(snapped_left, [2.0, 2.0])
        # right was already in the component → unchanged.
        np.testing.assert_allclose(snapped_right, [4.0, 2.0])

    def test_snap_to_depth_map_rejects_wrong_length_coord(self):
        detector = FishHeadTailDetector()
        depth = np.zeros((4, 4), dtype=np.float32)
        good = np.array([0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError):
            detector.snap_to_depth_map(depth, np.array([0.0, 0.0, 0.0], dtype=np.float32), good)


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
