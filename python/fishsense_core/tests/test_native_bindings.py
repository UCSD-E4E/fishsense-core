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
