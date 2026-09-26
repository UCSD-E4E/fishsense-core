"""fishsense_core's own CameraIntrinsics.

fishsense-services (v2) retires the v1 API SDK, so a core that imports
``fishsense_api_sdk`` cannot be installed there. The consumers
(:class:`RectifiedImage`, the laser detector's output rectification) read only
``camera_matrix`` and ``distortion_coefficients``, so the core carries a type
with exactly those two fields and keeps accepting anything else that has them.
"""

import dataclasses

import numpy as np
import pytest

import fishsense_core
from fishsense_core import CameraIntrinsics
from fishsense_core import _laser_detector as ld
from fishsense_core.image.image import Image
from fishsense_core.image.rectified_image import RectifiedImage

_K = [[2800.0, 0.0, 2007.0], [0.0, 2800.0, 1508.0], [0.0, 0.0, 1.0]]
_DIST = [-0.28, 0.11, 0.0, 0.0, -0.02]


class _DuckIntrinsics:
    """What v1's data worker passes: the SDK's CameraIntrinsics, by shape."""

    def __init__(self, k, dist):
        self.camera_matrix = np.array(k, dtype=float)
        self.distortion_coefficients = np.array(dist, dtype=float)


class _StubImage(Image):
    def __init__(self, data):
        self._stub = data
        super().__init__()

    def _get_data(self):
        return self._stub


# ---------------------------------------------------------------------------
# The type
# ---------------------------------------------------------------------------


class TestCameraIntrinsics:
    def test_converts_sequences_to_float64_arrays(self):
        intrinsics = CameraIntrinsics(_K, _DIST)

        assert isinstance(intrinsics.camera_matrix, np.ndarray)
        assert intrinsics.camera_matrix.dtype == np.float64
        assert intrinsics.camera_matrix.shape == (3, 3)
        assert isinstance(intrinsics.distortion_coefficients, np.ndarray)
        assert intrinsics.distortion_coefficients.dtype == np.float64
        np.testing.assert_array_equal(intrinsics.camera_matrix, _K)
        np.testing.assert_array_equal(intrinsics.distortion_coefficients, _DIST)

    def test_accepts_keyword_arguments(self):
        intrinsics = CameraIntrinsics(
            camera_matrix=np.array(_K), distortion_coefficients=np.array(_DIST)
        )
        np.testing.assert_array_equal(intrinsics.camera_matrix, _K)

    @pytest.mark.parametrize("n", [4, 5, 8])
    def test_accepts_opencv_distortion_lengths(self, n):
        dist = np.linspace(-0.1, 0.1, n)
        intrinsics = CameraIntrinsics(_K, dist)
        assert intrinsics.distortion_coefficients.shape == (n,)

    @pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
    def test_flattens_a_row_or_column_vector(self, shape):
        """``cv2.calibrateCamera`` returns distortion as ``(1, 5)``; the SDK
        squeezes it. Either orientation means the same five numbers."""
        intrinsics = CameraIntrinsics(_K, np.array(_DIST).reshape(shape))
        np.testing.assert_array_equal(intrinsics.distortion_coefficients, _DIST)

    @pytest.mark.parametrize(
        "k", [np.eye(2), np.eye(4), np.ones(9), np.ones((3, 3, 1))]
    )
    def test_rejects_a_camera_matrix_that_is_not_3x3(self, k):
        with pytest.raises(ValueError, match="3x3"):
            CameraIntrinsics(k, _DIST)

    @pytest.mark.parametrize("n", [0, 1, 3, 6, 7, 12, 14])
    def test_rejects_other_distortion_lengths(self, n):
        with pytest.raises(ValueError, match="4, 5 or 8"):
            CameraIntrinsics(_K, np.zeros(n))

    def test_rejects_a_2d_distortion_that_is_not_a_vector(self):
        """Eight numbers laid out 2x4 are not an unambiguous coefficient list."""
        with pytest.raises(ValueError, match="4, 5 or 8"):
            CameraIntrinsics(_K, np.zeros((2, 4)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_non_finite_values(self, bad):
        k = np.array(_K)
        k[0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            CameraIntrinsics(k, _DIST)
        dist = np.array(_DIST)
        dist[0] = bad
        with pytest.raises(ValueError, match="finite"):
            CameraIntrinsics(_K, dist)

    def test_is_frozen(self):
        intrinsics = CameraIntrinsics(_K, _DIST)
        with pytest.raises(dataclasses.FrozenInstanceError):
            intrinsics.camera_matrix = np.eye(3)  # type: ignore[misc]

    def test_arrays_are_read_only_copies(self):
        """Frozen all the way down: neither the caller's array nor ours can
        change the stored calibration after construction."""
        k = np.array(_K)
        intrinsics = CameraIntrinsics(k, _DIST)

        k[0, 0] = 1.0
        assert intrinsics.camera_matrix[0, 0] == 2800.0
        with pytest.raises(ValueError):
            intrinsics.camera_matrix[0, 0] = 1.0
        with pytest.raises(ValueError):
            intrinsics.distortion_coefficients[0] = 1.0

    def test_is_exported_at_the_package_root(self):
        from fishsense_core.camera_intrinsics import (  # pylint: disable=import-outside-toplevel
            CameraIntrinsics as FromModule,
        )

        assert fishsense_core.CameraIntrinsics is FromModule


# ---------------------------------------------------------------------------
# The consumers accept it — and still accept the duck-typed SDK object
# ---------------------------------------------------------------------------


class TestRectifiedImageAcceptsCameraIntrinsics:
    def _frame(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    def _k_small(self):
        return [[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]

    def test_rectifies_with_the_core_type(self):
        frame = self._frame()
        out = RectifiedImage(
            _StubImage(frame), CameraIntrinsics(self._k_small(), _DIST)
        ).data
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)  # the distortion did something

    def test_core_type_and_duck_type_rectify_identically(self):
        """No behaviour change: the SDK-shaped object v1 passes today and the
        core type give the same pixels."""
        frame = self._frame()
        ours = RectifiedImage(
            _StubImage(frame), CameraIntrinsics(self._k_small(), _DIST)
        ).data
        duck = RectifiedImage(
            _StubImage(frame), _DuckIntrinsics(self._k_small(), _DIST)
        ).data
        np.testing.assert_array_equal(ours, duck)


class TestLaserDetectorAcceptsCameraIntrinsics:
    def test_resolves_a_registered_core_type(self):
        registry = {7: CameraIntrinsics(_K, _DIST)}
        k, dist = ld._resolve_rectify_intrinsics(registry, 7, None, None)
        np.testing.assert_array_equal(k, _K)
        np.testing.assert_array_equal(dist, _DIST)

    def test_core_type_and_duck_type_rectify_identically(self):
        by_core = ld._resolve_rectify_intrinsics(
            {3: CameraIntrinsics(_K, _DIST)}, 3, None, None
        )
        by_duck = ld._resolve_rectify_intrinsics(
            {3: _DuckIntrinsics(_K, _DIST)}, 3, None, None
        )
        assert ld.rectify_prediction(1616.3, 1699.7, *by_core) == (
            ld.rectify_prediction(1616.3, 1699.7, *by_duck)
        )
