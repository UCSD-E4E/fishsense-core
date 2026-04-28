"""Unit tests for the Image base class and RectifiedImage."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from fishsense_core.image.image import Image


# ---------------------------------------------------------------------------
# Concrete stub so we can exercise the abstract base class
# ---------------------------------------------------------------------------

class _StubImage(Image):
    """Minimal concrete Image that returns a fixed ndarray."""

    def __init__(self, data: np.ndarray):
        self._data = data
        self._call_count = 0
        super().__init__()

    def _get_data(self) -> np.ndarray:
        self._call_count += 1
        return self._data


# ---------------------------------------------------------------------------
# Image base-class tests
# ---------------------------------------------------------------------------

class TestImage:
    def _make(self, h: int = 4, w: int = 6) -> _StubImage:
        data = np.zeros((h, w, 3), dtype=np.uint8)
        return _StubImage(data)

    def test_width(self):
        img = self._make(h=4, w=6)
        assert img.width == 6

    def test_height(self):
        img = self._make(h=4, w=6)
        assert img.height == 4

    def test_data_shape(self):
        data = np.ones((3, 5, 3), dtype=np.uint8)
        img = _StubImage(data)
        assert img.data.shape == (3, 5, 3)

    def test_data_content(self):
        data = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        img = _StubImage(data)
        np.testing.assert_array_equal(img.data, data)

    def test_lazy_loading_called_once(self):
        """_get_data must be called at most once, regardless of how many
        times width, height, and data are accessed."""
        img = self._make()
        _ = img.data
        _ = img.width
        _ = img.height
        _ = img.data
        assert img._call_count == 1

    def test_save_calls_imwrite(self):
        data = np.zeros((2, 2, 3), dtype=np.uint8)
        img = _StubImage(data)
        with patch("fishsense_core.image.image.cv2.imwrite") as mock_imwrite:
            img.save("output.png")
            mock_imwrite.assert_called_once_with("output.png", data)

    def test_image_is_abstract(self):
        """Image cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Image()  # type: ignore[abstract]

    def test_to_jpeg_bytes_round_trip(self):
        """to_jpeg_bytes must produce bytes that decode to the original shape."""
        rng = np.random.default_rng(seed=42)
        data = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        img = _StubImage(data)

        jpeg = img.to_jpeg_bytes()

        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == data.shape

    def test_to_jpeg_bytes_quality_is_wired(self):
        """Lower quality must yield strictly smaller output on a non-trivial image."""
        rng = np.random.default_rng(seed=7)
        data = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        img = _StubImage(data)

        high_quality = img.to_jpeg_bytes(quality=95)
        low_quality = img.to_jpeg_bytes(quality=20)

        assert len(low_quality) < len(high_quality)

    def test_to_jpeg_bytes_default_quality(self):
        """Default quality must match an explicit quality=95 call."""
        rng = np.random.default_rng(seed=11)
        data = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        img = _StubImage(data)

        assert img.to_jpeg_bytes() == img.to_jpeg_bytes(quality=95)

    def test_to_jpeg_bytes_raises_on_failure(self):
        """If cv2.imencode reports failure, to_jpeg_bytes must raise RuntimeError."""
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        img = _StubImage(data)
        with patch(
            "fishsense_core.image.image.cv2.imencode",
            return_value=(False, np.array([], dtype=np.uint8)),
        ):
            with pytest.raises(RuntimeError):
                img.to_jpeg_bytes()


# ---------------------------------------------------------------------------
# RectifiedImage tests
# ---------------------------------------------------------------------------

class TestRectifiedImage:
    def _make_intrinsics(self):
        intrinsics = MagicMock()
        intrinsics.camera_matrix = np.eye(3, dtype=np.float64)
        intrinsics.distortion_coefficients = np.zeros(5, dtype=np.float64)
        return intrinsics

    def test_undistort_is_called(self):
        """RectifiedImage must delegate to cv2.undistort with the right args."""
        from fishsense_core.image.rectified_image import RectifiedImage

        data = np.ones((4, 6, 3), dtype=np.uint8) * 128
        source = _StubImage(data)
        intrinsics = self._make_intrinsics()

        with patch("fishsense_core.image.rectified_image.cv2.undistort", return_value=data) as mock_ud:
            rect = RectifiedImage(source, intrinsics)
            _ = rect.data  # trigger lazy load

            mock_ud.assert_called_once_with(
                data,
                intrinsics.camera_matrix,
                intrinsics.distortion_coefficients,
            )

    def test_output_shape_preserved(self):
        """The output shape must match what cv2.undistort returns."""
        from fishsense_core.image.rectified_image import RectifiedImage

        data = np.zeros((4, 6, 3), dtype=np.uint8)
        source = _StubImage(data)
        intrinsics = self._make_intrinsics()

        with patch(
            "fishsense_core.image.rectified_image.cv2.undistort",
            return_value=np.zeros((4, 6, 3), dtype=np.uint8),
        ):
            rect = RectifiedImage(source, intrinsics)
            assert rect.data.shape == (4, 6, 3)
            assert rect.width == 6
            assert rect.height == 4
