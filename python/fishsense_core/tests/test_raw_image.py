"""Unit tests for RawImage."""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fishsense_core.image.raw_image import RawImage


# ---------------------------------------------------------------------------
# Fake rawpy so we can drive RawImage without a real raw fixture
# ---------------------------------------------------------------------------

class _FakeRaw:
    """Stand-in for rawpy's RawPy context manager."""

    def __init__(self, postprocess_output: np.ndarray):
        self._postprocess_output = postprocess_output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def postprocess(self, **_kwargs) -> np.ndarray:
        return self._postprocess_output


def _make_imread(captured_bytes: list, postprocess_output: np.ndarray):
    """Build a rawpy.imread replacement that captures input and returns a fake raw."""

    def _imread(file_like):
        # Drain so we can verify the input matches what we expect.
        captured_bytes.append(file_like.read())
        return _FakeRaw(postprocess_output)

    return _imread


def _synthetic_postprocess_output() -> np.ndarray:
    """A 16-bit RGB image with a non-degenerate brightness distribution.

    The auto-gamma branch in RawImage takes log(mean) of the V channel, so we
    need the mean to be > 1 and != 5100 (mid * 255) to avoid degenerate gamma.
    """
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 65535, size=(32, 48, 3), dtype=np.uint16)


# ---------------------------------------------------------------------------
# Bytes-vs-path equivalence
# ---------------------------------------------------------------------------

class TestRawImageBytesAndPath:
    def test_bytes_constructor_data_matches_path_constructor_data(self, tmp_path: Path):
        """RawImage(bytes).data must equal RawImage(path).data when the bytes
        and the file's contents are identical."""
        raw_bytes = b"\x00\x01\x02\x03fake-raw-payload" * 64
        raw_path = tmp_path / "fixture.raw"
        raw_path.write_bytes(raw_bytes)

        captured: list[bytes] = []
        postprocess_output = _synthetic_postprocess_output()
        fake_imread = _make_imread(captured, postprocess_output)

        with patch("fishsense_core.image.raw_image.rawpy.imread", side_effect=fake_imread):
            data_from_path = RawImage(raw_path).data
            data_from_bytes = RawImage(raw_bytes).data

        # rawpy.imread received the same bytes via both paths.
        assert len(captured) == 2
        assert captured[0] == raw_bytes
        assert captured[1] == raw_bytes

        np.testing.assert_array_equal(data_from_path, data_from_bytes)

    def test_path_branch_passes_real_file_handle(self, tmp_path: Path):
        """The Path branch must open the file on disk, not wrap it in BytesIO."""
        raw_bytes = b"path-branch-payload" * 16
        raw_path = tmp_path / "fixture.raw"
        raw_path.write_bytes(raw_bytes)

        seen_types: list[type] = []
        postprocess_output = _synthetic_postprocess_output()

        def _imread(file_like):
            seen_types.append(type(file_like))
            file_like.read()
            return _FakeRaw(postprocess_output)

        with patch("fishsense_core.image.raw_image.rawpy.imread", side_effect=_imread):
            _ = RawImage(raw_path).data

        assert len(seen_types) == 1
        assert not issubclass(seen_types[0], io.BytesIO)

    def test_bytes_branch_passes_bytesio(self):
        """The bytes branch must wrap the input in BytesIO — no temp file."""
        raw_bytes = b"bytes-branch-payload" * 16
        seen_types: list[type] = []
        postprocess_output = _synthetic_postprocess_output()

        def _imread(file_like):
            seen_types.append(type(file_like))
            file_like.read()
            return _FakeRaw(postprocess_output)

        with patch("fishsense_core.image.raw_image.rawpy.imread", side_effect=_imread):
            _ = RawImage(raw_bytes).data

        assert seen_types == [io.BytesIO]

    def test_invalid_source_type_raises(self):
        """Constructing with an unsupported type should fail when data is loaded."""
        with pytest.raises((TypeError, AttributeError)):
            _ = RawImage(12345).data  # type: ignore[arg-type]
