"""The decode chain, pinned byte-for-byte against real sensor data.

Every other test of :class:`RawImage` mocks ``rawpy`` with a seeded random
array, which exercises the plumbing (bytes vs. path, BytesIO vs. file handle)
and nothing about the decode itself. Swap ``equalize_adapthist`` for something
else and not one of them notices.

These do. They decode ``tests/fixtures/stage2_sample_crop.dng`` — a crop of a
real Olympus mosaic, see that directory's README — and hash the result. The
hash is not a substitute for understanding what changed; it is a tripwire that
makes a change to the decode *visible in a diff*, so that changing it is a
decision someone makes on purpose and re-pins.

**When one of these fails.** The failure message prints the hash that was
actually produced. Before re-pinning, work out which of these it is:

* You changed the decode. Re-pin, and say so in the commit message.
* ``uv.lock`` moved rawpy, scikit-image, OpenCV or NumPy. The chain is
  entirely their arithmetic, so a version bump can legitimately move the last
  bits. Re-pin, and note the version that moved.
* Neither — in which case something is wrong that these tests were written to
  catch, and the hash is the least interesting part of the report. Look at the
  channel means and the grain figure below first; they say *how* it moved.

The hashes were pinned on the versions in ``uv.lock`` at the time of writing:
rawpy 0.26.1, scikit-image 0.26.0, opencv-python-headless 5.0.0.93,
NumPy 2.4.4, x86_64 Linux.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from fishsense_core.image.linear_raw_image import LinearRawImage
from fishsense_core.image.raw_image import RawImage

FIXTURE = Path(__file__).parent / "fixtures" / "stage2_sample_crop.dng"

#: ``RawImage.data`` — uint8 BGR, the labeler-facing chain before rectification:
#: ``rawpy.postprocess`` -> auto-gamma to a mean V of 20 -> ``equalize_adapthist``.
GOLDEN_RAW_SHA256 = "e2636af517d5142adcd94dbdad5450fc4bb301905dbea675381fec74fb2c5078"

#: ``LinearRawImage.data`` — uint16 BGR, linear, sensor coordinates. The laser
#: detector's input. Nothing in the JPEG chain may move this; see
#: ``test_the_two_chains_are_independent``.
GOLDEN_LINEAR_SHA256 = "2f8faac9cd68e21d9d53e52326e8ac96d9744c37346508846b353068396b35a4"

#: ``LinearRawImage.bayer_excess`` — read off the undemosaiced mosaic, so it
#: depends on no decode setting at all.
GOLDEN_BAYER_EXCESS_SHA256 = (
    "6936f72c24ab70e993e55c3368c06891eab5ffd3d47f279cb4a5485a22e11442"
)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _assert_pinned(array: np.ndarray, expected: str, what: str) -> None:
    actual = _sha256(array)
    assert actual == expected, (
        f"{what} changed.\n"
        f"  expected {expected}\n"
        f"  actual   {actual}\n"
        "If this was deliberate, re-pin the constant in this file and say why "
        "in the commit message. If it was not, see this module's docstring."
    )


def _open_water(image: np.ndarray) -> tuple[slice, slice]:
    """The top-left fifth, which on this frame is open water.

    The same convention every noise figure in the enhancement evaluation uses,
    so numbers taken here line up with the ones recorded there.
    """
    height, width = image.shape[:2]
    return np.s_[: height // 5], np.s_[: width // 5]


def _grain(patch: np.ndarray) -> float:
    """High-frequency residual after a 3x3 median.

    A plain standard deviation would count the water's illumination gradient,
    which is not noise and which no decode setting removes.
    """
    plane = np.asarray(patch, dtype=np.float32)
    return float(np.std(plane - cv2.medianBlur(plane, 3)))


class TestFixture:
    def test_fixture_is_present(self):
        assert FIXTURE.is_file(), (
            f"missing decode fixture at {FIXTURE}; it is checked in, so this "
            "means a partial checkout rather than an environment problem"
        )

    def test_metadata_matches_the_source_sensor(self):
        """The crop carries the source .ORF's sensor description.

        These five properties are the whole of what ``postprocess`` reads
        before it starts interpolating, so if they survived the rewrap the
        decode is working on the same numbers the original frame gave it. A
        silent change here — the Bayer phase in particular — would move every
        pixel's colour while leaving the file perfectly readable.
        """
        import rawpy  # noqa: PLC0415 — kept local so a collect-only run is cheap

        with rawpy.imread(str(FIXTURE)) as raw:
            assert np.asarray(raw.raw_pattern).tolist() == [[1, 0], [2, 3]]
            assert raw.color_desc == b"RGBG"
            assert [int(b) for b in raw.black_level_per_channel] == [266, 266, 266, 267]
            assert int(raw.white_level) == 4095
            # The camera's topside daylight preset, carried through the DNG's
            # AsShotNeutral as a rational, hence the tolerance.
            red, green, blue, _ = raw.camera_whitebalance
            assert green == pytest.approx(1.0)
            assert red == pytest.approx(2.9375, rel=1e-5)
            assert blue == pytest.approx(1.7578125, rel=1e-5)


class TestRawImageGolden:
    def test_decode_is_unchanged(self):
        data = RawImage(FIXTURE).data

        assert data.shape == (480, 640, 3)
        assert data.dtype == np.uint8
        _assert_pinned(data, GOLDEN_RAW_SHA256, "RawImage.data")

    def test_decode_is_deterministic(self):
        """Two decodes of the same bytes agree.

        Not obvious enough to skip: ``equalize_adapthist`` interpolates
        between tile histograms and OpenCV's conversions are multithreaded,
        so a golden hash is only meaningful once repeatability is established.
        """
        np.testing.assert_array_equal(RawImage(FIXTURE).data, RawImage(FIXTURE).data)

    def test_bytes_and_path_decode_identically_on_a_real_frame(self):
        """The mocked version of this test cannot see a decode difference —
        it compares two calls that both return the same canned array."""
        np.testing.assert_array_equal(
            RawImage(FIXTURE).data, RawImage(FIXTURE.read_bytes()).data
        )

    def test_channel_means_are_pinned(self):
        """What the hash cannot tell you: which way it moved.

        Blue at 142 against red at 21 is the cyan cast — a topside daylight
        white balance applied through several metres of water. Red carries
        almost no signal, which is the fact every later decision turns on.
        """
        data = RawImage(FIXTURE).data
        blue, green, red = data.reshape(-1, 3).mean(axis=0)

        assert blue == pytest.approx(142.3458, rel=1e-4)
        assert green == pytest.approx(87.7722, rel=1e-4)
        assert red == pytest.approx(21.1809, rel=1e-4)

    def test_clahe_leaves_the_open_water_speckled(self):
        """The measured state of the current default, pinned before it changes.

        ``skimage.exposure.equalize_adapthist`` has no RGB branch: it reads an
        H×W×3 frame as a 3-D volume and defaults ``kernel_size`` to
        ``(H//8, W//8, max(3//8, 1))``. That trailing 1 is a channel-axis tile
        depth of one, so each colour channel is equalised independently, and
        near-uniform open water has its noise stretched to full scale — red
        hardest, since red has the narrowest histogram.

        Both figures below are that behaviour, measured on this frame's open
        water. They are here so that a change to the decode has to state what
        it did to them.
        """
        data = RawImage(FIXTURE).data
        rows, cols = _open_water(data)

        luma = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        assert _grain(luma[rows, cols]) == pytest.approx(14.5089, rel=1e-3)

        # Blue-minus-red over a patch of flat water: chroma speckle, not scene
        # content. Per-channel equalisation is what decorrelates the channels
        # enough to produce it.
        chroma = data[..., 0].astype(np.float64) - data[..., 2].astype(np.float64)
        assert float(np.std(chroma[rows, cols])) == pytest.approx(39.2972, rel=1e-3)


class TestLinearRawImageGolden:
    def test_decode_is_unchanged(self):
        image = LinearRawImage(FIXTURE)

        assert image.data.shape == (480, 640, 3)
        assert image.data.dtype == np.uint16
        _assert_pinned(image.data, GOLDEN_LINEAR_SHA256, "LinearRawImage.data")

    def test_bayer_excess_is_unchanged(self):
        excess = LinearRawImage(FIXTURE).bayer_excess

        assert excess is not None
        assert excess.shape == (480, 640, 2)
        assert excess.dtype == np.uint16
        _assert_pinned(
            excess, GOLDEN_BAYER_EXCESS_SHA256, "LinearRawImage.bayer_excess"
        )

    def test_the_two_chains_are_independent(self):
        """`RawImage` and `LinearRawImage` share only ``rawpy.postprocess``.

        This is the guarantee that keeps laser detection — a working,
        calibrated path — from taking a dependency on the decode work, and it
        is currently free only because enhancement lives downstream of here.
        Free-unless-somebody-moves-it is not a guarantee, so it is asserted:
        the linear decode is uint16 in sensor coordinates and the JPEG decode
        is uint8, and neither is derived from the other.
        """
        raw = RawImage(FIXTURE).data
        linear = LinearRawImage(FIXTURE).data

        assert raw.dtype == np.uint8 and linear.dtype == np.uint16
        assert raw.shape == linear.shape
        # Not a tautology: it fails the moment somebody makes one a rescale of
        # the other, which is exactly how such a dependency gets introduced.
        assert not np.array_equal(raw.astype(np.uint16) * 257, linear)
