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
  channel means and the water-noise figures below first; they say *how* it
  moved.

The hashes were pinned on the versions in ``uv.lock`` at the time of writing:
rawpy 0.26.1, scikit-image 0.26.0, opencv-python-headless 5.0.0.93,
NumPy 2.4.4, x86_64 Linux.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from fishsense_core.image.decode import (
    DecodeConfig,
    WhiteBalance,
    decode_linear_stage,
)
from fishsense_core.image.linear_raw_image import LinearRawImage
from fishsense_core.image.raw_image import RawImage

FIXTURE = Path(__file__).parent / "fixtures" / "stage2_sample_crop.dng"

#: ``RawImage.data`` at the **default** decode — uint8 BGR, the labeler-facing
#: chain before rectification: ``rawpy.postprocess`` -> auto-gamma to a mean V
#: of 20 -> global CIELAB L* percentile stretch.
GOLDEN_RAW_SHA256 = "fc4f8cd15a870d8435eb0e753af4ddb0f7265a533c3b8ddf3cb933a051f27f9c"

#: ``RawImage.data`` at ``DecodeConfig.production()`` — the chain that was
#: hard-coded before ``decode.py`` existed, ending in ``equalize_adapthist``
#: at skimage's defaults.
#:
#: **This hash is unchanged from the commit that introduced it.** That is the
#: point of keeping it: the decode was parameterised and its default flipped in
#: the same change, and this constant is the evidence that the parameterisation
#: was faithful rather than approximately faithful. If it ever has to move,
#: the refactor is what changed, not the default.
GOLDEN_PRODUCTION_SHA256 = (
    "e2636af517d5142adcd94dbdad5450fc4bb301905dbea675381fec74fb2c5078"
)

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


def _high_pass(patch: np.ndarray) -> np.ndarray:
    """Residual after a 3x3 median — the grain, without the scene.

    A plain standard deviation would count the water's illumination gradient,
    which is not noise and which no decode setting removes.
    """
    plane = np.asarray(patch, dtype=np.float32)
    return plane - cv2.medianBlur(plane, 3)


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

    def test_the_production_decode_is_still_reproducible(self):
        """`DecodeConfig.production()` is bit-identical to the old hard-coded
        chain. Parameterising a decode is only safe if you can show the
        parameterisation did not itself change anything."""
        data = RawImage(FIXTURE, config=DecodeConfig.production()).data

        assert data.shape == (480, 640, 3)
        assert data.dtype == np.uint8
        _assert_pinned(
            data, GOLDEN_PRODUCTION_SHA256, "RawImage.data at DecodeConfig.production()"
        )

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

        Blue far above red is the cyan cast — a topside daylight white balance
        applied through several metres of water. Red carries almost no signal,
        which is the fact every later decision turns on, and neither decode
        pretends otherwise: the stretch expands contrast without touching the
        cast, because removing it needs range-based physics rather than
        another global gain.
        """
        blue, green, red = RawImage(FIXTURE).data.reshape(-1, 3).mean(axis=0)
        assert blue == pytest.approx(215.5858, rel=1e-4)
        assert green == pytest.approx(160.1348, rel=1e-4)
        assert red == pytest.approx(112.0674, rel=1e-4)

        blue, green, red = (
            RawImage(FIXTURE, config=DecodeConfig.production())
            .data.reshape(-1, 3)
            .mean(axis=0)
        )
        assert blue == pytest.approx(142.3458, rel=1e-4)
        assert green == pytest.approx(87.7722, rel=1e-4)
        assert red == pytest.approx(21.1809, rel=1e-4)


class TestTheDefaultChanged:
    """The measured difference between the two chains, on this frame.

    ``equalize_adapthist`` is decorated ``@adapt_rgb(hsv_value)``, so on an
    H×W×3 frame it converts to HSV, equalises **V**, and converts back with
    hue and saturation restored untouched. Saturation is a *ratio*, so holding
    it fixed while amplifying V amplifies chroma noise in exact step with luma
    noise, and near-uniform open water — where the only local variation is
    noise — comes out as coloured speckle.

    A CIELAB L* map holds a* and b* fixed in *absolute* terms instead, so the
    same amplification lifts luma contrast and leaves the chroma noise where
    it was. Same water, grey grain rather than colour.

    That is a colour-space difference, not a local-vs-global one, and the
    control below says so: a *global* 4× expansion of V, through the same HSV
    path, decorrelates the water exactly as CLAHE does.

    (This is not the mechanism the proposal gave. It said skimage had no RGB
    branch and equalised each channel independently, handing red the largest
    gain. It does not — see ``test_decode.py`` for the direct check. The
    recommendation is unaffected; it rested on measured outputs.)

    These are figures from one 640×480 pool frame. The corpus-scale numbers —
    36.7× water-noise amplification, separation 0.259 → 0.397, and every
    paired model statistic spanning zero — were measured elsewhere and are
    cited in ``fishsense_core.image.decode``.
    """

    @staticmethod
    def _cross_channel_noise_correlation(data: np.ndarray) -> float:
        """How correlated the open water's grain is between red and green.

        High means grey grain — one noise source seen through three channels.
        Low means coloured speckle, which is the harder thing to look past.
        """
        rows, cols = _open_water(data)
        red, green = (
            _high_pass(data[..., channel][rows, cols]).ravel() for channel in (2, 1)
        )
        return float(np.corrcoef(red, green)[0, 1])

    def test_the_two_chains_disagree(self):
        assert not np.array_equal(
            RawImage(FIXTURE).data,
            RawImage(FIXTURE, config=DecodeConfig.production()).data,
        )

    def test_both_chains_amplify_the_water_noise_by_a_similar_factor(self):
        """The trade is not noise-for-contrast — they cost about the same.

        Pinned so that nobody reads the change as "we stopped amplifying
        noise". We did not; we changed what colour the amplified noise is.
        """
        production = RawImage(FIXTURE, config=DecodeConfig.production()).data
        default = RawImage(FIXTURE).data
        rows, cols = _open_water(default)

        def luma_grain(data):
            plane = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            return float(np.std(_high_pass(plane[rows, cols])))

        assert luma_grain(production) == pytest.approx(14.5089, rel=1e-3)
        assert luma_grain(default) == pytest.approx(20.0780, rel=1e-3)

    def test_the_hsv_path_leaves_the_water_noise_coloured(self):
        """The mechanism, measured: what CLAHE amplifies stays chromatic."""
        assert self._cross_channel_noise_correlation(
            RawImage(FIXTURE, config=DecodeConfig.production()).data
        ) == pytest.approx(0.5049, abs=5e-4)

        assert self._cross_channel_noise_correlation(
            RawImage(FIXTURE).data
        ) == pytest.approx(0.8984, abs=5e-4)

    def test_it_is_the_colour_space_and_not_the_locality(self):
        """The control that settles which explanation is right.

        Two *global* contrast expansions of the same size, one through HSV V
        and one through CIELAB L*. If locality were the cause they would land
        together; they land with their respective chains instead.
        """
        from skimage.color import hsv2rgb, lab2rgb, rgb2hsv, rgb2lab  # noqa: PLC0415
        from skimage.util import img_as_float, img_as_ubyte  # noqa: PLC0415

        from fishsense_core.image.decode import (  # noqa: PLC0415
            _decode_rgb16,
            auto_gamma,
        )

        pre_tone = auto_gamma(img_as_float(_decode_rgb16(FIXTURE, DecodeConfig(), None)), 20)

        def expanded(convert_to, convert_from, index, full_scale):
            space = convert_to(pre_tone)
            plane = space[..., index]
            space[..., index] = np.clip(
                (plane - plane.mean()) * 4.0 + plane.mean(), 0.0, full_scale
            )
            return img_as_ubyte(np.clip(convert_from(space), 0.0, 1.0)[:, :, ::-1])

        through_hsv = self._cross_channel_noise_correlation(
            expanded(rgb2hsv, hsv2rgb, 2, 1.0)
        )
        through_lab = self._cross_channel_noise_correlation(
            expanded(rgb2lab, lab2rgb, 0, 100.0)
        )

        assert through_hsv == pytest.approx(0.48, abs=0.03)
        assert through_lab == pytest.approx(0.89, abs=0.03)
        assert through_lab - through_hsv > 0.3

    def test_the_global_stretch_halves_the_chroma_speckle(self):
        """Blue-minus-red over flat water: speckle, not scene content."""

        def chroma_spread(data: np.ndarray) -> float:
            rows, cols = _open_water(data)
            difference = data[..., 0].astype(np.float64) - data[..., 2].astype(np.float64)
            return float(np.std(difference[rows, cols]))

        production = chroma_spread(
            RawImage(FIXTURE, config=DecodeConfig.production()).data
        )
        default = chroma_spread(RawImage(FIXTURE).data)

        assert production == pytest.approx(39.2972, rel=1e-3)
        assert default == pytest.approx(22.2533, rel=1e-3)
        assert default < production

    def test_dropping_clahe_is_the_faster_chain(self):
        """CLAHE is the expensive step, not an incidental one.

        Asserted as an operation count rather than a wall clock: on a
        3016×4014 frame ``equalize_adapthist`` measures 10.2 s of a 13.9 s
        decode, but a 640×480 fixture on a shared CI runner cannot time that
        honestly. What it *can* pin is that the default no longer calls it.
        """
        calls = []
        with patch(
            "fishsense_core.image.decode.equalize_adapthist",
            side_effect=lambda img, **kw: calls.append(kw) or img,
        ):
            RawImage(FIXTURE).data
            assert calls == []

            RawImage(FIXTURE, config=DecodeConfig.production()).data
            assert len(calls) == 1
            # No kwargs: skimage's own defaults, including the channel-axis
            # tile depth of one that this whole change is about.
            assert calls[0] == {}


class TestWhiteBalanceEstimation:
    """The estimators, run end to end against the fixture.

    None of these is recommended — every variant that gives red an independent
    gain from the bottom of its range failed on field frames. They are tested
    because they are reachable, and because the one claim made *about* them
    turned out not to hold.
    """

    QUAD = [[200, 150], [440, 330]]

    def _gains(self, white_balance, **kwargs):
        from fishsense_core.image.decode import (  # noqa: PLC0415
            resolve_white_balance,
        )

        config = DecodeConfig(white_balance=white_balance, **kwargs)
        return resolve_white_balance(FIXTURE, config, slate_quad=self.QUAD)

    def test_camera_and_auto_need_no_gains(self):
        assert self._gains(WhiteBalance.CAMERA) is None
        assert self._gains(WhiteBalance.RAWPY_AUTO) is None

    @pytest.mark.parametrize(
        "white_balance",
        [WhiteBalance.GRAY_WORLD, WhiteBalance.WHITE_PATCH, WhiteBalance.SLATE],
    )
    def test_each_estimator_composes_onto_the_camera_multipliers(self, white_balance):
        """Four multipliers in rawpy's [R, G1, B, G2] order, green pinned.

        Composing onto the camera's own multipliers rather than replacing them
        is what holds the colour matrix, demosaic and black levels fixed, so
        the difference between two configs is the white point and nothing else.
        """
        gains = self._gains(white_balance)

        assert gains is not None and len(gains) == 4
        assert all(g > 0 for g in gains)
        # G1 and G2 track each other; this sensor reports no separate G2.
        assert gains[1] == pytest.approx(gains[3])
        # Every estimator here pushes red up hard, which is the behaviour that
        # made them fail on field frames.
        assert gains[0] > 2.9375  # the camera's own red multiplier

    def test_slate_without_a_quad_says_what_is_missing(self):
        """The quad is per-image, so it cannot live on the config; the failure
        has to name what the caller forgot to pass."""
        from fishsense_core.image.decode import (  # noqa: PLC0415
            resolve_white_balance,
        )

        config = DecodeConfig(white_balance=WhiteBalance.SLATE)
        with pytest.raises(ValueError, match="slate_rectangle"):
            resolve_white_balance(FIXTURE, config)

    @pytest.mark.parametrize(
        ("white_balance", "half", "full"),
        [
            (WhiteBalance.GRAY_WORLD, 19.51, 22.62),
            (WhiteBalance.WHITE_PATCH, 5.69, 6.58),
            (WhiteBalance.SLATE, 5.78, 6.83),
        ],
    )
    def test_half_size_estimation_is_not_free(self, white_balance, half, full):
        """`wb_estimate_half_size` was defaulted on, justified as moving the
        gains "negligibly" because they are a global statistic.

        They are not. ``half_size`` skips the demosaic and averages each 2x2
        Bayer cell instead, which changes the noise floor of exactly the
        channel — red — that sits on it. Measured here at 14-15% on the red
        multiplier for all three estimators, which is why the default is now
        off. Pinned so that turning it back on has to argue with a number.
        """
        assert self._gains(white_balance, wb_estimate_half_size=True)[0] == (
            pytest.approx(half, rel=0.01)
        )
        assert self._gains(white_balance, wb_estimate_half_size=False)[0] == (
            pytest.approx(full, rel=0.01)
        )

    def test_the_default_is_the_full_resolution_estimate(self):
        assert DecodeConfig().wb_estimate_half_size is False
        np.testing.assert_allclose(
            self._gains(WhiteBalance.GRAY_WORLD),
            self._gains(WhiteBalance.GRAY_WORLD, wb_estimate_half_size=False),
        )

    def test_an_estimated_white_balance_changes_the_pixels(self):
        """Guards against the experiment silently being a no-op. A `user_wb`
        that failed to take effect would report "white balance does not
        matter", which is exactly the wrong conclusion to reach by accident.
        """
        from fishsense_core.image.decode import (  # noqa: PLC0415
            decode_rectified_stage,
        )

        baseline = decode_rectified_stage(FIXTURE, DecodeConfig())
        gray_world = decode_rectified_stage(
            FIXTURE, DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD)
        )
        assert not np.array_equal(baseline, gray_world)


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

    def test_the_jpeg_chain_cannot_reach_the_linear_decode(self):
        """Every tone knob, moved together, leaves the linear decode identical.

        Not a small effect needing a large sample to detect — exactly zero.
        This is the property that lets a decode change ship without re-running
        laser calibration, so it is asserted rather than reasoned about.
        """
        baseline = decode_linear_stage(FIXTURE, DecodeConfig())
        moved = decode_linear_stage(
            FIXTURE,
            DecodeConfig(
                auto_gamma_target=60,
                stretch_mode="per_channel",
                stretch_low=5.0,
                stretch_high=95.0,
                clahe_enabled=True,
                clahe_clip_limit=0.003,
                red_boost=0.4,
            ),
        )
        np.testing.assert_array_equal(baseline, moved)

    def test_white_balance_is_the_one_knob_that_does_reach_it(self):
        """The converse, and the reason white balance is treated differently.

        It lives inside ``rawpy.postprocess``, so it moves the laser
        detector's input as well as the labeler's JPEG — and the checkpoint
        was trained on ``use_camera_wb=True``, making a change here a
        systematic covariate shift rather than a mild perturbation. Nothing
        production calls can request it; see the next test.
        """
        baseline = decode_linear_stage(FIXTURE, DecodeConfig())
        gray_world = decode_linear_stage(
            FIXTURE, DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD)
        )
        assert not np.array_equal(baseline, gray_world)

    def test_linear_raw_image_exposes_no_decode_knob(self):
        """`LinearRawImage` is what production calls, and it takes no config.

        ``decode_linear_stage`` accepts one so the asymmetry above is
        testable. Letting the class accept one is how the laser path would
        quietly acquire a dependency on decode work done for the JPEG, so it
        does not.
        """
        import inspect  # noqa: PLC0415

        parameters = inspect.signature(LinearRawImage.__init__).parameters
        assert "config" not in parameters
        assert set(parameters) == {"self", "source", "bayer_upsample"}
