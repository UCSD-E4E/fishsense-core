"""The enhancement contract: what an enhancer may do to an array.

A shipped enhancer is strictly pixel-wise, because every measurement in this
pipeline is a pixel *coordinate*. That is mechanically checkable, so these
tests check the checker — including the two cases that motivated it, both of
which cost real time to diagnose the first time.
"""

import cv2
import numpy as np
import pytest

from fishsense_core.image.contract import (
    IDENTITY,
    GeometryViolation,
    guard,
    probe_geometry,
)


def _frame(height: int = 8, width: int = 6) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


class TestGuard:
    def test_a_value_only_enhancer_passes(self):
        guarded = guard(lambda a: (a // 2).astype(a.dtype), name="halve")
        source = _frame()
        out = guarded(source)

        assert out.shape == source.shape
        assert out.dtype == source.dtype

    def test_a_neighbourhood_filter_passes(self):
        """Value locality is deliberately allowed: CLAHE reads a
        neighbourhood, and CLAHE is a legitimate operator."""
        guarded = guard(lambda a: cv2.GaussianBlur(a, (3, 3), 0), name="blur")

        assert guarded(_frame()).shape == (8, 6, 3)

    @pytest.mark.parametrize(
        ("name", "enhancer"),
        [
            ("resize", lambda a: cv2.resize(a, (3, 4))),
            ("crop", lambda a: a[1:, 1:]),
            ("transpose", lambda a: np.swapaxes(a, 0, 1)),
            ("drop-channel", lambda a: a[..., :2]),
        ],
    )
    def test_a_geometry_change_is_refused(self, name, enhancer):
        with pytest.raises(GeometryViolation, match="shape"):
            guard(enhancer, name=name)(_frame())

    def test_a_dtype_change_is_refused(self):
        """The container's full scale is load-bearing: a chromaticity
        normalisation picks 255 vs 65535 off it, and cv2.imencode needs
        uint8."""
        with pytest.raises(GeometryViolation, match="dtype"):
            guard(lambda a: a.astype(np.uint16), name="widen")(_frame())

    def test_returning_something_other_than_an_array_is_refused(self):
        with pytest.raises(GeometryViolation, match="not an ndarray"):
            guard(lambda a: a.tolist(), name="listify")(_frame())

    def test_an_in_place_write_is_refused_with_the_rule_it_broke(self):
        """numpy's own message is a bare ValueError from inside the enhancer;
        what a reader needs to know is that this corrupts the control arm."""

        def mutate(image):
            image[0, 0] = 0
            return image

        with pytest.raises(GeometryViolation, match="in place"):
            guard(mutate, name="mutate")(_frame())

    def test_an_unrelated_value_error_is_not_swallowed(self):
        def broken(_image):
            raise ValueError("something else entirely")

        with pytest.raises(ValueError, match="something else entirely"):
            guard(broken, name="broken")(_frame())

    def test_the_input_is_never_modified(self):
        source = _frame()
        original = source.copy()
        guard(lambda a: a + 1, name="brighten")(source)

        np.testing.assert_array_equal(source, original)

    def test_the_identity_returns_a_writeable_array(self):
        """This is not tidiness — it is a segfault.

        The read-only view is what lets `guard` catch an in-place enhancer,
        but IDENTITY returns that same view, so every baseline arm would pass
        a read-only frame downstream. OpenCV ignores numpy's WRITEABLE flag
        and writes through anyway; the process dumps core with no traceback.
        """
        out = guard(IDENTITY, name="identity")(_frame())

        assert out.flags.writeable
        # And it does not alias the caller's buffer either.
        source = _frame()
        result = guard(IDENTITY, name="identity")(source)
        result[0, 0] = 255
        assert source[0, 0, 0] == 0


class TestProbeGeometry:
    """`guard` compares shape and dtype, which structurally cannot catch the
    three cases that matter most — a square-frame transpose, a one-pixel roll,
    and a resample round trip — all of which preserve both exactly while moving
    labelled coordinates."""

    def test_the_identity_moves_nothing(self):
        result = probe_geometry(IDENTITY, name="identity")

        assert result.displacement_px < 0.1
        assert not result.suspicious

    @pytest.mark.parametrize("sigma", [1.0, 6.0])
    def test_a_symmetric_blur_moves_nothing_however_wide(self, sigma):
        """A symmetric neighbourhood filter preserves the response centroid at
        any width, which is why CLAHE passes and a one-pixel roll does not."""
        blurred = probe_geometry(
            lambda a: cv2.GaussianBlur(a, (0, 0), sigma), name=f"blur{sigma}"
        )

        assert blurred.displacement_px < 0.1

    def test_a_value_only_enhancer_moves_nothing(self):
        assert (
            probe_geometry(
                lambda a: (a // 2).astype(a.dtype), name="halve"
            ).displacement_px
            < 0.1
        )

    def test_a_global_percentile_stretch_moves_nothing(self):
        """The reason the probe swaps two patches rather than brightening one:
        a global operator derives its mapping from the frame's histogram, and
        a swap leaves that histogram bit-identical, so the identical mapping is
        applied to both frames."""

        def stretch(image):
            plane = image.astype(np.float64)
            low, high = np.percentile(plane, 1), np.percentile(plane, 99)
            return np.clip((plane - low) / (high - low) * 255, 0, 255).astype(np.uint8)

        assert probe_geometry(stretch, name="stretch").displacement_px < 0.5

    def test_a_one_pixel_roll_is_refused(self):
        """The case a shape check cannot see, and the one that matters: 1 px of
        laser-dot displacement is 0.75% length error."""
        with pytest.raises(GeometryViolation, match="moved pixels"):
            probe_geometry(lambda a: np.roll(a, 1, axis=1), name="roll")

    def test_a_square_transpose_is_refused(self):
        with pytest.raises(GeometryViolation, match="moved pixels"):
            probe_geometry(lambda a: np.swapaxes(a, 0, 1), name="transpose")

    def test_a_horizontal_flip_is_refused(self):
        with pytest.raises(GeometryViolation, match="moved pixels"):
            probe_geometry(lambda a: a[:, ::-1].copy(), name="flip")

    def test_a_resample_round_trip_is_flagged_but_allowed(self):
        """A downsample-and-back blurs without displacing. That is a loss of
        detail rather than a displacement of it, so it is surfaced rather than
        refused — and if the detail matters, it shows up in the consumer's own
        numbers."""

        def wobble(image):
            height, width = image.shape[:2]
            small = cv2.resize(image, (width // 3, height // 3))
            return cv2.resize(small, (width, height))

        result = probe_geometry(wobble, name="resample")

        assert result.displacement_px < 0.5
        assert result.suspicious

    def test_the_measured_displacement_is_reported_not_hardcoded(self):
        """Otherwise `displacement_px` is decoration, the warning band never
        fires, and every enhancer reports a reassuring 0.000 whatever it did.

        Enough of a shift to be measured but not enough to refuse: the probe
        has to be able to report a number strictly between the two limits.
        """

        def half_pixel_shift(image):
            # An asymmetric 2-tap filter: a 0.25 px shift along x, no resample.
            kernel = np.array([[0.0, 0.0, 0.0], [0.0, 0.75, 0.25], [0.0, 0.0, 0.0]])
            return cv2.filter2D(image, -1, kernel)

        result = probe_geometry(half_pixel_shift, name="quarter-shift")

        assert 0.1 < result.displacement_px < 0.5
        assert result.suspicious

    def test_an_enhancer_that_ignores_its_input_is_not_this_probes_business(self):
        constant = np.zeros((384, 384, 3), dtype=np.uint8)
        result = probe_geometry(lambda _a: constant.copy(), name="constant")

        assert result.displacement_px == 0.0
        assert not result.suspicious
