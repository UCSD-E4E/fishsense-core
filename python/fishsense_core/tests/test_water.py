"""Tests for :mod:`fishsense_core.water`.

The attenuation fit and its inverse, exercised against synthetic water whose
coefficients are known — which is the only way to check a method whose real
validation (the published pure-water absorption spectrum) needs a corpus.
"""

import math

import numpy as np
import pytest

from fishsense_core.water.attenuation import (
    MIN_SAMPLES,
    as_polygon,
    fit_attenuation,
    fit_loglinear,
    sample_slate_patch,
)
from fishsense_core.water.seathru import estimate_veil, remove_water

#: Pool medians from the seven-dive slate fit, which land on Pope & Fry 1997.
POOL_BETA = (0.263, 0.040, 0.001)


def _slate_samples(beta=POOL_BETA, reflectance=(0.75, 0.80, 0.82), n=30, noise=0.0):
    """Frames of one dive: a constant-reflectance target at varying range."""
    rng = np.random.default_rng(5)
    ranges = np.linspace(0.6, 3.2, n)
    samples = []
    for z in ranges:
        row = {"range_m": float(z)}
        for key, rho, b in zip(("R", "G", "B"), reflectance, beta):
            value = rho * math.exp(-b * z)
            if noise:
                value *= math.exp(rng.normal(0.0, noise))
            row[key] = value
        samples.append(row)
    return samples


class TestLogLinearFit:
    def test_it_recovers_a_known_slope(self):
        z = list(np.linspace(0, 5, 20))
        fit = fit_loglinear(z, [3.0 - 0.4 * v for v in z])

        assert fit is not None
        assert fit.slope == pytest.approx(-0.4)
        assert fit.intercept == pytest.approx(3.0)
        assert fit.r2 == pytest.approx(1.0)
        assert not fit.spans_zero

    def test_pure_noise_gives_an_interval_that_spans_zero(self):
        """The property that lets the method say "no attenuation here" instead
        of inventing a correction for clear water."""
        rng = np.random.default_rng(0)
        z = list(np.linspace(0, 3, 40))
        fit = fit_loglinear(z, list(rng.normal(0.0, 1.0, 40)))

        assert fit is not None
        assert fit.spans_zero

    def test_too_few_points_is_none_not_a_number(self):
        z = list(range(MIN_SAMPLES - 1))
        assert fit_loglinear(z, z) is None

    def test_no_range_lever_is_none(self):
        """Attenuation is a slope against distance. Without a lever there is
        nothing to measure, and any number reported is noise dressed as a
        coefficient."""
        assert fit_loglinear([2.0] * 20, list(np.random.default_rng(1).normal(0, 1, 20))) is None

    def test_mismatched_lengths_are_none(self):
        assert fit_loglinear([1.0] * 10, [1.0] * 9) is None


class TestSlateSampling:
    def test_two_corners_become_a_rectangle(self):
        """`as_polygon` is re-exported here because this is where a reader
        looks for it; it is defined once, in `image.decode`."""
        assert as_polygon([[10.0, 20.0], [30.0, 50.0]]) == [
            (10.0, 20.0), (30.0, 20.0), (30.0, 50.0), (10.0, 50.0)
        ]

    def test_it_samples_the_paper_and_not_the_ink(self):
        image = np.zeros((128, 128, 3), dtype=np.float64)
        image[32:96, 32:96] = (0.5, 0.6, 0.7)  # BGR paper
        image[40:56, 32:96] = 0.02             # black markings

        sample = sample_slate_patch(image, [[32, 32], [95, 95]])

        assert sample is not None
        red, green, blue = sample
        assert (red, green, blue) == pytest.approx((0.7, 0.6, 0.5), rel=1e-6)

    def test_a_saturated_slate_carries_no_information_and_is_refused(self):
        """A blown-out slate sits at the container ceiling no matter how much
        water it was seen through; including it would flatten every slope
        toward zero."""
        image = np.full((128, 128, 3), 255, dtype=np.uint8)

        assert sample_slate_patch(image, [[32, 32], [95, 95]]) is None

    def test_a_degenerate_quad_is_refused_rather_than_sampled_as_a_streak(self):
        """Two identical x coordinates cannot bound an area; `cv2.fillPoly`
        would silently fill a one-pixel diagonal line several hundred pixels
        long, which clears a minimum-pixel guard while containing almost none
        of the slate."""
        image = np.full((128, 128, 3), 0.5)

        assert sample_slate_patch(image, [[32.0, 32.0], [32.0, 95.0]]) is None

    def test_a_quad_too_small_to_sample_is_refused(self):
        image = np.full((128, 128, 3), 0.5)

        assert sample_slate_patch(image, [[32, 32], [40, 40]]) is None


class TestAttenuationFit:
    def test_it_recovers_known_coefficients(self):
        fit = fit_attenuation(_slate_samples())

        assert fit is not None
        assert fit.beta == pytest.approx(POOL_BETA, abs=1e-9)
        assert fit.ordering_ok
        assert fit.trustworthy
        assert fit.n == 30
        assert fit.range_span == pytest.approx(2.6)

    def test_it_survives_realistic_scatter(self):
        fit = fit_attenuation(_slate_samples(noise=0.05))

        assert fit is not None
        assert fit.beta_r == pytest.approx(POOL_BETA[0], abs=0.05)
        assert fit.ordering_ok

    def test_the_differential_coefficients_match_the_absolute_ones(self):
        """`d_rg` is what the earlier annulus-around-the-dot method could
        measure, kept so the two are directly comparable."""
        fit = fit_attenuation(_slate_samples())

        assert fit is not None
        assert fit.d_rg == pytest.approx(fit.beta_r - fit.beta_g)
        assert fit.d_bg == pytest.approx(fit.beta_b - fit.beta_g)

    def test_clear_water_is_not_reported_as_trustworthy(self):
        """The negative control. Zero attenuation plus noise must not come back
        as a confident coefficient."""
        fit = fit_attenuation(_slate_samples(beta=(0.0, 0.0, 0.0), noise=0.05))

        assert fit is not None
        assert fit.beta_r_fit.spans_zero
        assert not fit.trustworthy

    def test_too_few_samples_is_none(self):
        assert fit_attenuation(_slate_samples(n=MIN_SAMPLES - 1)) is None

    def test_samples_without_a_range_are_dropped(self):
        samples = _slate_samples(n=12)
        for row in samples[:6]:
            row["range_m"] = None

        assert fit_attenuation(samples) is None

    def test_a_non_positive_channel_is_dropped_rather_than_logged(self):
        samples = _slate_samples(n=12)
        samples[0]["R"] = 0.0

        fit = fit_attenuation(samples)
        assert fit is not None
        assert fit.n == 11


class TestSeathru:
    @staticmethod
    def _observed(scene, beta, z, veil=(0.02, 0.10, 0.20)):
        """Forward model: what a camera at range z would record."""
        transmission = np.exp(-np.asarray(beta) * z)
        return scene * transmission + np.asarray(veil) * (1.0 - transmission)

    def test_it_inverts_the_forward_model_exactly(self):
        """Given the beta and veil that produced the frame, the recovery is the
        original scene. Nothing else in this module means anything if this
        does not hold."""
        rng = np.random.default_rng(2)
        scene = rng.uniform(0.05, 0.9, (16, 16, 3))
        veil = (0.02, 0.10, 0.20)
        observed = self._observed(scene, POOL_BETA, 2.5, veil)

        recovered = remove_water(observed, POOL_BETA, 2.5, veil=veil, normalize=False)

        np.testing.assert_allclose(recovered, scene, atol=1e-9)

    def test_it_lifts_red_relative_to_green(self):
        """Which is the whole point: red is absorbed fastest, so inverting the
        model gives it back the most."""
        scene = np.full((8, 8, 3), 0.4)
        observed = self._observed(scene, POOL_BETA, 3.0)

        before = observed[..., 0].mean() / observed[..., 1].mean()
        recovered = remove_water(observed, POOL_BETA, 3.0, normalize=False)
        after = recovered[..., 0].mean() / recovered[..., 1].mean()

        assert after > before

    def test_the_correction_grows_with_range(self):
        """At 1-1.5 m it is modest and most of the visible improvement is the
        tone treatment; the physics earns its place further out.

        Measured as how much red is lifted *relative to blue*, which is the
        cast. With no veil the correction is exactly
        ``exp((beta_r - beta_b) * z)``, so the claim is that the number is
        monotone in range and lands where the model says.
        """
        rng = np.random.default_rng(7)
        scene = rng.uniform(0.05, 0.9, (16, 16, 3))

        def red_over_blue_gain(z):
            observed = self._observed(scene, POOL_BETA, z, veil=(0.0, 0.0, 0.0))
            out = remove_water(
                observed, POOL_BETA, z, veil=(0.0, 0.0, 0.0), normalize=False
            )
            before = observed[..., 0].mean() / observed[..., 2].mean()
            return (out[..., 0].mean() / out[..., 2].mean()) / before

        assert red_over_blue_gain(5.0) > red_over_blue_gain(1.5) > red_over_blue_gain(0.5)
        assert red_over_blue_gain(5.0) == pytest.approx(
            math.exp((POOL_BETA[0] - POOL_BETA[2]) * 5.0), rel=1e-6
        )

        # And the recorded figure, on the field coefficients it was quoted for:
        # red about 2.7x green at 5 m, against 1.12x at 0.57 m and 1.35x at
        # 1.54 m. Those measurements pin a red-minus-green differential of
        # about 0.197 per metre, which this reproduces.
        differential = 0.197
        assert math.exp(differential * 0.57) == pytest.approx(1.12, rel=0.02)
        assert math.exp(differential * 1.54) == pytest.approx(1.35, rel=0.02)
        assert math.exp(differential * 5.0) == pytest.approx(2.7, rel=0.02)

    def test_zero_range_is_the_identity(self):
        rng = np.random.default_rng(3)
        scene = rng.uniform(0.0, 1.0, (8, 8, 3))

        np.testing.assert_allclose(
            remove_water(scene, POOL_BETA, 0.0, normalize=False), scene, atol=1e-12
        )

    def test_zero_beta_is_the_identity(self):
        rng = np.random.default_rng(4)
        scene = rng.uniform(0.0, 1.0, (8, 8, 3))

        np.testing.assert_allclose(
            remove_water(scene, (0.0, 0.0, 0.0), 4.0, normalize=False),
            scene,
            atol=1e-12,
        )

    def test_it_never_returns_negative_radiance(self):
        """Subtracting a veil larger than the signal happens in genuinely dark
        regions, and letting it go negative would invert those pixels once the
        result is renormalised."""
        dark = np.full((8, 8, 3), 0.001)
        out = remove_water(dark, POOL_BETA, 4.0, veil=(0.5, 0.5, 0.5))

        assert out.min() >= 0.0

    def test_a_negative_range_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            remove_water(np.zeros((4, 4, 3)), POOL_BETA, -1.0)

    def test_beta_must_be_three_coefficients(self):
        with pytest.raises(ValueError, match="three per-channel"):
            remove_water(np.zeros((4, 4, 3)), (0.2, 0.1), 1.0)

    def test_the_veil_estimate_reads_the_dark_tail(self):
        """Where reflectance is near zero the observation is essentially veil,
        which is the standard dark-channel argument. Low rather than zero, so
        one dead pixel does not set it."""
        scene = np.full((64, 64, 3), 0.6)
        scene[:8, :8] = (0.05, 0.12, 0.25)
        scene[0, 0] = 0.0  # a dead pixel that must not set the estimate

        veil = estimate_veil(scene)

        assert veil == pytest.approx((0.05, 0.12, 0.25), abs=1e-9)
