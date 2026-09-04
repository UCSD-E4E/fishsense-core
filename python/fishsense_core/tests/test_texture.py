"""Tests for :mod:`fishsense_core.image.texture`.

The metric exists because the obvious one could not do the job: a
high-frequency residual on the fish cannot distinguish a denoiser that erased
the grain from one that erased the scales too. So the thing to test is exactly
that discrimination — a filter that removes noise must score near 1.0 and a
filter that removes the lattice must score near 0.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from fishsense_core.image.texture import (
    SCALE_PERIOD_MAX_PX,
    SCALE_PERIOD_MIN_PX,
    SIGNIFICANCE,
    assess,
    grain_reduction,
    radial_power_spectrum,
    spectral_envelope,
    texture_power,
    texture_retention,
)

#: The period of the real lattice this was validated against.
LATTICE_PX = 4.5


def _scales(size=192, period=LATTICE_PX, amplitude=12.0, noise=3.0, seed=0):
    """A patch carrying a crossed sinusoidal lattice on a noise floor."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    lattice = amplitude * (
        np.sin(2 * np.pi * xx / period) + np.sin(2 * np.pi * yy / period)
    )
    return 90.0 + lattice + rng.normal(0.0, noise, (size, size))


def _water(size=192, noise=3.0, seed=1):
    """Open water: the same noise floor, no lattice."""
    rng = np.random.default_rng(seed)
    return 40.0 + rng.normal(0.0, noise, (size, size))


class TestRadialPowerSpectrum:
    def test_it_finds_the_lattice_frequency(self):
        freqs, power = radial_power_spectrum(_scales(noise=0.0))
        peak = freqs[int(np.argmax(power[1:])) + 1]

        assert 1.0 / peak == pytest.approx(LATTICE_PX, rel=0.15)

    def test_power_is_independent_of_patch_size(self):
        """Without normalising by the window's energy, ``|FFT|²`` grows with
        pixel count — a 320 px water patch read 4x above a 200 px fish patch at
        the same noise level. Every synthetic patch in the tests that missed it
        had been 128x128.
        """
        rng = np.random.default_rng(2)

        def mean_power(size):
            _, power = radial_power_spectrum(rng.normal(0.0, 5.0, (size, size)))
            return float(power[2:-2].mean())

        assert mean_power(320) == pytest.approx(mean_power(200), rel=0.1)

    def test_white_noise_reads_its_own_variance(self):
        rng = np.random.default_rng(3)
        _, power = radial_power_spectrum(rng.normal(0.0, 7.0, (256, 256)))

        assert float(power[2:-2].mean()) == pytest.approx(49.0, rel=0.15)

    def test_counts_are_returned_on_request(self):
        freqs, power, counts = radial_power_spectrum(_water(), with_counts=True)

        assert len(freqs) == len(power) == len(counts)
        assert counts.sum() == 192 * 192


class TestSpectralEnvelope:
    def test_it_passes_under_a_narrow_peak(self):
        """Which is what makes the peak's prominence measurable at all."""
        _, power = radial_power_spectrum(_scales())
        envelope = spectral_envelope(power)

        assert envelope.max() < power.max()

    def test_it_tracks_a_smooth_spectrum(self):
        _, power = radial_power_spectrum(_water())
        envelope = spectral_envelope(power)

        band = slice(3, -3)
        np.testing.assert_allclose(envelope[band], power[band], rtol=1.5)


class TestTexturePower:
    def test_a_lattice_is_many_sigma_above_the_estimate_noise(self):
        prominence, noise = texture_power(_scales())

        assert prominence > SIGNIFICANCE * noise

    def test_open_water_carries_no_measurable_peak(self):
        prominence, noise = texture_power(_water())

        assert prominence < SIGNIFICANCE * noise

    def test_the_band_excludes_grain_and_body_shading(self):
        """Below 3 px is the demosaic's own limit and nearly all grain; above
        24 px is body shading rather than scales."""
        assert SCALE_PERIOD_MIN_PX == 3.0
        assert SCALE_PERIOD_MAX_PX == 24.0

        too_fine, _ = texture_power(_scales(period=2.2))
        in_band, _ = texture_power(_scales(period=8.0))
        too_coarse, _ = texture_power(_scales(period=48.0))

        assert in_band > too_fine
        assert in_band > too_coarse


class TestTextureRetention:
    def test_the_identity_retains_everything(self):
        patch = _scales()
        assert texture_retention(patch, patch) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("sigma", "expected"),
        [(0.5, 0.69), (1.0, 0.14)],
    )
    def test_a_blur_grades_rather_than_collapsing(self, sigma, expected):
        """Validated on the real angelfish flank: sigma 0.5 gives 0.69,
        sigma 1.0 gives 0.14, sigma >= 1.5 gives 0.00. Graded, not brittle."""
        patch = _scales()
        retained = texture_retention(patch, gaussian_filter(patch, sigma))

        assert retained == pytest.approx(expected, abs=0.15)

    def test_a_heavy_blur_takes_the_lattice_entirely(self):
        patch = _scales()
        assert texture_retention(patch, gaussian_filter(patch, 2.0)) < 0.05

    def test_removing_only_noise_scores_near_one(self):
        """The discrimination the whole module exists for: a filter that takes
        the grain and leaves the lattice must not look like one that took
        both."""
        clean = _scales(noise=0.0)
        noisy = clean + np.random.default_rng(4).normal(0.0, 6.0, clean.shape)

        assert texture_retention(noisy, clean) == pytest.approx(1.0, abs=0.25)

    def test_added_noise_barely_moves_it(self):
        """Robustness: the metric must not read added grain as added texture.

        An earlier design that subtracted an open-water floor read 1.16 and
        2.82 here — texture *created* by denoising — because shot noise scales
        with brightness and water is the wrong floor level for a fish.
        """
        patch = _scales()
        noisier = patch + np.random.default_rng(5).normal(0.0, 8.0, patch.shape)

        assert texture_retention(patch, noisier) == pytest.approx(0.9, abs=0.2)

    def test_no_measurable_peak_is_undefined_not_zero(self):
        """0/0 is not a retention figure and must not be averaged into a
        table."""
        water = _water()

        assert np.isnan(texture_retention(water, gaussian_filter(water, 1.0)))


class TestGrainAndReport:
    def test_grain_reduction_reads_a_blur(self):
        water = _water()
        assert grain_reduction(water, gaussian_filter(water, 1.5)) > 3.0

    def test_grain_ignores_an_illumination_gradient(self):
        """A plain standard deviation read /1.2 where this reads /5: it was
        counting the water's gradient, which no denoiser removes and which is
        not noise."""
        flat = _water()
        gradient = np.linspace(0, 60, 192)[None, :] * np.ones((192, 1))

        assert grain_reduction(flat, flat + gradient) == pytest.approx(1.0, abs=0.05)

    def test_the_report_pairs_the_two_numbers_on_purpose(self):
        """Either on its own can be gamed: a blur wins on grain, the identity
        wins on retention."""
        report = assess(_scales(), _water(), gaussian_filter(_scales(), 1.5), gaussian_filter(_water(), 1.5))

        assert report.grain_reduction > 1.0
        assert report.texture_retained < 0.1
        assert "grain /" in str(report)

    def test_an_undefined_retention_says_so_in_words(self):
        report = assess(_water(), _water(), _water(), _water())

        assert np.isnan(report.texture_retained)
        assert "undefined" in str(report)
