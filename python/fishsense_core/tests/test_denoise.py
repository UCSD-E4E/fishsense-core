"""Tests for :mod:`fishsense_core.image.denoise`.

Small patches throughout: BM3D at the reference profile costs 133.8 s on a
12-megapixel frame, and none of the properties worth pinning need a big one.
"""

import warnings

import numpy as np
import pytest

from fishsense_core.image.contract import probe_geometry
from fishsense_core.image.denoise import (
    BM3D_PROFILES,
    BM3DConfig,
    TexturedNoiseRegion,
    bm3d_enhancer,
    denoise_luminance,
    noise_psd_from_water,
)
from fishsense_core.image.texture import texture_retention

pytest.importorskip("bm3d", reason="the `denoise` extra is not installed")


def _noisy_water(size=96, sigma=6.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.full((size, size), 110.0) + rng.normal(0, sigma, (size, size)), 0, 255)


def _frame(size=96, sigma=6.0, seed=1) -> np.ndarray:
    """uint8 RGB: flat water top-left, a lattice lower-right."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.full((size, size), 110.0)
    lattice = 14.0 * (np.sin(2 * np.pi * xx / 4.5) + np.sin(2 * np.pi * yy / 4.5))
    base[size // 2 :, size // 2 :] += lattice[size // 2 :, size // 2 :]
    base = base + rng.normal(0, sigma, (size, size))
    return np.clip(np.stack([base, base * 0.95, base * 0.9], axis=2), 0, 255).astype(
        np.uint8
    )


class TestConfig:
    def test_defaults_are_the_reference_profile(self):
        config = BM3DConfig()

        assert config.strength == 1.0
        assert config.profile == "np"
        # Chroma untouched by default: hue is what the species and slate labels
        # read.
        assert config.chroma_strength == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"psd_size": 4}, "psd_size"),
            ({"strength": 0.0}, "strength"),
            ({"chroma_strength": -1.0}, "chroma_strength"),
            ({"profile": "fast"}, "profile must be one of"),
        ],
    )
    def test_nonsense_is_refused_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BM3DConfig(**kwargs)

    @pytest.mark.parametrize("profile", BM3D_PROFILES)
    def test_every_advertised_profile_name_is_one_bm3d_accepts(self, profile):
        """`bm3d.bm3d` takes np/refilter/vn/high/vn_old/deb or a BM3DProfile
        object. "lc" is not among the strings — the low-complexity profile
        exists only as `bm3d.BM3DProfileLC` — so `profile="lc"`, which the cost
        figures quote, used to run the whole decode and then raise TypeError
        from inside the filter. Every name this module advertises has to work.
        """
        import bm3d  # noqa: PLC0415

        from fishsense_core.image.denoise import _resolve_profile  # noqa: PLC0415

        resolved = _resolve_profile(bm3d, profile)
        assert isinstance(resolved, (str, bm3d.BM3DProfile))
        if isinstance(resolved, str):
            assert resolved != "lc"

    def test_the_low_complexity_profile_actually_runs(self):
        """The one the docstrings offer as the cheaper option: 56.8 s/frame
        against the reference profile's 133.8 s."""
        water = _noisy_water(size=48)
        out = denoise_luminance(water, water, BM3DConfig(psd_size=16, profile="lc"))

        assert out.shape == water.shape
        assert out.std() < water.std()

    def test_a_profile_object_passes_straight_through(self):
        import bm3d  # noqa: PLC0415

        water = _noisy_water(size=48)
        config = BM3DConfig(psd_size=16, profile=bm3d.BM3DProfileLC())
        assert denoise_luminance(water, water, config).std() < water.std()


class TestNoisePsd:
    def test_white_noise_reads_its_own_variance_in_every_bin(self):
        """The normalisation convention the whole module depends on: white
        noise of variance sigma² reads sigma² per bin, the same convention as
        `texture.radial_power_spectrum`."""
        rng = np.random.default_rng(2)
        psd = noise_psd_from_water(rng.normal(0.0, 4.0, (256, 256)), size=64)

        # Skip DC and the outermost ring, which the Hann window shapes.
        interior = psd[8:-8, 8:-8]
        assert float(np.median(interior)) == pytest.approx(16.0, rel=0.2)

    def test_it_measures_the_noise_and_not_the_level(self):
        """Mean removal per tile, so a bright patch does not read as power."""
        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 4.0, (192, 192))
        dim = noise_psd_from_water(noise + 10.0, size=64)
        bright = noise_psd_from_water(noise + 200.0, size=64)

        np.testing.assert_allclose(dim, bright, rtol=1e-9)

    def test_the_psd_is_shifted_with_dc_at_the_centre(self):
        rng = np.random.default_rng(4)
        # Add a strong low-frequency ramp; its power belongs near the centre.
        ramp = np.linspace(0, 40, 128)[None, :] * np.ones((128, 1))
        psd = noise_psd_from_water(ramp + rng.normal(0, 1, (128, 128)), size=64)

        centre = psd[30:34, 30:34].mean()
        corner = psd[:4, :4].mean()
        assert centre > corner


class TestDenoiseLuminance:
    def test_it_removes_noise(self):
        water = _noisy_water()
        out = denoise_luminance(water, water, BM3DConfig(psd_size=32))

        assert out.std() < water.std()
        assert out.shape == water.shape

    def test_strength_is_monotone(self):
        water = _noisy_water()
        light = denoise_luminance(water, water, BM3DConfig(psd_size=32, strength=0.25))
        heavy = denoise_luminance(water, water, BM3DConfig(psd_size=32, strength=2.0))

        assert heavy.std() < light.std()

    def test_the_output_stays_in_range(self):
        water = _noisy_water()
        out = denoise_luminance(water, water, BM3DConfig(psd_size=32))

        assert out.min() >= 0.0 and out.max() <= 255.0


class TestEnhancer:
    def test_it_returns_a_uint8_frame_of_the_same_shape(self):
        frame = _frame()
        out = bm3d_enhancer(BM3DConfig(psd_size=32))(frame)

        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_it_quiets_the_water(self):
        frame = _frame()
        out = bm3d_enhancer(BM3DConfig(psd_size=32))(frame)
        water = np.s_[:32, :32]

        assert out[water].std() < frame[water].std()

    def test_it_keeps_the_lattice_it_is_chosen_for(self):
        """The whole reason BM3D is here rather than a blind-spot network: it
        groups similar patches and filters them jointly, so a periodic lattice
        is reinforced rather than averaged away."""
        frame = _frame()
        out = bm3d_enhancer(BM3DConfig(psd_size=32, strength=0.5))(frame)
        fish = np.s_[48:, 48:]

        retained = texture_retention(
            frame[fish][..., 0].astype(np.float64), out[fish][..., 0].astype(np.float64)
        )
        assert retained > 0.5

    def test_chroma_is_untouched_by_default(self):
        """a and b are the hue the species and slate labels read."""
        from skimage.color import rgb2lab

        frame = _frame()
        out = bm3d_enhancer(BM3DConfig(psd_size=32))(frame)

        before = rgb2lab(frame.astype(np.float64) / 255.0)[..., 1:]
        after = rgb2lab(out.astype(np.float64) / 255.0)[..., 1:]
        # Not bit-identical — L moved, and the round trip through sRGB is
        # nonlinear — but the hue must not have been filtered.
        assert float(np.median(np.abs(after - before))) < 1.5

    def test_chroma_filtering_takes_the_speckle_and_leaves_the_hue(self):
        """Scales are luminance texture, so chroma can be filtered far harder
        than L without touching them. What must survive is the *local mean* of
        a and b, which is the hue the species and slate labels read."""
        from skimage.color import rgb2lab  # noqa: PLC0415

        frame = _frame()
        config = BM3DConfig(psd_size=32, chroma_strength=2.0)
        out = bm3d_enhancer(config)(frame)

        before = rgb2lab(frame.astype(np.float64) / 255.0)[..., 1:]
        after = rgb2lab(out.astype(np.float64) / 255.0)[..., 1:]

        # Speckle down...
        assert after.std() < before.std()
        # ...hue held: the frame-wide mean of a and b barely moves.
        np.testing.assert_allclose(after.mean(axis=(0, 1)), before.mean(axis=(0, 1)), atol=2.0)

    def test_a_textured_psd_region_warns_rather_than_over_filtering_in_silence(self):
        """When a test frame's scale lattice extended into the PSD region, the
        noise model contained the scales and the filter erased them across the
        whole frame — retention 0.000 against 226 sigma of input. The top-left
        fifth is open water by convention, not by guarantee."""
        rng = np.random.default_rng(6)
        size = 96
        yy, xx = np.mgrid[0:size, 0:size]
        lattice = 110.0 + 16.0 * (
            np.sin(2 * np.pi * xx / 4.5) + np.sin(2 * np.pi * yy / 4.5)
        )
        textured = np.clip(
            np.stack([lattice + rng.normal(0, 3, (size, size))] * 3, axis=2), 0, 255
        ).astype(np.uint8)

        with pytest.warns(TexturedNoiseRegion, match="sigma spectral peak"):
            bm3d_enhancer(BM3DConfig(psd_size=32))(textured)

    def test_a_clean_water_region_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", TexturedNoiseRegion)
            bm3d_enhancer(BM3DConfig(psd_size=32))(_frame())

    def test_it_moves_no_pixel(self):
        """0.5 px is the hard limit; BM3D at half the measured PSD was measured
        at 0.009 px, a tenth of the mosaic-domain alternative's."""
        result = probe_geometry(
            bm3d_enhancer(BM3DConfig(psd_size=32, strength=0.5)),
            name="bm3d",
            size=128,
        )

        assert result.displacement_px < 0.5
