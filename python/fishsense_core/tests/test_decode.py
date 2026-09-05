"""Unit tests for :mod:`fishsense_core.image.decode`.

The golden tests in ``test_decode_golden.py`` pin what the chain *produces*.
These pin what its pieces *promise*: that the config rejects nonsense at
construction rather than deep inside a decode, that a stretch is a stretch and
not a crop, and that nothing here moves a pixel.
"""

import numpy as np
import pytest

from fishsense_core.image.decode import (
    DecodeConfig,
    WhiteBalance,
    apply_clahe,
    apply_red_boost,
    apply_stretch,
    as_polygon,
    auto_gamma,
    gray_world_gains,
    normalize_gains,
    postprocess_kwargs,
    rectify,
    slate_patch_gains,
    white_patch_gains,
)


def _scene(height: int = 96, width: int = 128, seed: int = 3) -> np.ndarray:
    """A float RGB frame in a narrow band, the way an underwater frame is.

    A gradient plus noise, occupying [0.10, 0.45] — flat enough that a stretch
    has something to do and textured enough that a warp would show.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 1.0, width)[None, :, None] * np.ones((height, 1, 1))
    base = 0.10 + 0.25 * ramp + rng.normal(0.0, 0.01, (height, width, 3))
    # Cyan cast: red starved, blue full.
    base *= np.array([0.35, 0.9, 1.0])
    return np.clip(base, 0.0, 1.0)


class TestDecodeConfigValidation:
    """Every one of these is a mistake that would otherwise surface as an
    exception from inside skimage, several seconds into a decode, with a
    traceback that names none of the fields involved."""

    def test_defaults_are_the_recommended_chain(self):
        config = DecodeConfig()

        assert config.white_balance is WhiteBalance.CAMERA
        assert config.stretch_mode == "luminance"
        assert config.clahe_enabled is False
        assert config.red_boost == 0.0
        assert config.auto_gamma_target == 20

    def test_production_is_the_old_chain(self):
        config = DecodeConfig.production()

        assert config.stretch_mode == "off"
        assert config.clahe_enabled is True
        assert config.clahe_mode == "value"
        assert config.clahe_clip_limit is None
        assert config.clahe_kernel_size is None

    def test_it_is_frozen_and_hashable(self):
        """Hashable so it can key a per-image cache; frozen so a cached result
        cannot be invalidated by someone mutating the key."""
        config = DecodeConfig()

        assert {config: "cached"}[DecodeConfig()] == "cached"
        assert config != DecodeConfig.production()
        with pytest.raises(AttributeError):
            config.clahe_enabled = True  # type: ignore[misc]

    @pytest.mark.parametrize("target", [0, -1])
    def test_non_positive_gamma_target_is_refused(self, target):
        with pytest.raises(ValueError, match="auto_gamma_target must be positive"):
            DecodeConfig(auto_gamma_target=target)

    @pytest.mark.parametrize("clip", [0.0, 1.5, -0.1])
    def test_clip_limit_outside_the_unit_interval_is_refused(self, clip):
        with pytest.raises(ValueError, match="clahe_clip_limit"):
            DecodeConfig(clahe_clip_limit=clip)

    def test_a_kernel_smaller_than_two_is_refused(self):
        with pytest.raises(ValueError, match="clahe_kernel_size"):
            DecodeConfig(clahe_kernel_size=1)

    def test_unknown_modes_are_refused(self):
        with pytest.raises(ValueError, match="stretch_mode"):
            DecodeConfig(stretch_mode="global")
        with pytest.raises(ValueError, match="clahe_mode"):
            DecodeConfig(clahe_mode="lab")

    def test_inverted_percentiles_are_refused(self):
        with pytest.raises(ValueError, match="0 <= low < high <= 100"):
            DecodeConfig(stretch_low=99.0, stretch_high=1.0)
        with pytest.raises(ValueError, match="0 <= low < high <= 100"):
            DecodeConfig(stretch_high=101.0)

    def test_percentile_triples_are_accepted_and_checked_elementwise(self):
        """Red is a narrow noise-dominated band while blue is broad, so one
        pair of percentiles for all three is the wrong shape of knob."""
        DecodeConfig(
            stretch_mode="per_channel",
            stretch_low=(2.0, 1.0, 1.0),
            stretch_high=(99.0, 99.0, 99.5),
        )

        with pytest.raises(ValueError, match="0 <= low < high <= 100"):
            DecodeConfig(
                stretch_mode="per_channel",
                stretch_low=(2.0, 1.0, 99.9),
                stretch_high=99.0,
            )
        with pytest.raises(ValueError, match="an \\(R, G, B\\) triple"):
            DecodeConfig(stretch_mode="per_channel", stretch_low=(1.0, 2.0))

    @pytest.mark.parametrize("mode", ["luminance", "off"])
    def test_a_triple_without_per_channel_is_refused(self, mode):
        """The luminance stretch maps CIELAB L*, which is one channel.

        Accepting a triple there would silently use its R entry for the whole
        frame while `label` went on advertising all three — a config that reads
        as tuned per channel and is not.
        """
        with pytest.raises(ValueError, match="needs stretch_mode='per_channel'"):
            DecodeConfig(stretch_mode=mode, stretch_low=(2.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="needs stretch_mode='per_channel'"):
            DecodeConfig(stretch_mode=mode, stretch_high=(99.0, 95.0, 90.0))

    def test_a_per_channel_triple_actually_reaches_each_channel(self):
        """The other half of the same bug: that the triple is not merely
        accepted but applied. Percentiles this far apart cannot produce the
        same output as any single pair."""
        scene = _scene()
        triple = apply_stretch(
            scene,
            DecodeConfig(
                stretch_mode="per_channel",
                stretch_low=(2.0, 10.0, 20.0),
                stretch_high=(99.0, 95.0, 90.0),
            ),
        )
        for entry_low, entry_high in ((2.0, 99.0), (10.0, 95.0), (20.0, 90.0)):
            scalar = apply_stretch(
                scene,
                DecodeConfig(
                    stretch_mode="per_channel",
                    stretch_low=entry_low,
                    stretch_high=entry_high,
                ),
            )
            assert not np.array_equal(triple, scalar)

    @pytest.mark.parametrize("field", ["red_boost", "red_boost_sigmas"])
    def test_negative_red_boost_is_refused(self, field):
        with pytest.raises(ValueError, match=field):
            DecodeConfig(**{field: -1.0})

    def test_labels_are_stable_and_filename_safe(self):
        assert DecodeConfig().label == "default"
        assert DecodeConfig.production().label == "noStretch-clahe"
        assert (
            DecodeConfig(clahe_enabled=True, clahe_mode="luminance").label
            == "lumaclahe"
        )
        assert (
            DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD).label == "grayworld"
        )
        assert DecodeConfig(red_boost=0.25, red_boost_sigmas=6.0).label == (
            "redboost0.25@6s"
        )
        for config in (
            DecodeConfig(),
            DecodeConfig.production(),
            DecodeConfig(
                stretch_mode="per_channel",
                stretch_low=(2.0, 1.0, 1.0),
                auto_gamma_target=40,
            ),
        ):
            assert config.label
            assert not set(config.label) & set("/\\ :*?\"<>|")


class TestPostprocessKwargs:
    def test_geometry_and_linearity_are_pinned_for_every_config(self):
        """``user_flip`` is one keyword away from rotating the frame, which
        would invalidate every label coordinate ever recorded against it."""
        for config in (DecodeConfig(), DecodeConfig.production()):
            kwargs = postprocess_kwargs(config, None)
            assert kwargs["user_flip"] == 0
            assert kwargs["gamma"] == (1, 1)
            assert kwargs["no_auto_bright"] is True
            assert kwargs["output_bps"] == 16

    def test_camera_and_auto_select_rawpy_flags(self):
        assert postprocess_kwargs(DecodeConfig(), None)["use_camera_wb"] is True
        auto = postprocess_kwargs(
            DecodeConfig(white_balance=WhiteBalance.RAWPY_AUTO), None
        )
        assert auto["use_auto_wb"] is True
        assert "use_camera_wb" not in auto

    def test_an_estimated_white_balance_without_gains_is_an_error(self):
        """Rather than silently falling back to the camera preset, which would
        make a white-balance experiment quietly a no-op — and 'white balance
        does not matter' is exactly the wrong conclusion to reach by accident.
        """
        config = DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD)
        with pytest.raises(ValueError, match="resolve_white_balance"):
            postprocess_kwargs(config, None)

        assert postprocess_kwargs(config, [2.0, 1.0, 1.5, 1.0])["user_wb"] == [
            2.0, 1.0, 1.5, 1.0
        ]


class TestGainEstimators:
    def test_gains_are_normalized_so_green_is_one(self):
        """A common factor across all three is an exposure change, not a white
        balance change; pinning green keeps the two separable."""
        assert normalize_gains((4.0, 2.0, 3.0)) == (2.0, 1.0, 1.5)

    @pytest.mark.parametrize("green", [0.0, float("nan"), float("inf")])
    def test_a_degenerate_green_falls_back_to_unity(self, green):
        assert normalize_gains((4.0, green, 3.0)) == (1.0, 1.0, 1.0)

    def test_gray_world_equalizes_the_means(self):
        # BGR, blue-heavy, the way an underwater frame is.
        image = np.zeros((8, 8, 3), dtype=np.float64)
        image[..., 0], image[..., 1], image[..., 2] = 0.8, 0.4, 0.1
        red, green, blue = gray_world_gains(image)

        assert green == 1.0
        assert red == pytest.approx(4.0)
        assert blue == pytest.approx(0.5)

    def test_a_dead_channel_passes_through_at_unity(self):
        """A zero channel is a broken decode, not a licence to return inf and
        blow the frame out."""
        image = np.zeros((8, 8, 3), dtype=np.float64)
        image[..., 1] = 0.4
        red, green, blue = gray_world_gains(image)

        assert (red, green, blue) == (1.0, 1.0, 1.0)

    def test_white_patch_ignores_a_single_specular_glint(self):
        """One hot pixel must not set the white point for the whole frame."""
        image = np.full((32, 32, 3), 0.2, dtype=np.float64)
        image[..., 1] = 0.4
        clean = white_patch_gains(image)

        image[0, 0, 2] = 1.0  # a glint on red
        assert white_patch_gains(image) == pytest.approx(clean)


class TestSlateQuad:
    def test_two_opposite_corners_become_a_rectangle(self):
        """``slate_rectangle`` stores two corners. ``cv2.fillPoly`` accepts
        that silently and fills a one-pixel diagonal line, which clears a
        minimum-pixel guard while containing almost none of the slate."""
        assert as_polygon([[10.0, 20.0], [30.0, 50.0]]) == [
            (10.0, 20.0), (30.0, 20.0), (30.0, 50.0), (10.0, 50.0)
        ]

    def test_a_real_polygon_passes_through(self):
        quad = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
        assert as_polygon(quad) == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

    @pytest.mark.parametrize(
        "rectangle",
        [None, [], [[1.0, 2.0]], [[1.0, 2.0], [1.0, 9.0]], [[1.0, 2.0], [9.0, 2.0]]],
    )
    def test_anything_that_cannot_bound_an_area_returns_none(self, rectangle):
        assert as_polygon(rectangle) is None

    def test_a_numpy_rectangle_does_not_raise(self):
        """`not array` raises on a numpy array, and callers legitimately pass
        one."""
        assert as_polygon(np.array([[10.0, 20.0], [30.0, 50.0]])) is not None

    def test_slate_gains_come_from_the_paper_not_the_ink(self):
        """The slate is white paper carrying black markings, so its mean is a
        paper/ink mixture that shifts with how much artwork is in view."""
        image = np.zeros((64, 64, 3), dtype=np.float64)
        # Inside the quad: mostly paper, a black stripe of "ink".
        image[16:48, 16:48] = (0.8, 0.4, 0.2)  # BGR paper, blue-cast
        image[20:24, 16:48] = 0.0

        red, green, blue = slate_patch_gains(image, [[16, 16], [47, 47]])
        assert green == 1.0
        assert red == pytest.approx(2.0)
        assert blue == pytest.approx(0.5)

    def test_a_degenerate_quad_is_refused_rather_than_sampled(self):
        image = np.ones((64, 64, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="does not bound an area"):
            slate_patch_gains(image, [[10.0, 10.0], [10.0, 40.0]])
        with pytest.raises(ValueError, match="too small"):
            slate_patch_gains(image, [[10.0, 10.0], [12.0, 12.0]])


class TestAutoGamma:
    def test_it_lifts_a_dark_frame(self):
        scene = _scene()
        assert auto_gamma(scene, 20).mean() > scene.mean()

    def test_a_higher_target_lifts_further(self):
        scene = _scene()
        assert auto_gamma(scene, 60).mean() > auto_gamma(scene, 20).mean()

    def test_a_frame_too_dark_to_lift_says_so(self):
        """The original expression is ``1 / (log(target * 255) / log(mean))``,
        which raises ZeroDivisionError at mean == 1 and ValueError below it,
        from inside the decode, naming nothing."""
        with pytest.raises(ValueError, match="too dark to auto-gamma"):
            auto_gamma(np.zeros((8, 8, 3)), 20)


class TestStretch:
    def test_off_is_the_identity(self):
        scene = _scene()
        assert apply_stretch(scene, DecodeConfig(stretch_mode="off")) is scene

    def test_luminance_expands_contrast_without_moving_the_cast(self):
        """The cyan cast is left alone on purpose: removing it needs
        range-based physics, not another global gain."""
        scene = _scene()
        stretched = apply_stretch(scene, DecodeConfig())

        assert stretched.std() > scene.std()
        # The cast survives: red stays the starved channel and blue the full
        # one, and their ratio stays well short of neutral. A per-channel map
        # is what would flatten it, and that is the next test.
        after = stretched.reshape(-1, 3).mean(axis=0)
        assert after[0] < after[1] < after[2]
        assert after[0] / after[2] < 0.85

    def test_per_channel_removes_the_cast(self):
        """Which is the point of having it, and the reason it is not the
        default: it is one of the variants that failed on field frames."""
        after = apply_stretch(
            _scene(), DecodeConfig(stretch_mode="per_channel")
        ).reshape(-1, 3).mean(axis=0)

        assert max(after) / min(after) < 1.2

    def test_a_flat_plane_is_left_alone(self):
        """Stretching a plane with no range would turn its noise into the
        entire signal."""
        flat = np.full((16, 16, 3), 0.3)
        result = apply_stretch(flat, DecodeConfig(stretch_mode="per_channel"))

        np.testing.assert_allclose(result, flat)

    def test_the_output_stays_in_range(self):
        for mode in ("per_channel", "luminance"):
            out = apply_stretch(_scene(), DecodeConfig(stretch_mode=mode))
            assert out.min() >= 0.0 and out.max() <= 1.0


class TestClahe:
    def test_disabled_is_the_identity(self):
        scene = _scene()
        assert apply_clahe(scene, DecodeConfig()) is scene

    def test_value_mode_is_skimages_own_rgb_handling(self):
        """``equalize_adapthist`` is ``@adapt_rgb(hsv_value)``-decorated: it
        equalizes HSV V and restores hue and saturation.

        This is the fact the original proposal got wrong — it held that
        skimage had no RGB branch and equalised each channel independently.
        Three channels that are exact scalings of one another settle it: per
        channel would drive their ratios to 1.0, V-only leaves them alone.
        """
        ramp = np.linspace(0.05, 0.5, 128)[None, :] * np.ones((96, 1))
        scene = np.stack([ramp * 0.3, ramp * 0.7, ramp], axis=2)

        out = apply_clahe(scene, DecodeConfig(clahe_enabled=True))
        ratio = out[..., 0] / np.maximum(out[..., 2], 1e-9)

        assert float(ratio.mean()) == pytest.approx(0.3, abs=0.01)

    def test_value_mode_holds_saturation_and_luminance_mode_holds_chroma(self):
        """Which is the whole difference between them.

        HSV saturation is a *ratio*, so holding it fixed while amplifying V
        amplifies chroma noise in step with luma noise. CIELAB a* and b* are
        absolute, so holding them fixed amplifies luma alone.
        """
        from skimage.color import rgb2hsv, rgb2lab  # noqa: PLC0415

        scene = _scene()
        value_mode = apply_clahe(scene, DecodeConfig(clahe_enabled=True))
        luminance_mode = apply_clahe(
            scene, DecodeConfig(clahe_enabled=True, clahe_mode="luminance")
        )

        def median_shift(after, before):
            return float(np.median(np.abs(after - before)))

        # Medians rather than maxima throughout. Both paths clip: CLAHE can
        # drive V to zero, where saturation is undefined, and lifting L* alone
        # pushes saturated pixels out of the sRGB gamut, where the clip back
        # into it moves a* and b*. Those are edge behaviours, not the claim.

        # The V path holds saturation and lets absolute chroma follow V up.
        assert median_shift(rgb2hsv(value_mode)[..., 1], rgb2hsv(scene)[..., 1]) < 0.01
        assert median_shift(rgb2lab(value_mode)[..., 1:], rgb2lab(scene)[..., 1:]) > 5.0

        # The L* path is the mirror image: absolute chroma held exactly,
        # saturation free to fall as L* rises.
        # Zero to float round-trip precision, not merely small.
        assert (
            median_shift(rgb2lab(luminance_mode)[..., 1:], rgb2lab(scene)[..., 1:])
            < 1e-9
        )
        assert (
            median_shift(rgb2hsv(luminance_mode)[..., 1], rgb2hsv(scene)[..., 1]) > 0.05
        )


    def test_the_clip_limit_and_kernel_reach_skimage(self):
        """Both are `None` by default, meaning skimage's own defaults. Neither
        was ever passed through in a test, so nothing checked that setting one
        did anything at all."""
        from unittest.mock import patch  # noqa: PLC0415

        scene = _scene()
        with patch(
            "fishsense_core.image.decode.equalize_adapthist",
            side_effect=lambda img, **kw: (captured.update(kw) or img),
        ):
            captured: dict = {}
            apply_clahe(scene, DecodeConfig(clahe_enabled=True))
            assert captured == {}

            captured = {}
            apply_clahe(
                scene,
                DecodeConfig(
                    clahe_enabled=True, clahe_clip_limit=0.003, clahe_kernel_size=16
                ),
            )
            assert captured == {"clip_limit": 0.003, "kernel_size": 16}

    def test_a_lower_clip_limit_amplifies_less(self):
        """The knob's whole purpose, and the reason it is exposed.

        Measured on flat water rather than on the gradient `_scene` builds: the
        clip only binds where a tile's local histogram is narrow, which is
        exactly the near-uniform region CLAHE over-amplifies and exactly what a
        gradient does not have. On the gradient both limits give bit-identical
        output, which is a fair description of the knob doing nothing there.
        """
        rng = np.random.default_rng(21)
        water = np.clip(
            0.30 + rng.normal(0.0, 0.004, (128, 128, 3)) * np.array([0.4, 1.0, 1.2]),
            0.0,
            1.0,
        )

        permissive = apply_clahe(
            water, DecodeConfig(clahe_enabled=True, clahe_clip_limit=0.01)
        )
        strict = apply_clahe(
            water, DecodeConfig(clahe_enabled=True, clahe_clip_limit=0.001)
        )

        assert strict.std() < permissive.std()
        # And both amplify the input, which is the behaviour the default
        # decode now avoids entirely.
        assert permissive.std() > water.std()

    def test_the_clahe_settings_are_named_in_the_label(self):
        assert (
            DecodeConfig(
                clahe_enabled=True, clahe_clip_limit=0.003, clahe_kernel_size=16
            ).label
            == "clahe-clip0.003-kernel16"
        )


class TestRedBoost:
    def test_zero_is_the_identity(self):
        scene = _scene()
        assert apply_red_boost(scene, DecodeConfig()) is scene

    def test_it_finds_a_coherent_blob_and_not_the_speckle(self):
        """The laser dot is a coherent blob roughly ten pixels across; the red
        speckle a stretch amplifies is pixel-scale. Blurring the red-excess map
        before thresholding is what tells them apart."""
        rng = np.random.default_rng(11)
        scene = np.zeros((96, 96, 3))
        scene[..., 1] = 0.5
        scene[..., 2] = 0.6
        scene[..., 0] = 0.1 + rng.normal(0.0, 0.03, (96, 96))  # noisy red
        yy, xx = np.mgrid[0:96, 0:96]
        dot = np.exp(-(((yy - 30) ** 2 + (xx - 64) ** 2) / (2 * 3.0**2)))
        scene[..., 0] = np.clip(scene[..., 0] + 0.5 * dot, 0.0, 1.0)

        boosted = apply_red_boost(
            scene, DecodeConfig(red_boost=0.4, red_boost_sigmas=6.0)
        )
        gain = boosted[..., 0] - scene[..., 0]

        assert gain[28:33, 62:67].mean() > 0.2
        # Open water, well away from the dot, must be untouched.
        assert gain[70:96, 0:30].max() < 0.05

    def test_it_writes_only_red_and_does_not_mutate_its_input(self):
        scene = _scene()
        original = scene.copy()
        boosted = apply_red_boost(scene, DecodeConfig(red_boost=0.5))

        np.testing.assert_array_equal(scene, original)
        np.testing.assert_array_equal(boosted[..., 1:], scene[..., 1:])


class TestGeometry:
    """Nothing in the tone chain may move a pixel.

    Measurements here are pixel *coordinates*, and 1 px of laser-dot error is
    0.75% length error. A shape check catches a resize or a crop; it cannot
    catch a one-pixel roll, so the response is located instead.
    """

    @staticmethod
    def _displacement(operation) -> float:
        """Where the output responds when one patch of the input is swapped
        with another.

        The patches are *swapped* rather than one brightened, so the frame's
        histogram is bit-identical between the two runs — otherwise a global
        operator (a percentile stretch, an auto-gamma) derives a different
        mapping for each and the whole frame responds, saying nothing about
        geometry.
        """
        rng = np.random.default_rng(0xF15E)
        base = rng.uniform(0.35, 0.55, (192, 192, 3))
        block = (slice(24, 48), slice(120, 144))
        patch = (slice(120, 144), slice(48, 72))
        base[block] = 0.85

        perturbed = base.copy()
        perturbed[patch], perturbed[block] = base[block].copy(), base[patch].copy()

        def centroid(field):
            strong = field >= field.max() * 0.10
            ys, xs = np.nonzero(strong)
            weights = field[ys, xs]
            total = weights.sum()
            return (xs * weights).sum() / total, (ys * weights).sum() / total

        expected = centroid(np.abs(perturbed - base).sum(axis=2))
        observed = centroid(
            np.abs(operation(perturbed) - operation(base)).sum(axis=2)
        )
        return float(np.hypot(*(np.subtract(observed, expected))))

    @pytest.mark.parametrize(
        "config",
        [
            DecodeConfig(),
            DecodeConfig.production(),
            DecodeConfig(stretch_mode="per_channel"),
            DecodeConfig(clahe_enabled=True, clahe_mode="luminance"),
        ],
        ids=lambda c: c.label,
    )
    def test_the_tone_chain_moves_nothing(self, config):
        def operation(image):
            return apply_red_boost(
                apply_clahe(apply_stretch(image, config), config), config
            )

        # 0.5 px is the hard limit, calibrated by measurement: identity,
        # Gaussian blurs and CLAHE all read under 0.1 px, a one-pixel roll
        # reads 1.01, and nothing legitimate lands in between.
        assert self._displacement(operation) < 0.5

    def test_the_probe_catches_a_one_pixel_roll(self):
        """Otherwise the test above is only asserting that the probe is
        insensitive."""
        assert self._displacement(lambda img: np.roll(img, 1, axis=1)) > 0.9


class TestRectify:
    def test_it_preserves_frame_size(self):
        """Which is why laser pixels, head/tail pixels and the JPEG are all one
        coordinate system."""
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        matrix = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])

        assert rectify(image, matrix, np.zeros(5)).shape == (48, 64, 3)

    def test_it_takes_plain_sequences(self):
        """So it is usable without ``CameraIntrinsics``, which arrives with the
        git-only API SDK and is an optional extra."""
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        matrix = [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]

        assert rectify(image, matrix, [0.0, 0.0, 0.0, 0.0, 0.0]).shape == (16, 16, 3)


class TestOptionalExtras:
    """`beta`/`range_m` (sea-thru) and `denoise`, both off by default.

    They are here for the ordering, which is forced rather than stylistic, and
    for the validation, which exists so a half-configured correction fails
    loudly instead of producing a frame that looks corrected and is not.
    """

    def test_they_are_off_by_default(self):
        config = DecodeConfig()

        assert config.beta is None
        assert config.range_m is None
        assert config.denoise is None

    def test_beta_and_range_are_meaningful_only_together(self):
        """Sea-thru cannot be a default even in principle: `RawImage(bytes)`
        has no idea which dive a frame came from or how far away the subject
        was, and the correction needs both."""
        DecodeConfig(beta=(0.263, 0.040, 0.001), range_m=1.5)

        with pytest.raises(ValueError, match="must be given together"):
            DecodeConfig(beta=(0.263, 0.040, 0.001))
        with pytest.raises(ValueError, match="must be given together"):
            DecodeConfig(range_m=1.5)

    def test_a_malformed_beta_is_refused(self):
        with pytest.raises(ValueError, match="three per-channel"):
            DecodeConfig(beta=(0.263, 0.040), range_m=1.0)
        with pytest.raises(ValueError, match="negative coefficient amplifies"):
            DecodeConfig(beta=(-0.1, 0.04, 0.001), range_m=1.0)
        with pytest.raises(ValueError, match="range_m must be non-negative"):
            DecodeConfig(beta=(0.263, 0.040, 0.001), range_m=-1.0)

    def test_sea_thru_runs_on_linear_radiance_before_the_auto_gamma(self):
        """Forced, not stylistic: `remove_water` inverts a radiance formation
        model, so applying it after a gamma curve inverts a curve that is not
        in the model. Checked by running the two orders and showing they
        disagree — if the placement did not matter, this test would be the one
        to delete.
        """
        from fishsense_core.image.decode import apply_seathru  # noqa: PLC0415

        scene = _scene()
        config = DecodeConfig(beta=(0.263, 0.040, 0.001), range_m=3.0)

        as_shipped = auto_gamma(apply_seathru(scene, config), 20)
        reversed_order = apply_seathru(auto_gamma(scene, 20), config)

        assert not np.allclose(as_shipped, reversed_order, atol=1e-3)

    def test_sea_thru_lifts_red_relative_to_blue(self):
        from fishsense_core.image.decode import apply_seathru  # noqa: PLC0415

        scene = _scene()
        corrected = apply_seathru(
            scene, DecodeConfig(beta=(0.263, 0.040, 0.001), range_m=3.0)
        )

        before = scene[..., 0].mean() / scene[..., 2].mean()
        after = corrected[..., 0].mean() / corrected[..., 2].mean()
        assert after > before

    def test_sea_thru_is_the_identity_when_off(self):
        from fishsense_core.image.decode import apply_seathru  # noqa: PLC0415

        scene = _scene()
        assert apply_seathru(scene, DecodeConfig()) is scene

    def test_denoise_is_the_identity_when_off(self):
        from fishsense_core.image.decode import apply_denoise  # noqa: PLC0415

        frame = (_scene() * 255).astype(np.uint8)
        assert apply_denoise(frame, DecodeConfig()) is frame

    def test_the_extras_are_named_in_the_label(self):
        from fishsense_core.image.denoise import BM3DConfig  # noqa: PLC0415

        assert DecodeConfig(beta=(0.26, 0.04, 0.001), range_m=1.5).label == (
            "seathru0.26_0.04_0.001@1.5m"
        )
        assert DecodeConfig(denoise=BM3DConfig(strength=0.5)).label == "denoise0.5"

    def test_a_denoiser_without_a_strength_still_labels(self):
        """`label` is used as a filename stem and in reports, so it must not be
        the thing that raises. The earlier
        ``f"...{getattr(o, 'strength', ''):g}"`` did: ":g" cannot format the
        empty-string fallback."""

        class Bare:  # a denoiser that is not a BM3DConfig
            pass

        assert DecodeConfig(denoise=Bare()).label == "denoise"

    def test_the_denoiser_actually_runs_through_the_decode(self):
        """`apply_denoise` flips BGR to RGB for the enhancer and back again;
        nothing tested that wiring end to end, only the enhancer alone."""
        pytest.importorskip("bm3d", reason="the `denoise` extra is not installed")

        from fishsense_core.image.decode import apply_denoise  # noqa: PLC0415
        from fishsense_core.image.denoise import BM3DConfig  # noqa: PLC0415

        rng = np.random.default_rng(9)
        # A frame whose channels are clearly distinguishable, so a swapped
        # flip would show up as a colour change rather than as nothing.
        frame = np.clip(
            np.stack(
                [
                    rng.normal(200, 6, (64, 64)),
                    rng.normal(120, 6, (64, 64)),
                    rng.normal(40, 6, (64, 64)),
                ],
                axis=2,
            ),
            0,
            255,
        ).astype(np.uint8)

        out = apply_denoise(frame, DecodeConfig(denoise=BM3DConfig(psd_size=16)))

        import cv2  # noqa: PLC0415

        assert out.shape == frame.shape and out.dtype == np.uint8
        # Luminance, not a single channel: the enhancer filters CIELAB L and
        # passes a and b through, so independent per-channel noise survives on
        # purpose and a per-channel SD would not move.
        assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).std() < cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        ).std()
        # Channel order preserved through the two flips: B stays the bright
        # one, R stays the dark one. A swapped flip inverts this.
        assert out[..., 0].mean() > out[..., 1].mean() > out[..., 2].mean()
        np.testing.assert_allclose(
            out.reshape(-1, 3).mean(axis=0),
            frame.reshape(-1, 3).mean(axis=0),
            atol=3.0,
        )

    def test_configuring_denoise_does_not_import_bm3d(self):
        """The `bm3d` extra is optional, so building a config that mentions it
        must not require it to be installed — only running the decode does.

        Checked in a fresh interpreter, because by the time this test runs in a
        full suite something else has already imported it.
        """
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        script = (
            "import sys;"
            "from fishsense_core.image.decode import DecodeConfig;"
            "from fishsense_core.image.denoise import BM3DConfig;"
            "DecodeConfig(denoise=BM3DConfig());"
            "print('bm3d' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False"

    def test_an_unimportable_denoiser_says_which_extra_to_install(self):
        """Rather than a bare ModuleNotFoundError from three frames down."""
        import builtins  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from fishsense_core.image import denoise as denoise_module  # noqa: PLC0415

        real_import = builtins.__import__

        def refuse_bm3d(name, *args, **kwargs):
            if name == "bm3d":
                raise ImportError("no bm3d here")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", refuse_bm3d):
            with pytest.raises(ImportError, match=r"fishsense_core\[denoise\]"):
                denoise_module._import_bm3d()  # pylint: disable=protected-access
