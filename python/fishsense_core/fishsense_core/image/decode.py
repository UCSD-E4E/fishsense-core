"""The raw decode chain, as parameters rather than as a fixed sequence.

:class:`~fishsense_core.image.raw_image.RawImage` used to hard-code one chain::

    rawpy.postprocess(gamma=(1,1), no_auto_bright=True, use_camera_wb=True,
                      output_bps=16, user_flip=0)
      -> auto-gamma from the frame's own mean V, targeting 20
      -> skimage.exposure.equalize_adapthist() at defaults

This module makes every step of it a field on :class:`DecodeConfig`, and
changes two of the defaults. Everything else here is opt-in and off.

The chain, in order, with the optional steps bracketed::

    postprocess -> [remove_water] -> auto-gamma -> [stretch] -> [clahe]
                -> [red boost] -> [denoise] -> uint8 BGR

and :class:`~fishsense_core.image.rectified_image.RectifiedImage` applies
``cv2.undistort`` after that, which is what puts pixels in label space.

Two of those positions are forced rather than stylistic.
:func:`~fishsense_core.water.seathru.remove_water` inverts a radiance formation
model, so it goes on linear radiance — after ``postprocess``, before the
auto-gamma; applying it later would invert a curve that is not in the model.
Denoising goes last because the enhancer it is expressed as reads a finished
uint8 frame.

The new defaults
----------------

``stretch_mode="luminance"`` and ``clahe_enabled=False``: a global CIELAB L*
percentile stretch in place of local histogram equalisation.

The reason is the **colour space each one works in**, and it is not the reason
that was originally written down — see "A correction" below.

``skimage.exposure.equalize_adapthist`` is decorated ``@adapt_rgb(hsv_value)``,
so on an H×W×3 frame it converts to HSV, equalises **V**, and converts back
with hue and saturation untouched. Saturation is a *ratio*, so holding it fixed
while amplifying V amplifies chroma noise in exact step with luma noise: flat
open water comes out as coloured speckle. A CIELAB L* map holds a* and b* fixed
in *absolute* terms, so the same amplification raises luma contrast and leaves
the chroma noise where it was — the same water reads as grey grain.

Measured on this repository's decode fixture, over its open-water patch:

===========================  ==========  ==========  =============
quantity                     CLAHE       L* stretch  before either
===========================  ==========  ==========  =============
noise amplification          ×3.89       ×3.16       —
whole-frame contrast (sd)    0.170       0.181       0.06
cross-channel noise corr.    0.51        0.90        0.40
8-bit water B-R spread       39.3        22.3        —
===========================  ==========  ==========  =============

So the two buy about the same contrast for about the same noise, and differ
almost entirely in whether that noise is coloured. The colour space is the
whole of it: replacing CLAHE with a *global* 4× expansion of V, through the
same HSV path, leaves the correlation at 0.48 — and a global 4× expansion of
L* puts it at 0.89. Locality sets how much a flat region is amplified; the
colour space sets what the amplification looks like.

The corpus-scale figures behind the recommendation were measured separately:
water-region noise amplification 36.7× at the old default against 1.0× with
CLAHE off, on a synthetic flat-water patch, and object-vs-water separation
improving 0.259 → 0.397. On a 3016×4014 frame ``equalize_adapthist`` is also
10.2 s of a 13.9 s decode, so dropping it is roughly a 3.7× speedup.

A correction
------------

The proposal this module implements said that ``equalize_adapthist`` "has no
RGB branch", reading an H×W×3 frame as a 3-D volume with a channel-axis tile
depth of one and so equalising each colour channel independently — and that
this handed the largest gain to red.

**That is not what it does**, on scikit-image 0.25 or 0.26 or on any version
in a long time. ``adapt_rgb``/``hsv_value`` intercepts the RGB case before
``kernel_size`` is ever computed, and the wrapped filter only ever sees the
2-D V plane. Verified directly: three channels that are exact scalings of one
another come back with their ratios intact (0.300 in, 0.2998 out), where
per-channel equalisation would have driven them to 1.0; and on the fixture's
open water, hue and saturation come out of CLAHE bit-identical.

The recommendation is unaffected — it rested on measured outputs, not on the
mechanism — but the mechanism above is the one that holds up, and it happens
to be the better argument.

What the evidence does and does not say
---------------------------------------

167 field frames × 6 decode arms × 2 models (FishIAL and SAM 3), fully paired,
scored against human head/tail labels: **every paired statistic spans zero for
every arm.** The change is null-risk for the model consumers and untested on
humans. The decisive evidence is a labeling trial that has not run. These
defaults are the recommendation, not an established result.

What deliberately did not change
--------------------------------

**White balance.** Every variant that gave red an independent gain from the
bottom of its range failed on field frames — gray-world, white-patch, and a
per-channel stretch. The estimators are here because the question gets asked,
not because any of them is recommended. The
cyan cast is left alone on purpose; removing it needs range-based physics, not
another global gain.

**The laser path.** :class:`~fishsense_core.image.linear_raw_image.LinearRawImage`
decodes to linear uint16 in sensor coordinates and never reads a JPEG, so
nothing in this module's tone chain can reach laser detection. That is what
keeps a working, calibrated path from taking a dependency on an experimental
one, and it is asserted in ``tests/test_decode_golden.py`` rather than left as
a property of the current arrangement.

**Geometry.** Nothing here moves a pixel. Measurements in this pipeline are
pixel *coordinates* — Label Studio percentages resolve through
``original_width/height`` into rectified pixels, which are then back-projected
against the laser ray — and 1 px of laser-dot error is 0.75% length error.
Reading a neighbourhood is fine; displacing one is not.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import cv2
import numpy as np
import rawpy
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb, rgb2lab  # pylint: disable=no-name-in-module
from skimage.exposure import adjust_gamma, equalize_adapthist  # pylint: disable=no-name-in-module
from skimage.util import img_as_float, img_as_ubyte

from fishsense_core.image.image import open_image_source

# cv2 and rawpy are extension modules whose members pylint cannot see. The
# length is docstrings: this module is where the decode's reasoning lives, and
# splitting the chain across files to satisfy a line count would hide it.
# pylint: disable=no-member,too-many-lines

_log = logging.getLogger(__name__)

__all__ = [
    "DecodeConfig",
    "WhiteBalance",
    "Gains",
    "apply_clahe",
    "apply_denoise",
    "apply_red_boost",
    "apply_seathru",
    "apply_stretch",
    "as_polygon",
    "auto_gamma",
    "decode_linear_stage",
    "decode_rectified_stage",
    "gray_world_gains",
    "normalize_gains",
    "postprocess_kwargs",
    "rectify",
    "resolve_white_balance",
    "slate_patch_gains",
    "white_patch_gains",
]

#: Per-channel multipliers in ``(R, G, B)`` order.
Gains = Tuple[float, float, float]

#: Stretch and CLAHE modes, as accepted by :class:`DecodeConfig`.
STRETCH_MODES = ("off", "per_channel", "luminance")
CLAHE_MODES = ("value", "luminance")


class WhiteBalance(enum.Enum):
    """Where the white point comes from.

    ``CAMERA`` is the only one recommended, and is the default. The rest exist
    because "did you try white balance?" is a question that gets asked, and
    because the answer — *yes, and every variant that gives red an independent
    gain from the bottom of its range fails on field frames* — is worth being
    able to reproduce rather than merely assert.
    """

    #: As-shot camera white balance, a topside daylight preset applied to a
    #: scene lit through metres of water. Wrong in principle and the best of
    #: the available options in practice.
    CAMERA = "camera"
    #: Scene averages to neutral. Cheap, and wrong underwater in a specific
    #: way: the water column genuinely is blue-green, so gray-world reads the
    #: water as a cast and over-corrects red.
    GRAY_WORLD = "grayworld"
    #: Brightest percentile is neutral. Better than gray-world when a
    #: genuinely white object is in frame, and the dive slate usually is one.
    WHITE_PATCH = "whitepatch"
    #: Fitted from the dive slate — a printed-white reference at a known place
    #: in the frame. The only one anchored to a real object rather than to an
    #: assumption about scene statistics. Needs the image's slate quad.
    SLATE = "slate"
    #: rawpy's own auto white balance, for reference.
    RAWPY_AUTO = "rawpyauto"


# ---------------------------------------------------------------------------
# White-balance gain estimation
# ---------------------------------------------------------------------------


def normalize_gains(gains: Gains) -> Gains:
    """Scale gains so green is 1.0.

    rawpy's ``user_wb`` multiplies raw channel values, so a common factor
    across all three is an *exposure* change, not a white-balance change.
    Pinning green separates the two: without it a white-balance experiment
    also changes overall brightness, the auto-gamma partly compensates, and no
    observed difference can be attributed to either cause.

    Green is the reference because it has twice the photosites of red or blue
    and therefore the least noise.
    """
    red, green, blue = gains
    if not green or not math.isfinite(green):
        return (1.0, 1.0, 1.0)
    return (red / green, 1.0, blue / green)


def _channel_stat(image: np.ndarray, reducer) -> Gains:
    """Reduce a BGR array to per-channel ``(R, G, B)`` statistics."""
    blue = float(reducer(image[:, :, 0]))
    green = float(reducer(image[:, :, 1]))
    red = float(reducer(image[:, :, 2]))
    return red, green, blue


def _gains_from_levels(levels: Gains) -> Gains:
    """Turn per-channel levels into gains that equalize them.

    A channel measuring zero is a broken decode, not a licence to return
    ``inf`` and blow the frame out — it is passed through at unity gain so the
    resulting frame is visibly wrong rather than invisibly saturated.
    """
    red, green, blue = levels
    reference = green if green > 0 else max(red, blue, 1.0)
    out = [reference / level if level > 0 else 1.0 for level in (red, green, blue)]
    return normalize_gains((out[0], out[1], out[2]))


def gray_world_gains(image: np.ndarray) -> Gains:
    """Gains that equalize the per-channel means. ``image`` is BGR."""
    return _gains_from_levels(_channel_stat(image, np.mean))


def white_patch_gains(image: np.ndarray, *, percentile: float = 99.0) -> Gains:
    """Gains that equalize the bright tail of each channel. ``image`` is BGR.

    A percentile rather than the maximum: one hot pixel or one specular glint
    off a fin must not set the white point for the whole frame.
    """
    return _gains_from_levels(
        _channel_stat(image, lambda plane: np.percentile(plane, percentile))
    )


def as_polygon(
    rectangle: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float]] | None:
    """Normalize a slate rectangle into a polygon ``cv2.fillPoly`` can use.

    ``DiveSlateLabel.slate_rectangle`` stores **two opposite corners**, not a
    polygon::

        [[1708.4, 1079.6], [2298.0, 1534.5]]

    Handing that to ``cv2.fillPoly`` does not fail. It fills a degenerate
    two-point polygon — a one-pixel-wide diagonal line. On a real frame that
    line runs several hundred pixels, so it clears a minimum-pixel guard and
    looks like a valid sample while containing almost none of the slate; every
    statistic taken from it is then a statistic of a streak. Fixing this took
    the usable-sample count on the attenuation fit from 131 to 218 and its r²
    from 0.05–0.82 to 0.86–0.96.

    Returns ``None`` for anything that cannot bound an area, so a caller that
    forgets to check gets nothing rather than a line.
    """
    # Explicit None/len rather than truthiness: a numpy array raises on
    # `not array`, and callers legitimately pass one.
    if rectangle is None or len(rectangle) == 0:
        return None
    points = [(float(p[0]), float(p[1])) for p in rectangle]
    if len(points) >= 3:
        return points
    if len(points) != 2:
        return None
    (x0, y0), (x1, y1) = points
    if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
        return None
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    return [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]


def slate_patch_gains(
    image: np.ndarray,
    quad: Sequence[Sequence[float]],
    *,
    percentile: float = 90.0,
) -> Gains:
    """Gains fitted from the dive slate's printed white. ``image`` is BGR.

    The bright tail *within the quad* rather than the quad's mean, because the
    slate is white paper carrying black markings — the mean is a paper/ink
    mixture that depends on how much artwork is in view, while the tail is the
    paper.

    It assumes the paper is spectrally flat and not blown out. It assumes
    nothing about the water column, which is the point.
    """
    polygon = as_polygon(quad)
    if polygon is None:
        raise ValueError(f"slate quad {quad!r} does not bound an area")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], 255)
    inside = mask.astype(bool)
    if inside.sum() < 16:
        raise ValueError(
            f"slate quad covers only {int(inside.sum())} px; too small to fit a "
            "white balance from"
        )

    levels = [
        float(np.percentile(image[:, :, channel][inside], percentile))
        for channel in (2, 1, 0)  # R, G, B
    ]
    return _gains_from_levels((levels[0], levels[1], levels[2]))


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeConfig:  # pylint: disable=too-many-instance-attributes
    """One decode chain, as data.

    A configuration object is the one place a long field list is the right
    shape: the alternative is several small ones that always travel together.

    Frozen and hashable, so it can key a per-image cache: decoding a 15 MB
    ``.ORF`` dominates any batch that does it more than once.

    The defaults are the recommended tone chain, **not** the chain production
    ran before this class existed. :meth:`production` returns that one.
    """

    #: Where the white point comes from. Unlike everything else here this
    #: lives inside ``rawpy.postprocess``, so it reaches the linear decode —
    #: i.e. the laser detector's input — as well as the JPEG. That asymmetry
    #: is why it is the one knob that cannot be treated as cosmetic.
    white_balance: WhiteBalance = WhiteBalance.CAMERA

    #: Target mean V for the auto-gamma lift, in 0-255. 20 is what production
    #: has always used — deliberately dark, and the reason the old CLAHE step
    #: had so much noise to amplify.
    auto_gamma_target: int = 20

    #: Global percentile contrast stretch, applied after the auto-gamma lift.
    #:
    #: The tool the underwater problem actually calls for. Attenuation and
    #: backscatter leave the scene occupying a narrow slice of the container's
    #: range — which is why everything reads flat — and backscatter in
    #: particular is roughly *additive*, a veiling floor over the whole frame.
    #: Subtracting a black point and rescaling removes that floor; a gain-only
    #: white balance structurally cannot, because it has no offset term.
    #:
    #: Unlike CLAHE it is ONE affine map over the whole frame, so a flat region
    #: stays flat *relative to the scene* instead of being handed its own local
    #: histogram and stretched on its own. That is the difference between
    #: recovering contrast and manufacturing it out of sensor noise.
    #:
    #: ``"luminance"`` (the default) maps CIELAB L* only, expanding contrast
    #: while leaving the colour cast alone. ``"per_channel"`` maps each channel
    #: independently, which also removes the cast — and is one of the variants
    #: that failed on field frames, so it is available rather than advised.
    stretch_mode: str = "luminance"
    #: Percentile mapped to black. Percentiles rather than min/max so one hot
    #: pixel or one specular glint cannot set the whole frame's mapping.
    #:
    #: Accepts a single value for all channels, or an ``(R, G, B)`` triple.
    #: The triple exists because the three channels are in completely
    #: different states underwater: red is a narrow, noise-dominated band while
    #: blue is broad, so one pair of percentiles for all three is the wrong
    #: shape of knob.
    #:
    #: A triple is only meaningful with ``stretch_mode="per_channel"``. The
    #: luminance stretch maps CIELAB L*, which is one channel, so there is
    #: nothing for the other two entries to apply to — that combination is
    #: refused rather than quietly using the first entry for everything.
    stretch_low: float | tuple[float, float, float] = 1.0
    #: Percentile mapped to white. Same single-or-triple rule.
    stretch_high: float | tuple[float, float, float] = 99.0

    #: Local histogram equalisation. **Off by default**, which is the change
    #: this class exists to make; see the module docstring for the measurement.
    clahe_enabled: bool = False
    #: Which channel CLAHE equalizes, when it is enabled at all.
    #:
    #: ``"value"`` hands the RGB frame straight to ``equalize_adapthist``,
    #: which — via ``@adapt_rgb(hsv_value)`` — equalizes HSV **V** and restores
    #: hue and saturation unchanged. That is what production ran, and it is the
    #: default here only so ``DecodeConfig.production()`` stays exactly
    #: reproducible. Holding *saturation* fixed is what makes it amplify chroma
    #: noise in step with luma noise; see the module docstring.
    #:
    #: ``"luminance"`` equalizes CIELAB **L\*** instead, holding a* and b*
    #: fixed in absolute terms, so local contrast lifts without the water
    #: turning coloured.
    #:
    #: (The mode was called ``"per_channel"`` in the evaluation harness this
    #: was ported from, on the belief that skimage equalised each channel
    #: separately. It does not. The name is corrected here rather than carried
    #: across, because it was the name that encoded the error.)
    clahe_mode: str = "value"
    #: ``None`` means skimage's default (0.01). Lower clips harder.
    clahe_clip_limit: float | None = None
    #: ``None`` means skimage's default (1/8 of each axis).
    clahe_kernel_size: int | None = None

    #: Additive gain applied to red where the *spatially coherent* red excess
    #: is high. 0.0 is off, and off is the default.
    #:
    #: This synthesises emphasis rather than restoring signal, which makes it a
    #: different kind of thing from every other knob here. Defensible as a
    #: target indicator on species and head/tail frames, where the laser marks
    #: which fish is being measured; circular if it were ever pointed at the
    #: laser-labeling task itself. Measured: laser-dot SNR 2.68 → 4.16 in a
    #: pool and 1.28 → 1.74 on a reef.
    red_boost: float = 0.0
    #: Local red-excess *contrast* above which the boost starts to apply.
    #:
    #: Deliberately a local contrast, not an absolute level. In the old decode
    #: the laser dot's absolute red excess ``R - max(G, B)`` is still
    #: **negative** (about -0.07, against -0.49 in the surrounding water),
    #: because the whole scene is cyan — so any absolute threshold either never
    #: fires or fires everywhere. What marks the dot is being redder than its
    #: neighbourhood.
    red_boost_threshold: float = 0.02
    #: Blur applied to the red-excess map before thresholding. This is the part
    #: that separates a coherent dot from pixel-scale speckle.
    red_boost_sigma: float = 1.5
    #: Radius of the local baseline the dot is measured against. With
    #: ``red_boost_sigma`` this is a difference-of-Gaussians blob detector on
    #: the red-excess map.
    red_boost_background_sigma: float = 12.0
    #: Threshold in robust sigmas of the blob response itself. When non-zero
    #: this replaces the absolute ``red_boost_threshold``.
    #:
    #: A fixed level cannot work across these scenes: the laser dot sits at the
    #: 99.999th percentile of red on a reef and at the **58.8th** in a pool,
    #: where a white shirt, skin and a painted model are all redder.
    #: Referencing the cut to the response's own noise makes it scene-adaptive.
    red_boost_sigmas: float = 0.0
    #: Red-excess span, above the threshold, over which the boost ramps to
    #: full. Set from the quantity's real range, not from 1.0: red excess runs
    #: about -0.4 in open water to +0.1 at a laser dot, so normalising over
    #: ``[threshold, 1.0]`` leaves the ramp permanently near zero.
    red_boost_span: float = 0.05

    #: Per-channel attenuation coefficients ``(R, G, B)`` per metre, for the
    #: Sea-thru correction. ``None`` is off, and off is the default.
    #:
    #: Sea-thru cannot be a default even in principle: ``RawImage(raw_bytes)``
    #: has no idea which dive a frame came from or how far away the subject
    #: was, and the correction needs both. Fit these per dive with
    #: :func:`fishsense_core.water.attenuation.fit_attenuation`; ``beta`` and
    #: ``range_m`` are only meaningful together, and supplying one without the
    #: other is refused.
    beta: tuple[float, float, float] | None = None
    #: Metric range to the subject, in metres — per image. See ``beta``.
    #:
    #: Applied uniformly across the frame, which is right for the subject the
    #: range was measured at and increasingly wrong for background at another
    #: distance. That is the ceiling on this correction, and lifting it needs a
    #: per-pixel depth map.
    range_m: float | None = None

    #: Denoising, applied at the very end of the chain. ``None`` is off, and
    #: off is the default — by a wide margin.
    #:
    #: A :class:`~fishsense_core.image.denoise.BM3DConfig`. It costs 15-36x the
    #: entire decode (133.8 s per 12-megapixel frame at the ``np`` profile,
    #: 56.8 s at ``lc``, against 3.7 s for the whole chain), so it cannot sit
    #: inline until that is solved. Typed loosely here so
    #: :mod:`fishsense_core.image.denoise` — and with it the optional ``bm3d``
    #: dependency — is imported only when something asks for it.
    denoise: Any | None = None

    #: Percentile for ``WHITE_PATCH`` and ``SLATE`` gain estimation.
    wb_percentile: float = 99.0
    #: Estimate white-balance gains from a half-resolution decode, then run the
    #: full decode once with the resulting ``user_wb``.
    #:
    #: **Off by default, because it is not free.** The reasoning for it — gains
    #: are a global statistic, so a quarter-cost pass moves them negligibly —
    #: does not survive measurement here. ``half_size`` skips the demosaic and
    #: averages each 2x2 Bayer cell instead, which changes the noise floor of
    #: exactly the channel that sits on it. On this repository's fixture the
    #: red multiplier shifts by 14-15% for every estimator: gray-world
    #: 22.62 -> 19.51, white-patch 6.58 -> 5.69, slate 6.84 -> 5.78.
    #:
    #: It is worse than that for ``WhiteBalance.SLATE``, which is not a global
    #: statistic at all — it is a percentile inside one quad, and at half size
    #: that quad has a quarter of the pixels to take a percentile of.
    #:
    #: Kept as a knob because the speed is real and a caller sweeping many
    #: frames may want it; a 14% error in a gain nobody recommends using is not
    #: the same kind of problem as a 14% error in the shipped decode.
    wb_estimate_half_size: bool = False

    def __post_init__(self) -> None:
        if self.auto_gamma_target <= 0:
            raise ValueError(
                f"auto_gamma_target must be positive, got {self.auto_gamma_target}: "
                "it is compared in log space, so a non-positive target makes "
                "math.log raise inside the decode"
            )
        if self.clahe_clip_limit is not None and not 0 < self.clahe_clip_limit <= 1:
            raise ValueError(
                f"clahe_clip_limit must be in (0, 1], got {self.clahe_clip_limit}"
            )
        if self.clahe_kernel_size is not None and self.clahe_kernel_size < 2:
            raise ValueError(
                f"clahe_kernel_size must be at least 2, got {self.clahe_kernel_size}"
            )
        if self.clahe_mode not in CLAHE_MODES:
            raise ValueError(
                f"clahe_mode must be one of {CLAHE_MODES}, got {self.clahe_mode!r}"
            )
        if self.stretch_mode not in STRETCH_MODES:
            raise ValueError(
                f"stretch_mode must be one of {STRETCH_MODES}, got {self.stretch_mode!r}"
            )
        self._validate_stretch_percentiles()
        if self.red_boost < 0:
            raise ValueError(f"red_boost must be >= 0, got {self.red_boost}")
        if self.red_boost_sigmas < 0:
            raise ValueError(
                f"red_boost_sigmas must be >= 0, got {self.red_boost_sigmas}"
            )
        self._validate_seathru()

    def _validate_seathru(self) -> None:
        """``beta`` and ``range_m`` are meaningful only together.

        Silently ignoring one of them would produce a frame that looks
        corrected and is not, which is worse than refusing.
        """
        if (self.beta is None) != (self.range_m is None):
            raise ValueError(
                "beta and range_m must be given together: beta is the water's "
                "attenuation per metre and range_m is how many metres, and "
                f"neither means anything alone (got beta={self.beta!r}, "
                f"range_m={self.range_m!r})"
            )
        if self.beta is not None:
            if len(self.beta) != 3:
                raise ValueError(
                    f"beta must be three per-channel coefficients (R, G, B), "
                    f"got {self.beta!r}"
                )
            if any(b < 0 for b in self.beta):
                raise ValueError(
                    f"beta must be non-negative — it is light lost per metre, so "
                    f"a negative coefficient amplifies with distance: {self.beta!r}"
                )
        if self.range_m is not None and self.range_m < 0:
            raise ValueError(f"range_m must be non-negative, got {self.range_m}")

    def _validate_stretch_percentiles(self) -> None:
        for name, value in (
            ("stretch_low", self.stretch_low),
            ("stretch_high", self.stretch_high),
        ):
            if isinstance(value, tuple) and len(value) != 3:
                raise ValueError(
                    f"{name} must be a number or an (R, G, B) triple, got {value!r}"
                )
        if self.stretch_mode != "per_channel" and (
            isinstance(self.stretch_low, tuple) or isinstance(self.stretch_high, tuple)
        ):
            raise ValueError(
                "an (R, G, B) percentile triple needs stretch_mode='per_channel'; "
                f"stretch_mode={self.stretch_mode!r} maps a single channel "
                "(CIELAB L*, or nothing at all), so two of the three entries "
                "would have nowhere to apply"
            )
        lows = self.stretch_low if isinstance(self.stretch_low, tuple) else (self.stretch_low,) * 3
        highs = (
            self.stretch_high if isinstance(self.stretch_high, tuple)
            else (self.stretch_high,) * 3
        )
        for low, high in zip(lows, highs):
            if not 0.0 <= low < high <= 100.0:
                raise ValueError(
                    "stretch percentiles must satisfy 0 <= low < high <= 100, got "
                    f"low={low}, high={high}"
                )

    @classmethod
    def production(cls) -> "DecodeConfig":
        """The chain shipped before this class existed.

        Auto-gamma to a mean V of 20, then ``equalize_adapthist`` at skimage's
        defaults. Kept constructible, and pinned in the tests, so the change of
        default is a thing you can measure rather than a thing you have to take
        on trust.
        """
        return cls(stretch_mode="off", clahe_enabled=True, clahe_mode="value")

    @property
    def label(self) -> str:
        """Short, stable, filename-safe name for reports and cache keys.

        Names how this config differs from the defaults, so the recommended
        chain is ``"default"`` and the old one is ``"noStretch-clahe"``.
        """
        parts: list[str] = []
        if self.white_balance is not WhiteBalance.CAMERA:
            parts.append(self.white_balance.value)
        if self.auto_gamma_target != 20:
            parts.append(f"gamma{self.auto_gamma_target}")
        if self.stretch_mode != "luminance":
            parts.append("noStretch" if self.stretch_mode == "off" else "stretchPc")
        if self.stretch_mode != "off" and (
            self.stretch_low != 1.0 or self.stretch_high != 99.0
        ):
            # "_" rather than "/" between the members of a triple: this string
            # is used as a filename stem.
            def fmt(value) -> str:
                if isinstance(value, tuple):
                    return "_".join(f"{x:g}" for x in value)
                return f"{value:g}"

            parts.append(f"p{fmt(self.stretch_low)}-{fmt(self.stretch_high)}")
        if self.clahe_enabled:
            parts.append("clahe" if self.clahe_mode == "value" else "lumaclahe")
            if self.clahe_clip_limit is not None:
                parts.append(f"clip{self.clahe_clip_limit:g}")
            if self.clahe_kernel_size is not None:
                parts.append(f"kernel{self.clahe_kernel_size}")
        if self.red_boost:
            parts.append(
                f"redboost{self.red_boost:g}"
                + (f"@{self.red_boost_sigmas:g}s" if self.red_boost_sigmas else "")
            )
        if self.beta is not None:
            parts.append(
                "seathru" + "_".join(f"{b:g}" for b in self.beta)
                + f"@{self.range_m:g}m"
            )
        if self.denoise is not None:
            parts.append(f"denoise{getattr(self.denoise, 'strength', ''):g}".rstrip())
        return "-".join(parts) if parts else "default"


# ---------------------------------------------------------------------------
# rawpy
# ---------------------------------------------------------------------------


def postprocess_kwargs(config: DecodeConfig, camera_gains: Sequence[float] | None) -> dict:
    """rawpy keywords for one decode.

    Everything except white balance is pinned. ``user_flip=0`` in particular is
    not optional: EXIF rotation would change the frame's shape and invalidate
    every label coordinate that has ever been recorded against it.
    """
    kwargs: dict = {
        "gamma": (1, 1),
        "no_auto_bright": True,
        "output_bps": 16,
        # rawpy's default, stated explicitly because both decodes depend on it
        # and one of them (the linear stage) used to say so while the other
        # relied on the default. Same value, one place.
        "output_color": rawpy.ColorSpace.sRGB,
        "user_flip": 0,
    }
    if config.white_balance is WhiteBalance.CAMERA:
        kwargs["use_camera_wb"] = True
    elif config.white_balance is WhiteBalance.RAWPY_AUTO:
        kwargs["use_auto_wb"] = True
    else:
        if camera_gains is None:
            raise ValueError(
                f"{config.white_balance} needs gains resolved by "
                "resolve_white_balance() before postprocess"
            )
        kwargs["user_wb"] = list(camera_gains)
    return kwargs


def _decode_rgb16(
    source: Path | bytes,
    config: DecodeConfig,
    camera_gains: Sequence[float] | None,
    *,
    half: bool = False,
) -> np.ndarray:
    """Raw source -> linear uint16 RGB, in sensor coordinates."""
    kwargs = postprocess_kwargs(config, camera_gains)
    if half:
        kwargs["half_size"] = True
    with open_image_source(source) as handle:
        with rawpy.imread(handle) as raw:
            return raw.postprocess(**kwargs)


def resolve_white_balance(
    source: Path | bytes,
    config: DecodeConfig,
    *,
    slate_quad: Sequence[Sequence[float]] | None = None,
) -> list[float] | None:
    """Absolute rawpy ``user_wb`` multipliers, or ``None`` for camera/auto WB.

    Estimated as *camera multipliers × correction*, where the correction comes
    from a camera-WB decode of this same frame. Composing rather than replacing
    matters: it holds the colour matrix, demosaic and black levels fixed, so
    the difference between two configs is the white point and nothing else.

    The probe decode is full-resolution unless
    :attr:`DecodeConfig.wb_estimate_half_size` says otherwise — see that
    field for what the cheaper pass costs.
    """
    if config.white_balance in (WhiteBalance.CAMERA, WhiteBalance.RAWPY_AUTO):
        return None

    with open_image_source(source) as handle:
        with rawpy.imread(handle) as raw:
            camera_wb = [float(v) for v in raw.camera_whitebalance]

    # The correction is measured against a camera-WB decode, so use one.
    probe_bgr = _decode_rgb16(
        source, DecodeConfig(), None, half=config.wb_estimate_half_size
    )[:, :, ::-1]

    if config.white_balance is WhiteBalance.GRAY_WORLD:
        correction = gray_world_gains(probe_bgr)
    elif config.white_balance is WhiteBalance.WHITE_PATCH:
        correction = white_patch_gains(probe_bgr, percentile=config.wb_percentile)
    elif config.white_balance is WhiteBalance.SLATE:
        if slate_quad is None:
            raise ValueError(
                "WhiteBalance.SLATE needs the image's slate quad "
                "(DiveSlateLabel.slate_rectangle); none was supplied"
            )
        quad = np.asarray(slate_quad, dtype=float)
        if config.wb_estimate_half_size:
            quad = quad / 2.0
        correction = slate_patch_gains(probe_bgr, quad, percentile=config.wb_percentile)
    else:  # pragma: no cover - the enum is closed
        raise ValueError(f"unhandled white balance {config.white_balance}")

    # camera_whitebalance is [R, G, B, G2]; the correction is (R, G, B).
    red, green, blue = correction
    g1 = camera_wb[1] * green
    g2 = (camera_wb[3] if len(camera_wb) > 3 and camera_wb[3] else camera_wb[1]) * green
    _log.debug("white balance %s: correction=%s", config.white_balance.value, correction)
    return [camera_wb[0] * red, g1, camera_wb[2] * blue, g2]


# ---------------------------------------------------------------------------
# The tone chain
# ---------------------------------------------------------------------------


def auto_gamma(img: np.ndarray, target: int) -> np.ndarray:
    """Lift a dark linear frame toward a mean V of ``target``.

    ``img`` is float RGB in [0, 1]. The exponent is derived from the frame's
    own mean V, including the ``* 255`` that makes "a target mean brightness of
    20" a good deal darker than it sounds. Preserved exactly as production had
    it: ``target`` moves the 20, not the formula.
    """
    # V of HSV is max(channels), so the BGR->HSV conversion on an RGB array is
    # harmless — and it is what production does. Reproduced rather than
    # corrected, because the point is to be bit-compatible with it.
    hsv = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2HSV)
    _, _, val = cv2.split(hsv)
    mean = float(np.mean(val))

    if mean <= 1.0:
        # log(mean) is 0 at mean == 1 and undefined below it, so the original
        # expression raises ZeroDivisionError or ValueError from inside the
        # decode. A frame this dark carries nothing to lift.
        raise ValueError(
            f"frame is too dark to auto-gamma: mean V is {mean:.4f}, and the "
            "exponent is 1 / (log(target * 255) / log(mean))"
        )

    gamma = 1 / (math.log(target * 255) / math.log(mean))
    _log.debug("auto-gamma: mean_brightness=%.2f gamma=%.4f", mean, gamma)
    return adjust_gamma(img, gamma=gamma)


def apply_stretch(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Global percentile contrast stretch. ``img`` is float RGB in [0, 1].

    One affine map over the whole frame (or over L*), derived from percentiles
    of the frame itself. That globality is the entire point — see
    :attr:`DecodeConfig.stretch_mode`.
    """
    if config.stretch_mode == "off":
        return img

    def _percentile_for(value, channel: int) -> float:
        # A tuple only reaches here from the per_channel branch below —
        # DecodeConfig refuses a triple in any other mode, precisely so that
        # this cannot silently return the R entry for all three.
        return value[channel] if isinstance(value, tuple) else value

    def _map(plane: np.ndarray, channel: int = 0) -> np.ndarray:
        low = float(np.percentile(plane, _percentile_for(config.stretch_low, channel)))
        high = float(np.percentile(plane, _percentile_for(config.stretch_high, channel)))
        if high - low < 1e-6:
            # A genuinely flat plane has no range to expand; stretching it
            # would turn its noise into the entire signal.
            return plane
        return np.clip((plane - low) / (high - low), 0.0, 1.0)

    if config.stretch_mode == "per_channel":
        out = np.empty_like(img)
        for channel in range(img.shape[2]):
            out[:, :, channel] = _map(img[:, :, channel], channel)
        return out

    lab = rgb2lab(img)
    lab[:, :, 0] = _map(lab[:, :, 0] / 100.0) * 100.0
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def apply_clahe(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Local contrast enhancement, in the mode ``config`` selects.

    ``img`` is float RGB in [0, 1] — the same array production handed
    ``equalize_adapthist``.
    """
    if not config.clahe_enabled:
        return img

    kwargs = {}
    if config.clahe_clip_limit is not None:
        kwargs["clip_limit"] = config.clahe_clip_limit
    if config.clahe_kernel_size is not None:
        kwargs["kernel_size"] = config.clahe_kernel_size

    if config.clahe_mode == "value":
        # skimage's own RGB handling: rgb2hsv -> CLAHE on V -> hsv2rgb, with
        # hue and saturation restored untouched.
        return equalize_adapthist(img, **kwargs)

    lab = rgb2lab(img)
    # L* is 0-100; equalize_adapthist wants a unit-range float, and rescaling
    # back by the same constant keeps the round trip exact for an untouched
    # image.
    lab[:, :, 0] = equalize_adapthist(lab[:, :, 0] / 100.0, **kwargs) * 100.0
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def apply_red_boost(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Lift red where the *spatially coherent* red excess is high.

    The laser dot is a coherent blob roughly ten pixels across; the red speckle
    a per-channel stretch amplifies is pixel-scale. Blurring the red-excess map
    before thresholding is what tells them apart — the blob survives the blur,
    the speckle averages away — so the gain lands on the dot and not the noise.

    Reads a neighbourhood but moves no pixel.
    """
    if not config.red_boost:
        return img

    red = img[:, :, 0]
    excess = red - np.maximum(img[:, :, 1], img[:, :, 2])
    # Difference of Gaussians on the red-excess map. The dot's *absolute*
    # excess is negative in these frames (the scene is cyan), so what is
    # thresholded has to be its excess relative to the water around it.
    coherent = gaussian_filter(excess, sigma=config.red_boost_sigma) - gaussian_filter(
        excess, sigma=config.red_boost_background_sigma
    )

    if config.red_boost_sigmas:
        # Robust sigma of the blob response, so the cut scales with whatever
        # noise this frame's red channel carries. MAD rather than std because
        # the dot itself is exactly the kind of outlier a std would absorb,
        # inflating the threshold until nothing passes.
        centre = float(np.median(coherent))
        sigma = 1.4826 * float(np.median(np.abs(coherent - centre)))
        threshold = centre + config.red_boost_sigmas * sigma
        span = max(config.red_boost_sigmas * sigma, 1e-6)
    else:
        threshold = config.red_boost_threshold
        # Soft ramp rather than a hard cut, so the dot does not acquire a
        # stamped edge a labeler could mistake for a real boundary.
        span = max(config.red_boost_span, 1e-6)

    weight = np.clip((coherent - threshold) / span, 0.0, 1.0)
    out = img.copy()
    out[:, :, 0] = np.clip(red + config.red_boost * weight, 0.0, 1.0)
    return out


def apply_seathru(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Invert the water column, given measured attenuation and a range.

    ``img`` is **linear** float RGB in [0, 1], straight out of
    ``rawpy.postprocess``. That placement is forced:
    :func:`~fishsense_core.water.seathru.remove_water` inverts a radiance
    formation model, and applying it after the auto-gamma would invert a curve
    that is not in the model.

    Off unless ``config.beta`` and ``config.range_m`` are both set, which they
    cannot be by default — see :attr:`DecodeConfig.beta`.

    Note ``normalize=False``. ``remove_water``'s own rescale-to-peak is for
    looking at the result on its own; inside this chain it is actively harmful,
    because the auto-gamma immediately downstream derives its exponent from the
    frame's mean brightness. Renormalising first makes the frame brighter, the
    auto-gamma then lifts less, and the starved red channel — which the
    inversion had just doubled relative to green — is pulled up less than it
    would have been. Measured on this repository's fixture at beta =
    (0.263, 0.040, 0.001) and 3 m: the linear R/G ratio improves 0.130 -> 0.258
    either way, but by the end of the chain ``normalize=True`` lands at 0.646
    against the uncorrected frame's 0.700, i.e. worse than doing nothing, while
    ``normalize=False`` lands at 0.706.

    (0.706 against 0.700 is also the honest headline: on a close-range pool
    frame the tone chain compresses away nearly all of the physics. The
    correction scales with range and earns its place further out.)
    """
    if config.beta is None or config.range_m is None:
        return img

    # Imported here so `fishsense_core.water` stays off the import path of
    # every decode that does not use it.
    # pylint: disable-next=import-outside-toplevel
    from fishsense_core.water.seathru import remove_water  # noqa: PLC0415

    # Clipped because dividing by the transmission can push a bright pixel in a
    # strongly attenuated channel above 1.0, and everything downstream of here
    # assumes [0, 1].
    return np.clip(
        remove_water(img, config.beta, config.range_m, normalize=False), 0.0, 1.0
    )


def apply_denoise(bgr: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Run the configured denoiser over a finished uint8 BGR frame.

    Last in the chain, because the enhancer is defined over a finished frame.
    Off by default and expensive enough that it cannot yet be anything else —
    see :attr:`DecodeConfig.denoise`.
    """
    if config.denoise is None:
        return bgr

    # pylint: disable-next=import-outside-toplevel
    from fishsense_core.image.denoise import bm3d_enhancer  # noqa: PLC0415

    # The enhancer is written over RGB; the decode's own order is BGR.
    return bm3d_enhancer(config.denoise)(bgr[:, :, ::-1])[:, :, ::-1]


# ---------------------------------------------------------------------------
# The two stages
# ---------------------------------------------------------------------------


def decode_rectified_stage(
    source: Path | bytes,
    config: DecodeConfig | None = None,
    *,
    slate_quad: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Raw source -> uint8 BGR: the labeler-facing chain, before undistortion.

    Kept in the same order and the same dtypes throughout: ``img_as_ubyte``
    round trips are lossy, so reordering the steps or skipping one changes the
    output even where the arithmetic looks equivalent.
    """
    config = config or DecodeConfig()
    camera_gains = resolve_white_balance(source, config, slate_quad=slate_quad)

    img = img_as_float(_decode_rgb16(source, config, camera_gains))
    img = apply_seathru(img, config)
    img = auto_gamma(img, config.auto_gamma_target)
    img = apply_stretch(img, config)
    img = apply_clahe(img, config)
    img = apply_red_boost(img, config)

    return apply_denoise(img_as_ubyte(img[:, :, ::-1]), config)


def decode_linear_stage(
    source: Path | bytes,
    config: DecodeConfig | None = None,
    *,
    slate_quad: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Raw source -> uint16 linear BGR, in sensor coordinates.

    The laser detector's decode. **Only white balance reaches here**: the
    auto-gamma, the stretch, CLAHE and the red boost all live in the JPEG chain
    only, which is why a change to the labeler-facing decode cannot move laser
    detection at all — not a small effect needing a large sample to detect,
    exactly zero.

    ``config`` is accepted so that asymmetry is *expressible* and therefore
    testable, not because varying it is a good idea.
    :class:`~fishsense_core.image.linear_raw_image.LinearRawImage` — what
    production actually calls — exposes no such knob, and a white-balance
    change here is a systematic covariate shift on the detector's primary
    input rather than a mild perturbation: the checkpoint was trained on
    ``use_camera_wb=True``, and the detector's first three channels are
    ``rgb / sum(rgb)``, which is per-pixel scale-invariant and so sees channel
    *ratios* and nothing else.
    """
    config = config or DecodeConfig()
    camera_gains = resolve_white_balance(source, config, slate_quad=slate_quad)
    rgb = _decode_rgb16(source, config, camera_gains)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def rectify(
    image: np.ndarray,
    camera_matrix: np.ndarray | Sequence[Sequence[float]],
    distortion: np.ndarray | Sequence[float],
) -> np.ndarray:
    """``cv2.undistort`` with the camera's intrinsics.

    The last step before a JPEG is encoded, and the step that puts pixels in
    label space. Size-preserving, which is why laser pixels, head/tail pixels
    and the JPEG are all one coordinate system.

    Takes plain arrays rather than a ``CameraIntrinsics``, so it is usable
    without the API SDK — which is an optional extra, see
    :class:`~fishsense_core.image.rectified_image.RectifiedImage`.
    """
    return cv2.undistort(
        image,
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
    )
