"""Fitting water attenuation coefficients against the dive slate.

The obstacle to Sea-thru everywhere else is that it needs a per-pixel metric
range map. This corpus has a metric range at one pixel per frame — the laser's
— and, in slate frames, a **known constant-reflectance target** at a known
place in the image.

That combination is what makes absolute coefficients recoverable. For a target
of fixed reflectance rho_c seen at range z::

    log(I_c) = log(J_c * rho_c) - beta_c * z + (backscatter)

so regressing log intensity on range across frames of one dive estimates
``beta_c`` itself. An earlier annulus-around-the-laser-dot method could not do
this: it sampled a different scene in every frame, so rho_c varied and only
*ratios* between channels were interpretable. It recovered the right sign on
all three field dives and the right null in the pool, but magnitudes varied
3.7x with r² of 0.18-0.34 — and the variance was exactly the reflectance term.

Assumptions, stated because they bound what the number means:

* **Reflectance is constant.** The slate is one printed sheet, so this holds
  across frames of a dive far better than it holds across scenes — but the
  bright tail inside the quad is used rather than its mean, because the mean is
  a paper/ink mixture that shifts with how much artwork is in view.
* **Illumination is constant within a dive.** It is not exactly: ambient light
  falls with depth and shifts with time of day. Residual illumination drift
  inflates the scatter and, if it correlates with range, biases the slope.
* **Backscatter is neglected.** The full model adds a veiling term that
  saturates with range, which flattens the curve at distance. Over the 1-3.7 m
  the slate frames span this is second-order; over longer ranges it is not, and
  a straight line would understate beta.

**The pool dives are the negative control.** Clear water has almost no
differential attenuation over 2 m, so a correct method must return
approximately the pure-water spectrum there and no more.

Validated: seven pool dives, 218 usable slate samples, r² 0.86-0.96, giving
beta_r +0.263, beta_g +0.040, beta_b +0.001 (pool medians) against Pope & Fry
1997's 0.24-0.34, 0.057 and 0.009 /m — with the expected
beta_r > beta_g > beta_b ordering in 7 of 8 fits. This is the only result in
the work this was ported from that is validated against an external standard
rather than against itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# `as_polygon` normalizes a `DiveSlateLabel.slate_rectangle`, which the
# white-balance estimator in `image.decode` needs too. Defined there and
# re-exported here so there is one implementation and no import cycle; this
# module is its natural home to read about.
from fishsense_core.image.decode import as_polygon

__all__ = [
    "MIN_RANGE_SPAN_M",
    "MIN_SAMPLES",
    "AttenuationFit",
    "LinearFit",
    "as_polygon",
    "fit_attenuation",
    "fit_loglinear",
    "sample_slate_patch",
]

#: Minimum range spread, in metres, for a slope to mean anything. Attenuation
#: is a slope against distance; without a lever there is nothing to measure,
#: and any number reported is noise dressed as a coefficient.
MIN_RANGE_SPAN_M = 0.4
MIN_SAMPLES = 8


@dataclass(frozen=True)
class LinearFit:
    """One ordinary least-squares slope, with enough to judge it."""

    slope: float
    intercept: float
    stderr: float
    r2: float
    n: int
    ci_low: float
    ci_high: float

    @property
    def spans_zero(self) -> bool:
        """True when the data cannot distinguish this slope from none.

        The property that lets the method say "no attenuation here" instead of
        inventing a correction for clear water.
        """
        return self.ci_low <= 0.0 <= self.ci_high


def fit_loglinear(z: Sequence[float], y: Sequence[float]) -> LinearFit | None:
    """OLS of ``y`` on ``z``. None when the data cannot support a slope."""
    n = len(z)
    if n < MIN_SAMPLES or n != len(y):
        return None
    if max(z) - min(z) < 1e-9:
        return None

    mean_z = sum(z) / n
    mean_y = sum(y) / n
    szz = sum((v - mean_z) ** 2 for v in z)
    if szz <= 0:
        return None
    slope = sum((a - mean_z) * (b - mean_y) for a, b in zip(z, y)) / szz
    intercept = mean_y - slope * mean_z

    residuals = [b - (intercept + slope * a) for a, b in zip(z, y)]
    sse = sum(e * e for e in residuals)
    sst = sum((b - mean_y) ** 2 for b in y)
    stderr = math.sqrt(sse / (n - 2) / szz) if n > 2 else float("nan")
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    # 1.96 rather than a t quantile: n is comfortably above 30 in every real
    # cohort here, and pretending to more precision than the model deserves
    # would be false comfort.
    return LinearFit(
        slope, intercept, stderr, r2, n,
        slope - 1.96 * stderr, slope + 1.96 * stderr,
    )


def sample_slate_patch(
    bgr: np.ndarray,
    quad: Sequence[Sequence[float]],
    *,
    percentile: float = 85.0,
    min_pixels: int = 400,
) -> tuple[float, float, float] | None:
    """Mean ``(R, G, B)`` of the slate's paper inside ``quad``, on a LINEAR frame.

    The bright tail rather than the mean: the slate is white paper carrying
    black markings, so its mean is a paper/ink mixture that shifts with how
    much artwork is in view, while the tail is the paper — and paper being one
    constant reflectance is the assumption the whole method rests on.

    Returns None when the quad is too small to sample or the slate is
    saturated. A blown-out slate sits at the container ceiling no matter how
    much water it was seen through, so it carries no attenuation information at
    all; including it would flatten every slope toward zero. Roughly 40% of the
    existing frames were rejected on this test.
    """
    import cv2  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    polygon = as_polygon(quad)
    if polygon is None:
        return None
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(  # pylint: disable=no-member
        mask, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], 255
    )
    inside = mask.astype(bool)
    if int(inside.sum()) < min_pixels:
        return None

    full_scale = (
        float(np.iinfo(bgr.dtype).max)
        if np.issubdtype(bgr.dtype, np.integer)
        else 1.0
    )
    pixels = bgr[inside].astype(np.float64)
    usable = pixels.max(axis=1) < full_scale * 0.98
    if usable.sum() < min_pixels // 2:
        return None

    pixels = pixels[usable]
    luminance = pixels.mean(axis=1)
    cutoff = float(np.percentile(luminance, percentile))
    tail = pixels[luminance >= cutoff]
    if len(tail) < 8:
        return None
    # BGR in, (R, G, B) out.
    return float(tail[:, 2].mean()), float(tail[:, 1].mean()), float(tail[:, 0].mean())


@dataclass(frozen=True)
class AttenuationFit:  # pylint: disable=too-many-instance-attributes
    """Per-dive attenuation, per channel, with the fits that produced it.

    The per-channel fits are carried alongside the coefficients rather than
    summarised away: a beta whose confidence interval spans zero is not a
    measurement, and :attr:`trustworthy` needs the fits to say so.
    """

    beta_r: float
    beta_g: float
    beta_b: float
    beta_r_fit: LinearFit
    beta_g_fit: LinearFit
    beta_b_fit: LinearFit
    #: Differential coefficients, directly comparable with the earlier
    #: annulus-around-the-dot result, which could only measure these.
    d_rg: float
    d_bg: float
    n: int
    range_span: float

    @property
    def beta(self) -> tuple[float, float, float]:
        """``(R, G, B)``, in the order :func:`~fishsense_core.water.seathru.remove_water`
        wants."""
        return (self.beta_r, self.beta_g, self.beta_b)

    @property
    def ordering_ok(self) -> bool:
        """Whether the result obeys the expected underwater ordering.

        Red is absorbed fastest and blue least, so beta_r > beta_g > beta_b is
        what physics predicts. A fit that violates it is reporting something
        other than water — illumination drift, a mis-sampled quad, or noise.
        """
        return self.beta_r > self.beta_g > self.beta_b

    @property
    def trustworthy(self) -> bool:
        """Enough lever, enough samples, and a red slope distinguishable from
        zero."""
        return (
            self.range_span >= MIN_RANGE_SPAN_M
            and self.n >= 20
            and not self.beta_r_fit.spans_zero
        )


def fit_attenuation(samples: Sequence[Mapping[str, Any]]) -> AttenuationFit | None:
    """Fit beta per channel from slate observations of one dive.

    ``samples`` carry ``range_m`` and linear ``R``, ``G``, ``B`` — i.e. the
    output of :func:`sample_slate_patch` paired with a range. The sign
    convention: beta is positive for light that is lost with distance, so it is
    the *negation* of the slope of log intensity on range.
    """
    usable = [
        s for s in samples
        if s.get("range_m") and min(float(s["R"]), float(s["G"]), float(s["B"])) > 0
    ]
    if len(usable) < MIN_SAMPLES:
        return None

    z = [float(s["range_m"]) for s in usable]
    fits = {
        key: fit_loglinear(z, [math.log(float(s[key])) for s in usable])
        for key in ("R", "G", "B")
    }
    if any(fit is None for fit in fits.values()):
        return None

    beta = {key: -fit.slope for key, fit in fits.items()}
    return AttenuationFit(
        beta_r=beta["R"], beta_g=beta["G"], beta_b=beta["B"],
        beta_r_fit=fits["R"], beta_g_fit=fits["G"], beta_b_fit=fits["B"],
        d_rg=beta["R"] - beta["G"], d_bg=beta["B"] - beta["G"],
        n=len(usable), range_span=max(z) - min(z),
    )
