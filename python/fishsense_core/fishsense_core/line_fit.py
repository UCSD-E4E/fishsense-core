"""Per-dive RANSAC line fit on positive laser labels.

This module is the source of truth. It was moved verbatim from
fishsense-lite@a8b2c3bc
(``services/fishsense-data-processing-workflow-worker/.../laser_label_validation/line_fit.py``),
which had vendored it from
https://github.com/UCSD-E4E/2026-05-02_laser_detector
``src/laser_detector/preprocessing/line_fit.py`` @ 3d5d2e8 (2026-05-02).

The numbers are load-bearing: v2's migrated ``dive_laser_lines`` rows were
computed by the fishsense-lite copy, and ``tests/test_line_fit_golden.py``
pins them. A change that moves them breaks parity with those rows. Two
deliberate breaks:

* ``label_noise_mad``, which fishsense-lite estimated from absolute distances
  (~0.59 sigma, see ``fit_dive_line``); rows from before the fix carry the
  old, smaller value.
* ``line_confidence`` on a dive whose inliers sit on one or two distinct
  locations, which fishsense-lite scored ``inf`` (confident) and this scores
  0 (see ``_line_confidence``). None of the golden cases is one.

Two surface differences from the research-repo module:

* Upstream operates on polars DataFrames as part of its batch pipeline;
  the public API here takes a numpy array of (x, y) positives plus a
  thin convenience that returns outlier indices, so the caller (an
  activity that already has a ``LaserLabel`` list from the SDK) doesn't
  pull polars in.
* The polars helpers ``fit_lines_per_dive`` and ``flag_label_outliers``
  are dropped — the per-dive RANSAC / outlier kernel is the only piece
  the validation activity needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Minimum positive labels required to attempt a line fit. A degenerate fit
# (2 points) always succeeds; we want enough redundancy for RANSAC to mean
# something.
MIN_POINTS_FOR_LINE = 5

# Minimum distinct inlier locations for a line to count as confident. Labels
# stacked on one or two pixels pass MIN_POINTS_FOR_LINE but determine the line
# no better than 1-2 points do. Not in fishsense-lite, which scored them inf;
# no golden case has fewer than 3, so parity is unaffected.
MIN_DISTINCT_LOCATIONS = 3

# RANSAC inlier tolerance in pixels (perpendicular distance). 4K frames + a
# 3 px laser blob → ~4 px is a generous-but-not-loose tolerance for label
# noise. Tunable; logged so we can revisit.
RANSAC_INLIER_TOL_PX = 4.0

# Max RANSAC iterations.
RANSAC_MAX_ITERS = 200

# Confidence threshold below which we say the line is ambiguous and the prior
# should not be applied. Eigenvalue ratio (along-line spread / perp spread).
LINE_CONFIDENCE_THRESHOLD = 5.0

# MAD → consistent estimator of σ for normally distributed residuals.
MAD_TO_SIGMA = 1.4826

# Floor on the MAD-derived σ used by ``flag_outliers``. On very small dives
# whose RANSAC inliers happen to be sub-pixel-tight, MAD collapses to ~0 and
# every label gets flagged. Labels can't reasonably be more precise than ~1 px
# at native 4K resolution, so any threshold below this is non-physical.
LABEL_NOISE_MAD_FLOOR_PX = 1.0

# Default σ multiple used by ``flag_outliers``. 3σ leaves a ~0.3% false-flag
# rate under Gaussian noise, which on a typical-size dive (~100 positives) is
# well under one expected false flag.
DEFAULT_OUTLIER_SIGMA = 3.0

#: Calibration frames (a frame carrying a completed `DiveSlateLabel`, or a
#: checkerboard frame) are judged against this absolute bound instead of the
#: 3 sigma one, because the dive line is fish-dominated and a genuine slate
#: dot legitimately sits a few px off it.
#:
#: Measured on prod 2026-09-13 against each dive's own fitted line: dive
#: 347's thirteen genuine slate dots sit 0.34-8.48 px off a line its 319 fish
#: dots define to 1.25 px median, and 349's twelve sit 2.87-5.98 px off a
#: 0.94 px line. The burst is a few seconds inside a dive whose fish frames
#: are minutes away, so a small in-plane rotation of the laser within its
#: mount moves the whole burst coherently -- and a dozen dots against
#: hundreds cannot pull the fit back toward themselves. The four real
#: mislabels on 347 are 45.57, 50.59, 83.99 and 130.18 px: a separate
#: population, a specular reflection or another object, and they must still
#: go.
#:
#: The bound sits between those populations rather than near either, because
#: the costs are wildly asymmetric. Superseding a genuine slate dot removes
#: the dive's only route to a calibration and cannot be undone by
#: relabelling (`get_laser_label_by_label_studio_id` filters superseded rows),
#: which is how prod dive 347 came to be calibrated from ONE frame and 349
#: from two dots 26 cm apart in range. A 10 px error in one calibration dot,
#: by contrast, is diluted by the others and caught downstream by the four
#: gates in `calibration_consistency`.
COARSE_CALIBRATION_TOLERANCE_PX = 20.0


@dataclass
class LineFit:  # pylint: disable=too-many-instance-attributes
    """A normalized line ``a*x + b*y + c = 0`` plus quality metrics.

    ``label_noise_mad`` is ``1.4826 * MAD`` of the signed perpendicular
    residuals of *every* row passed to :func:`fit_dive_line`, in px — the
    noise scale :func:`flag_outliers` thresholds against. It is a property of
    the population it was estimated from, not of the camera or the dive:
    drop rows and it shrinks. Removing the rows a previous
    :func:`flag_outliers` call flagged cuts the residual tail, so a fit of the
    survivors reports a smaller value and flags rows the first call passed.
    See :func:`flag_outliers` for the calling contract that follows.
    """

    a: float
    b: float
    c: float
    n_points: int
    inlier_count: int
    inlier_fraction: float
    residual_std: float  # perp-distance std among inliers, in px (RANSAC tightness)
    label_noise_mad: (
        float  # 1.4826 * MAD over ALL positive labels' SIGNED perp residual, in px
    )
    line_confidence: float  # along-line spread / perp spread (covariance eigenratio)

    @property
    def is_confident(self) -> bool:
        """Whether the line direction is determined well enough to act on."""
        return self.line_confidence >= LINE_CONFIDENCE_THRESHOLD

    def perpendicular_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perpendicular distance from each (x,y) to this line, in pixels."""
        return np.abs(self.a * x + self.b * y + self.c)


def _fit_line_total_least_squares(xy: np.ndarray) -> tuple[float, float, float]:
    """Fit a 2D line via SVD on centered points (total least squares).

    Returns normalized ``(a, b, c)`` for ``a*x + b*y + c = 0``.
    """
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    # SVD: smallest singular vector is the normal to the best-fit line
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    a, b = float(normal[0]), float(normal[1])
    norm = float(np.hypot(a, b))
    if norm == 0.0:
        return 1.0, 0.0, 0.0
    a, b = a / norm, b / norm
    c = float(-(a * centroid[0] + b * centroid[1]))
    return a, b, c


def _ransac_line(  # pylint: disable=too-many-locals
    xy: np.ndarray,
    tol_px: float,
    max_iters: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    """RANSAC line fit. Returns ``(a, b, c, inlier_mask)``."""
    n = xy.shape[0]
    best_inliers: np.ndarray | None = None
    best_count = -1

    for _ in range(max_iters):
        idx = rng.choice(n, size=2, replace=False)
        p0, p1 = xy[idx[0]], xy[idx[1]]
        dx, dy = p1 - p0
        norm = float(np.hypot(dx, dy))
        if norm == 0.0:
            continue
        a, b = -dy / norm, dx / norm
        c = -(a * p0[0] + b * p0[1])
        dist = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
        inliers = dist < tol_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 2:
        a, b, c = _fit_line_total_least_squares(xy)
        return a, b, c, np.ones(n, dtype=bool)

    a, b, c = _fit_line_total_least_squares(xy[best_inliers])
    dist = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
    inliers = dist < tol_px
    return a, b, c, inliers


def _line_confidence(xy: np.ndarray, a: float, b: float) -> float:
    """Ratio of along-line variance to perpendicular variance.

    A high ratio means the points are spread out along the line (well-determined
    direction). A low ratio means they cluster, leaving the line direction
    ambiguous.

    Points on fewer than ``MIN_DISTINCT_LOCATIONS`` distinct locations score
    0. Their perpendicular variance is exactly 0, which the ratio would read
    as a perfect line (``inf``); but one location fixes no direction and two
    fix it with no redundancy.
    """
    if np.unique(xy, axis=0).shape[0] < MIN_DISTINCT_LOCATIONS:
        return 0.0
    centered = xy - xy.mean(axis=0)
    along = np.array([-b, a])
    perp = np.array([a, b])
    var_along = float(np.var(centered @ along))
    var_perp = float(np.var(centered @ perp))
    if var_perp <= 1e-9:
        return float("inf")
    return var_along / var_perp


def fit_dive_line(  # pylint: disable=too-many-locals
    xy: np.ndarray,
    *,
    tol_px: float = RANSAC_INLIER_TOL_PX,
    max_iters: int = RANSAC_MAX_ITERS,
    rng: np.random.Generator | None = None,
) -> LineFit | None:
    """Fit a line to one dive's positive labels.

    Returns ``None`` when fewer than ``MIN_POINTS_FOR_LINE`` positives are
    available — RANSAC on 2-4 points is degenerate.

    Deterministic for identical input, but not order-invariant: RANSAC draws
    its point pairs by row index from ``rng``, so the same labels in another
    order, or with one label added, can settle on a different line. Pass a
    dive's labels in a stable order (e.g. by image id).
    """
    if xy.shape[0] < MIN_POINTS_FOR_LINE:
        return None
    rng = rng or np.random.default_rng(0)
    a, b, c, inliers = _ransac_line(xy, tol_px=tol_px, max_iters=max_iters, rng=rng)
    inlier_xy = xy[inliers]
    n = xy.shape[0]
    inlier_count = int(inliers.sum())

    dist_inliers = np.abs(a * inlier_xy[:, 0] + b * inlier_xy[:, 1] + c)
    residual_std = float(np.std(dist_inliers))
    confidence = _line_confidence(inlier_xy, a, b)

    # MAD on ALL positive labels — the population the outlier flag will be
    # applied against. ``residual_std`` is bounded by the RANSAC tolerance and
    # so under-states true label-noise scale; MAD is robust to the gross
    # outliers in the tail.
    #
    # Signed residuals, not distances: MAD_TO_SIGMA assumes a normal sample,
    # and |residual| is a folded normal whose scaled MAD is ~0.59 sigma. The
    # fishsense-lite copy (and so the migrated v2 ``dive_laser_lines`` rows)
    # used distances, which put the 3 sigma cut at ~1.78 sigma and flagged
    # ~7.6% of clean labels. Sign-invariant, so LAPACK's choice of normal
    # direction cannot move it.
    resid_all = a * xy[:, 0] + b * xy[:, 1] + c
    label_noise_mad = float(
        MAD_TO_SIGMA * np.median(np.abs(resid_all - np.median(resid_all)))
    )

    return LineFit(
        a=a,
        b=b,
        c=c,
        n_points=n,
        inlier_count=inlier_count,
        inlier_fraction=inlier_count / n,
        residual_std=residual_std,
        label_noise_mad=label_noise_mad,
        line_confidence=confidence,
    )


def flag_outliers(  # pylint: disable=too-many-arguments
    xy: np.ndarray,
    fit: LineFit,
    *,
    sigma: float = DEFAULT_OUTLIER_SIGMA,
    mad_floor_px: float = LABEL_NOISE_MAD_FLOOR_PX,
    calibration_mask: np.ndarray | None = None,
    coarse_tolerance_px: float = COARSE_CALIBRATION_TOLERANCE_PX,
) -> np.ndarray:
    """Boolean mask marking labels whose perpendicular distance to ``fit``
    exceeds ``sigma * max(label_noise_mad, mad_floor_px)``.

    Returns an all-False mask when ``fit`` is not confident — callers
    should not act on outlier flags from a low-confidence line.

    The MAD floor handles small-N dives where MAD collapses to sub-pixel
    values and would otherwise flag every label.

    ``calibration_mask`` marks the rows that are calibration observations —
    frames carrying a completed `DiveSlateLabel`. Those are judged against
    ``coarse_tolerance_px`` in absolute pixels instead, never the tighter of
    the two, because the dive line is fish-dominated and a genuine slate dot
    sits a few px off it; see `COARSE_CALIBRATION_TOLERANCE_PX`. Omit it and
    every row is judged at 3 sigma, which is what every caller got before
    this existed — the loose rule has to be asked for.

    **Calling contract: one pass over the whole population.** This is a
    single-pass judgement of the rows passed in, against a noise scale
    (``fit.label_noise_mad``) estimated from those same rows. It is *not*
    idempotent over its own survivors: fit and flag, drop what was flagged,
    refit and flag again, and more rows can be flagged — and again, pass after
    pass. Dropping the flagged tail shrinks the next noise estimate, so the
    3-sigma cut moves inward. Callers that act on the flags (e.g. superseding
    labels) must therefore fit and flag the **full** population every time —
    rows flagged on earlier runs included, in the same order (see
    :func:`fit_dive_line`) — and never re-flag the survivors of an earlier
    pass.

    Idempotence is deliberately not provided. An idempotent rule must flag
    its own fixed point in one call, and on dives whose line drifts or steps
    during the dive, that fixed point is the eroded set: labels that one line
    cannot represent, not mislabels.
    """
    if not fit.is_confident:
        return np.zeros(xy.shape[0], dtype=bool)
    effective_mad = max(fit.label_noise_mad, mad_floor_px)
    threshold = np.full(xy.shape[0], sigma * effective_mad, dtype=float)
    if calibration_mask is not None:
        mask = np.asarray(calibration_mask, dtype=bool)
        # `maximum`, not assignment: on a slate-only calibration dive the
        # line is fitted from the slate dots themselves, so 3 sigma there is
        # the *looser* of the two and tightening it would re-break the dives
        # this exists for.
        threshold[mask] = np.maximum(threshold[mask], coarse_tolerance_px)
    perp = fit.perpendicular_distance(xy[:, 0], xy[:, 1])
    return perp > threshold
