"""Unit tests for the per-dive RANSAC line-fit kernel.

Ported unchanged (imports aside) from fishsense-lite@a8b2c3bc, where they
pinned the vendored copy this module replaces. The exact numbers are pinned
separately in ``test_line_fit_golden.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fishsense_core.line_fit import (
    LABEL_NOISE_MAD_FLOOR_PX,
    MIN_POINTS_FOR_LINE,
    fit_dive_line,
    flag_outliers,
)


def _line_points(
    n: int,
    *,
    slope: float = 0.5,
    intercept: float = 100.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1000.0, n)
    ys = slope * xs + intercept
    if noise:
        ys = ys + rng.normal(0.0, noise, size=n)
    return np.column_stack([xs, ys])


def test_fit_returns_none_below_minimum_positives():
    xy = _line_points(MIN_POINTS_FOR_LINE - 1)
    assert fit_dive_line(xy) is None


def test_fit_recovers_clean_line_with_high_confidence():
    xy = _line_points(50, slope=0.3, intercept=200.0, noise=0.0)
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    # All points are colinear → all should be inliers and residuals tiny.
    assert fit.inlier_count == 50
    assert fit.inlier_fraction == pytest.approx(1.0)
    assert fit.residual_std < 1e-6
    # Spread along the line is large; perp variance is ~0 → confidence is huge.
    assert fit.is_confident


def test_fit_marks_low_confidence_when_points_cluster():
    rng = np.random.default_rng(1)
    # Tight cluster: along-line spread comparable to perpendicular spread.
    xy = rng.normal(loc=[500.0, 500.0], scale=[2.0, 2.0], size=(20, 2))
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert not fit.is_confident


def test_flag_outliers_catches_off_line_label():
    xy = _line_points(40, noise=0.5, seed=2)
    # Inject one obvious outlier: ~50px off the line.
    xy[10] = xy[10] + np.array([0.0, 50.0])
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert fit.is_confident
    mask = flag_outliers(xy, fit)
    assert mask[10]
    # No clean point should be flagged. Allow up to one false positive
    # to absorb very-tight Gaussian tails.
    assert mask.sum() <= 2


def test_flag_outliers_returns_all_false_for_low_confidence_fit():
    rng = np.random.default_rng(3)
    xy = rng.normal(loc=[500.0, 500.0], scale=[2.0, 2.0], size=(20, 2))
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert not fit.is_confident
    assert not flag_outliers(xy, fit).any()


def test_mad_floor_prevents_mass_flagging_when_inliers_are_subpixel_tight():
    """On a tiny noise-free dive every label was getting flagged before
    the floor was added — perpendicular MAD collapses to ~0 and the
    `3σ` threshold becomes sub-pixel."""
    xy = _line_points(MIN_POINTS_FOR_LINE + 2, noise=0.0)
    # Bump one point by exactly the floor — should be on the borderline,
    # not flagged.
    xy[2] = xy[2] + np.array([0.0, LABEL_NOISE_MAD_FLOOR_PX * 2.0])
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    if fit.is_confident:
        assert flag_outliers(xy, fit).sum() <= 1


def test_label_noise_mad_estimates_sigma_of_gaussian_label_noise():
    """Regression: the MAD was taken over *absolute* perpendicular distances.
    |d| of N(0, σ) is a folded normal whose scaled MAD is ~0.59σ, so the
    "3σ" cut sat at ~1.78σ and flagged ~7.6% of clean labels instead of the
    ~0.3% ``DEFAULT_OUTLIER_SIGMA`` promises. σ = 2.5 px keeps the 1 px floor
    out of play; at prod noise (~1.85 px) the old cut superseded ~7 good
    labels per 100-label dive."""
    sigma = 2.5
    xy = _line_points(2000, noise=sigma, seed=4)
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    # `_line_points` adds the noise in y; across a slope-0.5 line that is
    # sigma / hypot(1, 0.5) = 2.24 px perpendicular.
    assert fit.label_noise_mad == pytest.approx(sigma / np.hypot(1.0, 0.5), rel=0.1)
    # Expected ~0.27% of 2000 ≈ 5 under Gaussian noise; the bug flagged ~150.
    assert flag_outliers(xy, fit).sum() <= 20


# ---------------------------------------------------------------------------
# Not from fishsense-lite: positives stacked on one or two locations.
#
# The perpendicular variance is exactly 0 there, which the confidence ratio
# used to read as a perfect line (inf -> confident). One location fixes no
# direction, and two fix one with no redundancy (the reason fits below
# MIN_POINTS_FOR_LINE are refused) — so neither may be acted on.
# ---------------------------------------------------------------------------


def test_positives_on_one_location_are_not_confident():
    xy = np.tile([[812.0, 431.0]], (MIN_POINTS_FOR_LINE + 3, 1))
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert not fit.is_confident
    assert not flag_outliers(xy, fit).any()


def test_positives_on_two_locations_are_not_confident():
    xy = np.array([[100.0, 100.0]] * 4 + [[900.0, 500.0]] * 4)
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert not fit.is_confident


def test_line_through_two_stacked_locations_does_not_flag_a_third():
    """What the old behaviour cost: a line pinned by two spots was trusted,
    so a label anywhere else got flagged against it."""
    xy = np.array([[100.0, 100.0]] * 4 + [[900.0, 500.0]] * 4 + [[500.0, 900.0]])
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert fit.inlier_count == 8
    assert not fit.is_confident
    assert not flag_outliers(xy, fit).any()


def test_three_stacked_collinear_locations_stay_confident():
    """The fix keys on distinct locations, not on zero perpendicular
    variance: three agreeing spots are still a determined line."""
    xy = np.array([[100.0, 100.0]] * 3 + [[500.0, 300.0]] * 3 + [[900.0, 500.0]] * 3)
    fit = fit_dive_line(xy, rng=np.random.default_rng(0))
    assert fit is not None
    assert fit.is_confident


# ---------------------------------------------------------------------------
# The calling contract (see ``flag_outliers``): one pass over the full
# population, in a stable order. These pin both halves, so a change that
# makes flagging idempotent or order-invariant has to revisit the docstrings.
# ---------------------------------------------------------------------------


def _stepped_dive() -> np.ndarray:
    """200 labels on a line with 1 px noise, 8 gross mislabels 40-80 px off,
    and frames 80-129 stepped 5 px to one side — the dot's line moving for a
    stretch of the dive, as in prod dive 257."""
    rng = np.random.default_rng(0)
    n = 200
    xs = np.linspace(0.0, 1500.0, n)
    perp = rng.normal(0.0, 1.0, n)
    perp[80:130] += 5.0
    bad = rng.choice(n, 8, replace=False)
    perp[bad] += rng.choice([-1, 1], 8) * rng.uniform(40.0, 80.0, 8)
    normal = np.array([-0.3, 1.0]) / np.hypot(0.3, 1.0)
    return np.column_stack([xs, 0.3 * xs + 400.0]) + perp[:, None] * normal


def _fit_and_flag(xy: np.ndarray) -> np.ndarray:
    fit = fit_dive_line(xy)
    assert fit is not None
    return flag_outliers(xy, fit)


def test_refitting_the_full_population_reproduces_its_flags():
    """What the contract promises: a caller that re-fits every row each run,
    flagged ones included and in the same order, gets the same flags."""
    xy = _stepped_dive()
    first = _fit_and_flag(xy)
    assert first.any()
    np.testing.assert_array_equal(_fit_and_flag(xy.copy()), first)


def test_reflagging_the_survivors_flags_more():
    """What the contract forbids, and why: each pass over the survivors
    estimates a smaller noise scale and flags more, eroding the step."""
    xy = _stepped_dive()
    survivors = xy[~_fit_and_flag(xy)]
    assert fit_dive_line(survivors).label_noise_mad < fit_dive_line(xy).label_noise_mad
    assert _fit_and_flag(survivors).any()


def test_row_order_can_change_the_flags():
    """Why the contract says "in the same order": RANSAC samples by index."""
    xy = _stepped_dive()
    flags = _fit_and_flag(xy)
    changed = False
    for seed in range(20):
        perm = np.random.default_rng(seed).permutation(len(xy))
        if not np.array_equal(_fit_and_flag(xy[perm])[np.argsort(perm)], flags):
            changed = True
            break
    assert changed
