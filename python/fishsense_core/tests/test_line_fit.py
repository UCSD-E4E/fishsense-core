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
