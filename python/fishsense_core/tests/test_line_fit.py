"""Unit tests for the per-dive RANSAC line-fit kernel.

Ported unchanged (imports aside) from fishsense-lite@a8b2c3bc, where they
pinned the vendored copy this module replaces. The exact numbers are pinned
separately in ``test_line_fit_golden.py``.
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from fishsense_core.line_fit import (
    LABEL_NOISE_MAD_FLOOR_PX,
    MIN_POINTS_FOR_LINE,
    RANSAC_INLIER_TOL_PX,
    RANSAC_MAX_ITERS,
    LineConsensus,
    LineFit,
    consensus_line,
    fit_dive_line,
    fit_line_pca,
    flag_outliers,
    ransac_line,
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


# -- the line primitives, against the fitters they replace ------------------------
#
# fishsense-services v0.48.0 has two private line fitters. Their source is
# copied below verbatim (fishsense-services-processor/src/
# fishsense_services_processor/), as the oracle the primitives must reproduce.
# The pinned numbers further down were produced by the services' own modules.


def _svc_gated_ransac_line(xy: np.ndarray, rng, tol: float, iters: int = 500):
    """shadow_calibration/gated.py ``_ransac_line``, verbatim."""
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(xy), 2, replace=False)
        d = xy[j] - xy[i]
        if np.hypot(*d) < 1:
            continue
        nn = np.array([-d[1], d[0]]) / np.hypot(*d)
        inl = np.abs(xy @ nn - nn @ xy[i]) <= tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return None
    q = xy[best]
    m = q.mean(0)
    u = np.linalg.svd(q - m)[2][0]
    u = u if u[1] < 0 else -u
    return u, float(m @ np.array([-u[1], u[0]])), best


def _svc_line_through(xy: np.ndarray) -> np.ndarray:
    """automatic_calibration/v3.py ``_line_through``, verbatim."""
    m = xy.mean(axis=0)
    u = np.linalg.svd(xy - m)[2][0]
    n = np.array([-u[1], u[0]])
    return np.array([n[0], n[1], -n @ m])


def _svc_distances(xy: np.ndarray, line) -> np.ndarray:
    """automatic_calibration/v3.py ``_distances``, verbatim."""
    a, b, c = line
    return np.abs(xy @ np.array([a, b]) + c) / np.hypot(a, b)


def _svc_slate_consensus(slate_xy, tol: float = 10.0):
    """automatic_calibration/v3.py ``slate_consensus``, verbatim."""
    xy = np.asarray(slate_xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return list(range(len(xy))), None
    best = None
    for i, j in combinations(range(len(xy)), 2):
        d = xy[j] - xy[i]
        if np.hypot(*d) < 1.0:
            continue
        line = _svc_line_through(xy[[i, j]])
        dist = _svc_distances(xy, line)
        inl = dist <= tol
        key = (int(inl.sum()), -float(dist[inl].sum()))
        if best is None or key > best[0]:  # pylint: disable=unsubscriptable-object
            best = (key, inl)
    if best is None:
        return list(range(len(xy))), None
    inl = best[1]
    line = _svc_line_through(xy[inl])
    inl = _svc_distances(xy, line) <= tol
    return [int(i) for i in np.flatnonzero(inl)], line


def _gated_via_core(xy: np.ndarray):
    """gated.py's ``_ransac_line`` as a `ransac_line` call."""
    fit = ransac_line(
        xy,
        tol_px=10.0,
        max_iters=500,
        rng=np.random.default_rng(0),
        inclusive=True,
        min_separation_px=1.0,
        refit=fit_line_pca,
    )
    if fit is None:
        return None
    u, off = np.array([fit.b, -fit.a]), -fit.c
    if u[1] >= 0:
        u, off = -u, -off
    return u, off, fit.consensus


def _v3_via_core(xy: np.ndarray):
    """v3.py's ``slate_consensus`` as a `consensus_line` call."""
    fit = consensus_line(xy, tol_px=10.0)
    if fit is None:
        return list(range(len(xy))), None
    return [int(i) for i in np.flatnonzero(fit.inliers)], np.array(
        [fit.a, fit.b, fit.c]
    )


def _population(seed: int) -> np.ndarray:
    """A line of dots with gross outliers; every fourth kind snapped to
    integer pixels, to a 10 px grid, or with dots stacked on one pixel."""
    rng = np.random.default_rng(10_000 + seed)
    n = int(rng.integers(2, 60))
    x = rng.uniform(0, 4000, n)
    y = rng.uniform(-0.5, 0.5) * x + rng.uniform(0, 3000)
    y = y + rng.normal(0, rng.uniform(0.2, 8), n)
    k = int(rng.integers(0, n // 2 + 1))
    y[:k] = rng.uniform(0, 3000, k)
    xy = np.column_stack([x, y])
    kind = seed % 4
    if kind == 1:
        xy = np.round(xy)
    elif kind == 2:
        xy = np.round(xy / 10) * 10
    elif kind == 3:
        xy[: int(rng.integers(0, n))] = xy[0]
    return xy


@pytest.mark.parametrize("seed", range(40))
def test_ransac_line_reproduces_the_services_gated_fitter(seed):
    xy = _population(seed)
    want = _svc_gated_ransac_line(xy, np.random.default_rng(0), 10.0)
    got = _gated_via_core(xy)
    if want is None:
        assert got is None
        return
    np.testing.assert_array_equal(got[2], want[2])
    np.testing.assert_array_equal(got[0], want[0])
    assert got[1] == want[1]


@pytest.mark.parametrize("seed", range(40))
def test_consensus_line_reproduces_the_services_slate_consensus(seed):
    xy = _population(seed)
    want_kept, want_line = _svc_slate_consensus(xy)
    got_kept, got_line = _v3_via_core(xy)
    assert got_kept == want_kept
    if want_line is None:
        assert got_line is None
    else:
        np.testing.assert_array_equal(got_line, want_line)


#: fishsense-services' dive 279 rows (tests/fixtures/shadow_pool.json,
#: ``dive279``): a line of slate dots, one far off it (index 19).
_DIVE_279 = np.array(
    [
        [1970.7699083346777, 1340.0577294099733],
        [1976.4656982421875, 1340.16015625],
        [1977.484619140625, 1342.95703125],
        [1977.1553955078125, 1341.6544189453125],
        [1978.0667724609373, 1341.440185546875],
        [1976.8565673828125, 1341.03369140625],
        [1994.0660400390625, 1394.927978515625],
        [1994.16943359375, 1395.09375],
        [1994.0909423828125, 1393.39501953125],
        [1818.0263671875, 826.3372192382812],
        [1994.0108642578125, 1391.87451171875],
        [1994.3060302734373, 1393.1463623046875],
        [1850.322021484375, 902.7750854492188],
        [1958.6639404296875, 1276.1683349609375],
        [1817.89794921875, 809.791748046875],
        [1951.7452392578125, 1254.1253662109375],
        [1951.001708984375, 1254.4735107421875],
        [1955.3560791015625, 1264.2178955078125],
        [1997.2254638671875, 1407.9542236328125],
        [2086.385009765625, 1143.452880859375],
        [1951.4808853350955, 1273.4501455931647],
        [1951.89826623254, 1274.1973310719443],
        [1957.3295221686303, 1291.6907521640037],
        [1959.074373896789, 1295.0944246440747],
        [1958.160106434828, 1297.28098541928],
        [1982.8508760124248, 1379.7874238218888],
        [1983.89328632008, 1382.879969475159],
    ]
)


def _assert_line_up_to_sign(got, want):
    """LAPACK's choice of singular-vector sign is the build's, not ours (see
    test_line_fit_golden.py), so ``(a, b, c)`` matches up to one sign."""
    got, want = np.asarray(got), np.asarray(want)
    sign = 1.0 if got[:2] @ want[:2] > 0 else -1.0
    assert sign * got == pytest.approx(want, rel=1e-12)


def test_gated_fitter_on_dive_279_matches_the_services_pin():
    u, off, consensus = _gated_via_core(_DIVE_279)
    # u is canonicalised (u_y < 0), so no sign ambiguity here.
    assert u == pytest.approx([-0.2863295786200243, -0.9581311874724041], rel=1e-12)
    assert off == pytest.approx(1509.021256384672, rel=1e-12)
    assert np.flatnonzero(~consensus).tolist() == [19]


def test_slate_consensus_on_dive_279_matches_the_services_pin():
    kept, line = _v3_via_core(_DIVE_279)
    assert kept == [i for i in range(len(_DIVE_279)) if i != 19]
    _assert_line_up_to_sign(
        line, [0.9581311874724041, -0.2863295786200243, -1509.021256384672]
    )


def test_slate_consensus_tie_break_on_stacked_dots_matches_the_services_pin():
    """Five dots on one pixel: every line through it and one other dot holds
    six, with a summed distance that is zero up to rounding. v3 breaks that
    tie on the rounding of its own arithmetic, which `consensus_line` uses."""
    xy = np.array(
        [[2940.31666564, 675.89784121]] * 5
        + [
            [3808.81745463, -313.40074702],
            [3907.27229742, -330.65534112],
            [1031.97185252, 436.22802765],
        ]
    )
    kept, line = _v3_via_core(xy)
    assert kept == [0, 1, 2, 3, 4, 6]
    _assert_line_up_to_sign(
        line, [0.7211494072530239, 0.6927795698623137, -2588.655836266804]
    )


def test_fit_dive_line_is_ransac_line_with_its_defaults():
    xy = _population(5)
    fit = fit_dive_line(xy)
    line = ransac_line(
        xy, tol_px=RANSAC_INLIER_TOL_PX, max_iters=RANSAC_MAX_ITERS, rng=0
    )
    assert fit is not None and line is not None
    assert (fit.a, fit.b, fit.c) == (line.a, line.b, line.c)
    assert fit.inlier_count == int(line.inliers.sum())


def test_ransac_line_takes_a_seed_or_a_generator():
    xy = _population(9)
    by_seed = ransac_line(xy, tol_px=4.0, max_iters=50, rng=7)
    by_rng = ransac_line(xy, tol_px=4.0, max_iters=50, rng=np.random.default_rng(7))
    assert by_seed is not None and by_rng is not None
    assert (by_seed.a, by_seed.b, by_seed.c) == (by_rng.a, by_rng.b, by_rng.c)
    np.testing.assert_array_equal(by_seed.inliers, by_rng.inliers)


def test_the_primitives_find_no_line_through_one_pixel():
    xy = np.array([[10.0, 10.0]] * 6)
    assert ransac_line(xy, tol_px=4.0, max_iters=20) is None
    assert consensus_line(xy, tol_px=10.0) is None
    assert consensus_line(xy[:1], tol_px=10.0) is None
    # min_separation_px: pairs closer than it define no line.
    near = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    assert ransac_line(near, tol_px=4.0, max_iters=20, min_separation_px=1.0) is None
    assert ransac_line(near, tol_px=4.0, max_iters=20) is not None


def test_inclusive_decides_a_dot_at_exactly_the_tolerance():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [5.0, 4.0]])
    strict = consensus_line(xy, tol_px=4.0, inclusive=False)
    inclusive = consensus_line(xy, tol_px=4.0)
    assert strict is not None and inclusive is not None
    # Both masks, both modes: the consensus decides the tie, and the re-score
    # against the refit must honour `inclusive` too.
    assert strict.consensus.tolist() == [True, True, True, False]
    assert strict.inliers.tolist() == [True, True, True, False]
    assert inclusive.consensus.tolist() == [True, True, True, True]
    assert inclusive.inliers.tolist() == [True, True, True, True]



# -- the review's fixes ------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1])
def test_ransac_line_with_too_few_points_is_none_not_an_error(n):
    """It used to draw a pair first and raise 'Cannot take a larger sample
    than population'."""
    assert ransac_line(np.zeros((n, 2)), tol_px=10, max_iters=500) is None
    assert consensus_line(np.zeros((n, 2)), tol_px=10) is None


def test_rng_none_is_seed_zero_in_ransac_line_as_in_fit_dive_line():
    rng = np.random.default_rng(4)
    xy = np.column_stack([np.arange(30.0), np.arange(30.0) * 0.5]) + rng.normal(0, 3, (30, 2))
    want = ransac_line(xy, tol_px=4, max_iters=200, rng=0)
    assert ransac_line(xy, tol_px=4, max_iters=200, rng=None) == want
    assert ransac_line(xy, tol_px=4, max_iters=200, rng=None) == want


def test_line_consensus_compares_and_hashes():
    """The generated __eq__ raised on the arrays, and __hash__ did too."""
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [5.0, 30.0]])
    a, b = consensus_line(xy, tol_px=4.0), consensus_line(xy.copy(), tol_px=4.0)
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    c = consensus_line(xy, tol_px=4.0, inclusive=False)
    other = LineConsensus(c.a, c.b, c.c, ~c.inliers, c.consensus)
    assert other != c
    assert a != "not a fit"


def test_the_new_primitives_are_re_exported_from_laser():
    """`fishsense_core.laser` re-exports the line fit; the new names too."""
    from fishsense_core import laser, line_fit  # pylint: disable=import-outside-toplevel

    for name in ("ransac_line", "consensus_line", "LineConsensus", "fit_line_tls", "fit_line_pca"):
        assert getattr(laser, name) is getattr(line_fit, name)
        assert name in laser.__all__


def test_one_distance_expression_for_linefit_and_ransac():
    """LineFit.perpendicular_distance and ransac_line share the expression
    their pinned results depend on."""
    from fishsense_core import line_fit  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 4000, (50, 2))
    fit = LineFit(0.6, 0.8, -1234.5, 50, 50, 1.0, 0.0, 0.0, 1.0)
    np.testing.assert_array_equal(
        fit.perpendicular_distance(xy[:, 0], xy[:, 1]),
        line_fit._elementwise_distance(xy, (0.6, 0.8, -1234.5)),  # pylint: disable=protected-access
    )
