"""Golden numbers for the per-dive line fit.

v2 carries migrated ``dive_laser_lines`` rows (``a``/``b``/``c``,
``line_confidence``, ``residual_std``, ``label_noise_mad`` …) that were
computed by fishsense-lite's copy of this module. Any change that shifts
these numbers breaks parity with those rows, so the test fails on it — even a
change that looks like an improvement.

Every expected value below was produced by fishsense-lite@a8b2c3bc's
``laser_label_validation/line_fit.py`` under that repo's locked numpy
(2.5.1), fed the inputs built by ``_build_cases`` — not by this package.

Two deliberate tolerances, both measured by running that same source under
numpy 2.5.1 and 2.4.4:

* Floats are compared at ``rel=1e-12``. The two builds differ by up to
  1.4e-14 relative (the ``cluster`` case's ``residual_std``); an algorithmic
  change moves them by orders of magnitude more.
* ``(a, b, c)`` is compared up to one shared sign. The sign is LAPACK's
  choice of singular vector, not this code's: on the exactly colinear
  ``clean`` case the two numpy builds return opposite signs. Downstream
  consumers already canonicalise it (fishsense-api's ``_canonical_line``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fishsense_core.line_fit import fit_dive_line, flag_outliers

REL = 1e-12


def _line(n, *, slope, intercept, x_max, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, x_max, n)
    ys = slope * xs + intercept
    if noise:
        ys = ys + rng.normal(0.0, noise, size=n)
    return np.column_stack([xs, ys])


# Rows moved off the line in the "outliers" case, and by how many px in y.
OUTLIER_OFFSETS_PX = {7: 9.0, 19: -15.0, 33: 26.0, 48: -45.0, 62: 130.0, 81: -300.0}
# The "outliers" case's calibration rows: 19 (15 px, inside the coarse 20 px
# bound), 33 (26 px, outside it) and 90 (a clean row, which must stay clean).
CALIBRATION_ROWS = (19, 33, 90)


def _build_cases():
    clean = _line(30, slope=0.3, intercept=200.0, x_max=1000.0)
    noisy = _line(100, slope=-0.2, intercept=1500.0, x_max=3000.0, noise=1.5, seed=11)
    outliers = _line(100, slope=0.45, intercept=300.0, x_max=3500.0, noise=1.0, seed=12)
    for row, dy in OUTLIER_OFFSETS_PX.items():
        outliers[row, 1] += dy
    minimum = _line(5, slope=1.2, intercept=-40.0, x_max=800.0, noise=0.8, seed=13)
    cluster = np.random.default_rng(14).normal([500.0, 500.0], [2.0, 2.0], size=(20, 2))
    degenerate = _line(4, slope=0.5, intercept=100.0, x_max=1000.0)
    return {
        "clean": clean,
        "noisy": noisy,
        "outliers": outliers,
        "minimum": minimum,
        "cluster": cluster,
        "degenerate": degenerate,
    }


CASES = _build_cases()

# Sum of each input array. Guards the inputs, not the fit: if numpy's
# `default_rng` stream ever changes, this fails first and says so, instead of
# the fit assertions failing for a reason that has nothing to do with the fit.
INPUT_SUMS = {
    "clean": 25500.000000000004,
    "noisy": 270003.69982354814,
    "outliers": 283559.4366357013,
    "minimum": 4200.875500701528,
    "cluster": 20002.41160845857,
    "degenerate": 3400.0,
}

GOLDEN = {
    "clean": {
        "abc": (-0.2873478855663455, 0.9578262852211513, -191.56525704423024),
        "n_points": 30,
        "inlier_count": 30,
        "inlier_fraction": 1.0,
        # Floating-point dust on colinear points: pinned to "effectively 0".
        "residual_std": None,
        "label_noise_mad": None,
        "line_confidence": math.inf,
        "is_confident": True,
        "flagged": [],
    },
    "noisy": {
        "abc": (-0.19600102701354136, -0.9806036902896282, 1470.7622494741129),
        "n_points": 100,
        "inlier_count": 100,
        "inlier_fraction": 1.0,
        "residual_std": 0.7616956256496666,
        "label_noise_mad": 0.7730499434505185,
        "line_confidence": 440700.0739292007,
        "is_confident": True,
        "flagged": [],
    },
    "outliers": {
        "abc": (0.41031692513177315, -0.911942992160369, 273.73789019515607),
        "n_points": 100,
        "inlier_count": 94,
        "inlier_fraction": 0.94,
        "residual_std": 0.468066057311574,
        "label_noise_mad": 0.4836682221847358,
        "line_confidence": 1839806.0873257914,
        "is_confident": True,
        "flagged": [7, 19, 33, 48, 62, 81],
    },
    "minimum": {
        "abc": (-0.768447826357578, 0.6399124457035616, 25.70560567443806),
        "n_points": 5,
        "inlier_count": 5,
        "inlier_fraction": 1.0,
        "residual_std": 0.5259568804778961,
        "label_noise_mad": 0.2838218445446909,
        "line_confidence": 252183.52947625107,
        "is_confident": True,
        "flagged": [],
    },
    "cluster": {
        "abc": (0.995261756108503, -0.09723187146105297, -448.80467381968214),
        "n_points": 20,
        "inlier_count": 19,
        "inlier_fraction": 0.95,
        "residual_std": 1.0880327589093377,
        "label_noise_mad": 1.1620886586243588,
        "line_confidence": 1.7025817179512779,
        "is_confident": False,
        "flagged": [],
    },
}

# Rows the fit leaves outside the RANSAC tolerance; `inlier_count` is the
# complement, but pinning the rows proves *which* points were rejected.
OUTSIDE_RANSAC_TOL = {
    "clean": [],
    "noisy": [],
    "outliers": [7, 19, 33, 48, 62, 81],
    "minimum": [],
    "cluster": [5],
}

# flag_outliers on the "outliers" case with CALIBRATION_ROWS marked: row 19
# (15 px) is let through by the coarse bound, row 33 (26 px) is not.
FLAGGED_WITH_CALIBRATION = [7, 33, 48, 62, 81]


@pytest.mark.parametrize("name", sorted(CASES))
def test_inputs_are_the_ones_the_golden_values_were_computed_on(name):
    assert float(CASES[name].sum()) == INPUT_SUMS[name]


def test_degenerate_case_returns_none():
    assert fit_dive_line(CASES["degenerate"]) is None


def _assert_abc(fit, expected):
    got = np.array([fit.a, fit.b, fit.c])
    want = np.array(expected)
    sign = 1.0 if np.dot(got[:2], want[:2]) >= 0.0 else -1.0
    assert (sign * got).tolist() == pytest.approx(want.tolist(), rel=REL)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_line_fit_fields_match_fishsense_lite(name):
    xy, want = CASES[name], GOLDEN[name]
    fit = fit_dive_line(xy)
    assert fit is not None

    _assert_abc(fit, want["abc"])
    assert fit.n_points == want["n_points"]
    assert fit.inlier_count == want["inlier_count"]
    assert fit.inlier_fraction == want["inlier_fraction"]
    for field in ("residual_std", "label_noise_mad"):
        if want[field] is None:
            assert getattr(fit, field) < 1e-9
        else:
            assert getattr(fit, field) == pytest.approx(want[field], rel=REL)
    assert fit.line_confidence == pytest.approx(want["line_confidence"], rel=REL)
    assert fit.is_confident is want["is_confident"]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_inlier_mask_matches_fishsense_lite(name):
    xy = CASES[name]
    fit = fit_dive_line(xy)
    tol = 4.0  # RANSAC_INLIER_TOL_PX, spelled out so a retune fails here too
    outside = np.flatnonzero(fit.perpendicular_distance(xy[:, 0], xy[:, 1]) >= tol)
    assert outside.tolist() == OUTSIDE_RANSAC_TOL[name]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_flag_outliers_matches_fishsense_lite(name):
    xy = CASES[name]
    flagged = flag_outliers(xy, fit_dive_line(xy))
    assert flagged.dtype == bool
    assert np.flatnonzero(flagged).tolist() == GOLDEN[name]["flagged"]


def test_flag_outliers_with_calibration_mask_matches_fishsense_lite():
    xy = CASES["outliers"]
    mask = np.zeros(len(xy), dtype=bool)
    mask[list(CALIBRATION_ROWS)] = True
    flagged = flag_outliers(xy, fit_dive_line(xy), calibration_mask=mask)
    assert np.flatnonzero(flagged).tolist() == FLAGGED_WITH_CALIBRATION


def test_default_rng_seeding_matches_fishsense_lite():
    """``rng=None`` must mean ``default_rng(0)`` — the seeding every
    migrated row was computed with.

    At the default 200 iterations every seed tried converges on the same
    inlier set, so the cases above cannot see the seeding. One iteration
    makes the fit the single pair the generator draws first: ``default_rng(1)``
    gives confidence 1326328.67 here, not 1720875.06.
    """
    fit = fit_dive_line(CASES["outliers"], max_iters=1)
    _assert_abc(fit, (-0.4104936654544967, 0.9118634495480843, -273.1568013358901))
    assert fit.inlier_count == 94
    assert fit.residual_std == pytest.approx(0.5141654971276408, rel=REL)
    assert fit.label_noise_mad == pytest.approx(0.5894538224222293, rel=REL)
    assert fit.line_confidence == pytest.approx(1720875.0554145924, rel=REL)


def test_laser_module_re_exports_the_line_fit():
    """The public surface lives next to ``calibrate_laser``, the 3-D fit
    that consumes the same dots."""
    # pylint: disable=import-outside-toplevel
    from fishsense_core import laser, line_fit

    for name in (
        "LineFit",
        "fit_dive_line",
        "flag_outliers",
        "MIN_POINTS_FOR_LINE",
        "RANSAC_INLIER_TOL_PX",
        "RANSAC_MAX_ITERS",
        "LINE_CONFIDENCE_THRESHOLD",
        "MAD_TO_SIGMA",
        "LABEL_NOISE_MAD_FLOOR_PX",
        "DEFAULT_OUTLIER_SIGMA",
        "COARSE_CALIBRATION_TOLERANCE_PX",
    ):
        assert name in laser.__all__
        assert getattr(laser, name) is getattr(line_fit, name)
