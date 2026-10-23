"""Unit tests for the two-line (specular reflection) detector.

The first four tests are ported unchanged (imports aside) from
fishsense-services v0.48.0
``services/fishsense-services-processor/tests/test_reflection_detection.py``
(itself verbatim from fishsense-lite@77e8f8e5
services/fishsense-data-processing-workflow-worker/tests/
test_reflection_detection.py).

Prod dive 77 (2026-08-04): pool slate photos show the laser AND its specular
reflection ~45 px apart; labelers clicked the reflection on 34 of 79 frames.
The dots formed two coherent parallel lines — a population the RANSAC
validator cannot handle: the two-line split kills its confidence / trips the
MAX_OUTLIER_FRACTION refusal, so it silently skips the dive forever while
stage 13 consumes the poisoned mix. The detector names that failure mode
loudly instead.

The golden tests below pin the exact suspects. Every expected value was
produced by the fishsense-services v0.48.0 original (run against this
package's ``line_fit``, unchanged since 4.1.0, the version services pins), and
this port returns the same values bit for bit. ``angle_deg`` is compared to
an absolute 1e-6 deg (see the test).
"""

from __future__ import annotations

import numpy as np
import pytest

from fishsense_core.line_fit import fit_dive_line
from fishsense_core.laser_reflection import (
    ReflectionSuspect,
    detect_reflection_split,
)


def _line_points(n, x0, y0, dx, dy, noise=0.8, seed=3):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """``n`` points along ``(x0, y0) + t * (dx, dy)`` with isotropic noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 900, n)
    pts = np.stack([x0 + t * dx, y0 + t * dy], axis=1)
    return pts + rng.normal(0, noise, pts.shape)


def test_detects_parallel_reflection_split():
    """A parallel cluster 45 px off the laser line is flagged."""
    primary = _line_points(30, 1900.0, 800.0, 0.3, 1.0)
    artifact = primary[:20] + np.array([45.0, 0.0])  # parallel, 45px away
    xy = np.vstack([primary, artifact])
    fit = fit_dive_line(xy)

    suspect = detect_reflection_split(xy, fit)

    assert suspect is not None
    assert suspect.n_secondary >= 15
    assert 30.0 < suspect.separation_px < 60.0
    assert suspect.angle_deg < 3.0


def test_clean_line_with_random_outliers_is_not_a_split():
    """Scattered outliers do not form a second line."""
    primary = _line_points(40, 1900.0, 800.0, 0.3, 1.0)
    rng = np.random.default_rng(11)
    scatter = rng.uniform([1000, 500], [3000, 2500], size=(6, 2))
    xy = np.vstack([primary, scatter])
    fit = fit_dive_line(xy)

    assert detect_reflection_split(xy, fit) is None


def test_small_second_cluster_is_not_a_split():
    """Three off-line points are too few to be a line."""
    primary = _line_points(40, 1900.0, 800.0, 0.3, 1.0)
    artifact = primary[:3] + np.array([45.0, 0.0])  # only 3 points
    xy = np.vstack([primary, artifact])
    fit = fit_dive_line(xy)

    assert detect_reflection_split(xy, fit) is None


def test_crossing_line_is_not_a_reflection():
    """A second line at a steep angle is some other pathology — the
    reflection signature is parallelism."""
    primary = _line_points(30, 1900.0, 800.0, 0.3, 1.0)
    crossing = _line_points(20, 1200.0, 1500.0, 1.0, 0.05, seed=5)
    xy = np.vstack([primary, crossing])
    fit = fit_dive_line(xy)

    assert detect_reflection_split(xy, fit) is None


def test_no_primary_or_too_few_points_is_not_a_split():
    """No primary fit, or too few points for two lines, is never a split."""
    xy = _line_points(9, 1900.0, 800.0, 0.3, 1.0)
    assert detect_reflection_split(xy, None) is None
    # 9 < 2 * MIN_POINTS_FOR_LINE: too few to hold two lines.
    assert detect_reflection_split(xy, fit_dive_line(xy)) is None


# --- golden numbers ---------------------------------------------------------


def _services_reflection_mission():
    """fishsense-services' ``tests/_laser.py::reflection_mission`` as points:
    a noise-free laser line and its reflection 45 px to the left."""
    laser = [(1945.0 + 0.3 * i * 30.0, 800.0 + i * 30.0) for i in range(30)]
    reflection = [(1900.0 + 0.3 * i * 45.0, 800.0 + i * 45.0) for i in range(20)]
    return np.array(laser + reflection)


def _dive77_like(tilt_deg=0.0, extra_sep_px=0.0):
    """A synthetic dive 77: 79 frames, 34 clicked on a reflection 45 px off the
    laser line along its normal, prod-like 1.85 px perpendicular label noise.
    ``tilt_deg`` rotates the reflection cluster about its centroid;
    ``extra_sep_px`` pushes it further out."""
    rng = np.random.default_rng(77)
    direction = np.array([0.3, 1.0]) / np.hypot(0.3, 1.0)
    normal = np.array([-direction[1], direction[0]])
    t = rng.uniform(0, 1400, 79)
    pts = (
        np.array([1800.0, 600.0])
        + t[:, None] * direction
        + rng.normal(0, 1.85, 79)[:, None] * normal
    )
    refl = rng.choice(79, 34, replace=False)
    pts[refl] += (45.0 + extra_sep_px) * normal
    if tilt_deg:
        th = np.radians(tilt_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        centre = pts[refl].mean(axis=0)
        pts[refl] = (pts[refl] - centre) @ rot.T + centre
    return pts


def _services_split():
    primary = _line_points(30, 1900.0, 800.0, 0.3, 1.0)
    return np.vstack([primary, primary[:20] + np.array([45.0, 0.0])])


GOLDEN = {
    "services_split": (_services_split, (30, 20, 42.914710420656206, 0.07604547551613505)),
    "services_reflection_mission": (
        _services_reflection_mission,
        (30, 20, 43.10218283495192, 0.0),
    ),
    "dive77_like": (_dive77_like, (45, 34, 45.031652709399395, 0.008079875133415022)),
    "dive77_like_tilt2": (
        lambda: _dive77_like(tilt_deg=2.0),
        (45, 34, 41.342767258359686, 2.0080798751317865),
    ),
    "dive77_like_tilt5": (lambda: _dive77_like(tilt_deg=5.0), None),
    "dive77_like_sep250": (lambda: _dive77_like(extra_sep_px=205.0), None),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_suspects(name):
    """The port returns the services original's suspect exactly."""
    build, expected = GOLDEN[name]
    xy = build()
    suspect = detect_reflection_split(xy, fit_dive_line(xy))

    if expected is None:
        assert suspect is None
        return
    n_primary, n_secondary, separation_px, angle_deg = expected
    assert isinstance(suspect, ReflectionSuspect)
    assert suspect.n_primary == n_primary
    assert suspect.n_secondary == n_secondary
    assert suspect.separation_px == pytest.approx(separation_px, rel=1e-12)
    # ``angle_deg`` is arccos(|n1 . n2|) near cos = 1, where arccos's slope is
    # unbounded: a last-bit difference in the fitted normals (LAPACK build)
    # moves a 0.008 deg angle by ~5e-9 relative (seen on CI's openblas). An
    # absolute 1e-6 deg still fails on any real change to the fit.
    assert suspect.angle_deg == pytest.approx(angle_deg, rel=0, abs=1e-6)
