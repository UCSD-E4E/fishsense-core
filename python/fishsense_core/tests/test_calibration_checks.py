"""Tests for `fishsense_core.calibration_checks`: the outlier trim and the four
gates that guard `calibrate_laser`.

Ported from fishsense-services v0.48.0
`services/fishsense-services-processor/tests/` -- `test_robust_laser_fit.py`,
`test_calibration_consistency.py`, `test_calibration_baseline_gate.py`,
`test_calibration_observation_geometry.py`,
`test_calibration_describes_mission.py`,
`test_calibration_consistency_remedies.py` and
`test_calibration_gate_verdicts.py` -- with imports adapted and each file's
module-level fixtures prefixed so they can share one module. Each section
keeps its source's docstring as a comment. (`test_calibration_gates_are_wired.py`
tests services' orchestration, which is not ported.)

The golden section first pins numbers produced by the services implementation
on the same inputs, so a change to the arithmetic here shows up as a break in
parity with the refusal messages services has already stored.

These are synthetic geometry with a known answer, because there is no
ground-truth set of real calibrations to validate against. That is a real
limit: they pin that the checks behave as designed on the shapes seen in
prod, not that they are right on every real dive.
"""

# Seven ported suites plus the golden section; one module by request.
# pylint: disable=too-many-lines

from __future__ import annotations

import numpy as np
import pytest

from fishsense_core.calibration_checks import (
    ABSTAINED,
    DEFAULT_MAX_MISSION_MEDIAN_OFFSET_PX,
    DEFAULT_MAX_MISSION_P90_OFFSET_PX,
    MAX_BASELINE_M,
    MAX_TRIM_FRACTION,
    MIN_BASELINE_M,
    MIN_DISAGREEING_MISSION_DOTS,
    MIN_OBSERVATION_LEVER_M,
    PASSED,
    CalibrationDoesNotDescribeDiveError,
    CalibrationImplausibleError,
    CalibrationInconsistentError,
    CalibrationUnderdeterminedError,
    check_baseline_plausible,
    check_calibration_describes_mission,
    check_fit_self_consistency,
    check_observation_geometry,
    trim_outlying_observations,
)
from fishsense_core import calibration_checks


# ==========================================================================
# Golden values from the fishsense-services v0.48.0 implementation
#
# Produced by running services' `robust_fit` / `consistency` on these exact
# inputs. A fuzz of 3,000 random dives (trim -> `calibrate_laser` -> all four
# gates, 21,000 calls with every outcome represented) gave bit-identical
# results and identical refusal messages between the two implementations.
# ==========================================================================


def test_golden_contract_bounds():
    """The values services' `calibration_bounds` contract carries."""
    assert MIN_BASELINE_M == 0.097
    assert MAX_BASELINE_M == 0.145


def test_golden_baseline_bounds_are_overridable():
    """Services can keep passing its own contract values."""
    position = [0.0235 * 0.6, 0.0235 * 0.8, 0.0]
    assert check_baseline_plausible(position, min_baseline_m=0.02) == PASSED
    with pytest.raises(CalibrationImplausibleError, match=r"2\.35 cm .* 9\.7-14\.5 cm"):
        check_baseline_plausible(position)


def test_golden_dive_103_shape_trim():
    """Seventeen observations over 0.8-2.6 m, one gross outlier at index 11."""
    origin = np.array([0.0624, 0.0832, 0.0])
    axis = np.array([0.004, -0.006, 1.0]) / np.linalg.norm([0.004, -0.006, 1.0])
    rng = np.random.default_rng(103)
    depths = np.linspace(0.8, 2.6, 17)
    points = origin + (depths / axis[2])[:, None] * axis + rng.normal(0, 0.002, (17, 3))
    points[11] += [0.30, -0.20, 0.0]

    # pylint: disable-next=protected-access
    residuals = calibration_checks._residuals(points)
    assert residuals.max() == pytest.approx(0.3317196705762248, rel=1e-9)
    assert np.median(residuals) == pytest.approx(0.02139036281219016, rel=1e-9)

    kept = trim_outlying_observations(points)
    np.testing.assert_array_equal(kept, np.delete(points, 11, axis=0))


GOLDEN_K = np.array([[2840.0, 0.0, 2000.0], [0.0, 2860.0, 1450.0], [0.0, 0.0, 1.0]])
GOLDEN_ORIGIN = np.array([-0.031, -0.099, 0.0])
GOLDEN_AXIS = np.array([0.01, 0.03, 0.999]) / np.linalg.norm([0.01, 0.03, 0.999])


def _golden_dots() -> np.ndarray:
    rng = np.random.default_rng(7)
    out = []
    for z in np.linspace(0.4, 2.5, 20):
        q = GOLDEN_K @ (GOLDEN_ORIGIN + (z / GOLDEN_AXIS[2]) * GOLDEN_AXIS)
        out.append((q[0] / q[2] + rng.normal(0, 1.0), q[1] / q[2] + rng.normal(0, 1.0)))
    return np.array(out)


def test_golden_projected_ray():
    """The image line of the fitted ray, as services computes it."""
    # pylint: disable-next=protected-access
    centroid, direction = calibration_checks._project_ray_to_image_line(
        GOLDEN_ORIGIN, GOLDEN_AXIS, GOLDEN_K
    )
    np.testing.assert_allclose(centroid, [1958.19609622, 1310.01602512], rtol=1e-10)
    np.testing.assert_allclose(np.abs(direction), [0.29691895, 0.95490268], rtol=1e-7)


def test_golden_self_consistency_messages():
    """The statistics services reports for its own two refusal fixtures.

    Note the 10-deg axis error refuses through the *offset* branch: its image
    line is nearly parallel to the dots (0.07 deg) but 449 px away.
    """
    dots = _golden_dots()
    bad_axis = np.array([np.sin(np.radians(10.0)), 0.03, 0.99])
    bad_axis = bad_axis / np.linalg.norm(bad_axis)
    with pytest.raises(
        CalibrationInconsistentError,
        match=r"median offset 448\.6px .*; angle 0\.07 deg",
    ):
        check_fit_self_consistency(GOLDEN_ORIGIN, bad_axis, GOLDEN_K, dots)
    shifted = GOLDEN_ORIGIN + np.array([0.012, 0.0, 0.0])
    with pytest.raises(
        CalibrationInconsistentError,
        match=r"angle 6\.41 deg .*median offset 24\.8px",
    ):
        check_fit_self_consistency(shifted, GOLDEN_AXIS, GOLDEN_K, dots)


# ==========================================================================
# trim_outlying_observations -- ported from fishsense-services v0.48.0
# services/fishsense-services-processor/tests/test_robust_laser_fit.py
#
# Trimming outliers before the laser line fit.
#
# `calibrate_laser` is plain least squares with no outlier rejection, so a single
# badly-placed observation drags the whole line. That is not hypothetical: prod
# dive 77 shipped a calibration whose lengths reached +137% because labelers
# clicked the laser's specular reflection on about half the frames, and dive 103
# carries a 6.25 cm baseline from **17 otherwise clean observations** — the shape
# that neither a higher `MIN_LASER_POINTS` nor the baseline bound explains.
#
# The damage is amplified by what the fit is asked for. The reported origin is
# where the line crosses z=0, typically a metre or more behind the observations,
# so a small angular error from one bad point is levered into a large error in
# the baseline — which is exactly the quantity nothing downstream can see.
#
# These tests are built on synthetic geometry with a known answer, because there
# is no ground-truth set of real calibrations to validate against. That is a real
# limit: they pin that trimming recovers the clean line when the contamination is
# the shape we have seen, not that it helps on every real dive.
# ==========================================================================


#: A laser 10.4 cm off the camera centre, pointing essentially down the optical
#: axis — the fleet's real geometry.
TRIM_ORIGIN = np.array([0.0624, 0.0832, 0.0])
TRIM_AXIS = np.array([0.004, -0.006, 1.0]) / np.linalg.norm([0.004, -0.006, 1.0])


def _trim_clean_points(n: int = 16, *, noise: float = 0.0, seed: int = 0):
    """Observations along the true laser ray, spread over a real depth range."""
    rng = np.random.default_rng(seed)
    depths = np.linspace(0.5, 3.0, n)
    points = TRIM_ORIGIN[None, :] + (depths / TRIM_AXIS[2])[:, None] * TRIM_AXIS[None, :]
    if noise:
        points = points + rng.normal(0.0, noise, points.shape)
    return points


def test_clean_observations_are_all_kept():
    """Trimming must not fire on good data, or it silently discards signal."""
    points = _trim_clean_points()

    kept = trim_outlying_observations(points)

    assert len(kept) == len(points)


def test_realistic_noise_is_kept():
    """Sub-centimetre scatter is ordinary and must not be mistaken for outliers."""
    points = _trim_clean_points(noise=0.004)

    kept = trim_outlying_observations(points)

    assert len(kept) == len(points)


def test_a_single_gross_outlier_is_dropped():
    """One bad dot is the dive-103 shape: many clean points, one wrecking them."""
    points = _trim_clean_points(noise=0.002)
    points[7] = points[7] + np.array([0.35, -0.25, 0.0])

    kept = trim_outlying_observations(points)

    assert len(kept) == len(points) - 1
    # The survivor set must be the clean ones, not merely one fewer.
    assert not np.any(np.all(np.isclose(kept, points[7]), axis=1))


def test_trimming_recovers_the_clean_geometry():
    """The property that matters: the fit after trimming is the fit without
    the outlier, not just a different fit."""
    clean = _trim_clean_points(noise=0.002)
    contaminated = clean.copy()
    contaminated[3] = contaminated[3] + np.array([0.4, 0.3, 0.0])

    kept = trim_outlying_observations(contaminated)

    assert len(kept) == len(clean) - 1
    for point in kept:
        assert np.any(np.all(np.isclose(clean, point), axis=1))


def test_several_scattered_outliers_are_dropped():
    """Independently misplaced dots — the common case — are all removed.

    Displacements are fixed rather than sampled: a random draw occasionally
    lands a "outlier" close enough to the ray to be legitimately kept, which
    makes the test flaky about the threshold rather than about the property.
    """
    points = _trim_clean_points(n=20, noise=0.002)
    points[2] = points[2] + np.array([0.30, -0.20, 0.0])
    points[9] = points[9] + np.array([-0.25, 0.35, 0.0])
    points[15] = points[15] + np.array([0.40, 0.15, 0.0])

    kept = trim_outlying_observations(points)

    assert len(kept) == 17


def test_a_parallel_reflection_cluster_is_not_trimmed():
    """The dive-77 shape, and the limit of this approach — pinned deliberately.

    When a contiguous subset is displaced by a *constant* offset it forms a
    second line parallel to the true one. The total-least-squares fit then
    splits the difference between the two, so every observation carries a
    similar moderate residual and none of them looks like an outlier. No
    residual-based trim can separate them, however the threshold is tuned.

    That failure is already covered, by `check_fit_self_consistency`: two
    parallel dot lines make the fitted ray reproject at an angle to the dots it
    came from, which is exactly what that gate measures (dive 77 read 14 deg
    against a fleet baseline of <=1.6 deg).

    So the two guards are complementary, not redundant — trimming handles
    isolated bad dots, self-consistency handles systematic contamination. This
    test exists so nobody later "fixes" the trimmer to chase this case and
    starts discarding good observations in the process.
    """
    points = _trim_clean_points(n=20, noise=0.002)
    points[:5] = points[:5] + np.array([0.18, 0.0, 0.0])

    kept = trim_outlying_observations(points)

    assert len(kept) == 20


def test_it_refuses_to_trim_more_than_the_cap():
    """A cap, so a half-contaminated set cannot be silently 'cleaned'.

    If most observations disagree there is no majority to trust, and quietly
    fitting the larger half would produce a confident answer from data we have
    no reason to believe. Better to keep everything and let the baseline gate
    refuse the result.
    """
    points = _trim_clean_points(n=20, noise=0.002)
    # Half the set displaced — no defensible majority.
    points[:10] = points[:10] + np.array([0.3, 0.25, 0.0])

    kept = trim_outlying_observations(points)

    assert len(kept) >= len(points) * (1 - MAX_TRIM_FRACTION)


def test_too_few_observations_are_returned_untouched():
    """At the `MIN_LASER_POINTS` floor there is no redundancy to trim with.

    Two points always fit a line exactly, so every residual is zero and any
    trim rule is reading noise. Returning them unchanged keeps this from
    turning a thin-but-valid calibration into no calibration at all.
    """
    points = _trim_clean_points(n=2)

    kept = trim_outlying_observations(points)

    assert len(kept) == 2


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_inputs_pass_through(n):
    """Zero or one observation comes back as-is."""
    points = _trim_clean_points(n=max(n, 1))[:n]

    kept = trim_outlying_observations(points)

    assert len(kept) == n


def test_input_is_not_mutated():
    """The trim returns a new array; the caller's is untouched."""
    points = _trim_clean_points(noise=0.002)
    points[5] = points[5] + np.array([0.4, 0.0, 0.0])
    before = points.copy()

    trim_outlying_observations(points)

    assert np.array_equal(points, before)


# ==========================================================================
# check_fit_self_consistency -- ported from fishsense-services v0.48.0
# services/fishsense-services-processor/tests/test_calibration_consistency.py
#
# Unit tests for the stage-13 self-consistency gate (pure numpy, no cv2).
#
# The gate exists because of prod dive 77 (2026-08-04): labelers clicked the
# laser's specular reflection on ~half the pool slate frames, the laser dots
# formed two parallel lines, and the 3D fit split the difference — 14 deg off
# the dive's own dot line, with a fleet-outlier origin. The bad calibration was
# persisted, borrowed by a fish-model dive, and produced +31..+137% length
# errors. A fit that cannot reproject onto the very dots it was computed from
# must never persist. Fleet baseline: every good calibration reprojects onto
# its own dots at <=1.6 deg / <=4 px median; 77-bad was 14 deg / 17 px.
# ==========================================================================


SC_K = np.array(
    [
        [2840.0, 0.0, 2000.0],
        [0.0, 2860.0, 1450.0],
        [0.0, 0.0, 1.0],
    ]
)
SC_ORIGIN = np.array([-0.031, -0.099, 0.0])
SC_AXIS = np.array([0.01, 0.03, 0.999])
SC_AXIS = SC_AXIS / np.linalg.norm(SC_AXIS)


def _sc_project(p3: np.ndarray) -> tuple[float, float]:
    q = SC_K @ p3
    return float(q[0] / q[2]), float(q[1] / q[2])


def _sc_dots_on_ray(depths, noise_px: float = 0.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    for z in depths:
        t = (z - SC_ORIGIN[2]) / SC_AXIS[2]
        x, y = _sc_project(SC_ORIGIN + t * SC_AXIS)
        pts.append((x + rng.normal(0, noise_px), y + rng.normal(0, noise_px)))
    return np.array(pts)


def test_consistent_fit_passes():
    """Dots on the fitted ray with 1 px of noise pass."""
    dots = _sc_dots_on_ray(np.linspace(0.4, 2.5, 20), noise_px=1.0)
    # Must not raise.
    check_fit_self_consistency(SC_ORIGIN, SC_AXIS, SC_K, dots)


def test_rotated_axis_is_rejected():
    """A 10-deg axis error (the dive-77 class of failure) must be rejected."""
    dots = _sc_dots_on_ray(np.linspace(0.4, 2.5, 20), noise_px=1.0)
    bad_axis = np.array([np.sin(np.radians(10.0)), 0.03, 0.99])
    bad_axis = bad_axis / np.linalg.norm(bad_axis)
    with pytest.raises(CalibrationInconsistentError, match="angle"):
        check_fit_self_consistency(SC_ORIGIN, bad_axis, SC_K, dots)


def test_two_line_mixture_is_rejected():
    """Dots split across two parallel lines (reflection artifact): the fit
    projects between them, so the median offset blows the gate."""
    line_a = _sc_dots_on_ray(np.linspace(0.4, 2.5, 12), noise_px=1.0)
    line_b = line_a + np.array([45.0, 0.0])  # parallel artifact line
    dots = np.vstack([line_a, line_b])
    shifted_origin = SC_ORIGIN + np.array([0.008, 0.0, 0.0])  # fit splits the gap
    with pytest.raises(CalibrationInconsistentError, match="offset"):
        check_fit_self_consistency(shifted_origin, SC_AXIS, SC_K, dots)


def test_small_span_skips_gate():
    """Two nearly-coincident dots can't determine a line direction — the gate
    must skip (not raise) rather than reject on meaningless geometry."""
    dots = _sc_dots_on_ray([1.0, 1.005])
    bad_axis = np.array([np.sin(np.radians(10.0)), 0.03, 0.99])
    bad_axis = bad_axis / np.linalg.norm(bad_axis)
    # Must not raise despite the bad axis: span too small to judge.
    check_fit_self_consistency(SC_ORIGIN, bad_axis, SC_K, dots)


def test_offset_fit_is_rejected():
    """Right direction, wrong origin (parallel-shifted projection)."""
    dots = _sc_dots_on_ray(np.linspace(0.4, 2.5, 20), noise_px=1.0)
    shifted = SC_ORIGIN + np.array([0.012, 0.0, 0.0])  # ~30 px shift at ~1 m
    with pytest.raises(CalibrationInconsistentError, match="offset"):
        check_fit_self_consistency(shifted, SC_AXIS, SC_K, dots)


# ==========================================================================
# check_baseline_plausible -- ported from fishsense-services v0.48.0
# services/fishsense-services-processor/tests/test_calibration_baseline_gate.py
#
# The baseline plausibility gate — the one check that sees the failing axis.
#
# `check_fit_self_consistency` compares the fitted ray's *projection* against the
# 2D dots it came from. Many different 3D rays project to the same image line —
# the whole family lying in the plane through the camera centre and that line —
# and sliding the laser's offset along that family moves where the ray crosses
# z=0 while leaving the projection identical. So that gate constrains two of the
# ray's four degrees of freedom and the baseline sits in the two it cannot see.
# It is the epipolar blindness already documented for reprojection residual and
# scale, in a second guise.
#
# Measured 2026-09-11 over all 35 stored calibrations: the baseline is a rig
# constant. Its interquartile range is 9.99–10.45 cm — half a centimetre across
# both producers, every camera and two years — while 8 fits sit outside 8–13 cm,
# at 2.35, 2.60, 4.81, 4.90, 6.25, 6.91, 16.01 and 22.22 cm. Those 8 back 12
# dives and 663 of 3,104 measurements, and their length errors run -75% to +45%.
#
# Two of the eight are **slate**-derived, which is why this gate lives beside
# `check_fit_self_consistency` rather than in the checkerboard path: both
# producers feed the same fit and both can miss the same way.
# ==========================================================================


def _bl_position(baseline_m: float) -> np.ndarray:
    """A laser position with the given offset from the camera centre.

    The fit returns the origin with z implicit at 0 (the crossing of the
    camera plane), so the baseline is the norm of the x/y pair.
    """
    return np.array([baseline_m * 0.6, baseline_m * 0.8, 0.0])


@pytest.mark.parametrize("baseline_cm", [9.87, 10.12, 10.36, 10.45, 12.95])
def test_accepts_every_baseline_the_fleet_actually_produces(baseline_cm):
    """Real good fits must pass, or the gate is useless in practice.

    These are measured values from prod calibrations whose lengths grade
    within -12.7%, including the two extremes of the healthy band.
    """
    check_baseline_plausible(_bl_position(baseline_cm / 100))


@pytest.mark.parametrize(
    "baseline_cm", [2.35, 2.60, 4.81, 4.90, 6.25, 6.91, 16.01, 22.22]
)
def test_refuses_every_baseline_that_produced_bad_lengths(baseline_cm):
    """The eight known-bad fits, by their measured baselines.

    Parametrised on the real numbers rather than invented ones so the gate is
    pinned against the population it was derived from — if someone widens the
    bound far enough to re-admit these, this fails.
    """
    with pytest.raises(CalibrationImplausibleError):
        check_baseline_plausible(_bl_position(baseline_cm / 100))


def test_the_error_names_the_measured_baseline_and_the_bound():
    """An operator reading the log must be able to tell how far off it was."""
    with pytest.raises(CalibrationImplausibleError) as excinfo:
        check_baseline_plausible(_bl_position(0.0235))

    message = str(excinfo.value)
    assert "2.35" in message or "0.0235" in message
    assert "refus" in message.lower()


def test_a_zero_baseline_is_refused():
    """A laser at the camera centre is not a rig, it is a failed fit."""
    with pytest.raises(CalibrationImplausibleError):
        check_baseline_plausible(np.array([0.0, 0.0, 0.0]))


def test_a_non_finite_baseline_is_refused():
    """NaN comparisons are all False, so a naive bound would ACCEPT this."""
    with pytest.raises(CalibrationImplausibleError):
        check_baseline_plausible(np.array([float("nan"), 0.0, 0.0]))


def test_the_z_component_is_ignored():
    """The fit returns the origin at the z=0 crossing and pads z.

    Reading a three-component norm here would make the gate depend on padding
    that carries no information, and would drift if that convention changed.
    """
    check_baseline_plausible(np.array([0.104, 0.0, 5.0]))


def test_bounds_bracket_the_measured_fleet_cluster():
    """The constants themselves, so widening them is a deliberate act.

    The fleet sits at 9.99–10.45 cm interquartile. The bound must contain that
    comfortably and still exclude 9.51 cm and 16.01 cm, the nearest bad fits on
    each side.
    """
    assert MIN_BASELINE_M < 0.0987
    assert MIN_BASELINE_M > 0.0951
    assert MAX_BASELINE_M > 0.1296
    assert MAX_BASELINE_M < 0.1600


# ==========================================================================
# check_observation_geometry -- ported from fishsense-services v0.48.0
# services/fishsense-services-processor/tests/test_calibration_observation_geometry.py
#
# `MIN_LASER_POINTS` counts observations; it does not ask where they are.
#
# Stage 13 fits the laser ray from 3-D points, each the camera ray through a
# labelled dot intersected with the target plane. What determines the fitted
# *direction* is the lever arm — the spread of those points along the ray —
# against the label noise. One pixel of dot-label noise at range z is z/f metres
# of lateral error, so it rotates the fitted ray by about (z/f)/lever radians,
# and the measured sensitivity of length to that angle is ~30 % per degree at
# 2 m. The observation count says nothing about this: sixteen dots at one range
# determine the direction no better than two.
#
# Measured over the 32 stored calibrations whose observations are recoverable
# (2026-09-13), lever arm = depth spread of the extreme observations:
#
#     dive 341   30 obs   1.55 m   ->  0.1 % of length per px of label noise
#     dive 471    2 obs   1.56 m   ->  0.7 %
#     dive 279    3 obs   1.47 m   ->  0.6 %
#     dive 465    3 obs   1.34 m   ->  0.6 %
#     dive 383    2 obs   1.09 m   ->  0.7 %
#     dive 062   20 obs   1.02 m   ->  0.4 %     <- tightest sound calibration
#     dive 349    2 obs   0.26 m   ->  5.8 %     <- refused here
#     dive 107   16 obs   0.03 m   ->  5.5 %     <- refused here
#     dive 347    1 obs   0.00 m   ->  degenerate
#
# Two observations over a metre of range are worth more than sixteen at one
# distance, which is why this gate bounds the lever arm and not the count. The
# bound sits between the populations (1.02 m healthy, 0.26 m bad), nearer the
# bad side because a refusal wedges the dive in its cohort.
#
# Dive 107 is the case that shows the existing gates cannot do this: sixteen
# observations, a 12.95 cm baseline that `check_baseline_plausible` accepts as
# the fleet's high extreme, a dot span wide enough that
# `check_fit_self_consistency` does not abstain — and a 2.8 cm lever arm
# (1.97-2.00 m). Dive 526 had the same single-distance geometry at 6.6 cm
# (4.16-4.22 m) and was only caught because its fit happened to collapse to a
# 2.00 cm baseline.
# ==========================================================================


GEO_AXIS = np.array([0.00737, 0.03204, 0.99946])


def _geo_observations(depths, origin=(-0.0306, -0.0977, 0.0)) -> np.ndarray:
    """3-D observation points along the laser ray at the given depths — the
    shape `_gather_laser_points` hands to the fit."""
    origin = np.asarray(origin, float)
    axis = GEO_AXIS / np.linalg.norm(GEO_AXIS)
    ts = (np.asarray(depths, float) - origin[2]) / axis[2]
    return origin[None, :] + ts[:, None] * axis[None, :]


# --- accepted ---------------------------------------------------------------


def test_two_observations_over_a_wide_range_are_enough():
    """Dive 471's geometry: two dots, 1.56 m apart. Well determined."""
    check_observation_geometry(_geo_observations([1.12, 2.68]))


def test_a_long_burst_over_a_wide_range_is_accepted():
    """Thirty observations over 1.55 m of range."""
    check_observation_geometry(_geo_observations(np.linspace(1.19, 2.74, 30)))


def test_the_tightest_healthy_mission_is_accepted():
    """Dive 62: 1.02 m of lever over 20 observations — the closest sound
    calibration to the bound, so this pins that the bound does not refuse it.
    Dive 383's 1.09 m is the next tightest and is covered by the same margin."""
    check_observation_geometry(_geo_observations(np.linspace(1.38, 2.40, 20)))


# --- refused ----------------------------------------------------------------


def test_a_single_distance_burst_is_refused_however_many_observations():
    """Dive 107: sixteen observations inside 3 cm of range. Every other gate
    passes it; its 12.95 cm baseline is the fleet's highest."""
    rng = np.random.default_rng(0)
    depths = 1.97 + rng.uniform(0.0, 0.03, 16)
    with pytest.raises(CalibrationUnderdeterminedError, match="lever"):
        check_observation_geometry(_geo_observations(depths))


def test_two_close_observations_are_refused():
    """Dive 349: two dots 26 cm apart at 2.4 m — 5.8 % of length per pixel."""
    with pytest.raises(CalibrationUnderdeterminedError):
        check_observation_geometry(_geo_observations([2.361, 2.622]))


def test_a_single_observation_is_refused():
    """One observation has no lever arm at all."""
    with pytest.raises(CalibrationUnderdeterminedError):
        check_observation_geometry(_geo_observations([1.197]))


def test_coincident_observations_are_refused():
    """Two labels at the identical pixel on one frame are one observation."""
    with pytest.raises(CalibrationUnderdeterminedError):
        check_observation_geometry(_geo_observations([1.2, 1.2]))


def test_empty_is_refused_rather_than_crashing():
    """No observations is a refusal, not an exception from numpy."""
    with pytest.raises(CalibrationUnderdeterminedError):
        check_observation_geometry(np.zeros((0, 3)))


# --- the bound and the message ---------------------------------------------


def test_bound_sits_between_the_measured_populations():
    """Healthy reaches down to 1.02 m (dive 62, 20 observations); the bad ones
    are at 0.26 m and below (349, 107, 347). Nearer the bad side: a refusal is
    expensive. Pinned at 1.02 and not 1.09, so that raising the bound past the
    tightest *sound* calibration cannot pass with both tests green."""
    assert 0.26 < MIN_OBSERVATION_LEVER_M < 1.02


def test_the_message_reports_the_lever_and_the_count():
    """An operator can read the lever and the count off the refusal."""
    with pytest.raises(CalibrationUnderdeterminedError) as exc:
        check_observation_geometry(_geo_observations([2.361, 2.622]))
    message = str(exc.value)
    assert "0.26" in message
    assert "2 observation" in message
    assert "lever" in message


def test_non_finite_observations_are_dropped_not_counted():
    """A NaN/inf row neither adds lever nor poisons the max/min."""
    obs = _geo_observations([1.12, 2.68, 1.9])
    obs[2] = [np.nan, np.nan, np.nan]
    check_observation_geometry(obs)
    obs2 = _geo_observations([2.361, 2.622, 1.0])
    obs2[2] = [np.inf, 0.0, 0.0]
    with pytest.raises(CalibrationUnderdeterminedError):
        check_observation_geometry(obs2)


# ==========================================================================
# check_calibration_describes_mission -- ported from fishsense-services v0.48.0
# services/fishsense-services-processor/tests/test_calibration_describes_mission.py
#
# The gate that asks whether a calibration describes the dive it will measure.
#
# `check_fit_self_consistency` compares the fitted ray's projection against the
# dots it was computed from, so it is satisfied by construction and says nothing
# when those dots are few. On prod dive 347 it said nothing at all: the fit came
# from ONE frame carrying a duplicate laser label at the identical pixel, so the
# dot set had N=1 and the check abstained, while `MIN_LASER_POINTS = 2` was
# satisfied by counting the duplicate as a second observation. The calibration
# that resulted is 9.4 px away from the dive's own 321 laser dots.
#
# This gate asks the other question: project the fitted ray and compare it
# against **every live dot in the dive**, including the measurement frames the
# calibration was never fitted to. That is not tautological — it is the check
# that the laser was in the same state for the frames being measured as it was
# for the frames it was calibrated from, which is precisely what a mid-dive
# re-seat breaks (prod dive 490, where the board burst and the fish frames
# disagreed by 0.82 deg).
#
# Measured over all 32 stored calibrations against their dives' live dots
# (2026-09-13), median perpendicular offset in px:
#
#     0.43 - 0.88   twenty dives
#     1.31 - 1.55   six dives
#     2.01, 2.27    dives 279, 509
#     3.33, 3.92    dives 341, 349   <- real burst-vs-frames disagreement
#     5.37          dive 498         <- also caught by the baseline gate
#     9.42          dive 347         <- the degenerate fit, caught by nothing else
#
# The bounds sit between those populations, not against the healthy one — the
# same rule the baseline gate uses, and for the same reason: a refusal wedges
# the dive in its cohort, so a false positive is expensive.
#
# **It is blind to the baseline, deliberately and unavoidably.** Sliding the
# laser's offset along the family of rays that share an image line leaves the
# projection identical, so five of the six known-bad baselines sit at 0.43-1.35
# px here and pass. That is what `check_baseline_plausible` is for; the two
# gates are complementary and neither subsumes the other.
# ==========================================================================


# A realistic Olympus rig: the intrinsics and laser geometry of prod dive 347's
# camera, so the pixel scales in these tests are the ones the gate really sees.
MIS_K = np.array(
    [
        [2855.292793016781, 0.0, 2031.7894533541553],
        [0.0, 2881.0734769839373, 1447.6913944790074],
        [0.0, 0.0, 1.0],
    ]
)
MIS_ORIGIN = np.array([-0.0306, -0.0977, 0.0])
MIS_AXIS = np.array([0.00737, 0.03204, 0.99946])


def _mis_dots_on_ray(origin=MIS_ORIGIN, axis=MIS_AXIS, depths=None) -> np.ndarray:
    """The pixels a dot would occupy at a spread of ranges — i.e. dots that lie
    exactly on the projection of `origin` + t * `axis`."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    depths = np.linspace(0.6, 3.5, 40) if depths is None else np.asarray(depths)
    ts = (depths - origin[2]) / axis[2]
    points = np.asarray(origin, float)[None, :] + ts[:, None] * axis[None, :]
    homogeneous = (MIS_K @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _mis_perpendicular(dots: np.ndarray, px: float) -> np.ndarray:
    """Slide `dots` `px` pixels perpendicular to their own long axis."""
    centred = dots - dots.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    normal = np.array([-vt[0][1], vt[0][0]])
    return dots + px * normal


# --- the gate passes what it should -----------------------------------------


def test_dots_on_the_projected_ray_pass():
    """Dots exactly on the projected ray pass."""
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, _mis_dots_on_ray())


def test_realistic_label_noise_passes():
    """A sound dive sits at 0.4-2.3 px; a couple of px of label noise is normal."""
    rng = np.random.default_rng(0)
    dots = _mis_perpendicular(_mis_dots_on_ray(), 0.0) + rng.normal(0, 1.5, (40, 2))
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


def test_the_gate_is_blind_to_the_baseline_by_construction():
    """Scaling the laser's offset moves the ray but not its projection, so this
    gate cannot see a wrong baseline. Pinned so nobody expects it to."""
    dots = _mis_dots_on_ray()
    for scale in (0.5, 2.0, 4.0):
        check_calibration_describes_mission(MIS_ORIGIN * scale, MIS_AXIS, MIS_K, dots)


# --- and refuses what it should ---------------------------------------------


def test_a_uniformly_offset_dot_population_is_refused():
    """Dive 347's shape: the calibration's line sits ~9 px off every dot in the
    dive, because it was fitted from one frame that happened to lie elsewhere."""
    dots = _mis_perpendicular(_mis_dots_on_ray(), 9.4)
    with pytest.raises(CalibrationDoesNotDescribeDiveError, match="median"):
        check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


def test_a_rotated_ray_is_refused():
    """A mid-dive re-seat: the stored axis is a fraction of a degree off the one
    the measurement frames were shot with."""
    rotated = MIS_AXIS + np.array([0.004, -0.004, 0.0])
    dots = _mis_dots_on_ray()
    with pytest.raises(CalibrationDoesNotDescribeDiveError):
        check_calibration_describes_mission(MIS_ORIGIN, rotated, MIS_K, dots)


def test_a_wild_minority_is_caught_by_the_p90_even_when_the_median_is_fine():
    """Dive 498's shape: median 5.4 px but p90 29.7. A median-only bound would
    let a partial disagreement through."""
    dots = _mis_dots_on_ray()
    dots[-8:] = _mis_perpendicular(dots, 40.0)[-8:]
    with pytest.raises(CalibrationDoesNotDescribeDiveError, match="p90"):
        check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


def test_a_disagreeing_subset_is_refused_once_it_is_a_real_population():
    """Dive 498's true proportions: 13 of its 40 live dots sit beyond the p90
    bound. The floor must not be so high that this stops being refused."""
    dots = _mis_dots_on_ray(depths=np.linspace(0.6, 3.5, 40))
    dots[-13:] = _mis_perpendicular(dots, 40.0)[-13:]
    with pytest.raises(CalibrationDoesNotDescribeDiveError, match="p90"):
        check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


# --- the p90 branch needs a real subset, not one or two labels --------------
#
# The dots come from the whole dive, so they include labels the fit never saw
# and labels nobody has checked yet. The 3 sigma per-dive validator that would
# supersede a reflection mislabel only runs on dives whose laser labelling is
# *complete*, so during labelling the population is unpoliced -- and because a
# recorded refusal self-expires the moment any label on the dive changes, a
# refusal earned this way comes back every hour a labeler works.


def test_a_pair_of_mislabels_does_not_refuse_a_fit_they_never_fed():
    """Two reflection mislabels in a partly-labelled dive. `np.percentile`
    interpolates, so at N=18 the 90th percentile already reaches into the two
    worst values and lands at ~18 px with the fit itself exact."""
    dots = _mis_dots_on_ray(depths=np.linspace(0.6, 3.5, 18))
    dots[-2:] = _mis_perpendicular(dots, 60.0)[-2:]
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


def test_one_wild_label_does_not_refuse_at_the_minimum_dot_count():
    """At N=6 the 90th percentile is effectively the maximum, while the bounds
    were measured on populations of 13 to 321 dots."""
    dots = _mis_dots_on_ray(depths=np.linspace(0.6, 3.5, 6))
    dots[-1] = _mis_perpendicular(dots, 60.0)[-1]
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


def test_the_median_branch_still_fires_below_the_disagreeing_dot_floor():
    """The floor is on the p90 branch only. A dive whose every dot disagrees
    is refused however few dots it has — that is dive 347, and the whole
    point of the gate."""
    dots = _mis_perpendicular(_mis_dots_on_ray(depths=np.linspace(0.6, 3.5, 6)), 9.4)
    with pytest.raises(CalibrationDoesNotDescribeDiveError, match="median"):
        check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


# --- abstention, so the gate never fires on data that cannot answer ---------


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_abstains_on_too_few_dots(n):
    """Same posture as `check_fit_self_consistency`: too little to judge means
    say nothing, not refuse. A dive with a handful of dots says nothing about
    the fit; what exposes a fit that had nothing to constrain it is
    `check_observation_geometry`, on the lever arm of the observations — not
    a count, which is the quantity dive 347 satisfied with a duplicate."""
    dots = _mis_perpendicular(_mis_dots_on_ray(depths=np.linspace(0.6, 3.5, max(n, 1))), 60.0)
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots[:n])


def test_abstains_when_the_dots_do_not_span_enough_pixels():
    """All the dots at one range define no direction, so a perpendicular offset
    is not measurable against them."""
    dots = _mis_perpendicular(_mis_dots_on_ray(depths=np.full(30, 1.2)), 60.0)
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


def test_nan_and_infinite_dots_are_ignored_rather_than_poisoning_the_statistic():
    """Non-finite dots are dropped before the median and p90."""
    dots = _mis_dots_on_ray()
    dots[3] = [np.nan, np.nan]
    dots[7] = [np.inf, 0.0]
    check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)


# --- the bounds themselves --------------------------------------------------


def test_bounds_sit_between_the_measured_populations():
    """Healthy dives reach median 2.27 px (dive 509) with the two ambiguous
    ones at 3.33 and 3.92 (341, 349); the bad ones are 5.37 and 9.42 (498,
    347). Healthy p90 reaches 6.07 (341) against 18.44 and 29.69 for the bad.
    Widening past these re-admits a known-bad fit; tightening refuses dives
    whose calibrations are merely imperfect."""
    assert 3.92 < DEFAULT_MAX_MISSION_MEDIAN_OFFSET_PX < 9.42
    assert 6.07 < DEFAULT_MAX_MISSION_P90_OFFSET_PX < 18.44


def test_the_disagreeing_dot_floor_sits_between_the_measured_populations():
    """Re-measured 2026-09-13 over all 32 stored calibrations against their
    dives' live dots: **every one of the 30 sound calibrations has zero dots
    beyond the p90 bound.** The two refused have 13 (dive 498, of 40) and 139
    (dive 347, of 321). So the floor has only to clear the handful of stray
    labels a dive accumulates mid-labelling and stay below 13."""
    assert 2 < MIN_DISAGREEING_MISSION_DOTS < 13


def test_error_names_the_mission_statistic_so_a_refusal_is_actionable():
    """The refusal names the measured median and the bound."""
    dots = _mis_perpendicular(_mis_dots_on_ray(), 9.4)
    with pytest.raises(CalibrationDoesNotDescribeDiveError) as exc:
        check_calibration_describes_mission(MIS_ORIGIN, MIS_AXIS, MIS_K, dots)
    message = str(exc.value)
    assert "9.4" in message or "9.3" in message
    assert "px" in message
    assert str(int(DEFAULT_MAX_MISSION_MEDIAN_OFFSET_PX)) in message


# ==========================================================================
# Remedies are the caller's
#
# fishsense-services' REFUSAL_REMEDIES (its operator levers: the refusal
# record, priority 'none', calibration_target_id) used to be baked into every
# message here. Core's messages say what was measured and stop; a caller
# appends its own remedy text, and services passes its constant to keep its
# stored messages byte-identical. The text's own content tests stay in
# services, where its levers are.
# ==========================================================================

SERVICES_REMEDIES = (
    "The refusal is recorded, so this dive has left the calibration cohort "
    "and nothing is retrying it. It returns on its own once any laser or "
    "slate label on it is newer than the refusal, or its slate template or "
    "calibration target changes, so relabelling is enough and there is no "
    "flag to clear. If its observations cannot be fixed, park it with "
    "priority 'none' and a note; for a checkerboard dive that needs "
    "different frames rather than better labels, clear its "
    "calibration_target_id."
)


def _refusal(gate, **remedies):
    with pytest.raises(ValueError) as exc:
        gate(**remedies)
    return str(exc.value)


_GATES = {
    "baseline": lambda **kw: check_baseline_plausible([0.02, 0.0, 0.0], **kw),
    "geometry": lambda **kw: check_observation_geometry(np.zeros((3, 3)), **kw),
}


@pytest.mark.parametrize("gate", sorted(_GATES))
def test_a_core_refusal_carries_no_services_text(gate):
    message = _refusal(_GATES[gate])
    assert "calibration_target_id" not in message
    assert "priority 'none'" not in message
    assert message.endswith("Refusing to persist.")


@pytest.mark.parametrize("gate", sorted(_GATES))
def test_a_callers_remedies_are_appended_verbatim(gate):
    """Services passes its constant and gets the message it stores today."""
    message = _refusal(_GATES[gate], remedies=SERVICES_REMEDIES)
    assert message.endswith("Refusing to persist. " + SERVICES_REMEDIES)


# ==========================================================================
# gate verdicts -- ported from fishsense-services v0.48.0
# services/fishsense-services-processor/tests/test_calibration_gate_verdicts.py
#
# v2: each gate says what it did, so the calibration row can record it.
#
# v1's gates returned None whether they judged or abstained, so a stored
# calibration could not say which gates actually looked at it. PLAN.md §4.3 asks
# the laser calibration to record "each gate verdict"
# (`laser_calibrations.gate_verdicts`), and an abstention is exactly the case
# worth knowing about afterwards: dive 347's fit passed the self-consistency
# gate by abstaining on one dot. A refusal still raises, as in v1 (the other
# suites pin that); these pin the value returned when a gate does not refuse.
# ==========================================================================


VER_K = np.array([[2840.0, 0.0, 2000.0], [0.0, 2860.0, 1450.0], [0.0, 0.0, 1.0]])
VER_ORIGIN = np.array([-0.031, -0.099, 0.0])
VER_AXIS = np.array([0.01, 0.03, 0.999]) / np.linalg.norm([0.01, 0.03, 0.999])


def _ver_dots(depths) -> np.ndarray:
    ts = (np.asarray(depths, float) - VER_ORIGIN[2]) / VER_AXIS[2]
    points = VER_ORIGIN[None, :] + ts[:, None] * VER_AXIS[None, :]
    homogeneous = (VER_K @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _ver_points(depths) -> np.ndarray:
    ts = (np.asarray(depths, float) - VER_ORIGIN[2]) / VER_AXIS[2]
    return VER_ORIGIN[None, :] + ts[:, None] * VER_AXIS[None, :]


def test_a_judged_fit_is_reported_as_passed():
    """Each gate returns PASSED when it judged and did not refuse."""
    depths = np.linspace(0.6, 3.0, 20)
    assert check_observation_geometry(_ver_points(depths)) == PASSED
    assert check_fit_self_consistency(VER_ORIGIN, VER_AXIS, VER_K, _ver_dots(depths)) == PASSED
    assert check_baseline_plausible([0.0624, 0.0832, 0.0]) == PASSED
    assert (
        check_calibration_describes_mission(VER_ORIGIN, VER_AXIS, VER_K, _ver_dots(depths))
        == PASSED
    )


def test_an_abstention_is_reported_as_abstained_not_passed():
    """Dive 347's shape: one dot is too few to define a line, so the
    projection gates say nothing -- and the row should say they said nothing."""
    one = _ver_dots([1.2])
    assert check_fit_self_consistency(VER_ORIGIN, VER_AXIS, VER_K, one) == ABSTAINED
    assert check_calibration_describes_mission(VER_ORIGIN, VER_AXIS, VER_K, one) == ABSTAINED


def test_the_verdicts_are_distinct_strings():
    """The verdicts are the strings services records."""
    assert {PASSED, ABSTAINED} == {"passed", "abstained"}


# ==========================================================================
# The review's fixes
# ==========================================================================


def test_the_outlier_rule_is_line_fits_not_a_copy():
    from fishsense_core import line_fit  # pylint: disable=import-outside-toplevel

    assert calibration_checks.DEFAULT_OUTLIER_SIGMA is line_fit.DEFAULT_OUTLIER_SIGMA
    assert calibration_checks.MAD_TO_SIGMA is line_fit.MAD_TO_SIGMA
    for gone in ("OUTLIER_SIGMA", "DEFAULT_MIN_BASELINE_M", "DEFAULT_MAX_BASELINE_M",
                 "MIN_LASER_POINTS", "REFUSAL_REMEDIES"):  # fmt: skip
        assert gone not in calibration_checks.__all__


def test_the_trim_drops_a_non_finite_observation_instead_of_crashing():
    """One failed pose or depth used to make the SVD raise LinAlgError, before
    the geometry gate could refuse cleanly."""
    depths = np.linspace(0.6, 3.0, 12)
    points = _ver_points(depths)
    points[4] = [np.nan, 0.0, 1.0]
    kept = trim_outlying_observations(points)
    assert len(kept) == 11 and np.isfinite(kept).all()


def test_the_trim_refuses_a_malformed_array():
    with pytest.raises(TypeError, match=r"\(N, 3\)"):
        trim_outlying_observations(np.zeros((8, 2)))


def test_trimmed_points_reach_calibrate_laser_as_float32(monkeypatch):
    """The documented order, trim then calibrate_laser: the trim returns
    float64 and the native kernel takes float32, so the wrapper casts."""
    from fishsense_core import laser  # pylint: disable=import-outside-toplevel

    seen = {}

    class _Native:  # pylint: disable=too-few-public-methods
        @staticmethod
        def calibrate_laser(points):
            seen["dtype"], seen["shape"] = points.dtype, points.shape
            return np.zeros(3, np.float32), np.array([0, 0, 1], np.float32)

    monkeypatch.setattr(laser._native, "laser", _Native)  # pylint: disable=protected-access
    trimmed = trim_outlying_observations(_ver_points(np.linspace(0.6, 3.0, 12)))
    assert trimmed.dtype == np.float64
    laser.calibrate_laser(trimmed)
    assert seen == {"dtype": np.float32, "shape": (12, 3)}
    with pytest.raises(TypeError, match=r"\(N, 3\)"):
        laser.calibrate_laser(np.zeros((4, 2)))


@pytest.mark.parametrize(
    "bad", [np.zeros((10, 3)), np.zeros(20), np.zeros((10, 2, 1))], ids=["n-by-3", "flat", "3-d"]
)
def test_malformed_dots_are_a_type_error_not_a_refusal_or_an_abstention(bad):
    """(N, 3) homogeneous pixels used to make the self-consistency gate raise
    a matmul ValueError (read by a caller as a refusal) and the mission gate
    abstain (turning it off). Both now say it is the caller's bug."""
    with pytest.raises(TypeError, match=r"\(N, 2\)"):
        check_fit_self_consistency(VER_ORIGIN, VER_AXIS, VER_K, bad)
    with pytest.raises(TypeError, match=r"\(N, 2\)"):
        check_calibration_describes_mission(VER_ORIGIN, VER_AXIS, VER_K, bad)


def test_malformed_observations_are_a_type_error_not_a_refusal():
    with pytest.raises(TypeError, match=r"\(N, 3\)"):
        check_observation_geometry(np.zeros((10, 2)))


def test_a_non_finite_dot_is_dropped_before_judging():
    """A NaN dot used to make the span NaN (so no abstention) and then crash
    the SVD; it is dropped, and the gate judges the rest."""
    dots = _ver_dots(np.linspace(0.6, 3.0, 20))
    dots[3] = [np.nan, np.nan]
    assert check_fit_self_consistency(VER_ORIGIN, VER_AXIS, VER_K, dots) == PASSED
    assert check_calibration_describes_mission(VER_ORIGIN, VER_AXIS, VER_K, dots) == PASSED


@pytest.mark.parametrize(
    "axis",
    [np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.8, 1e-6]), np.array([np.nan, 0.0, 1.0])],
    ids=["in-z0-plane", "nearly", "nan"],
)
def test_an_axis_with_no_image_is_refused_not_crashed(axis):
    """An axis in or near the z=0 plane cannot cross the working depths; it
    used to divide by ~0 and crash the SVD."""
    dots = _ver_dots(np.linspace(0.6, 3.0, 20))
    with pytest.raises(CalibrationInconsistentError, match="no image"):
        check_fit_self_consistency(VER_ORIGIN, axis, VER_K, dots)
    with pytest.raises(CalibrationDoesNotDescribeDiveError, match="no image"):
        check_calibration_describes_mission(VER_ORIGIN, axis, VER_K, dots)
