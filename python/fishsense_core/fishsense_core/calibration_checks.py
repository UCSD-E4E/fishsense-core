"""The guards that make `calibrate_laser` safe to persist.

Ported from fishsense-services v0.48.0
`services/fishsense-services-processor/src/fishsense_services_processor/calibration/robust_fit.py`
(the outlier trim) and
`services/fishsense-services-processor/src/fishsense_services_processor/calibration/consistency.py`
(the gates), with the bounds from
`services/fishsense-services-contracts/src/fishsense_services_contracts/calibration_bounds.py`.
Those were in turn ported from fishsense-lite@77e8f8e5
(`robust_laser_fit.py`, `calibration_consistency.py` and
`libs/fishsense-shared/.../calibration_bounds.py`).

**Why core needs these.** `fishsense_core.laser.calibrate_laser` has no
outlier rejection and no judgement of its answer. Its direction is the
normalised sum of every pairwise difference between observations (each
flipped to point away from the camera), and its origin is where that
direction through their centroid crosses z = 0. That estimator weights
far-apart pairs most, so one badly-placed observation far down the range
drags the whole line, and nothing in the kernel says when the line is wrong.
Prod dive 77 is the worked example: labelers clicked
the laser's specular reflection on about half its pool slate frames, so the
dots formed two parallel lines ~45 px apart; the 3-D fit split the difference
-- 14 deg off the dive's own dot line, origin far outside the fleet family --
and the persisted calibration, borrowed by a fish-model dive, produced
+31..+137% length errors. Dive 103 is the quieter version -- a 6.25 cm
baseline from 17 otherwise clean observations, which neither a higher
`MIN_LASER_POINTS` nor the baseline bound explains.

**The geometry amplifies it.** The fit reports where the ray crosses z=0,
typically a metre or more behind the observations, so a small angular error
from one bad point is levered into a large error in the *baseline* -- the one
quantity `check_fit_self_consistency` provably cannot see, because sliding the
offset leaves the ray's projection unchanged.

The pieces, in the order a producer should run them (fishsense-services'
`calibration/fit.py` is the reference orchestration):

1. refuse below your minimum observation count (services refuses below 2);
2. `trim_outlying_observations` (one pass, capped; drops non-finite rows);
3. `fishsense_core.laser.calibrate_laser` on the trimmed points (it takes
   them as they come back);
4. the four gates, **observation geometry first** because the two projection
   gates abstain on exactly the degenerate geometry that most needs refusing
   (prod dives 347 and 107): `check_observation_geometry`,
   `check_fit_self_consistency`, `check_baseline_plausible`,
   `check_calibration_describes_mission`.

A gate that does not refuse **returns** a verdict (`PASSED` or `ABSTAINED`), so
a caller can record which gates actually judged the fit; a refusal raises one
of the four `ValueError` subclasses, whose message names the measured
statistic and the bound. Malformed input -- an array of the wrong shape -- is
a caller's bug, not a refusal, and raises `TypeError`, deliberately not a
`ValueError`, so a handler that records refusals cannot record it as one.

**What to do about a refusal is the caller's.** Every gate takes a
``remedies`` string appended to its message. fishsense-services passes its
operator levers there (its ``REFUSAL_REMEDIES``: the refusal record,
priority 'none', ``calibration_target_id``), so its stored messages stay
byte-identical; another caller passes its own, or nothing.

Pure numpy -- no cv2, no `_native` -- so every check is unit-testable on
synthetic geometry with a known answer, without the Rust kernel in the loop.
"""

from __future__ import annotations

import numpy as np

from fishsense_core.line_fit import DEFAULT_OUTLIER_SIGMA, MAD_TO_SIGMA

__all__ = [
    "ABSTAINED",
    "DEFAULT_MAX_ANGLE_DEG",
    "DEFAULT_MAX_MEDIAN_OFFSET_PX",
    "DEFAULT_MAX_MISSION_MEDIAN_OFFSET_PX",
    "DEFAULT_MAX_MISSION_P90_OFFSET_PX",
    "MAD_FLOOR_M",
    "MAX_BASELINE_M",
    "MAX_TRIM_FRACTION",
    "MIN_BASELINE_M",
    "MIN_AXIS_FORWARD",
    "MIN_DISAGREEING_MISSION_DOTS",
    "MIN_DOT_SPAN_PX",
    "MIN_MISSION_DOTS",
    "MIN_OBSERVATION_LEVER_M",
    "MIN_POINTS_TO_TRIM",
    "PASSED",
    "CalibrationDoesNotDescribeDiveError",
    "CalibrationImplausibleError",
    "CalibrationInconsistentError",
    "CalibrationUnderdeterminedError",
    "check_baseline_plausible",
    "check_calibration_describes_mission",
    "check_fit_self_consistency",
    "check_observation_geometry",
    "trim_outlying_observations",
]

# --- bounds (from fishsense-services-contracts `calibration_bounds`) --------
#
# Services' contracts package owns these numbers because its API also needs
# them (it excludes an already-stored implausible calibration from counting as
# a calibration at all). They are copied here as defaults; every gate that
# reads one takes it as a keyword argument, so services can keep passing its
# contract values and a disagreement stays visible at the call site.

#: Bounds on the baseline, `norm(laser_position[:2])`, in metres.
#:
#: The baseline is the offset between the camera centre and the laser on one
#: rig. It is hardware, not a property of a dive: over all 35 stored
#: calibrations its interquartile range is **9.99-10.45 cm**, half a
#: centimetre across both producers, every camera and two years.
#:
#: **Placed midway between the populations, not hard against the healthy one.**
#: The floor moved from 7.8 to 9.7 cm on 2026-09-12. Two fits inside the old
#: bound, 8.90 cm (dive 502, borrowed by 503/504) and 9.51 cm (dive 498), had
#: been called healthy on the strength of a ~-1 % median length error. The
#: range trend of a rigid target showed why the median lied: -14 to -18 % at
#: 0.8 m rising to ~0 at 4 m -- the short baseline's flat scale error and a
#: compensating angle error cancel exactly where the median and p90 sit. Sound
#: calibrations (10.3-10.5 cm) are flat across range to within 1 %. So the
#: nearest bad fit below is now 9.51 and the smallest sound one 9.87 (dive 94);
#: 9.7 sits between them. Above, healthy 12.95 and bad 16.01 are unchanged.
#:
#: **Widen only against re-measured data**, and measure with the range trend,
#: not the median: a wrong baseline is the one error the rest of the pipeline
#: provably cannot see. It scales every depth, hence every length, while
#: reprojection residual and self-consistency stay clean -- and, as above, a
#: compensating angle error can hide it from a known-length median too.
MIN_BASELINE_M = 0.097
MAX_BASELINE_M = 0.145

#: What a gate returns when it does not refuse: it judged and the fit passed,
#: or it had too little to judge and said nothing. A refusal raises.
PASSED = "passed"
ABSTAINED = "abstained"

def _with(remedies: str) -> str:
    """The caller's remedy text, as a message suffix."""
    return f" {remedies.strip()}" if remedies and remedies.strip() else ""


def _dots(array, what: str) -> np.ndarray:
    """``array`` as finite (N, 2) pixels; malformed is a caller's bug."""
    dots = np.asarray(array, dtype=float)
    if dots.ndim != 2 or dots.shape[1] != 2:
        raise TypeError(f"{what} must be an (N, 2) array of pixels, got shape {dots.shape}")
    return dots[np.isfinite(dots).all(axis=1)]


def _tls_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through `points` (2-D or 3-D) ->
    (centroid, unit direction).

    Not `fishsense_core.line_fit`'s TLS, which is 2-D and returns ``a x + b y
    + c = 0``; this one works in 3-D too and returns a point and a direction.
    Deliberately not `calibrate_laser` either: the trimmer only needs the line
    the points lie on, as a proxy for the one the kernel will fit.
    """
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    return centroid, vt[0]


# --- outlier trim (services `robust_fit.py`) --------------------------------
#
# Trimming is deliberately conservative. It is a guard against a few bad
# observations, not a way to rescue a badly contaminated dive: past
# `MAX_TRIM_FRACTION` it declines to trim at all and lets the baseline gate
# refuse the result, because when most observations disagree there is no
# majority worth trusting and quietly fitting the larger half would produce a
# confident answer from data we have no reason to believe.
#
# It cannot separate the dive-77 shape (a contiguous subset displaced by a
# constant offset forms a second, parallel line, and the TLS fit splits the
# difference so no single residual stands out). That is
# `check_fit_self_consistency`'s job; the two are complementary.

#: Floor on the MAD, in metres. Without it a tight set collapses the scale to
#: near zero and ordinary sub-millimetre scatter reads as a 3-sigma outlier --
#: the same reason `LABEL_NOISE_MAD_FLOOR_PX` exists on the 2-D fit.
MAD_FLOOR_M = 0.003

#: Below this many observations, trimming is skipped entirely. Two points fit a
#: line exactly, so every residual is zero and any rule is reading noise; a few
#: more still leave no redundancy worth spending. Returning them untouched
#: keeps a thin-but-valid calibration from becoming no calibration.
MIN_POINTS_TO_TRIM = 6

#: Never drop more than this fraction. A set where most observations disagree
#: has no majority to trust.
MAX_TRIM_FRACTION = 0.4


def _residuals(points: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each point to the best-fit line."""
    centroid, direction = _tls_line(points)
    offsets = points - centroid
    along = (offsets @ direction)[:, None] * direction[None, :]
    return np.linalg.norm(offsets - along, axis=1)


def trim_outlying_observations(
    points,
    *,
    sigma: float = DEFAULT_OUTLIER_SIGMA,
    mad_floor_m: float = MAD_FLOOR_M,
    max_trim_fraction: float = MAX_TRIM_FRACTION,
) -> np.ndarray:
    """Return `points` (N x 3, metres) without the observations that disagree
    with the ray, ready for `calibrate_laser`.

    Non-finite rows (a failed pose or depth) are dropped first: they are not
    observations. ``sigma`` defaults to `fishsense_core.line_fit`'s
    ``DEFAULT_OUTLIER_SIGMA``, the same rule on the same shape of problem in
    2-D, tuned on real dives; the residuals are judged against a TLS line, a
    proxy for the kernel's pairwise-sum direction (see the module docstring).

    One pass, not iterated to convergence. A single robust pass removes the
    gross disagreements that actually move the baseline, while repeated
    re-fitting on a shrinking set walks toward whichever subset happens to be
    most collinear -- which on a nearly-straight ray is close to arbitrary.

    Never mutates the input.
    """
    observations = np.asarray(points, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 3:
        raise TypeError(
            f"points must be an (N, 3) array of camera-space points, got shape "
            f"{observations.shape}"
        )
    observations = observations[np.isfinite(observations).all(axis=1)]
    if len(observations) < MIN_POINTS_TO_TRIM:
        return observations.copy()

    residuals = _residuals(observations)
    median = float(np.median(residuals))
    # Scaled MAD: MAD_TO_SIGMA makes it a consistent estimator of the standard
    # deviation for Gaussian noise, so `sigma` means what it says.
    mad = MAD_TO_SIGMA * float(np.median(np.abs(residuals - median)))
    scale = max(mad, mad_floor_m)

    keep = residuals <= median + sigma * scale
    dropped = int(len(observations) - keep.sum())
    if dropped > int(len(observations) * max_trim_fraction):
        # No trustworthy majority -- keep everything and let the baseline gate
        # judge the result, rather than confidently fitting the larger half.
        return observations.copy()
    return observations[keep]


# --- baseline plausibility -------------------------------------------------
#
# `check_fit_self_consistency` below cannot see the baseline, and that is
# structural rather than a threshold being too loose. It compares the fitted
# ray's PROJECTION against the 2D dots. Many different 3D rays project to the
# same image line -- the whole family lying in the plane through the camera
# centre and that line -- and sliding the laser's offset along that family
# moves where the ray crosses z=0 while leaving the projection identical. So
# it constrains two of the ray's four degrees of freedom, and the baseline
# sits in the two it cannot. Same epipolar blindness already documented for
# reprojection residual and scale.
#
# The baseline is checkable anyway, because it is not a property of the dive:
# it is the physical offset between camera and laser on one rig. Measured
# 2026-09-11 over all 35 stored calibrations, its interquartile range is
# 9.99-10.45 cm -- half a centimetre, across both producers, every camera and
# two years of dives.
#
# Against that, 8 fits sat outside 8-13 cm: 2.35, 2.60, 4.81, 4.90, 6.25,
# 6.91, 16.01 and 22.22 cm. They back 12 dives and 663 of 3,104 measurements,
# and grade -75% to +45% against the known-length targets. **Two are
# slate-derived**, which is why this lives beside the shared gate and not in
# the checkerboard path.


class CalibrationImplausibleError(ValueError):
    """The fitted laser sits at a physically implausible offset from the camera."""


def check_baseline_plausible(
    laser_position,
    *,
    min_baseline_m: float = MIN_BASELINE_M,
    max_baseline_m: float = MAX_BASELINE_M,
    remedies: str = "",
) -> str:
    """Raise `CalibrationImplausibleError` when the fitted baseline is not a rig.

    `laser_position` is the fit's origin: where the laser ray crosses the
    camera's z=0 plane. Its z component is padding (both producers emit the
    x/y pair and set z to 0), so only the in-plane offset is read -- taking a
    three-component norm would make the answer depend on a convention that
    carries no information.

    **Checked before persisting, not after.** A persisted calibration is
    immediately borrowable by sibling dives, so a bad one stops being one
    dive's problem the moment it is written.

    Returns `PASSED`.
    """
    position = np.asarray(laser_position, dtype=float)
    baseline = float(np.linalg.norm(position[:2]))

    # `isfinite` explicitly and first: every comparison against NaN is False,
    # so a bare range test would ACCEPT a NaN baseline. The safe reading of
    # "no measurable offset" is refusal.
    if not np.isfinite(baseline) or not min_baseline_m <= baseline <= max_baseline_m:
        raise CalibrationImplausibleError(
            f"fitted laser baseline {baseline * 100:.2f} cm is outside the "
            f"plausible range {min_baseline_m * 100:.1f}-"
            f"{max_baseline_m * 100:.1f} cm. The baseline is a property of the "
            f"camera+laser rig, not of the dive, and the fleet sits at "
            f"~10.4 cm; a value this far off means the fit, not the hardware. "
            f"It would scale every depth and every length by that factor, "
            f"invisibly to reprojection residual and to the self-consistency "
            f"gate. Refusing to persist.{_with(remedies)}"
        )
    return PASSED


# --- fit self-consistency ---------------------------------------------------
#
# The invariant is cheap and assumption-free: the fitted 3D ray, projected
# through the camera matrix, must coincide with the total-least-squares line
# through the input dots. Fleet baseline (every good calibration, all
# cameras): <=1.6 deg / <=4 px median; the dive-77 failure was 14 deg / 17 px
# median. Reprojection error also *predicts* measurement bias (0.9 px borrow
# -> +-3% lengths; 2-4 px -> -4..-15%), so the thresholds below bound
# downstream accuracy, not just fit hygiene.
#
# Fleet-derived thresholds: good fits sit at <=1.6 deg / <=4 px median, the
# known-bad fit at 14 deg / 17 px. The gap is wide; these sit in the middle
# with margin for noisier-but-sane dives.
DEFAULT_MAX_ANGLE_DEG = 3.0
DEFAULT_MAX_MEDIAN_OFFSET_PX = 8.0

# Below this dot spread the 2D line direction is numerically meaningless
# (e.g. MIN_LASER_POINTS=2 nearly-coincident observations), so the gate
# abstains rather than rejecting on noise.
MIN_DOT_SPAN_PX = 50.0


class CalibrationInconsistentError(ValueError):
    """The fitted laser ray does not reproject onto the dots it came from."""


#: A fitted axis whose forward (z) component is below this, after
#: normalising, is not a laser pointing into the scene: it cannot cross the
#: working depths, so its image is undefined. The projection gates refuse it.
MIN_AXIS_FORWARD = 1e-3


def _project_ray_to_image_line(
    laser_position: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """The image of the fitted laser ray -> (centroid, unit dir), or None
    when the ray has no image across the working depths: a non-finite fit,
    or an axis that does not point into the scene (z <= `MIN_AXIS_FORWARD`).

    Both projection gates need this same line: one compares it against the
    dots the fit came from, the other against every dot in the dive. Sampling
    the ray across the plausible working depth range and fitting is exact, not
    an approximation -- for a pinhole camera the image of a straight 3-D line
    is a straight line.
    """
    origin = np.asarray(laser_position, dtype=float)
    axis = np.asarray(laser_axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if not (np.all(np.isfinite(origin)) and np.isfinite(norm)) or norm == 0.0:
        return None
    axis = axis / norm
    if abs(axis[2]) < MIN_AXIS_FORWARD:
        return None
    depths = np.linspace(0.2, 4.0, 200)
    ts = (depths - origin[2]) / axis[2]
    ray_points = origin[None, :] + ts[:, None] * axis[None, :]
    homogeneous = (np.asarray(camera_matrix, dtype=float) @ ray_points.T).T
    image = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.all(np.isfinite(image)):
        return None
    return _tls_line(image)


def _no_image(remedies: str) -> str:
    return (
        "the fitted laser ray has no image across the 0.2-4 m working depths "
        "(non-finite, or its axis does not point into the scene), so it cannot "
        f"be the laser. Refusing to persist.{_with(remedies)}"
    )


# Seven arguments: the four inputs, then three keyword-only bounds a caller
# may override with its own contract values. Grouping them into an object
# would change the services call sites this was ported to serve. The locals
# are the statistics the refusal message reports; the body is a verbatim port.
def check_fit_self_consistency(  # pylint: disable=too-many-arguments,too-many-locals
    laser_position: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
    dots_xy: np.ndarray,
    *,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
    max_median_offset_px: float = DEFAULT_MAX_MEDIAN_OFFSET_PX,
    min_span_px: float = MIN_DOT_SPAN_PX,
    remedies: str = "",
) -> str:
    """Raise `CalibrationInconsistentError` when the fitted ray's projection
    disagrees with the 2D laser dots it was computed from.

    `laser_position` / `laser_axis` are the fit's origin (z padded to 0) and
    direction; `camera_matrix` is the 3x3 K. `dots_xy` is an (N, 2) array of
    the laser pixels whose observations fed the fit; non-finite rows are
    dropped. Abstains (returns `ABSTAINED`) when fewer than 2 remain or their
    spread is below `min_span_px` -- too degenerate to define a comparison
    line. Otherwise returns `PASSED`.

    Raises:
        CalibrationInconsistentError: The refusal; also when the fitted ray
            has no image (see `_project_ray_to_image_line`).
        TypeError: ``dots_xy`` is not an (N, 2) array.
    """
    dots = _dots(dots_xy, "dots_xy")
    if dots.shape[0] < 2:
        return ABSTAINED
    span = float(np.linalg.norm(dots.max(axis=0) - dots.min(axis=0)))
    if span < min_span_px:
        return ABSTAINED

    projected = _project_ray_to_image_line(laser_position, laser_axis, camera_matrix)
    if projected is None:
        raise CalibrationInconsistentError(_no_image(remedies))
    proj_centroid, proj_dir = projected
    _, dot_dir = _tls_line(dots)

    cos_angle = min(1.0, abs(float(proj_dir @ dot_dir)))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    normal = np.array([-proj_dir[1], proj_dir[0]])
    median_offset_px = float(np.median(np.abs((dots - proj_centroid) @ normal)))

    if angle_deg > max_angle_deg:
        raise CalibrationInconsistentError(
            f"fitted laser ray reprojects at angle {angle_deg:.2f} deg to the "
            f"input dot line (gate {max_angle_deg} deg); median offset "
            f"{median_offset_px:.1f}px. The fit disagrees with its own "
            f"observations — mixed dot populations (e.g. specular-reflection "
            f"mislabels) or corrupt slate poses. Refusing to persist."
            f"{_with(remedies)}"
        )
    if median_offset_px > max_median_offset_px:
        raise CalibrationInconsistentError(
            f"fitted laser ray reprojects with median offset "
            f"{median_offset_px:.1f}px from the input dot line "
            f"(gate {max_median_offset_px}px); angle {angle_deg:.2f} deg. "
            f"Refusing to persist.{_with(remedies)}"
        )
    return PASSED


# --- does the calibration describe the dive it will measure? ---------------
#
# The gate above compares the fitted ray against the dots it was computed
# from, so it is satisfied by construction, and it abstains when those dots
# are too few -- which is exactly the case it most needs to catch. Prod dive
# 347 got through both: the fit came from ONE frame carrying a duplicate laser
# label at the identical pixel, so `dots_xy` had N=1 and the check returned
# silently, while `MIN_LASER_POINTS = 2` was satisfied by counting the
# duplicate. The resulting calibration sits 9.4 px from the dive's own 321
# laser dots.
#
# This gate asks the other question, and it is not tautological: project the
# fitted ray and compare it against EVERY live dot in the dive, including the
# measurement frames the fit never saw. Those dots are where the lengths come
# from, so agreement there is the thing that actually matters, and
# disagreement means the laser was not in the same state for the frames being
# measured as for the frames it was calibrated from -- what a mid-dive re-seat
# does (prod dive 490: board burst and fish frames 0.82 deg apart).
#
# Measured over all 32 stored calibrations against their dives' live dots
# (2026-09-13), median perpendicular offset in px:
#
#     0.43-0.88   twenty dives
#     1.31-1.55   six dives
#     2.01, 2.27  dives 279, 509
#     3.33, 3.92  dives 341, 349   <- real burst-vs-frames disagreement
#     5.37        dive 498         <- also caught by the baseline gate
#     9.42        dive 347         <- the degenerate fit, caught by nothing else
#
# and p90 reaches 6.07 on the healthy side (dive 341) against 18.44 (347) and
# 29.69 (498). Both bounds sit between the populations rather than against the
# healthy one, the same rule the baseline gate uses and for the same reason: a
# refusal wedges the dive in its cohort, so a false positive is expensive.
#
# **Blind to the baseline, unavoidably.** Sliding the laser's offset along the
# family of rays sharing an image line leaves the projection identical, so five
# of the six known-bad baselines sit at 0.43-1.35 px here and pass. That is
# what `check_baseline_plausible` is for. Neither gate subsumes the other.
DEFAULT_MAX_MISSION_MEDIAN_OFFSET_PX = 6.0
DEFAULT_MAX_MISSION_P90_OFFSET_PX = 12.0
#: Below this many usable dots the dive cannot define a line, so the gate
#: abstains rather than refusing. A degenerate *fit* is not this gate's
#: abstention: it is caught by `check_observation_geometry`, which bounds the
#: lever arm of the observations rather than counting them -- the count is the
#: quantity that failed on dive 347, where `MIN_LASER_POINTS = 2` was
#: satisfied by a duplicate label at the identical pixel.
MIN_MISSION_DOTS = 6
#: The p90 branch needs a disagreeing SUBSET, not one or two stray labels, so
#: it also requires this many dots beyond the bound.
#:
#: The dots are the whole dive's, which means they include labels the fit
#: never saw and labels nobody has checked yet: the 3 sigma per-dive validator
#: that supersedes a reflection mislabel only runs once a dive's laser
#: labelling is *complete*, so mid-labelling the population is unpoliced. And
#: `np.percentile` interpolates, so at N=18 two 60 px mislabels already drag
#: the 90th percentile to ~18 px. Without a floor that refuses the dive, and a
#: recorded refusal self-expires the moment any label changes, so it would
#: come back every hour a labeler works.
#:
#: Re-measured 2026-09-13 over all 32 stored calibrations against their dives'
#: live dots: **every one of the 30 sound calibrations has zero dots beyond
#: the p90 bound**, against 13 of 40 (dive 498) and 139 of 321 (dive 347). So
#: the floor has only to clear a handful of strays and stay below 13.
MIN_DISAGREEING_MISSION_DOTS = 5


class CalibrationDoesNotDescribeDiveError(ValueError):
    """The fitted ray disagrees with the dive's own laser dots."""


# Seven arguments and the locals, for the same reasons as
# `check_fit_self_consistency`.
def check_calibration_describes_mission(  # pylint: disable=too-many-arguments,too-many-locals
    laser_position: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
    mission_dots_xy: np.ndarray,
    *,
    max_median_offset_px: float = DEFAULT_MAX_MISSION_MEDIAN_OFFSET_PX,
    max_p90_offset_px: float = DEFAULT_MAX_MISSION_P90_OFFSET_PX,
    min_disagreeing_dots: int = MIN_DISAGREEING_MISSION_DOTS,
    remedies: str = "",
) -> str:
    """Raise `CalibrationDoesNotDescribeDiveError` when the fitted ray's
    projection disagrees with the laser dots of the dive it will measure.

    `mission_dots_xy` is an (N, 2) array of every live laser pixel in the dive,
    not just the observations that fed the fit. Non-finite rows are dropped.
    Abstains below `MIN_MISSION_DOTS` usable dots or `MIN_DOT_SPAN_PX` of spread:
    a dive that cannot define a line cannot answer this. Those two are read
    from the module rather than taken as parameters -- they say when the
    question is unanswerable, which is not a per-caller policy.

    Two branches, and they fail differently. The median says the *whole* dive
    disagrees, which is a property of the fit. The p90 says a subset does,
    which can also be a handful of bad labels the fit never saw, so it fires
    only when at least `min_disagreeing_dots` sit beyond the bound.

    Raises:
        CalibrationDoesNotDescribeDiveError: The refusal; also when the fitted
            ray has no image (see `_project_ray_to_image_line`).
        TypeError: ``mission_dots_xy`` is not an (N, 2) array. It used to
            abstain, which turned the gate off for a caller bug.
    """
    dots = _dots(mission_dots_xy, "mission_dots_xy")
    if dots.shape[0] < MIN_MISSION_DOTS:
        return ABSTAINED
    span = float(np.linalg.norm(dots.max(axis=0) - dots.min(axis=0)))
    if span < MIN_DOT_SPAN_PX:
        return ABSTAINED

    projected = _project_ray_to_image_line(laser_position, laser_axis, camera_matrix)
    if projected is None:
        raise CalibrationDoesNotDescribeDiveError(_no_image(remedies))
    centroid, direction = projected
    normal = np.array([-direction[1], direction[0]])
    offsets = np.abs((dots - centroid) @ normal)
    median_px = float(np.median(offsets))
    p90_px = float(np.percentile(offsets, 90))
    n_disagreeing = int((offsets > max_p90_offset_px).sum())

    if median_px > max_median_offset_px:
        raise CalibrationDoesNotDescribeDiveError(
            f"calibration does not describe its dive: the fitted ray projects "
            f"with median offset {median_px:.1f}px from the dive's own "
            f"{dots.shape[0]} laser dots (gate {max_median_offset_px}px, p90 "
            f"{p90_px:.1f}px). The laser was not in the same state for the "
            f"frames being measured as for the frames it was calibrated from, "
            f"or the fit came from too few observations to constrain it. "
            f"Refusing to persist.{_with(remedies)}"
        )
    if p90_px > max_p90_offset_px and n_disagreeing >= min_disagreeing_dots:
        raise CalibrationDoesNotDescribeDiveError(
            f"calibration does not describe its dive: p90 offset {p90_px:.1f}px "
            f"from the dive's own {dots.shape[0]} laser dots (gate "
            f"{max_p90_offset_px}px, median {median_px:.1f}px), with "
            f"{n_disagreeing} of them beyond that bound (floor "
            f"{min_disagreeing_dots}). A real subset of the dive disagrees with "
            f"the fit -- either the laser moved partway through, or those "
            f"frames' labels are wrong. Refusing to persist.{_with(remedies)}"
        )
    return PASSED


# --- is the fit determined by its observations at all? ---------------------
#
# `MIN_LASER_POINTS` counts observations; it does not ask where they are. What
# determines the fitted ray's DIRECTION is the lever arm -- the spread of the
# observations along the ray -- against the label noise. One pixel of
# dot-label noise at range z is z/f metres of lateral error, so it rotates the
# fit by about (z/f)/lever radians, and the measured sensitivity of length to
# that angle is ~30 % per degree at 2 m. So two observations a metre apart
# determine the direction far better than sixteen at one range, and the count
# is the wrong quantity to bound.
#
# Measured over the 32 stored calibrations whose observations are recoverable
# (2026-09-13), lever arm and the resulting % of length per px of label noise:
#
#     dive 341   30 obs   1.55 m   0.1 %
#     dive 471    2 obs   1.56 m   0.7 %
#     dive 279    3 obs   1.47 m   0.6 %
#     dive 465    3 obs   1.34 m   0.6 %
#     dive 383    2 obs   1.09 m   0.7 %
#     dive 349    2 obs   0.26 m   5.8 %   <- refused
#     dive 062   20 obs   1.02 m   0.4 %   <- tightest sound calibration
#     dive 107   16 obs   0.03 m   5.5 %   <- refused
#     dive 526   15 obs   0.07 m           <- refused (its own fit collapses)
#     dive 347    1 obs   0.00 m   degenerate
#
# Dive 107 is why this gate is needed: sixteen observations, a dot span wide
# enough that `check_fit_self_consistency` does not abstain, and a 12.95 cm
# baseline that `check_baseline_plausible` accepts as the fleet's high
# extreme -- with 2.8 cm of lever (1.97-2.00 m). Dive 526 had the same
# single-distance geometry at 6.6 cm (4.16-4.22 m) and was caught only because
# its fit happened to collapse to 2.00 cm.
#
# Confirmed out of sample 2026-09-14, which is the strongest evidence for the
# bound: 107's fit reproduces its own 2.0 m working range to -0.12 % against
# the slate's solvePnP depth and 526's 4.2 m range to -17.25 %. Fitted jointly
# the two bursts give a 2.25 m lever and reproduce both to -0.22 % / -1.10 %
# (held-out medians -0.23 % / -1.19 %), so the two are one mount state and the
# single-burst fits are simply underdetermined.
#
# The bound sits between the populations -- 1.02 m is the tightest sound
# calibration and 0.26 m the loosest refused one -- nearer the bad side,
# because a refusal wedges the dive in its cohort.
MIN_OBSERVATION_LEVER_M = 0.60


class CalibrationUnderdeterminedError(ValueError):
    """The observations do not span enough range to determine the ray."""


def check_observation_geometry(
    observations_xyz: np.ndarray,
    *,
    min_lever_m: float = MIN_OBSERVATION_LEVER_M,
    remedies: str = "",
) -> str:
    """Raise `CalibrationUnderdeterminedError` when the 3-D observations are
    too tightly grouped along the ray to fix its direction.

    `observations_xyz` is the (N, 3) array of camera-space laser points that
    will be fitted. Non-finite rows are dropped before measuring. Unlike the
    other gates this one does not abstain on small N: one observation, or
    several at one distance, is exactly the case it exists to refuse.

    Returns `PASSED`.

    Raises:
        CalibrationUnderdeterminedError: The refusal.
        TypeError: ``observations_xyz`` is not an (N, 3) array.
    """
    points = np.asarray(observations_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise TypeError(
            "observations_xyz must be an (N, 3) array of camera-space points, got "
            f"shape {points.shape}"
        )
    points = points[np.isfinite(points).all(axis=1)]
    n = points.shape[0]
    lever = 0.0 if n < 2 else float(points[:, 2].max() - points[:, 2].min())
    if lever < min_lever_m:
        raise CalibrationUnderdeterminedError(
            f"laser observations span only {lever:.2f} m of range "
            f"({n} observation{'s' if n != 1 else ''}; gate {min_lever_m} m). "
            f"The lever arm, not the count, fixes the fitted direction: at this "
            f"spread one pixel of dot-label noise moves every length by several "
            f"percent. Shoot the target at two clearly different distances. "
            f"Refusing to persist.{_with(remedies)}"
        )
    return PASSED
