"""Two-line (specular reflection) detection for laser-dot populations.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/src/fishsense_services_processor/
laser_validation/reflection.py`` (itself verbatim from fishsense-lite@77e8f8e5
services/fishsense-data-processing-workflow-worker/src/
fishsense_data_processing_workflow_worker/laser_label_validation/
reflection.py). Only the imports changed: the line fit comes from
:mod:`fishsense_core.line_fit` directly, which is what ``fishsense_core.laser``
re-exports.

Prod dive 77 (2026-08-04): pool slate photos show two bright dots — the laser
and its specular reflection ~45 px away. Labelers clicked the reflection on 34
of 79 frames, so the dive's dots formed two coherent PARALLEL lines. That
population defeats the single-line RANSAC validator in exactly the wrong way:
the split kills its confidence / trips MAX_OUTLIER_FRACTION, so it silently
skips the dive forever — while stage 13 happily consumes the poisoned mix and
persists a calibration that is 14 deg off both lines.

This module names the failure mode: after the primary fit, look for a second
coherent line among the off-line points. Near-parallel + well-separated =
reflection signature. Detection only — choosing WHICH line is the real laser
needs cross-dive consensus (sibling dives on the same camera), which is
operator-level context the per-dive validator doesn't have. What to do about a
suspect (log it, hold the dive, supersede one line's labels) is pipeline
policy and stays with the caller.

It sits beside :mod:`fishsense_core.line_fit` because it is that fit's
companion: it reads a :class:`LineFit` and re-runs :func:`fit_dive_line` on the
points the primary line rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fishsense_core.line_fit import (
    LineFit,
    MIN_POINTS_FOR_LINE,
    RANSAC_INLIER_TOL_PX,
    fit_dive_line,
)

__all__ = [
    "MAX_PARALLEL_ANGLE_DEG",
    "MIN_SEPARATION_PX",
    "MAX_SEPARATION_PX",
    "ReflectionSuspect",
    "detect_reflection_split",
]

# A reflection is (near-)parallel to the true dot line — both are projections
# of the same beam geometry. Steeper crossings are some other pathology.
MAX_PARALLEL_ANGLE_DEG = 3.0

# Separation window: closer than the RANSAC tolerance would just be label
# noise; a reflection hundreds of px away stops looking like one (and stops
# being clickable-by-mistake).
MIN_SEPARATION_PX = 8.0
MAX_SEPARATION_PX = 200.0

# The secondary population must itself be line-like, not scatter.
MIN_SECONDARY_INLIER_FRACTION = 0.6


@dataclass
class ReflectionSuspect:
    """A coherent second dot line parallel to the primary."""

    n_primary: int
    n_secondary: int
    separation_px: float
    angle_deg: float


def detect_reflection_split(
    xy: np.ndarray,
    primary: LineFit | None,
    *,
    tol_px: float = RANSAC_INLIER_TOL_PX,
) -> ReflectionSuspect | None:
    # pylint: disable=too-many-return-statements
    # A flat chain of disqualifying guards, each returning None for its own
    # reason; collapsing them would obscure which criterion rejected.
    """Return a `ReflectionSuspect` when the off-line points form a second
    coherent line parallel to `primary`, else None."""
    if primary is None or xy.shape[0] < 2 * MIN_POINTS_FOR_LINE:
        return None

    residual = primary.perpendicular_distance(xy[:, 0], xy[:, 1])
    off = residual >= tol_px
    if int(off.sum()) < MIN_POINTS_FOR_LINE:
        return None

    secondary = fit_dive_line(xy[off])
    if secondary is None:
        return None
    if secondary.inlier_count < MIN_POINTS_FOR_LINE:
        return None
    if secondary.inlier_fraction < MIN_SECONDARY_INLIER_FRACTION:
        return None

    # Line coefficients are unit-normalized (a^2 + b^2 = 1), so the normals
    # compare directly and |c| differences are pixel distances.
    n1 = np.array([primary.a, primary.b])
    n2 = np.array([secondary.a, secondary.b])
    cos_angle = min(1.0, abs(float(n1 @ n2)))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    if angle_deg > MAX_PARALLEL_ANGLE_DEG:
        return None

    # Separation: median distance of the secondary population from the
    # primary line — robust even when the lines aren't perfectly parallel.
    separation_px = float(np.median(residual[off]))
    if not MIN_SEPARATION_PX <= separation_px <= MAX_SEPARATION_PX:
        return None

    return ReflectionSuspect(
        n_primary=int((~off).sum()),
        n_secondary=int(off.sum()),
        separation_px=separation_px,
        angle_deg=angle_deg,
    )
