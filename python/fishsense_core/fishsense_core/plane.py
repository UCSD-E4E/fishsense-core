"""The board plane — the quantity laser calibration actually consumes.

The dive-slate detector estimates a board pose, but laser calibration only
uses two things from it:

    slate_normal = rotation[:, 2]
    scale = (slate_normal @ camera_space_point) / (slate_normal @ ray)

i.e. the board's **normal** and its **offset along that normal**. Nothing else
survives. Verified numerically on real labels: rotating the board in its own
plane by 17 / 90 / 180 degrees, or sliding it up to 50 cm within the plane,
changes the recovered laser point by exactly 0 mm; only motion along the normal
does anything (1 cm -> 10.2 mm).

Three practical consequences:

* The estimator's target is a plane ``n . X = d`` in camera space — **3 DOF**,
  not a 6-DOF pose and certainly not a set of labeled points.
* A 180-degree in-plane flip yields an identical plane, so slate orientation
  ("upside down") provably cannot affect the calibration.
* Any correspondence error that amounts to an in-plane symmetry is harmless,
  which makes the estimate far more robust than point-matching would suggest.

Scale still matters — ``d`` is metric — so the template's physical geometry
(reference points / dpi -> inches -> metres) remains essential; a homography
alone recovers the plane only up to scale.

This is the bridge primitive between the slate detector's board pose and
:func:`fishsense_core.laser.calibrate_laser`: intersect each laser ray with the
board plane (:func:`laser_point_on_plane`) to get the 3-D points calibration
fits. It is pure NumPy — no native extension, no optional dependency.

Ported from ``slate_training.geometry`` (``UCSD-E4E/2026-07-31_slate_training``),
which now imports these definitions from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["Plane", "plane_from_pose", "plane_difference", "laser_point_on_plane"]


@dataclass(frozen=True)
class Plane:
    """A plane in camera space: ``normal . X = distance``, ``normal`` unit-length.

    ``distance`` is signed and carries the same sign convention as ``normal``;
    negating both describes the identical plane, which :func:`plane_difference`
    accounts for.
    """

    normal: np.ndarray
    distance: float


def plane_from_pose(rotation: np.ndarray, translation: np.ndarray) -> Plane:
    """Extract the board plane from a ``solvePnP`` pose.

    The board is the ``z = 0`` plane in body coordinates, so its camera-space
    normal is the third column of ``rotation`` and ``translation`` is a point
    on it.
    """
    normal = np.asarray(rotation, dtype=float)[:, 2]
    normal = normal / np.linalg.norm(normal)
    point = np.asarray(translation, dtype=float).reshape(3)
    return Plane(normal, float(normal @ point))


def plane_difference(a: Plane, b: Plane) -> Tuple[float, float]:
    """Compare two planes as ``(normal angle in degrees, offset in mm)``.

    This is the parameterization-independent accuracy metric: it is exactly the
    part of a predicted labeling that reaches the calibration, and it is blind
    to the in-plane freedoms the calibration ignores.

    Normal sign is normalized first, since ``(n, d)`` and ``(-n, -d)`` are one
    plane.
    """
    n_a, d_a = a.normal, a.distance
    n_b, d_b = b.normal, b.distance
    if n_a @ n_b < 0:
        n_b, d_b = -n_b, -d_b

    cos = float(np.clip(n_a @ n_b, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos)))
    return angle, abs(float(d_a - d_b)) * 1000.0


def laser_point_on_plane(
    plane: Plane, laser_xy: Tuple[float, float], camera_matrix: np.ndarray
) -> "np.ndarray | None":
    """Intersect the laser's back-projected ray with the board plane.

    Returns the 3-D hit in camera space, or ``None`` when the ray is parallel
    to the plane (or the projection degenerates) — the caller should drop that
    observation rather than emit a NaN.
    """
    k_inv = np.linalg.inv(np.asarray(camera_matrix, dtype=float))
    ray = k_inv @ np.array([laser_xy[0], laser_xy[1], 1.0], dtype=float)
    if np.any(np.isnan(ray)):
        return None

    denominator = float(plane.normal @ ray)
    if abs(denominator) < 1e-12:
        return None
    return ray * (plane.distance / denominator)
