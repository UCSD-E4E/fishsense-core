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
fits. It needs no native extension and no optional dependency. It is NumPy
at import; OpenCV is loaded only by :func:`pose_from_correspondences` (its
``solvePnP``), which recovers a target's pose from its 2-D <-> 3-D
correspondences. :func:`plane_from_correspondences` is that pose's plane; the
slate estimator calls the pose solver directly, since it also needs the pose
for its reprojection residual, so both producers share one solve.

Ported from ``slate_training.geometry`` (``UCSD-E4E/2026-07-31_slate_training``),
which now imports these definitions from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "Plane",
    "plane_from_pose",
    "plane_from_correspondences",
    "pose_from_correspondences",
    "plane_difference",
    "laser_point_on_plane",
]


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


#: solvePnP needs at least this many correspondences for a pose.
MIN_CORRESPONDENCES = 4

#: |z| above this in an (N, 3) body point means the target is not the z = 0
#: plane, which the solve assumes.
_PLANAR_Z_TOL_M = 1e-9


def pose_from_correspondences(
    body_points, image_points, camera_matrix
) -> "Tuple[np.ndarray, np.ndarray] | None":
    """``(rvec, tvec)`` of a planar target from its 2-D <-> 3-D pairs.

    ``body_points`` are the target's own coordinates in **metres**, ``(N, 2)``
    or ``(N, 3)`` with every z zero; ``image_points`` are where those landed in
    the **rectified** image, ``(N, 2)``, in the same order. Ordering is the
    caller's contract: ``solvePnP`` pairs them purely by position.
    ``camera_matrix`` is K, ``(3, 3)`` or flat ``(9,)``, any numeric type.

    ``solvePnP`` is given **zero distortion** on purpose: callers must detect
    their target on the rectified image. Passing raw (distorted) pixels yields
    a plausible, slightly wrong pose and no error.

    Returns ``None`` when this observation cannot give a pose, so a caller
    can skip it rather than fail a whole dive: fewer than
    :data:`MIN_CORRESPONDENCES` pairs (a partly detected board), a point that
    is not finite (a missed corner filled with NaN), ``solvePnP`` reporting
    failure or raising, or a non-finite pose. This is **not** a degeneracy
    check: coincident or collinear correspondences can still return a
    confident, meaningless pose, so an automatic producer must gate on its
    own detection quality first.

    Raises:
        ValueError: The inputs are malformed rather than unlucky, which is a
            caller's bug, not a frame to skip: body points that are not
            ``(N, 2)`` or ``(N, 3)``, image points that are not ``(N, 2)``,
            counts that differ, or a z column that is not zero (a target with
            real depth is not the plane this solves for).
    """
    body_in = np.asarray(body_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    if body_in.ndim != 2 or body_in.shape[1] not in (2, 3):
        raise ValueError(f"body_points must be (N, 2) or (N, 3); got shape {body_in.shape}")
    if image.ndim != 2 or image.shape[1] != 2:
        raise ValueError(f"image_points must be (N, 2); got shape {image.shape}")
    if len(body_in) != len(image):
        raise ValueError(
            f"{len(body_in)} body points but {len(image)} image points; they pair by position"
        )
    if body_in.shape[1] == 3:
        z = body_in[:, 2]
        if np.any(np.isfinite(z) & (np.abs(z) > _PLANAR_Z_TOL_M)):
            raise ValueError(
                "body_points' z column is not zero: the target must be the z = 0 plane "
                f"(largest |z| {float(np.nanmax(np.abs(z)))} m)"
            )
    if len(body_in) < MIN_CORRESPONDENCES:
        return None
    if not (np.all(np.isfinite(body_in)) and np.all(np.isfinite(image))):
        return None

    # pylint: disable-next=import-outside-toplevel
    import cv2

    # float32 body points, as fishsense-services and the slate estimator cast
    # them, so the recovered pose is the same.
    body = np.zeros((len(body_in), 3), dtype=np.float32)
    body[:, :2] = body_in[:, :2]
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    try:
        ret, rvec, tvec = cv2.solvePnP(  # pylint: disable=no-member
            body, image, k, np.zeros((5,))
        )
    except cv2.error:  # pylint: disable=catching-non-exception,no-member
        return None
    if not ret or not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None
    return rvec, tvec


def plane_from_correspondences(
    body_points, image_points, camera_matrix
) -> "Plane | None":
    """Recover the plane a planar target occupied, from 2-D <-> 3-D pairs.

    The plane of :func:`pose_from_correspondences`' pose, with its contract:
    same inputs, ``None`` for an observation that cannot give a pose,
    ``ValueError`` for malformed input. The target's shape never reaches the
    answer: a dive slate's six hand-clicked points and a checkerboard's grid
    of detected corners on the same physical plane yield the same
    :class:`Plane`.

    Ported from fishsense-services v0.48.0 ``calibration/geometry.py`` (itself
    from fishsense-lite@77e8f8e5), where malformed or short input made
    ``solvePnP`` raise and a NaN point could yield a NaN plane; here those
    are refused or skipped as above.
    """
    pose = pose_from_correspondences(body_points, image_points, camera_matrix)
    if pose is None:
        return None
    # pylint: disable-next=import-outside-toplevel
    import cv2

    rvec, tvec = pose
    rotation, _ = cv2.Rodrigues(rvec)  # pylint: disable=no-member
    # The body's +Z axis in camera space is the normal, and ``tvec`` (the body
    # origin) lies on the plane. The rotation about the normal is discarded,
    # which is why a checkerboard's 180-degree ambiguity is irrelevant here.
    return plane_from_pose(rotation, tvec)


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
