"""3D world-point projection from image coordinates.

Wraps the native ``WorldPointHandler`` Rust class. The handler holds a single
camera's inverted intrinsics matrix (K⁻¹) and exposes four projection methods:
``project_image_point`` (raw K⁻¹ ray), ``compute_world_point_from_depth`` (ray
scaled by a known depth), ``compute_world_point_from_laser`` (camera-ray
× known laser-line triangulation), and
``compute_world_point_from_laser_with_residual`` (the same triangulation plus
the closest-approach distance the solve computes on the way).

**Input dtype:** all numpy-array arguments are coerced to ``float64`` before
crossing the PyO3 boundary. Pass float32, float64, integer (e.g. raw pixel
coordinates from the API SDK), or anything else ``np.asarray`` accepts — the
wrapper handles the conversion. The native binding accepts ``float64`` only;
without this coercion, callers passing integer-dtype arrays hit a TypeError
from PyO3-numpy at the binding boundary.

**Input shape:** ``camera_intrinsics_inverted`` must be 3×3, ``image_point``
``[x, y]``, and ``laser_origin`` / ``laser_axis`` 3-vectors. Anything else is a
``ValueError`` from the native binding — in particular a homogeneous
``[x, y, w]`` pixel, whose ``w`` used to be dropped without complaint.

**Laser axis:** ``laser_axis`` is a *direction*. Its magnitude does not affect
the result — the native solve normalises it — so a raw ``target - origin``
difference works as well as the unit vector
:func:`fishsense_core.laser.calibrate_laser` returns. A zero-length axis has no
direction and is rejected with ``ValueError`` rather than answered with a
plausible-looking point.

**The laser point and length-at-depth.** Two module-level functions compose the
handler into the two numbers the measurement pipeline actually stores:
:func:`compute_laser_point` turns a laser dot into a :class:`LaserPoint` (depth,
range, residual) and :func:`measure_length_at_depth` measures head-to-tail across
a fronto-parallel plane at that depth. Both take the 3x3 camera matrix ``K``
itself, not K⁻¹, and plain arrays / tuples rather than label objects. Ported from
fishsense-services v0.48.0 ``laser_geometry.py`` (itself from
fishsense-lite@77e8f8e5); see :func:`compute_laser_point` for the rules the
services module documented, all of which still hold.
"""

from dataclasses import dataclass

import numpy as np

from fishsense_core import _native

_NativeWorldPointHandler = _native.world_point.WorldPointHandler  # pylint: disable=c-extension-no-member


def _f64(arr) -> np.ndarray:
    """Coerce ``arr`` to a float64 numpy array. No-op when already float64."""
    return np.asarray(arr, dtype=np.float64)


def _direction(arr, name: str = "laser_axis") -> np.ndarray:
    """Coerce ``arr`` to float64 and reject a degenerate direction vector.

    The magnitude is left alone — the native solve normalises it — but a zero
    or non-finite axis points nowhere, and triangulating against it would
    return an ordinary-looking wrong point.
    """
    vec = _f64(arr)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(
            f"{name} must be a finite, non-zero direction vector; got {vec!r}"
        )
    return vec


class WorldPointHandler:
    """Python wrapper around ``_native.world_point.WorldPointHandler``.

    All array inputs are coerced to float64 before the native call. See module
    docstring for rationale.
    """

    def __init__(self, camera_intrinsics_inverted):
        self._native = _NativeWorldPointHandler(_f64(camera_intrinsics_inverted))

    def project_image_point(self, image_point) -> np.ndarray:
        """Project an image-space ``[x, y]`` into camera-space via K⁻¹·[x, y, 1]."""
        return self._native.project_image_point(_f64(image_point))

    def compute_world_point_from_depth(self, image_point, depth: float) -> np.ndarray:
        """K⁻¹·[x, y, 1] · depth — the camera-space ray scaled by a known depth."""
        return self._native.compute_world_point_from_depth(_f64(image_point), float(depth))

    def compute_world_point_from_laser(
        self, laser_origin, laser_axis, image_point
    ) -> np.ndarray:
        """Triangulate camera ray vs. known laser line (least-squares closest point).

        ``laser_axis`` is a direction of any non-zero length; it is normalised
        natively, so the answer does not depend on its magnitude. A zero-length
        or non-finite axis raises ``ValueError``.

        The returned point always exists, including for a pixel no laser dot
        could have produced — use
        :meth:`compute_world_point_from_laser_with_residual` when the caller
        needs to tell those apart.
        """
        return self._native.compute_world_point_from_laser(
            _f64(laser_origin), _direction(laser_axis), _f64(image_point)
        )

    def compute_world_point_from_laser_with_residual(
        self, laser_origin, laser_axis, image_point
    ) -> tuple[np.ndarray, float]:
        """``(point, residual)`` — triangulation plus its closest-approach distance.

        ``residual`` is the distance, in the units of ``laser_origin``, between
        the camera ray and the laser line where they pass closest. It is ~0 when
        the two genuinely intersect, which makes it a direct per-dot check of a
        laser label against a calibration.

        **It only sees error across the laser's epipolar line** — the image line
        the laser sweeps out. A dot mislabelled *along* that line slides the
        answer up and down the laser with the residual pinned at zero: in a
        1.5 m scene, 100 px of along-epipolar error keeps the residual below
        1e-7 m while the depth falls to 0.87 m. Two further caveats: two lines
        through the camera centre intersect there exactly, so residual 0 can
        still mean a point at zero or negative depth; and the residual is
        metric, so a fixed threshold is stricter up close than far away. The
        solve runs in float32, so residuals below ~1e-5 m are noise for a
        metre-scale scene — don't threshold tighter than that.

        The worst case is an all-zero ``laser_origin`` — a missing or zeroed
        extrinsics row. With no baseline there is nothing to triangulate, so
        *every* dot in the dataset comes back at the camera centre reporting a
        perfect residual of 0; only the depth check catches it.

        Treat it as a necessary, not sufficient, check — pair it with a
        positive-depth and plausible-range test. Degenerate geometry (laser axis
        parallel to the camera ray) yields a non-finite point and residual.
        """
        point, residual = self._native.compute_world_point_from_laser_with_residual(
            _f64(laser_origin), _direction(laser_axis), _f64(image_point)
        )
        return point, float(residual)


@dataclass(frozen=True)
class LaserPoint:
    """Where the laser dot was, in metres, in camera coordinates.

    ``depth_m`` is the Z component and ``range_m`` the Euclidean norm. They
    differ by the dot's off-axis angle and are equal only on the optical axis,
    so they are carried separately rather than left for a caller to conflate:
    :func:`measure_length_at_depth` needs the depth specifically (it measures
    across a fronto-parallel plane), while "how far away was it" means the
    range.

    ``residual_m`` is how close the camera ray and the laser ray actually came
    to meeting -- ~0 when the dot is genuinely consistent with the calibration,
    which makes it a direct per-dot check of a laser label against its
    extrinsics. Necessary, not sufficient, and the ways it falls short are
    specific: it is blind to error *along* the laser's epipolar line (a dot
    slid along it moves the depth a long way with the residual pinned at the
    noise floor), two rays meeting at the camera centre report 0 at zero or
    negative depth, it is metric so one threshold is stricter close up than
    far away, and the float32 solve puts its noise floor near 1e-5 m at metre
    scale. Pair it with the depth gate; never replace the depth gate with it.
    """

    depth_m: float
    range_m: float
    residual_m: float


def _handler_for(camera_matrix) -> WorldPointHandler:
    """A handler for the camera matrix ``K`` (inverted here, in float64)."""
    return WorldPointHandler(np.linalg.inv(_f64(camera_matrix)))


def compute_laser_point(dot_xy, laser_position, laser_axis, camera_matrix) -> LaserPoint:
    """Triangulate the laser dot from its pixel and the rig's calibration.

    ``dot_xy`` is the dot's ``[x, y]`` pixel; ``laser_position`` and
    ``laser_axis`` are the laser's origin and direction in camera space (as
    :func:`fishsense_core.laser.calibrate_laser` returns them); ``camera_matrix``
    is the 3x3 ``K``. The dot's pixel fixes a ray out of the camera; the
    calibration fixes the laser's ray in space; the dot is the least-squares
    closest point between the two, and ``residual_m`` is how far apart they
    still were there.

    Ported from fishsense-services v0.48.0 ``laser_geometry.py`` (itself from
    fishsense-lite@77e8f8e5). The one change is the signature: services took a
    label (``.x``/``.y``) and a calibration (``.laser_position``/
    ``.laser_axis``) object; core takes the values, so the call is
    ``compute_laser_point((label.x, label.y), cal.laser_position,
    cal.laser_axis, K)``. The numbers are identical.

    **The triangulation has no failure mode, and that is the thing to
    understand about it.** It returns the closest point between the two rays,
    which always exists -- so for a pixel no real laser dot could have produced
    it answers just as confidently, with a point at or behind the camera. A
    laser ray parallel to the image plane through the camera centre gives the
    origin; a dot on the wrong side of the principal point for the laser's
    offset gives a negative Z. Both are the *correct* answer to the question
    asked; neither corresponds to an observation, and neither is non-finite, so
    an ``isfinite`` guard misses both. A length computed at a negative depth is
    numerically identical to one at its positive twin (the back-projection is
    linear in depth, so head and tail move together). **Callers must gate on
    ``depth_m > 0``.** The residual is not a substitute: see
    :class:`LaserPoint` for what it cannot see.

    The axis magnitude does not matter -- the native solve normalises it -- and
    a zero-length axis raises ``ValueError`` rather than coming back as a
    plausible-looking point. The axis belongs to a dive's calibration, so a
    directionless one makes every image in that dive unprocessable; that is
    the caller's blast radius to decide, so the raise is not caught here.
    """
    laser3d, residual = _handler_for(
        camera_matrix
    ).compute_world_point_from_laser_with_residual(
        laser_position, laser_axis, dot_xy
    )
    return LaserPoint(
        depth_m=float(laser3d[2]),
        range_m=float(np.linalg.norm(laser3d)),
        residual_m=float(residual),
    )


def measure_length_at_depth(head_xy, tail_xy, depth_m: float, camera_matrix) -> float:
    """Head-to-tail distance in metres, with both keypoints placed at ``depth_m``.

    ``head_xy`` / ``tail_xy`` are ``[x, y]`` pixels and ``camera_matrix`` the
    3x3 ``K``; ``depth_m`` is normally :attr:`LaserPoint.depth_m` (the Z
    component, not the range).

    Single-depth by construction: the laser gives one depth per frame, so the
    fish is measured as its *projection* onto a plane at that depth. An
    out-of-plane fish therefore reads short, never long -- the foreshortening
    the accuracy analysis attributes its one-sided negative tail to. The
    result is unchanged by flipping the sign of ``depth_m``, which is why the
    ``depth_m > 0`` gate belongs on the depth, not on the length.

    Ported from fishsense-services v0.48.0 ``laser_geometry.py`` (itself from
    fishsense-lite@77e8f8e5); services took a head/tail label object
    (``.head_x``/``.head_y``/``.tail_x``/``.tail_y``), core takes the two
    pixels.
    """
    handler = _handler_for(camera_matrix)
    head3d = handler.compute_world_point_from_depth(head_xy, depth_m)
    tail3d = handler.compute_world_point_from_depth(tail_xy, depth_m)
    return float(np.linalg.norm(head3d - tail3d))


__all__ = [
    "LaserPoint",
    "WorldPointHandler",
    "compute_laser_point",
    "measure_length_at_depth",
]
