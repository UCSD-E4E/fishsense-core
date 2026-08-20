"""3D world-point projection from image coordinates.

Wraps the native ``WorldPointHandler`` Rust class. The handler holds a single
camera's inverted intrinsics matrix (K⁻¹) and exposes three projection methods:
``project_image_point`` (raw K⁻¹ ray), ``compute_world_point_from_depth`` (ray
scaled by a known depth), and ``compute_world_point_from_laser`` (camera-ray
× known laser-line triangulation).

**Input dtype:** all numpy-array arguments are coerced to ``float64`` before
crossing the PyO3 boundary. Pass float32, float64, integer (e.g. raw pixel
coordinates from the API SDK), or anything else ``np.asarray`` accepts — the
wrapper handles the conversion. The native binding accepts ``float64`` only;
without this coercion, callers passing integer-dtype arrays hit a TypeError
from PyO3-numpy at the binding boundary.

**Laser axis:** ``laser_axis`` is a *direction*. Its magnitude does not affect
the result — the native solve normalises it — so a raw ``target - origin``
difference works as well as the unit vector
:func:`fishsense_core.laser.calibrate_laser` returns. A zero-length axis has no
direction and is rejected with ``ValueError`` rather than answered with a
plausible-looking point.
"""

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
        """
        return self._native.compute_world_point_from_laser(
            _f64(laser_origin), _direction(laser_axis), _f64(image_point)
        )


__all__ = ["WorldPointHandler"]
