"""3D world-point projection from image coordinates.

Wraps the native ``WorldPointHandler`` Rust class. The handler holds a single
camera's inverted intrinsics matrix (K⁻¹) and exposes three projection methods:
``project_image_point`` (raw K⁻¹ ray), ``compute_world_point_from_depth`` (ray
scaled by a known depth), and ``compute_world_point_from_laser`` (camera-ray
× known laser-line triangulation).
"""

from fishsense_core import _native

WorldPointHandler = _native.world_point.WorldPointHandler  # pylint: disable=c-extension-no-member

__all__ = ["WorldPointHandler"]
