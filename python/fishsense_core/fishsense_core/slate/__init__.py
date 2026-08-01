"""Dive-slate detection — recovers the board plane feeding laser calibration.

The classical estimator (:func:`estimate_plane`) is the CPU pipeline: template
search → ECC → ``solvePnP`` → board plane. It is pure ``cv2`` + ``numpy`` (base
install). The optional learned localization mask enters through
``estimate_plane(board_mask=...)`` and only adds search candidates, so a
missing or wrong mask costs search time, never coverage.

See :mod:`fishsense_core.plane` for the plane the estimate ultimately produces,
and the bridge (``laser_point_on_plane``) into
:func:`fishsense_core.laser.calibrate_laser`.
"""

from fishsense_core.slate.estimator import (
    BoardEstimate,
    estimate_plane,
    homography_from_quad,
    order_quad,
    rotate_quad,
    segment_board,
    template_corners,
)

__all__ = [
    "BoardEstimate",
    "estimate_plane",
    "homography_from_quad",
    "order_quad",
    "rotate_quad",
    "segment_board",
    "template_corners",
]
