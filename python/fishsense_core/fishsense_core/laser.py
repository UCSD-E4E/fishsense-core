"""Laser calibration and laser-dot detection."""
import logging

import numpy as np

from fishsense_core import _native
from fishsense_core._laser_detector import (
    DEFAULT_RIG_PRIOR_BBOX,
    LaserDetector,
    LaserPrediction,
)

__all__ = [
    "DEFAULT_RIG_PRIOR_BBOX",
    "LaserDetector",
    "LaserPrediction",
    "calibrate_laser",
]

_log = logging.getLogger(__name__)


def calibrate_laser(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the laser from a set of 3-D points.

    Delegates to the native Rust implementation.

    Args:
        points: (N, 3) float32 array of observed laser points.

    Returns:
        A ``(origin, orientation)`` tuple of 1-D float32 arrays.
    """
    _log.debug("calibrate_laser called with %d points", len(points))
    origin, orientation = _native.laser.calibrate_laser(points)  # pylint: disable=c-extension-no-member
    _log.debug(
        "calibrate_laser result: origin=%s orientation=%s",
        origin,
        orientation,
    )
    return origin, orientation
