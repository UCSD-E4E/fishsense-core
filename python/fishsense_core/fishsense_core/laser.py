"""Laser calibration, laser-dot detection, and the per-dive 2-D line fit."""
import logging

import numpy as np

from fishsense_core import _native
from fishsense_core._laser_detector import (
    DEFAULT_RIG_PRIOR_BBOX,
    LaserDetector,
    LaserPrediction,
    rectify_prediction,
    unrectify_prediction,
)
from fishsense_core.line_fit import (
    COARSE_CALIBRATION_TOLERANCE_PX,
    DEFAULT_OUTLIER_SIGMA,
    LABEL_NOISE_MAD_FLOOR_PX,
    LINE_CONFIDENCE_THRESHOLD,
    MAD_TO_SIGMA,
    MIN_POINTS_FOR_LINE,
    RANSAC_INLIER_TOL_PX,
    RANSAC_MAX_ITERS,
    LineFit,
    fit_dive_line,
    flag_outliers,
)

__all__ = [
    "COARSE_CALIBRATION_TOLERANCE_PX",
    "DEFAULT_OUTLIER_SIGMA",
    "DEFAULT_RIG_PRIOR_BBOX",
    "LABEL_NOISE_MAD_FLOOR_PX",
    "LINE_CONFIDENCE_THRESHOLD",
    "LaserDetector",
    "LaserPrediction",
    "LineFit",
    "MAD_TO_SIGMA",
    "MIN_POINTS_FOR_LINE",
    "RANSAC_INLIER_TOL_PX",
    "RANSAC_MAX_ITERS",
    "calibrate_laser",
    "fit_dive_line",
    "flag_outliers",
    "rectify_prediction",
    "unrectify_prediction",
]

_log = logging.getLogger(__name__)


def calibrate_laser(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the laser from a set of 3-D points.

    Delegates to the native Rust implementation.

    Args:
        points: (N, 3) observed laser points, metres. Any real dtype: they
            are cast to the float32 the native kernel takes, so the float64
            :func:`fishsense_core.calibration_checks.trim_outlying_observations`
            returns can be passed straight in.

    Returns:
        A ``(origin, orientation)`` tuple of 1-D float32 arrays.

    Raises:
        TypeError: ``points`` is not an (N, 3) array.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise TypeError(f"points must be an (N, 3) array, got shape {points.shape}")
    _log.debug("calibrate_laser called with %d points", len(points))
    origin, orientation = _native.laser.calibrate_laser(points)  # pylint: disable=c-extension-no-member
    _log.debug(
        "calibrate_laser result: origin=%s orientation=%s",
        origin,
        orientation,
    )
    return origin, orientation
