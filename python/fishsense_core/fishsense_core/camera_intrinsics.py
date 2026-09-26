"""Camera intrinsics: the two arrays that rectification reads."""

from dataclasses import dataclass

import numpy as np

# The distortion models OpenCV reads that a FishSense calibration produces:
# (k1, k2, p1, p2), + k3, + k4..k6 (rational). 12 and 14 (thin-prism, tilted)
# are valid to OpenCV but no camera in the fleet is calibrated with them.
_DISTORTION_LENGTHS = (4, 5, 8)


@dataclass(frozen=True, eq=False)
class CameraIntrinsics:
    """A camera matrix and its distortion coefficients, validated.

    This is the core's own intrinsics type, so nothing in fishsense_core
    needs the API SDK. It is *accepted, not required*: :class:`RectifiedImage`
    and :class:`LaserDetector` read only ``camera_matrix`` and
    ``distortion_coefficients``, so any object with those two attributes —
    including ``fishsense_api_sdk``'s ``CameraIntrinsics``, which v1's data
    worker passes — keeps working unchanged.

    Both fields are stored as read-only ``float64`` copies. The camera matrix
    must be 3x3; the distortion must hold 4, 5 or 8 coefficients, given flat
    or as a ``(1, N)`` / ``(N, 1)`` vector (``cv2.calibrateCamera`` returns
    ``(1, 5)``). Every value must be finite.

    ``eq=False`` because array fields have no single truth value; compare the
    arrays with ``np.array_equal`` instead.
    """

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    def __post_init__(self):
        camera_matrix = np.array(self.camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix must be 3x3, got shape {camera_matrix.shape}"
            )

        distortion = np.array(self.distortion_coefficients, dtype=np.float64)
        if distortion.ndim == 2 and 1 in distortion.shape:
            distortion = distortion.reshape(-1)
        if distortion.ndim != 1 or distortion.size not in _DISTORTION_LENGTHS:
            raise ValueError(
                "distortion_coefficients must hold 4, 5 or 8 values, got shape "
                f"{distortion.shape}"
            )

        for name, array in (
            ("camera_matrix", camera_matrix),
            ("distortion_coefficients", distortion),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite, got {array.tolist()}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
