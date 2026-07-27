""" "Module for rectified images using camera intrinsics to correct distortion."""

import logging

import cv2
import numpy as np

try:
    from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
except ImportError as exc:  # pragma: no cover - depends on install extras
    raise ImportError(
        "RectifiedImage requires the FishSense API SDK, which is not a base "
        "dependency (it is git-only and would make the wheel un-pip-installable). "
        "Install the extra: pip install 'fishsense_core[rectified]'. Note the "
        "laser detector and its own output rectification do not need this."
    ) from exc

from fishsense_core.image.image import Image

_log = logging.getLogger(__name__)


class RectifiedImage(Image):
    """Represents a rectified image using camera intrinsics to correct distortion."""

    # pylint: disable=no-member

    def __init__(self, image: Image, intrinsics: CameraIntrinsics):
        self.__image = image
        self.__intrinsics = intrinsics

        super().__init__()

    def _get_data(self) -> np.ndarray:
        _log.debug("rectifying image with camera intrinsics")
        result = cv2.undistort(
            self.__image.data,
            self.__intrinsics.camera_matrix,
            self.__intrinsics.distortion_coefficients,
        )
        _log.debug("rectification complete: shape=%s", result.shape)
        return result
