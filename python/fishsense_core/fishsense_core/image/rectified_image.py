""" "Module for rectified images using camera intrinsics to correct distortion."""

import logging

import cv2
import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics

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
