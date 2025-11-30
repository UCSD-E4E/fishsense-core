""" "Module for rectified images using camera intrinsics to correct distortion."""

import cv2
import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics

from fishsense_core.image.image import Image


class RectifiedImage(Image):
    """Represents a rectified image using camera intrinsics to correct distortion."""

    # pylint: disable=no-member

    def __init__(self, image: Image, intrinsics: CameraIntrinsics):
        self.__image = image
        self.__intrinsics = intrinsics

        super().__init__()

    def _get_data(self) -> np.ndarray:
        return cv2.undistort(
            self.__image.data,
            self.__intrinsics.camera_matrix,
            self.__intrinsics.distortion_coefficients,
        )
