"""FishSense Core Package"""

import math
from pathlib import Path

import cv2
import numpy as np
import rawpy
from skimage.exposure import adjust_gamma, equalize_adapthist # pylint: disable=no-name-in-module
from skimage.util import img_as_float, img_as_ubyte

from fishsense_core.image.image import Image


class RawImage(Image):
    """Represents a raw image loaded from a file path."""

    # pylint: disable=no-member

    def __init__(self, path: Path):
        self.__path = path

        super().__init__()

    def _get_data(self) -> np.ndarray:
        """Loads the raw image from the file path and processes it."""
        with self.__path.open("rb") as f:
            with rawpy.imread(f) as raw:
                img = img_as_float(
                    raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=True,
                        output_bps=16,
                        user_flip=0,
                    )
                )

                hsv = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2HSV)
                _, _, val = cv2.split(hsv)

                mid = 20
                mean = np.mean(val)
                mean_log = math.log(mean)
                mid_log = math.log(mid * 255)
                gamma = mid_log / mean_log
                gamma = 1 / gamma

                img = adjust_gamma(img, gamma=gamma)
                img = equalize_adapthist(img)

                img = img_as_ubyte(img[:, :, ::-1])

        return img
