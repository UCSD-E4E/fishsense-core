"""Abstract base classes for FishSense Core."""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

_log = logging.getLogger(__name__)


class Image(ABC):
    """Abstract base class for images."""

    # pylint: disable=no-member

    @property
    def width(self) -> int:
        """Returns the width of the image."""
        self.__load_image()

        return self.__width

    @property
    def height(self) -> int:
        """Returns the height of the image."""
        self.__load_image()

        return self.__height

    @property
    def data(self) -> np.ndarray:
        """Returns the image data as a NumPy array."""
        self.__load_image()

        return self.__data

    def __init__(self) -> None:
        self.__width: int | None = None
        self.__height: int | None = None
        self.__data: np.ndarray | None = None

    def __load_image(self) -> None:
        if self.__data is None:
            _log.debug("%s: loading image data", type(self).__name__)
            self.__data = self._get_data()
            self.__height, self.__width = self.__data.shape[:2]
            _log.debug(
                "%s: loaded image %dx%d", type(self).__name__, self.__width, self.__height
            )

    @abstractmethod
    def _get_data(self) -> np.ndarray:
        """Loads the image data."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Saves the image to the specified file path."""
        _log.debug("%s: saving image to %s", type(self).__name__, path)
        cv2.imwrite(path, self.data)
