"""The labeler-facing raw decode."""

import logging
from pathlib import Path

import numpy as np

from fishsense_core.image.decode import DecodeConfig, decode_rectified_stage
from fishsense_core.image.image import Image

_log = logging.getLogger(__name__)


class RawImage(Image):
    """A raw image decoded for viewing and for the fish models.

    ``data`` is uint8 BGR. The chain is
    :func:`~fishsense_core.image.decode.decode_rectified_stage`, and every step
    of it is a field on ``config``.

    The default ``config`` applies a **global CIELAB L\\* percentile stretch
    with CLAHE off**, which is a change from the local histogram equalisation
    this class used to hard-code. ``DecodeConfig.production()`` reconstructs
    the old chain exactly. See :mod:`fishsense_core.image.decode` for the
    measurement behind the change and for what the evidence does not say.
    """

    def __init__(self, source: Path | bytes, *, config: DecodeConfig | None = None):
        self.__source = source
        self.__config = config or DecodeConfig()

        super().__init__()

    @property
    def config(self) -> DecodeConfig:
        """The decode this image was built with."""
        return self.__config

    def _get_data(self) -> np.ndarray:
        if isinstance(self.__source, (bytes, bytearray, memoryview)):
            _log.debug("loading raw image from %d bytes", len(self.__source))
        else:
            _log.debug("loading raw image: %s", self.__source)

        img = decode_rectified_stage(self.__source, self.__config)

        _log.debug(
            "raw image loaded: shape=%s decode=%s", img.shape, self.__config.label
        )
        return img
