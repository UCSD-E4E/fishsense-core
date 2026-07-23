"""Linear (no-CLAHE) raw decoding for the laser detector.

:class:`fishsense_core.image.RawImage` applies auto-gamma and CLAHE, which is
right for viewing and for the fish models but *wrong* for laser detection:
CLAHE saturates bright laser blobs across all channels, destroying the
wavelength selectivity the detector's 6-channel input depends on. This module
provides the decode the detector was actually trained on — linear 16-bit BGR
in sensor coordinates, plus the per-super-cell Bayer-excess channels.

Ported from ``preprocessing/image_loader.py`` (``_decode_raw_linear`` and
``_decode_raw_bayer_excess``) in the laser-detector training repo.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import rawpy

from fishsense_core.image.image import Image, open_image_source

_log = logging.getLogger(__name__)

#: Accepted values for ``bayer_upsample``. See
#: :attr:`LinearRawImage.VALID_UPSAMPLES` for which to use and why.
VALID_UPSAMPLES = ("repeat", "bilinear")


def upsample_super_cells(half: np.ndarray, mode: str = "repeat") -> np.ndarray:
    """Lift a half-resolution super-cell array to full resolution.

    ``"repeat"`` puts each super-cell value at its 2×2 block's top-left
    corner; ``"bilinear"`` puts it at the block centroid (cv2's pixel-centre
    convention, ``src = (dst + 0.5) * scale - 0.5``). The two therefore differ
    by half a super-cell — one full pixel — everywhere the signal has
    gradient, which is why they are not interchangeable.
    """
    if mode not in VALID_UPSAMPLES:
        raise ValueError(f"mode must be one of {VALID_UPSAMPLES}, got {mode!r}")
    if mode == "repeat":
        return np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)
    half_h, half_w = half.shape[:2]
    return cv2.resize(  # pylint: disable=no-member
        half, (half_w * 2, half_h * 2),
        interpolation=cv2.INTER_LINEAR,  # pylint: disable=no-member
    )


class LinearRawImage(Image):
    """A raw image decoded linearly, in sensor coordinates, without CLAHE.

    ``data`` is uint16 BGR. ``bayer_excess`` is a uint16 ``[H, W, 2]`` array of
    per-super-cell green/red excess upsampled to full resolution; it is
    computed lazily on first access and cached.

    Unlike :class:`~fishsense_core.image.RawImage` this applies **no** EXIF
    rotation (``user_flip=0``). The camera rig is a body-frame property — the
    laser sits at a fixed place in the sensor's coordinate frame — so the
    detector's static rig prior only holds in sensor coordinates.
    """

    # pylint: disable=no-member

    #: Upsample used to lift the half-resolution super-cell excess to full
    #: resolution. ``"repeat"`` places each super-cell value at the block's
    #: top-left; ``"bilinear"`` places it at the block centroid, which is
    #: geometrically correct but shifts every pixel by half a super-cell
    #: relative to ``"repeat"``.
    #:
    #: The default is ``"repeat"`` because that is what the published
    #: checkpoint's Bayer cache, audit numbers, and bias-offset calibration
    #: were all produced with. ``"bilinear"`` is the better transform in
    #: isolation, but pairing it with the ``"repeat"``-era bias offset
    #: reintroduces roughly the (-1.1, -2.1) px bias the offset exists to
    #: cancel. Only switch if you also re-calibrate the offset.
    VALID_UPSAMPLES = VALID_UPSAMPLES

    def __init__(self, source: Path | bytes, *, bayer_upsample: str = "repeat"):
        if bayer_upsample not in self.VALID_UPSAMPLES:
            raise ValueError(
                f"bayer_upsample must be one of {self.VALID_UPSAMPLES}, "
                f"got {bayer_upsample!r}"
            )
        self.__source = source
        self.__bayer_upsample = bayer_upsample
        self.__bayer_excess: np.ndarray | None = None

        super().__init__()

    def _get_data(self) -> np.ndarray:
        """Decodes the raw image to linear 16-bit BGR in sensor coordinates."""
        with open_image_source(self.__source) as f:
            with rawpy.imread(f) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    gamma=(1, 1),          # linear — no gamma correction
                    no_auto_bright=True,   # no histogram stretch
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    user_flip=0,           # no EXIF rotation (sensor coords)
                )

        _log.debug("linear raw image decoded: shape=%s", rgb.shape)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @property
    def bayer_excess(self) -> np.ndarray | None:
        """Per-super-cell green/red excess, upsampled to full resolution.

        Returns a uint16 ``[H, W, 2]`` array of ``(G_excess, R_excess)``, or
        ``None`` if the Bayer pattern could not be interpreted.
        """
        if self.__bayer_excess is None:
            self.__bayer_excess = self.__decode_bayer_excess()
        return self.__bayer_excess

    # pylint: disable-next=too-many-locals
    def __decode_bayer_excess(self) -> np.ndarray | None:
        """Computes ``(G_excess, R_excess)`` from the undemosaiced mosaic.

        Each photosite sees one band, so a green laser saturates the G
        photosites while leaving R/B headroom (and vice versa). Per 2×2 cell::

            G_avg    = (G1 + G2) / 2
            G_excess = max(0, G_avg - max(R, B))
            R_excess = max(0, R     - max(G_avg, B))

        These survive demosaicing and chromaticity normalization, both of
        which wash the signal out.
        """
        with open_image_source(self.__source) as f:
            with rawpy.imread(f) as raw:
                mosaic = raw.raw_image_visible.copy()
                pattern = np.asarray(raw.raw_pattern)
                color_desc = raw.color_desc.decode()
                black = list(raw.black_level_per_channel)

        if mosaic.ndim != 2:
            _log.warning("unexpected raw_image_visible shape %s", mosaic.shape)
            return None
        height, width = mosaic.shape

        color_at = {
            (di, dj): color_desc[int(pattern[di, dj])]
            for di in range(2)
            for dj in range(2)
        }
        r_offsets = [off for off, c in color_at.items() if c == "R"]
        g_offsets = [off for off, c in color_at.items() if c == "G"]
        b_offsets = [off for off, c in color_at.items() if c == "B"]
        if len(r_offsets) != 1 or len(g_offsets) != 2 or len(b_offsets) != 1:
            _log.warning(
                "unexpected Bayer pattern (R=%d G=%d B=%d)",
                len(r_offsets), len(g_offsets), len(b_offsets),
            )
            return None

        def plane(off: tuple[int, int]) -> np.ndarray:
            """Black-level-corrected photosite plane at a 2x2 offset."""
            idx = int(pattern[off[0], off[1]])
            level = int(black[idx]) if idx < len(black) else 0
            return np.maximum(
                mosaic[off[0]::2, off[1]::2].astype(np.int32) - level, 0
            )

        r_arr = plane(r_offsets[0])
        b_arr = plane(b_offsets[0])
        g_avg = (plane(g_offsets[0]) + plane(g_offsets[1])) // 2

        g_excess = np.maximum(g_avg - np.maximum(r_arr, b_arr), 0).astype(np.uint16)
        r_excess = np.maximum(r_arr - np.maximum(g_avg, b_arr), 0).astype(np.uint16)
        half = np.stack([g_excess, r_excess], axis=2)

        return upsample_super_cells(half, self.__bayer_upsample)[:height, :width]
