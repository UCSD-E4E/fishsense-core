"""Which colour laser made this dot, read off the dot's own pixels.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/src/fishsense_services_processor/
laser_predict/laser_color.py``, which was itself ported verbatim from
fishsense-lite@77e8f8e5 services/fishsense-data-processing-
workflow-worker/src/fishsense_data_processing_workflow_worker/laser_color.py.
The code is unchanged; only this paragraph and the next one (which name
core's own forward transform and coordinate types) are new.

Coordinates. :meth:`~fishsense_core.laser.LaserDetector.predict` reports
sensor pixels by default, and rectified ones with ``rectify_output=True``
(the forward map is ``rectify_prediction``, ``cv2.undistortPoints(..., P=K)``,
the point form of :class:`~fishsense_core.image.rectified_image.RectifiedImage`'s
``cv2.undistort``). :class:`~fishsense_core.image.linear_raw_image.LinearRawImage`
is in sensor coordinates. :func:`rectified_to_sensor_point` is the inverse,
so a rectified point can be sampled on a sensor-frame image. It takes the two
arrays rather than an intrinsics object; pass
``intrinsics.camera_matrix, intrinsics.distortion_coefficients`` from a
:class:`~fishsense_core.CameraIntrinsics` (or anything with those attributes).

The laser LS config offers "Red Laser" and "Green Laser" and the pre-annotation
hardcoded "Red Laser", which is wrong for about a quarter of prod: 143 dives
are entirely red, 88 entirely green. Colour is a property of the rig for a
given dive, and nothing was tracking it — `resolve_laser_predict_inputs` still
sends `wavelength=None`, so the detector runs on its unknown-wavelength
channel and the pipeline had no colour to report.

The measurement, over 332 human-labelled dots that still have a processed JPEG
in Garage: comparing R against G **at the dot** separates red from green at
98.19%. All 6 misses are frames where the labelled point sits a pixel or two
off the dot, which is why this samples a small neighbourhood and weights it
toward the bright pixels rather than reading one pixel.

Taking `max(R-G)` over a ±7 px window instead scores 94.88% — *worse* than the
single centre pixel. A window that size is mostly background, and an
underwater scene is blue-green, so the max finds the scene, not the laser.
The dot is the bright thing in its own neighbourhood; that is the signal.

Per frame this is advisory. Populate takes the dive-level majority, which is
what turns a 98% per-frame rule into 4/4 dives correct on the corpus — and it
matches the physics, since one rig shoots one colour for a whole dive.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

__all__ = ["classify_laser_color", "rectified_to_sensor_point"]

#: Half-width of the sampled neighbourhood, in pixels. Wide enough to survive
#: a prediction landing a couple of pixels off the dot, narrow enough that the
#: dot still dominates the bright tail.
DEFAULT_SAMPLE_RADIUS = 6

#: Fraction of the window, by luminance, treated as "the dot". The dot is a
#: few pixels in a 13x13 window, so this is deliberately a small tail.
_BRIGHT_QUANTILE = 0.90

#: Below this separation (in 8-bit levels, after normalising) the channels are
#: too close to call and the frame abstains rather than casting a vote.
MIN_COLOR_MARGIN = 4.0


def rectified_to_sensor_point(
    x: float,
    y: float,
    camera_matrix: Sequence[Sequence[float]],
    distortion: Sequence[float],
) -> Tuple[float, float]:
    """Map a rectified pixel back to the sensor pixel it came from.

    The detector reports rectified coordinates (`rectify_output=True`) because
    that is the space labelers work in, but `LinearRawImage` — the only decoded
    image the predict activity holds — is in sensor coordinates. Sampling one
    at the other's coordinates reads the wrong pixels, and near the laser
    region that is tens of pixels, which is more than enough to miss the dot
    and still return a confident-looking colour.

    Inverse of `cv2.undistortPoints(..., P=K)`: undo K, apply the distortion
    polynomial, reapply K. `test_laser_color` round-trips it against OpenCV's
    forward transform rather than trusting the algebra.
    """
    k = np.asarray(camera_matrix, dtype=np.float64)
    d = np.asarray(distortion, dtype=np.float64).ravel()

    # undistortPoints with distCoeffs=None is just K^-1 into normalised coords.
    normalized = cv2.undistortPoints(  # pylint: disable=no-member
        np.array([[[float(x), float(y)]]], dtype=np.float64), k, None
    ).reshape(1, 2)
    points_3d = np.hstack([normalized, np.ones((1, 1), dtype=np.float64)])
    projected, _ = cv2.projectPoints(  # pylint: disable=no-member
        points_3d, np.zeros(3), np.zeros(3), k, d.reshape(1, -1)
    )
    return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def classify_laser_color(  # pylint: disable=too-many-locals
    bgr: np.ndarray,
    x: float,
    y: float,
    radius: int = DEFAULT_SAMPLE_RADIUS,
) -> Tuple[str | None, float | None]:
    """Return `("red" | "green" | None, margin)` for the dot at `(x, y)`.

    `bgr` is any BGR array — 8-bit or 16-bit linear, in whatever frame `(x, y)`
    is expressed in; the caller is responsible for handing over coordinates
    that match the image (see `rectified_to_sensor_point`). The margin is
    signed R−G in 8-bit levels, positive for red, so a caller can tell a
    decisive frame from a marginal one.

    `(None, None)` means "no opinion": the point is outside the frame, or the
    channels are too close to call. Abstaining is the right failure here —
    populate counts votes, so a coin-flip vote is worse than no vote.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
        return None, None

    height, width = bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < width and 0 <= yi < height):
        return None, None

    patch = bgr[
        max(0, yi - radius) : yi + radius + 1,
        max(0, xi - radius) : xi + radius + 1,
    ].astype(np.float64)
    if patch.size == 0:
        return None, None

    blue, green, red = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    full_scale = (
        float(np.iinfo(bgr.dtype).max)
        if np.issubdtype(bgr.dtype, np.integer)
        else max(float(patch.max()), 1.0)
    )

    # Normalise to 8-bit levels so a 16-bit linear frame and an 8-bit JPEG
    # produce comparable margins -- the threshold was measured on the latter.
    scale = 255.0 / max(float(patch.max()), 1.0)

    # A close dot blows its core out to white, which carries no colour at all.
    # Drop saturated pixels *before* picking the bright tail, not after: the
    # core is the brightest thing in the window, so a tail taken first is
    # entirely core and there is nothing left to discard. What remains is the
    # halo, which is where the colour actually lives.
    # "Saturated" means at the *container's* ceiling (255 / 65535), never at
    # this patch's own maximum: the dot is by construction the brightest thing
    # in the window, so a patch-relative test would discard exactly the pixels
    # being asked about and read the background instead.
    usable = patch.max(axis=2) < full_scale * 0.99
    if usable.sum() < 3:
        usable = np.ones(patch.shape[:2], dtype=bool)

    # "The dot" = the brightest tail of what is left. Chosen over the centre
    # pixel because the caller passes a *predicted* point, and over a
    # max-over-window because that finds the blue-green scene instead.
    luminance = blue + green + red
    cutoff = float(np.quantile(luminance[usable], _BRIGHT_QUANTILE))
    sample = usable & (luminance >= cutoff)
    if not sample.any():
        return None, None

    margin = float(np.median((red - green)[sample]) * scale)
    if abs(margin) < MIN_COLOR_MARGIN:
        return None, margin
    return ("red" if margin > 0 else "green"), margin
