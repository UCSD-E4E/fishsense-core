"""Which colour laser made this dot, read off the dot's own pixels.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/src/fishsense_services_processor/
laser_predict/laser_color.py``, which was itself ported verbatim from
fishsense-lite@77e8f8e5 services/fishsense-data-processing-
workflow-worker/src/fishsense_data_processing_workflow_worker/laser_color.py.
Two deliberate changes from that code, both in :func:`classify_laser_color`
(see its docstring): the margin is scaled by the brightest *usable* pixel,
so a saturated dot is stretched like an unsaturated one, and a float image
is never tested for saturation against its own maximum.

**Coordinates.** :meth:`~fishsense_core.laser.LaserDetector.predict` reports
**sensor** pixels by default, the frame
:class:`~fishsense_core.image.linear_raw_image.LinearRawImage` is in, so a
default prediction samples a ``LinearRawImage`` directly, with no mapping.
Only ``rectify_output=True`` reports **rectified** pixels (the frame
:class:`~fishsense_core.image.rectified_image.RectifiedImage` and human labels
are in). To sample a rectified point on a sensor-frame image, map it back with
:func:`fishsense_core.laser.unrectify_prediction` (also importable here as
:func:`rectified_to_sensor_point`, fishsense-services' name for it). Never map
a point that is already in sensor pixels: that distorts it twice, by tens of
pixels near the laser region, and the 13x13 window then misses the dot while
still returning a plausible colour.

**Why this exists.** The laser labelling config offers "Red Laser" and "Green
Laser", and fishsense-lite's pre-annotation hardcoded "Red Laser", which is
wrong for about a quarter of production: 143 dives are entirely red, 88
entirely green. Colour is a property of the rig for a whole dive.

The measurement, over 332 human-labelled dots on 8-bit processed JPEGs:
comparing R against G **at the dot** separates red from green at 98.19%. All
6 misses are frames where the labelled point sits a pixel or two off the dot,
which is why this samples a small neighbourhood and weights it toward the
bright pixels rather than reading one pixel. That figure was measured on the
code before the two changes above; they move only frames with a saturated
core (the uint8 ones included) and float images.

Taking ``max(R-G)`` over a ±7 px window instead scores 94.88%, *worse* than the
single centre pixel. A window that size is mostly background, and an
underwater scene is blue-green, so the max finds the scene, not the laser.
The dot is the bright thing in its own neighbourhood; that is the signal.

Per frame this is advisory: take the dive-level majority of the votes. That
is what turns a 98% per-frame rule into 4/4 dives correct on the corpus, and
it matches the physics, since one rig shoots one colour for a whole dive.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from fishsense_core._laser_detector import unrectify_prediction

__all__ = [
    "DEFAULT_SAMPLE_RADIUS",
    "MIN_COLOR_MARGIN",
    "classify_laser_color",
    "rectified_to_sensor_point",
]

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

    :func:`fishsense_core.laser.unrectify_prediction` under the name
    fishsense-services calls it. Only for a point that is in rectified pixels
    (a ``rectify_output=True`` prediction, a human label); see the module
    docstring's note on coordinates.
    """
    return unrectify_prediction(x, y, np.asarray(camera_matrix), np.asarray(distortion))


def classify_laser_color(  # pylint: disable=too-many-locals
    bgr: np.ndarray,
    x: float,
    y: float,
    radius: int = DEFAULT_SAMPLE_RADIUS,
    *,
    full_scale: float | None = None,
) -> Tuple[str | None, float | None]:
    """Return ``("red" | "green" | None, margin)`` for the dot at ``(x, y)``.

    ``bgr`` is a BGR array, 8-bit or 16-bit linear (or float, see
    ``full_scale``), in whatever frame ``(x, y)`` is expressed in; the caller
    hands over coordinates that match the image (see the module docstring's
    note on coordinates). The margin is signed R−G in 8-bit levels, positive
    for red, so a caller can tell a decisive frame from a marginal one.

    Two kinds of abstention, told apart by the margin:

    * ``(None, None)``: no reading was possible. The image is not a 3-channel
      array, or the point is not finite or falls outside the frame.
    * ``(None, margin)``: a reading was taken but ``|margin|`` is under
      :data:`MIN_COLOR_MARGIN`, too close to call. A vote counter must
      check the colour, not the margin, to skip it.

    Abstaining is the right failure: a vote counter is better served by no
    vote than a coin flip.

    Args:
        full_scale: The image's saturation ceiling. Defaults to the dtype's
            maximum for an integer image (255, 65535). A float image has no
            ceiling the array can tell us, so with the default none of its
            pixels is treated as saturated; pass the ceiling (1.0 for a
            normalised image, the sensor's white level for linear counts) to
            drop a blown-out core. It is never taken from the patch itself:
            the dot is by construction the patch's brightest thing, so a
            patch-relative ceiling discards the pixels being asked about.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
        return None, None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None, None

    height, width = bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < width and 0 <= yi < height):
        return None, None

    # In frame, so the patch holds at least the (xi, yi) pixel.
    patch = bgr[
        max(0, yi - radius) : yi + radius + 1,
        max(0, xi - radius) : xi + radius + 1,
    ].astype(np.float64)
    blue, green, red = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    if full_scale is None:
        full_scale = (
            float(np.iinfo(bgr.dtype).max)
            if np.issubdtype(bgr.dtype, np.integer)
            else math.inf
        )

    # A close dot blows its core out to white, which carries no colour at all.
    # Drop saturated pixels *before* picking the bright tail, not after: the
    # core is the brightest thing in the window, so a tail taken first is
    # entirely core and there is nothing left to discard. What remains is the
    # halo, which is where the colour actually lives.
    brightest = patch.max(axis=2)
    usable = brightest < full_scale * 0.99
    if usable.sum() < 3:
        usable = np.ones(patch.shape[:2], dtype=bool)

    # Normalise to 8-bit levels by the brightest pixel actually read, so a
    # 16-bit linear frame and an 8-bit JPEG produce comparable margins (the
    # threshold was measured on the latter), and a saturated dot is stretched
    # like an unsaturated one. Scaling by the whole patch let a blown-out
    # core set the scale and shrink the halo's margin, by ~256x in 16-bit.
    scale = 255.0 / max(float(brightest[usable].max()), 1.0)

    # "The dot" = the brightest tail of what is left. Chosen over the centre
    # pixel because the caller passes a *predicted* point, and over a
    # max-over-window because that finds the blue-green scene instead. The
    # tail is never empty: the quantile is at most the maximum.
    luminance = blue + green + red
    cutoff = float(np.quantile(luminance[usable], _BRIGHT_QUANTILE))
    sample = usable & (luminance >= cutoff)

    margin = float(np.median((red - green)[sample]) * scale)
    if abs(margin) < MIN_COLOR_MARGIN:
        return None, margin
    return ("red" if margin > 0 else "green"), margin
