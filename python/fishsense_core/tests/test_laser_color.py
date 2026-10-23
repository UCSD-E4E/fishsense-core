"""Reading the laser's colour off its own pixels.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/tests/test_laser_color.py``, itself
ported verbatim from fishsense-lite@77e8f8e5 services/fishsense-data-processing-
workflow-worker/tests/test_laser_color.py. The original tests are unchanged
but for the import; the round trips through core's own forward transforms and
the golden values at the end are new.

The laser LS config offers "Red Laser" and "Green Laser", and the pre-annotation
used to hardcode "Red Laser" on every frame. That is wrong for roughly a
quarter of prod: of the dives with completed laser labels, 143 are entirely
red and 88 entirely green (the 31 "mixed" ones carry a 1.2% minority, which is
labeler slips rather than a laser changing colour mid-dive).

So colour is a per-dive constant that nothing was tracking, and the dot itself
is the evidence. Measured over 332 human-labelled dots that still have a
processed JPEG in Garage: comparing R and G *at the dot* separates red from
green at 98.19%, and every one of the 6 misses is a frame where the labelled
point sits a pixel or two off the dot.

Two things that look like improvements and are not:

* Taking max(R-G) over a +-7px window scores 94.88% -- worse. A window that
  size is mostly background, and an underwater scene is blue-green, so the max
  finds the scene rather than the dot. Hence the brightest-pixels rule below:
  the dot is the bright thing in its own neighbourhood.
* Trusting any single frame. Populate takes the dive-level majority instead,
  which is what turns a 98% per-frame rule into 4/4 dives correct.

The coordinate detail is the one that would fail silently: the detector returns
*rectified* pixels, while `LinearRawImage` is in **sensor** coordinates. Sample
one at the other's coordinates and you read the wrong pixels -- near the laser
region that is tens of pixels, easily missing the dot entirely, and the result
still looks like a plausible colour.
"""

# pylint: disable=missing-function-docstring,no-member,unbalanced-tuple-unpacking

from __future__ import annotations

import numpy as np
import pytest

from fishsense_core.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.image import Image
from fishsense_core.image.rectified_image import RectifiedImage
from fishsense_core.laser_color import (
    MIN_COLOR_MARGIN,
    classify_laser_color,
    rectified_to_sensor_point,
)

_K = [[2833.0, 0.0, 2010.0], [0.0, 2858.0, 1450.0], [0.0, 0.0, 1.0]]
_D = [-0.28, 0.12, 0.0005, -0.0003, -0.03]


def _frame(h=400, w=400, bg=(90, 70, 60)):
    """A dim blue-green-ish background, like an underwater scene (BGR)."""
    img = np.zeros((h, w, 3), np.uint16)
    img[:, :] = bg
    return img


def _dot(img, x, y, bgr, radius=4):
    ys, xs = np.ogrid[: img.shape[0], : img.shape[1]]
    img[((xs - x) ** 2 + (ys - y) ** 2) <= radius**2] = bgr
    return img


# --- coordinate mapping -----------------------------------------------------


@pytest.mark.parametrize(
    "sensor_xy",
    [(2010.0, 1450.0), (1600.0, 600.0), (2400.0, 1900.0), (1000.0, 300.0)],
)
def test_rectified_to_sensor_inverts_the_undistort(sensor_xy):
    """Round-trip against OpenCV's own forward transform.

    `undistortPoints(..., P=K)` is precisely what maps a sensor pixel into the
    rectified frame the detector reports in, so mapping back has to land on the
    pixel we started from.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    src = np.array([[list(sensor_xy)]], dtype=np.float64)
    rectified = cv2.undistortPoints(
        src, np.array(_K), np.array(_D), P=np.array(_K)
    ).reshape(2)
    back = rectified_to_sensor_point(float(rectified[0]), float(rectified[1]), _K, _D)
    assert back == pytest.approx(sensor_xy, abs=0.5)


def test_mapping_actually_moves_the_point():
    """A no-op mapping would pass the round-trip test above and still be
    wrong. At the laser region the distortion shift is tens of pixels --
    enough to miss the dot completely."""
    # The region's top-left corner, where the shift is ~48 px. It is only
    # ~5 px near the frame centre, so a test point matters here.
    x, y = 1580.0, 395.0
    sx, sy = rectified_to_sensor_point(x, y, _K, _D)
    assert np.hypot(sx - x, sy - y) > 10.0


def test_zero_distortion_is_the_identity():
    assert rectified_to_sensor_point(1234.0, 567.0, _K, [0.0] * 5) == pytest.approx(
        (1234.0, 567.0), abs=1e-6
    )


# --- colour classification --------------------------------------------------


def test_reads_a_red_dot():
    img = _dot(_frame(), 200, 200, (40, 60, 4000))
    color, margin = classify_laser_color(img, 200, 200)
    assert color == "red" and margin > 0


def test_reads_a_green_dot():
    img = _dot(_frame(), 200, 200, (40, 4000, 60))
    color, margin = classify_laser_color(img, 200, 200)
    assert color == "green" and margin < 0


def test_tolerates_the_point_being_a_couple_of_pixels_off():
    """The production caller passes a *predicted* point, not a human label."""
    img = _dot(_frame(), 200, 200, (40, 4000, 60))
    assert classify_laser_color(img, 202, 198)[0] == "green"


def test_is_not_fooled_by_a_blue_green_background():
    """An underwater scene is greenish everywhere; only the dot counts. This
    is the case the +-7px max-over-window rule got wrong."""
    img = _frame(bg=(400, 800, 120))
    _dot(img, 200, 200, (200, 300, 4000))
    assert classify_laser_color(img, 200, 200)[0] == "red"


def test_returns_none_outside_the_frame():
    assert classify_laser_color(_frame(), 5000, 5000) == (None, None)


def test_saturated_white_core_still_reads_from_the_halo():
    """A close dot blows out to white, which carries no colour. The halo
    around it does -- 4 of the 6 misses in the corpus look like this."""
    img = _frame()
    _dot(img, 200, 200, (300, 3000, 300), radius=7)  # green halo
    _dot(img, 200, 200, (65535, 65535, 65535), radius=3)  # blown core
    assert classify_laser_color(img, 200, 200)[0] == "green"


def test_margin_scales_with_separation():
    """A washed-out dot must report a smaller margin than a vivid one, so a
    close call can be recognised as one."""
    vivid = classify_laser_color(_dot(_frame(), 200, 200, (40, 60, 4000)), 200, 200)[1]
    washed = classify_laser_color(_dot(_frame(), 200, 200, (300, 380, 460)), 200, 200)[
        1
    ]
    assert vivid > washed > 0


def test_uint8_and_uint16_agree():
    """The JPEG corpus this was tuned on is 8-bit; production samples 16-bit
    linear. The call must not depend on the container's scale."""
    small = _dot(np.full((400, 400, 3), 60, np.uint8), 200, 200, (10, 15, 250))
    big = _dot(_frame(bg=(60, 60, 60)), 200, 200, (2560, 3840, 64000))
    assert classify_laser_color(small, 200, 200)[0] == "red"
    assert classify_laser_color(big, 200, 200)[0] == "red"


# --- round trips through core's own forward transforms ---------------------
#
# The tests above round-trip against a bare `cv2.undistortPoints`. These use
# the two places core actually moves a point into the rectified frame: the
# laser detector's `rectify_prediction` (its `rectify_output=True` path) and
# `RectifiedImage` (the `cv2.undistort` that puts pixels in label space).


@pytest.mark.parametrize(
    "sensor_xy",
    [(2010.0, 1450.0), (1600.0, 600.0), (2400.0, 1900.0), (1000.0, 300.0)],
)
def test_round_trip_through_the_detectors_rectify_prediction(sensor_xy):
    """`rectify_prediction` is float32 and inherits `undistortPoints`' fixed
    iteration count, so the round trip is not exact. 0.5 px is the bound the
    services test used; measured worst case here is 0.0099 px, at the
    (1000, 300) corner."""
    from fishsense_core._laser_detector import (  # pylint: disable=import-outside-toplevel
        rectify_prediction,
    )

    rx, ry = rectify_prediction(*sensor_xy, np.array(_K), np.array(_D))
    back = rectified_to_sensor_point(rx, ry, _K, _D)
    assert back == pytest.approx(sensor_xy, abs=0.5)


class _ArrayImage(Image):  # pylint: disable=too-few-public-methods
    def __init__(self, data: np.ndarray):
        self.__data = data
        super().__init__()

    def _get_data(self) -> np.ndarray:
        return self.__data


@pytest.mark.parametrize(
    "sensor_xy", [(2010.0, 1450.0), (1597.9, 439.5), (2420.3, 1910.7)]
)
def test_round_trip_through_rectified_image(sensor_xy):
    """A dot drawn in the sensor frame, rectified by `RectifiedImage`, found
    again and mapped back, lands on the pixel it was drawn at.

    `cv2.undistort` builds its remap by distorting each rectified pixel --
    the same forward model `rectified_to_sensor_point` evaluates -- so this is
    bounded by the dot's centroid and bilinear resampling, not by an
    iterative solve. 0.1 px tolerance; measured error is <= 0.008 px.
    """
    sx, sy = sensor_xy
    ys, xs = np.mgrid[:3000, :4000].astype(np.float32)
    sensor = np.exp(-((xs - sx) ** 2 + (ys - sy) ** 2) / (2 * 3.0**2))
    rectified = RectifiedImage(
        _ArrayImage(sensor), CameraIntrinsics(np.array(_K), np.array(_D))
    ).data
    cy, cx = np.unravel_index(int(np.argmax(rectified)), rectified.shape)
    win = rectified[cy - 12 : cy + 13, cx - 12 : cx + 13].astype(np.float64)
    wy, wx = np.mgrid[cy - 12 : cy + 13, cx - 12 : cx + 13]
    rx, ry = (wx * win).sum() / win.sum(), (wy * win).sum() / win.sum()
    back = rectified_to_sensor_point(rx, ry, _K, _D)
    assert back == pytest.approx(sensor_xy, abs=0.1)


def test_accepts_core_camera_intrinsics_arrays():
    """The function takes the two arrays; a `CameraIntrinsics`' attributes
    (read-only float64, distortion flattened) give the same answer as lists."""
    intr = CameraIntrinsics(np.array(_K), np.array([_D]))
    assert rectified_to_sensor_point(
        1580.0, 395.0, intr.camera_matrix, intr.distortion_coefficients
    ) == rectified_to_sensor_point(1580.0, 395.0, _K, _D)


# --- golden values ----------------------------------------------------------
#
# Produced by fishsense-services v0.48.0's own `laser_color.py` on these exact
# inputs. Exact float equality for the mapping: the port runs the same OpenCV
# calls in the same order.


@pytest.mark.parametrize(
    ("rectified_xy", "sensor_xy"),
    [
        ((2010.0, 1450.0), (2010.0, 1450.0)),
        ((1580.0, 395.0), (1597.9068028938752, 439.4939539991607)),
        ((100.0, 100.0), (374.5450445832073, 295.4256267604942)),
        ((3900.0, 2900.0), (3620.1199846822146, 2686.739113146596)),
    ],
)
def test_rectified_to_sensor_golden(rectified_xy, sensor_xy):
    assert rectified_to_sensor_point(*rectified_xy, _K, _D) == pytest.approx(
        sensor_xy, rel=1e-12
    )


def _saturated_green():
    img = _frame()
    _dot(img, 200, 200, (300, 3000, 300), radius=7)
    return _dot(img, 200, 200, (65535, 65535, 65535), radius=3)


@pytest.mark.parametrize(
    ("make", "xy", "expected"),
    [
        (lambda: _dot(_frame(), 200, 200, (40, 60, 4000)), (200, 200), ("red", 251.175)),
        (
            lambda: _dot(_frame(), 200, 200, (40, 4000, 60)),
            (202, 198),
            ("green", -251.175),
        ),
        (_saturated_green, (200, 200), ("green", -10.505836575875486)),
        (
            lambda: _dot(_frame(), 200, 200, (300, 380, 460)),
            (200, 200),
            ("red", 44.34782608695652),
        ),
        (
            lambda: _dot(_frame(), 200, 200, (300, 301, 302)),
            (200, 200),
            (None, 0.8443708609271523),
        ),
    ],
)
def test_classify_golden(make, xy, expected):
    color, margin = classify_laser_color(make(), *xy)
    assert color == expected[0]
    assert margin == pytest.approx(expected[1], rel=1e-12)
    if color is None:
        assert abs(margin) < MIN_COLOR_MARGIN
