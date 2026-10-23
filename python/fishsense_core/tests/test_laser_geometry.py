"""The laser point (depth, range, residual) and length-at-depth.

Ported from fishsense-services v0.48.0 ``tests/test_laser_geometry.py``
(itself from fishsense-lite@77e8f8e5). Test names, bodies and reasons are
services'; the one adaptation is the signature: core takes the dot, the
laser's origin/axis and the head/tail pixels as values, not label and
calibration objects.

These pin the two properties a second caller can get wrong: that ``depth_m``
is the Z component (not the Euclidean distance), and that
``measure_length_at_depth`` composes into exactly the length the inline
handler sequence produces. ``test_matches_services_numbers`` pins the values
services v0.48.0 produced on the same inputs, so a later change to the kernel
or the composition shows up as a number, not as a silent rescale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fishsense_core import world_point
from fishsense_core.camera_intrinsics import CameraIntrinsics
from fishsense_core.world_point import (
    DEFAULT_MIN_PARALLAX_RAD,
    LaserPoint,
    WorldPointHandler,
    compute_laser_point,
    measure_length_at_depth,
)

POSITION = np.array([0.104, 0.0, 0.0])
AXIS = np.array([0.0, 0.0, 1.0])
DOT = (1200.0, 800.0)
HEAD, TAIL = (900.0, 700.0), (1400.0, 760.0)


def _intrinsics(fx=2000.0, fy=2000.0, cx=1000.0, cy=750.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _on_laser_dot(intrinsics, position=POSITION, axis=AXIS, distance=1.5):
    """The pixel a dot ``distance`` metres down the laser projects to."""
    origin = np.asarray(position, dtype=float)
    direction = np.asarray(axis, dtype=float)
    truth = origin + distance * direction / np.linalg.norm(direction)
    projected = intrinsics @ truth
    return truth, (projected[0] / projected[2], projected[1] / projected[2])


def test_depth_is_the_z_component_of_the_projected_laser_point():
    """The number the length back-projects head and tail against. Pinned
    against the kernel directly so nothing can quietly substitute the slant
    distance, which is a different -- and larger -- number."""
    intrinsics = _intrinsics()
    expected = WorldPointHandler(np.linalg.inv(intrinsics)).compute_world_point_from_laser(
        POSITION, AXIS, np.array(DOT)
    )

    point = compute_laser_point(DOT, POSITION, AXIS, intrinsics)

    assert isinstance(point, LaserPoint)
    assert point.depth_m == pytest.approx(float(expected[2]))


def test_range_is_the_euclidean_distance_to_the_dot():
    """`range_m` is the norm of the triangulated point, not its Z."""
    intrinsics = _intrinsics()
    expected = WorldPointHandler(np.linalg.inv(intrinsics)).compute_world_point_from_laser(
        POSITION, AXIS, np.array(DOT)
    )

    point = compute_laser_point(DOT, POSITION, AXIS, intrinsics)

    assert point.range_m == pytest.approx(float(np.linalg.norm(expected)))


def test_range_exceeds_depth_for_an_off_axis_dot():
    """The two are only equal on the optical axis. A caller that treats them
    as interchangeable is wrong by the off-axis angle, and silently so."""
    point = compute_laser_point((1600.0, 1200.0), POSITION, AXIS, _intrinsics())

    assert point.range_m > point.depth_m


def test_range_and_depth_agree_with_the_pixels_off_axis_angle():
    """``range_m / depth_m`` is fixed by the pixel alone: the recovered point
    lies on the camera ray through it, so the ratio is |K^-1·[x,y,1]| over its
    Z component, whatever depth the laser happens to put it at."""
    intrinsics = _intrinsics()
    dot = (1600.0, 1200.0)
    direction = np.linalg.inv(intrinsics) @ np.array([dot[0], dot[1], 1.0])
    expected_ratio = float(np.linalg.norm(direction) / direction[2])

    point = compute_laser_point(dot, POSITION, AXIS, intrinsics)

    assert point.range_m / point.depth_m == pytest.approx(expected_ratio)


def test_a_non_unit_laser_axis_does_not_change_the_answer():
    """Until fishsense-core 3.0.0 the kernel silently assumed ||axis|| = 1 and
    returned garbage otherwise -- doubling a unit axis flipped the depth
    negative. A regression there would rescale every depth and every fish
    length in silence."""
    intrinsics = _intrinsics()
    unit = compute_laser_point(DOT, POSITION, AXIS, intrinsics)

    for scale in (2.0, 0.5, 17.3):
        scaled = compute_laser_point(DOT, POSITION, (0.0, 0.0, scale), intrinsics)
        assert scaled.depth_m == pytest.approx(unit.depth_m, rel=1e-6)
        assert scaled.range_m == pytest.approx(unit.range_m, rel=1e-6)


def test_a_zero_length_axis_raises():
    """A calibration with no direction cannot place a dot anywhere, and the
    kernel says so instead of answering with a plausible ~1 cm depth."""
    with pytest.raises(ValueError, match="laser_axis"):
        compute_laser_point(DOT, POSITION, (0.0, 0.0, 0.0), _intrinsics())


def test_residual_is_near_zero_for_a_consistent_dot():
    """The dot really is on the laser ray, so the two rays meet: the
    closest-approach distance collapses to the float32 noise floor."""
    intrinsics = _intrinsics()
    truth, dot = _on_laser_dot(intrinsics)

    point = compute_laser_point(dot, POSITION, AXIS, intrinsics)

    assert point.depth_m == pytest.approx(truth[2], rel=1e-3)
    # 10x the float32 solve's ~1e-5 m noise floor at metre scale: a
    # consistent dot is far below it, and this does not ride the floor.
    assert point.residual_m < 1e-4


def test_residual_grows_when_the_dot_cannot_be_on_the_laser():
    """A dot displaced *across* the laser's epipolar line cannot lie on the
    laser at any depth, and the residual says so."""
    intrinsics = _intrinsics()
    _, (x, y) = _on_laser_dot(intrinsics)

    # The laser is offset in x, so its epipolar line runs horizontally here;
    # displace in y to move across it.
    off = compute_laser_point((x, y + 150.0), POSITION, AXIS, intrinsics)

    assert off.residual_m > 1e-3


def test_residual_is_blind_along_the_epipolar_line():
    """The limit that stops the residual being a sufficient check.

    Sliding the dot *along* the laser's epipolar line moves the recovered
    depth a long way while the rays still meet, so the residual stays at the
    noise floor. The positive-depth and plausible-range checks are not
    redundant with it.
    """
    intrinsics = _intrinsics()
    truth, (x, y) = _on_laser_dot(intrinsics)

    along = compute_laser_point((x - 100.0, y), POSITION, AXIS, intrinsics)
    across = compute_laser_point((x, y + 100.0), POSITION, AXIS, intrinsics)

    assert along.residual_m < across.residual_m / 100.0
    assert abs(along.depth_m - truth[2]) > 0.1, "yet the depth moved a lot"


def test_length_at_depth_matches_the_inline_composition():
    """Byte-for-byte the sequence services' stage 14 ran inline: project the
    laser for a depth, back-project head and tail at that depth, take the
    norm."""
    intrinsics = _intrinsics()
    handler = WorldPointHandler(np.linalg.inv(intrinsics))
    depth = float(
        handler.compute_world_point_from_laser(POSITION, AXIS, np.array(DOT))[2]
    )
    head3d = handler.compute_world_point_from_depth(np.array(HEAD), depth)
    tail3d = handler.compute_world_point_from_depth(np.array(TAIL), depth)
    expected = float(np.linalg.norm(head3d - tail3d))

    point = compute_laser_point(DOT, POSITION, AXIS, intrinsics)
    actual = measure_length_at_depth(HEAD, TAIL, point.depth_m, intrinsics)

    assert actual == expected


def test_geometry_with_no_intersection_yields_a_non_positive_depth():
    """How failure actually presents, which is not how it reads.

    A laser along the camera's own x axis, origin (0.104, 0, 0) and axis
    (1, 0, 0), passes through the camera centre, so it meets *every* camera
    ray there: the kernel answers with the origin, depth 0, residual ~0 (an
    exact intersection, not a missed one), rather than NaN or an exception.
    It is just as willing to return a *negative* depth when the dot sits on
    the wrong side of the principal point for the laser's offset. Neither is
    caught by an ``isfinite`` check; is_measurable rejects both.
    """
    intrinsics = _intrinsics()

    parallel = compute_laser_point(DOT, POSITION, (1.0, 0.0, 0.0), intrinsics)
    assert parallel.depth_m <= 0.0
    assert np.isfinite(parallel.depth_m), "not NaN -- an isfinite guard misses it"
    assert parallel.residual_m < 1e-4, "the lines do meet, at the camera centre"
    assert not parallel.is_measurable()

    # Laser offset along +x, so a real dot always lands right of the principal
    # point. A label to the left of it inverts the solve.
    wrong_side = compute_laser_point((800.0, 700.0), POSITION, AXIS, intrinsics)
    assert wrong_side.depth_m < 0.0
    assert np.isfinite(wrong_side.depth_m)
    assert not wrong_side.is_measurable()


def test_a_negative_depth_would_give_a_plausible_length_so_it_is_refused():
    """Why the guard belongs on the depth and not on the length: the
    back-projection is linear in depth, so flipping its sign moves head and
    tail together and leaves their separation identical. A length cannot
    reveal a behind-the-camera depth, so measure_length_at_depth refuses one
    (services' copy measured it)."""
    intrinsics = _intrinsics()
    handler = WorldPointHandler(np.linalg.inv(intrinsics))

    def span(depth):
        return float(
            np.linalg.norm(
                handler.compute_world_point_from_depth(HEAD, depth)
                - handler.compute_world_point_from_depth(TAIL, depth)
            )
        )

    assert span(-1.5) == pytest.approx(span(1.5))
    assert measure_length_at_depth(HEAD, TAIL, 1.5, intrinsics) == pytest.approx(span(1.5))
    with pytest.raises(ValueError, match="finite and positive"):
        measure_length_at_depth(HEAD, TAIL, -1.5, intrinsics)


def test_accepts_integer_pixels_and_lists():
    """Raw label pixels arrive as ints and calibrations as lists; the result
    is the same as for the float arrays."""
    intrinsics = _intrinsics()
    as_float = compute_laser_point(np.array(DOT), POSITION, AXIS, intrinsics)
    as_int = compute_laser_point((1200, 800), list(POSITION), [0, 0, 1], intrinsics.tolist())

    assert as_int == as_float


# Realistic rig: an FSL-01-like camera matrix and a laser offset down-left of
# the lens, toed in. The first dot lies on the laser ~2 m out (to 0.1 px); the
# second is inconsistent with the calibration by ~10 cm. Values are what
# fishsense-services v0.48.0 `laser_geometry` returned on the same inputs
# (identical, bit for bit, to this port), from the float32 native solve.
_REAL_K = np.array([[2988.7, 0.0, 2006.0], [0.0, 2988.7, 1496.0], [0.0, 0.0, 1.0]])
_REAL_POSITION = (-0.0425, -0.1167, 0.0)
_REAL_AXIS = (0.0147, 0.0623, 0.9979)
_REAL_HEAD, _REAL_TAIL = (1400.2, 1302.5), (2620.9, 1388.1)


@pytest.mark.parametrize(
    ("dot", "depth", "range_", "residual", "length"),
    [
        ((1986.4, 1507.8), 1.9955137, 1.9955721, None, 0.8170460),
        ((2104.3, 1611.8), 2.2293150, 2.2321925, 0.10460509, 0.9127739),
        ((1500.0, 1000.0), 0.4005047, 0.4116063, 0.04011405, 0.1639832),
    ],
)
def test_matches_services_numbers(dot, depth, range_, residual, length):
    """The port returns what services v0.48.0 did on a realistic rig."""
    point = compute_laser_point(dot, _REAL_POSITION, _REAL_AXIS, _REAL_K)

    assert point.depth_m == pytest.approx(depth, rel=1e-5)
    assert point.range_m == pytest.approx(range_, rel=1e-5)
    if residual is None:
        # A consistent dot: at the float32 noise floor, not a pinnable number.
        assert point.residual_m < 1e-4
    else:
        assert point.residual_m == pytest.approx(residual, rel=1e-4)
    assert measure_length_at_depth(
        _REAL_HEAD, _REAL_TAIL, point.depth_m, _REAL_K
    ) == pytest.approx(length, rel=1e-5)


# -- near the laser's vanishing point -----------------------------------------
#
# The laser here runs parallel to the optical axis, so its vanishing point is
# the principal point (1000, 750). A real dot at depth Z lands at
# x = 1000 + 2000 * 0.104 / Z: 1020.8 px at 10 m, 1002.08 px at 100 m.


def test_a_real_dot_is_measurable():
    intrinsics = _intrinsics()
    _, pixel = _on_laser_dot(intrinsics)
    point = compute_laser_point(pixel, POSITION, AXIS, intrinsics)
    assert point.is_measurable()
    # The parallax is the angle the 0.104 m baseline subtends at 1.5 m.
    assert point.parallax_rad == pytest.approx(math.atan(0.104 / 1.5), rel=1e-3)


@pytest.mark.parametrize("pixel_x", [1000.5, 1000.05])
def test_a_dot_near_the_vanishing_point_is_not_measurable(pixel_x):
    """Finite, and maybe positive, but the camera ray and the laser are nearly
    parallel: one pixel of dot error moves this depth by more than itself.
    depth_m > 0 alone would let it through."""
    point = compute_laser_point((pixel_x, 750.0), POSITION, AXIS, _intrinsics())
    assert point.parallax_rad < DEFAULT_MIN_PARALLAX_RAD
    assert not point.is_measurable()


def test_a_dot_at_the_vanishing_point_is_not_measurable_whatever_comes_back():
    """Exactly parallel: the solve divides by ~0 and may return inf, nan or a
    finite number of either sign. is_measurable rejects all of them."""
    point = compute_laser_point((1000.0, 750.0), POSITION, AXIS, _intrinsics())
    assert point.parallax_rad == pytest.approx(0.0, abs=1e-12)
    assert not point.is_measurable()


@pytest.mark.parametrize(
    "fields",
    [
        dict(depth_m=math.inf, range_m=math.inf, residual_m=0.0, parallax_rad=0.1),
        dict(depth_m=math.nan, range_m=1.0, residual_m=0.0, parallax_rad=0.1),
        dict(depth_m=1.0, range_m=1.0, residual_m=math.nan, parallax_rad=0.1),
        dict(depth_m=-1.0, range_m=1.0, residual_m=0.0, parallax_rad=0.1),
        dict(depth_m=0.0, range_m=0.0, residual_m=0.0, parallax_rad=0.1),
        dict(depth_m=1.0, range_m=1.0, residual_m=0.0, parallax_rad=0.001),
    ],
    ids=["inf", "nan-depth", "nan-residual", "behind", "zero", "low-parallax"],
)
def test_is_measurable_rejects(fields):
    assert not LaserPoint(**fields).is_measurable()


def test_is_measurable_honours_a_maximum_depth_and_a_parallax_floor():
    point = LaserPoint(depth_m=12.0, range_m=12.0, residual_m=0.0, parallax_rad=0.008)
    assert point.is_measurable()
    assert not point.is_measurable(max_depth_m=10.0)
    assert not point.is_measurable(min_parallax_rad=0.01)


@pytest.mark.parametrize("depth", [math.inf, -math.inf, math.nan, 0.0, -1.5])
def test_no_length_at_a_depth_that_cannot_carry_one(depth):
    """An inf or behind-the-camera depth can never become a stored length."""
    with pytest.raises(ValueError, match="finite and positive"):
        measure_length_at_depth(HEAD, TAIL, depth, _intrinsics())


# -- the camera matrix ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3)),
        np.zeros((3, 3)),
        np.array([[2000.0, 0, 1000], [0, math.nan, 750], [0, 0, 1]]),
    ],
    ids=["not-3x3", "singular", "nan"],
)
def test_a_bad_camera_matrix_is_a_value_error(bad):
    """Not a LinAlgError a ValueError handler would miss, and not a NaN K
    silently turning every depth into NaN."""
    with pytest.raises(ValueError, match="camera_matrix"):
        compute_laser_point(DOT, POSITION, AXIS, bad)
    with pytest.raises(ValueError, match="camera_matrix"):
        measure_length_at_depth(HEAD, TAIL, 1.5, bad)


def test_camera_intrinsics_are_accepted_as_they_are():
    intrinsics = _intrinsics()
    camera = CameraIntrinsics(intrinsics, np.zeros(5))
    assert compute_laser_point(DOT, POSITION, AXIS, camera) == compute_laser_point(
        DOT, POSITION, AXIS, intrinsics
    )
    assert measure_length_at_depth(HEAD, TAIL, 1.5, camera) == measure_length_at_depth(
        HEAD, TAIL, 1.5, intrinsics
    )


def test_k_is_inverted_once_per_camera_not_per_call():
    world_point._handler_from_bytes.cache_clear()  # pylint: disable=protected-access
    intrinsics = _intrinsics()
    for _ in range(5):
        point = compute_laser_point(DOT, POSITION, AXIS, intrinsics)
        measure_length_at_depth(HEAD, TAIL, abs(point.depth_m) or 1.0, intrinsics)
    info = world_point._handler_from_bytes.cache_info()  # pylint: disable=protected-access
    assert (info.misses, info.hits) == (1, 9)
