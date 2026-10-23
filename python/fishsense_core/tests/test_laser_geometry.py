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

import numpy as np
import pytest

from fishsense_core.world_point import (
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
    assert point.residual_m < 1e-5


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

    A laser axis parallel to the image plane never meets the camera ray, and
    the kernel answers with the origin rather than NaN or an exception. It is
    just as willing to return a *negative* depth when the dot sits on the
    wrong side of the principal point for the laser's offset. Neither is
    caught by an ``isfinite`` check, so callers gate on ``depth_m > 0``.
    """
    intrinsics = _intrinsics()

    parallel = compute_laser_point(DOT, POSITION, (1.0, 0.0, 0.0), intrinsics)
    assert parallel.depth_m <= 0.0
    assert np.isfinite(parallel.depth_m), "not NaN -- an isfinite guard misses it"

    # Laser offset along +x, so a real dot always lands right of the principal
    # point. A label to the left of it inverts the solve.
    wrong_side = compute_laser_point((800.0, 700.0), POSITION, AXIS, intrinsics)
    assert wrong_side.depth_m < 0.0
    assert np.isfinite(wrong_side.depth_m)


def test_a_length_survives_a_negative_depth_unchanged():
    """The reason the guard belongs on the depth and not on the length: the
    back-projection is linear in depth, so flipping its sign moves head and
    tail together and leaves their separation identical."""
    intrinsics = _intrinsics()

    positive = measure_length_at_depth(HEAD, TAIL, 1.5, intrinsics)
    negative = measure_length_at_depth(HEAD, TAIL, -1.5, intrinsics)

    assert negative == pytest.approx(positive)


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
