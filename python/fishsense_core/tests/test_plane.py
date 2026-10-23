"""Tests for the board-plane primitive.

The load-bearing property is that only the plane survives into laser
calibration: in-plane rotation and translation must produce *zero* change,
along-normal motion must shift the offset by exactly that distance, and the
metric is blind to the ``(n, d) == (-n, -d)`` sign ambiguity. These are the
invariances `slate_training.geometry` documents from real data; here they are
pinned synthetically where the expected value is exact.
"""

import cv2
import numpy as np
import pytest

from fishsense_core.plane import (
    Plane,
    laser_point_on_plane,
    plane_difference,
    plane_from_correspondences,
    plane_from_pose,
)


def _rx(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def test_plane_from_pose_uses_third_column_and_normalizes():
    """The normal is the rotation's third column; the offset is normal . t."""
    rot = _rx(0.4)
    t = np.array([0.1, -0.2, 1.5])
    plane = plane_from_pose(rot, t)
    np.testing.assert_allclose(plane.normal, rot[:, 2], atol=1e-12)
    assert np.isclose(np.linalg.norm(plane.normal), 1.0)
    assert np.isclose(plane.distance, rot[:, 2] @ t)


def test_in_plane_rotation_leaves_plane_unchanged():
    """Rotating the board about its own normal (incl. 180 deg) is invisible."""
    rot, t = _rx(0.4), np.array([0.1, -0.2, 1.5])
    base = plane_from_pose(rot, t)
    for theta in (0.3, np.pi / 2, np.pi, 2.1):
        spun = plane_from_pose(rot @ _rz(theta), t)  # Rz fixes the 3rd column
        angle, offset = plane_difference(base, spun)
        assert angle == pytest.approx(0.0, abs=1e-9)
        assert offset == pytest.approx(0.0, abs=1e-9)


def test_in_plane_translation_leaves_plane_unchanged():
    """Sliding the board within its plane (up to 50 cm) is invisible."""
    rot, t = _rx(0.4), np.array([0.1, -0.2, 1.5])
    base = plane_from_pose(rot, t)
    normal = rot[:, 2]
    # Two independent in-plane directions (orthogonal to the normal).
    for in_plane in (rot[:, 0], rot[:, 1], 0.5 * rot[:, 0] - 0.3 * rot[:, 1]):
        assert abs(normal @ in_plane) < 1e-12  # genuinely in-plane
        slid = plane_from_pose(rot, t + 0.5 * in_plane)  # 50 cm
        angle, offset = plane_difference(base, slid)
        assert angle == pytest.approx(0.0, abs=1e-9)
        assert offset == pytest.approx(0.0, abs=1e-9)


def test_along_normal_translation_shifts_offset_by_exactly_that_distance():
    """Moving 1 cm along the normal moves the offset by exactly 10 mm."""
    rot, t = _rx(0.4), np.array([0.1, -0.2, 1.5])
    base = plane_from_pose(rot, t)
    moved = plane_from_pose(rot, t + 0.01 * rot[:, 2])  # 1 cm along the normal
    angle, offset = plane_difference(base, moved)
    assert angle == pytest.approx(0.0, abs=1e-9)
    assert offset == pytest.approx(10.0, abs=1e-9)  # 1 cm -> 10 mm, exactly


def test_plane_difference_is_blind_to_sign():
    """(n, d) and (-n, -d) are one plane, so they differ by nothing."""
    n = np.array([0.0, -np.sin(0.4), np.cos(0.4)])
    a = Plane(n, 1.5)
    b = Plane(-n, -1.5)  # the identical plane, opposite sign convention
    angle, offset = plane_difference(a, b)
    assert angle == pytest.approx(0.0, abs=1e-9)
    assert offset == pytest.approx(0.0, abs=1e-9)


def test_plane_difference_reports_normal_angle_in_degrees():
    """A 10-degree tilt of the normal reads as 10."""
    a = Plane(np.array([0.0, 0.0, 1.0]), 1.0)
    b = Plane(np.array([0.0, np.sin(np.radians(10)), np.cos(np.radians(10))]), 1.0)
    angle, _ = plane_difference(a, b)
    assert angle == pytest.approx(10.0, abs=1e-9)


def test_laser_point_on_plane_intersects():
    """The principal-point ray meets z = 2 at (0, 0, 2)."""
    # K = identity intrinsics; a laser at the principal point back-projects to
    # the +z ray, which meets the plane z = 2 at (0, 0, 2).
    k = np.eye(3)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    hit = laser_point_on_plane(plane, (0.0, 0.0), k)
    np.testing.assert_allclose(hit, [0.0, 0.0, 2.0], atol=1e-12)


def test_laser_point_on_plane_parallel_ray_returns_none():
    """A ray parallel to the plane has no hit."""
    k = np.eye(3)
    # Plane whose normal is orthogonal to the +z ray -> no intersection.
    plane = Plane(np.array([1.0, 0.0, 0.0]), 1.0)
    assert laser_point_on_plane(plane, (0.0, 0.0), k) is None


def test_laser_point_on_plane_nan_input_returns_none():
    """A NaN pixel yields None, not a NaN point."""
    k = np.eye(3)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    assert laser_point_on_plane(plane, (float("nan"), 0.0), k) is None


# ── plane_from_correspondences ────────────────────────────────────────
#
# Ported from fishsense-services v0.48.0 `tests/test_calibration_geometry.py`
# (itself from fishsense-lite@77e8f8e5), adapted to `Plane(normal, distance)`.
# Conventions: +Z is depth, projection is a plain pinhole through
# `CAMERA_MATRIX`, and `solvePnP` gets zero distortion because callers feed it
# rectified points.

CAMERA_MATRIX = np.array(
    [
        [3000.0, 0.0, 2048.0],
        [0.0, 3000.0, 1536.0],
        [0.0, 0.0, 1.0],
    ]
)

# A deliberately tilted pose. A fronto-parallel plane would hide a whole class
# of sign and transpose errors, because its normal is +Z whatever you do to the
# rotation.
_RVEC = np.array([0.12, -0.20, 0.05])
_TVEC = np.array([-0.06, 0.03, 1.40])


def _pose() -> tuple:
    rotation, _ = cv2.Rodrigues(_RVEC)  # pylint: disable=no-member
    return rotation, _TVEC


def _to_camera(body_xy: np.ndarray) -> np.ndarray:
    """Lift planar body points (z=0) into camera space under the test pose."""
    rotation, tvec = _pose()
    body = np.zeros((len(body_xy), 3))
    body[:, :2] = body_xy
    return (rotation @ body.T + tvec.reshape(3, 1)).T


def _project(point_camera: np.ndarray) -> tuple:
    p = CAMERA_MATRIX @ point_camera
    return float(p[0] / p[2]), float(p[1] / p[2])


def _slate_body() -> np.ndarray:
    """Six scattered points, the shape a hand-labelled dive slate produces."""
    return np.array(
        [
            [0.0, 0.0],
            [0.203, 0.0],
            [0.0, 0.254],
            [0.203, 0.254],
            [0.101, 0.0],
            [0.101, 0.254],
        ]
    )


def _checkerboard_body(cols: int = 14, rows: int = 10, square: float = 0.025):
    """Interior corners of a `cols` x `rows` grid — what detection returns.

    Deliberately a different origin, count and spacing from `_slate_body`: the
    point of the seam is that none of that reaches the answer.
    """
    xs, ys = np.meshgrid(np.arange(cols) * square, np.arange(rows) * square)
    return np.column_stack([xs.ravel(), ys.ravel()])


def _observed(body_xy: np.ndarray) -> np.ndarray:
    return np.array([_project(p) for p in _to_camera(body_xy)])


def test_recovers_the_plane_a_known_pose_put_the_target_on():
    """A tilted known pose comes back as its plane."""
    body = _slate_body()
    plane = plane_from_correspondences(body, _observed(body), CAMERA_MATRIX)

    assert plane is not None
    rotation, tvec = _pose()
    # `rotation[:, 2]` is the plane normal: the body's +Z axis in camera space.
    np.testing.assert_allclose(plane.normal, rotation[:, 2], atol=1e-6)
    assert plane.distance == pytest.approx(float(rotation[:, 2] @ tvec), abs=1e-6)
    # Every correspondence must satisfy the plane equation it produced.
    for point in _to_camera(body):
        assert abs(float(plane.normal @ point) - plane.distance) < 1e-7


def test_accepts_three_column_body_points_and_ignores_z():
    """(N, 3) body points behave as (N, 2): the z column is discarded."""
    body = _slate_body()
    body3 = np.column_stack([body, np.full(len(body), 9.0)])
    a = plane_from_correspondences(body, _observed(body), CAMERA_MATRIX)
    b = plane_from_correspondences(body3, _observed(body), CAMERA_MATRIX)
    np.testing.assert_array_equal(a.normal, b.normal)
    assert a.distance == b.distance


def test_solvepnp_does_not_refuse_a_degenerate_correspondence_set():
    """**A trap for automatic producers, pinned because it is not obvious.**
    Six coincident body points against six coincident pixels describe no pose
    at all, and `solvePnP` still returns success — so
    `plane_from_correspondences` hands back a confident, meaningless plane.

    The None path exists (it is `solvePnP`'s own `ret=False`) but it is NOT a
    degeneracy check. A producer that generates correspondences automatically
    must gate on its own detection quality — a partially-detected or collinear
    board has to be rejected before it gets this far.
    """
    body = np.zeros((6, 2))
    image = np.full((6, 2), 2048.0)

    plane = plane_from_correspondences(body, image, CAMERA_MATRIX)

    assert plane is not None  # documents the gap, does not endorse it


def test_recovers_the_three_d_position_of_a_dot_on_the_plane():
    """A dot on the target, not a correspondence, is recovered in 3-D."""
    body = _slate_body()
    plane = plane_from_correspondences(body, _observed(body), CAMERA_MATRIX)

    # A point on the target that is NOT one of the correspondences.
    truth = _to_camera(np.array([[0.077, 0.161]]))[0]
    recovered = laser_point_on_plane(plane, _project(truth), CAMERA_MATRIX)

    assert recovered is not None
    np.testing.assert_allclose(recovered, truth, atol=1e-6)


def test_a_ray_parallel_to_a_recovered_style_plane_yields_no_intersection():
    """The one real failure mode of the intersection itself: a ray that never
    meets the plane has no 3-D position, and None keeps it out of the fit."""
    pixel = (2600.0, 1800.0)
    ray = np.linalg.inv(CAMERA_MATRIX) @ np.array([pixel[0], pixel[1], 1.0])
    # A plane whose normal is perpendicular to that exact ray.
    normal = np.cross(ray, np.array([0.0, 1.0, 0.0]))
    normal = normal / np.linalg.norm(normal)
    assert abs(float(normal @ ray)) < 1e-12

    parallel = Plane(normal, float(normal @ np.array([0.0, 0.0, 1.5])))

    assert laser_point_on_plane(parallel, pixel, CAMERA_MATRIX) is None


def test_the_target_shape_does_not_reach_the_answer():
    """**The seam.** A slate's six hand-clicked points and a checkerboard's 140
    detected corners, describing the same physical plane, must recover the same
    dot position — the target differs only in how `(body_points,
    image_points)` are obtained, and nothing downstream can tell which it was.
    """
    slate_body = _slate_body()
    board_body = _checkerboard_body()

    slate_plane = plane_from_correspondences(
        slate_body, _observed(slate_body), CAMERA_MATRIX
    )
    board_plane = plane_from_correspondences(
        board_body, _observed(board_body), CAMERA_MATRIX
    )
    assert slate_plane is not None and board_plane is not None

    # Same physical plane, expressed from two different body origins.
    angle, offset_mm = plane_difference(slate_plane, board_plane)
    assert angle < 1e-4
    assert offset_mm < 1e-3
    np.testing.assert_allclose(slate_plane.normal, board_plane.normal, atol=1e-6)

    truth = _to_camera(np.array([[0.088, 0.133]]))[0]
    pixel = _project(truth)

    from_slate = laser_point_on_plane(slate_plane, pixel, CAMERA_MATRIX)
    from_board = laser_point_on_plane(board_plane, pixel, CAMERA_MATRIX)

    np.testing.assert_allclose(from_slate, truth, atol=1e-6)
    np.testing.assert_allclose(from_board, truth, atol=1e-6)
    # Services pinned 1e-9 here in float32 via the native handler; core's
    # float64 intersection gets the two to agree at least that well.
    np.testing.assert_allclose(from_slate, from_board, atol=1e-9)


# Recorded by running fishsense-services v0.48.0
# `calibration/geometry.py` (`plane_from_correspondences` ->
# `laser_point_on_plane`) on the fixtures above. Its ray comes from the native
# float32 `WorldPointHandler`; core intersects in float64, so the two agree to
# the float32 ray's precision (largest gap measured: 7.7e-8 m), not bit-for-bit.
# The plane itself (normal and offset) matched services exactly.
_SERVICES_NORMAL = {
    "slate": [-0.19512278620172688, -0.1238416766929502, 0.9729287422102165],
    "board": [-0.19512292417414814, -0.12384155144716631, 0.9729287304817802],
}
_SERVICES_DISTANCE = {"slate": 1.3700923611910987, "board": 1.3700923338117583}
_PARITY_BODY_XY = ([0.077, 0.161], [0.088, 0.133], [0.3, 0.2], [-0.1, 0.4])
_SERVICES_DOTS = {
    "slate": [
        [0.00547471847526106, 0.19253993477995052, 1.433820393360061],
        [0.01796322144827784, 0.16518883884383687, 1.4328435344768755],
        [0.22136105759930802, 0.23959311823488147, 1.4831061087934878],
        [-0.18247319732328263, 0.4228775414815408, 1.4254461669459617],
    ],
    "board": [
        [0.00547471833971168, 0.19253993001282516, 1.4338203578598852],
        [0.01796322107087761, 0.16518883537328422, 1.4328435043733954],
        [0.22136105607232157, 0.23959311658212712, 1.483106098562768],
        [-0.18247318549648966, 0.42287751407321333, 1.425446074557288],
    ],
}


@pytest.mark.parametrize("target", ["slate", "board"])
def test_reproduces_the_services_result(target):
    """Core matches the numbers fishsense-services v0.48.0 produced."""
    body = _slate_body() if target == "slate" else _checkerboard_body()
    plane = plane_from_correspondences(body, _observed(body), CAMERA_MATRIX)
    assert plane is not None

    np.testing.assert_allclose(plane.normal, _SERVICES_NORMAL[target], atol=1e-9)
    assert plane.distance == pytest.approx(_SERVICES_DISTANCE[target], abs=1e-9)

    for body_xy, expected in zip(_PARITY_BODY_XY, _SERVICES_DOTS[target]):
        pixel = _project(_to_camera(np.array([body_xy]))[0])
        dot = laser_point_on_plane(plane, pixel, CAMERA_MATRIX)
        assert dot is not None
        # 0.5 um: float32-ray precision at ~1.4 m, with headroom for the
        # native handler's summation order.
        np.testing.assert_allclose(dot, expected, rtol=0, atol=5e-7)
