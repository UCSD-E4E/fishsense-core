"""Tests for the board-plane primitive.

The load-bearing property is that only the plane survives into laser
calibration: in-plane rotation and translation must produce *zero* change,
along-normal motion must shift the offset by exactly that distance, and the
metric is blind to the ``(n, d) == (-n, -d)`` sign ambiguity. These are the
invariances `slate_training.geometry` documents from real data; here they are
pinned synthetically where the expected value is exact.
"""

import numpy as np
import pytest

from fishsense_core.plane import (
    Plane,
    laser_point_on_plane,
    plane_difference,
    plane_from_pose,
)


def _rx(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def test_plane_from_pose_uses_third_column_and_normalizes():
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
    rot, t = _rx(0.4), np.array([0.1, -0.2, 1.5])
    base = plane_from_pose(rot, t)
    moved = plane_from_pose(rot, t + 0.01 * rot[:, 2])  # 1 cm along the normal
    angle, offset = plane_difference(base, moved)
    assert angle == pytest.approx(0.0, abs=1e-9)
    assert offset == pytest.approx(10.0, abs=1e-9)  # 1 cm -> 10 mm, exactly


def test_plane_difference_is_blind_to_sign():
    n = np.array([0.0, -np.sin(0.4), np.cos(0.4)])
    a = Plane(n, 1.5)
    b = Plane(-n, -1.5)  # the identical plane, opposite sign convention
    angle, offset = plane_difference(a, b)
    assert angle == pytest.approx(0.0, abs=1e-9)
    assert offset == pytest.approx(0.0, abs=1e-9)


def test_plane_difference_reports_normal_angle_in_degrees():
    a = Plane(np.array([0.0, 0.0, 1.0]), 1.0)
    b = Plane(np.array([0.0, np.sin(np.radians(10)), np.cos(np.radians(10))]), 1.0)
    angle, _ = plane_difference(a, b)
    assert angle == pytest.approx(10.0, abs=1e-9)


def test_laser_point_on_plane_intersects():
    # K = identity intrinsics; a laser at the principal point back-projects to
    # the +z ray, which meets the plane z = 2 at (0, 0, 2).
    k = np.eye(3)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    hit = laser_point_on_plane(plane, (0.0, 0.0), k)
    np.testing.assert_allclose(hit, [0.0, 0.0, 2.0], atol=1e-12)


def test_laser_point_on_plane_parallel_ray_returns_none():
    k = np.eye(3)
    # Plane whose normal is orthogonal to the +z ray -> no intersection.
    plane = Plane(np.array([1.0, 0.0, 0.0]), 1.0)
    assert laser_point_on_plane(plane, (0.0, 0.0), k) is None


def test_laser_point_on_plane_nan_input_returns_none():
    k = np.eye(3)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    assert laser_point_on_plane(plane, (float("nan"), 0.0), k) is None
