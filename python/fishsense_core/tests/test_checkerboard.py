"""Unit tests for the checkerboard corner detector.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/tests/test_checkerboard_detection.py``
and ``tests/test_checkerboard_pitch.py`` (the former itself from
fishsense-lite@77e8f8e5
services/fishsense-data-processing-workflow-worker/tests/test_checkerboard_detection.py),
merged into one file. Imports are adapted; the tests are unchanged.

The detector's whole job is to produce the `(body_points, image_points)` that
`fishsense_core.plane.plane_from_correspondences` turns into a plane — the
same pair a `MissionSlate`'s template points and hand-clicked corners produce
today. Everything downstream is already shared and already tested.

What is pinned here is the part that can go wrong *silently*, because
`solvePnP` does not refuse a bad correspondence set — it returns success and a
confident, meaningless pose:

  * correspondences are index-aligned and built from the **detected** grid,
    never the nominal board;
  * a grid bigger than the declared board is refused (wrong target linked);
  * a degenerate near-collinear detection is refused;
  * a sub-grid recovers the *same plane* as the full board, which is what
    makes partial detections usable at all — and how much less sharply it
    does so once real corner noise is in play.

v2 change: the detector takes the target's pitch per axis (`pitch_x_m`,
`pitch_y_m`; `calibration_targets` is versioned with one per axis, PLAN.md
§4.3). These are v1's tests, run with the two equal -- which is every migrated
target, since v1 kept one `square_size_m`. What changes when they differ is
pinned in the per-axis pitch section at the end of this file.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fishsense_core import checkerboard as sut
from fishsense_core.plane import Plane, laser_point_on_plane, plane_from_pose

# cv2 is a C extension pylint cannot introspect.
# pylint: disable=no-member

SQUARE_PX = 40
SQUARE_SIZE_M = 0.0254

# The E4E board: 15 x 11 squares -> 14 x 10 interior corners.
BOARD_COLS_SQ, BOARD_ROWS_SQ = 15, 11
BOARD_COLS, BOARD_ROWS = BOARD_COLS_SQ - 1, BOARD_ROWS_SQ - 1

CAMERA_MATRIX = np.array([[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]])


def plane_from_correspondences(body_points, image_points, camera_matrix) -> Plane | None:
    """The plane a planar target occupied, from 2-D <-> 3-D pairs.

    A test-local copy of fishsense-services' `calibration.geometry
    .plane_from_correspondences` (zero-distortion `solvePnP` on a z=0 body),
    returning core's `Plane`. Core gains its own
    `fishsense_core.plane.plane_from_correspondences` in a sibling change; this
    copy keeps these tests independent of when that lands.
    """
    body = np.zeros((len(body_points), 3), dtype=np.float32)
    body[:, :2] = np.asarray(body_points, dtype=np.float64)[:, :2]
    image = np.asarray(image_points, dtype=np.float64)
    ret, rvec, tvec = cv2.solvePnP(body, image, camera_matrix, np.zeros((5,)))
    if not ret:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return plane_from_pose(rotation, tvec)


def point_in_laser_region(x: float, y: float, region) -> bool:
    """Whether `(x, y)` lies inside the convex `region`, edges included.

    A test-local copy of `fishsense_services_contracts.laser_region
    .point_in_laser_region` (the consumer of `board_hull`): a point is inside
    iff it is on the same side of every directed edge.
    """
    poly = [tuple(v) for v in region]
    if len(poly) < 3:
        return False
    positive = negative = False
    for index, (ax, ay) in enumerate(poly):
        bx, by = poly[(index + 1) % len(poly)]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross > 0:
            positive = True
        elif cross < 0:
            negative = True
        if positive and negative:
            return False
    return True


def _render(cols_sq: int = BOARD_COLS_SQ, rows_sq: int = BOARD_ROWS_SQ, border=80):
    """A flat checkerboard on a white ground.

    The white border is not decoration — `findChessboardCornersSB` requires a
    quiet boundary around the pattern.
    """
    height = rows_sq * SQUARE_PX + 2 * border
    width = cols_sq * SQUARE_PX + 2 * border
    img = np.full((height, width), 255, np.uint8)
    for row in range(rows_sq):
        for col in range(cols_sq):
            if (row + col) % 2 == 0:
                y_0, x_0 = border + row * SQUARE_PX, border + col * SQUARE_PX
                img[y_0 : y_0 + SQUARE_PX, x_0 : x_0 + SQUARE_PX] = 0
    return img


def _warped(img):
    """The board seen at an angle, so the recovered pose is a real one."""
    height, width = img.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32(
        [
            [width * 0.10, height * 0.05],
            [width * 0.92, height * 0.16],
            [width * 0.88, height * 0.95],
            [width * 0.06, height * 0.86],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(img, matrix, (width, height), borderValue=255)


def _detect(img, **overrides):
    kwargs = {
        "max_rows": BOARD_ROWS,
        "max_cols": BOARD_COLS,
        "pitch_x_m": SQUARE_SIZE_M,
        "pitch_y_m": SQUARE_SIZE_M,
    }
    kwargs.update(overrides)
    return sut.detect_checkerboard(img, **kwargs)


# ---------- the happy path ----------


def test_detects_the_full_board():
    """The whole board comes back, in either orientation, index-aligned."""
    detected = _detect(_render())

    assert detected is not None
    assert sorted((detected.rows, detected.cols)) == sorted((BOARD_ROWS, BOARD_COLS))
    assert len(detected.body_points) == BOARD_ROWS * BOARD_COLS
    assert len(detected.image_points) == len(detected.body_points)


def test_body_points_carry_the_measured_pitch():
    """The one number that sets the scale of every downstream length."""
    detected = _detect(_render())

    grid = detected.body_points.reshape(detected.rows, detected.cols, 2)
    along_cols = np.diff(grid[0, :, 0])
    down_rows = np.diff(grid[:, 0, 1])
    assert np.allclose(along_cols, SQUARE_SIZE_M)
    assert np.allclose(down_rows, SQUARE_SIZE_M)


def test_accepts_a_colour_image():
    """The pipeline hands it `RectifiedImage.data`, which is BGR."""
    colour = cv2.cvtColor(_render(), cv2.COLOR_GRAY2BGR)

    assert _detect(colour) is not None


def test_correspondences_recover_a_plane():
    """The detector's output is exactly what the plane fit consumes."""
    detected = _detect(_warped(_render()))
    assert detected is not None

    plane = plane_from_correspondences(
        detected.body_points, detected.image_points, CAMERA_MATRIX
    )

    assert plane is not None
    assert np.isclose(np.linalg.norm(plane.normal), 1.0)


# ---------- partial views: the property that makes them usable ----------


def test_a_partial_view_is_detected_as_a_sub_grid():
    """Occlusion and the frame edge cut the board down; ~96% still detect.

    Note the exact patternSize would fail outright here — which is why the
    detector asks for a small grid and reads the real one back from the
    detector's own metadata, rather than asserting the nominal board.
    """
    board = _warped(_render())
    cropped = board[:, : int(board.shape[1] * 0.66)].copy()

    detected = _detect(cropped)

    assert detected is not None
    assert detected.rows * detected.cols < BOARD_ROWS * BOARD_COLS


def _project(body_points, rvec, tvec):
    """Where a body grid lands in a rectified image, exactly.

    Analytic rather than detected, because the property below is about the
    geometry and detected corners carry sub-pixel noise that would swamp it —
    the noise is characterised separately in the next test.
    """
    points_3d = np.column_stack(
        [body_points, np.zeros(len(body_points), dtype=np.float64)]
    )
    projected, _ = cv2.projectPoints(
        points_3d, rvec, tvec, CAMERA_MATRIX, np.zeros((5,))
    )
    return projected.reshape(-1, 2)


def test_a_sub_grid_recovers_the_identical_plane():
    """The whole justification for accepting sub-grids.

    A sub-grid of a regular grid has the same pitch, and only the plane
    reaches the answer — the sub-board's unknown offset into the full board
    lands in the in-plane pose, which nothing downstream reads. So a detector
    that returns 10 x 10 of a 14 x 10 board, with no idea *which* 10 x 10, is
    still answering the question stage 13 asks.
    """
    rvec = np.array([0.28, -0.16, 0.07])
    tvec = np.array([-0.10, -0.08, 1.40])
    grid = np.stack(
        np.meshgrid(np.arange(BOARD_COLS), np.arange(BOARD_ROWS)), axis=-1
    ).astype(np.float64)
    body = grid * SQUARE_SIZE_M
    image = _project(body.reshape(-1, 2), rvec, tvec).reshape(BOARD_ROWS, BOARD_COLS, 2)

    dot = np.array([620.0, 500.0])

    def _hit(body_grid, image_grid):
        points = body_grid.reshape(-1, 2)
        # Re-based to the sub-grid's own origin, as the detector emits it.
        points = points - points.min(axis=0)
        plane = plane_from_correspondences(
            points, image_grid.reshape(-1, 2), CAMERA_MATRIX
        )
        return laser_point_on_plane(plane, dot, CAMERA_MATRIX)

    whole = _hit(body, image)
    for rows, cols in (
        (slice(2, None), slice(None, 7)),
        (slice(None, 5), slice(4, None)),
    ):
        assert np.allclose(whole, _hit(body[rows, cols], image[rows, cols]), atol=1e-6)


def test_a_partial_view_agrees_with_the_full_board_to_within_its_conditioning():
    """Detected independently, a sub-grid answers the same question less sharply.

    Not a contradiction of the exact property above: the corners themselves
    agree to 0.025 px between the two runs, but a 10 x 10 sub-grid spans less
    of the frame than the full 14 x 10, so the same corner error buys a worse
    pose. Measured here at ~1.6% of a 1.43 m depth.

    Recorded rather than gated on, because the defence against it is already
    in place and is the right one: `calibrate_laser` is a least-squares fit
    over every observation in the dive (28-133 frames in this corpus, against
    a `MIN_LASER_POINTS` of 2), and `check_fit_self_consistency` refuses a fit
    that disagrees with the dots it came from. Discarding small detections
    instead would throw away usable frames to fix a term the aggregate already
    handles.
    """
    board = _warped(_render())
    cropped = board[:, : int(board.shape[1] * 0.66)].copy()

    full = _detect(board)
    partial = _detect(cropped)
    assert full is not None and partial is not None
    assert (partial.rows, partial.cols) != (full.rows, full.cols)

    dot = np.array([620.0, 500.0])
    from_full = laser_point_on_plane(
        plane_from_correspondences(full.body_points, full.image_points, CAMERA_MATRIX),
        dot,
        CAMERA_MATRIX,
    )
    from_partial = laser_point_on_plane(
        plane_from_correspondences(
            partial.body_points, partial.image_points, CAMERA_MATRIX
        ),
        dot,
        CAMERA_MATRIX,
    )

    relative = np.linalg.norm(from_full - from_partial) / np.linalg.norm(from_full)
    assert relative < 0.03, relative


def test_the_grid_orientation_does_not_change_the_plane():
    """The detected grid may come back transposed or mirrored, and must not matter.

    The board is a symmetric grid, so the detector's indexing is only defined
    up to the 8 symmetries of a rectangle. A mirrored assignment is realised
    by a real rotation (the board flipped 180 degrees about an in-plane axis),
    which flips the plane normal — and the ray-plane intersection divides one
    normal by the other, so the sign cancels.

    This is why the classic checkerboard headaches (180-degree ambiguity,
    which corner is the origin, consistent ordering between frames) are all
    irrelevant to *this* use, and why building body points on the detected
    shape is safe.
    """
    detected = _detect(_warped(_render()))
    assert detected is not None

    body = detected.body_points.reshape(detected.rows, detected.cols, 2)
    image = detected.image_points.reshape(detected.rows, detected.cols, 2)

    dot = np.array([620.0, 500.0])
    straight = laser_point_on_plane(
        plane_from_correspondences(
            body.reshape(-1, 2), image.reshape(-1, 2), CAMERA_MATRIX
        ),
        dot,
        CAMERA_MATRIX,
    )
    # Mirror the image traversal while leaving the body grid as it is: the
    # body frame is now a reflection of the physical board's.
    mirrored = laser_point_on_plane(
        plane_from_correspondences(
            body.reshape(-1, 2), image[:, ::-1, :].reshape(-1, 2), CAMERA_MATRIX
        ),
        dot,
        CAMERA_MATRIX,
    )

    assert np.allclose(straight, mirrored, atol=1e-3)


# ---------- refusals ----------


def test_returns_none_when_there_is_no_board():
    """A blank frame is refused, not answered."""
    assert _detect(np.full((600, 800), 255, np.uint8)) is None


def test_refuses_a_grid_larger_than_the_declared_board():
    """A different board means a different pitch, i.e. a wrong scale.

    The 2025 pool-test board is 24 x 17 where the E4E board is 14 x 10. A
    frame linked to the wrong target would otherwise be fitted at the wrong
    scale and calibrate cleanly — the error reprojection residual cannot see.
    """
    fine_board = _render(cols_sq=25, rows_sq=18)

    assert _detect(fine_board) is None


@pytest.mark.parametrize(
    ("detected", "why"),
    [
        ((9, 20), "one side fits, the other overruns"),
        ((8, 15), "the overrun is a single corner"),
        ((17, 24), "the whole 2025 board"),
    ],
)
def test_a_grid_that_overruns_either_side_is_refused(detected, why):
    """ "Fits inside the declared board" is elementwise, both sides.

    A lexicographic compare of the sorted pairs passes 9 x 20 against a
    10 x 14 board — 9 < 10 decides it and the 20 is never looked at. That is
    not a hypothetical shape: it is a partial view of the 2025 pool board
    (24 x 17), which would then be fitted at the E4E board's 4.2 cm pitch. A
    wrong pitch is a wrong scale, and scale is the one error term reprojection
    residual provably cannot see — so it would calibrate cleanly and measure
    every fish in the dive wrong.

    Asserted on the predicate rather than through an image because rendering a
    board that detects at exactly 9 x 20 is not something a test can pin down;
    the predicate is the part that has to be right.
    """
    assert (
        sut.pitch_per_detected_axis(
            detected[0], detected[1], max_rows=BOARD_ROWS, max_cols=BOARD_COLS,
            pitch_x_m=SQUARE_SIZE_M, pitch_y_m=SQUARE_SIZE_M,
        )
        is None
    ), why


@pytest.mark.parametrize(
    "detected",
    [
        (10, 14),  # the whole board
        (14, 10),  # the whole board, transposed
        (10, 10),  # a sub-grid
        (3, 3),  # the smallest accepted sub-grid
        (10, 3),
    ],
)
def test_a_grid_that_fits_either_way_round_is_accepted(detected):
    """Orientation is free, so either way round fits."""
    assert (
        sut.pitch_per_detected_axis(
            detected[0], detected[1], max_rows=BOARD_ROWS, max_cols=BOARD_COLS,
            pitch_x_m=SQUARE_SIZE_M, pitch_y_m=SQUARE_SIZE_M,
        )
        is not None
    )


def test_accepts_the_declared_board_at_its_own_size():
    """The refusal above must be about the geometry, not about strictness."""
    fine_board = _render(cols_sq=25, rows_sq=18)

    assert _detect(fine_board, max_rows=17, max_cols=24) is not None


@pytest.mark.parametrize("side", [1, 2])
def test_refuses_a_degenerate_grid(side):
    """`solvePnP` accepts near-collinear points and returns confident junk.

    A one- or two-corner-wide strip does not determine a plane, and nothing
    downstream would notice: there is no residual check between here and
    `LaserExtrinsics`.
    """
    strip = _render(cols_sq=side + 1, rows_sq=BOARD_ROWS_SQ)

    assert _detect(strip) is None


# ---------- is the dot actually ON the board? ----------
#
# The calibration assumes the dot lies on the board's plane, and
# `laser_point_on_plane` intersects the camera ray with that plane whether or
# not the physical board reaches that far — a dot that missed and landed on the
# floor behind gets a confident, wrong depth. Nothing downstream can see it:
# `check_fit_self_consistency` compares the fitted ray's reprojection against
# the 2-D dot line, and that line is the laser's epipolar line, identical
# either way. The dots are right; only the depths are wrong.
#
# So the frame has to answer it, and the detected corners already do. This is
# the checkerboard's equivalent of the `Slate, Laser on slate` marker, minus
# the human.


def _accepts(img, point) -> bool:
    """Would this frame contribute an observation with the dot at `point`?

    False covers both refusals — no trustworthy board detected, or a board
    whose hull does not contain the dot. They are the same outcome for the
    caller: the frame is dropped rather than fitted.
    """
    detected = _detect(img)
    if detected is None:
        return False
    return point_in_laser_region(point[0], point[1], sut.board_hull(detected))


def test_board_hull_is_the_grid_outline_in_order():
    """Four vertices, traversing the quad — `point_in_laser_region` is a
    convex same-side test, so a scrambled order would admit the wrong set."""
    detected = _detect(_warped(_render()))
    hull = sut.board_hull(detected)

    assert len(hull) == 4
    grid = detected.image_points.reshape(detected.rows, detected.cols, 2)
    for expected, actual in zip(
        [grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]], hull, strict=True
    ):
        assert np.allclose(expected, actual)


def test_a_dot_on_the_board_is_accepted():
    """A dot in the middle of the board lies inside its hull."""
    board = _warped(_render())
    height, width = board.shape[:2]

    assert _accepts(board, (width // 2, height // 2))


def test_a_dot_off_the_board_is_rejected():
    """The case the gate exists for: the laser missed and hit what was behind."""
    board = _warped(_render())

    assert not _accepts(board, (5.0, 5.0))


def test_the_hull_is_inset_by_one_square_and_that_is_deliberate():
    """Interior corners only, so the hull sits one full square inside the board.

    A dot in that outer ring is genuinely on the board and is still refused.
    That is the conservative direction and it is chosen: dilating the hull
    outward to reach the physical edge would also erode the occlusion margin
    the gate depends on by exactly one square, which is what makes a
    two-square occluder detectable. There are 28-113 dotted frames per dive
    against a `MIN_LASER_POINTS` of 2, so frames are the cheap thing here.
    """
    detected = _detect(_render())
    grid = detected.image_points.reshape(detected.rows, detected.cols, 2)
    # One square outward from a corner, along the grid's own diagonal.
    outer = grid[0, 0] + (grid[0, 0] - grid[1, 1]) * 0.5

    assert not point_in_laser_region(
        float(outer[0]), float(outer[1]), sut.board_hull(detected)
    )


@pytest.mark.parametrize("squares", [2, 3, 4, 6])
def test_an_occluder_means_a_dot_on_it_is_refused(squares):
    """The coupling the gate leans on, measured rather than assumed.

    Something occluding the board removes the corners under it, so the
    detector returns a sub-rectangle beside it and the hull no longer covers
    the occluder. Measured against OpenCV 4.13: a 2x2-square occluder collapses
    a 10x14 grid to 10x6, a 4x4 to 5x10, and half the board occluded still
    detects 7x10 and still refuses a dot in the occluded half.

    The boundary is roughly one square: an occluder that small leaves the grid
    intact and a dot on it passes. That needs a sub-4cm object sitting exactly
    where the dot lands, and an off-board point sits off the common laser ray,
    so the fit's own residual is the second line.
    """
    board = _warped(_render())
    height, width = board.shape[:2]
    size = squares * SQUARE_PX
    centre_y, centre_x = height // 2, width // 2
    occluded = board.copy()
    occluded[
        centre_y - size // 2 : centre_y + size // 2,
        centre_x - size // 2 : centre_x + size // 2,
    ] = 128

    assert not _accepts(occluded, (centre_x, centre_y))


@pytest.mark.parametrize("radius", [4, 8, 12])
def test_a_saturated_dot_never_costs_us_its_own_frame(radius):
    """The gate must not eat the very observations it exists to protect.

    A bright dot up to 12 px across a ~32 px corner spacing leaves a true
    lattice that still spans it, and the frame is kept.
    """
    board = _warped(_render())
    height, width = board.shape[:2]
    centre = (width // 2, height // 2)
    dotted = board.copy()
    cv2.circle(dotted, centre, radius, 255, -1)

    assert _accepts(dotted, centre)


@pytest.mark.parametrize("radius", [16, 20])
def test_a_larger_dot_makes_the_detector_skip_rows_and_the_frame_is_refused(radius):
    """**The 2x scale error, provoked for real.** At 16 px and up the detector
    falls back to a 7 x 10 grid of the 10 x 14 board -- and it is every other
    row: adjacent detected rows are ~66 px apart where the squares are ~33.
    Labelled one pitch apart, that halves the board's distance along that
    axis. The services port (and this test, before it) accepted it as
    "a sub-grid that still spans the dot". Its homography is perfect; its
    checker parity is ~0.13, and the frame is refused.
    """
    board = _warped(_render())
    height, width = board.shape[:2]
    centre = (width // 2, height // 2)
    dotted = board.copy()
    cv2.circle(dotted, centre, radius, 255, -1)

    cv2.setRNGSeed(0)
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        dotted, (3, 3), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER
    )
    assert found
    rows, cols = int(meta.shape[0]), int(meta.shape[1])
    grid = np.asarray(corners, np.float64).reshape(rows, cols, 2)
    step_rows = np.median(np.linalg.norm(np.diff(grid, axis=0), axis=2))
    step_cols = np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2))
    assert step_rows / step_cols == pytest.approx(2.0, rel=0.1), "every other row"

    assert _detect(dotted) is None


def test_a_mis_latticed_detection_is_refused():
    """The silent 2x scale error, pinned directly.

    Corners two squares apart, labelled one square apart in `body_points`,
    place the board at half its true distance — halving every depth and every
    length downstream. Nothing else catches it: `solvePnP` succeeds, the plane
    is perfectly self-consistent, and `check_fit_self_consistency` compares
    2-D dots that are unaffected by the error.

    Constructed rather than provoked through the detector, because the exact
    occluder that elicits it sits on a stability boundary and would make this
    a flaky test rather than a statement about the guard.
    """
    grid = np.stack(
        np.meshgrid(np.arange(BOARD_COLS), np.arange(BOARD_ROWS)), axis=-1
    ).astype(np.float64)
    body = (grid * SQUARE_SIZE_M).reshape(-1, 2)
    # The image says two squares per step; the body points say one.
    image = (grid * SQUARE_PX * 2.0).reshape(-1, 2)

    spacing = SQUARE_PX * 2.0
    residual = sut.grid_residual_px(body, image)

    # A uniformly scaled grid IS still a perfect grid, so the homography
    # absorbs it — which is why `checker_parity` exists (tested below on a
    # real image).
    assert residual < sut.MAX_GRID_RESIDUAL_FRACTION * spacing

    # What it does catch is the incoherent case: corners that are not a
    # consistent lattice at all, which is what the detector actually returned
    # on a heavily occluded board (median spacing 73 px where every genuine
    # detection sat at 32 px).
    scrambled = image.copy()
    scrambled[::3] += SQUARE_PX * 0.8
    assert sut.grid_residual_px(body, scrambled) > (
        sut.MAX_GRID_RESIDUAL_FRACTION * spacing
    )


def test_the_quality_gate_keeps_every_genuine_detection():
    """The gate must not cost us frames it was not aimed at.

    Measured across the shapes the corpus actually produces: full board, seen
    at an angle, cropped, and occluded. The worst corner sits at <=1.5 % of a
    square where the threshold is 5 %.
    """
    board = _warped(_render())
    height, width = board.shape[:2]
    occluded = board.copy()
    occluded[
        height // 2 - SQUARE_PX : height // 2 + SQUARE_PX,
        width // 2 - SQUARE_PX : width // 2 + SQUARE_PX,
    ] = 128

    for label, img in (
        ("flat", _render()),
        ("warped", board),
        ("cropped", board[:, : int(width * 0.66)].copy()),
        ("occluded", occluded),
    ):
        detected = _detect(img)
        assert detected is not None, label
        spacing = sut.median_corner_spacing(
            detected.image_points, detected.rows, detected.cols
        )
        residual = sut.grid_residual_px(detected.body_points, detected.image_points)
        # The worst corner, not the median: an occluder's edge corner reaches
        # ~1.5 % of the spacing, against the 5 % threshold.
        assert residual < 0.02 * spacing, (label, residual, spacing)


# ---------- v2: per-axis pitch (was test_checkerboard_pitch.py) ----------
#
# v2: a calibration target's pitch is per axis, and each axis gets its own.
#
# v1 modelled a board with one `square_size_m`. PLAN.md §4.3 asks for per-axis
# pitch, because the E4E board is ~0.3-0.7 % anisotropic (wuwnet, §2.7) and one
# scalar hides it; `calibration_targets` stores `pitch_x_m` and `pitch_y_m`.
# Pitch is scale, and scale is the one error the rest of the pipeline provably
# cannot see (`fishsense_core.checkerboard`'s module docstring), so the two must
# land on the right axes.
#
# The convention (the target's, as seeded in the API's `test_reference_data`:
# the E4E board is `interior_rows=10, interior_cols=14, pitch_x_m=0.04223,
# pitch_y_m=0.04211`): `pitch_x_m` is the spacing between adjacent corners
# **along a row** (column to column, across the `interior_cols` axis) and
# `pitch_y_m` the spacing **down a column** (row to row).
#
# The detector's grid orientation is free (v1: the same board comes back as
# 10 x 14 or 14 x 10), so which detected axis is which declared axis has to be
# read from the detected shape. When the shape cannot say -- a sub-grid that
# fits the board either way round -- and the two pitches differ, the frame is
# refused rather than guessed: frames are the cheap thing (28-133 per dive
# against `MIN_LASER_POINTS` of 2). With equal pitches (every migrated v1
# target) nothing changes from v1.

PITCH_X, PITCH_Y = 0.04223, 0.04211


def _detect_anisotropic(img, **overrides):
    kwargs = {
        "max_rows": BOARD_ROWS,
        "max_cols": BOARD_COLS,
        "pitch_x_m": PITCH_X,
        "pitch_y_m": PITCH_Y,
    }
    kwargs.update(overrides)
    return sut.detect_checkerboard(img, **kwargs)


def _steps(detected):
    grid = detected.body_points.reshape(detected.rows, detected.cols, 2)
    along_cols = np.diff(grid[0, :, 0])  # col index -> x
    down_rows = np.diff(grid[:, 0, 1])  # row index -> y
    return along_cols, down_rows


def test_the_long_axis_of_the_e4e_board_takes_pitch_x():
    """The board's 14-corner axis is its `interior_cols` axis, whichever way
    round the detector indexed it, so that axis is spaced at `pitch_x_m`."""
    detected = _detect_anisotropic(_warped(_render()))
    assert detected is not None
    along_cols, down_rows = _steps(detected)

    if detected.cols == BOARD_COLS:
        assert np.allclose(along_cols, PITCH_X) and np.allclose(down_rows, PITCH_Y)
    else:
        assert detected.rows == BOARD_COLS
        assert np.allclose(along_cols, PITCH_Y) and np.allclose(down_rows, PITCH_X)


@pytest.mark.parametrize(
    ("detected", "expected"),
    [
        ((10, 14), (PITCH_X, PITCH_Y)),  # the declared orientation
        ((14, 10), (PITCH_Y, PITCH_X)),  # transposed: cols are declared rows
        ((9, 12), (PITCH_X, PITCH_Y)),  # a sub-grid only the declared way fits
        ((12, 9), (PITCH_Y, PITCH_X)),  # ... only the transposed way fits
    ],
)
def test_the_detected_shape_says_which_pitch_is_which(detected, expected):
    """`(col_pitch, row_pitch)` for the detected grid."""
    assert sut.pitch_per_detected_axis(
        *detected,
        max_rows=BOARD_ROWS,
        max_cols=BOARD_COLS,
        pitch_x_m=PITCH_X,
        pitch_y_m=PITCH_Y,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("detected", [(10, 10), (3, 3), (10, 3), (5, 8)])
def test_an_ambiguous_sub_grid_is_refused_when_the_pitches_differ(detected):
    """A shape that fits the board either way round cannot say which axis is
    which, and a guess is a scale error of the anisotropy."""
    assert (
        sut.pitch_per_detected_axis(
            *detected,
            max_rows=BOARD_ROWS,
            max_cols=BOARD_COLS,
            pitch_x_m=PITCH_X,
            pitch_y_m=PITCH_Y,
        )
        is None
    )


@pytest.mark.parametrize("detected", [(10, 10), (3, 3), (10, 14), (14, 10)])
def test_equal_pitches_are_v1s_behaviour(detected):
    """Every migrated target: one pitch on both axes, any orientation."""
    assert sut.pitch_per_detected_axis(
        *detected,
        max_rows=BOARD_ROWS,
        max_cols=BOARD_COLS,
        pitch_x_m=0.0254,
        pitch_y_m=0.0254,
    ) == (0.0254, 0.0254)


def test_a_grid_that_fits_neither_way_is_refused():
    """A grid larger than the board either way round has no pitch."""
    assert (
        sut.pitch_per_detected_axis(
            17,
            24,
            max_rows=BOARD_ROWS,
            max_cols=BOARD_COLS,
            pitch_x_m=0.0254,
            pitch_y_m=0.0254,
        )
        is None
    )


def test_an_ambiguous_partial_view_is_dropped_only_when_the_pitches_differ():
    """Through the detector: a cropped board detects as a sub-grid that fits
    either way; anisotropic, the frame is dropped; isotropic, it is kept."""
    board = _warped(_render())
    cropped = board[:, : int(board.shape[1] * 0.66)].copy()
    isotropic = _detect_anisotropic(cropped, pitch_x_m=0.0254, pitch_y_m=0.0254)
    assert isotropic is not None
    short, long = sorted((isotropic.rows, isotropic.cols))
    assert short <= min(BOARD_ROWS, BOARD_COLS) and long <= min(
        BOARD_ROWS, BOARD_COLS
    ), "the crop must detect as an ambiguous sub-grid for this test to mean it"

    assert _detect_anisotropic(cropped) is None


# ---------- golden: parity with the fishsense-services original ----------
#
# Recorded by running fishsense-services v0.48.0's `checkerboard.detection`
# and this module side by side on the same synthetic frames (OpenCV 5.0.0,
# the locked `opencv-python-headless`): identical grid shapes, identical body
# points, and corner coordinates equal to 0.0 px. Pinned here so a change in
# either the port or the locked OpenCV shows up as a diff, not a drift.
#
# Only unoccluded frames are pinned. `findChessboardCornersSB` draws from
# OpenCV's global RNG, so on an occluded board the sub-grid it settles on can
# change from call to call (a 4x4-square occluder: 4x10, 5x3 or 3x3 across
# 20 calls); the full, warped and cropped boards are stable.

_GOLDEN = {
    # name: (rows, cols, first corner, last corner, column sums, median spacing)
    "flat": (
        10,
        14,
        (639.5234985351562, 479.51416015625),
        (119.51322937011719, 119.51603698730469),
        (53131.54863739014, 41931.482261657715),
        40.001534543126226,
    ),
    "warped": (
        14,
        10,
        (151.7974090576172, 427.54443359375),
        (596.3683471679688, 180.62864685058594),
        (52526.112396240234, 42418.78079223633),
        32.05706846116085,
    ),
    "cropped": (
        10,
        10,
        (466.7195739746094, 167.38645935058594),
        (151.7845916748047, 427.551025390625),
        (30976.546112060547, 29669.245315551758),
        32.14281053902681,
    ),
}


def _golden_frame(name):
    if name == "flat":
        return _render()
    board = _warped(_render())
    if name == "cropped":
        return board[:, : int(board.shape[1] * 0.66)].copy()
    return board


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_detection_matches_the_services_original(name):
    """Grid shape, corners and spacing as fishsense-services produced them."""
    rows, cols, first, last, sums, spacing = _GOLDEN[name]

    detected = _detect(_golden_frame(name))

    assert detected is not None
    assert (detected.rows, detected.cols) == (rows, cols)
    np.testing.assert_allclose(detected.image_points[0], first, atol=1e-4)
    np.testing.assert_allclose(detected.image_points[-1], last, atol=1e-4)
    np.testing.assert_allclose(detected.image_points.sum(axis=0), sums, atol=1e-2)
    assert sut.median_corner_spacing(
        detected.image_points, rows, cols
    ) == pytest.approx(spacing, abs=1e-4)


def test_anisotropic_golden_puts_pitch_y_on_the_detected_columns():
    """The warped board detects transposed (14 x 10), so its columns are the
    declared rows and are spaced at `pitch_y_m` -- as services computed it."""
    detected = _detect_anisotropic(_golden_frame("warped"))

    assert detected is not None
    assert (detected.rows, detected.cols) == (14, 10)
    along_cols, down_rows = _steps(detected)
    np.testing.assert_allclose(along_cols, PITCH_Y)
    np.testing.assert_allclose(down_rows, PITCH_X)
    np.testing.assert_allclose(
        detected.image_points[0], _GOLDEN["warped"][2], atol=1e-4
    )


# ---------- the review's fixes ----------


def _occluded(squares: int):
    board = _warped(_render())
    height, width = board.shape[:2]
    half = squares * SQUARE_PX // 2
    board[height // 2 - half : height // 2 + half, width // 2 - half : width // 2 + half] = 128
    return board


def _shape(detected):
    return None if detected is None else (detected.rows, detected.cols)


@pytest.mark.parametrize("squares", [2, 3, 4, 6])
def test_the_same_frame_gives_the_same_detection_every_time(squares):
    """OpenCV's detector draws from its random generator on an occluded board;
    unseeded, a 4-square occluder came back as 4x10, 5x3 or 3x3. Here the
    generator is scrambled between calls and the answer must not move."""
    img = _occluded(squares)
    first = _detect(img)
    for scramble in range(1, 8):
        cv2.setRNGSeed(scramble * 7919)
        cv2.randu(np.empty((64, 64), np.uint8), 0, 255)  # advance it further
        again = _detect(img)
        assert _shape(again) == _shape(first)
        if first is not None:
            np.testing.assert_array_equal(again.image_points, first.image_points)


def test_the_seed_is_a_parameter():
    img = _occluded(4)
    assert _shape(_detect(img, seed=0)) == _shape(_detect(img, seed=0))


def test_a_lattice_at_twice_the_pitch_is_refused_by_its_parity():
    """The 2x scale error the homography cannot see. Every other corner of a
    genuine detection, labelled one pitch apart, is a perfect grid (its worst
    residual is ~0.02 px) but its cell centres are corner junctions."""
    img = _render()
    genuine = _detect(img)
    assert sut.checker_parity(img, genuine.body_points, genuine.image_points,
                              genuine.rows, genuine.cols) > 0.99  # fmt: skip
    grid = genuine.image_points.reshape(genuine.rows, genuine.cols, 2)[::2, ::2]
    rows, cols = grid.shape[:2]
    body = (
        np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), axis=-1).reshape(-1, 2)
        * SQUARE_SIZE_M
    ).astype(np.float64)
    image = grid.reshape(-1, 2)
    assert sut.grid_residual_px(body, image) < 0.1, "a perfect grid to the homography"
    assert sut.checker_parity(img, body, image, rows, cols) < sut.MIN_CHECKER_PARITY


@pytest.mark.parametrize("squares", [0, 2, 3, 4, 6])
def test_every_genuine_detection_clears_both_gates(squares):
    img = _warped(_render()) if squares == 0 else _occluded(squares)
    detected = _detect(img)
    assert detected is not None
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert sut.checker_parity(gray, detected.body_points, detected.image_points,
                              detected.rows, detected.cols) > 0.95  # fmt: skip
    spacing = sut.median_corner_spacing(detected.image_points, detected.rows, detected.cols)
    assert sut.grid_residual_px(detected.body_points, detected.image_points) < 0.02 * spacing


def test_the_residual_is_the_worst_corners_not_the_median():
    """A minority of corners a full square off: the median hid it."""
    img = _render()
    genuine = _detect(img)
    image = genuine.image_points.copy()
    image[:10] += SQUARE_PX  # 10 of 140 corners latticed one square off
    spacing = sut.median_corner_spacing(genuine.image_points, genuine.rows, genuine.cols)
    assert sut.grid_residual_px(genuine.body_points, image) > sut.MAX_GRID_RESIDUAL_FRACTION * spacing


@pytest.mark.parametrize(
    ("pitch_y", "fits"),
    [(0.04223 * (1 + 4e-4), True), (0.04223 * (1 + 2.8e-3), False)],
    ids=["0.04%-same", "0.28%-different"],
)
def test_pitches_are_equal_within_a_tolerance(pitch_y, fits):
    """An ambiguous 10x10 sub-grid: a 0.04 % anisotropy is one square size
    (no frame dropped), the E4E board's 0.28 % is not (no guess made)."""
    got = sut.pitch_per_detected_axis(
        10, 10, max_rows=10, max_cols=14, pitch_x_m=0.04223, pitch_y_m=pitch_y
    )
    assert (got is not None) == fits


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_grey_bgr_and_bgra_are_accepted(channels):
    grey = _render()
    img = {
        1: grey[:, :, None],
        3: cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR),
        4: cv2.cvtColor(grey, cv2.COLOR_GRAY2BGRA),
    }[channels]
    assert _shape(_detect(img)) == _shape(_detect(grey))


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_a_non_8_bit_image_is_a_clear_error(dtype):
    """It used to reach OpenCV and raise cv2.error from inside the detector."""
    with pytest.raises(TypeError, match="uint8"):
        _detect(_render().astype(dtype))


def test_an_unknown_shape_is_a_clear_error():
    with pytest.raises(ValueError, match="grey, BGR or BGRA"):
        _detect(np.zeros((40, 40, 2), np.uint8))


# ---------- regressions from 4.3.0 on real frames ----------
#
# fishsense-services ran 4.3.0 over all 745 production checkerboard frames
# (2026-10-24, 17 missions, the E4E 10 x 14 board). Two ways it dropped good
# boards:
#  * 11 full boards refused for ONE corner a laser dot had dragged 9-11 px,
#    0.18-0.20 of the spacing, every other corner under 1 px (e.g. mission
#    509 capture 133042: worst 0.201, second-worst 0.018);
#  * 18 boards refused because seed 0 drew a partial grid while another
#    seed found the whole board.
# The tests below put those shapes through the public API.


def _real_corners(img):
    """The detector's own corners and grid on a rendered board."""
    cv2.setRNGSeed(0)
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        img, sut._REQUESTED_PATTERN, sut._DETECTION_FLAGS  # pylint: disable=protected-access
    )
    assert found
    return np.asarray(corners, np.float64).reshape(-1, 2), meta.shape[:2]


def _detector_returning(monkeypatch, by_seed):
    """OpenCV's detector, answering per seed: by_seed(seed) -> (corners,
    shape), or None for "no grid found"."""
    state = {"seed": None}
    real_seed = cv2.setRNGSeed

    def set_seed(seed):
        state["seed"] = seed
        real_seed(seed)

    def find(_gray, _pattern, _flags):
        answer = by_seed(state["seed"])
        if answer is None:
            return False, None, None
        corners, shape = answer
        return True, corners.reshape(-1, 1, 2).astype(np.float32), np.zeros(shape)

    monkeypatch.setattr(sut.cv2, "setRNGSeed", set_seed)
    monkeypatch.setattr(sut.cv2, "findChessboardCornersSBWithMeta", find)


def _subgrid(corners, shape, rows, cols):
    """The top-left rows x cols corners of a detected grid."""
    return corners.reshape(*shape, 2)[:rows, :cols].reshape(-1, 2), (rows, cols)


def test_a_board_with_one_corner_dragged_by_the_dot_is_kept_and_repaired(monkeypatch):
    """4.3.0 refused it on the worst corner. The dragged corner is replaced by
    where the other corners put it, so the pose never sees the dragged
    position and the shape stays the full grid."""
    img = _render()
    corners, shape = _real_corners(img)
    spacing = sut.median_corner_spacing(corners, *shape)
    dragged = corners.copy()
    dragged[37] += [0.2 * spacing, 0.0]  # mission 509's 0.201
    _detector_returning(monkeypatch, lambda seed: (dragged, shape))

    detected = _detect(img)

    assert detected is not None and detected.repaired == (37,)
    assert detected.image_points.shape == (shape[0] * shape[1], 2)
    assert np.linalg.norm(detected.image_points[37] - corners[37]) < 0.05 * spacing
    others = np.arange(len(corners)) != 37
    np.testing.assert_allclose(detected.image_points[others], dragged[others].astype(np.float32))


def test_a_board_seed_zero_misses_is_found_by_the_next_seed(monkeypatch):
    """4.3.0 tried seed 0 only, and on 18 production frames that draw was a
    partial grid the gates refuse while another seed found the whole board."""
    img = _render()
    corners, shape = _real_corners(img)
    spacing = sut.median_corner_spacing(corners, *shape)
    refused = corners.copy()
    refused[[3, 40, 90]] += [0.3 * spacing, 0.0]  # a grid the gates refuse
    _detector_returning(monkeypatch, lambda seed: (refused, shape) if seed == 0 else (corners, shape))

    detected = _detect(img)

    assert detected is not None and (detected.rows, detected.cols) == shape


def test_a_partial_passing_grid_loses_to_a_full_board_on_a_later_seed(monkeypatch):
    """The largest passing grid wins: a 6 x 14 sub-grid shrinks the outline
    and would refuse a dot in the missing rows as off the board."""
    img = _render()
    corners, shape = _real_corners(img)
    part = _subgrid(corners, shape, 6, shape[1])
    _detector_returning(monkeypatch, lambda seed: part if seed == 0 else (corners, shape))
    assert (_detect(img).rows, _detect(img).cols) == shape


def test_a_full_board_on_the_first_seed_ends_the_search(monkeypatch):
    img = _render()
    corners, shape = _real_corners(img)
    tried = []

    def by_seed(seed):
        tried.append(seed)
        return corners, shape

    _detector_returning(monkeypatch, by_seed)
    assert _detect(img) is not None and tried == [0]


def test_no_grid_on_the_first_seed_ends_the_search(monkeypatch):
    """A frame with no board must not cost eight exhaustive detections."""
    img = _render()
    tried = []
    _detector_returning(monkeypatch, lambda seed: tried.append(seed))
    assert _detect(img) is None and tried == [0]


def test_the_earliest_seed_wins_a_tie(monkeypatch):
    img = _render()
    corners, shape = _real_corners(img)
    first = _subgrid(corners, shape, 6, shape[1])
    later = (corners.reshape(*shape, 2)[shape[0] - 6 :].reshape(-1, 2), (6, shape[1]))
    _detector_returning(monkeypatch, lambda seed: first if seed == 0 else later)
    np.testing.assert_allclose(_detect(img).image_points, first[0].astype(np.float32))


def test_a_board_with_a_corner_a_square_off_is_still_refused(monkeypatch):
    """Tolerating a dragged corner must not tolerate a lattice error."""
    img = _render()
    corners, shape = _real_corners(img)
    spacing = sut.median_corner_spacing(corners, *shape)
    wrong = corners.copy()
    wrong[37] += [spacing, 0.0]
    _detector_returning(monkeypatch, lambda seed: (wrong, shape))
    assert _detect(img) is None


def test_three_displaced_corners_are_still_refused(monkeypatch):
    img = _render()
    corners, shape = _real_corners(img)
    spacing = sut.median_corner_spacing(corners, *shape)
    off = corners.copy()
    off[[10, 50, 100]] += [0.2 * spacing, 0.0]
    _detector_returning(monkeypatch, lambda seed: (off, shape))
    assert _detect(img) is None


def test_a_small_grid_cannot_pass_on_repairs(monkeypatch):
    """Two of a 3 x 3 grid's nine corners off is 22 % of it, not a dragged
    corner: a grid needs CORNERS_PER_REPAIR corners for each repair."""
    img = _render()
    corners, shape = _real_corners(img)
    small, small_shape = _subgrid(corners, shape, 3, 3)
    spacing = sut.median_corner_spacing(small, *small_shape)
    small = small.copy()
    small[[1, 7]] += [0.3 * spacing, 0.0]
    _detector_returning(monkeypatch, lambda seed: (small, small_shape))
    assert _detect(img) is None
    assert sut.CORNERS_PER_REPAIR > 9


def test_the_corner_repaired_is_the_dragged_one_and_lands_on_the_lattice():
    """The dragged corner, here the grid's own corner where the least-squares
    fit leans toward it most, is the one picked by its deleted residual and
    is put back where the lattice says it is."""
    col, row = np.meshgrid(np.arange(10), np.arange(10))
    body = np.stack([col.ravel(), row.ravel()], axis=1).astype(np.float64) * SQUARE_SIZE_M
    spacing = 40.0
    image = np.stack([col.ravel(), row.ravel()], axis=1).astype(np.float64) * spacing + 200.0
    truth = image.copy()
    image[0] += [0.3 * spacing, 0.3 * spacing]  # the grid's own corner
    repaired, which = sut._repair_displaced_corners(body, image, spacing)  # pylint: disable=protected-access
    assert which == (0,)
    np.testing.assert_allclose(repaired[0], truth[0], atol=0.05 * spacing)


def test_an_outline_corner_is_repaired_onto_the_lattice(monkeypatch):
    """The hull is the lattice's: a dot dragging the grid's own corner must
    not bend the quad the on-board test uses."""
    img = _render()
    corners, shape = _real_corners(img)
    spacing = sut.median_corner_spacing(corners, *shape)
    dragged = corners.copy()
    dragged[0] += [0.3 * spacing, 0.3 * spacing]
    _detector_returning(monkeypatch, lambda seed: (dragged, shape))

    detected = _detect(img)

    assert detected.repaired == (0,)
    assert np.linalg.norm(np.array(sut.board_hull(detected)[0]) - corners[0]) < 0.05 * spacing


def test_a_twice_the_pitch_lattice_is_still_refused_on_every_seed(monkeypatch):
    """Mission 522 capture 134474's shape: every other column."""
    img = _render()
    corners, (rows, cols) = _real_corners(img)
    every_other = corners.reshape(rows, cols, 2)[:, ::2].reshape(-1, 2)
    _detector_returning(monkeypatch, lambda seed: (every_other, (rows, (cols + 1) // 2)))
    assert _detect(img) is None


def test_a_frame_whose_first_seed_gives_the_full_board_keeps_that_grid():
    """Every frame 4.3.0 accepted with the whole board keeps its grid."""
    img = _render()
    np.testing.assert_array_equal(_detect(img).image_points, _detect(img, seed=0).image_points)


def test_seed_still_pins_a_single_attempt(monkeypatch):
    img = _render()
    tried = []
    _detector_returning(monkeypatch, lambda seed: tried.append(seed))
    assert _detect(img, seed=5) is None
    assert tried == [5]


@pytest.mark.parametrize(
    "kwargs, match",
    [({"seeds": ()}, "empty"), ({"seeds": (3, 4), "seed": 0}, "not both")],
    ids=["empty", "both"],
)
def test_the_seed_arguments_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _detect(_render(), **kwargs)


def test_a_one_shot_iterable_of_seeds_is_read_once(monkeypatch):
    img = _render()
    corners, shape = _real_corners(img)
    _detector_returning(monkeypatch, lambda seed: None if seed == 0 else (corners, shape))
    seeds = iter((1, 2))
    assert _detect(img, seeds=seeds) is not None


def test_a_clean_board_repairs_nothing():
    assert _detect(_render()).repaired == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_points": np.zeros((139, 2))},
        {"body_points": np.zeros((139, 2))},
        {"repaired": (140,)},
    ],
    ids=["image-short", "body-short", "repaired-out-of-range"],
)
def test_an_inconsistent_detection_cannot_be_built(kwargs):
    fields = {
        "rows": 10, "cols": 14,
        "body_points": np.zeros((140, 2)), "image_points": np.zeros((140, 2)),
    }  # fmt: skip
    fields.update(kwargs)
    with pytest.raises(ValueError):
        sut.DetectedCheckerboard(**fields)
