"""Find a checkerboard's interior corners and pair them with board coordinates.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/src/fishsense_services_processor/checkerboard/detection.py``
(itself from fishsense-lite@77e8f8e5
``services/fishsense-data-processing-workflow-worker/src/``
``fishsense_data_processing_workflow_worker/checkerboard_detection.py``).

This produces the checkerboard's ``(body_points, image_points)`` — the same
pair a dive slate's template points and hand-clicked corners produce — from
which :func:`fishsense_core.plane.plane_from_correspondences` recovers the
board plane. Everything after that — the pose, the plane, the ray-plane
intersection (:func:`fishsense_core.plane.laser_point_on_plane`), the laser fit
(:func:`fishsense_core.laser.calibrate_laser`) — already knows nothing about
what the target was. In fishsense-services this fed stage 13
(``calibration_geometry``) in place of a ``MissionSlate``.

**The detector is asked for a small grid, and told the real one.**
`findChessboardCornersSB` with `CALIB_CB_LARGER` returns whatever grid it
actually found, and reports its shape in the metadata; asking for the nominal
board instead is worse in both directions, measured against OpenCV 4.13:

* a partial view fails outright. A board with its right third out of frame
  returns nothing for the exact `patternSize` and a clean 10 x 10 sub-grid for
  a small one — and occlusion, glare and the frame edge cut the board down
  often enough that the survey saw sub-grids on many frames.
* a wrong-but-plausible `patternSize` also fails outright, so an operator who
  linked the wrong target would see "no board here" rather than a diagnosis.

Reading the shape back also *is* the safety property. A nominal full-board
grid paired against a partial detection mis-pairs every correspondence, and
`solvePnP` does not refuse that: it returns success and a confident,
meaningless pose. There is no residual check anywhere between here and
`LaserExtrinsics`, so nothing downstream would notice.

**Orientation is free, and deliberately unconstrained.** The detected grid
comes back transposed or mirrored depending on how the board sits in frame
(the survey saw the same board reported as both 10 x 14 and 14 x 10). It does
not matter: the board is a symmetric grid, so its indexing is only defined up
to the symmetries of a rectangle, a mirrored assignment is realised by a real
rotation (the board flipped about an in-plane axis) which merely flips the
plane normal, and the ray-plane intersection divides one normal by the other
so the sign cancels. Only the plane reaches the answer — never the in-plane
pose — which is why the classic checkerboard headaches are all irrelevant
here. ``tests/test_checkerboard.py`` pins that.

**Detect on the RECTIFIED image, at full resolution.** `solvePnP` is called
with zero distortion, which is only correct because the pipeline feeds it
undistorted imagery; raw pixels give a plausible, slightly wrong pose and no
error. Downscaling is its own hazard — at half scale the 2023 board came back
one column short, and a silently smaller grid is a mis-pairing risk, not just
fewer points.

v2 change: **the pitch is per axis** (`calibration_targets.pitch_x_m`,
`pitch_y_m`; fishsense-services PLAN.md §4.3). `pitch_x_m` spaces corners along a declared row
(across the `interior_cols` axis), `pitch_y_m` down a column. Since the
detected grid's orientation is free, `pitch_per_detected_axis` reads which is
which from the detected shape, and refuses a frame whose shape fits the board
either way round when the two pitches differ: a guess there is a scale error
the size of the anisotropy, and nothing downstream could see it. Equal pitches
(every target migrated from v1) behave exactly as v1's single square size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# cv2 is a C extension pylint cannot introspect.
# pylint: disable=no-member

__all__ = [
    "DetectedCheckerboard",
    "MAX_GRID_RESIDUAL_FRACTION",
    "MIN_GRID_SIDE",
    "board_hull",
    "detect_checkerboard",
    "grid_residual_px",
    "median_corner_spacing",
    "pitch_per_detected_axis",
]


#: Smallest grid side accepted, in interior corners.
#:
#: A one- or two-corner-wide strip is near-collinear and does not determine a
#: plane — and `solvePnP` will not say so. Six coincident points return
#: *success* and a confident, meaningless pose, so an automatic producer has
#: to gate on its own detection quality because nothing after it will. Three
#: is the smallest grid that spans two independent in-plane directions with a
#: point to spare in each; every sub-grid the corpus survey returned was
#: comfortably above it (the smallest was 6 x 14).
MIN_GRID_SIDE = 3

#: How far the detected corners may sit from a perfect grid before the
#: detection is refused, as a fraction of the median corner spacing.
#:
#: A regular planar grid imaged by a pinhole camera maps to the image by a
#: HOMOGRAPHY, exactly — so the residual of the best-fit homography is a direct
#: test of "is this actually a grid?", scale-free and independent of how far
#: away the board is.
#:
#: **This is not hygiene, it catches a silent 2x scale error.** On a heavily
#: occluded board the detector can return corners that are TWO squares apart
#: while `body_points` label them one apart — measured: a genuine detection has
#: a median corner spacing of 32 px, and one of these came back at 73 px. The
#: pose then places the board at half its true distance, every depth halves,
#: every length halves, and nothing downstream can see it: the plane is
#: perfectly self-consistent, `solvePnP` succeeds, and
#: `check_fit_self_consistency` compares 2-D dots that are unaffected.
#:
#: Measured on synthetic boards (flat, warped, cropped, and five occluder
#: sizes): every genuine detection lands at 0.011-0.030 px, i.e. <=0.1% of a
#: square, while the mis-latticed one lands at 12.155 px (16.7%). The gap is
#: three orders of magnitude, so this threshold sits far from both edges —
#: ~50x above the worst genuine case and ~3x below the bad one. Re-measure on
#: real underwater frames before tightening it; blur and backscatter will
#: raise the genuine floor, and the safe direction here is to drop frames
#: (there are 28-133 per calibration dive against `MIN_LASER_POINTS` of 2).
MAX_GRID_RESIDUAL_FRACTION = 0.05

#: The grid asked for. See the module docstring — the real one is read back
#: from the detector's metadata, so this only has to be small enough not to
#: exclude a heavily-cropped view.
_REQUESTED_PATTERN = (MIN_GRID_SIDE, MIN_GRID_SIDE)

#: `EXHAUSTIVE` raises the detection rate, `ACCURACY` upsamples for sub-pixel
#: precision (these corners are the whole input to the pose), and `LARGER`
#: is what lets the detector return a grid other than the one requested.
_DETECTION_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER


@dataclass(frozen=True)
class DetectedCheckerboard:
    """A detected grid and its correspondences, index-aligned.

    `body_points` are board coordinates in **metres**, `image_points` the
    rectified pixels they landed on, both `(rows * cols, 2)` in the detector's
    own row-major order. Ordering is the contract `solvePnP` depends on: it
    pairs them purely by position.
    """

    rows: int
    cols: int
    body_points: np.ndarray
    image_points: np.ndarray


def grid_residual_px(body_points: np.ndarray, image_points: np.ndarray) -> float:
    """Median reprojection error of the corners under their best homography.

    Zero for a real grid, large for corners that only look like one. See
    `MAX_GRID_RESIDUAL_FRACTION` for why this is load-bearing rather than
    tidy.

    `method=0` is the plain least-squares fit, deliberately not RANSAC: a
    robust fit would discard the very corners that reveal the grid is wrong.
    """
    body = np.asarray(body_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    matrix, _ = cv2.findHomography(body, image, method=0)
    if matrix is None:
        return float("inf")

    projected = (matrix @ np.hstack([body, np.ones((len(body), 1))]).T).T
    scale = projected[:, 2:3]
    if not np.all(np.isfinite(scale)) or np.any(scale == 0):
        return float("inf")
    projected = projected[:, :2] / scale
    return float(np.median(np.linalg.norm(projected - image, axis=1)))


def median_corner_spacing(image_points: np.ndarray, rows: int, cols: int) -> float:
    """Median distance between horizontally adjacent detected corners.

    Public because the lattice-verification stage reports it per frame. It is
    the machine-readable counterpart of the question that stage puts to a
    human: a lattice resolved at twice the true pitch shows twice the corner
    spacing over the same board, so this number moves with the fault even
    though `grid_residual_px` cannot see it. Recomputing it there instead
    would be the same arithmetic in two places, and the two could disagree.
    """
    grid = np.asarray(image_points, dtype=np.float64).reshape(rows, cols, 2)
    return float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))


def board_hull(detected: DetectedCheckerboard) -> list[list[float]]:
    """The detected grid's outline, as a convex quad in draw order.

    Used to answer the question the calibration otherwise just assumes: **was
    the laser dot actually on the board?** `laser_point_on_plane` intersects
    the camera ray with the board's *infinite* plane, so a dot that missed the
    board and landed on whatever was behind it gets a confident, wrong depth,
    and nothing downstream can tell — `check_fit_self_consistency` compares the
    fitted ray's reprojection against the 2-D dot line, which is the laser's
    epipolar line and is identical either way. The dots are right; only the
    depths are wrong. This is the checkerboard's equivalent of the
    `Slate, Laser on slate` marker, minus the human.

    A grid is the projective image of a rectangle, so its convex hull is just
    the quadrilateral of its four extreme corners — no hull algorithm needed.
    Returned in traversal order because `point_in_laser_region` is a
    same-side-of-every-edge test.

    **Inset by one full square from the physical board**, because only interior
    corners are detectable: the outermost one sits a square in from the edge.
    A dot in that ring is on the board and is still refused. That is the
    deliberate direction — dilating outward to reach the true edge would also
    erode, by exactly one square, the margin that makes an occluder detectable
    (a 2x2-square occluder is what collapses a 10x14 grid to 10x6). Frames are
    the cheap thing: 28-133 per calibration dive against `MIN_LASER_POINTS`
    of 2.
    """
    grid = detected.image_points.reshape(detected.rows, detected.cols, 2)
    return [
        [float(x), float(y)]
        for x, y in (grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0])
    ]


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _fits_declared_board(rows: int, cols: int, max_rows: int, max_cols: int) -> bool:
    """Whether a `rows x cols` detection can be part of the declared board.

    Compared as unordered pairs because the detected grid's orientation is
    free (see the module docstring), so a 10 x 14 detection of a 14 x 10 board
    is the same board.

    A detection *larger* than the declared board is refused rather than
    trimmed: it means the frame holds a different board, so the stored pitch
    does not apply, and a wrong pitch is a wrong scale — the one error term
    the reprojection residual provably cannot see. The 2025 pool-test board
    (24 x 17) against the E4E board (14 x 10) is exactly that case.

    **Elementwise, on both sides.** Comparing the sorted pairs with `<=`
    directly is a lexicographic list compare, not a "fits inside" test: it
    accepts 9 x 20 against a 10 x 14 board, because 9 < 10 settles the
    comparison and the 20 is never looked at. A partial view of the 2025 board
    lands in exactly that shape and would then be fitted at the E4E board's
    pitch.
    """
    short, long = sorted((rows, cols))
    max_short, max_long = sorted((max_rows, max_cols))
    return short <= max_short and long <= max_long


def pitch_per_detected_axis(  # pylint: disable=too-many-arguments
    rows: int,
    cols: int,
    *,
    max_rows: int,
    max_cols: int,
    pitch_x_m: float,
    pitch_y_m: float,
) -> tuple[float, float] | None:
    """`(col_pitch, row_pitch)` for a detected `rows x cols` grid, or None.

    v2. The detected grid is the declared one (`max_rows x max_cols`) either
    as declared -- its columns run along a declared row, spaced `pitch_x_m`
    -- or transposed, when its columns are the declared rows. The shape says
    which when only one way round fits the board. When both do (a square
    board, or a sub-grid within the shorter side) it cannot, and a guess is
    only harmless when the two pitches are equal. None when neither
    orientation fits, or the orientation is ambiguous and the pitches differ.
    """
    declared = rows <= max_rows and cols <= max_cols
    transposed = rows <= max_cols and cols <= max_rows
    if pitch_x_m == pitch_y_m:
        return (pitch_x_m, pitch_y_m) if declared or transposed else None
    if declared and not transposed:
        return pitch_x_m, pitch_y_m
    if transposed and not declared:
        return pitch_y_m, pitch_x_m
    return None


def detect_checkerboard(  # pylint: disable=too-many-locals
    image: np.ndarray,
    *,
    max_rows: int,
    max_cols: int,
    pitch_x_m: float,
    pitch_y_m: float,
) -> DetectedCheckerboard | None:
    """Detect the board in `image` and pair its corners with board coordinates.

    `image` must be the **rectified** frame, full resolution, grayscale or
    BGR. `max_rows` / `max_cols` are the declared board's interior corners and
    act as an upper bound, not a target. `pitch_x_m` / `pitch_y_m` are the
    measured grid pitch along a declared row and down a declared column (v1:
    one `square_size_m`).

    Returns None when no usable board is found — no detection, a degenerate
    strip, a grid larger than the declared board, metadata that disagrees
    with the corners returned, or (v2) a grid whose axes cannot be told apart
    on a board whose pitches differ. Every one of those is a case where
    continuing would hand `solvePnP` a correspondence set it would accept and
    answer wrongly, so the caller drops the frame and the dive is fitted from
    the others.
    """
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        _to_grayscale(image), _REQUESTED_PATTERN, _DETECTION_FLAGS
    )
    if not found or corners is None or meta is None:
        return None

    rows, cols = int(meta.shape[0]), int(meta.shape[1])
    if min(rows, cols) < MIN_GRID_SIDE or not _fits_declared_board(
        rows, cols, max_rows, max_cols
    ):
        return None

    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    # Belt and braces on the detector's own contract: if the metadata grid and
    # the corner count ever disagree, the reshape below would pair the wrong
    # points and solvePnP would accept it silently.
    if len(image_points) != rows * cols:
        return None

    pitches = pitch_per_detected_axis(
        rows,
        cols,
        max_rows=max_rows,
        max_cols=max_cols,
        pitch_x_m=float(pitch_x_m),
        pitch_y_m=float(pitch_y_m),
    )
    if pitches is None:
        return None
    col_pitch, row_pitch = pitches

    # Board coordinates on the DETECTED grid, at the measured pitch. Its
    # origin is wherever the detected sub-grid starts and its axes are
    # whichever way round the detector indexed the board — both are in-plane
    # pose, which nothing downstream reads. Which axis takes which pitch is
    # not, and is `pitch_per_detected_axis`'s answer.
    col_index, row_index = np.meshgrid(np.arange(cols), np.arange(rows))
    body_points = np.stack(
        [col_index.ravel() * col_pitch, row_index.ravel() * row_pitch], axis=1
    ).astype(np.float64)

    # Is it actually a grid? Refused here rather than downstream because
    # everything downstream accepts it silently — see
    # `MAX_GRID_RESIDUAL_FRACTION`.
    #
    # `isfinite` first and explicitly: a NaN spacing makes every comparison
    # False, so a bare `spacing <= 0` would ACCEPT it and then divide the
    # threshold by nothing. The safe reading of "no measurable spacing" is
    # refusal.
    spacing = median_corner_spacing(image_points, rows, cols)
    if (
        not math.isfinite(spacing)
        or spacing <= 0
        or grid_residual_px(body_points, image_points)
        > MAX_GRID_RESIDUAL_FRACTION * spacing
    ):
        return None

    return DetectedCheckerboard(
        rows=rows,
        cols=cols,
        body_points=body_points,
        image_points=image_points,
    )
