"""Find a checkerboard's interior corners and pair them with board coordinates.

Ported from fishsense-services v0.48.0
``services/fishsense-services-processor/src/fishsense_services_processor/checkerboard/detection.py``
(itself from fishsense-lite@77e8f8e5
``services/fishsense-data-processing-workflow-worker/src/``
``fishsense_data_processing_workflow_worker/checkerboard_detection.py``).

This produces the checkerboard's ``(body_points, image_points)`` — the same
pair a dive slate's template points and hand-clicked corners produce — from
which ``fishsense_core.plane.plane_from_correspondences`` recovers the board
plane (that function arrives with the plane PR, #95, so merge it first).
Everything after that — the pose, the plane, the ray-plane intersection
(:func:`fishsense_core.plane.laser_point_on_plane`), the laser fit
(:func:`fishsense_core.laser.calibrate_laser`) — already knows nothing about
what the target was.

**Determinism.** On an occluded board ``findChessboardCornersSB`` draws from
OpenCV's random generator, so the same frame could come back as a different
sub-grid on each call (measured: a 4-square occluder gave 4 x 10, 5 x 3 and
3 x 3 over 20 calls), and with it different correspondences, planes and a
different laser fit. :func:`detect_checkerboard` therefore seeds that
generator (``seed``, default 0) immediately before detecting. OpenCV's
generator is per thread, so this resets the calling thread's only.

**The detector is asked for a small grid, and told the real one.**
`findChessboardCornersSB` with `CALIB_CB_LARGER` returns whatever grid it
actually found, and reports its shape in the metadata; asking for the nominal
board instead is worse in both directions, measured against OpenCV 5.0 (the
floor this package requires; the services port measured the same on 4.13):

* a partial view fails outright. A board with its right third out of frame
  returns nothing for the exact `patternSize` and a clean 10 x 10 sub-grid for
  a small one — and occlusion, glare and the frame edge cut the board down
  often enough that the survey saw sub-grids on many frames.
* a wrong-but-plausible `patternSize` also fails outright, so an operator who
  linked the wrong target would see "no board here" rather than a diagnosis.

Reading the shape back also *is* the safety property. A nominal full-board
grid paired against a partial detection mis-pairs every correspondence, and
`solvePnP` does not refuse that: it returns success and a confident,
meaningless pose, and nothing downstream would notice.

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

**The pitch is per axis.** `pitch_x_m` spaces corners along a declared row
(across the `interior_cols` axis), `pitch_y_m` down a column; the E4E board's
differ by ~0.3 %. Since the detected grid's orientation is free,
`pitch_per_detected_axis` reads which is which from the detected shape, and
refuses a frame whose shape fits the board either way round when the two
pitches differ by more than :data:`PITCH_EQUAL_REL_TOL`: a guess there is a
scale error the size of the anisotropy, and nothing downstream could see it.
Pitches within that tolerance behave as one square size.
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
    "MIN_CHECKER_PARITY",
    "MIN_GRID_SIDE",
    "PITCH_EQUAL_REL_TOL",
    "board_hull",
    "checker_parity",
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

#: How far the **worst** detected corner may sit from a perfect grid before
#: the detection is refused, as a fraction of the median corner spacing.
#:
#: A regular planar grid imaged by a pinhole camera maps to the image by a
#: HOMOGRAPHY, exactly — so the residual of the best-fit homography is a direct
#: test of "is this actually a grid?", scale-free and independent of how far
#: away the board is. It is the worst corner's, not the median's: a median is
#: itself robust, and would hide a minority of corners latticed a square off,
#: which solvePnP would then pair with the wrong body points.
#:
#: Measured on synthetic boards (flat, warped, cropped, and five occluder
#: sizes) under OpenCV 5.0: the worst genuine corner sits at 0.05-1.5 % of the
#: spacing (an occluder's edge corner is the 1.5 %), against 5 % here. An
#: incoherent detection (corners that are not one lattice) lands far above.
#: Re-measure on real underwater frames before tightening it; the safe
#: direction is to drop frames (there are 28-133 per calibration dive).
#:
#: It **cannot** see a uniform scale error: corners two squares apart labelled
#: one apart are still a perfect grid. :data:`MIN_CHECKER_PARITY` catches that.
MAX_GRID_RESIDUAL_FRACTION = 0.05

#: The least :func:`checker_parity` a detection must show.
#:
#: **This catches a silent 2x scale error.** On a heavily occluded board the
#: detector can return corners that are TWO squares apart while `body_points`
#: label them one apart (measured: a genuine detection's median corner spacing
#: was 32 px, one of these 73 px). The pose then places the board at half its
#: true distance, every depth and length halves, and nothing else can see it:
#: the homography fits perfectly, `solvePnP` succeeds, the plane is
#: self-consistent. But in a true lattice each cell is one square, so cell
#: centres alternate dark and light like the board itself (parity ~1); in a 2x
#: lattice every cell centre lands on a corner junction and reads mid-grey
#: (parity ~0). Measured: 0.999-1.000 on every genuine detection above, 0.084
#: on a 2x lattice cut from a genuine one.
MIN_CHECKER_PARITY = 0.5

#: Pitches closer than this, relatively, are treated as one square size, so an
#: orientation-ambiguous sub-grid is accepted rather than refused. The E4E
#: board's two pitches differ by 0.28 %, which stays "different"; a guess
#: within 0.1 % is a scale error below any length that reaches a fish.
PITCH_EQUAL_REL_TOL = 1e-3

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
    """Worst corner's reprojection error under the corners' best homography.

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
    return float(np.max(np.linalg.norm(projected - image, axis=1)))


def checker_parity(  # pylint: disable=too-many-locals
    gray: np.ndarray,
    body_points: np.ndarray,
    image_points: np.ndarray,
    rows: int,
    cols: int,
) -> float:
    """How strongly the detected cells alternate dark and light, in ``[0, 1]``.

    Each cell of the detected grid (between four adjacent corners) is mapped
    to the image through the corners' homography, and the grey level at its
    centre (a 5x5 mean) is correlated with the board's parity ``(-1)^(r + c)``.
    A true lattice reads ~1: its cells are single squares. A lattice at twice
    the pitch reads ~0: its cell centres are corner junctions, half dark and
    half light. Absolute, so which colour the first square is does not matter.
    See :data:`MIN_CHECKER_PARITY`.
    """
    body = np.asarray(body_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    matrix, _ = cv2.findHomography(body, image, method=0)
    if matrix is None or rows < 2 or cols < 2:
        return 0.0
    grid = body.reshape(rows, cols, 2)
    centres = (grid[:-1, :-1] + grid[1:, :-1] + grid[:-1, 1:] + grid[1:, 1:]) / 4.0
    r_index, c_index = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    signs = np.where((r_index + c_index) % 2 == 0, 1.0, -1.0).ravel()
    points = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    patch = np.asarray(gray, dtype=np.float32)
    values = np.array(
        [float(cv2.getRectSubPix(patch, (5, 5), (float(x), float(y))).mean()) for x, y in points]
    )
    centred = values - values.mean()
    norm = float(np.linalg.norm(centred)) * float(np.linalg.norm(signs))
    if not math.isfinite(norm) or norm == 0.0:
        return 0.0
    return abs(float(centred @ signs) / norm)


def median_corner_spacing(image_points: np.ndarray, rows: int, cols: int) -> float:
    """Median distance between horizontally adjacent detected corners.

    The scale :data:`MAX_GRID_RESIDUAL_FRACTION` is measured against, and
    public so a caller can report it per frame: a lattice at twice the true
    pitch shows twice the spacing over the same board.
    """
    grid = np.asarray(image_points, dtype=np.float64).reshape(rows, cols, 2)
    return float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))


def board_hull(detected: DetectedCheckerboard) -> list[list[float]]:
    """The detected grid's outline, as a convex quad in draw order.

    Used to answer the question the calibration otherwise just assumes: **was
    the laser dot actually on the board?** `laser_point_on_plane` intersects
    the camera ray with the board's *infinite* plane, so a dot that missed the
    board and landed on whatever was behind it gets a confident, wrong depth,
    and nothing downstream can tell: a self-consistency check compares the
    fitted ray against the 2-D dot line, which is identical either way. The
    dots are right; only the depths are wrong.

    A grid is the projective image of a rectangle, so its convex hull is just
    the quadrilateral of its four extreme corners — no hull algorithm needed.
    Returned in traversal order, for a same-side-of-every-edge point test.

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
    """An 8-bit single-channel view of ``image``, which the detector needs.

    Grey ``(H, W)`` or ``(H, W, 1)``, BGR ``(H, W, 3)`` or BGRA ``(H, W, 4)``.

    Raises:
        TypeError: The image is not uint8. A 16-bit or float frame is a
            caller's decode path, not an unlucky frame: rescaling it here
            would be a guess at its range.
        ValueError: Any other shape.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise TypeError(f"the checkerboard detector needs a uint8 image, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 2:
        return image
    raise ValueError(f"expected a grey, BGR or BGRA image, got shape {image.shape}")


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

    This is also the one test of whether the detection fits the board at all.
    The detected grid is the declared one (`max_rows x max_cols`) either as
    declared -- its columns run along a declared row, spaced `pitch_x_m` -- or
    transposed, when its columns are the declared rows. The shape says which
    when only one way round fits the board. When both do (a square board, or
    a sub-grid within the shorter side) it cannot, and a guess is only
    harmless when the two pitches agree within :data:`PITCH_EQUAL_REL_TOL`.

    None when neither orientation fits, or the orientation is ambiguous and
    the pitches differ. A detection *larger* than the declared board fits
    neither way: it means the frame holds a different board, so the stored
    pitch does not apply (the 2025 pool-test board, 24 x 17, against the E4E
    board, 14 x 10). The two orientations are tested axis by axis, never as a
    sorted-pair comparison: comparing sorted pairs with ``<=`` is
    lexicographic and would accept 9 x 20 against a 10 x 14 board.
    """
    declared = rows <= max_rows and cols <= max_cols
    transposed = rows <= max_cols and cols <= max_rows
    if math.isclose(pitch_x_m, pitch_y_m, rel_tol=PITCH_EQUAL_REL_TOL):
        return (pitch_x_m, pitch_y_m) if declared or transposed else None
    if declared and not transposed:
        return pitch_x_m, pitch_y_m
    if transposed and not declared:
        return pitch_y_m, pitch_x_m
    return None


# Keyword-only board description plus the seed; one return per refusal, each
# a distinct reason a frame is dropped.
# pylint: disable-next=too-many-locals,too-many-arguments,too-many-return-statements
def detect_checkerboard(
    image: np.ndarray,
    *,
    max_rows: int,
    max_cols: int,
    pitch_x_m: float,
    pitch_y_m: float,
    seed: int = 0,
) -> DetectedCheckerboard | None:
    """Detect the board in `image` and pair its corners with board coordinates.

    `image` must be the **rectified** frame, full resolution, uint8, grey,
    BGR or BGRA. `max_rows` / `max_cols` are the declared board's interior
    corners and act as an upper bound, not a target. `pitch_x_m` /
    `pitch_y_m` are the measured grid pitch along a declared row and down a
    declared column. `seed` seeds OpenCV's random generator for the call (see
    the module docstring's note on determinism): the same frame and seed give
    the same detection every time.

    Returns None when no usable board is found — no detection, a degenerate
    strip, a grid larger than the declared board, metadata that disagrees
    with the corners returned, a grid whose axes cannot be told apart on a
    board whose pitches differ, corners that are not one lattice, or a
    lattice at the wrong pitch. Every one of those is a case where continuing
    would hand `solvePnP` a correspondence set it would accept and answer
    wrongly, so the caller drops the frame and the dive is fitted from the
    others.

    Raises:
        TypeError: The image is not uint8.
        ValueError: The image is not grey, BGR or BGRA.
    """
    gray = _to_grayscale(image)
    cv2.setRNGSeed(int(seed))
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        gray, _REQUESTED_PATTERN, _DETECTION_FLAGS
    )
    if not found or corners is None or meta is None:
        return None

    rows, cols = int(meta.shape[0]), int(meta.shape[1])
    if min(rows, cols) < MIN_GRID_SIDE:
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
    # A perfect grid at the wrong pitch: see `MIN_CHECKER_PARITY`.
    if checker_parity(gray, body_points, image_points, rows, cols) < MIN_CHECKER_PARITY:
        return None

    return DetectedCheckerboard(
        rows=rows,
        cols=cols,
        body_points=body_points,
        image_points=image_points,
    )
