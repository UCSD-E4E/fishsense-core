"""Classical board-plane estimator — the bar a learned model must beat.

Pipeline, exploiting that the board is a known rigid planar target whose
printed geometry we hold exactly:

    rectified photo
      -> segment the bright board quad
      -> seed a homography from the template page corners to that quad
      -> resolve the 4-fold corner ambiguity by pattern correlation
      -> refine the homography densely (ECC) against the binarized template
      -> project template reference points through H
      -> solvePnP against metric geometry -> board plane

Only the plane is the product (see :mod:`fishsense_core.plane`): in-plane rotation and
translation are invisible to the calibration, so the 4-fold seed ambiguity
only has to be resolved well enough for ECC to converge, not perfectly.

Pure-logic helpers are module-level and unit-tested; the OpenCV-heavy stages
are exercised by the integration run over the real corpus.

Ported verbatim from ``slate_training.baseline``
(``UCSD-E4E/2026-07-31_slate_training``); fishsense-core is the canonical
home and the training repo imports the estimator from here. Pure ``cv2`` +
``numpy`` — base install, no optional dependency. The optional learned mask
enters through ``estimate_plane(board_mask=...)``; it only adds candidates,
so a missing or wrong mask costs search time, never coverage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from fishsense_core.plane import Plane, plane_from_pose

# cv2 is a C extension pylint cannot introspect; matches the convention in the
# image modules. The many-argument OpenCV pipeline stages and the module length
# are inherent to a ported classical CV algorithm.
# pylint: disable=no-member,too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-lines

INCH_TO_M = 0.0254

__all__ = [
    "BoardEstimate",
    "order_quad",
    "rotate_quad",
    "template_corners",
    "homography_from_quad",
    "segment_board",
    "estimate_plane",
]


@dataclass
class BoardEstimate:
    """A recovered board plane plus the diagnostics needed to gate on it."""

    plane: Plane
    homography: np.ndarray
    image_points: np.ndarray
    reprojection_rms: float
    ecc_score: float
    board_area_px: float


def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Order 4 points cyclically around the board, starting top-left-most.

    Sorted by angle about the centroid rather than the usual sum/difference
    trick: for a board rotated near 45 degrees the sum/difference test ties and
    emits a duplicated corner, which silently produces a degenerate homography.
    Angular sort is a true permutation for any non-degenerate quad.

    With image y pointing down, ascending angle yields TL, TR, BR, BL for an
    axis-aligned board. The start corner is the minimum of x+y; a tie there is
    harmless because the caller scores all four rotations anyway.
    """
    pts = np.asarray(points, dtype=float).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def rotate_quad(quad: np.ndarray, k: int) -> np.ndarray:
    """Cyclically rotate an ordered quad by `k` corners.

    The board is a rectangle, so a page-corner correspondence is only defined
    up to 4 rotations; the caller scores each and keeps the best.
    """
    return np.roll(np.asarray(quad, dtype=float), -int(k) % 4, axis=0)


def template_corners(width: float, height: float) -> np.ndarray:
    """Page corners of a template render, in the same TL/TR/BR/BL order."""
    return np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=float
    )


def homography_from_quad(
    template_size: Tuple[float, float], quad: np.ndarray
) -> np.ndarray:
    """Homography mapping template page pixels onto an ordered image quad."""
    width, height = template_size
    return cv2.getPerspectiveTransform(
        template_corners(width, height).astype(np.float32),
        np.asarray(quad, dtype=np.float32),
    )


def local_std(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Standard deviation of intensity in a `ksize` box around each pixel.

    This is *the* cue that finds the board. Measured over 8 frames spanning all
    four labeled templates, separation of board from background
    (`scripts/probe_cues.py`, d-prime against background MAD):

        local_std      12.00   (8/8 frames usable)
        local_range     5.39   (8/8)
        grad_density    3.33   (6/8)
        grey-world V   -0.31   (0/8)   <- colour correction does not help
        grey            -0.02   <- why the first detector failed
        saturation      -0.06

    Absolute brightness is worthless underwater: the white board's median grey
    is 101 against a frame median of 102. But a black pattern on a light panel
    is a strong *local contrast* signature that smooth water and fine reef
    texture both lack.
    """
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (ksize, ksize))
    mean_sq = cv2.boxFilter(f * f, -1, (ksize, ksize))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def segment_board(
    bgr: np.ndarray,
    min_area_frac: float = 1e-3,
    max_area_frac: float = 0.9,
    min_contrast: int = 25,
    window_frac: float = 0.01,
) -> np.ndarray | None:
    """Find the printed pattern as the highest-local-contrast region.

    Returns the ordered quad bounding the *pattern* (not the board's white
    margin) — `estimate_plane` seeds its homography from the template's
    pattern quad to match, so the two are consistent.

    Guards that fire on real frames: a near-uniform image (no board in view, a
    white-out) has no meaningful split and must not yield a "board"; and a
    region covering essentially the whole frame is background.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if int(gray.max()) - int(gray.min()) < min_contrast:
        return None

    # Window sized to the pattern's stroke scale. Too small and the response
    # hugs edges instead of filling the pattern; too large and the halo
    # swamps a distant board.
    diagonal = float(np.hypot(*gray.shape[:2]))
    window = max(5, int(window_frac * diagonal)) | 1
    contrast = local_std(cv2.GaussianBlur(gray, (5, 5), 0), window)

    scaled = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((max(3, window // 2) | 1,) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    candidates = board_candidates(
        mask, gray.shape, window, min_area_frac, max_area_frac
    )
    return candidates[0] if candidates else None


def board_candidates(
    mask: np.ndarray,
    shape: Tuple[int, int],
    window: int,
    min_area_frac: float = 1e-3,
    max_area_frac: float = 0.9,
    aspect_range: Tuple[float, float] = (0.25, 4.0),
    limit: int = 12,
) -> List[np.ndarray]:
    """Plausible pattern quads from a local-contrast mask, largest first.

    Returning several matters: local contrast separates the board from *water*
    (d-prime 12) but **not from reef**, which is also textured and much larger,
    so the biggest blob is usually coral. Measured on real frames, the
    largest-blob rule returned regions of ~3964x1174 px against a true board of
    ~287x181. The template is the only reliable arbiter, so the caller scores
    these candidates by actual alignment rather than trusting area.

    Filters applied here are cheap and shape-only: area bounds and a
    perspective-tolerant aspect window.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(shape[0] * shape[1])
    min_area, max_area = min_area_frac * frame_area, max_area_frac * frame_area

    scored: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        if min(rect_w, rect_h) < 1.0:
            continue
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if not aspect_range[0] <= aspect <= aspect_range[1]:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
        # The contrast response bleeds ~window/2 past the pattern; shrink back.
        scored.append((area, order_quad(_shrink_quad(box, window / 2.0))))

    scored.sort(key=lambda t: -t[0])
    return [quad for _, quad in scored[:limit]]


def local_normalize(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Zero-mean, unit-variance the image within a `ksize` neighbourhood.

    This is what makes template *matching* possible here. Raw intensity carries
    no signal (board grey 101 vs frame 102), but after local normalization the
    board's dark-pattern-on-light-panel structure becomes directly comparable
    to the binarized template render, independent of how the water has shifted
    exposure across the frame.
    """
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (ksize, ksize))
    variance = np.maximum(cv2.boxFilter(f * f, -1, (ksize, ksize)) - mean * mean, 0.0)
    return (f - mean) / (np.sqrt(variance) + 1.0)


def template_search(
    gray: np.ndarray,
    template_gray: np.ndarray,
    scales: Sequence[float] | None = None,
    angles: Sequence[float] | None = None,
    top_k: int = 8,
    template_points: Sequence[Sequence[float]] | None = None,
    point_scale: float = 1.0,
) -> List[Tuple[float, np.ndarray]]:
    """Exhaustive scale x in-plane-rotation search for the pattern.

    Sidesteps the failure mode that killed the connected-component approach:
    the board's contrast blob merges with reef into one component, so it is
    never a separable candidate. A sliding search has no such notion — it scores
    every location independently.

    Out-of-plane foreshortening is *not* searched. `cv2.matchTemplate` is
    tolerant of moderate perspective, and the ECC refinement downstream
    recovers the full homography from a fronto-parallel seed, so searching it
    here would multiply cost for nothing.

    Returns up to `top_k` ``(score, quad)`` pairs, best first, where `quad` is
    the pattern's estimated image quad at that scale/rotation.
    """
    if scales is None:
        # Board spans roughly 5%-70% of the frame's larger dimension.
        # 11 steps. Finer sampling was tried (19 steps) and is **worse**
        # overall: 67% -> 63% seeded. It helps a board that sits mid-gap
        # (V-Slate 4: 44% -> 50%) but costs more elsewhere (Tic-Tac-Toe
        # 83% -> 50%, V-Slate 1 73% -> 69%), because every extra hypothesis is
        # another chance for background to produce a high max score. More
        # candidates raise the noise floor faster than they sharpen the signal.
        scales = np.geomspace(0.05, 0.7, 11)
    if angles is None:
        # Full 360, not 180. The tape patterns are deliberately haphazard and
        # **not** 180-degree symmetric — that asymmetry is exactly what makes
        # board orientation legible to a labeler. A chevron held at 225 degrees
        # has no match anywhere in a 0-180 sweep, so the search settles on
        # background. Verified on real failures: large, close, high-contrast
        # boards were missed purely for being rotated past 180 degrees.
        # Corpus-wide this lifted seeded coverage 53% -> 60% and rescued
        # V-Slate 3 outright (1607mm -> 89mm median offset).
        #
        # Tic-Tac-Toe's *plane* metrics moved adversely over the same change
        # (median angle 9.3 -> 45.3 deg) while its seeded pixel accuracy
        # improved sharply (81 -> 4.5 px). That is 6 frames; too few to act on,
        # and a symmetry-aware sweep was tried and reverted because the grids'
        # renders are not actually half-turn symmetric (handwritten digits and
        # borders break it, even though the fiducial *points* are). Revisit
        # with more grid data.
        angles = np.arange(0.0, 360.0, 22.5)

    pattern = template_pattern_quad(template_gray, template_points, point_scale)
    x0, y0 = pattern.min(axis=0)
    x1, y1 = pattern.max(axis=0)
    patch = template_gray[int(y0):int(y1), int(x0):int(x1)]
    if patch.size == 0:
        return []

    target = float(max(gray.shape[:2]))
    norm_window = max(5, int(0.01 * np.hypot(*gray.shape[:2]))) | 1
    scene = local_normalize(gray, norm_window)

    results: List[Tuple[float, np.ndarray]] = []
    for scale in scales:
        width = max(12, int(target * scale))
        height = max(12, int(width * patch.shape[0] / patch.shape[1]))
        if height >= gray.shape[0] or width >= gray.shape[1]:
            continue
        resized = cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)

        for angle in angles:
            rotated, corners = _rotate_patch(resized, float(angle))
            if rotated.shape[0] >= gray.shape[0] or rotated.shape[1] >= gray.shape[1]:
                continue
            # Zero-mean the template so TM_CCOEFF_NORMED compares structure,
            # not brightness — the whole point of the local normalization.
            probe = rotated.astype(np.float32)
            probe -= probe.mean()

            response = cv2.matchTemplate(scene, probe, cv2.TM_CCOEFF_NORMED)
            _, peak, _, location = cv2.minMaxLoc(response)
            offset = np.array([location[0], location[1]], dtype=float)
            results.append((float(peak), order_quad(corners + offset)))

    results.sort(key=lambda t: -t[0])
    return results[:top_k]


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=float)


def refine_at_full_res(
    image_gray: np.ndarray,
    template_gray: np.ndarray,
    homography: np.ndarray,
    pad_frac: float = 0.35,
    iterations: int = 200,
) -> Tuple[np.ndarray, float]:
    """Re-refine `homography` on a full-resolution crop around the board.

    Why this matters more than any other refinement: the plane's **offset** is
    the error term the calibration actually feels (a 1 cm shift along the
    normal moves the laser point ~10 mm, whereas 20 deg of tilt moves it only
    ~9 mm), and offset is set by the board's apparent *scale*. Estimating scale
    to mm at 2 m needs sub-pixel boundary accuracy, which is unobtainable when
    the coarse pass sees a board only ~65 px across.

    Cropping keeps the cost bounded — we refine a few hundred pixels of board at
    native resolution instead of a 12 MP frame — and the template is rendered at
    the matching scale so ECC starts well-conditioned.

    Returns the refined full-resolution homography and its ECC score; on
    failure the input is returned with score 0.
    """
    pattern = template_pattern_quad(template_gray).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(
        pattern.astype(np.float64), homography
    ).reshape(-1, 2)

    span = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    if not np.isfinite(span) or span < 8:
        return np.asarray(homography, dtype=float), 0.0

    pad = span * pad_frac
    x0 = int(max(0, projected[:, 0].min() - pad))
    y0 = int(max(0, projected[:, 1].min() - pad))
    x1 = int(min(image_gray.shape[1], projected[:, 0].max() + pad))
    y1 = int(min(image_gray.shape[0], projected[:, 1].max() + pad))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return np.asarray(homography, dtype=float), 0.0
    crop = image_gray[y0:y1, x0:x1]

    # Render the template so its pattern is about the same pixel size as the
    # board in the crop; ECC converges far better at matched scale.
    pattern_span = max(np.ptp(pattern[:, 0, 0]), np.ptp(pattern[:, 0, 1]))
    t_scale = float(np.clip(span / max(pattern_span, 1.0), 0.02, 1.0))
    t_img = cv2.resize(template_gray, None, fx=t_scale, fy=t_scale,
                       interpolation=cv2.INTER_AREA)
    if min(t_img.shape[:2]) < 8:
        return np.asarray(homography, dtype=float), 0.0

    shift = _translation(-x0, -y0)
    seed = shift @ homography @ np.linalg.inv(_scale_matrix(t_scale))
    seed = seed / seed[2, 2]

    refined, score = _refine_ecc(t_img, crop, seed, iterations=iterations)
    lifted = np.linalg.inv(shift) @ refined @ _scale_matrix(t_scale)
    return lifted / lifted[2, 2], score


def _rotate_patch(patch: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a patch about its centre, expanding the canvas to fit.

    Returns the rotated patch and its four original-corner positions in the
    rotated patch's own coordinate frame, so a match location can be turned
    straight back into a quad.
    """
    height, width = patch.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]

    rotated = cv2.warpAffine(patch, matrix, (new_w, new_h), borderValue=int(patch.mean()))
    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=float
    )
    mapped = (matrix[:, :2] @ corners.T).T + matrix[:, 2]
    return rotated, mapped


def _for_matching(gray: np.ndarray) -> np.ndarray:
    """Locally-normalized float image — the space where the board is visible."""
    window = max(5, int(0.02 * np.hypot(*gray.shape[:2]))) | 1
    return local_normalize(gray, window)


def mask_candidates(
    mask: np.ndarray,
    shape: Tuple[int, int],
    threshold: float = 0.5,
    limit: int = 3,
    pad_frac: float = 0.10,
) -> List[np.ndarray]:
    """Candidate board quads from a learned segmentation mask.

    `mask` is a probability map (uint8 0-255 or float 0-1) at any resolution;
    it is scaled to `shape` (H, W). Returns the largest components' quads,
    biggest first, slightly padded so a tight mask still contains the board.

    These are *additional* candidates, not a replacement. The classical
    contrast blobs remain in the pool, so a wrong or empty mask can only cost
    search time — never coverage. That matters because the mask's weakest
    folds are board types it has never seen (Tic-Tac-Toe 0.67), which is
    exactly where the classical path must still work.
    """
    probs = np.asarray(mask, dtype=np.float32)
    if probs.max() > 1.5:
        probs = probs / 255.0
    binary = (probs > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    height, width = shape[:2]
    binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return []
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    quads: List[np.ndarray] = []
    for index in order[:limit]:
        x = stats[index, cv2.CC_STAT_LEFT]
        y = stats[index, cv2.CC_STAT_TOP]
        w = stats[index, cv2.CC_STAT_WIDTH]
        h = stats[index, cv2.CC_STAT_HEIGHT]
        if w < 4 or h < 4:
            continue
        px, py = w * pad_frac, h * pad_frac
        x0, y0 = max(0.0, x - px), max(0.0, y - py)
        x1, y1 = min(float(width), x + w + px), min(float(height), y + h + py)
        quads.append(order_quad(
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)))
    return quads


def shrink_to_pattern(
    quad: np.ndarray, ratio_x: float, ratio_y: float
) -> np.ndarray:
    """Convert a whole-board quad into the pattern's sub-quad.

    The learned mask predicts the **board** (the printed page), but every seed
    maps the *pattern* extent onto its candidate. Feeding a board-sized quad
    stretches the pattern across the whole page, which scores poorly on
    correlation — so the candidate never survives to ECC refinement and the
    mask has literally no effect (observed: bit-identical results with and
    without masks).

    The template gives the exact ratio of pattern extent to page extent, so
    the conversion is a scale about the quad's centre.
    """
    quad = np.asarray(quad, dtype=float)
    centre = quad.mean(axis=0)
    scaled = quad - centre
    scaled[:, 0] *= float(ratio_x)
    scaled[:, 1] *= float(ratio_y)
    return scaled + centre


def contrast_mask(gray: np.ndarray, window_frac: float = 0.01) -> Tuple[np.ndarray, int]:
    """Binary local-contrast mask plus the window size used to build it."""
    diagonal = float(np.hypot(*gray.shape[:2]))
    window = max(5, int(window_frac * diagonal)) | 1
    contrast = local_std(cv2.GaussianBlur(gray, (5, 5), 0), window)
    scaled = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((max(3, window // 2) | 1,) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask, window


def _shrink_quad(quad: np.ndarray, amount: float) -> np.ndarray:
    """Contract a quad toward its centroid by roughly `amount` pixels."""
    quad = np.asarray(quad, dtype=float)
    centre = quad.mean(axis=0)
    radii = np.linalg.norm(quad - centre, axis=1)
    scale = np.clip(1.0 - amount / np.maximum(radii, 1e-6), 0.1, 1.0)
    return centre + (quad - centre) * scale[:, None]


def template_pattern_quad(
    template_gray: np.ndarray,
    template_points: Sequence[Sequence[float]] | None = None,
    scale: float = 1.0,
    pad_frac: float = 0.12,
) -> np.ndarray:
    """Ordered quad bounding the *fiducial pattern* in a template render.

    Prefer `template_points` when available: the reference points literally
    define the pattern, so their bounding box is the thing we want to match.

    The dark-pixel fallback is a trap and is only used when points are absent.
    Every real template has a page border and edge marks, so "all dark pixels"
    spans **100% of the page** — measured on all three labeled families. That
    makes the matched patch ~78% white margin for the V-Slates and 3x too large
    for the grids, leaving so little signal that a thin dark object (a diver's
    speargun, in the failure that exposed this) can outscore a real board.

    `scale` converts point coordinates when the render has been resized;
    `pad_frac` widens the box so stroke width beyond the corner points is
    included.
    """
    height, width = template_gray.shape[:2]

    if template_points is not None and len(template_points) >= 3:
        pts = np.asarray(template_points, dtype=float) * float(scale)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_x = (x1 - x0) * pad_frac
        pad_y = (y1 - y0) * pad_frac
        x0 = max(0.0, x0 - pad_x)
        y0 = max(0.0, y0 - pad_y)
        x1 = min(float(width), x1 + pad_x)
        y1 = min(float(height), y1 + pad_y)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            return order_quad(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
            )

    dark = (template_gray < 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return template_corners(width, height)
    points = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    return order_quad(cv2.boxPoints(cv2.minAreaRect(points)).astype(float))


def _scale_matrix(factor: float) -> np.ndarray:
    """Homogeneous matrix scaling image coordinates by `factor`."""
    return np.array(
        [[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )


def rescale_homography(
    homography: np.ndarray, template_scale: float, image_scale: float
) -> np.ndarray:
    """Lift a homography computed at reduced resolution back to full size.

    With ``template_small = S_t · template_full`` and
    ``image_small = S_i · image_full``, a warp `H_s` mapping small template to
    small image corresponds to ``H = S_i⁻¹ · H_s · S_t`` at full resolution.

    This is what makes the baseline tractable: ECC on 4200×2550 template
    against a 4014×3016 frame is minutes per call, while the same fit at ~1/8
    scale is milliseconds and — because a homography is scale-free — carries no
    loss of accuracy in the recovered plane beyond sub-pixel seeding.
    """
    lifted = (
        np.linalg.inv(_scale_matrix(image_scale))
        @ np.asarray(homography, dtype=float)
        @ _scale_matrix(template_scale)
    )
    return lifted / lifted[2, 2]


def _correlation(template_gray, image_gray, warp) -> float:
    """Cheap normalized correlation of the template warped into the image.

    Used to pick among the four corner rotations before paying for ECC, which
    turns 4 expensive refinements into 4 cheap warps plus 1 refinement.
    """
    height, width = image_gray.shape[:2]
    warped = cv2.warpPerspective(template_gray, warp, (width, height))
    mask = cv2.warpPerspective(
        np.full_like(template_gray, 255), warp, (width, height)
    ) > 0
    if mask.sum() < 50:
        return -1.0
    a = warped[mask].astype(np.float64)
    b = image_gray[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-9 else -1.0


def _refine_ecc(
    template_gray: np.ndarray, image_gray: np.ndarray, warp: np.ndarray,
    iterations: int = 60,
) -> Tuple[np.ndarray, float]:
    """Dense homography refinement of `warp` (template -> image).

    Returns the refined warp and its ECC correlation; on failure to converge
    the input warp is returned with a score of 0 so the caller can gate.
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-5)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, refined = cv2.findTransformECC(
                template_gray,
                image_gray,
                warp.astype(np.float32),
                cv2.MOTION_HOMOGRAPHY,
                criteria,
                None,
                5,
            )
        return np.asarray(refined, dtype=float), float(score)
    except cv2.error:
        return np.asarray(warp, dtype=float), 0.0


def estimate_plane(
    bgr: np.ndarray,
    template_gray: np.ndarray,
    template_points: Sequence[Sequence[float]],
    dpi: float,
    camera_matrix: np.ndarray,
    full_res_refine: bool = False,
    template_blur: float = 0.0,
    board_mask: np.ndarray | None = None,
) -> BoardEstimate | None:
    """Recover the board plane from one rectified frame.

    `template_gray` is the binarized template render at `dpi`;
    `template_points` are its reference points in that same render-pixel space.
    Returns None when the board can't be found or PnP fails.
    """
    image_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if int(image_gray.max()) - int(image_gray.min()) < 25:
        return None

    t_h, t_w = template_gray.shape[:2]

    # Fit at reduced resolution, then lift (see `rescale_homography`). Both
    # scales are chosen so the board spans a few hundred pixels -- enough for
    # ECC to lock on, cheap enough to run per frame.
    t_scale = min(1.0, 320.0 / max(t_w, t_h))
    i_scale = min(1.0, 900.0 / max(image_gray.shape[:2]))
    t_small = cv2.resize(template_gray, None, fx=t_scale, fy=t_scale,
                         interpolation=cv2.INTER_AREA)
    if template_blur > 0:
        # Match the template's sharpness to the board's. The render is a hard
        # binary edge; the real board is scattered and motion-blurred, and ECC
        # correlates appearance, not geometry.
        k = int(template_blur * 4) | 1
        t_small = cv2.GaussianBlur(t_small, (k, k), template_blur)
    i_small = cv2.resize(image_gray, None, fx=i_scale, fy=i_scale,
                         interpolation=cv2.INTER_AREA)

    # Note the asymmetry, which is measured rather than assumed: the sliding
    # search runs on a *locally normalized* image because absolute intensity
    # carries no board signal at frame scale (grey 101 vs 102), but ECC refines
    # against **raw grey**. Normalizing before ECC costs accuracy (20.1 deg ->
    # 25.5 deg, 65 mm -> 142 mm) because the normalization window is comparable
    # to the chevron's stroke width and suppresses the pattern ECC locks onto.
    # Finding the board and refining onto it want different representations.

    # Two independent candidate sources, both in reduced-image coordinates.
    # Blob candidates are cheap but merge with reef; the sliding search has no
    # segmentation step to fail, at the cost of an exhaustive scan.
    mask, window = contrast_mask(image_gray)
    quads = [q * i_scale for q in board_candidates(mask, image_gray.shape, window)]
    if board_mask is not None:
        # Learned candidates go in alongside the classical ones; the winner is
        # still decided by template correlation, so a bad mask costs nothing.
        # The mask outlines the whole board, so shrink to the pattern's extent
        # before seeding -- see `shrink_to_pattern`.
        page_quad = template_pattern_quad(template_gray, template_points)
        ratio_x = float(np.ptp(page_quad[:, 0])) / max(float(t_w), 1.0)
        ratio_y = float(np.ptp(page_quad[:, 1])) / max(float(t_h), 1.0)
        quads += [
            shrink_to_pattern(q, ratio_x, ratio_y) * i_scale
            for q in mask_candidates(board_mask, image_gray.shape, pad_frac=0.0)
        ]
    quads += [q for _, q in template_search(
        i_small, t_small, template_points=template_points, point_scale=t_scale)]
    if not quads:
        return None
    # Both sides of the seed must measure the same thing: `segment_board`
    # returns the pattern quad, so map from the template's pattern quad rather
    # than its page corners (the page carries a white margin the detector
    # never sees).
    pattern_quad = template_pattern_quad(t_small, template_points, t_scale)

    # Score every (candidate region x corner rotation) by cheap correlation and
    # refine only the winner. This is where the board is actually chosen —
    # area alone picks reef, the template does not.
    seeds = [
        cv2.getPerspectiveTransform(
            pattern_quad.astype(np.float32),
            rotate_quad(quad, k).astype(np.float32),
        )
        for quad in quads
        for k in range(4)
    ]
    scores = [_correlation(t_small, i_small, s) for s in seeds]
    order = np.argsort(scores)[::-1]

    # Refine the few best seeds: correlation ranks roughly but not perfectly,
    # and ECC converges to a much sharper optimum when it starts near one.
    best: Tuple[float, np.ndarray] | None = None
    for index in order[:3]:
        refined, score = _refine_ecc(t_small, i_small, seeds[int(index)])
        # `best[0]` is guarded by the short-circuit; pylint's flow analysis
        # doesn't see the None-narrowing through `or`.
        # pylint: disable-next=unsubscriptable-object
        if best is None or score > best[0]:
            best = (score, refined)

    assert best is not None
    ecc_score, refined = best
    homography = rescale_homography(refined, t_scale, i_scale)

    # Full-resolution crop refinement is implemented and tested
    # (`refine_at_full_res`) but **off by default: measured worse**, offset
    # 127mm -> 216mm over the corpus. More resolution does not help because the
    # limiting factor is not pixel count but appearance mismatch — a crisp
    # binarized template against a genuinely blurry underwater board. Reducing
    # resolution was implicitly matching their sharpness.
    if full_res_refine:
        tuned, score = refine_at_full_res(image_gray, template_gray, homography)
        if score > 0.0:
            homography, ecc_score = tuned, score

    quad = quads[int(order[0]) // 4]

    template_pts = np.asarray(template_points, dtype=float).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(template_pts, homography).reshape(-1, 2)

    body = np.zeros((len(template_pts), 3), dtype=np.float32)
    body[:, :2] = (np.asarray(template_points, dtype=float) / float(dpi)) * INCH_TO_M

    ok, rvec, tvec = cv2.solvePnP(
        body, image_points.astype(np.float64),
        np.asarray(camera_matrix, dtype=float), np.zeros((5,))
    )
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        body, rvec, tvec, np.asarray(camera_matrix, dtype=float), np.zeros((5,))
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)

    return BoardEstimate(
        plane=plane_from_pose(rotation, tvec.reshape(3)),
        homography=homography,
        image_points=image_points,
        reprojection_rms=float(np.sqrt((residual**2).mean())),
        ecc_score=ecc_score,
        board_area_px=float(cv2.contourArea(quad.astype(np.float32))),
    )
