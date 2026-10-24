"""Label-free laser calibration by size constancy -- no slate size, no labels.

This replaces :func:`fishsense_core.laser.calibrate_laser`'s input when there
are no labelled 3-D points: it fits the beam from the detector's dots and the
apparent size of whatever rigid object the dot is on.

Provenance: ported from fishsense-services v0.48.0
`services/fishsense-services-processor/src/fishsense_services_processor/
automatic_results/size_constancy.py`, which ported cscw-fishsense2027@96a8da07
calibration_analysis/e1j_size_constancy/register.py (`pair_scale`; `score`'s
weighted solve, its one robust pass and its connectivity rule) and
slate_unknown.py (`frame`, `fit_tv`), with the beam built as e2e_measurement/
score.py `calibrations()` builds its label-free one. The method is
wuwnet-fishsense2026@5532b99 fishsense_wuwnet/laser.py
(`close_with_apparent_size`); the paper is cscw PAPER.md §4.3. Where core
departs from that port it says so below, under **Departures**.

It is NumPy, SciPy and OpenCV (SIFT, RANSAC), not Rust: the registration is
OpenCV's, and the only consumers are server-side. Mobile measures from a
depth map and has no use for it.

**The idea.** A dot at range Z sits at ``t = t_v + A / Z`` along the dive's
laser line, and a rigid object the dot is on appears with size ``s ∝ 1/Z``. So
``s_k / (t_k - t_v)`` is constant across frames only for the true vanishing
position ``t_v``; the object's physical size cancels. Metric scale comes from
the laser's offset ``|O|``, fixed by the mount (fleet median 104.0 mm until the
CAD value arrives, cscw score.py `O_DESIGN_M`).

**Sizes with no labels** (register.py). SIFT features within 350 px of the dot
in each frame; every pair of frames is registered with RANSAC similarity
transforms, peeled off as up to three motion layers; the object is the layer
whose inliers surround the dot with the most matches within 80 px of it (the
dot is on the object by construction); its local scale at the dot is the size
ratio. A weighted least-squares solve over all pairs gives each frame's log
size up to one constant, after dropping pairs inconsistent with the rest.

What the services port changed from cscw, all kept here:

* **the features come from the rectified frame** (`extract_features`), where
  cscw read the raw at half size through rawpy. Both are halved,
  CLAHE-equalised greys with the same mask, detector and matcher; the dot is
  in rectified pixels in both, so features and dot share one space (cscw's
  were ~ a lens distortion apart);
* **refusals** (`fit_label_free`), which cscw left to the reader: too few
  frames (`MIN_FRAMES`, slate_unknown's), too few matched pairs (register's
  "only N matched pairs" skip), too little range spread (`MIN_SIZE_RATIO`, the
  paper's ~1.5x), and an optimum pinned at the search bracket (unconstrained);
* frames come with their dive line when one is known (cscw's `lines.psv`, the
  production dive line), else the line through the frames' own dots (cscw's
  fallback). Oriented to negative y, cscw's FSL-mount convention;
* **a frame whose size disagrees with its place on the line is dropped**
  (`trim_frames`, 2026-10-19), one at a time, while one is more than
  `FRAME_RESIDUAL_MAX` off. A detector dot can sit on the line at the wrong
  range -- mission 80 had one far down the line on a slate as near as its
  nearest frames, and with it the vanishing point was 360 px off (standard
  error 324 px), without it 11 px (0.9 px). Each frame is judged by the fit of
  the others (`deleted_residuals`), so a wrong frame cannot shape the fit it
  is judged by -- setting the bracket, pinning the optimum, pushing good
  frames out first -- and after each drop the sizes are re-solved from the
  kept frames' own pairs, so the dropped frame's pairs no longer shape them.
  The pairs already have their own loop check (`solve_log_sizes`); this is
  the same idea a level up. A fit pinned at the bracket after the trim is
  refused (``unconstrained``), and so is one the trim leaves too few frames
  when it was pinned before it.

**Departures** from the services port, each a case it got wrong:

* **One connected component.** A frame had a size if *any* pair touched it,
  so two clusters of frames that never matched each other were sized
  relative to each other by the arbitrary gauge alone, and fitted. Now only
  the largest connected component of the pair graph has sizes, and that
  graph is taken **after** the robust pass, so a frame whose only pairs were
  dropped has no size rather than a made-up one. When the graph is one
  component (every dive the services fixtures cover) the solve is unchanged.
* **A malformed pair is an error** (`near` < 1, a non-finite scale, an index
  out of range or a self-pair), where `near` = 0 used to turn every residual
  into NaN and silently zero every size.
* **With no dive line, the direction follows the trim.** It came from every
  frame before the trim and was never recomputed, so the frame the trim
  dropped still tilted it; now it is refitted from the kept frames and the
  trim repeated until the kept set stops changing.
* **A collapsed similarity is no layer.** Matches that all land on one spot
  in the other frame fit a similarity of scale ~0, and it was the object's
  if it held the dot. cscw's own pairs hold one (session 114's (7, 23), log
  scale -37.18, its inliers 0 px apart), and under OpenCV 5 the same
  features give six (sessions 63, 71, 114; -39 to -42), mission 63's two
  connecting frames with no real pair and moving its vanishing position
  from -2101 to -5988 px. `pair_scale` now skips a layer whose inliers sit
  within `RANSAC_PX` RMS of one spot in either frame (these sit at <= 0.001
  px; real pairs' at >= 33) or whose ratio is beyond `MAX_PAIR_LOG_SCALE`
  (10x; real pairs reach 2.9x), and `solve_log_sizes` drops a pair beyond
  that bound before its first solve, whoever measured it. On the ten
  sessions' features this is cscw's registration exactly, under 4.14 or 5,
  less that one pair; session 114 then keeps frame 7, whose 16 other pairs
  are good, and lands 2.2 px from the known-size fit (cscw's 6.4).

**The shadow copy.** fishsense-services v0.48.0 also carries
`shadow_calibration/fit.py`, a port of cscw-fishsense2027@8312780a
`analysis/codesign/size_constancy_sam.py`, with its own `fit_tv` and `beam`.
They are this module's `fit_vanishing_position` and `beam_from_vanishing`
line for line: the same bracket (``[t_min - 4005, t_min - 5]`` px, bounded
Brent), the same vanishing point ``t_v u + off n`` with ``n = (-u1, u0)``,
the same unit axis ``K^-1 [v, 1]`` and the same origin (``|O|`` along the
line direction in normalised coordinates, z = 0, ``|O| = 0.104`` m). Its
`port_fit` differs only in its bootstrap: 500 draws from one generator per
``(session, variant)``, ``default_rng([session, variant])``, where
`fit_label_free` takes 200 from ``default_rng(0)``. That is `bootstrap_se`'s
``draws`` and ``rng``, so the shadow fit is ``bootstrap_se(t, s,
rng=np.random.default_rng([session, variant]), draws=500)``; its published
pool rows reproduce through this module (tests/test_size_constancy.py). Its
error metrics against a reference calibration (`err_deg` at ``F_PX = 2850``,
`angle_deg`) are evaluation, not calibration, and stay with the shadow stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "BOOTSTRAP_DRAWS",
    "deleted_residuals",
    "FRAME_RESIDUAL_MAX",
    "FrameFeatures",
    "LabelFreeFit",
    "LogSizes",
    "MIN_FRAMES",
    "MIN_PAIR_NEAR",
    "MIN_SIZE_RATIO",
    "NEAR_PX",
    "O_DESIGN_M",
    "PairScale",
    "ROI_PX",
    "beam_from_vanishing",
    "bootstrap_se",
    "extract_features",
    "fit_label_free",
    "fit_vanishing_position",
    "line_direction",
    "pair_scale",
    "pair_scales",
    "solve_log_sizes",
    "trim_frames",
]

#: The laser's offset from the camera, metres: the fleet median of 31 stored
#: calibrations, standing in for the mount's CAD value (cscw score.py).
O_DESIGN_M = 0.104
#: Full-resolution px around the dot whose features are kept (register.ROI).
ROI_PX = 700
#: A layer's score: its inliers within this many full-res px of the dot
#: (register.NEAR; "must be on the scale of the object, not the scene").
NEAR_PX = 80
#: Matches a similarity needs (register.MIN_INLIERS).
MIN_INLIERS = 8
#: A pair needs this many matches near the dot, or its weight is zero and its
#: residual undefined (register.score's `r[1] >= 3`).
MIN_PAIR_NEAR = 3
#: RANSAC reprojection threshold, px (register: 6.0).
RANSAC_PX = 6.0
#: The largest |log size ratio| a pair may have: 10x, a frame ten times the
#: other's range. A plausibility bound, applied to every pair the solve takes
#: (`solve_log_sizes`), wherever it came from. Of cscw's 1874 golden pairs,
#: 1873 reach 1.07 (2.9x); the other is session 114's collapsed one, -37.18.
MAX_PAIR_LOG_SCALE = float(np.log(10.0))
#: Lowe's ratio (register: 0.75).
LOWE_RATIO = 0.75
#: SIFT features per frame (register: 4000).
SIFT_FEATURES = 4000
#: Frames a fit needs (slate_unknown.MIN_FRAMES).
MIN_FRAMES = 8
#: Apparent-size spread a fit needs: the paper's "at least about 1.5x in
#: distance" (§4.3); sessions under 1.12x are unconstrained.
MIN_SIZE_RATIO = 1.5
#: The most a frame's log size may sit off the relation fitted without it
#: (`deleted_residuals`): 10 % in size. On cscw's ten sessions (human dots) the
#: largest is 0.058, but one frame of session 114 at 0.41; mission 80's
#: wrong-range detector frame was 0.92 (2026-10-19). Pairs carry ~1 %.
FRAME_RESIDUAL_MAX = 0.10
#: Bootstrap draws for the standard error (register / slate_unknown: 200).
BOOTSTRAP_DRAWS = 200
#: fit_tv's bracket: from 4000 px below to 5 px below the nearest dot.
_BRACKET_PX = 4000.0
_BRACKET_GAP_PX = 5.0
#: An optimum this close to the bracket's edge is the edge, not a minimum.
_AT_BOUND_PX = 1.0


@dataclass(frozen=True)
class FrameFeatures:
    """SIFT keypoints (full-resolution rectified px) and their descriptors."""

    xy: np.ndarray
    desc: Optional[np.ndarray]


@dataclass(frozen=True)
class PairScale:
    """``log s_j - log s_i``, and the matches near the dot that carried it."""

    i: int
    j: int
    log_scale: float
    near: int


@dataclass(frozen=True)
class LogSizes:
    """Per-frame log sizes, up to the gauge ``mean log s = 0``."""

    log_sizes: np.ndarray
    #: Frames some pair touches; the rest have no size.
    connected: np.ndarray
    #: SD of every pair's residual after the solve (register's pair_resid_sd).
    residual_sd: float


# A result record: every field is reported, so the count is the payload's.
@dataclass(frozen=True)
class LabelFreeFit:  # pylint: disable=too-many-instance-attributes
    """A dive's label-free calibration, accepted or refused, with what it saw."""

    outcome: str
    refusal_reason: Optional[str]
    laser_position: Optional[list[float]]
    laser_axis: Optional[list[float]]
    vanishing_px: Optional[float]
    line_direction: Optional[list[float]]
    line_offset_px: Optional[float]
    #: Indices (into the frames given) the fit used.
    used: list[int]
    frames_used: int
    pairs: int
    size_ratio: Optional[float]
    se_px: Optional[float]
    pair_residual_sd: Optional[float]
    o_mag_m: float


def extract_features(image_bgr: np.ndarray, dot: Sequence[float]) -> FrameFeatures:
    """SIFT around the dot, as register.py `extract` does on its half-size
    grey: CLAHE (2.0, 8x8), a disc of `ROI_PX`/2 (full-res) about the dot,
    `SIFT_FEATURES` features; keypoints returned in full-resolution px."""
    # cv2's members are built at import time; pylint cannot see them.
    # pylint: disable=no-member
    import cv2  # pylint: disable=import-outside-toplevel

    half = cv2.resize(
        image_bgr,
        (image_bgr.shape[1] // 2, image_bgr.shape[0] // 2),
        interpolation=cv2.INTER_AREA,
    )
    grey = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    mask = np.zeros_like(grey)
    cv2.circle(
        mask, (int(dot[0] / 2), int(dot[1] / 2)), int(ROI_PX / 2), 255, -1
    )  # fmt: skip
    keypoints, desc = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(
        grey, mask
    )
    xy = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2) * 2
    return FrameFeatures(xy=xy, desc=desc)


def pair_scale(  # pylint: disable=too-many-locals
    a: FrameFeatures, b: FrameFeatures, dot_a: Sequence[float], matcher=None
) -> Optional[tuple[float, int]]:
    """``(log s_b - log s_a, matches near the dot)`` for the object the dot is
    on in frame `a`, or None (register.py `pair_scale`, SELECT=near,
    MODEL=sim).

    Sequential RANSAC peels off up to three similarity motions; the object's
    is the one whose inlier hull contains the dot, with the most inliers within
    `NEAR_PX` of it. Its local scale at the dot, ``sqrt|det J|``, is the size
    ratio. A layer is no candidate when that scale is not measured: a
    reflection (``det J <= 0``), inliers within `RANSAC_PX` RMS of one spot in
    either frame (a collapse, which fits any small enough scale), or a ratio
    beyond `MAX_PAIR_LOG_SCALE`."""
    # cv2's members are built at import time; pylint cannot see them.
    # pylint: disable=no-member
    import cv2  # pylint: disable=import-outside-toplevel

    if a.desc is None or b.desc is None or len(a.desc) < 10 or len(b.desc) < 10:
        return None
    matcher = matcher or cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(a.desc, b.desc, k=2)
    good = [
        p[0] for p in knn if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance
    ]
    if len(good) < MIN_INLIERS:
        return None
    pa = a.xy[[g.queryIdx for g in good]]
    pb = b.xy[[g.trainIdx for g in good]]
    x, y = float(dot_a[0]), float(dot_a[1])
    best = None
    for _ in range(3):
        if len(pa) < MIN_INLIERS:
            break
        m, inl = cv2.estimateAffinePartial2D(
            pa, pb, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX
        )
        if m is None or inl.sum() < MIN_INLIERS:
            break
        inl = inl.ravel().astype(bool)
        hull = cv2.convexHull(pa[inl].astype(np.float32))
        inside = cv2.pointPolygonTest(hull, (x, y), True)
        # A similarity (estimateAffinePartial2D) is affine: its Jacobian is
        # its linear part everywhere, so the local scale needs no point.
        det = np.linalg.det(m[:, :2])
        near = int((np.hypot(*(pa[inl] - (x, y)).T) < NEAR_PX).sum())
        log_scale = float(0.5 * np.log(det)) if det > 0 else np.nan
        # Inliers that sit inside the RANSAC tolerance in either frame fit
        # every scale small enough to keep them there: it is not measured.
        spread = min(_rms_radius(pa[inl]), _rms_radius(pb[inl]))
        if (
            np.isfinite(log_scale)
            and spread >= RANSAC_PX
            and abs(log_scale) <= MAX_PAIR_LOG_SCALE
            and inside > 0
            and (best is None or near > best[1])  # pylint: disable=unsubscriptable-object
        ):
            best = (log_scale, near)
        pa, pb = pa[~inl], pb[~inl]
    return best


def _rms_radius(xy: np.ndarray) -> float:
    """RMS distance of points from their centroid, px."""
    return float(np.sqrt(((xy - xy.mean(axis=0)) ** 2).sum(axis=1).mean()))


def pair_scales(features: Sequence[FrameFeatures], dots: np.ndarray) -> list[PairScale]:
    """Every pair register.score keeps: matched, with `MIN_PAIR_NEAR` matches
    near the dot."""
    # cv2's members are built at import time; pylint cannot see them.
    # pylint: disable=no-member
    import cv2  # pylint: disable=import-outside-toplevel

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    out = []
    for i, j in combinations(range(len(features)), 2):
        r = pair_scale(features[i], features[j], dots[i], matcher)
        if r and r[1] >= MIN_PAIR_NEAR:
            out.append(PairScale(i, j, r[0], r[1]))
    return out


def _check_pairs(n: int, pairs: Sequence[PairScale]) -> None:
    """A pair the solve cannot weigh or place is the caller's error."""
    for p in pairs:
        if not (0 <= p.i < n and 0 <= p.j < n) or p.i == p.j:
            raise ValueError(f"pair ({p.i}, {p.j}) is not two distinct frames of {n}")
        if not p.near >= 1:
            raise ValueError(
                f"pair ({p.i}, {p.j}) has near={p.near}: its weight sqrt(near) must be "
                "positive (pair_scales keeps only pairs with MIN_PAIR_NEAR or more)"
            )
        if not np.isfinite(p.log_scale):
            raise ValueError(f"pair ({p.i}, {p.j}) has a non-finite log_scale {p.log_scale}")


def _largest_component(n: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    """Boolean mask of the largest connected component of the pair graph
    (ties to the one holding the lowest frame); a frame no edge touches is in
    none."""
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    touched = np.zeros(n, bool)
    for i, j in edges:
        touched[i] = touched[j] = True
        parent[root(i)] = root(j)
    roots = np.array([root(k) for k in range(n)])
    best, best_size = None, 0
    for k in range(n):
        if touched[k]:
            size = int((roots == roots[k]).sum())
            if size > best_size:
                best, best_size = roots[k], size
    return (roots == best) & touched if best is not None else touched


def _weighted_solve(n: int, pairs: Sequence[PairScale], members: np.ndarray):
    """Least-squares log sizes over ``members`` from the pairs among them,
    rows weighted by ``sqrt(near)``, gauge ``mean log s = 0`` over them."""
    rows = np.zeros((len(pairs), n))
    for k, p in enumerate(pairs):
        rows[k, p.j], rows[k, p.i] = 1, -1
    w = np.sqrt([p.near for p in pairs])
    a = np.vstack([rows * w[:, None], members.astype(float)])
    b = np.append(np.array([p.log_scale for p in pairs]) * w, 0.0)
    return a, b, w


def solve_log_sizes(n: int, pairs: Sequence[PairScale]) -> LogSizes:
    """Per-frame log sizes from pairwise ratios (register.score).

    Rows weighted by ``sqrt(near)``, gauge ``mean log s = 0``. A wrong pair
    (another layer that passed the selection) disagrees with the loop of other
    pairs through the same frames: one pass drops pairs beyond 4 robust SDs,
    the SD floored at 1 % of scale so an almost-perfect majority cannot
    shrink the tolerance until everything is rejected; then re-solve.

    Only frames in the **largest connected component of the pairs the robust
    pass kept** are ``connected``, and the re-solve is over them alone: a
    frame in another component, or whose only pairs were dropped, has no
    size relative to the rest, so it gets none (its entry is NaN) rather
    than one the gauge made up. With every frame in one component this is
    the services port's solve exactly.

    A pair beyond `MAX_PAIR_LOG_SCALE` is dropped before the first solve, not
    left to the robust pass: one at -37 (session 114's) drags that solve far
    enough to fail every other pair through its frames, so a frame with good
    pairs loses its size. The bound is here, not only in `pair_scale`, for
    pairs that come from elsewhere (cscw's files, services' shadow track).

    Raises:
        ValueError: A pair with ``near`` < 1, a non-finite ``log_scale``, or
            indices that are not two distinct frames in ``range(n)``.
    """
    _check_pairs(n, pairs)
    pairs = [p for p in pairs if abs(p.log_scale) <= MAX_PAIR_LOG_SCALE]
    if not pairs:
        return LogSizes(np.full(n, np.nan), np.zeros(n, bool), float("nan"))
    a, b, w = _weighted_solve(n, pairs, np.ones(n, bool))
    logs = np.linalg.lstsq(a, b, rcond=None)[0]
    r = (a[:-1] @ logs - b[:-1]) / w
    mad = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.01)
    kept = [p for p, ok in zip(pairs, np.abs(r) <= 4 * mad) if ok]
    connected = _largest_component(n, [(p.i, p.j) for p in kept])
    own = [p for p in kept if connected[p.i] and connected[p.j]]
    a, b, w = _weighted_solve(n, own, connected)
    logs = np.linalg.lstsq(a, b, rcond=None)[0]
    logs[~connected] = np.nan
    # Residual SD over every pair whose frames both have sizes, as before.
    sized = [k for k, p in enumerate(pairs) if connected[p.i] and connected[p.j]]
    resid = np.array(
        [(logs[pairs[k].j] - logs[pairs[k].i]) - pairs[k].log_scale for k in sized]
    )
    return LogSizes(logs, connected, float(resid.std()) if len(resid) else float("nan"))


def line_direction(line: Optional[Sequence[float]], dots: np.ndarray) -> np.ndarray:
    """Unit direction along the dive line ``a x + b y + c = 0``, or through
    the dots when there is none; oriented to negative y, where t grows away
    from the vanishing point on FishSense Lite mounts (slate_unknown.frame)."""
    if line is not None:
        a, b = float(line[0]), float(line[1])
        n = np.array([a, b]) / np.hypot(a, b)
        u = np.array([n[1], -n[0]])
    else:
        xy = np.asarray(dots, float)
        u = np.linalg.svd(xy - xy.mean(0))[2][0]
    return u if u[1] < 0 else -u


def _objective(t: np.ndarray, s: np.ndarray):
    return lambda tv: np.var(np.log(s / (t - tv)))


def fit_vanishing_position(t: np.ndarray, s: np.ndarray) -> float:
    """The ``t_v`` that makes ``s / (t - t_v)`` most nearly constant
    (slate_unknown.fit_tv): bounded, from 4000 px below to 5 px below the
    nearest dot."""
    from scipy.optimize import (  # pylint: disable=import-outside-toplevel
        minimize_scalar,
    )

    t, s = np.asarray(t, float), np.asarray(s, float)
    hi = t.min() - _BRACKET_GAP_PX
    return float(
        minimize_scalar(
            _objective(t, s), bounds=(hi - _BRACKET_PX, hi), method="bounded"
        ).x
    )


def deleted_residuals(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Each frame's ``|log s - log(t - t_v) - c|`` under the fit of the other
    frames (`t_v` and the median offset `c` without it); infinite for a frame
    at or below that fit's vanishing position."""
    t, s = np.asarray(t, float), np.asarray(s, float)
    log_s = np.log(s)
    out = np.empty(len(t))
    for i in range(len(t)):  # pylint: disable=consider-using-enumerate
        rest = np.arange(len(t)) != i
        tv = fit_vanishing_position(t[rest], s[rest])
        c = np.median(log_s[rest] - np.log(t[rest] - tv))
        gap = t[i] - tv
        out[i] = np.inf if gap <= 0 else abs(log_s[i] - np.log(gap) - c)
    return out


def _sizes_of(pairs: Sequence[PairScale], keep: np.ndarray):
    """`keep`'s connected frames and their sizes, from their own pairs only."""
    index = {int(f): k for k, f in enumerate(keep)}
    own = [
        PairScale(index[p.i], index[p.j], p.log_scale, p.near)
        for p in pairs
        if p.i in index and p.j in index
    ]
    sizes = solve_log_sizes(len(keep), own)
    return keep[sizes.connected], np.exp(sizes.log_sizes[sizes.connected])


def trim_frames(
    t: np.ndarray,
    pairs: Sequence[PairScale],
    keep: np.ndarray,
    s: np.ndarray,
    max_residual: float = FRAME_RESIDUAL_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """``(kept frames, their sizes)``: while the frame with the largest
    deleted residual is beyond `max_residual`, drop it and re-solve the sizes
    from the rest's pairs. `t` is every frame's place on the line, `keep`
    the frames in play with sizes `s`."""
    t = np.asarray(t, float)
    keep, s = np.asarray(keep), np.asarray(s, float)
    while len(keep) > 2:
        r = deleted_residuals(t[keep], s)
        worst = int(np.argmax(r))
        if r[worst] <= max_residual:
            break
        keep, s = _sizes_of(pairs, np.delete(keep, worst))
    return keep, s


def bootstrap_se(
    t: np.ndarray,
    s: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    draws: int = BOOTSTRAP_DRAWS,
) -> float:
    """SD of `fit_vanishing_position` over frame resamples (px)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    t, s = np.asarray(t, float), np.asarray(s, float)
    fits = [
        fit_vanishing_position(t[i], s[i])
        for i in (rng.integers(0, len(t), len(t)) for _ in range(draws))
    ]
    return float(np.std(fits))


def beam_from_vanishing(
    vanishing_px: float,
    direction: Sequence[float],
    line_offset_px: float,
    camera_matrix,
    o_mag_m: float = O_DESIGN_M,
) -> tuple[np.ndarray, np.ndarray]:
    """``(laser_position, laser_axis)`` from the vanishing position along the
    line (cscw score.py `calibrations()`): the vanishing point is
    ``v = t_v u + offset n``; the axis is ``K^-1 [v, 1]``, unit; the origin is
    ``|O|`` along the locus direction in normalised coordinates, in the
    camera's plane (z = 0)."""
    k = np.asarray(camera_matrix, float)
    u = np.asarray(direction, float)
    n = np.array([-u[1], u[0]])
    v = vanishing_px * u + line_offset_px * n
    axis = np.linalg.solve(k, [v[0], v[1], 1.0])
    o = np.array([u[0] / k[0, 0], u[1] / k[1, 1]])
    origin = np.append(o_mag_m * o / np.linalg.norm(o), 0.0)
    return origin, axis / np.linalg.norm(axis)


def _pinned(t: np.ndarray, tv: float) -> bool:
    """The optimum is the search bracket's edge, not a minimum."""
    hi = float(np.min(t)) - _BRACKET_GAP_PX
    return tv > hi - _AT_BOUND_PX or tv < hi - _BRACKET_PX + _AT_BOUND_PX


def _refused(reason: str, o_mag_m: float, **seen) -> LabelFreeFit:
    fields = {
        "laser_position": None, "laser_axis": None, "vanishing_px": None,
        "line_direction": None, "line_offset_px": None, "used": [],
        "frames_used": 0, "pairs": 0, "size_ratio": None, "se_px": None,
        "pair_residual_sd": None, "o_mag_m": o_mag_m,
    }  # fmt: skip
    fields.update(seen)
    return LabelFreeFit(outcome="refused", refusal_reason=reason, **fields)


# One refusal per gate, and the gates' thresholds are the caller's to set.
# pylint: disable-next=too-many-arguments,too-many-locals,too-many-return-statements
def fit_label_free(
    *,
    dots,
    pairs: Sequence[PairScale],
    camera_matrix,
    line: Optional[Sequence[float]] = None,
    min_frames: int = MIN_FRAMES,
    min_size_ratio: float = MIN_SIZE_RATIO,
    o_mag_m: float = O_DESIGN_M,
    rng: np.random.Generator | None = None,
) -> LabelFreeFit:
    """A dive's label-free calibration from frames of one rigid object with
    the dot on it: their dots (rectified px) and the pairwise size ratios.

    Refused (no beam) with too few matched pairs (fewer than frames, as
    register.score skips), too few connected frames, too little size spread,
    or an optimum at the search bracket's edge."""
    dots = np.asarray(dots, float).reshape(-1, 2)
    if len(pairs) < len(dots) or not pairs:
        return _refused("too_few_matched_pairs", o_mag_m, pairs=len(pairs))
    sizes = solve_log_sizes(len(dots), pairs)
    used = np.flatnonzero(sizes.connected)
    seen = {
        "used": [int(i) for i in used],
        "frames_used": len(used),
        "pairs": len(pairs),
        "pair_residual_sd": sizes.residual_sd,
    }
    if len(used) < min_frames:
        return _refused("too_few_frames", o_mag_m, **seen)
    s = np.exp(sizes.log_sizes[used])
    seen["size_ratio"] = float(s.max() / s.min())
    if seen["size_ratio"] < min_size_ratio:
        return _refused("too_little_range_spread", o_mag_m, **seen)
    u = line_direction(line, dots[used])
    pinned = _pinned(dots[used] @ u, fit_vanishing_position(dots[used] @ u, s))
    start, start_s = used, s
    used, s = trim_frames(dots @ u, pairs, start, start_s)
    if line is None:
        # The direction came from every frame, the one the trim dropped
        # included: refit it from the kept frames and trim again from the
        # start until the kept set stops changing (a few rounds at most; each
        # round can only drop frames the previous direction kept).
        for _ in range(len(start)):
            u = line_direction(None, dots[used])
            again, again_s = trim_frames(dots @ u, pairs, start, start_s)
            if np.array_equal(again, used):
                break
            used, s = again, again_s
        else:
            u = line_direction(None, dots[used])
    seen.update(used=[int(i) for i in used], frames_used=len(used))
    if len(used) < min_frames:
        return _refused("unconstrained" if pinned else "too_few_frames", o_mag_m, **seen)
    seen["size_ratio"] = float(s.max() / s.min())
    if seen["size_ratio"] < min_size_ratio:
        return _refused("too_little_range_spread", o_mag_m, **seen)
    xy = dots[used]
    t = xy @ u
    tv = fit_vanishing_position(t, s)
    offset = float(np.median(xy @ np.array([-u[1], u[0]])))
    seen.update(
        vanishing_px=tv, line_direction=[float(c) for c in u], line_offset_px=offset
    )
    if _pinned(t, tv):
        return _refused("unconstrained", o_mag_m, **seen)
    origin, axis = beam_from_vanishing(tv, u, offset, camera_matrix, o_mag_m)
    return LabelFreeFit(
        outcome="accepted",
        refusal_reason=None,
        laser_position=[float(c) for c in origin],
        laser_axis=[float(c) for c in axis],
        se_px=bootstrap_se(t, s, rng=rng),
        o_mag_m=o_mag_m,
        **seen,
    )
