"""What an image enhancer is, and what it is not allowed to do.

An enhancer that ships in front of this pipeline is **strictly pixel-wise**.
Measurements here come from pixel *coordinates*, not pixel values: Label Studio
percentages resolve through ``original_width``/``original_height`` into
rectified pixels, and those are back-projected against the laser ray. A
value-only enhancer is therefore geometrically free. One that shifts an edge by
a pixel is not — 1 px of laser-dot error is 0.75% length error, and head/tail
keypoints are clicked on edges.

That is mechanically checkable, so it is checked rather than reviewed.
:func:`guard` costs two attribute reads per call and fires at the moment of the
violation, naming the enhancer, instead of surfacing later as an unexplained
bias. :func:`probe_geometry` runs once per enhancer and catches what a shape
check structurally cannot.

What is deliberately *not* forbidden is value locality. CLAHE reads a
neighbourhood, and CLAHE is a legitimate operator. "Pixel-wise" means no pixel
moves, not that no pixel may consult its neighbours.

Nothing in :mod:`fishsense_core` calls this yet. It is the vocabulary an
enhancer has to be expressed in before one is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "IDENTITY",
    "Enhancer",
    "GeometryViolation",
    "MAX_DISPLACEMENT_PX",
    "ProbeResult",
    "WARN_DISPLACEMENT_PX",
    "guard",
    "probe_geometry",
]

#: An enhancement function: takes a BGR array, returns one of identical shape
#: and dtype. Called with uint8 at the rectified/JPEG stage and uint16 at the
#: linear-raw stage, so an enhancer that cares must branch on ``dtype``.
Enhancer = Callable[[np.ndarray], np.ndarray]


class GeometryViolation(RuntimeError):
    """An enhancer changed the frame's geometry or dtype.

    Raised rather than warned: a geometry change invalidates every coordinate
    comparison downstream of it, so continuing would produce numbers that look
    fine and mean nothing.
    """


def IDENTITY(image: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """The baseline enhancer. Named so reports print something better than
    ``<lambda>``."""
    return image


def guard(enhancer: Enhancer, *, name: str) -> Enhancer:
    """Wrap ``enhancer`` so it cannot silently change geometry or dtype.

    Also defends the *control*: when two arms score from the same decoded
    array, an enhancer that mutates its input in place corrupts the baseline.
    That failure mode reads as the enhancement helping, which is the single
    most expensive way to be wrong, so the input is handed over read-only.
    """

    def _guarded(image: np.ndarray) -> np.ndarray:
        source = image
        # A read-only view makes an in-place write raise instead of corrupting
        # the control. Enhancers that legitimately need to write must copy
        # first, which is the correct discipline anyway. numpy's own error is a
        # bare ValueError from deep inside the enhancer, so it is translated
        # here into the rule that was actually broken.
        view = source.view()
        view.flags.writeable = False
        try:
            out = enhancer(view)
        except ValueError as exc:
            if "read-only" not in str(exc):
                raise
            raise GeometryViolation(
                f"enhancer {name!r} wrote to its input in place; when two arms "
                "score from the same decoded array this corrupts the control "
                "and reads as the enhancement helping. Copy first."
            ) from exc

        if not isinstance(out, np.ndarray):
            raise GeometryViolation(
                f"enhancer {name!r} returned {type(out).__name__}, not an ndarray"
            )
        if out.shape != source.shape:
            raise GeometryViolation(
                f"enhancer {name!r} changed shape {source.shape} -> {out.shape}; "
                "enhancement must be strictly pixel-wise (no warp, resize, crop, "
                "transpose, or channel change) because every measurement in this "
                "pipeline is a pixel coordinate"
            )
        if out.dtype != source.dtype:
            raise GeometryViolation(
                f"enhancer {name!r} changed dtype {source.dtype} -> {out.dtype}; "
                "the container's full scale is load-bearing (a chromaticity "
                "normalisation picks 255 vs 65535 off it, and cv2.imencode needs "
                "uint8)"
            )

        # Hand the consumer a writeable array it can safely own.
        #
        # This is not tidiness — it is a segfault. The read-only view above is
        # what lets `guard` catch an in-place enhancer, but an enhancer that
        # legitimately returns its input unchanged (IDENTITY, i.e. every
        # baseline arm) returns that same read-only view. OpenCV does not
        # respect numpy's WRITEABLE flag and writes into the buffer anyway, and
        # the frame then crashes the process inside the laser detector —
        # verified: passing a read-only uint16 frame to it dumps core with no
        # traceback. Copying costs one array on the arms that alias their
        # input, and nothing on the arms that already allocate.
        if not out.flags.writeable or out.base is source or out.base is view:
            out = out.copy()
        return out

    _guarded.__name__ = f"guarded[{name}]"
    return _guarded


# ---------------------------------------------------------------------------
# The locality probe
# ---------------------------------------------------------------------------

#: Displacement above which the probe refuses the enhancer outright, in pixels.
#:
#: Calibrated against measurement, not chosen. Over identity, a halve, a 5 px
#: and a 31 px Gaussian, and CLAHE, the measured displacement is <= 0.074 px —
#: a symmetric neighbourhood filter preserves the response centroid however
#: wide it is. A one-pixel roll measures 1.01, a square transpose 7.07, a
#: horizontal flip 20.99. Nothing legitimate observed lands between 0.1 and 1.0.
MAX_DISPLACEMENT_PX = 0.5

#: Displacement above which the probe reports the enhancer as suspicious but
#: still allows it. Resample round trips land here — 0.07 px at 0.5x, 0.16 at
#: 0.33x, 0.34 at 0.7x. Those are a loss of detail rather than a displacement
#: of it, so they are surfaced rather than refused.
WARN_DISPLACEMENT_PX = 0.1

#: Side length of the synthetic probe frame.
#:
#: 384, not 64, and the difference is load-bearing for tile-adaptive
#: enhancers. CLAHE derives a separate mapping per tile (skimage's default tile
#: is 1/8 of each axis), so on a small probe the perturbation spans several
#: tiles and the response is asymmetric enough to shift the centroid ~0.8 px —
#: refusing a legitimate operator. Measured across probe sizes, CLAHE reads
#: 0.82 / 0.76 / 0.72 / 0.000 px at 64 / 128 / 256 / 384 while a one-pixel roll
#: reads 1.00 at every size. The artefact is the probe's, and it resolves at
#: 384.
DEFAULT_PROBE_SIZE = 384


@dataclass(frozen=True)
class ProbeResult:
    """What the locality probe measured.

    ``displacement_px`` belongs next to the enhancer's name wherever its
    results are reported: it is the geometric assurance behind them.
    """

    name: str
    displacement_px: float
    suspicious: bool


def _response_centroid(field: np.ndarray) -> tuple[float, float] | None:
    """Magnitude-weighted centroid of a response field, or None if it is flat.

    The 10% floor drops the far tail of a wide kernel, which is
    noise-dominated and would jitter the centroid.
    """
    peak = float(field.max())
    if peak <= 0:
        return None
    strong = field >= peak * 0.10
    ys, xs = np.nonzero(strong)
    weights = field[ys, xs].astype(float)
    total = weights.sum()
    if total <= 0:
        return None
    return float((xs * weights).sum() / total), float((ys * weights).sum() / total)


def probe_geometry(  # pylint: disable=too-many-locals
    enhancer: Enhancer,
    *,
    name: str,
    max_displacement_px: float = MAX_DISPLACEMENT_PX,
    size: int = DEFAULT_PROBE_SIZE,
    dtype: "np.dtype | type" = np.uint8,
) -> ProbeResult:
    """Assert ``enhancer`` does not move pixels. Run once per enhancer.

    :func:`guard` compares shape and dtype, which catches a resize, a crop and
    a transpose of a non-square frame. It structurally cannot catch the three
    that matter most — a square-frame transpose, a one-pixel roll, and a
    resample round trip — all of which preserve shape and dtype exactly while
    moving labelled coordinates.

    The probe perturbs a small off-centre patch of an otherwise textured input
    and measures where the output changed in response. A value-only or
    symmetric-neighbourhood enhancer answers at the perturbation; a warp
    answers somewhere else. It works for any enhancer whose output depends on
    its input, and it is indifferent to what the enhancer does to values —
    only to where the consequences land.

    What it does *not* measure is resampling loss. A downsample-and-back blurs
    without displacing, so it passes with ``suspicious`` set rather than
    raising.

    Returns:
        A :class:`ProbeResult` carrying the measured displacement.

    Raises:
        GeometryViolation: if the response is displaced past
            ``max_displacement_px``.
    """
    rng = np.random.default_rng(0xF15E)
    info = np.iinfo(dtype)
    # Mid-grey with light texture, plus one bright block. A flat field makes a
    # histogram method degenerate, and a warp needs gradient to reveal itself.
    base = rng.integers(
        int(info.max * 0.40), int(info.max * 0.60), size=(size, size, 3)
    ).astype(dtype)
    half = max(3, size // 9)
    by0, bx0 = size // 6, int(size * 0.62)
    block = (slice(by0, by0 + 2 * half), slice(bx0, bx0 + 2 * half))
    base[block] = int(info.max * 0.85)

    # The perturbation SWAPS two patches rather than brightening one, which
    # keeps the frame's histogram bit-identical.
    #
    # That matters for a whole class of enhancer. A *global* operator — a
    # percentile stretch, an auto-gamma — derives its mapping from the frame's
    # histogram. Brighten one patch and the mapping itself changes, so every
    # pixel's output moves and the response says nothing about geometry;
    # worse, for a stretch the largest changes land at the histogram's
    # extremes, wherever those happen to be, and the measured "displacement" is
    # an artefact of where the image is bright. With the histogram held fixed a
    # global operator applies the identical mapping to both frames, so the only
    # response is where the content actually moved — which is the question.
    cy, cx = size // 4, size // 3
    patch = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    perturbed = base.copy()
    perturbed[patch], perturbed[block] = base[block].copy(), base[patch].copy()

    guarded = guard(enhancer, name=name)
    before = guarded(base).astype(np.int64)
    after = guarded(perturbed).astype(np.int64)

    # Where the input changed, and where the output responded. Comparing the
    # two is what makes this independent of where the patches happen to sit.
    expected = _response_centroid(
        np.abs(perturbed.astype(np.int64) - base.astype(np.int64)).sum(axis=2)
    )
    observed = _response_centroid(np.abs(after - before).sum(axis=2))
    if expected is None or observed is None:
        # The enhancer's output does not depend on its input here — a
        # constant, or a no-op on this synthetic field. Nothing to conclude
        # about geometry, and not this function's business to complain.
        return ProbeResult(name=name, displacement_px=0.0, suspicious=False)

    displacement = float(
        math.hypot(observed[0] - expected[0], observed[1] - expected[1])
    )
    if displacement > max_displacement_px:
        raise GeometryViolation(
            f"enhancer {name!r} moved pixels: the input changed around "
            f"({expected[0]:.2f}, {expected[1]:.2f}) but the output responded "
            f"around ({observed[0]:.2f}, {observed[1]:.2f}), a displacement of "
            f"{displacement:.2f} px (limit {max_displacement_px}). Enhancement "
            "must be strictly pixel-wise — 1 px of laser-dot displacement is "
            "0.75% length error, and head/tail keypoints are clicked on edges."
        )
    return ProbeResult(
        name=name,
        displacement_px=displacement,
        suspicious=displacement > WARN_DISPLACEMENT_PX,
    )
