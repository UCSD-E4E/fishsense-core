"""The lasered fish: one mask at the laser dot, keypointed.

This is the head/tail stage every FishSense pipeline runs between the laser
dot and the length: cut one window around the first dot, segment it, keep
the mask a dot lands on, and keypoint that mask. It used to live, copy by
copy, in fishsense-lite's workflow worker, fishsense-services' processor
(``headtail_predict``), the coral-gardeners detector and the head/tail
keypoint experiment (``UCSD-E4E/2026-10-22_headtail_keypoints``), each
importing the previous one's private names to be sure of seeing *the same
fish*. Ported from fishsense-services' ``headtail_predict/predict.py`` and
``geometry.py`` (themselves ported from fishsense-lite@77e8f8e5); behaviour
is unchanged and its choices are measured:

* **crop, don't tile**: one :data:`CROP_WIDTH` x :data:`CROP_HEIGHT` window
  centred on the *first* laser dot, clamped inside the frame (never padded
  or truncated); the gate tries every dot, first hit wins;
* **SAM 3.1** prompted :data:`SAM3_PROMPT`, behind the :class:`Segmenter`
  seam, with the embedded Mask R-CNN (:class:`FishSegmentation`) as the
  no-GPU fallback;
* the head and tail are fishsense-core's own :class:`FishHeadTailDetector`
  on the kept mask (the "SAM rule"), lifted back into frame pixels;
* **a fish longer than the window** (``grow_cut_masks=True``, the automatic
  chain's since 2026-10-22): a kept mask that runs into an edge of the window
  that is not the frame's own is segmented again on the whole frame, because
  the cut mask's ends, and the head and tail read off them, are the window's
  (lab mission 383: a 686 mm grouper cut at both ends measured 554 mm). The
  cut mask is never kept: the whole frame answers, or the frame abstains.
  Off by default, as for the human pre-fill path, whose output is pinned.
* **SAM's score as a gate** (``score_gate``, the automatic chain's; paper
  §6.3, fishsense-services' ``AUTOMATIC_SAM_SCORE_GATE`` is 0.5): masks
  scoring under it are dropped *before* the laser gate, so a weak mask under
  the dot is no fish and a confident one off it does not count. The kept
  mask's score is reported on :attr:`FishAtDot.score` whenever the segmenter
  scores its masks (:class:`ScoredSegmenter`), gate or not; with
  ``grow_cut_masks`` it is the whole frame's mask's. Off by default.

What this module does **not** do is the service's job and stays there: decode
the JPEG, map a status to a database row, name a laser *label* (a caller gets
the dot's index and looks the label up), retry policy, and GPU detection.

Hard-won rules, kept: SAM 3.1 runs under ``torch.autocast(bfloat16)`` entered
as a context manager (its weights are bfloat16 and ``Sam3Processor`` sets up
no autocast; without one every frame dies in ``vitdet.forward`` with a dtype
mismatch, which in production looped on all 356 images of dive 94 on
2026-09-08), on a PIL image (an HWC array yields three-pixel-wide masks);
tensors are detached and moved to the CPU before numpy; the Mask R-CNN
fallback is fed BGR and its instance map is split into binary masks.

The geometry (:func:`crop_origin`, :func:`cut_by_window`, :func:`lift_point`,
:func:`mask_at_point`, :func:`mask_box`, :func:`silhouette_ratio`) is kept as
separate functions because these are the parts that fail *plausibly*: a wrong
crop origin displaces every keypoint by a constant and still produces a
fish-shaped answer.
"""

from __future__ import annotations

# The geometry signatures and the result are the ported module's, kept
# argument-for-argument so the service's call sites and tests read across; the
# kernel's locals are the stage's steps.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,too-many-locals

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from fishsense_core.fish import FishHeadTailDetector, FishSegmentation

__all__ = [
    "CROP_HEIGHT",
    "CROP_WIDTH",
    "SAM3_PROMPT",
    "STATUS_HEADTAIL_FAILED",
    "STATUS_LASER_OFF_ALL_FISH",
    "STATUS_NO_DETECTIONS",
    "STATUS_PREDICTED",
    "FishAtDot",
    "FishialSegmenter",
    "Sam3Segmenter",
    "ScoredSegmenter",
    "Segmenter",
    "crop_origin",
    "cut_by_window",
    "find_fish_at_dot",
    "lift_point",
    "mask_at_point",
    "mask_box",
    "silhouette_ratio",
    "to_numpy",
]

_log = logging.getLogger(__name__)

#: The window cut around the first laser dot, in rectified-frame pixels. One
#: window, not tiles: a fish under the dot fits, and SAM sees it at full
#: resolution.
CROP_WIDTH = 1800
CROP_HEIGHT = 1350

#: The concept SAM 3.1 is prompted with.
SAM3_PROMPT = "fish"

#: A mask was kept and keypointed.
STATUS_PREDICTED = "predicted"
#: The segmenter returned no masks at all.
STATUS_NO_DETECTIONS = "no_detections"
#: No dot landed on any mask (or there were no dots). An abstention, not an
#: error.
STATUS_LASER_OFF_ALL_FISH = "laser_off_all_fish"
#: A mask was kept but the head/tail detector could not fit it.
STATUS_HEADTAIL_FAILED = "headtail_failed"


# ── geometry ────────────────────────────────────────────────────────────────


def crop_origin(
    laser_x: float,
    laser_y: float,
    frame_w: int,
    frame_h: int,
    crop_w: int = CROP_WIDTH,
    crop_h: int = CROP_HEIGHT,
) -> Tuple[int, int]:
    """Top-left of the crop window centred on the laser dot, clamped inside
    the frame. Clamped rather than padded or truncated, which keeps the fish
    in view and the input shape constant."""
    max_x = max(0, frame_w - crop_w)
    max_y = max(0, frame_h - crop_h)
    ox = int(min(max(0, laser_x - crop_w // 2), max_x))
    oy = int(min(max(0, laser_y - crop_h // 2), max_y))
    return ox, oy


def cut_by_window(
    binary: np.ndarray, origin_x: int, origin_y: int, frame_w: int, frame_h: int
) -> bool:
    """Whether a crop-local mask runs into an edge of the window that is not
    the frame's own: the fish goes on past what the segmenter saw, so the
    mask's ends, and the head and tail read off them, are the window's. An
    edge the window was clamped to is the photo's, which no bigger window
    can see past."""
    h, w = binary.shape[:2]
    return bool(
        (origin_x > 0 and binary[:, 0].any())
        or (origin_x + w < frame_w and binary[:, -1].any())
        or (origin_y > 0 and binary[0, :].any())
        or (origin_y + h < frame_h and binary[-1, :].any())
    )


def lift_point(point: Sequence[float], origin_x: int, origin_y: int) -> Tuple[float, float]:
    """Move a crop-local point back into rectified-frame coordinates, the
    space the laser labels and a labeller's clicks are in."""
    return (float(point[0]) + origin_x, float(point[1]) + origin_y)


def _gate(
    masks: Iterable[np.ndarray], points: Sequence[Sequence[float]]
) -> Optional[Tuple[int, int]]:
    """The laser gate, once: ``(dot index, mask index)`` for the first dot
    that lands on any mask, trying dots in order and masks in order. None
    when none does. :func:`mask_at_point` and the kernel both read this, so
    the dot a result names, and the score it reports, are the ones of the
    mask it chose by construction."""
    masks = list(masks)
    for index, (px, py) in enumerate(points):
        xi, yi = int(round(px)), int(round(py))
        for k, mask in enumerate(masks):
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                return index, k
    return None


def mask_at_point(
    masks: Iterable[np.ndarray], points: Sequence[Sequence[float]]
) -> Optional[np.ndarray]:
    """The laser gate: the first mask whose pixel at a laser dot is set. Every
    dot is tried, first hit wins. None (an abstention, not an error) when no
    dot lands on any mask."""
    masks = list(masks)
    hit = _gate(masks, points)
    return None if hit is None else masks[hit[1]]


def silhouette_ratio(mask_area_px: int, length_px: float) -> Optional[float]:
    """``mask_area / length**2``: how fish-shaped a detection is (a real fish
    runs ~0.15-0.30). Recorded on every result so a caller can gate on it
    later without re-predicting. None for a degenerate length."""
    if not length_px:
        return None
    return mask_area_px / (length_px * length_px)


def mask_box(
    binary: np.ndarray, origin_x: int, origin_y: int
) -> Optional[Tuple[int, int, int, int]]:
    """The mask's box, ``(x_min, y_min, x_max, y_max)``, half-open, lifted
    into frame pixels like the keypoints. What a species or keypoint stage
    crops by. None for an empty mask. A tuple, so a result that carries it
    stays hashable."""
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    return (
        int(xs.min()) + origin_x,
        int(ys.min()) + origin_y,
        int(xs.max()) + 1 + origin_x,
        int(ys.max()) + 1 + origin_y,
    )


# ── segmenters ──────────────────────────────────────────────────────────────


class Segmenter(Protocol):  # pylint: disable=too-few-public-methods
    """Anything that turns a BGR image into image-local binary masks.
    Production passes :class:`Sam3Segmenter` or :class:`FishialSegmenter`;
    tests a stub. It is handed the window, and with ``grow_cut_masks`` the
    whole frame too."""

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Instance masks of ``image_bgr``, each ``(H, W)`` and truthy on the
        instance. Empty when nothing was found."""


class ScoredSegmenter(Segmenter, Protocol):  # pylint: disable=too-few-public-methods
    """A :class:`Segmenter` that also scores its masks. :func:`find_fish_at_dot`
    calls ``segment_scored`` instead of ``segment`` when a segmenter has it,
    which is what lets it gate on and report the kept mask's score."""

    def segment_scored(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, Optional[float]]]:
        """``(mask, score)`` per instance, the masks exactly ``segment``'s and
        in its order. A score is None when the backend gave none."""


def to_numpy(mask: Any) -> np.ndarray:
    """A mask as an ndarray, whatever the backend handed back: a device
    tensor is detached and moved to the CPU first (``np.asarray`` on a CUDA
    tensor raises)."""
    if getattr(mask, "detach", None) is not None:
        mask = mask.detach().cpu()
    return np.asarray(mask)


class Sam3Segmenter:  # pylint: disable=too-few-public-methods
    """SAM 3.1 behind the :class:`Segmenter` seam.

    ``processor`` is a ``Sam3Processor``, as :func:`fishsense_core.fish.sam3.build_segmenter`
    or :func:`~fishsense_core.fish.sam3.load_segmenter` returns one. ``torch``,
    ``PIL`` and ``sam3`` are the caller's to install (see :mod:`fishsense_core.fish.sam3`).
    """

    def __init__(self, processor: Any, prompt: str = SAM3_PROMPT):
        self._processor = processor
        self._prompt = prompt

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """One concept-prompted segmentation, under bfloat16 autocast entered
        as a context manager (the thread is reused), on a PIL image (an HWC
        array yields three-pixel-wide masks)."""
        return [mask for mask, _ in self.segment_scored(image_bgr)]

    def segment_scored(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, Optional[float]]]:
        """:meth:`segment`'s masks, each with SAM's score from
        ``state["scores"]``: one ``(N,)`` tensor, moved to the host once like
        the masks, as float32 (numpy has no bfloat16). Every score is None
        when SAM returned none; a count that does not match the masks
        raises, since pairing them would name the wrong mask's score."""
        # cv2 is a C extension pylint cannot introspect; torch and PIL are the
        # caller's and load on first use.
        # pylint: disable=import-outside-toplevel,import-error,no-member
        import cv2
        import PIL.Image
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type, dtype=torch.bfloat16):
            state = self._processor.set_image(PIL.Image.fromarray(rgb))
            state = self._processor.set_text_prompt(self._prompt, state)
        masks = state.get("masks") if hasattr(state, "get") else None
        if masks is None:
            return []
        if getattr(masks, "detach", None) is not None:
            # One (N, 1, H, W) tensor: one device-to-host transfer, not N.
            masks = to_numpy(masks)
        masks = [to_numpy(m).squeeze() for m in masks]
        scores = state.get("scores")
        if scores is None:
            return [(m, None) for m in masks]
        if getattr(scores, "float", None) is not None:
            scores = scores.float()
        # One (N,) tensor, one transfer, as for the masks.
        values = [float(s) for s in to_numpy(scores).reshape(-1)]
        if len(values) != len(masks):
            raise ValueError(f"SAM 3.1 returned {len(masks)} masks but {len(values)} scores")
        return list(zip(masks, values))


class FishialSegmenter:  # pylint: disable=too-few-public-methods
    """The embedded Mask R-CNN (:class:`fishsense_core.fish.FishSegmentation`)
    behind the :class:`Segmenter` seam: the fallback where there is no GPU.
    BGR in, any orientation (the native ``inference`` rotates a portrait
    image in and its mask back out), its instance label map split into binary
    masks."""

    def __init__(self, segmentation: Any):
        self._segmentation = segmentation

    @classmethod
    def load(cls) -> "FishialSegmenter":
        """Build and load the embedded model. ``load_model`` is called here,
        before the segmenter is handed out: unloaded, ``inference`` raises
        on every frame."""
        segmentation = FishSegmentation()
        segmentation.load_model()
        return cls(segmentation)

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Binary masks of each instance in a BGR image."""
        labels = np.asarray(self._segmentation.inference(image_bgr))
        ids = [int(i) for i in np.unique(labels) if int(i) != 0]
        return [(labels == i) for i in ids]


# ── the stage ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FishAtDot:
    """What the stage found for one frame. Points and the box are in frame
    pixels; the mask is local to ``(crop_x, crop_y)``, which is ``(0, 0)``
    when the whole frame answered."""

    status: str
    width: int
    height: int
    crop_x: int
    crop_y: int
    #: Index into the ``laser_points`` the stage was given of the dot the
    #: answer came from: the dot on the kept mask when one was kept, else
    #: the crop's centre, the first. None only when there were no dots. A
    #: caller that tracks laser *labels* uses it to name the one a later
    #: correction would make this result stale.
    dot: Optional[int] = None
    #: The kept mask, ``(h, w)`` uint8 in {0, 1}; None when none was. Left
    #: out of equality and hashing: an array has no truth value.
    mask: Optional[np.ndarray] = field(default=None, compare=False)
    mask_bbox: Optional[Tuple[int, int, int, int]] = None
    mask_area_px: Optional[int] = None
    head: Optional[Tuple[float, float]] = None
    tail: Optional[Tuple[float, float]] = None
    silhouette_ratio: Optional[float] = None
    #: The kept mask's segmenter score (SAM's), when the segmenter scores its
    #: masks (:class:`ScoredSegmenter`); with ``grow_cut_masks`` the whole
    #: frame's mask's. None when no mask was kept, or for a segmenter with no
    #: scores (:class:`FishialSegmenter`, a plain :class:`Segmenter`).
    score: Optional[float] = None

    @property
    def length_px(self) -> Optional[float]:
        """Head-to-tail distance in frame pixels, when both were found."""
        if self.head is None or self.tail is None:
            return None
        return float(np.hypot(self.head[0] - self.tail[0], self.head[1] - self.tail[1]))


def find_fish_at_dot(
    frame_bgr: np.ndarray,
    laser_points: Sequence[Sequence[float]],
    segmenter: Segmenter,
    *,
    crop_w: int = CROP_WIDTH,
    crop_h: int = CROP_HEIGHT,
    grow_cut_masks: bool = False,
    head_tail_detector: Any = None,
    score_gate: Optional[float] = None,
) -> FishAtDot:
    """Crop, segment, gate, keypoint, and lift back to frame pixels.

    Args:
        frame_bgr: The rectified frame, ``(H, W, 3)`` uint8 BGR: the exact
            image the labeller is shown.
        laser_points: Laser dots in frame pixels, ``[(x, y), ...]`` (a list
            or an ``(N, 2)`` array). The crop is centred on the first; every
            one is tried at the gate.
        segmenter: A :class:`Segmenter`.
        crop_w: Window size; the defaults are production's.
        crop_h: Window size; the defaults are production's.
        grow_cut_masks: Segment the whole frame when the kept mask is cut by
            the window (:func:`cut_by_window`). The automatic chain's; off for
            the human pre-fill, whose output is pinned.
        head_tail_detector: Anything with ``find_head_tail_img(mask)``;
            defaults to :class:`fishsense_core.fish.FishHeadTailDetector`.
        score_gate: Keep only masks scoring at least this (inclusive), before
            the laser gate and on both passes; no mask left is
            ``no_detections``. The automatic chain's (fishsense-services
            passes its ``AUTOMATIC_SAM_SCORE_GATE``, 0.5); None, the default,
            gates nothing. Needs a :class:`ScoredSegmenter` whose masks all
            carry a score: an unscored mask cannot pass a gate, so it raises
            (``TypeError`` / ``ValueError``) rather than abstaining.

    Returns:
        A :class:`FishAtDot`. Abstentions are statuses, not exceptions: a
        decoded frame always yields a result.
    """
    height, width = frame_bgr.shape[:2]
    if len(laser_points) == 0:
        return FishAtDot(STATUS_LASER_OFF_ALL_FISH, width, height, 0, 0)

    origin_x, origin_y = crop_origin(
        laser_points[0][0], laser_points[0][1], width, height, crop_w, crop_h
    )
    crop = np.ascontiguousarray(
        frame_bgr[origin_y : origin_y + crop_h, origin_x : origin_x + crop_w]
    )
    masks, scores = _segment(segmenter, crop, score_gate)
    # With no mask kept, the answer came from the crop, centred on the first
    # dot, so the abstention names it.
    placed = {"width": width, "height": height, "crop_x": origin_x, "crop_y": origin_y, "dot": 0}
    if not masks:
        return FishAtDot(STATUS_NO_DETECTIONS, **placed)

    local_points = [(px - origin_x, py - origin_y) for px, py in laser_points]
    hit = _gate(masks, local_points)
    if hit is None:
        return FishAtDot(STATUS_LASER_OFF_ALL_FISH, **placed)
    dot, kept = hit
    mask, score = masks[kept], scores[kept]

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if grow_cut_masks and cut_by_window(binary, origin_x, origin_y, width, height):
        # The fish runs on past the window. The cut mask's length is known to
        # be wrong, so it is never kept: the whole frame answers, or the
        # frame abstains.
        origin_x = origin_y = 0
        placed.update(crop_x=0, crop_y=0)
        local_points = [(float(px), float(py)) for px, py in laser_points]
        masks, scores = _segment(segmenter, np.ascontiguousarray(frame_bgr), score_gate)
        if not masks:
            return FishAtDot(STATUS_NO_DETECTIONS, **placed)
        hit = _gate(masks, local_points)
        if hit is None:
            return FishAtDot(STATUS_LASER_OFF_ALL_FISH, **placed)
        dot, kept = hit
        mask, score = masks[kept], scores[kept]
        binary = (np.asarray(mask) > 0).astype(np.uint8)

    return _keypoint(
        binary,
        origin=(origin_x, origin_y),
        frame_size=(width, height),
        dot=dot,
        score=score,
        head_tail_detector=head_tail_detector,
    )


def _segment(
    segmenter: Segmenter, image_bgr: np.ndarray, score_gate: Optional[float]
) -> Tuple[List[np.ndarray], List[Optional[float]]]:
    """One segmentation, as parallel ``(masks, scores)`` lists, the masks
    under ``score_gate`` dropped. A segmenter without ``segment_scored`` is
    called exactly as before and scores nothing."""
    segment_scored = getattr(segmenter, "segment_scored", None)
    if segment_scored is None:
        if score_gate is not None:
            raise TypeError(
                f"score_gate={score_gate} needs a segmenter with segment_scored; "
                f"{type(segmenter).__name__} scores nothing"
            )
        masks = list(segmenter.segment(image_bgr))
        return masks, [None] * len(masks)
    pairs = [(m, None if s is None else float(s)) for m, s in segment_scored(image_bgr)]
    if score_gate is not None:
        if any(s is None for _, s in pairs):
            raise ValueError(f"score_gate={score_gate} given but a mask came back unscored")
        pairs = [(m, s) for m, s in pairs if s >= score_gate]
    return [m for m, _ in pairs], [s for _, s in pairs]


def _keypoint(
    binary: np.ndarray,
    *,
    origin: Tuple[int, int],
    frame_size: Tuple[int, int],
    dot: int,
    score: Optional[float],
    head_tail_detector: Any,
) -> FishAtDot:
    """Keypoint one kept mask and lift the result into frame coordinates."""
    origin_x, origin_y = origin
    area = int(np.count_nonzero(binary))
    common = {
        "width": frame_size[0],
        "height": frame_size[1],
        "crop_x": origin_x,
        "crop_y": origin_y,
        "dot": dot,
        "score": score,
        "mask": binary,
        # The kept mask's box, on a failure too: the fish is known even when
        # its keypoints are not.
        "mask_bbox": mask_box(binary, origin_x, origin_y),
        "mask_area_px": area,
    }

    if head_tail_detector is None:
        head_tail_detector = FishHeadTailDetector()
    try:
        head, tail = head_tail_detector.find_head_tail_img(binary * 255)
    # A native call with no documented exception hierarchy; one unfittable
    # mask must not fail the caller's whole batch.
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:
        _log.warning("find_head_tail_img failed: %s", exc)
        return FishAtDot(STATUS_HEADTAIL_FAILED, **common)

    head_xy = lift_point(head, origin_x, origin_y)
    tail_xy = lift_point(tail, origin_x, origin_y)
    length = float(np.hypot(head_xy[0] - tail_xy[0], head_xy[1] - tail_xy[1]))
    return FishAtDot(
        STATUS_PREDICTED,
        head=head_xy,
        tail=tail_xy,
        silhouette_ratio=silhouette_ratio(area, length),
        **common,
    )
