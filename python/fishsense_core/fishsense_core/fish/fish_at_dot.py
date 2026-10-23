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
is unchanged and three of its choices are measured:

* **crop, don't tile**: one :data:`CROP_WIDTH` x :data:`CROP_HEIGHT` window
  centred on the *first* laser dot, clamped inside the frame (never padded
  or truncated); the gate tries every dot, first hit wins;
* **SAM 3.1** prompted :data:`SAM3_PROMPT`, behind the :class:`Segmenter`
  seam, with the embedded Mask R-CNN (:class:`FishSegmentation`) as the
  no-GPU fallback;
* the head and tail are fishsense-core's own :class:`FishHeadTailDetector`
  on the kept mask (the "SAM rule"), lifted back into frame pixels.

What this module does **not** do is the service's job and stays there: decode
the JPEG, map a status to a database row, name a laser *label* (a caller gets
the dot's index and looks the label up), retry policy, and GPU detection.

Hard-won rules, kept: SAM 3.1 runs under ``torch.autocast(bfloat16)`` entered
as a context manager (its weights are bfloat16 and ``Sam3Processor`` sets up
no autocast; without one every frame dies in ``vitdet.forward`` with a dtype
mismatch, which in production looped on all 356 images of dive 94 on
2026-09-08), on a PIL image (an HWC array yields three-pixel-wide masks);
tensors are detached and moved to the CPU before numpy; the Mask R-CNN
fallback is fed BGR, landscape crops only, and its instance map is split into
binary masks.

The geometry (:func:`crop_origin`, :func:`lift_point`, :func:`mask_at_point`,
:func:`mask_box`, :func:`silhouette_ratio`) is kept as separate functions
because these are the parts that fail *plausibly*: a wrong crop origin
displaces every keypoint by a constant and still produces a fish-shaped
answer.
"""

from __future__ import annotations

# The geometry signatures and the result are the ported module's, kept
# argument-for-argument so the service's call sites and tests read across; the
# kernel's locals are the stage's steps. cv2 is a C extension pylint cannot
# introspect (no-member); torch, PIL and sam3 load lazily by design.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,too-many-locals,no-member

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

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
    "Segmenter",
    "crop_origin",
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


def lift_point(point: Sequence[float], origin_x: int, origin_y: int) -> Tuple[float, float]:
    """Move a crop-local point back into rectified-frame coordinates, the
    space the laser labels and a labeller's clicks are in."""
    return (float(point[0]) + origin_x, float(point[1]) + origin_y)


def mask_at_point(
    masks: Iterable[np.ndarray], points: Sequence[Sequence[float]]
) -> Optional[np.ndarray]:
    """The laser gate: the first mask whose pixel at a laser dot is set. Every
    dot is tried, first hit wins. None (an abstention, not an error) when no
    dot lands on any mask."""
    masks = list(masks)
    for px, py in points:
        xi, yi = int(round(px)), int(round(py))
        for mask in masks:
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                return mask
    return None


def _dot_on_mask(points: Sequence[Sequence[float]], binary: np.ndarray) -> Optional[int]:
    """Index of the first dot that lands on ``binary`` (crop-local), if any."""
    for index, (px, py) in enumerate(points):
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < binary.shape[0] and 0 <= xi < binary.shape[1] and binary[yi, xi]:
            return index
    return None


def silhouette_ratio(mask_area_px: int, length_px: float) -> Optional[float]:
    """``mask_area / length**2``: how fish-shaped a detection is (a real fish
    runs ~0.15-0.30). Recorded on every result so a caller can gate on it
    later without re-predicting. None for a degenerate length."""
    if not length_px:
        return None
    return mask_area_px / (length_px * length_px)


def mask_box(binary: np.ndarray, origin_x: int, origin_y: int) -> Optional[List[int]]:
    """The mask's box, ``[x_min, y_min, x_max, y_max)``, lifted into frame
    pixels like the keypoints. What a species or keypoint stage crops by.
    None for an empty mask."""
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    return [
        int(xs.min()) + origin_x,
        int(ys.min()) + origin_y,
        int(xs.max()) + 1 + origin_x,
        int(ys.max()) + 1 + origin_y,
    ]


# ── segmenters ──────────────────────────────────────────────────────────────


class Segmenter(Protocol):  # pylint: disable=too-few-public-methods
    """Anything that turns a BGR crop into crop-local binary masks. Production
    passes :class:`Sam3Segmenter` or :class:`FishialSegmenter`; tests a stub."""

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Instance masks of ``image_bgr``, each ``(H, W)`` and truthy on the
        instance. Empty when nothing was found."""


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
        # pylint: disable=import-outside-toplevel,import-error
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
        return [to_numpy(m).squeeze() for m in masks]


class FishialSegmenter:  # pylint: disable=too-few-public-methods
    """The embedded Mask R-CNN (:class:`fishsense_core.fish.FishSegmentation`)
    behind the :class:`Segmenter` seam: the fallback where there is no GPU.
    BGR in, landscape crops only, its instance label map split into binary
    masks."""

    def __init__(self, segmentation: Any):
        self._segmentation = segmentation

    @classmethod
    def load(cls) -> "FishialSegmenter":
        """Build and load the embedded model. ``load_model`` is called here,
        before the segmenter is handed out: unloaded, ``inference`` raises
        on every frame."""
        # pylint: disable-next=import-outside-toplevel
        from fishsense_core.fish import FishSegmentation

        segmentation = FishSegmentation()
        segmentation.load_model()
        return cls(segmentation)

    def segment(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Binary masks of each instance in a landscape BGR crop; ``[]`` for
        a portrait or square one, which the model returns empty for anyway."""
        height, width = image_bgr.shape[:2]
        if width <= height:
            _log.warning(
                "fallback segmenter given a %dx%d (non-landscape) crop; "
                "FishSegmentation returns an empty mask for these",
                width,
                height,
            )
            return []
        labels = np.asarray(self._segmentation.inference(image_bgr))
        ids = [int(i) for i in np.unique(labels) if int(i) != 0]
        return [(labels == i) for i in ids]


# ── the stage ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FishAtDot:
    """What the stage found for one frame. Points and the box are in frame
    pixels; the mask is crop-local (offset by ``(crop_x, crop_y)``)."""

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
    #: The kept mask, ``(crop_h, crop_w)`` uint8 in {0, 1}; None when none was.
    mask: Optional[np.ndarray] = None
    mask_bbox: Optional[List[int]] = None
    mask_area_px: Optional[int] = None
    head: Optional[Tuple[float, float]] = None
    tail: Optional[Tuple[float, float]] = None
    silhouette_ratio: Optional[float] = None

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
    head_tail_detector: Any = None,
) -> FishAtDot:
    """Crop, segment, gate, keypoint, and lift back to frame pixels.

    Args:
        frame_bgr: The rectified frame, ``(H, W, 3)`` uint8 BGR: the exact
            image the labeller is shown.
        laser_points: Laser dots in frame pixels, ``[(x, y), ...]``. The crop
            is centred on the first; every one is tried at the gate.
        segmenter: A :class:`Segmenter`.
        crop_w: Window size; the defaults are production's.
        crop_h: Window size; the defaults are production's.
        head_tail_detector: Anything with ``find_head_tail_img(mask)``;
            defaults to :class:`fishsense_core.fish.FishHeadTailDetector`.

    Returns:
        A :class:`FishAtDot`. Abstentions are statuses, not exceptions: a
        decoded frame always yields a result.
    """
    height, width = frame_bgr.shape[:2]
    if not laser_points:
        return FishAtDot(STATUS_LASER_OFF_ALL_FISH, width, height, 0, 0)

    origin_x, origin_y = crop_origin(
        laser_points[0][0], laser_points[0][1], width, height, crop_w, crop_h
    )
    crop = np.ascontiguousarray(
        frame_bgr[origin_y : origin_y + crop_h, origin_x : origin_x + crop_w]
    )
    masks = segmenter.segment(crop)
    # With no mask kept, the answer came from the crop, centred on the first
    # dot, so the abstention names it.
    placed = {"width": width, "height": height, "crop_x": origin_x, "crop_y": origin_y, "dot": 0}
    if not masks:
        return FishAtDot(STATUS_NO_DETECTIONS, **placed)

    local_points = [(px - origin_x, py - origin_y) for px, py in laser_points]
    mask = mask_at_point(masks, local_points)
    if mask is None:
        return FishAtDot(STATUS_LASER_OFF_ALL_FISH, **placed)

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    return _keypoint(
        binary,
        origin=(origin_x, origin_y),
        frame_size=(width, height),
        local_points=local_points,
        head_tail_detector=head_tail_detector,
    )


def _keypoint(
    binary: np.ndarray,
    *,
    origin: Tuple[int, int],
    frame_size: Tuple[int, int],
    local_points: Sequence[Sequence[float]],
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
        "dot": _dot_on_mask(local_points, binary),
        "mask": binary,
        # The kept mask's box, on a failure too: the fish is known even when
        # its keypoints are not.
        "mask_bbox": mask_box(binary, origin_x, origin_y),
        "mask_area_px": area,
    }

    if head_tail_detector is None:
        # pylint: disable-next=import-outside-toplevel
        from fishsense_core.fish import FishHeadTailDetector

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
