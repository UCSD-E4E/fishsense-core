"""Snout/fork keypoints from a crop of the lasered fish: the SimCC model.

A top-down keypoint model in the RTMPose design, ported from
``UCSD-E4E/2026-10-22_headtail_keypoints`` (``scripts/kpnet.py`` and
``scripts/hk.py``), where it was trained on 19,654 human head/tail clicks
drawn without a pre-fill, from 117 dives, and scored on fish **length** against
the SAM rule (:func:`fishsense_core.fish.fish_at_dot.find_fish_at_dot`'s
head/tail on the mask). Run ``simcc_sam``, epoch 12, selected on validation
dives by a rule fixed before any test prediction:

* **Reef** (7 calibrated dives, agreement with the human pipeline): mean |d|
  5.5% -> 3.0%, with the paired difference -2.5 points per frame (95% CI -3.2
  to -1.8). Nearly all of the gain is at the fork (7.6% -> 1.2% of length) and
  in orientation (22% -> 0% swapped). The hogfish over-length of +4.7% goes
  away.
* **Pool** (10 fish-model dives, tape truth): **worse** under production's
  per-fish p90 estimator, 10.2% -> 15.4% (CI +2.2 to +7.8 points). Its frames
  are less biased, but a long over-length tail on two plastic models moves the
  p90; only 34 training frames come from the pool.

So the model should not replace the rule for fish models without either more
pool training data or an estimator matched to its error shape. See the
experiment's ``results/SUMMARY.md``.

**The crop rule is part of the model.** Every crop, in training, validation
and here, is a square centred on the SAM 3.1 mask box of the fish under the
laser dot (:func:`crop_from_box`): side = :data:`BOX_SCALE` x box diagonal +
:data:`CROP_PAD`, clipped to the frame. The box is the one
:class:`~fishsense_core.fish.fish_at_dot.FishAtDot` carries (``mask_bbox``), so
the model and the rule see the same fish chosen the same way. A different
crop is a different input distribution; ``test_headtail_keypoints.py`` pins
the rule to the experiment's stored crops.

**One input difference is unmeasured.** The experiment's crops were JPEG
files: cut from staged frames that were themselves JPEGs, and encoded again
(``sam_boxes.py``, ``cv2.imencode``). :meth:`HeadTailKeypointDetector.predict`
is fed the decoded rectified frame, with no compression at all. The numbers
above were measured on the double-JPEG input; the gap to production's input
has not been, and anyone swapping the model in on those numbers should
measure it (or feed it a JPEG round trip).

Input: the crop warped to :data:`IN` x :data:`IN` (longer side fills, grey
:data:`PAD_VALUE` outside). Backbone: timm ConvNeXt-Tiny, last stage (stride
32). Head (SimCC): per keypoint a classification over ``IN * K`` bins for x and
for y (1/:data:`K` px), decoded as the softmax-weighted mean in a
+-:data:`RADIUS`-bin window around the argmax. Keypoint 0 is the snout (the
human "head"), 1 the fork (the human "tail"); augmentation kept their identity,
so the model learned orientation (0.4% swapped on validation against the
rule's 15%).

``torch`` and ``timm`` are the ``[headtail-keypoints]`` extra and load lazily;
the crop rule, the warp and the decode are numpy and importable without them.
"""

from __future__ import annotations

# torch and timm load lazily (the extra); cv2 is a C extension pylint cannot
# introspect. The affine takes the five augmentation knobs training used, so
# a test can drive it the way training did.
# pylint: disable=import-outside-toplevel,import-error,no-member
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Tuple

import cv2
import numpy as np

from fishsense_core._optional import require
from fishsense_core.models import Manifest, WeightStore, fetch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "BACKBONE",
    "BOX_SCALE",
    "CROP_PAD",
    "IN",
    "K",
    "MODEL_NAME",
    "NK",
    "PAD_VALUE",
    "RADIUS",
    "HeadTailKeypointDetector",
    "HeadTailKeypoints",
    "affine",
    "build_model",
    "crop_from_box",
    "decode",
    "lift",
    "normalize",
    "square_crop",
    "warp",
]

_log = logging.getLogger(__name__)

#: Manifest name of the checkpoint (``fishsense_core.models``).
MODEL_NAME = "headtail-keypoints"
_EXTRA = "headtail-keypoints"

#: Network input side, SimCC bins per pixel, decode window in bins, keypoints.
IN, K, RADIUS, NK = 512, 2, 6, 2
#: The crop rule: side = BOX_SCALE x mask-box diagonal + CROP_PAD. The box
#: diagonal is ~1.1x the head-tail length.
BOX_SCALE, CROP_PAD = 1.4, 128
#: Ultralytics' letterbox grey, outside the crop after the warp.
PAD_VALUE = 114
#: The timm backbone; built with ``pretrained=False`` here, the checkpoint
#: carries every weight.
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


# ── the crop rule ───────────────────────────────────────────────────────────


def square_crop(
    cx: float,
    cy: float,
    extent: float,
    frame_w: int,
    frame_h: int,
    scale: float,
    pad: int = CROP_PAD,
) -> Tuple[int, int, int, int]:
    """A square of side ``scale * extent + pad`` centred on ``(cx, cy)``,
    clipped to the frame: ``(x0, y0, x1, y1)``, half-open. The floor/ceil
    pair is the experiment's; one pixel either way changes the crop."""
    half = 0.5 * (scale * extent + pad)
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    return x0, y0, min(frame_w, int(np.ceil(cx + half))), min(frame_h, int(np.ceil(cy + half)))


def crop_from_box(
    bbox: Sequence[float], frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    """The model's crop of a fish whose SAM mask box is ``bbox``
    (``[x0, y0, x1, y1)`` in frame pixels): :data:`BOX_SCALE` x the box
    diagonal + :data:`CROP_PAD`, square, centred on the box, clipped."""
    x0, y0, x1, y1 = bbox
    return square_crop(
        0.5 * (x0 + x1),
        0.5 * (y0 + y1),
        float(np.hypot(x1 - x0, y1 - y0)),
        frame_w,
        frame_h,
        scale=BOX_SCALE,
    )


# ── crop <-> network input ──────────────────────────────────────────────────


def affine(
    h: int,
    w: int,
    rot: float = 0.0,
    scale: float = 1.0,
    shift: Tuple[float, float] = (0.0, 0.0),
    flip: bool = False,
) -> np.ndarray:
    """2x3 matrix, crop pixels -> network input pixels. The crop's longer
    side fills :data:`IN` at scale 1; the crop's centre lands on the input's
    centre. The other knobs are training's augmentation."""
    s = scale * IN / max(h, w)
    a = math.radians(rot)
    c, si = math.cos(a) * s, math.sin(a) * s
    m = np.array([[c, -si, 0.0], [si, c, 0.0]])
    if flip:
        m[0, :2] *= -1
    cx, cy = w / 2 + shift[0] * w, h / 2 + shift[1] * h
    m[:, 2] = np.array([IN / 2, IN / 2]) - m[:, :2] @ np.array([cx, cy])
    return m


def warp(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The crop under ``m``, as an :data:`IN` x :data:`IN` BGR image, grey
    where the crop does not reach."""
    return cv2.warpAffine(
        img,
        m,
        (IN, IN),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAD_VALUE,) * 3,
    )


def normalize(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float32 CHW, ImageNet-normalised (the backbone's)."""
    x = (img_bgr[:, :, ::-1].astype(np.float32) - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def lift(points_in: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Network-input pixels back to crop pixels: ``m``'s inverse."""
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)
    return points_in @ inv[:2, :2].T + inv[:2, 2]


def decode(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SimCC logits ``(..., IN * K)`` -> coordinate in input pixels ``(...)``
    and the peak probability. The softmax-weighted mean within +-:data:`RADIUS`
    bins of the argmax, so a second mode elsewhere does not pull the point."""
    logits = np.asarray(logits, np.float64)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    am = p.argmax(-1)[..., None]
    bins = np.arange(p.shape[-1]).reshape((1,) * (p.ndim - 1) + (-1,))
    w = p * (np.abs(bins - am) <= RADIUS)
    coord = (w * bins).sum(-1) / np.maximum(w.sum(-1), 1e-12) / K
    return coord, p.max(-1)


def _check_image(img: Any, what: str) -> np.ndarray:
    """The contract every image here shares, enforced: ``(H, W, 3)`` uint8
    BGR with pixels in it. A float or 16-bit frame would be normalised with
    the uint8 ImageNet constants and give plausible garbage, so it is
    refused rather than run."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{what} must be (H, W, 3) BGR; got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"{what} must be uint8 (0-255), as the model was trained; got {arr.dtype}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{what} is empty: shape {arr.shape}")
    return arr


# ── the model ───────────────────────────────────────────────────────────────


_SIMCC: Any = None


def _simcc_class() -> Any:
    """The network class, defined once torch and timm are importable and then
    kept: one class, so instances compare by type and a whole module
    pickles (``torch.save(module)`` finds it at ``headtail_keypoints.SimCC``
    through the module's ``__getattr__``)."""
    global _SIMCC  # pylint: disable=global-statement
    if _SIMCC is not None:
        return _SIMCC
    torch = require("torch", _EXTRA, "HeadTailKeypointDetector")
    timm = require("timm", _EXTRA, "HeadTailKeypointDetector")
    nn = torch.nn

    class SimCC(nn.Module):  # pylint: disable=too-few-public-methods
        """``forward(x) -> (logits_x, logits_y)``, each ``(B, NK, IN * K)``:
        ConvNeXt-Tiny's last stage, a 7x7 reduce to 32 channels per
        keypoint, a per-keypoint MLP, and two classifiers over ``IN * K``
        bins. ``pretrained`` fetches the backbone's ImageNet weights
        (training's start); a checkpoint load overwrites them."""

        def __init__(self, pretrained: bool = False):
            super().__init__()
            self.backbone = timm.create_model(
                BACKBONE, pretrained=pretrained, features_only=True, out_indices=[3]
            )
            channels = self.backbone.feature_info.channels()[-1]
            side = IN // 32
            self.reduce = nn.Sequential(nn.Conv2d(channels, NK * 32, 7, padding=3), nn.GELU())
            self.mlp = nn.Sequential(
                nn.LayerNorm(32 * side * side), nn.Linear(32 * side * side, 512), nn.GELU()
            )
            self.cls_x = nn.Linear(512, IN * K)
            self.cls_y = nn.Linear(512, IN * K)

        def forward(self, x):  # pylint: disable=missing-function-docstring
            f = self.reduce(self.backbone(x)[-1])  # (B, NK*32, S, S)
            f = f.reshape(f.shape[0], NK, -1)  # per keypoint
            h = self.mlp(f)
            return self.cls_x(h), self.cls_y(h)

    SimCC.__module__ = __name__
    SimCC.__qualname__ = SimCC.__name__
    _SIMCC = SimCC
    return SimCC


def __getattr__(name: str) -> Any:
    """``headtail_keypoints.SimCC``: the network class, built on first use.
    This is how pickle finds it when a whole module is saved."""
    if name == "SimCC":
        return _simcc_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_model(pretrained: bool = False) -> "torch.nn.Module":
    """The SimCC network, exactly as trained. ``pretrained`` fetches the
    backbone's ImageNet weights; a checkpoint load overwrites them."""
    return _simcc_class()(pretrained=pretrained)


def _checkpoint_safe_globals() -> list:
    """What the checkpoint's non-tensor payload (``epoch``, ``length_val``:
    numpy scalars from pandas) needs the restricted unpickler to allow,
    beyond its defaults. Resolved by name so a numpy without one of them
    (the ``numpy.core`` path is numpy 1's) is not an error."""
    allowed = []
    for path in (
        "numpy.core.multiarray.scalar",
        "numpy._core.multiarray.scalar",
        "numpy.dtype",
        "numpy.dtypes.Float64DType",
        "numpy.dtypes.Float32DType",
        "numpy.dtypes.Int64DType",
    ):
        module_name, _, attr = path.rpartition(".")
        try:
            obj = getattr(__import__(module_name, fromlist=[attr]), attr)
        except (ImportError, AttributeError):
            continue
        allowed.append(obj)
    return allowed


@dataclass(frozen=True)
class HeadTailKeypoints:
    """One fish's snout and fork in frame pixels, the crop they came from,
    and the mean peak probability of the four SimCC distributions (a
    relative confidence, not calibrated)."""

    head: Tuple[float, float]
    tail: Tuple[float, float]
    confidence: float
    crop: Tuple[int, int, int, int]

    @property
    def length_px(self) -> float:
        """Snout-to-fork distance in frame pixels."""
        return float(np.hypot(self.head[0] - self.tail[0], self.head[1] - self.tail[1]))


class HeadTailKeypointDetector:
    """The trained SimCC model, from a crop to snout and fork.

    Build it with :meth:`from_store` (the pinned checkpoint, verified by hash;
    ``HuggingFaceStore()`` fetches it from the public
    ``ucsde4e/fishsense-headtail-keypoint`` repo at its pinned commit)
    or :meth:`from_checkpoint`; then :meth:`predict` a frame given the fish's
    SAM mask box, or :meth:`predict_crops` crops already cut by
    :func:`crop_from_box`.
    """

    def __init__(self, module: "torch.nn.Module", device: "torch.device | str | None" = None):
        torch = require("torch", _EXTRA, "HeadTailKeypointDetector")
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls, path: str | os.PathLike[str], *, device: "torch.device | str | None" = None
    ) -> "HeadTailKeypointDetector":
        """Load the experiment's checkpoint format, ``{"model": state_dict,
        "epoch": int, "length_val": {...}}`` (``best_length.pt``).

        The file is read with torch's restricted unpickler
        (``weights_only=True``, plus the numpy scalar types the payload
        holds), so a tampered checkpoint cannot run code on load; it fails to
        unpickle. It is still **not verified** against the manifest: that is
        :meth:`from_store`'s job, and the path to use for bytes from a shared
        bucket or repo. The state dict is loaded strictly, so a checkpoint for
        another design fails here, not quietly at the output.
        """
        torch = require("torch", _EXTRA, "HeadTailKeypointDetector")
        with torch.serialization.safe_globals(_checkpoint_safe_globals()):
            checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        module = build_model(pretrained=False)
        module.load_state_dict(checkpoint["model"], strict=True)
        _log.info(
            "loaded head/tail keypoint model %s (epoch %s)",
            Path(path).name,
            checkpoint.get("epoch"),
        )
        return cls(module, device=device)

    @classmethod
    def from_store(
        cls,
        store: WeightStore | None,
        *,
        cache_dir: str | os.PathLike[str],
        version: str | None = None,
        device: "torch.device | str | None" = None,
        manifest: Manifest | None = None,
    ) -> "HeadTailKeypointDetector":
        """Load the pinned checkpoint through :func:`fishsense_core.models.fetch`,
        verified by hash before loading. ``store=None`` loads only from
        ``cache_dir`` and never touches the network.

        Raises:
            KeyError: The manifest has no such version.
            fishsense_core.models.ModelIntegrityError: The store's bytes are
                not the pinned checkpoint.
            fishsense_core.models.ModelUnavailable: Not cached and no store
                has it.
        """
        path = fetch(MODEL_NAME, version, store=store, cache_dir=cache_dir, manifest=manifest)
        return cls.from_checkpoint(path, device=device)

    # -- inference ----------------------------------------------------------

    def _autocast(self) -> Any:
        """bf16 autocast on CUDA only, as trained and evaluated."""
        torch = self._torch
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def predict_crops(
        self, crops: Sequence[np.ndarray], batch: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keypoints ``(N, 2, 2)`` in crop pixels (``[snout, fork]``, each
        ``(x, y)``) and confidence ``(N,)`` for BGR uint8 crops cut by
        :func:`crop_from_box`.

        Raises:
            TypeError: A crop is not uint8.
            ValueError: A crop is not ``(H, W, 3)``, or is empty.
        """
        torch = self._torch
        crops = [_check_image(c, f"crop {i}") for i, c in enumerate(crops)]
        keypoints, confidences = [], []
        with torch.no_grad():
            for start in range(0, len(crops), batch):
                chunk = crops[start : start + batch]
                mats = [affine(*c.shape[:2]) for c in chunk]
                x = torch.from_numpy(
                    np.stack([normalize(warp(c, m)) for c, m in zip(chunk, mats)])
                ).to(self.device)
                with self._autocast():
                    logits_x, logits_y = self.module(x)
                px, cx = decode(logits_x.float().cpu().numpy())
                py, cy = decode(logits_y.float().cpu().numpy())
                points = np.stack([px, py], -1)  # (b, NK, 2) input pixels
                for m, q in zip(mats, points):
                    keypoints.append(lift(q, m))
                confidences.append(((cx + cy) / 2).mean(-1))
        if not keypoints:
            return np.zeros((0, NK, 2)), np.zeros((0,))
        return np.array(keypoints), np.concatenate(confidences)

    def predict(
        self, frame_bgr: np.ndarray, mask_bbox: Sequence[float] | None
    ) -> HeadTailKeypoints:
        """Snout and fork of the fish whose SAM mask box is ``mask_bbox``, in
        frame pixels. ``frame_bgr`` is the rectified frame the box was found
        on, ``(H, W, 3)`` uint8 BGR; ``mask_bbox`` is ``FishAtDot.mask_bbox``.

        Raises:
            ValueError: ``mask_bbox`` is None (the stage abstained: check
                ``FishAtDot.status`` before calling), or the box lies outside
                the frame so the crop is empty, or the frame is not
                ``(H, W, 3)``.
            TypeError: The frame is not uint8.
        """
        if mask_bbox is None:
            raise ValueError(
                "mask_bbox is None: the stage abstained on this frame (no fish "
                "under the dot), so there is nothing to keypoint. Check "
                "FishAtDot.status before calling predict."
            )
        frame_bgr = _check_image(frame_bgr, "frame")
        height, width = frame_bgr.shape[:2]
        x0, y0, x1, y1 = crop_from_box(mask_bbox, width, height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"mask_bbox {tuple(mask_bbox)} lies outside the {width}x{height} frame: "
                "the crop is empty. Is this the frame the box was found on?"
            )
        crop = np.ascontiguousarray(frame_bgr[y0:y1, x0:x1])
        points, confidence = self.predict_crops([crop])
        head, tail = points[0]
        return HeadTailKeypoints(
            head=(float(head[0]) + x0, float(head[1]) + y0),
            tail=(float(tail[0]) + x0, float(tail[1]) + y0),
            confidence=float(confidence[0]),
            crop=(x0, y0, x1, y1),
        )
