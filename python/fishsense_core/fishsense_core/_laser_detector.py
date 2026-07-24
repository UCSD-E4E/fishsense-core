"""Laser-dot detector inference stack.

A port of the production inference recipe from the FishSense laser-detector
training repo (``UCSD-E4E/2026-05-02_laser_detector``), specifically
``src/laser_detector/inference.py`` (``predict_frame`` /
``predict_frame_with_cascade``) plus the bias-offset and rectification steps
that ``train.py::_run_inference`` applies around them.

The public surface is re-exported from :mod:`fishsense_core.laser`; import
``LaserDetector`` from there.

Fidelity notes — this module deliberately mirrors the reference implementation
rather than improving on it. Where the reference does something surprising, the
behaviour is reproduced and the surprise is documented in a comment, because
the published accuracy numbers (val hit_n3 = 0.9081) were measured with those
surprises in place. Do not "fix" them without re-running the audit.

``torch`` and ``segmentation-models-pytorch`` are optional dependencies; they
are imported lazily so that importing :mod:`fishsense_core` stays cheap for
callers that never touch the detector. Install with::

    pip install 'fishsense_core[laser-detector]'
"""

from __future__ import annotations

# This module mirrors one reference module end to end, so it is deliberately
# kept as a single unit: `too-many-lines` and the argument-count checks fight
# that, and the mask builders genuinely take a bbox, a center, a sigma and a
# floor. `import-outside-toplevel` / `import-error` are the point of the
# optional-dependency design — torch is not installed in the base environment.
# `no-member` is a known pylint false positive on `np.mgrid[...]`, which it
# infers as a plain tuple.
# pylint: disable=too-many-lines,import-outside-toplevel,import-error
# pylint: disable=too-many-arguments,too-many-positional-arguments,no-member

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Recipe constants
#
# These mirror `laser_detector.data` / `laser_detector.inference`. Values that
# differ from the reference module's *defaults* because production overrides
# them on the command line are called out individually.
# --------------------------------------------------------------------------

DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 256

# `laser_detector.data.WAVELENGTH_CHANNEL`. Note this is a {0.0, 0.5, 1.0}
# encoding, NOT the {-1, 0, +1} encoding described in some downstream docs;
# getting this wrong degrades green-wavelength frames specifically.
WAVELENGTH_CHANNEL: dict[str, float] = {"red": 1.0, "green": 0.0}
UNKNOWN_WAVELENGTH_CHANNEL = 0.5

# `laser_detector.inference.DEFAULT_RIG_PRIOR_*`. Static prior on laser
# position in Olympus TG-6 *sensor* coordinates (no EXIF rotation).
DEFAULT_RIG_PRIOR_BBOX: tuple[int, int, int, int] = (1100, 700, 2950, 2180)
DEFAULT_RIG_PRIOR_CENTER: tuple[float, float] = (1977.0, 1343.0)
DEFAULT_RIG_PRIOR_SIGMA: tuple[float, float] = (300.0, 300.0)
# The reference module defaults this to 0.5, but the production recipe passes
# `--rig-prior-floor 1.0`, which saturates the Gaussian and makes the prior a
# pure hard bbox. We default to the production value.
DEFAULT_RIG_PRIOR_FLOOR = 1.0

DEFAULT_LINE_MASK_CORRIDOR_PX = 25.0
DEFAULT_TAU_LINE = 5.0
DEFAULT_ALPHA_MAX = 0.3
DEFAULT_REFINE_WINDOW = 256
DEFAULT_BAYER_EXCESS_SCALE = 4096.0
DEFAULT_INFERENCE_BATCH_SIZE = 8

# Per-checkpoint calibration, subtracted from the final prediction. Two
# constraints determine these values, and both must hold or the offset is
# wrong:
#   1. Bayer-excess must use the `np.repeat` upsample — the checkpoint, its
#      audit numbers, and this calibration were all produced against it (see
#      `LinearRawImage.VALID_UPSAMPLES`).
#   2. Inference must run in fp32, NOT bf16 autocast. The offset was refit on
#      the fp32 pipeline after the training repo disabled bf16 at inference
#      (upstream issue #13: bf16 rounding made the argmax hardware-dependent).
#      The obsolete Ada-bf16 offset was (-0.200, -0.006); do not use it.
# This is why `predict()` defaults `use_bf16=False`.
CHECKPOINT_BIAS_OFFSETS: dict[str, tuple[float, float]] = {
    "run3_epoch_021.pt": (-0.179, -0.023),
    # run7 (HRNet) has not been independently refit on the fp32 pipeline
    # upstream; this is the run3 fp32 value as a best available estimate.
    # Recalibrate against run7 before relying on it for that checkpoint.
    "run7_hrnet_w18_epoch_021.pt": (-0.179, -0.023),
}
DEFAULT_CHECKPOINT = "run3_epoch_021.pt"
DEFAULT_HF_REPO = "ucsde4e/fishsense-laser-detector"

# Encoder backbone per checkpoint. The published checkpoints' embedded `cfg`
# dicts predate the `encoder_name` / `decoder_interpolation` fields, so the
# values that were in force at training time have to be recorded here.
# `decoder_interpolation="nearest"` is `train.py`'s default.
CHECKPOINT_ENCODERS: dict[str, str] = {
    "run3_epoch_021.pt": "resnet34",
    "run7_hrnet_w18_epoch_021.pt": "tu-hrnet_w18",
}

# Content hashes (sha256 of the raw .pt bytes) of the published checkpoints, so
# a local copy resolves to its canonical name — and therefore its encoder and
# bias offset — regardless of the filename it was saved under. Filename is a
# fragile identity: a run3 checkpoint saved as `epoch_021.pt` would otherwise
# lose its bias-offset calibration silently. Identity by content is not.
CHECKPOINT_SHA256: dict[str, str] = {
    "bd3ab8f5e273da37a1f2dfc2c6c6a36735b89ae26ff821b71b1f8acce3a74d68":
        "run3_epoch_021.pt",
    "17a4cd13358fe98093ad06cb9097d7ced844b305784cc32a6d26ead63dae6577":
        "run7_hrnet_w18_epoch_021.pt",
}
DEFAULT_DECODER_INTERPOLATION = "nearest"
DEFAULT_PRESENCE_HIDDEN = 128


def _require_torch() -> Any:
    """Import torch, or raise with an actionable message."""
    try:
        import torch  # noqa: PLC0415 - optional heavy dependency
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "The laser detector requires PyTorch. Install the optional extra: "
            "pip install 'fishsense_core[laser-detector]'"
        ) from exc
    return torch


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def chromaticity_norm(image_rgb: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """RGB → per-pixel chromaticity ``c_i = i / (R + G + B)``.

    Port of ``laser_detector.data._chromaticity_norm``. Accepts uint8 or
    uint16; the scale factor is chosen from the dtype, so passing a uint8
    image where the model expects uint16 silently changes nothing (the
    normalization is scale-invariant) but passing a *gamma-corrected* or
    CLAHE'd image does change the result — see :class:`LinearRawImage`.

    Returns float32 ``[H, W, 3]``.
    """
    scale = 65535.0 if image_rgb.dtype == np.uint16 else 255.0
    rgb = image_rgb.astype(np.float32) / scale
    intensity = np.maximum(rgb.sum(axis=2, keepdims=True), eps)
    return rgb / intensity


def _reflect_pad(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pad the bottom/right edges to ``(h, w)`` with BORDER_REFLECT_101."""
    src_h, src_w = image.shape[:2]
    pad_h = max(h - src_h, 0)
    pad_w = max(w - src_w, 0)
    if pad_h == 0 and pad_w == 0:
        return image
    return cv2.copyMakeBorder(  # pylint: disable=no-member
        image, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101  # pylint: disable=no-member
    )


def _preprocess_tile(
    tile_bgr: np.ndarray,
    wavelength_value: float,
    bayer_excess_tile: np.ndarray | None = None,
    bayer_excess_scale: float = DEFAULT_BAYER_EXCESS_SCALE,
) -> np.ndarray:
    """BGR tile → float32 ``[C, H, W]`` model input.

    C=4 (chromaticity + wavelength) or C=6 when Bayer-excess G/R is supplied.
    """
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    chrom = chromaticity_norm(rgb)
    h, w = chrom.shape[:2]
    wavelength_channel = np.full((h, w, 1), wavelength_value, dtype=np.float32)
    parts = [chrom, wavelength_channel]
    if bayer_excess_tile is not None:
        parts.append(bayer_excess_tile.astype(np.float32) / bayer_excess_scale)
    stacked = np.concatenate(parts, axis=2)
    return np.transpose(stacked, (2, 0, 1)).copy()


def _wavelength_value(wavelength: str | None) -> float:
    if wavelength is None:
        return UNKNOWN_WAVELENGTH_CHANNEL
    return WAVELENGTH_CHANNEL.get(wavelength, UNKNOWN_WAVELENGTH_CHANNEL)


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGrid:
    """Tile origins covering one frame, plus the reflect-padded extent."""

    origins: list[tuple[int, int]]
    padded_h: int
    padded_w: int
    original_h: int
    original_w: int


def compute_tile_grid(
    h: int,
    w: int,
    *,
    tile: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> TileGrid:
    """Tile ``(x, y)`` origins covering an ``h`` × ``w`` image.

    Port of ``laser_detector.inference.compute_tile_grid``. The last tile on
    each axis snaps to the image edge, so it may overlap its predecessor by
    more than ``overlap``. Images smaller than ``tile`` get a single tile and
    are reflect-padded up.
    """
    stride = tile - overlap
    xs = [0] if w <= tile else list(range(0, w - tile, stride)) + [w - tile]
    ys = [0] if h <= tile else list(range(0, h - tile, stride)) + [h - tile]
    origins = [(x, y) for y in ys for x in xs]
    return TileGrid(
        origins=origins,
        padded_h=max(h, tile),
        padded_w=max(w, tile),
        original_h=h,
        original_w=w,
    )


# --------------------------------------------------------------------------
# Heatmap masks
# --------------------------------------------------------------------------


# pylint: disable-next=too-many-locals
def _rig_prior_for_tile(
    tile_origin_x: int,
    tile_origin_y: int,
    tile: int,
    bbox: tuple[int, int, int, int],
    center: tuple[float, float],
    sigma: tuple[float, float],
    floor: float,
) -> np.ndarray:
    """``[tile, tile]`` float32 rig-prior mask in [0, 1] for one tile.

    Outside ``bbox`` → 0 (hard reject). Inside → ``max(floor, gaussian)``, so
    ``floor=1.0`` degenerates to a pure hard bbox (the production setting).
    """
    ys, xs = np.mgrid[
        tile_origin_y : tile_origin_y + tile,
        tile_origin_x : tile_origin_x + tile,
    ].astype(np.float32)

    bx0, by0, bx1, by1 = bbox
    in_bbox = (xs >= bx0) & (xs < bx1) & (ys >= by0) & (ys < by1)

    cx, cy = center
    sx, sy = sigma
    gauss = np.exp(-0.5 * (((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2))
    soft = np.maximum(gauss, floor).astype(np.float32)
    return np.where(in_bbox, soft, 0.0).astype(np.float32)


def _line_mask_for_tile(
    tile_origin_x: int,
    tile_origin_y: int,
    tile: int,
    line_abc: tuple[float, float, float],
    corridor_px: float,
) -> np.ndarray:
    """``[tile, tile]`` binary corridor mask around ``a*x + b*y + c = 0``."""
    a, b, c = line_abc
    norm = float((a * a + b * b) ** 0.5)
    if norm <= 1e-12:
        return np.ones((tile, tile), dtype=np.float32)
    ys, xs = np.mgrid[
        tile_origin_y : tile_origin_y + tile,
        tile_origin_x : tile_origin_x + tile,
    ].astype(np.float32)
    dist = np.abs(a * xs + b * ys + c) / norm
    return (dist <= corridor_px).astype(np.float32)


# --------------------------------------------------------------------------
# Peak refinement and line snapping
# --------------------------------------------------------------------------


def subpixel_refine_peak(
    heatmap_2d: "torch.Tensor | np.ndarray", x: int, y: int
) -> tuple[float, float]:
    """Separable 3-point parabolic refinement of an integer peak.

    For each axis, ``delta = 0.5 * (v_minus - v_plus) /
    (v_minus - 2*v_center + v_plus)``, rejected (→ 0) when the fit is
    degenerate or lands outside ``(-0.5, 0.5)``.

    Must be called on heatmap **logits**, not sigmoid probabilities: under
    bf16 autocast the sigmoid saturates to exactly 1.0 across the whole peak
    and the parabola becomes meaningless. See :meth:`LaserDetector.predict`.
    """
    h, w = heatmap_2d.shape[-2:]
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return float(x), float(y)

    # Duck-typed rather than isinstance-checked against torch.Tensor, so this
    # function (and the tests covering it) stay usable without torch installed.
    def get(i: int, j: int) -> float:
        value = heatmap_2d[i, j]
        return float(value.item() if hasattr(value, "item") else value)

    v_c = get(y, x)
    v_xm, v_xp = get(y, x - 1), get(y, x + 1)
    v_ym, v_yp = get(y - 1, x), get(y + 1, x)
    den_x = v_xm - 2.0 * v_c + v_xp
    den_y = v_ym - 2.0 * v_c + v_yp
    dx = 0.5 * (v_xm - v_xp) / den_x if abs(den_x) > 1e-12 else 0.0
    dy = 0.5 * (v_ym - v_yp) / den_y if abs(den_y) > 1e-12 else 0.0
    if not -0.5 < dx < 0.5:
        dx = 0.0
    if not -0.5 < dy < 0.5:
        dy = 0.0
    return float(x) + dx, float(y) + dy


def _project_point_onto_line(
    x: float, y: float, a: float, b: float, c: float
) -> tuple[float, float]:
    """Orthogonal projection of ``(x, y)`` onto ``a*x + b*y + c = 0``."""
    norm_sq = a * a + b * b
    if norm_sq <= 1e-12:
        return x, y
    t = (a * x + b * y + c) / norm_sq
    return x - t * a, y - t * b


def soft_snap_to_line(
    x: float,
    y: float,
    *,
    line_abc: tuple[float, float, float],
    line_confidence: float,
    pred_confidence: float,
    tau_line: float = DEFAULT_TAU_LINE,
    alpha_max: float = DEFAULT_ALPHA_MAX,
) -> tuple[float, float, float]:
    """Blend a prediction toward its projection onto the dive line.

    ``alpha = clip(sigmoid(line_confidence - tau_line) * (1 - pred_confidence),
    0, alpha_max)``; confident predictions stay free to disagree with the line.
    Returns ``(x, y, alpha)``.
    """
    line_strength = 1.0 / (1.0 + np.exp(-(line_confidence - tau_line)))
    alpha = max(0.0, min(float(line_strength * (1.0 - pred_confidence)), alpha_max))
    if alpha <= 0.0:
        return x, y, 0.0
    proj_x, proj_y = _project_point_onto_line(x, y, *line_abc)
    return (1.0 - alpha) * x + alpha * proj_x, (1.0 - alpha) * y + alpha * proj_y, alpha


def rectify_prediction(
    pred_x: float, pred_y: float, k: np.ndarray, dist: np.ndarray
) -> tuple[float, float]:
    """Raw pixel space → rectified (undistorted) pixel space.

    ``cv2.undistortPoints(..., P=K)``. See the coordinate-frame note on
    :meth:`LaserDetector.predict`.
    """
    pts = np.asarray([[[pred_x, pred_y]]], dtype=np.float32)
    k = np.asarray(k, dtype=np.float32).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float32).reshape(-1)
    out = cv2.undistortPoints(pts, k, dist, P=k)  # pylint: disable=no-member
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaserPrediction:
    """One frame's laser-dot prediction.

    Attributes:
        x: Predicted column, or ``None`` when no laser was detected.
        y: Predicted row, or ``None`` when no laser was detected.
        confidence: Frame-level presence confidence — the max presence sigmoid
            over all tiles. Reported even when ``x``/``y`` are ``None``.
    """

    x: float | None
    y: float | None
    confidence: float

    @property
    def is_detected(self) -> bool:
        """Whether a laser position was produced for this frame."""
        return self.x is not None and self.y is not None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def _build_module(encoder_name: str, in_channels: int, decoder_interpolation: str) -> Any:
    """Construct the UNet + presence-head module.

    Structurally identical to ``laser_detector.model.LaserDetector`` so that
    published checkpoints load with ``strict=True``.
    """
    torch = _require_torch()
    try:
        import segmentation_models_pytorch as smp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "The laser detector requires segmentation-models-pytorch. Install "
            "the optional extra: pip install 'fishsense_core[laser-detector]'"
        ) from exc

    # pylint: disable-next=import-outside-toplevel
    from torch import nn

    # pylint: disable-next=too-few-public-methods
    class _LaserDetectorModule(nn.Module):
        """UNet heatmap head + per-tile presence head sharing one encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=1,
                encoder_weights=None,
                decoder_interpolation=decoder_interpolation,
            )
            bottleneck_dim = self.unet.encoder.out_channels[-1]
            self.presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bottleneck_dim, DEFAULT_PRESENCE_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(DEFAULT_PRESENCE_HIDDEN, 1),
            )

        def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            """Returns heatmap logits ``[B, 1, H, W]`` and presence logits ``[B]``."""
            features = self.unet.encoder(x)
            presence_logits = self.presence_head(features[-1]).squeeze(-1)
            heatmap_logits = self.unet.segmentation_head(self.unet.decoder(features))
            return {
                "heatmap_logits": heatmap_logits,
                "presence_logits": presence_logits,
            }

    _ = torch  # imported for the lazy-dependency check above
    return _LaserDetectorModule()


def _extract_state_dict(checkpoint: dict) -> dict:
    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    raise KeyError(
        "checkpoint has no recognizable state dict "
        f"(keys: {sorted(checkpoint)[:10]})"
    )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _canonical_checkpoint_name(path: Path) -> str | None:
    """Resolve ``path`` to a published checkpoint's canonical filename.

    Matches by filename first, then falls back to hashing the file contents —
    so a renamed copy of a published checkpoint still resolves to the name that
    keys its encoder and bias offset. Returns ``None`` for an unrecognized
    checkpoint.
    """
    if path.name in CHECKPOINT_ENCODERS:
        return path.name
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CHECKPOINT_SHA256.get(digest)


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class LaserDetector:
    """Runs the FishSense laser-dot detector on full-resolution dive frames.

    Load with :meth:`from_pretrained` (HuggingFace) or :meth:`from_checkpoint`
    (local file), then call :meth:`predict` per frame.

    **Input contract.** The model was trained on *linear* 16-bit BGR — rawpy
    with ``gamma=(1, 1)``, ``no_auto_bright``, ``use_camera_wb``, and no EXIF
    rotation — and explicitly *not* on the CLAHE + auto-gamma pipeline that
    :class:`~fishsense_core.image.raw_image.RawImage` produces. CLAHE saturates bright
    laser blobs across all channels and destroys the wavelength selectivity
    the 6-channel input depends on. Use
    :class:`~fishsense_core.image.linear_raw_image.LinearRawImage` to decode frames for this
    detector; passing a ``RawImage`` will produce degraded, not obviously
    wrong, results.

    **Coordinate frame.** :meth:`predict` returns **raw pixel space** by
    default — the same frame the raw image is in, which is what the model
    actually predicts in. Labels in the FishSense corpus live in *rectified*
    pixel space because the labeling UI renders via
    ``RectifiedImage(RawImage(...))``. The discrepancy is small (median
    0.02 px, p99 1.01 px) but load-bearing for downstream 3D reconstruction.
    Pass ``rectify_output=True`` together with ``camera_matrix`` and
    ``distortion`` to emit rectified coordinates instead.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        module: Any,
        *,
        in_channels: int,
        bias_offset: tuple[float, float] = (0.0, 0.0),
        device: "torch.device | str | None" = None,
        checkpoint_name: str | None = None,
    ):
        torch = _require_torch()
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.in_channels = int(in_channels)
        self.bias_offset = bias_offset
        self.checkpoint_name = checkpoint_name
        self.module = module.to(self.device).eval()
        _log.debug(
            "LaserDetector ready: device=%s in_channels=%d bias_offset=%s",
            self.device,
            self.in_channels,
            self.bias_offset,
        )

    # -- construction -------------------------------------------------------

    @classmethod
    # pylint: disable-next=too-many-arguments
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: "torch.device | str | None" = None,
        encoder_name: str | None = None,
        bias_offset: tuple[float, float] | None = None,
        decoder_interpolation: str = DEFAULT_DECODER_INTERPOLATION,
    ) -> "LaserDetector":
        """Load a detector from a local ``.pt`` checkpoint.

        Args:
            path: Checkpoint file.
            device: Torch device; defaults to CUDA when available.
            encoder_name: smp encoder backbone. Resolved automatically for a
                published checkpoint (by content, so a renamed copy still
                resolves); required for an unrecognized one.
            bias_offset: ``(dx, dy)`` subtracted from the final prediction.
                Resolved automatically for a published checkpoint. For an
                unrecognized checkpoint this is **required** — it is a
                per-checkpoint calibration that cannot be inferred from the
                weights, so passing ``(0.0, 0.0)`` is the explicit way to run
                without one.
            decoder_interpolation: smp UNet decoder upsample mode. The
                published checkpoints predate this being recorded in the
                checkpoint config; ``"nearest"`` was the training default.

        Raises:
            ValueError: If the checkpoint is not a recognized published one and
                ``encoder_name`` or ``bias_offset`` was not supplied.
        """
        torch = _require_torch()
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("cfg") or {}
        # Identify the checkpoint by content, not just filename, so a renamed
        # copy of a published checkpoint keeps its encoder and bias calibration.
        canonical = _canonical_checkpoint_name(path)
        name = canonical or path.name

        in_channels = int(cfg.get("in_channels", 6))
        if encoder_name is None:
            encoder_name = CHECKPOINT_ENCODERS.get(name) or cfg.get("encoder_name")
            if encoder_name is None:
                raise ValueError(
                    f"cannot infer encoder for unrecognized checkpoint "
                    f"{path.name!r}; pass encoder_name= explicitly"
                )
        if bias_offset is None:
            if name not in CHECKPOINT_BIAS_OFFSETS:
                # Never silently default to (0, 0): the bias offset is a
                # per-checkpoint calibration and running without it leaves a
                # constant sub-pixel error the caller may not notice.
                raise ValueError(
                    f"no bias offset known for unrecognized checkpoint "
                    f"{path.name!r}. It is a per-checkpoint calibration and "
                    "cannot be inferred from the weights. Pass bias_offset= "
                    "explicitly, or bias_offset=(0.0, 0.0) to run without one. "
                    f"Known checkpoints: {sorted(CHECKPOINT_BIAS_OFFSETS)}."
                )
            bias_offset = CHECKPOINT_BIAS_OFFSETS[name]
        decoder_interpolation = cfg.get(
            "decoder_interpolation", decoder_interpolation
        )

        module = _build_module(encoder_name, in_channels, decoder_interpolation)
        module.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        _log.info(
            "loaded laser detector %s (encoder=%s, in_channels=%d, epoch=%s)",
            name,
            encoder_name,
            in_channels,
            checkpoint.get("epoch"),
        )
        return cls(
            module,
            in_channels=in_channels,
            bias_offset=bias_offset,
            device=device,
            checkpoint_name=name,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_HF_REPO,
        *,
        filename: str = DEFAULT_CHECKPOINT,
        revision: str = "main",
        device: "torch.device | str | None" = None,
        **kwargs: Any,
    ) -> "LaserDetector":
        """Download a published checkpoint from HuggingFace and load it.

        Args:
            repo_id: HuggingFace model repo.
            filename: Checkpoint within the repo. Defaults to the production
                ResNet-34 checkpoint.
            revision: Git revision to pin.
            device: Torch device; defaults to CUDA when available.
        """
        try:
            # pylint: disable-next=import-outside-toplevel
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "from_pretrained requires huggingface_hub. Install the "
                "optional extra: pip install 'fishsense_core[laser-detector]'"
            ) from exc

        path = hf_hub_download(repo_id, filename, revision=revision)
        return cls.from_checkpoint(path, device=device, **kwargs)

    # -- inference ----------------------------------------------------------

    def _autocast(self, autocast_dtype: Any) -> Any:
        """bf16 autocast on CUDA only, matching the reference."""
        if autocast_dtype is not None and self.device.type == "cuda":
            return self._torch.autocast(
                device_type=self.device.type, dtype=autocast_dtype
            )
        return _NullContext()

    # The recipe genuinely has this many independent knobs; grouping them into
    # a config object would only move the argument list somewhere else.
    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def _predict_coarse(
        self,
        image_bgr: np.ndarray,
        *,
        wavelength: str | None,
        tile: int,
        overlap: int,
        batch_size: int,
        presence_threshold: float | None,
        autocast_dtype: Any,
        line_abc: tuple[float, float, float] | None,
        line_confidence: float,
        tau_line: float,
        alpha_max: float,
        rig_prior: bool,
        rig_prior_bbox: tuple[int, int, int, int],
        rig_prior_center: tuple[float, float],
        rig_prior_sigma: tuple[float, float],
        rig_prior_floor: float,
        bayer_excess_image: np.ndarray | None,
        bayer_excess_scale: float,
        subpixel_refine: bool,
        line_mask_corridor_px: float | None,
    ) -> LaserPrediction:
        """Single-pass tiled inference. Port of ``inference.predict_frame``."""
        torch = self._torch
        grid = compute_tile_grid(*image_bgr.shape[:2], tile=tile, overlap=overlap)
        padded = _reflect_pad(image_bgr, grid.padded_h, grid.padded_w)
        wavelength_value = _wavelength_value(wavelength)

        bayer_padded = (
            _reflect_pad(bayer_excess_image, grid.padded_h, grid.padded_w)
            if bayer_excess_image is not None
            else None
        )

        tile_arrays = [
            _preprocess_tile(
                padded[y : y + tile, x : x + tile],
                wavelength_value,
                bayer_excess_tile=(
                    None if bayer_padded is None
                    else bayer_padded[y : y + tile, x : x + tile]
                ),
                bayer_excess_scale=bayer_excess_scale,
            )
            for x, y in grid.origins
        ]
        tile_batch = torch.from_numpy(np.stack(tile_arrays))

        rig_masks = None
        if rig_prior:
            rig_masks = [
                torch.from_numpy(
                    _rig_prior_for_tile(
                        ox, oy, tile, rig_prior_bbox, rig_prior_center,
                        rig_prior_sigma, rig_prior_floor,
                    )
                )
                for ox, oy in grid.origins
            ]

        # The corridor mask requires a line; note that the reference only
        # threads `line_abc` through when soft-snap is enabled, so in the
        # reference pipeline corridor masking is implicitly coupled to
        # soft-snap being on. We reproduce the same gating.
        line_masks = None
        if (
            line_mask_corridor_px is not None
            and line_abc is not None
            and line_confidence > 0.0
        ):
            line_masks = [
                torch.from_numpy(
                    _line_mask_for_tile(ox, oy, tile, line_abc, line_mask_corridor_px)
                )
                for ox, oy in grid.origins
            ]

        best_value = -1.0
        best_xy: tuple[float | None, float | None] = (None, None)
        best_local: tuple[int, int] | None = None
        best_origin: tuple[int, int] | None = None
        best_heatmap_2d = None
        presence_max = 0.0

        autocast_ctx = self._autocast(autocast_dtype)
        for chunk_start in range(0, len(tile_arrays), batch_size):
            chunk = tile_batch[chunk_start : chunk_start + batch_size].to(
                self.device, non_blocking=True
            )
            with autocast_ctx:
                out = self.module(chunk)

            # fp32 BEFORE sigmoid. Under bf16 autocast, sigmoid of any logit
            # above ~5.5 saturates to exactly 1.0, several pixels tie at the
            # peak, and `max()` breaks ties toward the lowest row-major index
            # — biasing predictions up and to the left by 1-2 px.
            heatmap_logits = out["heatmap_logits"].float()
            heatmap_probs = torch.sigmoid(heatmap_logits)
            # Reference order: sigmoid then .float(). Presence only feeds a
            # scalar confidence and the argmax is unaffected, so this is kept
            # bit-identical to the reference rather than "fixed".
            presence_probs = torch.sigmoid(out["presence_logits"]).float()

            if rig_masks is not None:
                chunk_masks = torch.stack(
                    [rig_masks[chunk_start + i] for i in range(heatmap_probs.shape[0])]
                ).to(heatmap_probs.device)
                heatmap_probs = heatmap_probs * chunk_masks.unsqueeze(1)
            if line_masks is not None:
                chunk_line_masks = torch.stack(
                    [line_masks[chunk_start + i] for i in range(heatmap_probs.shape[0])]
                ).to(heatmap_probs.device)
                heatmap_probs = heatmap_probs * chunk_line_masks.unsqueeze(1)

            flat = heatmap_probs.view(heatmap_probs.shape[0], -1)
            max_vals, max_idx = flat.max(dim=1)

            for i, (mv, mi) in enumerate(zip(max_vals.tolist(), max_idx.tolist())):
                if mv > best_value:
                    best_value = mv
                    local_y, local_x = divmod(mi, tile)
                    ox, oy = grid.origins[chunk_start + i]
                    best_xy = (float(local_x + ox), float(local_y + oy))
                    best_local = (local_x, local_y)
                    best_origin = (ox, oy)
                    if subpixel_refine:
                        # Refine on the *unmasked* logits, matching the
                        # reference: the masks shape the argmax but not the
                        # parabola fitted around it.
                        best_heatmap_2d = heatmap_logits[i, 0].detach().cpu()
            presence_max = max(presence_max, float(presence_probs.max().item()))

        if presence_threshold is not None and presence_max < presence_threshold:
            return LaserPrediction(x=None, y=None, confidence=presence_max)

        pred_x, pred_y = best_xy
        if pred_x is not None:
            if subpixel_refine and best_heatmap_2d is not None:
                local_x, local_y = best_local
                rx, ry = subpixel_refine_peak(best_heatmap_2d, local_x, local_y)
                pred_x = float(best_origin[0]) + rx
                pred_y = float(best_origin[1]) + ry
            # Never report a position inside the reflect-padded margin.
            pred_x = min(pred_x, float(grid.original_w - 1))
            pred_y = min(pred_y, float(grid.original_h - 1))
            if line_abc is not None and line_confidence > 0.0:
                pred_x, pred_y, _ = soft_snap_to_line(
                    pred_x, pred_y,
                    line_abc=line_abc,
                    line_confidence=line_confidence,
                    pred_confidence=presence_max,
                    tau_line=tau_line,
                    alpha_max=alpha_max,
                )
                pred_x = max(0.0, min(pred_x, float(grid.original_w - 1)))
                pred_y = max(0.0, min(pred_y, float(grid.original_h - 1)))

        return LaserPrediction(x=pred_x, y=pred_y, confidence=presence_max)

    # pylint: disable-next=too-many-arguments,too-many-locals
    def _predict_cascade(
        self,
        image_bgr: np.ndarray,
        *,
        refine_window: int,
        wavelength: str | None,
        tile: int,
        overlap: int,
        batch_size: int,
        presence_threshold: float | None,
        autocast_dtype: Any,
        line_abc: tuple[float, float, float] | None,
        line_confidence: float,
        tau_line: float,
        alpha_max: float,
        rig_prior: bool,
        rig_prior_bbox: tuple[int, int, int, int],
        rig_prior_center: tuple[float, float],
        rig_prior_sigma: tuple[float, float],
        rig_prior_floor: float,
        bayer_excess_image: np.ndarray | None,
        bayer_excess_scale: float,
        subpixel_refine: bool,
        line_mask_corridor_px: float | None,
    ) -> LaserPrediction:
        """Two-pass inference. Port of ``predict_frame_with_cascade``."""
        torch = self._torch

        # The coarse pass needs the line only to build the corridor mask;
        # soft-snap is deferred until after refinement by forcing alpha to 0.
        coarse_line_abc = line_abc if line_mask_corridor_px is not None else None
        coarse_line_conf = (
            line_confidence if line_mask_corridor_px is not None else 0.0
        )
        coarse = self._predict_coarse(
            image_bgr,
            wavelength=wavelength, tile=tile, overlap=overlap,
            batch_size=batch_size, presence_threshold=presence_threshold,
            autocast_dtype=autocast_dtype,
            line_abc=coarse_line_abc, line_confidence=coarse_line_conf,
            tau_line=tau_line, alpha_max=0.0,
            rig_prior=rig_prior, rig_prior_bbox=rig_prior_bbox,
            rig_prior_center=rig_prior_center, rig_prior_sigma=rig_prior_sigma,
            rig_prior_floor=rig_prior_floor,
            bayer_excess_image=bayer_excess_image,
            bayer_excess_scale=bayer_excess_scale,
            subpixel_refine=subpixel_refine,
            line_mask_corridor_px=line_mask_corridor_px,
        )
        if coarse.x is None or coarse.y is None:
            return coarse

        h, w = image_bgr.shape[:2]
        half = refine_window // 2
        cx, cy = int(round(coarse.x)), int(round(coarse.y))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, x0 + refine_window), min(h, y0 + refine_window)
        x0, y0 = max(0, x1 - refine_window), max(0, y1 - refine_window)

        crop = _reflect_pad(image_bgr[y0:y1, x0:x1], refine_window, refine_window)
        bayer_crop = (
            _reflect_pad(
                bayer_excess_image[y0:y1, x0:x1], refine_window, refine_window
            )
            if bayer_excess_image is not None
            else None
        )

        arr = _preprocess_tile(
            crop,
            _wavelength_value(wavelength),
            bayer_excess_tile=bayer_crop,
            bayer_excess_scale=bayer_excess_scale,
        )
        batch = torch.from_numpy(arr[None]).to(self.device, non_blocking=True)

        with self._autocast(autocast_dtype):
            out = self.module(batch)

        # Same fp32-before-sigmoid rule as the coarse pass.
        heatmap_logits = out["heatmap_logits"][0].float()
        heatmap_probs = torch.sigmoid(heatmap_logits)
        presence_prob = float(torch.sigmoid(out["presence_logits"][0]).max().item())

        # Note: pass 2 applies neither the rig prior nor the corridor mask.
        # That matches the reference — the crop is already centered on a peak
        # that survived both masks.
        flat = heatmap_probs.view(-1)
        refined_value = float(flat.max().item())
        local_y, local_x = divmod(int(flat.argmax().item()), refine_window)
        if subpixel_refine:
            rx, ry = subpixel_refine_peak(heatmap_logits[0], local_x, local_y)
            refined_x, refined_y = float(x0) + rx, float(y0) + ry
        else:
            refined_x, refined_y = float(x0 + local_x), float(y0 + local_y)

        if presence_threshold is not None and presence_prob < presence_threshold:
            return coarse
        # Reference comparison, reproduced verbatim: this compares a heatmap
        # *probability* against a *presence* confidence. The two are not the
        # same quantity, but the published accuracy numbers were measured with
        # this check in place, so it is not "corrected" here.
        if refined_value < 0.5 * coarse.confidence:
            return coarse

        refined_x = max(0.0, min(refined_x, float(w - 1)))
        refined_y = max(0.0, min(refined_y, float(h - 1)))

        final_conf = max(coarse.confidence, presence_prob)
        if line_abc is not None and line_confidence > 0.0:
            refined_x, refined_y, _ = soft_snap_to_line(
                refined_x, refined_y,
                line_abc=line_abc,
                line_confidence=line_confidence,
                pred_confidence=final_conf,
                tau_line=tau_line,
                alpha_max=alpha_max,
            )
            refined_x = max(0.0, min(refined_x, float(w - 1)))
            refined_y = max(0.0, min(refined_y, float(h - 1)))

        return LaserPrediction(x=refined_x, y=refined_y, confidence=final_conf)

    # pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    def predict(
        self,
        image: Any,
        *,
        wavelength: str | None = None,
        bayer_excess: np.ndarray | None = None,
        dive_line: tuple[float, float, float, float] | None = None,
        cascade: bool = True,
        subpixel_refine: bool = True,
        rig_prior: bool = True,
        rig_prior_bbox: tuple[int, int, int, int] = DEFAULT_RIG_PRIOR_BBOX,
        rig_prior_center: tuple[float, float] = DEFAULT_RIG_PRIOR_CENTER,
        rig_prior_sigma: tuple[float, float] = DEFAULT_RIG_PRIOR_SIGMA,
        rig_prior_floor: float = DEFAULT_RIG_PRIOR_FLOOR,
        line_mask_corridor_px: float | None = DEFAULT_LINE_MASK_CORRIDOR_PX,
        soft_snap: bool = True,
        tau_line: float = DEFAULT_TAU_LINE,
        alpha_max: float = DEFAULT_ALPHA_MAX,
        refine_window: int = DEFAULT_REFINE_WINDOW,
        tile: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_TILE_OVERLAP,
        batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        presence_threshold: float | None = None,
        bayer_excess_scale: float = DEFAULT_BAYER_EXCESS_SCALE,
        use_bf16: bool = False,
        apply_bias_offset: bool = True,
        rectify_output: bool = False,
        camera_matrix: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
    ) -> LaserPrediction:
        """Detect the laser dot in a single frame.

        Defaults reproduce the production recipe: rig prior with a hard bbox,
        a ±25 px line corridor, two-pass cascade, sub-pixel refinement,
        soft-snap to the dive line, and the checkpoint's bias offset.

        Args:
            image: A :class:`~fishsense_core.image.LinearRawImage` (or any
                object with a ``.data`` array), or a BGR ndarray. Must be
                linear — see the class docstring on why ``RawImage`` is wrong
                for this model.
            wavelength: ``"red"``, ``"green"``, or ``None`` for unknown.
            bayer_excess: ``[H, W, 2]`` uint16 Bayer-excess G/R at full
                resolution. Required for 6-channel checkpoints. When ``image``
                is a ``LinearRawImage`` this is taken from it automatically.
            dive_line: ``(a, b, c, confidence)`` for the dive's fitted line,
                used for the corridor mask and soft-snap. ``None`` disables
                both.
            cascade: Run the two-pass refinement.
            subpixel_refine: Parabolic sub-pixel peak refinement, on logits.
            rig_prior: Apply the static rig-position prior.
            rig_prior_floor: ``1.0`` (default) makes the prior a hard bbox.
            line_mask_corridor_px: Corridor half-width; ``None`` disables.
            soft_snap: Blend the result toward the dive line.
            presence_threshold: When set, frames whose presence confidence
                falls below this return a non-detection. The production audit
                leaves this unset and thresholds downstream instead, so the
                default is ``None``.
            use_bf16: Use bf16 autocast (CUDA only; ignored on CPU). Defaults
                to ``False`` to match production, which disabled bf16 at
                inference: bf16 rounding collapses near-tied heatmap peaks and
                the argmax then resolves differently across GPU architectures
                (upstream issue #13). The published bias offset is calibrated
                for the fp32 pipeline, so enabling this both loses
                reproducibility and mis-pairs with the offset.
            apply_bias_offset: Subtract the checkpoint's calibration offset.
            rectify_output: Return rectified rather than raw pixel
                coordinates. Requires ``camera_matrix`` and ``distortion``.
            camera_matrix: 3×3 intrinsics ``K`` for this rig.
            distortion: Distortion coefficients for this rig.

        Returns:
            A :class:`LaserPrediction` in raw pixel space unless
            ``rectify_output`` is set.

        Raises:
            ValueError: If required inputs for the selected options are
                missing, or the image channel count contradicts the
                checkpoint.
        """
        torch = self._torch

        image_bgr, from_image_obj = _as_bgr_array(image)
        if bayer_excess is None and from_image_obj is not None:
            bayer_excess = from_image_obj

        if self.in_channels == 6 and bayer_excess is None:
            raise ValueError(
                "this checkpoint expects 6 input channels, so Bayer-excess G/R "
                "is required; pass a LinearRawImage or an explicit "
                "bayer_excess array"
            )
        if bayer_excess is not None and bayer_excess.shape[:2] != image_bgr.shape[:2]:
            raise ValueError(
                f"bayer_excess shape {bayer_excess.shape[:2]} does not match "
                f"image shape {image_bgr.shape[:2]}"
            )
        if rectify_output and (camera_matrix is None or distortion is None):
            raise ValueError(
                "rectify_output=True requires camera_matrix and distortion"
            )

        line_abc: tuple[float, float, float] | None = None
        line_confidence = 0.0
        if dive_line is not None:
            a, b, c, conf = dive_line
            line_abc, line_confidence = (a, b, c), float(conf)
        # The reference only forwards the line when soft-snap is enabled,
        # which also gates the corridor mask. Preserve that coupling.
        if not soft_snap and line_mask_corridor_px is None:
            line_abc, line_confidence = None, 0.0

        autocast_dtype = torch.bfloat16 if use_bf16 else None
        common = {
            "wavelength": wavelength,
            "tile": tile,
            "overlap": overlap,
            "batch_size": batch_size,
            "presence_threshold": presence_threshold,
            "autocast_dtype": autocast_dtype,
            "line_abc": line_abc,
            "line_confidence": line_confidence,
            "tau_line": tau_line,
            "alpha_max": alpha_max if soft_snap else 0.0,
            "rig_prior": rig_prior,
            "rig_prior_bbox": rig_prior_bbox,
            "rig_prior_center": rig_prior_center,
            "rig_prior_sigma": rig_prior_sigma,
            "rig_prior_floor": rig_prior_floor,
            "bayer_excess_image": bayer_excess,
            "bayer_excess_scale": bayer_excess_scale,
            "subpixel_refine": subpixel_refine,
            "line_mask_corridor_px": line_mask_corridor_px,
        }

        with torch.inference_mode():
            if cascade:
                pred = self._predict_cascade(
                    image_bgr, refine_window=refine_window, **common
                )
            else:
                pred = self._predict_coarse(image_bgr, **common)

        if pred.x is None or pred.y is None:
            return pred

        x, y = pred.x, pred.y
        if apply_bias_offset:
            h, w = image_bgr.shape[:2]
            x = max(0.0, min(x - self.bias_offset[0], float(w - 1)))
            y = max(0.0, min(y - self.bias_offset[1], float(h - 1)))

        # Rectify last: the bias offset was calibrated against label residuals
        # in raw space, so it has to cancel there before the coordinate-frame
        # conversion happens.
        if rectify_output:
            x, y = rectify_prediction(x, y, camera_matrix, distortion)

        return LaserPrediction(x=x, y=y, confidence=pred.confidence)


def _as_bgr_array(image: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize ``image`` to a BGR ndarray, plus Bayer-excess when available.

    Returns ``(bgr, bayer_excess_or_None)``.
    """
    bayer = getattr(image, "bayer_excess", None)
    data = image.data if hasattr(image, "data") else image
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(
            f"expected an [H, W, 3] BGR image, got shape {data.shape}"
        )
    return data, bayer
