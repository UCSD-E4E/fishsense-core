"""Board-mask inference — the checkpoint slot for `fishsense-core[slate]`.

Mirrors the laser detector's surface (`LaserDetector.from_checkpoint`) so the
data-worker integration looks familiar:

    from fishsense_core.slate import BoardMasker

    masker = BoardMasker.from_checkpoint("/e4efs/models/board_unet_v1.pt")
    mask = masker.predict(bgr)                    # HxW float32 probabilities
    result = predict_slate(..., board_mask=mask)  # optional everywhere

**Torch is imported lazily.** This module is importable, and `predict_slate`
fully usable, without the `[mask]` extra installed — the classical path costs
~13 points of coverage (80% -> 67% seeded) but works. Build the wrapper against
the classical path first and wire the checkpoint in later; nothing here has to
exist for that to ship.

Checkpoint format is deliberately small and self-describing::

    {"model": <state_dict>, "size": (width, height)}

`size` travels with the weights because the preprocessing must match training
exactly (the model was trained at 512x384); reading it from the file removes
one thing a caller can get wrong.

Cost on CPU: **202 ms/frame** at 512x384 with 4 threads, 1.08M params, 4.4 MB.
No GPU, no `nodeAffinity`.

Ported from ``slate_training.mask`` (``UCSD-E4E/2026-07-31_slate_training``,
same owner); fishsense-core is the canonical home. Torch + huggingface-hub
are the ``[slate]`` extra — this module imports without them (they load
lazily in ``build_unet`` / ``from_pretrained`` / ``predict``), so the base
install can import it and the classical ``estimate_plane(board_mask=None)``
path stays fully usable without the extra.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import cv2
import numpy as np

# torch / huggingface_hub load lazily (they are the [slate] extra) — that is the
# whole point, so import-outside-toplevel / import-error are expected. cv2 is a
# C extension pylint cannot introspect (no-member). The from_pretrained arg list
# mirrors LaserDetector.from_pretrained.
# pylint: disable=import-outside-toplevel,import-error,no-member
# pylint: disable=too-many-arguments,too-many-positional-arguments

__all__ = [
    "BoardMasker",
    "build_unet",
    "preprocess",
    "DEFAULT_HF_REPO",
    "DEFAULT_CHECKPOINT",
    "MASK_MEAN",
    "MASK_STD",
]

# Mirrors the laser detector's convention (ucsde4e/fishsense-laser-detector).
# Note the data-worker Dockerfile bakes that checkpoint in at build time so the
# activity never reaches HuggingFace at run time -- preemptible NRP pods with
# no HF token. Do the same here.
DEFAULT_HF_REPO = "ucsde4e/fishsense-slate-detector"
DEFAULT_CHECKPOINT = "board_unet_v0.1.0.pt"

# Training-time normalization. Must match `scripts/train_mask.py` exactly; a
# mismatch here degrades the mask silently rather than raising.
MASK_MEAN = 0.45
MASK_STD = 0.25


def build_unet(channels: Tuple[int, int, int, int] = (24, 48, 96, 192)) -> Any:
    """Construct the board-segmentation UNet (1.08M params at the default).

    Defined here rather than imported from `scripts/` so the architecture ships
    with the installable package — a checkpoint is useless to a consumer who
    cannot instantiate the model that matches it.
    """
    # The nested nn.Module is a straightforward UNet; its docstring/attribute
    # counts are not worth churning a ported architecture over.
    # pylint: disable=missing-class-docstring,missing-function-docstring
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def block(cin: int, cout: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3, c4 = channels
            self.d1, self.d2, self.d3 = block(3, c1), block(c1, c2), block(c2, c3)
            self.bottom = block(c3, c4)
            self.u3, self.c3 = nn.ConvTranspose2d(c4, c3, 2, 2), block(c3 * 2, c3)
            self.u2, self.c2 = nn.ConvTranspose2d(c3, c2, 2, 2), block(c2 * 2, c2)
            self.u1, self.c1 = nn.ConvTranspose2d(c2, c1, 2, 2), block(c1 * 2, c1)
            self.head = nn.Conv2d(c1, 1, 1)

        def forward(self, x: Any) -> Any:
            d1 = self.d1(x)
            d2 = self.d2(F.max_pool2d(d1, 2))
            d3 = self.d3(F.max_pool2d(d2, 2))
            b = self.bottom(F.max_pool2d(d3, 2))
            x = self.c3(torch.cat([self.u3(b), d3], 1))
            x = self.c2(torch.cat([self.u2(x), d2], 1))
            x = self.c1(torch.cat([self.u1(x), d1], 1))
            return self.head(x)

    return UNet()


def preprocess(bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize and normalize a BGR frame into CHW float32, as trained.

    Note **BGR, not RGB** — training read frames with `cv2.imread`, so the
    channel order is baked into the weights. Feeding RGB will not error; it
    will just quietly produce a worse mask.
    """
    resized = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MASK_MEAN) / MASK_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


class BoardMasker:
    """Loads a board-mask checkpoint and predicts board probability maps."""

    def __init__(self, model: Any, size: Tuple[int, int], device: str = "cpu"):
        self._model = model
        self._size = size
        self._device = device

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) the model expects; comes from the checkpoint."""
        return self._size

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", threads: int | None = 4
    ) -> "BoardMasker":
        """Load a checkpoint written by `scripts/train_mask.py`.

        `threads` caps torch's CPU thread pool. Defaulted because the
        data-worker runs many activities concurrently and an uncapped pool
        will happily consume every core for a 200 ms model.
        """
        import torch  # noqa: PLC0415

        if not os.path.exists(path):
            raise FileNotFoundError(f"board mask checkpoint not found: {path}")
        if threads:
            torch.set_num_threads(int(threads))

        payload = torch.load(path, map_location=device)
        if "model" not in payload or "size" not in payload:
            raise ValueError(
                f"{path} is not a board-mask checkpoint "
                "(expected keys 'model' and 'size')"
            )
        model = build_unet()
        model.load_state_dict(payload["model"])
        model.eval().to(device)
        return cls(model, tuple(payload["size"]), device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_HF_REPO,
        *,
        filename: str = DEFAULT_CHECKPOINT,
        revision: str = "main",
        device: str = "cpu",
        threads: int | None = 4,
    ) -> "BoardMasker":
        """Download a published checkpoint from HuggingFace and load it.

        Signature mirrors `LaserDetector.from_pretrained` so the two detectors
        read the same way at the call site. `device` defaults to CPU rather
        than CUDA-when-available: this model is 202 ms/frame on CPU and the
        activity is deliberately not GPU-scheduled.
        """
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "from_pretrained requires huggingface_hub. Install the "
                "optional extra: pip install 'fishsense_core[slate]'"
            ) from exc

        path = hf_hub_download(repo_id, filename, revision=revision)
        return cls.from_checkpoint(path, device=device, threads=threads)

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Board probability map for one frame.

        Returns a float32 array in [0, 1] at the model's own resolution, not
        the frame's — `baseline.mask_candidates` rescales it, so there is no
        need to resize here and no ambiguity about which resolution wins.
        """
        import torch  # noqa: PLC0415

        tensor = torch.from_numpy(preprocess(bgr, self._size))[None].to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.sigmoid(logits)[0, 0]
        return probs.detach().cpu().numpy().astype(np.float32)
