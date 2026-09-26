"""SAM 3.1 concept segmenter, loaded through the model manifest.

``sam3`` is **not** a dependency of fishsense_core, not even an optional extra.
It is published only as a git repository, and a git-only requirement in a
wheel's metadata is what makes the wheel impossible to ``pip install``. The
caller installs it. fishsense-lite pins ``sam3`` from
``git+https://github.com/facebookresearch/sam3.git`` at commit
``8e451d5eb43c817b64ae7577fb7b9ae223db88a9``, the commit this loader was
written against.

SAM 3.1 has no manifest entry yet: ``facebook/sam3.1`` is gated and does not
publish the checkpoint's hash. Until the ``model-weights`` copy is hashed into
``models.toml``, :func:`load_segmenter` raises ``KeyError`` for the default
model. :func:`build_segmenter` loads a checkpoint the caller has already
fetched.
"""

from __future__ import annotations

import os
from typing import Any

from fishsense_core.models import Manifest, WeightStore, fetch

SAM3_MODEL_NAME = "sam3"


class Sam3RequiresGpu(RuntimeError):
    """SAM 3.1 cannot be built without CUDA. Retrying on the same host cannot
    help, so a job runner should treat this as non-retryable."""


def _require_cuda() -> None:
    try:
        import torch  # pylint: disable=import-outside-toplevel,import-error

        available = bool(torch.cuda.is_available())
    except Exception:  # pylint: disable=broad-except
        available = False
    if not available:
        # `build_sam3_image_model` reaches `PositionEmbeddingSine`, which
        # precomputes its cache with a hardcoded `device="cuda"`, so
        # construction fails before any device handling is consulted.
        raise Sam3RequiresGpu(
            "SAM 3.1 requires a GPU: build_sam3_image_model allocates its "
            "position-encoding cache on a hardcoded device='cuda', and this "
            "process has no usable CUDA device."
        )


def build_segmenter(checkpoint_path: str | os.PathLike[str]) -> Any:
    """Build a ``Sam3Processor`` from a local checkpoint.

    **Loading prints four missing keys, and they are benign**:
    ``backbone.vision_backbone.convs.3.{conv_1x1,conv_3x3}.{weight,bias}``.
    The neck builds a conv for each of four scales, but the backbone discards
    the lowest-resolution level, so that conv was never trained. Upstream's
    ``_load_checkpoint`` uses ``strict=False`` and only prints, which means a
    wrong checkpoint would load just as quietly. That is why
    :func:`load_segmenter` verifies the file by hash first.

    Raises:
        Sam3RequiresGpu: No usable CUDA device.
        ImportError: ``sam3`` is not installed.
    """
    _require_cuda()
    # pylint: disable=import-outside-toplevel,import-error
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=str(checkpoint_path))
    model.eval()
    return Sam3Processor(model)


def load_segmenter(
    store: WeightStore | None,
    *,
    cache_dir: str | os.PathLike[str],
    version: str | None = None,
    manifest: Manifest | None = None,
) -> Any:
    """Fetch the pinned SAM 3.1 checkpoint (verified) and build the segmenter.

    The GPU check runs **before** the fetch, so a host that cannot run SAM
    does not download a 3.5 GB checkpoint first.

    Raises:
        Sam3RequiresGpu: No usable CUDA device.
        KeyError: The manifest has no such SAM version.
        fishsense_core.models.ModelIntegrityError: The store's bytes are not
            the pinned checkpoint.
        fishsense_core.models.ModelUnavailable: Not cached and no store has it.
    """
    _require_cuda()
    path = fetch(SAM3_MODEL_NAME, version, store=store, cache_dir=cache_dir, manifest=manifest)
    return build_segmenter(path)
