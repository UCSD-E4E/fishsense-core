"""Import an optional dependency, or say which extra installs it.

torch, timm, segmentation-models-pytorch, huggingface_hub and bm3d are
extras, not base dependencies: torch alone pulls its own CUDA runtime, which
would bloat every wheel and collide with the onnxruntime CPU/+cu12 split the
released wheels rely on. So the modules that need them import lazily, through
here, and a missing one fails with the ``pip install`` line that fixes it
rather than a bare ImportError. This is the only copy of that guard.
"""

from __future__ import annotations

import importlib
from typing import Any, Sequence

__all__ = ["require"]


def require(module: str, extra: str | Sequence[str], feature: str, why: str = "") -> Any:
    """Import ``module`` and return it.

    Args:
        module: The importable name, e.g. ``"huggingface_hub"``.
        extra: The ``fishsense_core[...]`` extra that installs it, or several
            when more than one does (the message offers each).
        feature: What the caller was doing, e.g. ``"HuggingFaceStore"``.
        why: Optional sentence on why it is not a base dependency.

    Raises:
        ImportError: It is not installed; chained from the original.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extras = [extra] if isinstance(extra, str) else list(extra)
        install = " or ".join(f"pip install 'fishsense_core[{e}]'" for e in extras)
        reason = f" {why.strip()}" if why else ""
        noun = "extra" if len(extras) == 1 else "extras"
        raise ImportError(
            f"{feature} requires {module}.{reason} Install the optional {noun}: {install}"
        ) from exc
