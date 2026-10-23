"""Import an optional dependency, or say which extra installs it.

torch, timm and huggingface_hub are extras, not base dependencies: torch
alone pulls its own CUDA runtime, which would bloat every wheel and collide
with the onnxruntime CPU/+cu12 split the released wheels rely on. So the
modules that need them import lazily, through here, and a missing one fails
with the ``pip install`` line that fixes it rather than a bare ImportError.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["require"]


def require(module: str, extra: str, feature: str) -> Any:
    """Import ``module`` and return it.

    Raises:
        ImportError: It is not installed. The message names ``feature`` (what
            the caller was doing) and the ``fishsense_core[extra]`` that
            provides it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires {module}. Install the optional extra: "
            f"pip install 'fishsense_core[{extra}]'"
        ) from exc
