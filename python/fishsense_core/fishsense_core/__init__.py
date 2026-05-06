"""FishSense Core Package"""

import ctypes
import os
import sys
from glob import glob

__version__ = "0.1.0"


def _preload_nvidia_libs() -> None:
    """Pre-load CUDA shared libraries from any installed ``nvidia-*`` pip
    packages so ONNX Runtime's CUDA provider can resolve them via ``dlopen``.

    ORT's ``libonnxruntime_providers_cuda.so`` has DT_NEEDED entries like
    ``libcudart.so.12`` with no rpath. When fishsense_core is imported into a
    "cold" Python process (no torch/etc already loaded), the dynamic loader
    can't find those libs even though they're sitting in
    ``site-packages/nvidia/*/lib``. We replicate what torch's ``__init__``
    does: walk the ``nvidia/`` subpackages and ``dlopen`` each ``.so*`` with
    ``RTLD_GLOBAL`` so symbols are visible to subsequent loads.

    Failures are intentionally swallowed: this is a best-effort preload, and
    every "expected" failure path (no nvidia-* installed, libs already loaded
    by another importer, no CUDA at all, non-CUDA build) should be silent.
    """
    if not sys.platform.startswith("linux"):
        return

    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        return

    nvidia_root = os.path.dirname(nvidia.__file__)
    if not os.path.isdir(nvidia_root):
        return

    for lib in sorted(glob(os.path.join(nvidia_root, "*", "lib", "*.so*"))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_nvidia_libs()
