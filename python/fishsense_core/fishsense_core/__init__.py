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


def _set_ort_dylib_path() -> None:
    """Point ort's ``load-dynamic`` mode at the ``onnxruntime`` pip package's
    bundled ``libonnxruntime.so``.

    The ``+cu12`` wheel is built with ``--features cuda`` → ``ort/load-dynamic``,
    so the extension links no ONNX Runtime of its own; it ``dlopen``s
    ``libonnxruntime.so`` from ``$ORT_DYLIB_PATH`` at first use. The CUDA wheel
    depends on ``onnxruntime-gpu`` — a manylinux_2_27 build that loads on
    glibc < 2.38, unlike pyke's prebuilt ORT-CUDA — so resolve that here.

    No-op when ``ORT_DYLIB_PATH`` is already set (respect explicit overrides),
    when ``onnxruntime`` isn't installed, or for the default CPU build (which
    statically links ORT and never reads the variable).
    """
    if os.environ.get("ORT_DYLIB_PATH"):
        return

    try:
        import onnxruntime  # type: ignore[import-not-found]
    except ImportError:
        return

    capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
    libs = sorted(glob(os.path.join(capi_dir, "libonnxruntime.so*")))
    if libs:
        os.environ["ORT_DYLIB_PATH"] = libs[-1]


_preload_nvidia_libs()
_set_ort_dylib_path()
