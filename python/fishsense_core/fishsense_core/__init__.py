"""FishSense Core Package"""

import ctypes
import os
import sys
from glob import glob

from fishsense_core.camera_intrinsics import CameraIntrinsics

__all__ = ["CameraIntrinsics"]

__version__ = "0.1.0"


def _preload_nvidia_libs() -> None:
    """Pre-load CUDA shared libraries from any installed ``nvidia-*`` pip
    packages so ONNX Runtime's CUDA provider can resolve them via ``dlopen``.

    ORT's ``libonnxruntime_providers_cuda.so`` has DT_NEEDED entries like
    ``libcudart.so.12`` with no rpath. When fishsense_core is imported into a
    "cold" Python process (no torch/etc already loaded), the dynamic loader
    can't find those libs even though they're sitting in
    ``site-packages/nvidia/*/lib``. We replicate what torch's ``__init__``
    does: ``dlopen`` a fixed list of runtime libraries (`_CUDA_RUNTIME_LIBS`),
    in dependency order, from the ``nvidia/`` subpackages with
    ``RTLD_GLOBAL`` so symbols are visible to subsequent loads.

    Failures are intentionally swallowed: this is a best-effort preload, and
    every "expected" failure path (no nvidia-* installed, libs already loaded
    by another importer, no CUDA at all, non-CUDA build) should be silent.
    Because it runs at package-import time, it must never be the reason
    ``import fishsense_core`` fails — so discovery is guarded too, not just
    the individual ``dlopen`` calls.

    A list, not every ``.so*`` in the tree: the wheels also ship interposers.
    ``libnvblas``, global, shadows every CPU BLAS in the process with its
    ``sgemm_`` & co., and with no ``nvblas.conf`` it has no CPU fallback, so
    the next CPU GEMM (torch's convolution) called a null pointer;
    ``libcufftw`` does the same to FFTW. Nor does ORT need nvshmem's MPI/PMI
    bootstraps or the profiler libraries.
    """
    if not sys.platform.startswith("linux"):
        return

    try:
        import nvidia  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
    except ImportError:
        return

    try:
        roots = _nvidia_search_roots(nvidia)
    except Exception:  # pylint: disable=broad-except
        return

    for pattern in _CUDA_RUNTIME_LIBS:
        for root in roots:
            try:
                libs = sorted(glob(os.path.join(root, "*", "lib", pattern)))
            except OSError:
                continue
            for lib in libs:
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


#: What ORT's CUDA provider loads, dependencies first: cuBLAS needs cuBLASLt
#: (else it resolves one by RUNPATH, maybe a mismatched system copy, as
#: torch's list notes), cuSPARSE nvJitLink, cuSOLVER both. cuDNN 9 dlopens
#: its sub-libraries itself, from its own $ORIGIN; they are listed anyway so
#: it loads as it always has. Covers the cu12 layout (one package per
#: library) and cu13's (all under ``nvidia/cu13/lib``).
_CUDA_RUNTIME_LIBS = (
    "libcudart.so*",
    "libnvJitLink.so*",
    "libcublasLt.so*",
    "libcublas.so*",
    "libcusparse.so*",
    "libcusolver.so*",
    "libcufft.so*",
    "libcurand.so*",
    "libnvrtc.so*",
    "libnvrtc-builtins.so*",
    "libcudnn*.so*",
)


def _nvidia_search_roots(nvidia) -> list:
    """Directories under which to look for ``<pkg>/lib/*.so*``.

    The ``nvidia-*-cu12`` wheels each contribute a subpackage to a shared
    ``nvidia`` *namespace* package. Namespace packages set ``__file__`` to
    ``None`` and carry their location only in ``__path__``, so reading
    ``__file__`` alone raises ``TypeError`` in exactly the configuration that
    matters most (torch installed). Prefer ``__path__``; fall back to
    ``__file__`` for a conventional package.
    """
    roots = [str(p) for p in getattr(nvidia, "__path__", [])]
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]
    return [root for root in roots if os.path.isdir(root)]


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
        import onnxruntime  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
    except ImportError:
        return

    capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
    libs = sorted(glob(os.path.join(capi_dir, "libonnxruntime.so*")))
    if libs:
        os.environ["ORT_DYLIB_PATH"] = libs[-1]


_preload_nvidia_libs()
_set_ort_dylib_path()
