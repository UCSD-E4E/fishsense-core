"""Tests for the NVIDIA-CUDA library preloader in `fishsense_core/__init__.py`.

The preloader only matters at import time on Linux with a CUDA-enabled wheel
and an `nvidia-*-cu*` pip package on the path. It must be a no-op (no raised
exception) on every other configuration: macOS/Windows, no `nvidia` package,
or a non-CUDA build.
"""
# pylint: disable=import-error
import ctypes
import os
import sys
import types

import pytest

import fishsense_core


def test_preload_helper_is_idempotent_and_safe():
    """Calling the preloader directly must never raise, regardless of host."""
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access


def test_preload_no_op_when_nvidia_package_missing(monkeypatch):
    """If `import nvidia` fails, the preloader returns silently."""
    monkeypatch.setitem(sys.modules, "nvidia", None)  # forces ImportError
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access


def test_preload_no_op_when_nvidia_dir_missing(monkeypatch):
    """A bogus `nvidia.__file__` should not crash — `isdir` short-circuits."""
    fake = types.ModuleType("nvidia")
    fake.__file__ = "/definitely/not/a/real/path/__init__.py"
    monkeypatch.setitem(sys.modules, "nvidia", fake)
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="linux-only path")
def test_preload_runs_on_linux():
    """Smoke check that the linux branch is reached without errors when
    invoked. Relies on global no-raise contract."""
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access


# --------------------------------------------------------------------------
# Namespace-package handling
#
# The `nvidia-*-cu12` wheels each contribute a subpackage to a shared `nvidia`
# *namespace* package. Verified against a real torch install: `nvidia.__file__`
# exists but is None, and only `nvidia.__path__` (a `_NamespacePath`) names the
# directory. Every fake in the tests above sets `__file__` to a string, so this
# — the only shape that occurs in practice — went uncovered, and
# `os.path.dirname(None)` raised TypeError out of `import fishsense_core`.
# --------------------------------------------------------------------------


def _fake_nvidia_tree(root):
    """Build an `nvidia/<pkg>/lib/<lib>.so*` tree and return the root dir."""
    nvidia_root = root / "nvidia"
    for pkg, lib in (("cublas", "libcublas.so.12"), ("cudnn", "libcudnn.so.9")):
        lib_dir = nvidia_root / pkg / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / lib).write_bytes(b"")
    return nvidia_root


def _namespace_module(path):
    """A stand-in matching a real namespace package: __file__ present but None."""
    module = types.ModuleType("nvidia")
    module.__file__ = None
    module.__path__ = [str(path)]
    return module


def test_preload_handles_namespace_package(monkeypatch, tmp_path):
    """Regression: a namespace `nvidia` must not raise out of import.

    Before the fix this raised `TypeError: expected str, bytes or os.PathLike
    object, not NoneType`, breaking `import fishsense_core` in any environment
    with torch installed — i.e. every laser-detector environment.
    """
    nvidia_root = _fake_nvidia_tree(tmp_path)
    monkeypatch.setitem(sys.modules, "nvidia", _namespace_module(nvidia_root))
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access


def test_preload_actually_loads_libs_from_namespace_path(monkeypatch, tmp_path):
    """The namespace path must still be *searched*, not just survived.

    Guards against a fix that only dodges the crash (e.g. bailing out when
    `__file__` is None). That would silently disable the preloader and bring
    back the ORT `dlopen` failures it exists to prevent — a regression no
    no-raise test would catch.
    """
    nvidia_root = _fake_nvidia_tree(tmp_path)
    monkeypatch.setitem(sys.modules, "nvidia", _namespace_module(nvidia_root))

    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access

    assert sorted(os.path.basename(p) for p in loaded) == [
        "libcublas.so.12",
        "libcudnn.so.9",
    ]


def test_preload_falls_back_to_file_for_regular_package(monkeypatch, tmp_path):
    """A conventional (non-namespace) `nvidia` package still resolves."""
    nvidia_root = _fake_nvidia_tree(tmp_path)
    module = types.ModuleType("nvidia")
    module.__file__ = str(nvidia_root / "__init__.py")
    monkeypatch.setitem(sys.modules, "nvidia", module)

    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access

    assert len(loaded) == 2


def test_preload_swallows_dlopen_failures(monkeypatch, tmp_path):
    """A library that fails to load must not abort the remaining preloads."""
    nvidia_root = _fake_nvidia_tree(tmp_path)
    monkeypatch.setitem(sys.modules, "nvidia", _namespace_module(nvidia_root))

    def _boom(path, mode=0):
        raise OSError(f"cannot load {path}")

    monkeypatch.setattr(ctypes, "CDLL", _boom)
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access


def test_preload_survives_malformed_nvidia_module(monkeypatch):
    """An `nvidia` module of an unanticipated shape must not break import.

    This is the general form of the namespace bug: the preloader is
    best-effort, runs at package-import time, and so must never be the reason
    `import fishsense_core` fails.
    """

    class _Hostile(types.ModuleType):
        @property
        def __path__(self):
            raise RuntimeError("no __path__ for you")

    monkeypatch.setitem(sys.modules, "nvidia", _Hostile("nvidia"))
    fishsense_core._preload_nvidia_libs()  # pylint: disable=protected-access
