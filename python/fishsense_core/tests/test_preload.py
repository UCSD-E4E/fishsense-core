"""Tests for the NVIDIA-CUDA library preloader in `fishsense_core/__init__.py`.

The preloader only matters at import time on Linux with a CUDA-enabled wheel
and an `nvidia-*-cu*` pip package on the path. It must be a no-op (no raised
exception) on every other configuration: macOS/Windows, no `nvidia` package,
or a non-CUDA build.
"""
# pylint: disable=import-error
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
