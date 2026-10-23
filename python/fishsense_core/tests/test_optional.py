"""fishsense_core._optional.require: the one lazy-import guard for the extras."""

import sys

import pytest

from fishsense_core._optional import require


def test_returns_the_module_when_installed():
    assert require("json", "whatever", "the test") is sys.modules["json"]


def test_names_the_feature_and_the_extra_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as excinfo:
        require("torch", "laser-detector", "The laser detector")
    assert str(excinfo.value) == (
        "The laser detector requires torch. Install the optional extra: "
        "pip install 'fishsense_core[laser-detector]'"
    )
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_several_extras_are_each_offered(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError) as excinfo:
        require("huggingface_hub", ("headtail-keypoints", "laser-detector"), "HuggingFaceStore")
    assert str(excinfo.value) == (
        "HuggingFaceStore requires huggingface_hub. Install the optional extras: "
        "pip install 'fishsense_core[headtail-keypoints]' or "
        "pip install 'fishsense_core[laser-detector]'"
    )


def test_a_reason_is_carried_into_the_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "bm3d", None)
    with pytest.raises(ImportError, match=r"requires bm3d\. It is slow\. Install the optional extra"):
        require("bm3d", "denoise", "BM3D denoising", why="It is slow.")


def test_every_hand_rolled_guard_is_gone():
    """The helper is the only copy: no module raises its own
    'Install the optional extra' message any more."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    import fishsense_core  # pylint: disable=import-outside-toplevel

    root = Path(fishsense_core.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "_optional.py" and "Install the optional extra" in path.read_text()
    ]
    assert not offenders, offenders
