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
