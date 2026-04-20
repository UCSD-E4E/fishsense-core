"""Fish detection, segmentation, and measurement."""
# pylint: disable=c-extension-no-member,no-name-in-module,import-error
from fishsense_core._native.fish import FishSegmentation

__all__ = ["FishSegmentation"]
