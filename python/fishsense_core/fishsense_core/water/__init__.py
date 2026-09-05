"""Water-column physics: measuring attenuation, and inverting it.

Two modules, in the order they are used:

* :mod:`fishsense_core.water.attenuation` fits per-channel attenuation
  coefficients from observations of a constant-reflectance target — the dive
  slate — at known metric ranges.
* :mod:`fishsense_core.water.seathru` inverts the image formation model with
  those coefficients and a range, recovering the direct signal.

Both are pure functions over arrays and scalars. Neither knows about dives,
labels or the decode chain, and nothing in the decode calls them by default —
:class:`~fishsense_core.image.decode.DecodeConfig` has ``beta`` and ``range_m``
off, and cannot do otherwise: ``RawImage(raw_bytes)`` has no idea which dive a
frame came from or how far away the subject was, and the correction needs both.
"""
