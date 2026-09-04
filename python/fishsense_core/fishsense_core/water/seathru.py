"""Inverting the underwater image formation model with measured coefficients.

The revised model (Akkaynak & Treibitz) for a target of reflectance rho_c at
range z::

    I_c = J_c * rho_c * exp(-beta_c * z)  +  B_c * (1 - exp(-beta_c * z))
          \\_______ direct signal _______/    \\____ backscatter / veil ____/

Both unknowns here are **measured**, which is the whole reason this is worth
doing after gray-world, white-patch and a per-channel stretch all failed:
``beta`` comes from the slate fit in
:mod:`fishsense_core.water.attenuation`, validated against the published
pure-water absorption spectrum across seven dives, and ``z`` from the laser's
metric range. A measured beta cannot over-correct into magenta the way a
statistical white balance does, because it is not inferred from the picture it
is correcting.

**The limitation is range, and it is real.** Sea-thru proper needs a per-pixel
depth map; the corpus this was built against has range at exactly one pixel, so
:func:`remove_water` treats z as uniform across the frame. That is right for
the fish the laser is on and increasingly wrong for background at another
distance. Scaling a monocular depth estimate by the laser's metric range is the
step that would lift the restriction; it is not implemented.

**Where it goes in the chain is forced, not stylistic.** This inverts a
radiance formation model, so it belongs on linear radiance — after
``rawpy.postprocess`` and *before* the auto-gamma. Applying it later inverts a
curve that is not in the model.

**What to expect.** The correction scales with range, so at the 1-1.5 m of
close-up field frames it is modest: a fitted beta implies gains of about
1.78/1.33/1.36 at 1.5 m, i.e. red gets roughly a third more than green. At 5 m
the same coefficients give red about 2.7x green, which is where the cast
actually hurts. Most of the visible improvement on close frames comes from the
tone treatment, not the physics; the physics earns its place further out.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["DEFAULT_VEIL_PERCENTILE", "estimate_veil", "remove_water"]

#: Percentile of each channel taken as the backscatter estimate. Where
#: reflectance is near zero the observation is essentially veil, which is the
#: standard dark-channel argument. Low rather than zero so one dead pixel does
#: not set it.
DEFAULT_VEIL_PERCENTILE = 0.5


def estimate_veil(
    linear_rgb: np.ndarray, *, percentile: float = DEFAULT_VEIL_PERCENTILE
) -> tuple[float, float, float]:
    """Per-channel backscatter estimate from the frame's dark tail.

    ``linear_rgb`` must be linear and in [0, 1]: the model is written in
    radiance, and estimating a veil through a gamma curve would return a number
    that is not the veil.
    """
    return tuple(  # type: ignore[return-value]
        float(np.percentile(linear_rgb[:, :, c], percentile)) for c in range(3)
    )


def remove_water(
    linear_rgb: np.ndarray,
    beta: Sequence[float],
    range_m: float,
    *,
    veil: Sequence[float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Recover direct signal from a linear frame, given beta and a range.

    Args:
        linear_rgb: linear RGB in [0, 1]. Applying this after gamma or a tone
            curve would invert a curve that is not in the model.
        beta: per-channel attenuation per metre, in ``(R, G, B)`` order.
        range_m: the metric range, applied uniformly — see the module
            docstring for what that costs.
        veil: backscatter per channel; estimated from the dark tail when None.
        normalize: rescale the result into [0, 1]. Off for tests, and for
            anyone who wants the radiance ratios preserved exactly.

    Returns:
        The direct-signal estimate, same shape as the input.
    """
    if range_m < 0:
        raise ValueError(f"range must be non-negative, got {range_m}")

    beta_arr = np.asarray(beta, dtype=np.float64)
    if beta_arr.shape != (3,):
        raise ValueError(f"beta must be three per-channel coefficients, got {beta!r}")

    veil_arr = np.asarray(
        estimate_veil(linear_rgb) if veil is None else veil, dtype=np.float64
    )
    transmission = np.exp(-beta_arr * float(range_m))

    # I = J*t + B*(1-t)  =>  J = (I - B*(1-t)) / t
    #
    # Clipped at zero rather than allowed negative: subtracting a veil larger
    # than the signal happens in genuinely dark regions, and letting it go
    # negative would invert those pixels once the result is renormalised.
    direct = (linear_rgb - veil_arr * (1.0 - transmission)) / transmission
    direct = np.clip(direct, 0.0, None)

    if normalize:
        peak = float(direct.max())
        if peak > 0:
            direct = direct / peak
    return direct
