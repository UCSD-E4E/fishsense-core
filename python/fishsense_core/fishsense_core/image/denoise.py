"""BM3D on the luminance of a finished decode, with a measured noise PSD.

**Not a default, and not close to one.** BM3D costs 15-36x the entire decode —
133.8 s per 12-megapixel frame at the ``np`` profile and 56.8 s at
``BM3DProfileLC``, against 3.7 s for the whole chain — so it cannot sit inline
until that is solved. It is here because it is the only denoiser measured that
keeps a fine scale lattice, and because that question is now a *measurable* one
(see :mod:`fishsense_core.image.texture`) rather than a matter of taste.

``bm3d`` is an optional extra: ``pip install 'fishsense_core[denoise]'``.

Why this, after Noise2Void
--------------------------

Every N2V variant erased the fine reticulated scale lattice completely —
retention 0.00 for random crops, fish-centred crops and a shallow net alike —
and the reason is geometric rather than a training choice. The lattice's 4.5 px
period in the output is 2.25 px in the half-resolution photosite planes N2V
works on: at the plane's Nyquist limit, where a blind-spot network cannot tell
a periodic pattern from pixel noise. Anything that hopes to keep fine scales
has to work at full resolution.

BM3D (Dabov et al. 2007) is the classical answer for exactly this texture. It
groups similar patches across the image and filters them jointly in a 3D
transform, so a periodic lattice — as self-similar as image content gets — is
reinforced rather than averaged away. It is non-learned, deterministic, and can
only remove.

Two choices particular to this pipeline
---------------------------------------

**It takes a noise PSD, not a sigma.** Demosaicing colours the noise: it rolls
off at high frequency. A white-noise sigma would over-smooth there, which is
precisely where the scales live. An open-water patch's spectrum *is* the PSD.

**It sits at the JPEG stage, on L only.** After demosaic, in CIELAB, with a and
b passed through untouched — so hue, which the species and slate labels depend
on, does not move, and the enhancer is geometrically inert and passes
:func:`~fishsense_core.image.contract.probe_geometry` like any other. The
mosaic-domain alternative could offer neither.

What it is measured to do
-------------------------

===============================  ==========  ======  ============  ======
                                 retention   grain   displacement  time
===============================  ==========  ======  ============  ======
Noise2Void (Bayer domain)        0.00        /4.78   0.111 px      30 s
BM3D, 0.5x measured PSD          **0.84**    /3.67   **0.009 px**  173 s
BM3D, 1.0x measured PSD          0.66        /3.47   0.002 px      194 s
===============================  ==========  ======  ============  ======

Half the measured PSD keeps 84% of the scale peak with three-quarters of N2V's
grain reduction, and moves nothing — displacement at the phase-correlation
floor, a tenth of N2V's. At full strength the rock takes on BM3D's familiar
painted look and the scales visibly weaken.

**Neither denoiser dominates**, and which is right depends on a number nobody
has yet: the scale pitch identity work will need in output pixels. On a fine
~4.5 px lattice, N2V erases it and BM3D keeps 86%. On coarser 10-20 px body
scales and stripes across 12 scanned frames, N2V keeps them (>= 0.84 on 10 of
12) while BM3D 0.5 keeps 0.43-0.98, losing most on the noisiest frames where it
works hardest. Above Nyquist the blind-spot network sees the periodicity and
preserves it; BM3D's 8x8 collaborative filtering favours sharp fine lattices
over soft coarse modulation.

The underlying limit is unchanged either way: fish texture is only about 1.6x
the grain in the same frequency band, and no filter in any domain separates
them cleanly. That is an information limit, not a tuning failure, and it is why
this is opt-in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom
from skimage.color import lab2rgb, rgb2lab  # pylint: disable=no-name-in-module

from fishsense_core.image.texture import SIGNIFICANCE, texture_power

__all__ = [
    "BM3D_PROFILES",
    "BM3DConfig",
    "TexturedNoiseRegion",
    "bm3d_enhancer",
    "denoise_luminance",
    "noise_psd_from_water",
]

#: Profile names :class:`BM3DConfig` accepts.
#:
#: ``bm3d.bm3d`` takes either one of its own strings —
#: ``np``/``refilter``/``vn``/``high``/``vn_old``/``deb`` — or a
#: ``BM3DProfile`` object. ``"lc"`` is **not** among the strings: the
#: low-complexity profile exists only as ``bm3d.BM3DProfileLC``, so passing
#: the obvious ``profile="lc"`` runs the entire decode and then raises
#: ``TypeError`` from inside the filter, minutes in. It is accepted here and
#: resolved to that object, because it is the profile the cost measurements
#: quote (56.8 s/frame against the reference profile's 133.8 s) and a name
#: that is documented has to work.
BM3D_PROFILES = ("np", "lc", "refilter", "vn", "high", "vn_old", "deb")


class TexturedNoiseRegion(UserWarning):
    """The region the noise PSD was measured from carries real texture.

    BM3D removes whatever the PSD says is noise. When a test frame's scale
    lattice extended into the PSD region, the noise model contained the scales
    and the filter erased them across the whole frame — retention 0.000 against
    226 sigma of input. The top-left fifth is open water by convention, not by
    guarantee, so the region is checked for a significant spectral peak and the
    frame is processed with this warning rather than over-filtered in silence.
    """


def _import_bm3d():
    """Import ``bm3d``, restoring the NumPy alias it still needs.

    bm3d 4.0.x calls ``np.trapz``, which NumPy 2 removed in favour of
    ``np.trapezoid``. Same function, new name; the alias is restored before
    import so the package loads on this NumPy.
    """
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    try:
        import bm3d  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "BM3D denoising requires the `bm3d` package, which is not a base "
            "dependency: it costs 15-36x the entire decode and nothing here "
            "enables it by default. Install the extra: "
            "pip install 'fishsense_core[denoise]'."
        ) from exc
    return bm3d


@dataclass(frozen=True)
class BM3DConfig:
    """How hard to filter, and against what noise model."""

    #: Tile used to estimate the PSD from open water.
    psd_size: int = 64
    #: Scales the PSD handed to BM3D. 1.0 is as measured; 0.5 is the setting
    #: that kept 84% of the scale peak.
    strength: float = 1.0
    #: BM3D profile: one of :data:`BM3D_PROFILES`, or a ``bm3d.BM3DProfile``
    #: object. ``"np"`` is the reference quality at 133.8 s/frame; ``"lc"`` is
    #: 56.8 s. Neither is fast enough to sit inline.
    profile: str = "np"
    #: PSD multiplier for the a and b channels; 0 leaves them untouched.
    #:
    #: Scales are luminance texture, so chroma can be filtered far harder than
    #: L without touching them — and what L-only filtering leaves behind is a
    #: faint coloured mottle in water and rock. Only the speckle goes: the
    #: local mean of a and b, which is the hue the species and slate labels
    #: read, is preserved.
    chroma_strength: float = 0.0

    def __post_init__(self) -> None:
        # Validated as a *name*, without importing bm3d: the package is an
        # optional extra, and building a config must not require it. The
        # alternative is discovering the typo after a full-frame decode.
        if isinstance(self.profile, str) and self.profile not in BM3D_PROFILES:
            raise ValueError(
                f"profile must be one of {BM3D_PROFILES} or a bm3d.BM3DProfile "
                f"object, got {self.profile!r}"
            )
        if self.psd_size < 8:
            raise ValueError(f"psd_size must be at least 8, got {self.psd_size}")
        if self.strength <= 0:
            raise ValueError(f"strength must be positive, got {self.strength}")
        if self.chroma_strength < 0:
            raise ValueError(
                f"chroma_strength must be >= 0, got {self.chroma_strength}"
            )


def noise_psd_from_water(water: np.ndarray, size: int = 64) -> np.ndarray:
    """Noise power spectral density from a textureless patch.

    Welch-style: the patch is cut into ``size``-square tiles, each mean-removed
    and Hann-windowed, and their ``|FFT|²`` averaged. Normalized by the
    window's energy so white noise of variance sigma² reads sigma² in every bin
    — the same convention as
    :func:`fishsense_core.image.texture.radial_power_spectrum`. Returned
    fft-shifted, DC at the centre.
    """
    arr = np.asarray(water, dtype=np.float64)
    height, width = arr.shape
    size = int(min(size, height, width))
    window = np.outer(np.hanning(size), np.hanning(size))
    norm = float((window**2).sum())

    accumulated = np.zeros((size, size))
    tiles = 0
    for y in range(0, height - size + 1, size // 2):
        for x in range(0, width - size + 1, size // 2):
            tile = arr[y : y + size, x : x + size]
            tile = tile - tile.mean()
            accumulated += np.abs(np.fft.fft2(tile * window)) ** 2 / norm
            tiles += 1
    return np.fft.fftshift(accumulated / max(tiles, 1))


def _psd_for_image(psd: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a small PSD onto an image-sized grid, in ``bm3d``'s convention.

    The package wants a PSD the size of the image in unnormalized-FFT units:
    white noise of variance sigma² is ``sigma² * M * N`` in every bin. Ours is
    per-bin variance, so the factor is M*N.
    """
    height, width = shape
    resampled = zoom(psd, (height / psd.shape[0], width / psd.shape[1]), order=1)
    resampled = np.maximum(resampled, 1e-12)
    return np.fft.ifftshift(resampled) * float(height * width)


def _resolve_profile(bm3d, profile):
    """Turn a :data:`BM3D_PROFILES` name into what ``bm3d.bm3d`` accepts.

    Only ``"lc"`` needs translating; every other name is one of the package's
    own strings, and a ``BM3DProfile`` object passes straight through.
    """
    if profile == "lc":
        return bm3d.BM3DProfileLC()
    return profile


def denoise_luminance(
    lum: np.ndarray, water: np.ndarray, config: BM3DConfig | None = None
) -> np.ndarray:
    """BM3D on a [0, 255] luminance array, noise model taken from ``water``."""
    bm3d = _import_bm3d()

    config = config or BM3DConfig()
    arr = np.asarray(lum, dtype=np.float64) / 255.0
    psd = noise_psd_from_water(
        np.asarray(water, dtype=np.float64) / 255.0, config.psd_size
    )
    psd_image = _psd_for_image(psd * config.strength, arr.shape)
    out = bm3d.bm3d(
        arr, sigma_psd=psd_image, profile=_resolve_profile(bm3d, config.profile)
    )
    return np.clip(np.asarray(out, dtype=np.float64) * 255.0, 0.0, 255.0)


def _warn_if_region_is_textured(region: np.ndarray) -> None:
    prominence, noise = texture_power(region)
    if noise > 0 and prominence > SIGNIFICANCE * noise:
        warnings.warn(
            f"noise-PSD region has a {prominence / noise:.1f} sigma spectral peak "
            "in the scale band; BM3D will treat that texture as noise and remove "
            "it everywhere",
            TexturedNoiseRegion,
            stacklevel=3,
        )


def bm3d_enhancer(config: BM3DConfig | None = None):
    """Build an :data:`~fishsense_core.image.contract.Enhancer`.

    uint8 RGB in, uint8 RGB out; L filtered, a and b kept unless
    ``chroma_strength`` says otherwise. The open-water patch is taken as the
    top-left fifth of the frame, the same convention every noise figure in this
    work uses — and checked, because it is a convention rather than a
    guarantee.
    """
    config = config or BM3DConfig()

    def enhance(image: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image)
        lab = rgb2lab(rgb.astype(np.float64) / 255.0)
        lum = lab[..., 0] * 2.55
        height, width = lum.shape
        region = np.s_[
            : max(height // 5, config.psd_size), : max(width // 5, config.psd_size)
        ]
        _warn_if_region_is_textured(lum[region])
        lab[..., 0] = denoise_luminance(lum, lum[region], config) / 2.55

        if config.chroma_strength > 0:
            # a and b live on roughly [-128, 128]; shift to [0, 255] for the
            # same [0, 1] range the L path uses, and take each channel's own
            # PSD from the same open-water region.
            chroma_config = BM3DConfig(
                psd_size=config.psd_size,
                strength=config.chroma_strength,
                profile=config.profile,
            )
            for channel in (1, 2):
                plane = lab[..., channel] + 128.0
                lab[..., channel] = (
                    denoise_luminance(plane, plane[region], chroma_config) - 128.0
                )

        return np.rint(np.clip(lab2rgb(lab) * 255.0, 0, 255)).astype(np.uint8)

    enhance.__name__ = f"bm3d[{config.strength:g}]"
    return enhance
