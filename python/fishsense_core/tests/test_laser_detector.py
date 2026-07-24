"""Tests for the laser-detector inference recipe.

The values asserted here were captured from the reference implementation in
the training repo (``UCSD-E4E/2026-05-02_laser_detector``,
``src/laser_detector/inference.py``) — every expectation in this file was
verified to match it exactly. They are hard-coded rather than computed so the
suite is self-contained and so a regression shows up as a test failure rather
than as a quiet accuracy loss.

Several tests exist specifically to pin down bugs that were expensive to find
upstream; those are called out individually.
"""

# pylint: disable=protected-access

import numpy as np
import pytest

from fishsense_core import _laser_detector as ld
from fishsense_core.image import linear_raw_image as lri


# --------------------------------------------------------------------------
# Recipe constants
# --------------------------------------------------------------------------


def test_wavelength_encoding_is_zero_to_one():
    """Guards the {0.0, 0.5, 1.0} encoding.

    A {-1, 0, +1} encoding appears in some downstream documentation. Using it
    degrades green frames specifically, and does so silently — the model still
    produces plausible-looking output.
    """
    assert ld.WAVELENGTH_CHANNEL == {"red": 1.0, "green": 0.0}
    assert ld.UNKNOWN_WAVELENGTH_CHANNEL == 0.5
    assert ld._wavelength_value("red") == 1.0
    assert ld._wavelength_value("green") == 0.0
    assert ld._wavelength_value(None) == 0.5
    assert ld._wavelength_value("infrared") == 0.5


def test_rig_prior_defaults_to_production_hard_bbox():
    """Production passes ``--rig-prior-floor 1.0``; the reference library
    default of 0.5 would leave a soft Gaussian in place."""
    assert ld.DEFAULT_RIG_PRIOR_BBOX == (1100, 700, 2950, 2180)
    assert ld.DEFAULT_RIG_PRIOR_FLOOR == 1.0


def test_known_checkpoint_bias_offset():
    # fp32-pipeline offset (upstream issue #13). The obsolete Ada-bf16 value
    # was (-0.200, -0.006); this must not silently revert to it.
    assert ld.CHECKPOINT_BIAS_OFFSETS["run3_epoch_021.pt"] == (-0.179, -0.023)


def test_inference_defaults_to_fp32():
    """Production disabled bf16 at inference (issue #13); the bias offset is
    calibrated for fp32, so the default must not re-enable autocast."""
    import inspect  # noqa: PLC0415

    assert inspect.signature(ld.LaserDetector.predict).parameters[
        "use_bf16"
    ].default is False


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


def test_tile_grid_full_sensor_frame():
    grid = ld.compute_tile_grid(3016, 4014)
    xs = sorted({x for x, _ in grid.origins})
    ys = sorted({y for _, y in grid.origins})
    assert xs == [0, 768, 1536, 2304, 2990]
    assert ys == [0, 768, 1536, 1992]
    assert len(grid.origins) == 20
    # Last tile on each axis snaps to the edge rather than keeping stride.
    assert xs[-1] == 4014 - 1024
    assert ys[-1] == 3016 - 1024
    assert (grid.padded_h, grid.padded_w) == (3016, 4014)


def test_tile_grid_pads_small_images():
    grid = ld.compute_tile_grid(500, 700)
    assert grid.origins == [(0, 0)]
    assert (grid.padded_h, grid.padded_w) == (1024, 1024)
    assert (grid.original_h, grid.original_w) == (500, 700)


def test_tile_grid_origins_are_row_major():
    grid = ld.compute_tile_grid(2048, 2048)
    assert grid.origins == sorted(grid.origins, key=lambda o: (o[1], o[0]))


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def test_chromaticity_norm_sums_to_one_and_scales_by_dtype():
    rgb16 = np.array([[[1000, 2000, 3000]]], dtype=np.uint16)
    chrom = ld.chromaticity_norm(rgb16)
    assert chrom.dtype == np.float32
    np.testing.assert_allclose(chrom.sum(axis=2), 1.0, rtol=1e-6)
    np.testing.assert_allclose(chrom[0, 0], [1 / 6, 2 / 6, 3 / 6], rtol=1e-5)

    # uint8 uses a /255 scale; chromaticity is scale-invariant so the ratios
    # match, which is exactly why a CLAHE'd image is not detectably wrong.
    rgb8 = np.array([[[10, 20, 30]]], dtype=np.uint8)
    np.testing.assert_allclose(
        ld.chromaticity_norm(rgb8)[0, 0], [1 / 6, 2 / 6, 3 / 6], rtol=1e-5
    )


def test_chromaticity_norm_black_pixel_uses_eps_floor():
    chrom = ld.chromaticity_norm(np.zeros((1, 1, 3), dtype=np.uint16))
    np.testing.assert_array_equal(chrom[0, 0], [0.0, 0.0, 0.0])


def test_preprocess_tile_channel_layout():
    tile = np.zeros((8, 8, 3), dtype=np.uint16)
    bayer = np.full((8, 8, 2), 2048, dtype=np.uint16)

    four = ld._preprocess_tile(tile, 1.0, None)
    assert four.shape == (4, 8, 8)
    np.testing.assert_allclose(four[3], 1.0)

    six = ld._preprocess_tile(tile, 0.0, bayer)
    assert six.shape == (6, 8, 8)
    np.testing.assert_allclose(six[3], 0.0)
    # Bayer channels are divided by the 4096 scale.
    np.testing.assert_allclose(six[4], 0.5)
    np.testing.assert_allclose(six[5], 0.5)


def test_preprocess_tile_converts_bgr_to_rgb():
    """The model consumes RGB chromaticity; the pipeline carries BGR."""
    tile = np.zeros((1, 1, 3), dtype=np.uint16)
    tile[0, 0] = [65535, 0, 0]  # pure blue in BGR
    out = ld._preprocess_tile(tile, 1.0, None)
    # Channel 2 of the RGB chromaticity triple is blue.
    np.testing.assert_allclose(out[:3, 0, 0], [0.0, 0.0, 1.0], atol=1e-6)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def test_rig_prior_floor_one_is_a_hard_bbox():
    mask = ld._rig_prior_for_tile(
        1000, 600, 256, ld.DEFAULT_RIG_PRIOR_BBOX, ld.DEFAULT_RIG_PRIOR_CENTER,
        ld.DEFAULT_RIG_PRIOR_SIGMA, 1.0,
    )
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # Tile spans x 1000-1255, y 600-855; bbox starts at (1100, 700).
    assert mask[0, 0] == 0.0        # (1000, 600) outside
    assert mask[150, 150] == 1.0    # (1150, 750) inside


def test_rig_prior_soft_floor_keeps_gaussian():
    mask = ld._rig_prior_for_tile(
        1900, 1300, 64, ld.DEFAULT_RIG_PRIOR_BBOX, ld.DEFAULT_RIG_PRIOR_CENTER,
        ld.DEFAULT_RIG_PRIOR_SIGMA, 0.5,
    )
    assert mask.max() <= 1.0
    assert mask.min() >= 0.5
    assert len(np.unique(mask)) > 2


def test_line_corridor_mask_matches_perpendicular_distance():
    # Horizontal line y = 100, unit-normalized as (0, 1, -100).
    mask = ld._line_mask_for_tile(0, 50, 128, (0.0, 1.0, -100.0), 25.0)
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # Row index r corresponds to y = 50 + r; inside when |y - 100| <= 25.
    assert mask[25, 0] == 1.0    # y = 75
    assert mask[75, 0] == 1.0    # y = 125
    assert mask[24, 0] == 0.0    # y = 74
    assert mask[76, 0] == 0.0    # y = 126


def test_degenerate_line_mask_is_all_ones():
    mask = ld._line_mask_for_tile(0, 0, 16, (0.0, 0.0, 0.0), 25.0)
    np.testing.assert_array_equal(mask, np.ones((16, 16), dtype=np.float32))


# --------------------------------------------------------------------------
# Sub-pixel refinement
# --------------------------------------------------------------------------


def test_subpixel_refine_recovers_known_offset():
    heat = np.zeros((9, 9), dtype=np.float32)
    # Parabola with its vertex a quarter-pixel right of the integer peak.
    for i in range(9):
        for j in range(9):
            heat[i, j] = -((j - 4.25) ** 2) - ((i - 3.75) ** 2)
    x, y = ld.subpixel_refine_peak(heat, 4, 4)
    assert x == pytest.approx(4.25, abs=1e-6)
    assert y == pytest.approx(3.75, abs=1e-6)


def test_subpixel_refine_rejects_edges_and_degenerate_fits():
    heat = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    # Edge peaks return the integer position untouched.
    assert ld.subpixel_refine_peak(heat, 0, 5) == (0.0, 5.0)
    assert ld.subpixel_refine_peak(heat, 15, 5) == (15.0, 5.0)
    assert ld.subpixel_refine_peak(heat, 5, 0) == (5.0, 0.0)
    assert ld.subpixel_refine_peak(heat, 5, 15) == (5.0, 15.0)
    # A flat neighborhood has a zero denominator on both axes.
    assert ld.subpixel_refine_peak(np.ones((8, 8), np.float32), 4, 4) == (4.0, 4.0)


def test_subpixel_refine_is_useless_on_saturated_probabilities():
    """Why refinement runs on logits, not sigmoid probabilities.

    Under bf16 autocast, sigmoid saturates near a confident peak and the
    neighborhood collapses to v_center ~= v_neighbor ~= 0.99805. The parabola
    then carries no information — this test documents that the degenerate
    input yields no shift, so a port that refines on probabilities silently
    loses all sub-pixel precision instead of failing loudly.
    """
    saturated = np.full((16, 16), 0.99805, dtype=np.float32)
    saturated[8, 8] = 1.0
    assert ld.subpixel_refine_peak(saturated, 8, 8) == (8.0, 8.0)

    # The same peak in logit space still refines.
    logits = np.full((16, 16), 6.24, dtype=np.float32)
    logits[8, 8] = 9.0
    logits[8, 9] = 7.0
    x, _ = ld.subpixel_refine_peak(logits, 8, 8)
    assert x != 8.0


# --------------------------------------------------------------------------
# Soft snap
# --------------------------------------------------------------------------


def test_soft_snap_alpha_is_capped_and_confidence_weighted():
    line = (0.0, 1.0, -100.0)  # y = 100
    # Uncertain prediction, confident line → maximum blend.
    x, y, alpha = ld.soft_snap_to_line(
        50.0, 120.0, line_abc=line, line_confidence=20.0, pred_confidence=0.0
    )
    assert alpha == pytest.approx(0.3)
    assert x == 50.0                       # movement is perpendicular only
    assert y == pytest.approx(120.0 + 0.3 * (100.0 - 120.0))

    # Confident prediction → the line barely moves it.
    _, _, alpha_conf = ld.soft_snap_to_line(
        50.0, 120.0, line_abc=line, line_confidence=20.0, pred_confidence=0.99
    )
    assert alpha_conf < 0.02


def test_soft_snap_is_noop_for_unconfident_line():
    line = (0.0, 1.0, -100.0)
    x, y, alpha = ld.soft_snap_to_line(
        50.0, 120.0, line_abc=line, line_confidence=0.0, pred_confidence=0.0
    )
    # sigmoid(0 - 5) ~= 0.0067, small but non-zero.
    assert alpha == pytest.approx(0.0066928, abs=1e-6)
    assert (x, y) != (50.0, 120.0)


def test_project_point_onto_degenerate_line_is_identity():
    assert ld._project_point_onto_line(3.0, 4.0, 0.0, 0.0, 0.0) == (3.0, 4.0)


# --------------------------------------------------------------------------
# Rectification and coordinate frame
# --------------------------------------------------------------------------


def test_rectify_is_near_identity_at_principal_point():
    k = np.array([[2800.0, 0, 2007.0], [0, 2800.0, 1508.0], [0, 0, 1]])
    dist = np.array([-0.28, 0.11, 0.0, 0.0, -0.02])
    x, y = ld.rectify_prediction(2007.0, 1508.0, k, dist)
    assert x == pytest.approx(2007.0, abs=1e-3)
    assert y == pytest.approx(1508.0, abs=1e-3)


def test_rectify_moves_off_axis_points():
    k = np.array([[2800.0, 0, 2007.0], [0, 2800.0, 1508.0], [0, 0, 1]])
    dist = np.array([-0.28, 0.11, 0.0, 0.0, -0.02])
    x, y = ld.rectify_prediction(500.0, 400.0, k, dist)
    assert (x, y) != (500.0, 400.0)


# --------------------------------------------------------------------------
# Prediction container
# --------------------------------------------------------------------------


def test_prediction_detection_flag():
    assert ld.LaserPrediction(1.0, 2.0, 0.9).is_detected
    assert not ld.LaserPrediction(None, None, 0.1).is_detected


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_as_bgr_array_rejects_non_bgr():
    with pytest.raises(ValueError, match=r"\[H, W, 3\] BGR"):
        ld._as_bgr_array(np.zeros((10, 10), dtype=np.uint16))


def test_as_bgr_array_picks_up_bayer_excess_from_image_object():
    class _Fake:
        data = np.zeros((4, 4, 3), dtype=np.uint16)
        bayer_excess = np.zeros((4, 4, 2), dtype=np.uint16)

    bgr, bayer = ld._as_bgr_array(_Fake())
    assert bgr.shape == (4, 4, 3)
    assert bayer is not None and bayer.shape == (4, 4, 2)


def test_as_bgr_array_accepts_plain_ndarray():
    bgr, bayer = ld._as_bgr_array(np.zeros((4, 4, 3), dtype=np.uint16))
    assert bgr.shape == (4, 4, 3)
    assert bayer is None


# --------------------------------------------------------------------------
# Bayer super-cell upsample
#
# This choice has already been reverted once upstream. It is load-bearing:
# the published checkpoint's bias offset was calibrated against `np.repeat`,
# so silently switching to bilinear reintroduces roughly the (-1.1, -2.1) px
# bias that offset exists to cancel. Verified bit-exact against all 25 frames
# of the ucsde4e/fishsense-laser-detector-validation oracle bundle.
# --------------------------------------------------------------------------


def test_upsample_defaults_to_repeat():
    half = np.arange(4, dtype=np.uint16).reshape(2, 2)
    np.testing.assert_array_equal(
        lri.upsample_super_cells(half), lri.upsample_super_cells(half, "repeat")
    )
    assert lri.LinearRawImage.VALID_UPSAMPLES == ("repeat", "bilinear")


def test_repeat_upsample_places_value_at_block_top_left():
    half = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    full = lri.upsample_super_cells(half, "repeat")
    assert full.shape == (4, 4)
    np.testing.assert_array_equal(
        full,
        np.array([[10, 10, 20, 20],
                  [10, 10, 20, 20],
                  [30, 30, 40, 40],
                  [30, 30, 40, 40]], dtype=np.uint16),
    )


def test_bilinear_upsample_differs_from_repeat_on_a_gradient():
    """The two modes disagree wherever the signal has gradient.

    A flat array is the one case where they agree, so testing on flat input
    would pass for either implementation and catch nothing.
    """
    half = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100
    rep = lri.upsample_super_cells(half, "repeat")
    bil = lri.upsample_super_cells(half, "bilinear")
    assert rep.shape == bil.shape
    assert np.abs(rep.astype(np.int64) - bil.astype(np.int64)).max() > 0

    flat = np.full((4, 4), 7, dtype=np.uint16)
    np.testing.assert_array_equal(
        lri.upsample_super_cells(flat, "repeat"),
        lri.upsample_super_cells(flat, "bilinear"),
    )


def test_upsample_rejects_unknown_mode():
    half = np.zeros((2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="mode must be one of"):
        lri.upsample_super_cells(half, "lanczos")


def test_linear_raw_image_rejects_unknown_upsample():
    with pytest.raises(ValueError, match="bayer_upsample must be one of"):
        lri.LinearRawImage(b"", bayer_upsample="lanczos")


# --------------------------------------------------------------------------
# Checkpoint identity resolution
#
# The bias offset is a per-checkpoint calibration keyed by the checkpoint's
# identity. Filename is a fragile identity — a run3 checkpoint saved under any
# other name (as a downstream audit hit) would otherwise lose its offset
# silently. Resolution falls back to content hash, and an unrecognized
# checkpoint must fail loudly rather than default to a zero offset.
# --------------------------------------------------------------------------


def test_canonical_name_matches_published_filename(tmp_path):
    p = tmp_path / "run3_epoch_021.pt"
    p.write_bytes(b"not actually a checkpoint")  # filename hit, contents unread
    assert ld._canonical_checkpoint_name(p) == "run3_epoch_021.pt"


def test_canonical_name_resolves_renamed_copy_by_content(tmp_path, monkeypatch):
    """A published checkpoint saved under a different name still resolves."""
    body = b"pretend this is the run3 checkpoint bytes"
    import hashlib

    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setitem(ld.CHECKPOINT_SHA256, digest, "run3_epoch_021.pt")
    p = tmp_path / "epoch_021.pt"  # the exact rename that bit the downstream audit
    p.write_bytes(body)
    assert ld._canonical_checkpoint_name(p) == "run3_epoch_021.pt"


def test_canonical_name_is_none_for_unknown_checkpoint(tmp_path):
    p = tmp_path / "some_other_model.pt"
    p.write_bytes(b"unrecognized")
    assert ld._canonical_checkpoint_name(p) is None


def test_published_checkpoint_hashes_map_to_canonical_names():
    # Every hashed checkpoint must name one that also has an encoder and offset.
    for name in ld.CHECKPOINT_SHA256.values():
        assert name in ld.CHECKPOINT_ENCODERS
        assert name in ld.CHECKPOINT_BIAS_OFFSETS


def test_from_checkpoint_raises_on_unknown_bias(tmp_path):
    """An unrecognized checkpoint with no explicit bias must raise, not run
    with a silent (0, 0) offset."""
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "mystery.pt"
    # cfg supplies the encoder so resolution reaches the bias check, which is
    # the behaviour under test; the state dict is never loaded (raise is first).
    torch.save(
        {"cfg": {"encoder_name": "resnet34", "in_channels": 6},
         "model_state_dict": {}},
        ckpt,
    )
    with pytest.raises(ValueError, match="no bias offset known"):
        ld.LaserDetector.from_checkpoint(ckpt, device="cpu")
