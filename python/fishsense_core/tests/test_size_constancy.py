"""Label-free size-constancy laser calibration: the pure port, pinned.

Ported from fishsense-services v0.48.0 `services/fishsense-services-processor/
tests/test_size_constancy.py` (imports aside, unchanged) and
`tests/test_shadow_fit.py` (re-expressed on this module: the shadow stage's
`fit_tv`/`beam` are this module's functions, see its docstring), with both
fixtures copied verbatim from that suite's `tests/fixtures/`. The last section
pins fishsense-services' own `fit_label_free` output, produced by running
v0.48.0's module on the golden fixture (bit-identical to this port).

Ported from cscw-fishsense2027@96a8da07
calibration_analysis/e1j_size_constancy/register.py (`pair_scale`, `score`'s
weighted solve) and slate_unknown.py (`frame`, `fit_tv`), with the beam built
as e2e_measurement/score.py `calibrations()` builds its label-free one; the
method is wuwnet-fishsense2026@5532b99 fishsense_wuwnet/laser.py
`close_with_apparent_size` (paper §4.3).

**Golden**: `fixtures/size_constancy_golden.json` holds, for the paper's ten
calibration sessions (Table 3), the pair equations cscw's own `pair_scale`
produced from its SIFT features, the dots, the dive line, the camera matrix
and the stored calibration's axis; and cscw's output (`labelfree_tv.csv`).
The port must reproduce that output -- the vanishing position to a
micro-pixel, and so Table 3: 9 of 10 sessions within 0.05 deg of the
known-size calibration, 10 of 10 within 0.15 deg.

The rest are synthetic, CPU-only: a beam the fit must recover, a texture
whose scale the registration must recover, and the refusals.
"""

# The test names are the documentation; cv2's members are built at import time,
# out of pylint's sight.
# pylint: disable=missing-function-docstring,no-member

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fishsense_core import size_constancy as sc

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "size_constancy_golden.json").read_text()
)["dives"]
#: cscw's nominal focal length for turning pixels into degrees (slate_unknown.F_PX).
F_PX = 2850.0


def _pairs(d):
    return [sc.PairScale(int(i), int(j), float(v), int(n)) for i, j, v, n in d["pairs"]]


def _stored_vanishing(d, u):
    """The stored (known-size) calibration's vanishing point, along the line."""
    k = np.array(d["camera_matrix"])
    ax = np.array(d["stored_axis"])
    v = np.array([k[0, 0] * ax[0] / ax[2] + k[0, 2], k[1, 1] * ax[1] / ax[2] + k[1, 2]])
    return float(v @ u)


# -- golden: cscw's labelfree_tv.csv, from cscw's own pair equations -----------------


# Session 114 is core's one departure from cscw's solve. Frame 7's pairs
# include one at log scale -37.18 (a factor of ~1e16), which drags its first
# solve so far that every one of its pairs fails the robust pass. cscw still
# counted it "connected" and fitted the size the gauge made up for it (its
# bootstrap SE was 4.41 px); core gives it no size. The other nine sessions are
# cscw's exactly.
_CSCW_DEPARTS = {"114"}


@pytest.mark.parametrize("mission", sorted(set(GOLDEN) - _CSCW_DEPARTS, key=int))
def test_reproduces_cscws_label_free_fit(mission):
    d = GOLDEN[mission]
    want = d["expected"]
    dots = np.array(d["dots"])

    sizes = sc.solve_log_sizes(len(dots), _pairs(d))
    u = sc.line_direction(d["line"], dots[sizes.connected])
    t = dots[sizes.connected] @ u
    tv = sc.fit_vanishing_position(t, np.exp(sizes.log_sizes[sizes.connected]))

    assert int(sizes.connected.sum()) == want["frames"]
    assert u == pytest.approx([want["ux"], want["uy"]], abs=1e-12)
    assert sizes.residual_sd == pytest.approx(want["pair_resid_sd"], rel=1e-6)
    assert np.ptp(sizes.log_sizes[sizes.connected]) == pytest.approx(
        np.log(want["size_ratio"]), rel=1e-9
    )
    assert tv == pytest.approx(want["tv_fit"], abs=1e-4)
    assert tv - _stored_vanishing(d, u) == pytest.approx(want["err_px"], abs=1e-4)


def test_session_114_drops_the_frame_cscw_gave_a_made_up_size():
    """Every pair through frame 7 fails the robust pass, so it has no size
    relative to the rest; cscw fitted one anyway. Without it the solve alone
    lands where cscw's fit only got after its trim dropped the frame."""
    d = GOLDEN["114"]
    dots = np.array(d["dots"])
    sizes = sc.solve_log_sizes(len(dots), _pairs(d))
    assert np.flatnonzero(~sizes.connected).tolist() == [7]
    assert np.isnan(sizes.log_sizes[7])
    u = sc.line_direction(d["line"], dots[sizes.connected])
    t = dots[sizes.connected] @ u
    tv = sc.fit_vanishing_position(t, np.exp(sizes.log_sizes[sizes.connected]))
    assert tv == pytest.approx(-2273.7843259011893, abs=_TV_ABS)
    assert sizes.residual_sd == pytest.approx(0.009128963751628413, rel=1e-6)


def test_bootstrap_reproduces_cscws_standard_errors():
    """cscw draws every session's bootstrap from one generator, in Table 3's
    order; so must this, to reproduce its SEs. Session 114 is last, so the
    draws before it are unchanged; its own SE falls from cscw's 4.41 px to
    0.73 px once frame 7's made-up size is gone."""
    rng = np.random.default_rng(0)
    for mission in sorted(GOLDEN, key=int):
        d = GOLDEN[mission]
        dots = np.array(d["dots"])
        sizes = sc.solve_log_sizes(len(dots), _pairs(d))
        xy = dots[sizes.connected]
        t = xy @ sc.line_direction(d["line"], xy)
        se = sc.bootstrap_se(t, np.exp(sizes.log_sizes[sizes.connected]), rng=rng)
        want = 0.7291465961104082 if mission in _CSCW_DEPARTS else d["expected"]["se_px"]
        assert se == pytest.approx(want, rel=1e-6), mission


def test_table_3_all_ten_within_0_05_deg():
    """cscw: 9 of 10 within 0.05 deg, all within 0.15. Session 114's frame 7
    (0.129 deg in cscw) is now out of the solve; 0.0485 deg."""
    errors = []
    for d in GOLDEN.values():
        fit = sc.fit_label_free(
            dots=np.array(d["dots"]),
            pairs=_pairs(d),
            camera_matrix=d["camera_matrix"],
            line=d["line"],
            min_frames=1,
            min_size_ratio=1.0,
        )
        u = np.array(fit.line_direction)
        errors.append(np.degrees((fit.vanishing_px - _stored_vanishing(d, u)) / F_PX))
    errors = np.abs(errors)
    assert (errors <= 0.05).all()
    assert np.median(errors) == pytest.approx(0.02, abs=0.01)


def test_the_beam_is_built_as_cscws_label_free_calibration():
    """score.py `calibrations()`: v = tv u + offset n; the axis is K^-1 [v, 1];
    the origin is |O| along the locus direction in normalised coordinates."""
    d = GOLDEN["71"]
    k = np.array(d["camera_matrix"])
    want = d["expected"]
    u = np.array([want["ux"], want["uy"]])
    n = np.array([-u[1], u[0]])
    v = want["tv_fit"] * u + want["line_offset"] * n
    axis = np.linalg.solve(k, [v[0], v[1], 1.0])
    o = np.array([u[0] / k[0, 0], u[1] / k[1, 1]])
    origin = np.append(0.104 * o / np.linalg.norm(o), 0.0)

    got_origin, got_axis = sc.beam_from_vanishing(
        want["tv_fit"], u, want["line_offset"], k
    )

    assert got_axis == pytest.approx(axis / np.linalg.norm(axis), abs=1e-12)
    assert got_origin == pytest.approx(origin, abs=1e-12)


# -- synthetic: a beam the fit must recover ----------------------------------------

K = np.array([[2850.0, 0.0, 2000.0], [0.0, 2850.0, 1500.0], [0.0, 0.0, 1.0]])
TRUE_O = np.append(0.104 * np.array([-0.29, -0.957]) / np.hypot(0.29, 0.957), 0.0)
TRUE_D = np.array([0.012, 0.035, 1.0])


def _dot_at(z):
    p = TRUE_O + (z - TRUE_O[2]) / TRUE_D[2] * TRUE_D
    q = K @ (p / p[2])
    return q[:2]


def _beam_angle(a, b):
    a, b = np.asarray(a) / np.linalg.norm(a), np.asarray(b) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def _exact_pairs(z):
    """Exact pairwise size ratios of an object 1/z in apparent size."""
    logs = -np.log(z)
    n = len(z)
    return [sc.PairScale(i, j, logs[j] - logs[i], 20)
            for i in range(n) for j in range(i + 1, n)]  # fmt: skip


def test_recovers_a_known_beam_from_apparent_size_alone():
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])

    fit = sc.fit_label_free(
        dots=dots, pairs=_exact_pairs(z), camera_matrix=K, line=None
    )

    assert fit.outcome == "accepted", fit.refusal_reason
    assert _beam_angle(fit.laser_axis, TRUE_D) < 0.01
    assert np.linalg.norm(fit.laser_position) == pytest.approx(0.104)
    assert fit.laser_position[2] == 0.0
    assert fit.size_ratio == pytest.approx(3.0)
    assert fit.frames_used == 12


def test_a_wrong_object_size_does_not_move_the_beam():
    """Size constancy: the object's physical size cancels."""
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    u = sc.line_direction(None, dots)
    t = dots @ u
    small = sc.fit_vanishing_position(t, 1.0 / z)
    large = sc.fit_vanishing_position(t, 7.0 / z)
    assert small == pytest.approx(large, abs=1e-3)
    assert (
        _beam_angle(
            sc.beam_from_vanishing(small, u, float(np.median(dots @ [-u[1], u[0]])), K)[
                1
            ],
            TRUE_D,
        )
        < 0.01
    )


def test_a_bad_pair_is_dropped_by_the_loop_check():
    """Pairs carry ~1 % scale noise (register's floor); one matched another
    layer and is 35 % off. The loop of the other pairs outvotes it."""
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    rng = np.random.default_rng(3)
    clean = [sc.PairScale(p.i, p.j, p.log_scale + rng.normal(0, 0.01), p.near)
             for p in _exact_pairs(z)]  # fmt: skip
    bad = list(clean)
    bad[5] = sc.PairScale(bad[5].i, bad[5].j, bad[5].log_scale + 0.3, bad[5].near)

    want = sc.fit_label_free(dots=dots, pairs=clean, camera_matrix=K)
    got = sc.fit_label_free(dots=dots, pairs=bad, camera_matrix=K)

    assert got.vanishing_px == pytest.approx(want.vanishing_px, abs=0.5)
    assert _beam_angle(got.laser_axis, TRUE_D) < 0.05


def _noisy_pairs(z, sd=0.01, seed=3):
    rng = np.random.default_rng(seed)
    return [sc.PairScale(p.i, p.j, p.log_scale + rng.normal(0, sd), p.near)
            for p in _exact_pairs(z)]  # fmt: skip


def test_a_frame_whose_size_disagrees_with_its_range_is_dropped():
    """A dot on the line but at the wrong range: far down the line, on a slate
    as near as the nearest frames (mission 80's one such frame moved its
    vanishing point 360 px, 2026-10-19). It is dropped and the fit refitted."""
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    want = sc.fit_label_free(dots=dots, pairs=_noisy_pairs(z), camera_matrix=K)

    sizes = np.append(z, 1.2)  # its slate's size: as at 1.2 m
    pairs = _noisy_pairs(sizes)
    got = sc.fit_label_free(
        dots=np.vstack([dots, _dot_at(8.0)]), pairs=pairs, camera_matrix=K
    )

    assert got.outcome == "accepted", got.refusal_reason
    assert 12 not in got.used and got.frames_used == 12
    assert got.vanishing_px == pytest.approx(want.vanishing_px, abs=0.5)
    assert _beam_angle(got.laser_axis, TRUE_D) < 0.05


def _with_a_bad_frame(z_dot, z_size, n=12):
    """`n` good frames at 1-3 m and one more whose dot is at `z_dot` on a
    slate sized as at `z_size`; the pairs, and the same pairs without it."""
    z = np.linspace(1.0, 3.0, n)
    dots = np.vstack([[_dot_at(zi) for zi in z], [_dot_at(z_dot)]])
    pairs = _noisy_pairs(np.append(z, z_size))
    without = [p for p in pairs if n not in (p.i, p.j)]
    return dots, pairs, without


def test_a_trimmed_frames_pairs_no_longer_shape_the_sizes():
    """The sizes are re-solved without the dropped frame's pairs, so the fit
    is the fit of the frames kept, exactly (review of #136)."""
    dots, pairs, without = _with_a_bad_frame(8.0, 1.2)

    got = sc.fit_label_free(dots=dots, pairs=pairs, camera_matrix=K)
    want = sc.fit_label_free(dots=dots[:12], pairs=without, camera_matrix=K)

    assert got.used == want.used == list(range(12))
    assert got.vanishing_px == pytest.approx(want.vanishing_px, abs=1e-6)


@pytest.mark.parametrize(
    "z_dot, z_size",
    [(8.0, 0.3), (5.0, 10.0), (0.7, 5.0)],
    ids=["pins-the-fit", "sets-the-bracket", "nearest-far-slate"],
)
def test_one_wrong_range_frame_costs_only_itself(z_dot, z_size):
    """Judged by a fit it took no part in, the wrong frame is the one dropped:
    it cannot pin the fit to the bracket it sets, or push good frames out
    first (review of #136: it was refused `unconstrained` or
    `too_few_frames`)."""
    dots, pairs, without = _with_a_bad_frame(z_dot, z_size)

    got = sc.fit_label_free(dots=dots, pairs=pairs, camera_matrix=K)
    want = sc.fit_label_free(dots=dots[:12], pairs=without, camera_matrix=K)

    assert got.outcome == "accepted", got.refusal_reason
    assert got.used == list(range(12))
    assert got.vanishing_px == pytest.approx(want.vanishing_px, abs=1e-6)
    assert _beam_angle(got.laser_axis, TRUE_D) < 0.05


def test_frames_within_the_noise_are_all_kept():
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    fit = sc.fit_label_free(dots=dots, pairs=_noisy_pairs(z, sd=0.015), camera_matrix=K)
    assert fit.frames_used == 12


@pytest.mark.parametrize("mission", sorted(set(GOLDEN) - {"114"}, key=int))
def test_the_paper_sessions_lose_no_frame_but_114s_one(mission):
    """cscw's frames (human dots on labelled slate frames) are consistent, so
    nine of the ten sessions are fitted exactly as cscw fitted them."""
    d = GOLDEN[mission]
    fit = sc.fit_label_free(
        dots=np.array(d["dots"]), pairs=_pairs(d), camera_matrix=d["camera_matrix"],
        line=d["line"], min_frames=1, min_size_ratio=1.0,
    )  # fmt: skip
    assert fit.frames_used == d["expected"]["frames"]
    assert fit.vanishing_px == pytest.approx(d["expected"]["tv_fit"], abs=1e-4)


def test_session_114_loses_one_frame_and_moves_toward_the_known_size_fit():
    d = GOLDEN["114"]
    fit = sc.fit_label_free(
        dots=np.array(d["dots"]), pairs=_pairs(d), camera_matrix=d["camera_matrix"],
        line=d["line"], min_frames=1, min_size_ratio=1.0,
    )  # fmt: skip
    u = np.array(fit.line_direction)
    err = abs(fit.vanishing_px - _stored_vanishing(d, u))
    assert fit.frames_used == d["expected"]["frames"] - 1
    assert err < abs(d["expected"]["err_px"])


def test_refuses_when_the_trim_leaves_too_few_frames():
    z = np.linspace(1.0, 3.0, 7)
    dots = np.vstack([[_dot_at(zi) for zi in z], [_dot_at(8.0)]])
    fit = sc.fit_label_free(
        dots=dots, pairs=_noisy_pairs(np.append(z, 1.2)), camera_matrix=K
    )
    assert (fit.outcome, fit.refusal_reason) == ("refused", "too_few_frames")


# -- the refusals ----------------------------------------------------------------


def test_refuses_too_few_frames():
    z = np.linspace(1.0, 3.0, 5)
    dots = np.array([_dot_at(zi) for zi in z])
    fit = sc.fit_label_free(dots=dots, pairs=_exact_pairs(z), camera_matrix=K)
    assert (fit.outcome, fit.refusal_reason) == ("refused", "too_few_frames")
    assert fit.laser_axis is None and fit.laser_position is None


def test_refuses_too_little_range_spread():
    """Paper §4.3: the method needs about 1.5x in distance; sessions under
    1.12x are unconstrained."""
    z = np.linspace(2.0, 2.6, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    fit = sc.fit_label_free(dots=dots, pairs=_exact_pairs(z), camera_matrix=K)
    assert (fit.outcome, fit.refusal_reason) == ("refused", "too_little_range_spread")
    assert fit.size_ratio == pytest.approx(1.3)


def test_refuses_frames_no_pair_connects():
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    fit = sc.fit_label_free(dots=dots, pairs=[], camera_matrix=K)
    assert (fit.outcome, fit.refusal_reason) == ("refused", "too_few_matched_pairs")


def _two_clusters(z, split):
    """Exact pairs within frames [0, split) and within [split, n), none across."""
    logs, n = -np.log(z), len(z)
    return [
        sc.PairScale(i, j, logs[j] - logs[i], 20)
        for i in range(n) for j in range(i + 1, n)
        if (i < split) == (j < split)
    ]  # fmt: skip


def test_two_unlinked_clusters_are_not_sized_against_each_other():
    """Two passes of the object that never cross-match: every frame has pairs,
    but the gauge alone would set one cluster's sizes against the other's.
    Only the larger cluster is sized."""
    z = np.linspace(1.0, 3.0, 12)
    sizes = sc.solve_log_sizes(12, _two_clusters(z, 5))
    assert sizes.connected.tolist() == [False] * 5 + [True] * 7
    assert np.isnan(sizes.log_sizes[:5]).all()
    # Within the cluster the ratios are the true ones.
    got = sizes.log_sizes[5:] - sizes.log_sizes[5:].mean()
    want = -np.log(z[5:]) - (-np.log(z[5:])).mean()
    assert got == pytest.approx(want, abs=1e-12)


def test_a_fit_over_two_unlinked_clusters_uses_one():
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    fit = sc.fit_label_free(dots=dots, pairs=_two_clusters(z, 4), camera_matrix=K)
    assert fit.used == list(range(4, 12))


def test_a_frame_whose_only_pairs_fail_the_robust_pass_has_no_size():
    """Session 114's case: frame 7's two pairs contradict each other (one a
    wrong-layer match ~e^5 off), so its first solve splits the difference and
    both fail the robust pass. With no pair left it has no size, not one the
    gauge made up. (A lone pair could not be judged: with no loop, nothing
    contradicts it.)"""
    z = np.linspace(1.0, 3.0, 12)
    logs = -np.log(z)
    pairs = [p for p in _exact_pairs(z) if 7 not in (p.i, p.j)]
    pairs.append(sc.PairScale(3, 7, logs[7] - logs[3], 20))
    pairs.append(sc.PairScale(4, 7, logs[7] - logs[4] + 5.0, 20))
    sizes = sc.solve_log_sizes(12, pairs)
    assert not sizes.connected[7] and np.isnan(sizes.log_sizes[7])
    # The single robust pass (the paper's) also loses frames 3 and 4 here: the
    # outlier's error spreads into them in the first solve. Every frame that
    # keeps a size has the true one; none has a made-up one.
    sized = sizes.connected
    got = sizes.log_sizes[sized] - sizes.log_sizes[sized].mean()
    want = logs[sized] - logs[sized].mean()
    assert got == pytest.approx(want, abs=1e-12)
    assert np.isnan(sizes.log_sizes[~sized]).all()


@pytest.mark.parametrize(
    "bad",
    [
        sc.PairScale(0, 1, 0.1, 0),
        sc.PairScale(0, 1, float("nan"), 5),
        sc.PairScale(0, 1, float("inf"), 5),
        sc.PairScale(0, 12, 0.1, 5),
        sc.PairScale(3, 3, 0.0, 5),
    ],
    ids=["near-zero", "nan-scale", "inf-scale", "out-of-range", "self-pair"],
)
def test_a_malformed_pair_is_an_error_not_silent_zeros(bad):
    """near=0 used to make every residual NaN, reject every pair and return
    all-zero sizes with no error."""
    pairs = _exact_pairs(np.linspace(1.0, 3.0, 12)) + [bad]
    with pytest.raises(ValueError):
        sc.solve_log_sizes(12, pairs)


def test_with_no_line_the_direction_follows_the_trim():
    """A detector dot far off the true line, on a frame whose size puts it at
    the wrong range: the trim drops it, and the direction is refitted from the
    frames that remain, so it no longer tilts the line."""
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    dots[6] = dots[6] + np.array([120.0, 40.0])  # off the beam's line
    sizes = 1.0 / z
    sizes[6] *= 1.6  # and at the wrong range for its place
    fit = sc.fit_label_free(dots=dots, pairs=_exact_pairs(1.0 / sizes), camera_matrix=K)
    assert 6 not in fit.used
    clean = np.delete(dots, 6, axis=0)
    want = sc.line_direction(None, clean)
    assert fit.line_direction == pytest.approx(want.tolist(), abs=1e-12)


def test_a_refusal_reports_the_offset_it_was_given():
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    fit = sc.fit_label_free(dots=dots, pairs=[], camera_matrix=K, o_mag_m=0.12)
    assert (fit.outcome, fit.o_mag_m) == ("refused", 0.12)


def test_the_result_has_no_services_only_fields():
    assert "scale_sd" not in {f.name for f in sc.LabelFreeFit.__dataclass_fields__.values()}


def test_refuses_a_fit_pinned_at_the_search_bound():
    """Unconstrained: the optimum is the bracket's edge, not a minimum."""
    z = np.linspace(1.0, 3.0, 12)
    dots = np.array([_dot_at(zi) for zi in z])
    # Sizes that grow with range: no vanishing point explains them.
    pairs = _exact_pairs(1.0 / z)
    fit = sc.fit_label_free(dots=dots, pairs=pairs, camera_matrix=K)
    assert (fit.outcome, fit.refusal_reason) == ("refused", "unconstrained")


# -- registration: the scale between two frames ------------------------------------


def _texture(seed=0, size=(1600, 2200)):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(size[0] // 8, size[1] // 8), dtype=np.uint8)
    img = cv2.resize(noise, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_registration_recovers_the_scale_about_the_dot():
    dot = np.array([1100.0, 800.0])
    a = _texture()
    scale = 1.25
    m = cv2.getRotationMatrix2D((float(dot[0]), float(dot[1])), 0.0, scale)
    b = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]))

    fa = sc.extract_features(a, dot)
    fb = sc.extract_features(b, dot)
    got = sc.pair_scale(fa, fb, dot)

    assert got is not None
    log_scale, near = got
    assert log_scale == pytest.approx(np.log(scale), abs=0.01)
    assert near >= sc.MIN_PAIR_NEAR


def test_registration_refuses_a_similarity_that_collapses_the_object():
    """Matches that all land on one spot in `b` fit a similarity of scale ~0:
    a 'size ratio' no two frames of a dive have. Under OpenCV 5 cscw's mission
    63 features produce two such pairs (log scale -41 and -42), which connect
    two more frames and move the vanishing position 3900 px."""
    rng = np.random.default_rng(0)
    dot = np.array([1000.0, 800.0])
    xy_a = (dot + rng.uniform(-60, 60, (60, 2))).astype(np.float32)
    xy_b = (np.array([1500.0, 500.0]) + rng.normal(0, 0.5, (60, 2))).astype(np.float32)
    desc = rng.normal(size=(60, 128)).astype(np.float32)

    got = sc.pair_scale(
        sc.FrameFeatures(xy=xy_a, desc=desc), sc.FrameFeatures(xy=xy_b, desc=desc), dot
    )

    assert got is None


def test_registration_needs_features():
    empty = sc.FrameFeatures(xy=np.zeros((0, 2), np.float32), desc=None)
    assert sc.pair_scale(empty, empty, np.array([0.0, 0.0])) is None


def _cscw_features() -> Path | None:
    """cscw's SIFT features (65 MB, not vendored), from a sibling checkout."""
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent
            / "cscw-fishsense2027/calibration_analysis/e1j_size_constancy/features"
        )
        if candidate.is_dir():
            return candidate
    return None


CSCW_FEATURES = _cscw_features()


@pytest.mark.skipif(CSCW_FEATURES is None, reason="needs ../cscw-fishsense2027")
@pytest.mark.parametrize("mission", ["63", "94"])
def test_registration_reproduces_cscws_pairs_from_its_features(mission):
    """The registration itself, against the golden pairs, when the sibling
    checkout is present. cscw ran OpenCV 4.14; under 5 the RANSAC refinement
    moves log scales by up to 1.2e-7 (the vanishing position by 3e-6 px), and
    near-zero log scales make that a large relative error, so the tolerance
    is absolute. Pairs carry ~1 %."""
    d = GOLDEN[mission]
    features = []
    for image_id in d["image_ids"]:
        z = np.load(CSCW_FEATURES / f"{image_id}.npz", allow_pickle=True)
        features.append(sc.FrameFeatures(xy=z["xy"], desc=z["desc"]))

    got = sc.pair_scales(features, np.array(d["dots"]))

    assert [(p.i, p.j, p.near) for p in got] == [(i, j, n) for i, j, _, n in d["pairs"]]
    assert [p.log_scale for p in got] == pytest.approx(
        [v for _, _, v, _ in d["pairs"]], rel=0, abs=1e-6
    )


# -- the shadow stage's copy: cscw's size_constancy_sam.py pool, through this module --
#
# fishsense-services' `shadow_calibration/fit.py` `port_fit` is `fit_vanishing_position`,
# `beam_from_vanishing` and `bootstrap_se` with 500 draws from
# ``default_rng([session, variant])``; its error metrics are computed here as it
# computes them. The published rows have 5 decimals: each must match to the last
# printed digit (services tests/test_shadow_fit.py).

POOL = json.loads((Path(__file__).parent / "fixtures/shadow_pool.json").read_text())
SESSIONS = {s["session"]: s for s in POOL["sessions"]}
#: The published CSV prints 5 decimals.
PRINTED = 5.000001e-6
#: size_constancy_sam.N_BOOT, and the script's variants, whose index seeds each
#: one's bootstrap (services shadow_calibration/fit.py `N_BOOT`, `VARIANT_INDEX`).
N_BOOT = 500
VARIANT_INDEX = {"sam3": 0, "corners": 1, "sam3_all": 2, "sam3_predline": 3}


#: Side by side in one environment the two ports agree bit for bit; across
#: numpy/BLAS builds Brent's bounded search (xatol 1e-5) lands a few 1e-8 px
#: apart, which moves a bootstrap SE by ~4e-8 relative (CI, ubuntu numpy). The
#: services goldens use the same scale (SE rel=1e-6).
_TV_ABS = 1e-6
_SE_REL = 1e-6


def _shadow_fit(session, variant, line):  # pylint: disable=too-many-locals
    """services `port_fit`, on this module."""
    sized = [f for f in session["frames"] if f["size"] is not None]
    xy = np.array([[f["dot_x"], f["dot_y"]] for f in sized])
    s = np.array([f["size"] for f in sized])
    u = np.array(session[line]["u"])
    k = np.array(session["camera_matrix"])
    t = xy @ u
    tv = sc.fit_vanishing_position(t, s)
    rng = np.random.default_rng([session["session"], VARIANT_INDEX[variant]])
    se_px = sc.bootstrap_se(t, s, rng=rng, draws=N_BOOT)
    _, axis = sc.beam_from_vanishing(tv, u, session[line]["offset"], k)
    ref = np.array(session["stored_laser_axis"], float)
    ref = ref / np.linalg.norm(ref)
    vs = k @ ref
    err_px = tv - float((vs[:2] / vs[2]) @ u)
    return {
        "tv": tv,
        "frames": len(t),
        "size_spread": float(s.max() / s.min()),
        "se_px": se_px,
        "se_deg": float(np.degrees(se_px / F_PX)),
        "err_deg": float(np.degrees(err_px / F_PX)),
        "angle_deg": float(np.degrees(np.arccos(np.clip(axis @ ref, -1, 1)))),
        "axis": axis,
    }


def test_the_pool_constants_are_the_scripts():
    assert (F_PX, N_BOOT) == (POOL["f_px"], POOL["n_boot"]) == (2850.0, 500)


@pytest.mark.parametrize("session", sorted(SESSIONS))
@pytest.mark.parametrize("variant, line", [("sam3_all", "line"),
                                           ("sam3_predline", "detector_line")])  # fmt: skip
def test_each_pool_session_reproduces_its_published_row(session, variant, line):
    s = SESSIONS[session]
    published = s["published"][variant]
    fit = _shadow_fit(s, variant, line)

    assert fit["frames"] == published["frames"]
    assert fit["size_spread"] == pytest.approx(published["size_spread"], abs=PRINTED)
    assert fit["err_deg"] == pytest.approx(published["err_deg"], abs=PRINTED)
    assert fit["angle_deg"] == pytest.approx(published["angle_deg"], abs=PRINTED)
    assert fit["se_deg"] == pytest.approx(published["se_deg"], abs=PRINTED)


def test_the_pool_medians_are_the_papers():
    """Median |err_deg| 0.0232, median angle 0.029, 10/10 within 0.05; the
    detector line's median angle 0.047."""
    fits = [_shadow_fit(SESSIONS[k], "sam3_all", "line") for k in sorted(SESSIONS)]
    errs = [abs(f["err_deg"]) for f in fits]
    assert np.median(errs) == pytest.approx(0.02322, abs=1e-5)
    assert np.median([f["angle_deg"] for f in fits]) == pytest.approx(0.029, abs=5e-4)
    assert sum(e <= 0.05 for e in errs) == 10
    pred = [
        _shadow_fit(SESSIONS[k], "sam3_predline", "detector_line")
        for k in sorted(SESSIONS)
    ]
    assert np.median([f["angle_deg"] for f in pred]) == pytest.approx(0.047, abs=5e-4)


def test_fit_vanishing_position_is_the_scripts_bounded_search():
    """slate_unknown.fit_tv: Brent on [t_min - 4005, t_min - 5]. On exact
    sizes ``s = c (t - tv)`` it recovers tv."""
    t = np.linspace(200.0, 900.0, 12)
    for tv in (-500.0, -2000.0, -3000.0):
        assert sc.fit_vanishing_position(t, 0.07 * (t - tv)) == pytest.approx(
            tv, abs=0.05
        )


@pytest.mark.parametrize(
    "session, variant, tv, se_px",
    [
        (62, "sam3_all", -2053.4441329241617, 2.324314018459537),
        (62, "sam3_predline", -2059.7819344063414, 2.198782938546873),
        (63, "sam3_all", -2099.765314615446, 1.5104688964348734),
        (63, "sam3_predline", -2116.6266969673193, 1.4161367408977719),
    ],
)
def test_pool_fits_match_the_services_shadow_port(session, variant, tv, se_px):
    """fishsense-services v0.48.0 `port_fit` output, bit-identical to this
    module's composition when run side by side."""
    line = "line" if variant == "sam3_all" else "detector_line"
    fit = _shadow_fit(SESSIONS[session], variant, line)
    assert fit["tv"] == pytest.approx(tv, abs=_TV_ABS)
    assert fit["se_px"] == pytest.approx(se_px, rel=_SE_REL)


# -- parity: fishsense-services v0.48.0's own fit_label_free, on the golden fixture --


@pytest.mark.parametrize(
    "mission, tv, se_px, frames, axis, origin",
    [
        ("63", -2100.751418909571, 2.849386283702527, 12,
         [0.0006830644359840647, 0.02781982942565242, 0.999612720264055],
         [-0.030925636596909285, -0.0992955437120715, 0.0]),
        ("71", -2157.674039712723, 0.9615177815109391, 32,
         [0.031876685543251945, 0.04736781840562558, 0.9983687528655272],
         [-0.03206242143204709, -0.09893432736878441, 0.0]),
        ("80", -2075.52982343564, 1.1864945722318934, 31,
         [0.021941865426729087, 0.027812417254195643, 0.9993723149998073],
         [-0.029640994130566624, -0.09968656613080679, 0.0]),
        # services: tv -2273.5939549207674, se 0.8149804544450181, axis
        # [0.02113307546381264, 0.03277202789457865, 0.9992394044017269]. Both
        # drop frame 7; core's sizes are solved with its bad pairs already out.
        ("114", -2273.7843259011893, 0.8122183526337636, 28,
         [0.021162177941014645, 0.032831945741439605, 0.9992368215611472],
         [-0.045454413480002474, -0.09354088034217428, 0.0]),
    ],
)  # fmt: skip
def test_matches_the_services_fit(mission, tv, se_px, frames, axis, origin):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    d = GOLDEN[mission]
    fit = sc.fit_label_free(
        dots=np.array(d["dots"]), pairs=_pairs(d), camera_matrix=d["camera_matrix"],
        line=d["line"], min_frames=1, min_size_ratio=1.0,
    )  # fmt: skip
    assert fit.outcome == "accepted"
    assert fit.frames_used == frames
    assert fit.vanishing_px == pytest.approx(tv, abs=_TV_ABS)
    assert fit.se_px == pytest.approx(se_px, rel=_SE_REL)
    assert fit.laser_axis == pytest.approx(axis, abs=1e-9)
    assert fit.laser_position == pytest.approx(origin, abs=1e-12)
