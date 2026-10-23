"""The head/tail stage: crop at the dot, segment, gate, keypoint, lift.

Ported from fishsense-services' processor (``tests/test_headtail_geometry.py``,
``test_headtail_predict.py``, ``test_headtail_sam3_adapter.py``), themselves
ported from fishsense-lite@77e8f8e5. Names, bodies and reasons are theirs;
what changed is the seam: the kernel takes a decoded frame and returns a
:class:`FishAtDot` with a dot *index*, where the service took JPEG bytes and
returned its own row with a laser label id.

Everything but the SAM 3.1 adapter's autocast tests runs without torch or a
GPU (the module lives under ``fishsense_core.fish``, so ``_native`` is always
present), which is the point: these are the parts that fail *plausibly*. A
bad crop origin displaces every keypoint by a constant and still looks like a
fish.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fishsense_core.fish import fish_at_dot as sut
from fishsense_core.fish.fish_at_dot import (
    CROP_HEIGHT,
    CROP_WIDTH,
    FishialSegmenter,
    Sam3Segmenter,
    crop_origin,
    cut_by_window,
    find_fish_at_dot,
    lift_point,
    mask_at_point,
    mask_box,
    silhouette_ratio,
    to_numpy,
)

FRAME_W, FRAME_H = 4014, 3016
CROP_W, CROP_H = CROP_WIDTH, CROP_HEIGHT


def test_the_window_is_productions():
    assert (CROP_W, CROP_H) == (1800, 1350)


# ── geometry ────────────────────────────────────────────────────────────────


def test_crop_is_centred_on_the_laser_when_there_is_room():
    ox, oy = crop_origin(2000, 1500, FRAME_W, FRAME_H, CROP_W, CROP_H)
    assert (ox, oy) == (2000 - CROP_W // 2, 1500 - CROP_H // 2)


@pytest.mark.parametrize(
    "lx,ly",
    [
        (0, 0),
        (10, 10),
        (FRAME_W - 1, FRAME_H - 1),
        (FRAME_W - 5, 20),
        (20, FRAME_H - 5),
    ],
)
def test_crop_stays_inside_the_frame_at_every_edge_and_corner(lx, ly):
    """Clamped, never padded: padding shows the model a black band and
    truncating changes its aspect, both on exactly the edge frames."""
    ox, oy = crop_origin(lx, ly, FRAME_W, FRAME_H, CROP_W, CROP_H)
    assert 0 <= ox <= FRAME_W - CROP_W
    assert 0 <= oy <= FRAME_H - CROP_H


def test_crop_origin_handles_a_window_larger_than_the_frame():
    """Degenerate but must not produce a negative origin."""
    ox, oy = crop_origin(10, 10, 800, 600, 1800, 1350)
    assert (ox, oy) == (0, 0)


@pytest.mark.parametrize(
    "dot,origin",
    [
        ((2071.0137, 1367.2976), (1171, 692)),
        ((2070.8269, 1367.0896), (1170, 692)),
        ((2073.2759, 1366.2087), (1173, 691)),
    ],
    ids=["3339-human", "3339-auto", "3340-human"],
)
def test_crop_origin_reproduces_productions_windows(dot, origin):
    """Three rows of ``results/sam_boxes.jsonl`` in
    UCSD-E4E/2026-10-22_headtail_keypoints: production's predictor on
    4014x3016 test frames. The default window must land where it did."""
    assert crop_origin(dot[0], dot[1], FRAME_W, FRAME_H) == origin


def test_lift_undoes_the_crop():
    """The one bug in this stage that looks entirely plausible in the output:
    every prediction displaced by the crop origin."""
    assert lift_point((10.0, 20.0), 1200, 900) == (1210.0, 920.0)


def test_lift_round_trips_with_crop_origin():
    lx, ly = 2500.0, 1800.0
    ox, oy = crop_origin(lx, ly, FRAME_W, FRAME_H, CROP_W, CROP_H)
    in_crop = (lx - ox, ly - oy)
    assert lift_point(in_crop, ox, oy) == (lx, ly)


def test_mask_at_point_picks_the_mask_covering_the_dot():
    """The gate: the laser says which fish, not "the biggest one"."""
    a = np.zeros((100, 100), dtype=np.uint8)
    a[0:10, 0:10] = 1
    b = np.zeros((100, 100), dtype=np.uint8)
    b[50:60, 50:60] = 1
    assert mask_at_point([a, b], [(55, 55)]) is b


def test_mask_at_point_returns_none_when_the_dot_is_on_background():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[0:10, 0:10] = 1
    assert mask_at_point([a], [(80, 80)]) is None


def test_mask_at_point_tries_every_laser_point():
    """461 prod images carry two valid laser labels; first hit wins."""
    a = np.zeros((100, 100), dtype=np.uint8)
    a[50:60, 50:60] = 1
    assert mask_at_point([a], [(5, 5), (55, 55)]) is a


def test_mask_at_point_ignores_out_of_bounds_points():
    a = np.zeros((100, 100), dtype=np.uint8)
    a[50:60, 50:60] = 1
    assert mask_at_point([a], [(500, 500)]) is None


def test_silhouette_ratio_is_area_over_length_squared():
    assert silhouette_ratio(2500, 100.0) == pytest.approx(0.25)


def test_silhouette_ratio_is_none_for_a_degenerate_length():
    """A zero-length prediction has no meaningful shape; None rather than a
    division error or a misleading infinity."""
    assert silhouette_ratio(2500, 0.0) is None


def test_mask_box_is_half_open_and_lifted():
    m = np.zeros((20, 30), dtype=np.uint8)
    m[3:8, 10:14] = 1
    assert mask_box(m, 100, 200) == (110, 203, 114, 208)


def test_mask_box_of_an_empty_mask_is_none():
    assert mask_box(np.zeros((4, 4), dtype=np.uint8), 0, 0) is None


class TestCutByWindow:
    """A mask touching an edge of the window that is not the frame's own was
    cut by the window; one touching an edge the window was clamped to was
    cut by the photo, which no bigger window can see past."""

    def _mask(self, **edges):
        m = np.zeros((CROP_H, CROP_W), dtype=np.uint8)
        m[600:700, 800:1000] = 1
        if edges.get("left"):
            m[650, 0] = 1
        if edges.get("right"):
            m[650, -1] = 1
        if edges.get("top"):
            m[0, 900] = 1
        if edges.get("bottom"):
            m[-1, 900] = 1
        return m

    def test_an_interior_mask_is_not_cut(self):
        assert cut_by_window(self._mask(), 1100, 800, FRAME_W, FRAME_H) is False

    @pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
    def test_touching_an_interior_window_edge_is_cut(self, edge):
        assert cut_by_window(self._mask(**{edge: True}), 1100, 800, FRAME_W, FRAME_H) is True

    def test_touching_the_frames_own_edge_is_not_cut(self):
        # Window clamped to the top-left corner: its left and top edges are
        # the photo's.
        assert cut_by_window(self._mask(left=True, top=True), 0, 0, FRAME_W, FRAME_H) is False
        # Clamped to the bottom-right corner likewise.
        ox, oy = FRAME_W - CROP_W, FRAME_H - CROP_H
        assert cut_by_window(self._mask(right=True, bottom=True), ox, oy, FRAME_W, FRAME_H) is False

    def test_a_clamped_window_is_still_cut_on_its_free_edges(self):
        assert cut_by_window(self._mask(right=True), 0, 0, FRAME_W, FRAME_H) is True


# ── the kernel ──────────────────────────────────────────────────────────────


def _frame(width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    return np.full((height, width, 3), 40, dtype=np.uint8)


class _Stub:
    """Returns fixed crop-local masks, and records what it was handed."""

    def __init__(self, masks):
        self._masks = masks
        self.seen_shape = None

    def segment(self, image):
        self.seen_shape = image.shape[:2]
        return self._masks


def _fish_mask(cx: int, cy: int, half_len: int = 200, half_h: int = 60):
    """An ellipse, so the head/tail detector has a real principal axis."""
    m = np.zeros((CROP_H, CROP_W), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (half_len, half_h), 0, 0, 360, 1, -1)
    return m


class _Exploding:
    def find_head_tail_img(self, _mask):
        raise RuntimeError("native detector failed")


class _Ends:
    """A head/tail detector that needs no native module: the mask's extreme
    columns, so the lift is the only thing under test."""

    def find_head_tail_img(self, mask):
        ys, xs = np.nonzero(mask)
        left, right = int(np.argmin(xs)), int(np.argmax(xs))
        return (np.array([xs[right], ys[right]], float), np.array([xs[left], ys[left]], float))


def _find(frame, points, stub, **kwargs):
    kwargs.setdefault("head_tail_detector", _Ends())
    return find_fish_at_dot(frame, points, stub, **kwargs)


def test_predicts_and_lifts_keypoints_into_frame_coordinates():
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    stub = _Stub([_fish_mask(int(laser[0] - ox), int(laser[1] - oy))])

    result = _find(_frame(), [laser], stub)

    assert result.status == "predicted"
    assert stub.seen_shape == (CROP_H, CROP_W), "model must see the crop, not the frame"
    assert (result.crop_x, result.crop_y) == (ox, oy)
    assert result.head[0] == pytest.approx(laser[0], abs=260)
    assert result.tail[0] == pytest.approx(laser[0], abs=260)
    assert abs(result.head[0] - result.tail[0]) == pytest.approx(400, abs=40)
    assert result.length_px == pytest.approx(400, abs=40)
    assert (result.width, result.height) == (FRAME_W, FRAME_H)


def test_keypoints_are_not_left_in_crop_coordinates():
    """A missing lift puts every keypoint within the crop's own extent, which
    still looks like a fish."""
    laser = (3500.0, 2500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    assert ox > 0 and oy > 0, "fixture must use an off-origin crop to be meaningful"
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))

    result = _find(_frame(), [laser], _Stub([mask]))

    assert result.head[0] > CROP_W or result.head[1] > CROP_H


def test_laser_on_no_mask_abstains():
    mask = _fish_mask(100, 100, 40, 20)
    result = _find(_frame(), [(2000.0, 1500.0)], _Stub([mask]))
    assert result.status == "laser_off_all_fish"
    assert result.head is None and result.mask is None


def test_no_masks_abstains_as_no_detections():
    result = _find(_frame(), [(2000.0, 1500.0)], _Stub([]))
    assert result.status == "no_detections"


def test_gate_picks_the_lasered_fish_not_the_largest():
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    big = _fish_mask(300, 300, 400, 150)
    small = _fish_mask(int(laser[0] - ox), int(laser[1] - oy), 120, 40)

    result = _find(_frame(), [laser], _Stub([big, small]))

    assert result.status == "predicted"
    assert result.mask_area_px == pytest.approx(int(np.count_nonzero(small)), rel=0.01)
    assert np.array_equal(result.mask, small)


def test_records_which_dot_chose_the_fish():
    """Provenance: a caller whose laser label is later superseded must be
    able to tell this result is stale. It gets the dot's index."""
    laser_a = (1800.0, 1500.0)
    laser_b = (2300.0, 1500.0)
    ox, oy = crop_origin(*laser_a, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser_b[0] - ox), int(laser_b[1] - oy), 120, 50)
    assert mask[int(laser_a[1] - oy), int(laser_a[0] - ox)] == 0, "fixture: A must miss"
    assert mask[int(laser_b[1] - oy), int(laser_b[0] - ox)] == 1, "fixture: B must hit"

    result = _find(_frame(), [laser_a, laser_b], _Stub([mask]))

    assert (result.status, result.dot) == ("predicted", 1)


def test_crop_is_centred_on_the_first_laser_point_only():
    """A second dot outside that window cannot be gated on; in the corpus that
    never happens, which is why cropping on the first point is safe."""
    far = (200.0, 200.0)
    laser = (3800.0, 2800.0)
    ox, oy = crop_origin(far[0], far[1], FRAME_W, FRAME_H, CROP_W, CROP_H)
    assert not (ox <= laser[0] < ox + CROP_W and oy <= laser[1] < oy + CROP_H)

    mask = _fish_mask(CROP_W // 2, CROP_H // 2)
    result = _find(_frame(), [far, laser], _Stub([mask]))

    assert (result.crop_x, result.crop_y) == (ox, oy)


def test_silhouette_ratio_is_recorded():
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))

    result = _find(_frame(), [laser], _Stub([mask]))

    assert result.silhouette_ratio is not None
    assert 0.05 < result.silhouette_ratio < 1.0


def test_no_laser_points_abstains_and_names_no_dot():
    stub = _Stub([])
    result = _find(_frame(), [], stub)
    assert result.status == "laser_off_all_fish"
    assert result.dot is None
    assert stub.seen_shape is None, "nothing to segment without a dot"


def test_an_unfittable_mask_is_a_headtail_failure_not_an_error():
    """The detector is a native call with no documented exceptions; one bad
    mask must not fail the caller's whole batch."""
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)

    result = _find(
        _frame(),
        [laser],
        _Stub([_fish_mask(int(laser[0] - ox), int(laser[1] - oy))]),
        head_tail_detector=_Exploding(),
    )

    assert result.status == "headtail_failed"
    assert result.mask_area_px > 0 and result.head is None


def test_a_smaller_window_is_honoured():
    """The experiment runs production's stage on stored crops smaller than
    the window: the clamp then feeds the whole crop."""
    frame = _frame(900, 700)
    stub = _Stub([])
    _find(frame, [(450.0, 350.0)], stub, crop_w=1800, crop_h=1350)
    assert stub.seen_shape == (700, 900)


def test_dots_may_be_an_array():
    """The laser detector and the line fit hand over (N, 2) arrays; an array
    has no truth value, so the empty check must not ask for one."""
    laser = np.array([[2000.0, 1500.0], [2100.0, 1500.0]])
    ox, oy = crop_origin(laser[0, 0], laser[0, 1], FRAME_W, FRAME_H, CROP_W, CROP_H)
    stub = _Stub([_fish_mask(int(laser[0, 0] - ox), int(laser[0, 1] - oy))])

    result = _find(_frame(), laser, stub)

    assert (result.status, result.dot) == ("predicted", 0)
    assert _find(_frame(), np.zeros((0, 2)), _Stub([])).status == "laser_off_all_fish"


def test_results_with_masks_compare_and_hash():
    """The result is a frozen dataclass; the mask has no truth value, so it
    sits out of equality and hashing or both would raise."""
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))

    a = _find(_frame(), [laser], _Stub([mask]))
    b = _find(_frame(), [laser], _Stub([mask.copy()]))

    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a.mask is not None


class TestAnAbstentionNamesItsDot:
    """v1 left the dot NULL on every abstention, so a laser correction never
    made one stale: the frame kept its "no fish" forever. A kept mask names
    the dot on it, as a prediction does; with no mask kept, the dot the crop
    was centred on, the first, is the one the answer came from."""

    LASER = (2000.0, 1500.0)

    def test_no_detections_names_the_crop_centre(self):
        result = _find(_frame(), [self.LASER, (2100.0, 1500.0)], _Stub([]))
        assert (result.status, result.dot) == ("no_detections", 0)

    def test_laser_off_all_fish_names_the_crop_centre(self):
        result = _find(
            _frame(), [self.LASER, (2100.0, 1500.0)], _Stub([_fish_mask(100, 100, 40, 20)])
        )
        assert (result.status, result.dot) == ("laser_off_all_fish", 0)

    def test_headtail_failed_names_the_dot_on_the_mask(self):
        miss, hit = (1800.0, 1500.0), (2300.0, 1500.0)
        ox, oy = crop_origin(*miss, FRAME_W, FRAME_H, CROP_W, CROP_H)
        mask = _fish_mask(int(hit[0] - ox), int(hit[1] - oy), 120, 50)

        result = _find(_frame(), [miss, hit], _Stub([mask]), head_tail_detector=_Exploding())

        assert (result.status, result.dot) == ("headtail_failed", 1)


class TestTheKeptMasksBox:
    """The kept mask's box rides on the result, in frame pixels, so a species
    or keypoint stage crops exactly the fish SAM 3.1 segmented at the dot,
    with no second SAM pass, and nothing else."""

    LASER = (3500.0, 2500.0)

    def _mask(self):
        ox, oy = crop_origin(*self.LASER, FRAME_W, FRAME_H, CROP_W, CROP_H)
        assert ox > 0 and oy > 0, "an off-origin crop, or a missing lift passes"
        mask = np.zeros((CROP_H, CROP_W), dtype=np.uint8)
        local_x, local_y = int(self.LASER[0] - ox), int(self.LASER[1] - oy)
        mask[local_y - 40 : local_y + 30, local_x - 200 : local_x + 150] = 1
        box = (local_x - 200 + ox, local_y - 40 + oy, local_x + 150 + ox, local_y + 30 + oy)
        return mask, box

    def test_a_prediction_carries_its_masks_box_in_frame_pixels(self):
        mask, box = self._mask()

        result = _find(_frame(), [self.LASER], _Stub([mask]))

        assert result.status == "predicted"
        assert result.mask_bbox == box

    def test_the_box_is_the_kept_masks_not_another(self):
        mask, box = self._mask()
        elsewhere = np.zeros_like(mask)
        elsewhere[10:20, 10:20] = 1

        result = _find(_frame(), [self.LASER], _Stub([elsewhere, mask]))

        assert result.mask_bbox == box

    def test_an_unfittable_mask_still_names_its_box(self):
        """The mask was kept, so the fish is known even when its keypoints
        are not: the next stage can still crop it."""
        mask, box = self._mask()

        result = _find(_frame(), [self.LASER], _Stub([mask]), head_tail_detector=_Exploding())

        assert (result.status, result.mask_bbox) == ("headtail_failed", box)

    @pytest.mark.parametrize("masks", [[], "off"], ids=["no-masks", "laser-off"])
    def test_no_kept_mask_means_no_box(self, masks):
        if masks == "off":
            masks = [_fish_mask(100, 100, 40, 20)]
        result = _find(_frame(), [self.LASER], _Stub(masks))
        assert result.mask_bbox is None


def test_the_real_detector_is_the_default():
    """With no detector passed, the stage runs fishsense-core's own
    FishHeadTailDetector on the kept mask: the SAM rule as production
    computes it."""
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))

    result = find_fish_at_dot(_frame(), [laser], _Stub([mask]))

    assert result.status == "predicted"
    assert abs(result.head[0] - result.tail[0]) == pytest.approx(400, abs=40)
    assert result.head[1] == pytest.approx(laser[1], abs=70)


# ── a fish longer than the window ───────────────────────────────────────────
#
# The window is 1800x1350 around the dot; a fish longer than it came back cut
# by the window's straight edges, and its head and tail were read off the cut
# (lab mission 383, 2026-10-22: a 686 mm Nassau Grouper measured 554 mm). With
# grow_cut_masks, a mask touching an edge of the window that is not the frame's
# own is segmented again on the whole frame. Ported from fishsense-services'
# tests/test_automatic_frames.py.

DOT = (2000.0, 1500.0)


def _fish_in_frame(cx, cy, half_len, half_h=150):
    m = np.zeros((FRAME_H, FRAME_W), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (half_len, half_h), 0, 0, 360, 1, -1)
    return m


def _window_of(frame_mask, dot=DOT):
    ox, oy = crop_origin(*dot, FRAME_W, FRAME_H, CROP_W, CROP_H)
    return np.ascontiguousarray(frame_mask[oy : oy + CROP_H, ox : ox + CROP_W])


class _Windowed:
    """A segmenter that answers the window and the whole frame each with
    their own masks, as SAM would see them."""

    def __init__(self, window, frame):
        self.answers = {(CROP_H, CROP_W): window, (FRAME_H, FRAME_W): frame}
        self.shapes = []

    def segment(self, image):
        self.shapes.append(image.shape[:2])
        return self.answers[image.shape[:2]]


LONG_FISH = _fish_in_frame(*DOT, half_len=1300)


class TestGrowCutMasks:
    def test_a_fish_longer_than_the_window_is_measured_on_the_whole_frame(self):
        seg = _Windowed([_window_of(LONG_FISH)], [LONG_FISH])

        r = _find(_frame(), [DOT], seg, grow_cut_masks=True)

        assert seg.shapes == [(CROP_H, CROP_W), (FRAME_H, FRAME_W)]
        assert r.status == "predicted"
        assert (r.crop_x, r.crop_y) == (0, 0)
        assert r.mask_bbox[0] <= DOT[0] - 1290 and r.mask_bbox[2] >= DOT[0] + 1290
        ends = sorted((r.head[0], r.tail[0]))
        assert ends[0] == pytest.approx(DOT[0] - 1300, abs=30)
        assert ends[1] == pytest.approx(DOT[0] + 1300, abs=30)
        assert r.mask.shape == (FRAME_H, FRAME_W), "the whole frame's mask, at (0, 0)"

    def test_off_by_default_the_cut_mask_is_kept(self):
        """The human pre-fill path's output is pinned: without the flag the
        window's answer stands, cut or not."""
        seg = _Windowed([_window_of(LONG_FISH)], [LONG_FISH])

        r = _find(_frame(), [DOT], seg)

        assert seg.shapes == [(CROP_H, CROP_W)]
        assert r.status == "predicted"
        assert (r.crop_x, r.crop_y) == crop_origin(*DOT, FRAME_W, FRAME_H, CROP_W, CROP_H)
        assert r.length_px < CROP_W

    def test_a_fish_inside_the_window_is_segmented_once(self):
        fish = _fish_in_frame(*DOT, half_len=500)
        seg = _Windowed([_window_of(fish)], [])

        r = _find(_frame(), [DOT], seg, grow_cut_masks=True)

        assert seg.shapes == [(CROP_H, CROP_W)]
        assert r.status == "predicted"
        assert (r.crop_x, r.crop_y) == crop_origin(*DOT, FRAME_W, FRAME_H, CROP_W, CROP_H)

    def test_a_mask_cut_only_by_the_frames_own_edge_is_not_segmented_again(self):
        """Near the frame's edge the window is clamped there: a fish running
        off the photo is cut by the photo, which a bigger window cannot help."""
        dot = (300.0, 1500.0)
        fish = _fish_in_frame(150, 1500, half_len=600)  # runs off the left edge
        seg = _Windowed([_window_of(fish, dot)], [])

        r = _find(_frame(), [dot], seg, grow_cut_masks=True)

        assert seg.shapes == [(CROP_H, CROP_W)]
        assert r.status == "predicted" and r.crop_x == 0

    def test_a_cut_fish_the_whole_frame_puts_no_mask_under_the_dot_abstains(self):
        """The cut mask's length is known to be wrong, so it is never kept."""
        elsewhere = _fish_in_frame(DOT[0] + 1500, 400, half_len=300)
        seg = _Windowed([_window_of(LONG_FISH)], [elsewhere])

        r = _find(_frame(), [DOT], seg, grow_cut_masks=True)

        assert (r.status, r.dot) == ("laser_off_all_fish", 0)
        assert r.head is None and r.mask is None and (r.crop_x, r.crop_y) == (0, 0)

    def test_a_cut_fish_the_whole_frame_finds_nothing_on_is_no_detection(self):
        seg = _Windowed([_window_of(LONG_FISH)], [])

        r = _find(_frame(), [DOT], seg, grow_cut_masks=True)

        assert (r.status, r.crop_x, r.crop_y) == ("no_detections", 0, 0)

    def test_the_second_pass_names_the_dot_on_the_whole_frame_mask(self):
        miss = (DOT[0], 2100.0)  # below the fish; its window still holds DOT
        seg = _Windowed([_window_of(LONG_FISH, miss)], [LONG_FISH])
        # The window at `miss` catches the fish's lower part, cut at its top.
        assert _window_of(LONG_FISH, miss)[0].any()

        r = _find(_frame(), [miss, DOT], seg, grow_cut_masks=True)

        assert (r.status, r.dot) == ("predicted", 1)


# ── mask conversion ─────────────────────────────────────────────────────────


class TestMaskConversion:
    """SAM3 returns device tensors; `np.asarray` on a CUDA tensor raises."""

    def test_detaches_and_moves_a_device_tensor(self):
        class _DeviceTensor:
            def __init__(self, data):
                self._data = data
                self.detached = False

            def __array__(self, *args, **kwargs):
                raise TypeError("can't convert cuda:0 device type tensor to numpy")

            def detach(self):
                self.detached = True
                return self

            def cpu(self):
                return np.asarray(self._data)

        tensor = _DeviceTensor([[1, 0], [0, 1]])
        out = to_numpy(tensor)
        assert tensor.detached
        assert out.tolist() == [[1, 0], [0, 1]]

    def test_passes_a_plain_array_through(self):
        assert to_numpy(np.zeros((2, 2))).shape == (2, 2)


# ── the fallback backend ────────────────────────────────────────────────────


class TestFishialSegmenter:
    class _Recording:
        def __init__(self, labels):
            self._labels = labels
            self.seen = None

        def inference(self, image):
            self.seen = image
            return self._labels

    def test_splits_the_instance_label_map_into_binary_masks(self):
        labels = np.zeros((40, 60), dtype=np.int32)
        labels[5:10, 5:15] = 1
        labels[20:25, 30:50] = 7  # ids are arbitrary, not contiguous

        masks = FishialSegmenter(self._Recording(labels)).segment(
            np.zeros((40, 60, 3), dtype=np.uint8)
        )

        assert len(masks) == 2
        assert sorted(int(np.asarray(m).sum()) for m in masks) == [50, 100]
        assert all(np.asarray(m).dtype == bool for m in masks)

    def test_background_is_not_returned_as_a_mask(self):
        labels = np.zeros((20, 40), dtype=np.int32)
        segmenter = FishialSegmenter(self._Recording(labels))
        assert segmenter.segment(np.zeros((20, 40, 3), dtype=np.uint8)) == []

    def test_the_frame_is_passed_through_as_bgr(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # blue channel only, in BGR
        recorder = self._Recording(np.zeros((20, 40), dtype=np.int32))

        FishialSegmenter(recorder).segment(frame)

        assert np.array_equal(recorder.seen, frame), "frame was converted"

    @pytest.mark.parametrize("shape", [(40, 20, 3), (30, 30, 3)], ids=["portrait", "square"])
    def test_a_portrait_or_square_crop_goes_to_the_model_as_is(self, shape):
        """The service's copy refused these, from a time when the native
        ``inference`` returned an all-zero mask for them. It now rotates a
        portrait image in and its mask back out, and takes a square one
        natively, so the guard would only lose fish."""
        labels = np.zeros(shape[:2], dtype=np.int32)
        labels[5:10, 5:10] = 1
        recorder = self._Recording(labels)

        masks = FishialSegmenter(recorder).segment(np.zeros(shape, dtype=np.uint8))

        assert recorder.seen is not None and recorder.seen.shape == shape
        assert len(masks) == 1 and int(masks[0].sum()) == 25

    def test_load_calls_load_model_before_the_segmenter_is_published(self, monkeypatch):
        """Unloaded, `inference` raises on every frame."""

        class _Segmentation:
            def __init__(self):
                self.loaded = False

            def load_model(self):
                self.loaded = True

            def inference(self, _image):
                if not self.loaded:
                    raise ValueError("model has not been loaded")
                return np.zeros((4, 8), dtype=np.int32)

        built = _Segmentation()
        monkeypatch.setattr(sut, "FishSegmentation", lambda: built)

        got = FishialSegmenter.load()

        assert built.loaded is True, "load_model() was never called"
        assert got.segment(np.zeros((4, 8, 3), dtype=np.uint8)) == []


# ── the SAM 3.1 adapter (needs torch) ───────────────────────────────────────


class TestSam3Segmenter:
    """SAM 3.1's weights are bfloat16 and `Sam3Processor` sets up no autocast:
    without a context, every frame dies in `vitdet.forward` with a dtype
    mismatch, a plain retryable RuntimeError, which in prod looped on all 356
    images of dive 94 while holding a GPU (2026-09-08). Every other test
    stubs the very seam this adapter implements, so only these can see it."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    class _RecordingProcessor:
        """A stand-in with `Sam3Processor`'s REAL signatures: `set_image`
        returns the state, `set_text_prompt(prompt, state)` takes it back and
        *is* the inference call, and the result is a dict."""

        def __init__(self, torch, device_type):
            self._torch = torch
            self._device_type = device_type
            self.enabled_at_set_image = None
            self.enabled_at_predict = None
            self.dtype_at_predict = None
            self.image_type = None
            self.prompt = None
            self.state_round_tripped = False

        def _sample(self):
            return (
                self._torch.is_autocast_enabled(self._device_type),
                self._torch.get_autocast_dtype(self._device_type),
            )

        def set_image(self, image, _state=None):
            self.enabled_at_set_image = self._sample()[0]
            self.image_type = type(image).__name__
            return {"sentinel": object()}

        def set_text_prompt(self, prompt, state):
            self.enabled_at_predict, self.dtype_at_predict = self._sample()
            self.prompt = prompt
            self.state_round_tripped = "sentinel" in state
            state["masks"] = []
            return state

    def _device_type(self):
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def _processor(self):
        return self._RecordingProcessor(self.torch, self._device_type())

    def _segment(self, processor):
        return Sam3Segmenter(processor).segment(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_inference_runs_under_bfloat16_autocast(self):
        processor = self._processor()

        assert not self._segment(processor)

        assert processor.enabled_at_set_image is True, (
            "the vision backbone runs inside set_image -- that is where the dtype "
            "mismatch was raised"
        )
        assert processor.enabled_at_predict is True
        assert processor.dtype_at_predict is self.torch.bfloat16

    def test_state_is_carried_from_set_image_into_the_prompt_call(self):
        processor = self._processor()
        self._segment(processor)
        assert processor.state_round_tripped is True

    def test_the_model_is_handed_a_pil_image_not_the_raw_array(self):
        """An HWC ndarray yields three-pixel-wide masks, silently."""
        processor = self._processor()
        self._segment(processor)
        assert processor.image_type == "Image", processor.image_type

    def test_the_prompt_is_fish(self):
        processor = self._processor()
        self._segment(processor)
        assert processor.prompt == "fish"

    def test_autocast_does_not_leak_past_the_call(self):
        """A context manager, not `__enter__()`: the caller's thread is reused."""
        self._segment(self._processor())
        assert self.torch.is_autocast_enabled(self._device_type()) is False

    @staticmethod
    def _processor_returning(masks):
        class _Processor:
            def set_image(self, _image):
                return {}

            def set_text_prompt(self, _prompt, state):
                state["masks"] = masks
                return state

        return _Processor()

    def test_masks_are_squeezed_numpy(self):
        masks = [np.zeros((1, 5, 7))]  # a list of (1, H, W), as SAM returns
        got = Sam3Segmenter(self._processor_returning(masks)).segment(
            np.zeros((5, 7, 3), dtype=np.uint8)
        )
        assert len(got) == 1 and got[0].shape == (5, 7)

    def test_a_batch_tensor_crosses_to_the_host_once(self):
        """SAM hands back one (N, 1, H, W) device tensor; moving it per
        instance is N syncs and N transfers on the per-frame hot path."""

        class _Batch:
            """A stand-in device tensor: iterable only once on the host."""

            def __init__(self, data):
                self._data = data
                self.transfers = 0

            def detach(self):
                return self

            def cpu(self):
                self.transfers += 1
                return self._data

            def __iter__(self):
                raise AssertionError("iterated on the device")

        batch = _Batch(np.ones((3, 1, 5, 7)))
        got = Sam3Segmenter(self._processor_returning(batch)).segment(
            np.zeros((5, 7, 3), dtype=np.uint8)
        )
        assert batch.transfers == 1
        assert len(got) == 3 and all(m.shape == (5, 7) for m in got)


# ── SAM's score as the gate ─────────────────────────────────────────────────
#
# The automatic chain gates on SAM's own confidence (paper §6.3): only masks
# scoring at least the gate (fishsense-services' AUTOMATIC_SAM_SCORE_GATE,
# 0.5) reach the laser gate, and the kept mask's score is recorded. Ported
# from fishsense-services' tests/test_automatic_frames.py (`_Scored`,
# `_Gated`, `ScoredSam3Adapter`).

GATE = 0.5


def _scored_mask_at(x, y, half_len=200, half_h=60):
    ox, oy = crop_origin(*DOT, FRAME_W, FRAME_H, CROP_W, CROP_H)
    return _fish_mask(int(x - ox), int(y - oy), half_len, half_h)


class _Scored:
    """A scored segmenter: (mask, score) pairs, crop-local. Its plain
    `segment` must never be called when `segment_scored` exists."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.calls = 0

    def segment(self, image):
        raise AssertionError("segment called on a scored segmenter")

    def segment_scored(self, image):
        self.calls += 1
        assert image.shape[:2] == (CROP_H, CROP_W)
        return self.pairs


class _WindowedScored(_Windowed):
    def segment(self, image):
        raise AssertionError("segment called on a scored segmenter")

    def segment_scored(self, image):
        self.shapes.append(image.shape[:2])
        return self.answers[image.shape[:2]]


def _gated(segmenter, dot=DOT, **kwargs):
    kwargs.setdefault("score_gate", GATE)
    return _find(_frame(), [dot], segmenter, **kwargs)


class TestScoreGate:
    def test_a_fish_at_the_dot_is_keypointed_with_its_score(self):
        r = _gated(_Scored([(_scored_mask_at(*DOT), 0.81)]))

        assert r.status == "predicted"
        assert r.score == pytest.approx(0.81)
        assert isinstance(r.score, float)
        assert r.head[0] == pytest.approx(DOT[0], abs=260)
        assert r.mask_bbox is not None

    def test_a_mask_below_the_gate_is_not_kept(self):
        r = _gated(_Scored([(_scored_mask_at(*DOT), 0.49)]))
        assert r.status == "no_detections"
        assert r.score is None and r.head is None and r.mask is None

    def test_the_gate_is_inclusive(self):
        r = _gated(_Scored([(_scored_mask_at(*DOT), 0.5)]))
        assert r.status == "predicted" and r.score == pytest.approx(0.5)

    def test_a_confident_fish_off_the_dot_does_not_count(self):
        """The dot must be on the mask; a weak mask under it is not kept either."""
        r = _gated(
            _Scored(
                [(_scored_mask_at(*DOT), 0.3), (_scored_mask_at(DOT[0] + 600, DOT[1]), 0.95)]
            )
        )
        assert r.status == "laser_off_all_fish" and r.score is None

    def test_the_kept_masks_score_is_the_one_under_the_dot(self):
        r = _gated(
            _Scored(
                [(_scored_mask_at(DOT[0] + 600, DOT[1]), 0.95), (_scored_mask_at(*DOT), 0.7)]
            )
        )
        assert r.status == "predicted" and r.score == pytest.approx(0.7)

    def test_the_gate_runs_before_the_laser_gate(self):
        """A weak mask first in order and under the dot must not shadow a
        passing one under it too: the gate drops it before the first-hit rule."""
        r = _gated(
            _Scored([(_scored_mask_at(*DOT), 0.2), (_scored_mask_at(*DOT, half_len=300), 0.6)])
        )
        assert r.status == "predicted" and r.score == pytest.approx(0.6)
        assert r.mask_area_px == int(np.count_nonzero(_scored_mask_at(*DOT, half_len=300)))

    def test_an_unfittable_kept_mask_still_names_its_score(self):
        r = _gated(_Scored([(_scored_mask_at(*DOT), 0.81)]), head_tail_detector=_Exploding())
        assert r.status == "headtail_failed" and r.score == pytest.approx(0.81)

    def test_without_a_gate_a_scored_segmenter_still_reports_its_score(self):
        r = _find(_frame(), [DOT], _Scored([(_scored_mask_at(*DOT), 0.3)]))
        assert r.status == "predicted" and r.score == pytest.approx(0.3)

    def test_a_plain_segmenter_scores_nothing_and_is_unchanged(self):
        r = _find(_frame(), [DOT], _Stub([_scored_mask_at(*DOT)]))
        assert r.status == "predicted" and r.score is None

    def test_a_gate_on_a_segmenter_with_no_scores_raises(self):
        with pytest.raises(TypeError, match="segment_scored"):
            _gated(_Stub([_scored_mask_at(*DOT)]))

    def test_a_gate_on_an_unscored_mask_raises(self):
        with pytest.raises(ValueError, match="unscored"):
            _gated(_Scored([(_scored_mask_at(*DOT), None)]))

    def test_the_score_survives_equality_and_hashing(self):
        a = _gated(_Scored([(_scored_mask_at(*DOT), 0.81)]))
        b = _gated(_Scored([(_scored_mask_at(*DOT), 0.81)]))
        c = _gated(_Scored([(_scored_mask_at(*DOT), 0.82)]))
        assert a == b and hash(a) == hash(b)
        assert a != c


class TestScoreGateWithGrowCutMasks:
    def test_a_fish_longer_than_the_window_is_measured_on_the_whole_frame(self):
        seg = _WindowedScored([(_window_of(LONG_FISH), 0.93)], [(LONG_FISH, 0.88)])

        r = _gated(seg, grow_cut_masks=True)

        assert seg.shapes == [(CROP_H, CROP_W), (FRAME_H, FRAME_W)]
        assert r.status == "predicted"
        assert (r.crop_x, r.crop_y) == (0, 0)
        assert r.mask_bbox[0] <= DOT[0] - 1290 and r.mask_bbox[2] >= DOT[0] + 1290
        ends = sorted((r.head[0], r.tail[0]))
        assert ends[0] == pytest.approx(DOT[0] - 1300, abs=30)
        assert ends[1] == pytest.approx(DOT[0] + 1300, abs=30)
        assert r.score == pytest.approx(0.88)  # the whole frame's mask's

    def test_a_fish_inside_the_window_is_segmented_once(self):
        fish = _fish_in_frame(*DOT, half_len=500)
        seg = _WindowedScored([(_window_of(fish), 0.9)], [])

        r = _gated(seg, grow_cut_masks=True)

        assert seg.shapes == [(CROP_H, CROP_W)]
        assert r.status == "predicted" and r.score == pytest.approx(0.9)
        assert (r.crop_x, r.crop_y) == crop_origin(*DOT, FRAME_W, FRAME_H, CROP_W, CROP_H)

    def test_a_mask_cut_only_by_the_frames_own_edge_is_not_segmented_again(self):
        dot = (300.0, 1500.0)
        fish = _fish_in_frame(150, 1500, half_len=600)  # runs off the left edge
        seg = _WindowedScored([(_window_of(fish, dot), 0.9)], [])

        r = _gated(seg, dot=dot, grow_cut_masks=True)

        assert seg.shapes == [(CROP_H, CROP_W)]
        assert r.status == "predicted" and r.crop_x == 0

    def test_a_cut_fish_the_whole_frame_puts_no_mask_under_the_dot_abstains(self):
        elsewhere = _fish_in_frame(DOT[0] + 1500, 400, half_len=300)
        seg = _WindowedScored([(_window_of(LONG_FISH), 0.93)], [(elsewhere, 0.9)])

        r = _gated(seg, grow_cut_masks=True)

        assert r.status == "laser_off_all_fish"
        assert r.head is None and r.score is None

    def test_a_cut_fish_whose_whole_frame_mask_fails_the_gate_is_no_detection(self):
        seg = _WindowedScored([(_window_of(LONG_FISH), 0.93)], [(LONG_FISH, 0.4)])

        r = _gated(seg, grow_cut_masks=True)

        assert r.status == "no_detections" and r.score is None


class TestSam3SegmenterScores:
    """`ScoredSam3Adapter.segment_scored`'s contract, on core's adapter."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    @staticmethod
    def _processor_returning(state_items):
        class _Processor:
            def set_image(self, _image):
                return {}

            def set_text_prompt(self, _prompt, state):
                state.update(state_items)
                return state

        return _Processor()

    def _scored(self, **state_items):
        return Sam3Segmenter(self._processor_returning(state_items)).segment_scored(
            np.zeros((5, 7, 3), dtype=np.uint8)
        )

    def test_masks_come_with_their_scores_in_order(self):
        got = self._scored(
            masks=[np.zeros((1, 5, 7)), np.ones((1, 5, 7))],
            scores=self.torch.tensor([0.9, 0.4]),
        )
        assert [m.shape for m, _ in got] == [(5, 7), (5, 7)]
        assert [s for _, s in got] == pytest.approx([0.9, 0.4])
        assert all(isinstance(s, float) for _, s in got)

    def test_bfloat16_scores_are_read(self):
        """numpy has no bfloat16: the scores are cast before crossing."""
        got = self._scored(
            masks=[np.zeros((1, 5, 7))],
            scores=self.torch.tensor([0.75], dtype=self.torch.bfloat16),
        )
        assert got[0][1] == pytest.approx(0.75)

    def test_a_batch_score_tensor_crosses_to_the_host_once(self):
        class _Batch:
            def __init__(self, data):
                self._data = data
                self.transfers = 0

            def float(self):
                return self

            def detach(self):
                return self

            def cpu(self):
                self.transfers += 1
                return self._data

            def __iter__(self):
                raise AssertionError("iterated on the device")

        scores = _Batch(np.array([0.9, 0.8, 0.7]))
        got = self._scored(masks=np.ones((3, 1, 5, 7)), scores=scores)
        assert scores.transfers == 1
        assert [s for _, s in got] == pytest.approx([0.9, 0.8, 0.7])

    def test_no_scores_means_none_and_segment_is_unchanged(self):
        masks = [np.zeros((1, 5, 7))]
        assert [s for _, s in self._scored(masks=masks)] == [None]
        got = Sam3Segmenter(self._processor_returning({"masks": masks})).segment(
            np.zeros((5, 7, 3), dtype=np.uint8)
        )
        assert len(got) == 1 and got[0].shape == (5, 7)

    def test_no_masks_is_empty(self):
        assert not self._scored()

    def test_a_score_count_that_does_not_match_the_masks_raises(self):
        with pytest.raises(ValueError, match="2 masks but 1 scores"):
            self._scored(
                masks=[np.zeros((1, 5, 7)), np.zeros((1, 5, 7))],
                scores=self.torch.tensor([0.9]),
            )

    def test_the_kernel_gates_on_sams_scores(self):
        """End to end through the adapter: a weak SAM mask under the dot is
        dropped, a passing one is kept with its score."""
        weak = _fish_mask(100, 100)[None]
        strong = _fish_mask(100, 100, half_len=150)[None]
        seg = Sam3Segmenter(
            self._processor_returning(
                {"masks": [weak, strong], "scores": self.torch.tensor([0.3, 0.8])}
            )
        )
        r = _find(_frame(), [(100.0, 100.0)], seg, score_gate=GATE)
        assert r.status == "predicted" and r.score == pytest.approx(0.8)
        assert r.mask_area_px == int(np.count_nonzero(strong))
