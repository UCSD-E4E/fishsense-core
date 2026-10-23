"""The SimCC head/tail keypoint model: the crop rule, the warp and its
inverse, the decode, and the lift from network input to frame pixels.

The crop rule and the decode are numpy and run everywhere. The model tests
need torch (and timm for the real network); they skip without the extra.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fishsense_core.fish import headtail_keypoints as sut
from fishsense_core.fish.headtail_keypoints import (
    IN,
    K,
    NK,
    PAD_VALUE,
    RADIUS,
    HeadTailKeypointDetector,
    affine,
    crop_from_box,
    decode,
    lift,
    normalize,
    warp,
)
from fishsense_core.models import LocalDirStore, Manifest, ModelIntegrityError

FRAME_W, FRAME_H = 4014, 3016


def test_the_design_constants_are_the_experiments():
    """Changing any of these is a different model; the checkpoint would load
    (or not) and the output would quietly shift."""
    assert (IN, K, RADIUS, NK) == (512, 2, 6, 2)
    assert (sut.BOX_SCALE, sut.CROP_PAD, PAD_VALUE) == (1.4, 128, 114)
    assert sut.BACKBONE == "convnext_tiny.fb_in22k_ft_in1k"


# ── the crop rule ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bbox,crop",
    [
        ((1900, 1255, 2270, 1496), (1711, 1002, 2459, 1749)),
        ((1900, 1259, 2270, 1496), (1713, 1005, 2457, 1750)),
        ((1898, 1261, 2270, 1497), (1711, 1006, 2457, 1752)),
        ((1898, 1254, 2277, 1497), (1708, 996, 2467, 1755)),
        ((1905, 1270, 2240, 1498), (1724, 1036, 2421, 1732)),
        # Clipped at the top of the frame.
        ((1049, 937, 2671, 1577), (575, 0, 3145, 2542)),
        ((1046, 910, 2670, 1559), (569, 0, 3147, 2523)),
        ((1047, 909, 2670, 1560), (570, 0, 3147, 2523)),
        ((1119, 702, 2458, 1582), (602, 0, 2975, 2328)),
    ],
    ids=[
        "3339-auto", "3339-human", "3340-auto", "3340-human", "3342-auto",
        "4145-human-clipped", "4147-auto-clipped", "4147-human-clipped", "4376-auto-clipped",
    ],
)  # fmt: skip
def test_crop_from_box_reproduces_the_experiments_crops(bbox, crop):
    """Rows of ``results/sam_crops.csv`` in
    UCSD-E4E/2026-10-22_headtail_keypoints, with the SAM mask box each was
    cut from (``results/sam_boxes.jsonl``). The rule reproduces all 5,073
    stored crops; these nine stand for them. A one-pixel change to the
    floor/ceil pair fails this."""
    assert crop_from_box(bbox, FRAME_W, FRAME_H) == crop


def test_crop_from_box_is_square_when_unclipped():
    x0, y0, x1, y1 = crop_from_box((1000, 1000, 1400, 1100), FRAME_W, FRAME_H)
    assert x1 - x0 == y1 - y0
    # 1.4 x the diagonal (~412) + 128 -> about 705 px a side.
    assert x1 - x0 == pytest.approx(1.4 * np.hypot(400, 100) + 128, abs=2)


def test_crop_from_box_never_leaves_the_frame():
    x0, y0, x1, y1 = crop_from_box((10, 10, 300, 100), 800, 600)
    assert (x0, y0) == (0, 0)
    assert x1 <= 800 and y1 <= 600


# ── warp and lift ───────────────────────────────────────────────────────────


def test_affine_fills_the_input_with_the_longer_side_and_centres_the_crop():
    m = affine(300, 600)
    centre = np.array([300.0, 150.0, 1.0])
    assert m @ centre == pytest.approx([IN / 2, IN / 2])
    assert m[0, 0] == pytest.approx(IN / 600)
    # The crop's corners land on the input's edges along x, inside along y.
    assert (m @ np.array([0.0, 0.0, 1.0]))[0] == pytest.approx(0.0)
    assert (m @ np.array([600.0, 300.0, 1.0]))[0] == pytest.approx(IN)


def test_lift_inverts_affine_for_every_knob():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 400, size=(5, 2))
    for rot, scale, shift, flip in [
        (0.0, 1.0, (0.0, 0.0), False),
        (37.0, 1.1, (0.02, -0.04), True),
        (270.0, 0.8, (-0.06, 0.06), False),
    ]:
        m = affine(400, 300, rot=rot, scale=scale, shift=shift, flip=flip)
        forward = pts @ m[:, :2].T + m[:, 2]
        assert lift(forward, m) == pytest.approx(pts)


def test_warp_pads_with_letterbox_grey():
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = warp(crop, affine(100, 200))
    assert out.shape == (IN, IN, 3)
    assert out[5, IN // 2].tolist() == [PAD_VALUE] * 3, "above the crop: grey"
    assert out[IN // 2, IN // 2].tolist() == [255] * 3, "inside the crop: the crop"


def test_normalize_is_rgb_chw_imagenet():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # red, in BGR
    x = normalize(img)
    assert x.shape == (3, 4, 4)
    assert x[0, 0, 0] == pytest.approx((255 - 0.485 * 255) / (0.229 * 255))
    assert x[2, 0, 0] == pytest.approx(-0.406 / 0.225)


# ── decode ──────────────────────────────────────────────────────────────────


def test_decode_reads_a_one_hot_bin_as_its_pixel():
    logits = np.full((NK, IN * K), -30.0)
    logits[0, 100] = 30.0  # bin 100 -> 50.0 px
    logits[1, 701] = 30.0  # bin 701 -> 350.5 px
    coord, peak = decode(logits)
    assert coord == pytest.approx([50.0, 350.5])
    assert peak == pytest.approx([1.0, 1.0], abs=1e-6)


def test_decode_takes_the_sub_bin_mean_of_a_gaussian():
    """Training's target is a Gaussian of sigma 4 bins; the decode reads it
    back to well under a bin (the +-RADIUS window is cut off-centre, which
    pulls the mean a little toward the argmax: 123.62 here, not 123.7)."""
    bins = np.arange(IN * K, dtype=float)
    target = 123.7 * K  # a sub-bin location, in bins
    logits = -((bins - target) ** 2) / (2 * 4.0**2)
    coord, _ = decode(logits[None])
    assert coord[0] == pytest.approx(123.7, abs=0.1)
    assert 123.5 <= coord[0] <= 123.7, "between the argmax bin and the true centre"


def test_decode_ignores_a_second_mode_outside_the_window():
    """The softmax-weighted mean over all bins would land between two peaks;
    the windowed one stays on the argmax."""
    logits = np.full(IN * K, -30.0)
    logits[200] = 10.0
    logits[800] = 9.0  # nearly as strong, far away
    coord, _ = decode(logits)
    assert coord == pytest.approx(200 / K, abs=1e-6)


def test_decode_batches_over_leading_dims():
    logits = np.full((3, NK, IN * K), -30.0)
    for b in range(3):
        logits[b, :, 10 * (b + 1)] = 30.0
    coord, peak = decode(logits)
    assert coord.shape == (3, NK) and peak.shape == (3, NK)
    assert coord[:, 0] == pytest.approx([5.0, 10.0, 15.0])


# ── the detector (needs torch) ──────────────────────────────────────────────


class _Peaked:
    """A stand-in network: logits peaked at fixed *input* pixels, so the test
    can watch the lift back through the warp and the crop."""

    def __init__(self, torch, points_in):
        self._torch = torch
        self._points = np.asarray(points_in, float)  # (NK, 2) input px
        self.seen = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        self.seen.append(x)
        b = x.shape[0]
        lx = np.full((b, NK, IN * K), -30.0, np.float32)
        ly = np.full((b, NK, IN * K), -30.0, np.float32)
        for k, (px, py) in enumerate(self._points):
            lx[:, k, int(round(px * K))] = 30.0
            ly[:, k, int(round(py * K))] = 30.0
        return self._torch.from_numpy(lx), self._torch.from_numpy(ly)


@pytest.fixture(name="torch")
def _torch():
    return pytest.importorskip("torch")


def test_predict_lifts_input_pixels_through_the_warp_and_the_crop(torch):
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    bbox = (1900, 1255, 2270, 1496)
    x0, y0, x1, y1 = crop_from_box(bbox, FRAME_W, FRAME_H)
    h, w = y1 - y0, x1 - x0
    m = affine(h, w)
    # Pick frame pixels, push them to input pixels, and have the net "find" them.
    head_frame, tail_frame = (2250.0, 1300.0), (1920.0, 1400.0)
    crop_pts = np.array([head_frame, tail_frame]) - [x0, y0]
    input_pts = crop_pts @ m[:, :2].T + m[:, 2]
    net = _Peaked(torch, input_pts)

    got = HeadTailKeypointDetector(net, device="cpu").predict(frame, bbox)

    # Half a bin of quantisation in input pixels, scaled back up to the crop.
    tol = 0.5 / K * max(h, w) / IN + 1e-6
    assert got.head == pytest.approx(head_frame, abs=tol)
    assert got.tail == pytest.approx(tail_frame, abs=tol)
    assert got.crop == (x0, y0, x1, y1)
    assert got.length_px == pytest.approx(np.hypot(330, 100), abs=2 * tol)
    assert got.confidence == pytest.approx(1.0, abs=1e-5)
    assert tuple(net.seen[0].shape) == (1, 3, IN, IN), "one normalised IN x IN input"


def test_predict_crops_batches_and_keeps_order(torch):
    net = _Peaked(torch, [(IN / 2, IN / 2), (IN / 4, IN / 4)])
    det = HeadTailKeypointDetector(net, device="cpu")
    crops = [np.zeros((200, 400, 3), np.uint8)] * 3 + [np.zeros((400, 200, 3), np.uint8)] * 2

    points, conf = det.predict_crops(crops, batch=2)

    assert points.shape == (5, NK, 2) and conf.shape == (5,)
    assert [int(x.shape[0]) for x in net.seen] == [2, 2, 1]
    # The input's centre is the crop's centre, whatever its aspect.
    assert points[0, 0] == pytest.approx([200.0, 100.0], abs=0.3)
    assert points[4, 0] == pytest.approx([100.0, 200.0], abs=0.3)


def test_predict_crops_of_nothing_is_empty(torch):
    det = HeadTailKeypointDetector(_Peaked(torch, [(0, 0), (0, 0)]), device="cpu")
    points, conf = det.predict_crops([])
    assert points.shape == (0, NK, 2) and conf.shape == (0,)


class TestPredictRefusesWhatItCannotMeasure:
    """Each of these used to be an opaque failure (a TypeError on unpacking
    None, an OpenCV assertion, or no failure at all and garbage keypoints)."""

    @pytest.fixture(autouse=True)
    def _det(self, torch):
        self.det = HeadTailKeypointDetector(_Peaked(torch, [(10, 10), (20, 20)]), device="cpu")
        self.frame = np.zeros((300, 400, 3), dtype=np.uint8)

    def test_an_abstentions_none_box_is_a_clear_error(self):
        """FishAtDot.mask_bbox is None on every abstention; the README's
        flow passes it straight in."""
        with pytest.raises(ValueError, match="abstained"):
            self.det.predict(self.frame, None)

    def test_a_box_outside_the_frame_is_named(self):
        with pytest.raises(ValueError, match="outside the 400x300 frame"):
            self.det.predict(self.frame, (500, 500, 600, 600))

    def test_a_float_frame_is_refused_not_normalised_as_uint8(self):
        with pytest.raises(TypeError, match="uint8"):
            self.det.predict(self.frame.astype(np.float32), (100, 100, 200, 150))

    def test_a_uint16_frame_is_refused(self):
        with pytest.raises(TypeError, match="uint8"):
            self.det.predict(self.frame.astype(np.uint16), (100, 100, 200, 150))

    def test_a_grey_frame_is_refused(self):
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            self.det.predict(self.frame[:, :, 0], (100, 100, 200, 150))

    def test_predict_crops_checks_each_crop(self):
        good = np.zeros((50, 80, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="crop 1 is empty"):
            self.det.predict_crops([good, np.zeros((0, 80, 3), dtype=np.uint8)])
        with pytest.raises(TypeError, match="crop 0"):
            self.det.predict_crops([good.astype(np.float32)])


class TestTheNetworkIsOneClass:
    """The experiment had SimCC at module level. Defined inside build_model
    it would be a new class per call: isinstance across instances fails and
    a whole module cannot be pickled or torch.save'd."""

    @pytest.fixture(autouse=True)
    def _timm(self, torch):
        pytest.importorskip("timm")

    def test_two_builds_share_a_type(self):
        assert type(sut.build_model()) is type(sut.build_model())
        assert type(sut.build_model()) is sut.SimCC

    def test_the_class_is_found_by_its_pickled_name(self):
        klass = sut.SimCC
        assert (klass.__module__, klass.__qualname__) == (sut.__name__, "SimCC")
        assert pickle.loads(pickle.dumps(klass)) is klass

    def test_a_whole_module_round_trips_through_torch_save(self, torch):
        module = sut.build_model()
        buf = io.BytesIO()
        torch.save(module, buf)
        buf.seek(0)
        back = torch.load(buf, weights_only=False)
        assert type(back) is sut.SimCC
        for a, b in zip(back.state_dict().values(), module.state_dict().values()):
            assert torch.equal(a, b)


def test_the_real_network_has_the_checkpoints_shape(torch):
    """Random weights, one forward: the module is the trained design (its
    state dict loads strictly), and the head emits IN*K bins per keypoint."""
    pytest.importorskip("timm")
    module = sut.build_model(pretrained=False)
    with torch.no_grad():
        lx, ly = module(torch.zeros(1, 3, IN, IN))
    assert tuple(lx.shape) == (1, NK, IN * K) and tuple(ly.shape) == (1, NK, IN * K)
    keys = set(module.state_dict())
    assert {"cls_x.weight", "cls_y.weight", "reduce.0.weight", "mlp.1.weight"} <= keys
    assert any(k.startswith("backbone.") for k in keys)


def test_from_checkpoint_loads_the_experiments_format_strictly(torch, tmp_path):
    """``{"model": state_dict, "epoch": ..., "length_val": {...}}``, and a
    state dict for another design fails here, not quietly at the output."""
    pytest.importorskip("timm")
    module = sut.build_model(pretrained=False)
    path = tmp_path / "best_length.pt"
    # length_val is a pandas summary: numpy scalars, which the restricted
    # unpickler refuses unless allowed (they are).
    length_val = {"mean_abs_d": np.float64(0.0267), "within10": np.float64(0.951), "n": np.int64(1739)}
    torch.save({"model": module.state_dict(), "epoch": 12, "length_val": length_val}, path)

    det = HeadTailKeypointDetector.from_checkpoint(path, device="cpu")
    assert det.device.type == "cpu"
    for a, b in zip(det.module.state_dict().values(), module.state_dict().values()):
        assert torch.equal(a, b)

    torch.save({"model": {"cls_x.weight": torch.zeros(1)}, "epoch": 1}, path)
    with pytest.raises(RuntimeError):
        HeadTailKeypointDetector.from_checkpoint(path, device="cpu")


class _Payload:  # pylint: disable=too-few-public-methods
    """Any class the unpickler was not told about stands for a tampered file."""

    def __reduce__(self):
        return (_Payload, ())


def test_from_checkpoint_does_not_unpickle_arbitrary_objects(torch, tmp_path):
    """A .pt file is a pickle; with weights_only=False a tampered one runs
    code on load. The restricted unpickler refuses anything but tensors,
    containers and the numpy scalars the payload needs."""
    path = tmp_path / "tampered.pt"
    torch.save({"model": {}, "epoch": 1, "evil": _Payload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        HeadTailKeypointDetector.from_checkpoint(path, device="cpu")


# ── the manifest ────────────────────────────────────────────────────────────

BLOB = b"stand-in checkpoint"


def _manifest(body: bytes = BLOB) -> Manifest:
    return Manifest.parse(f"""
[[model]]
name = "{sut.MODEL_NAME}"
version = "simcc-sam-e12"
default = true
[[model.artifact]]
targets = ["server"]
filename = "best_length.pt"
sha256 = "{hashlib.sha256(body).hexdigest()}"
size = {len(body)}
""")


def _bundle(root: Path, body: bytes = BLOB) -> Path:
    (root / sut.MODEL_NAME / "simcc-sam-e12").mkdir(parents=True)
    (root / sut.MODEL_NAME / "simcc-sam-e12" / "best_length.pt").write_bytes(body)
    return root


def test_from_store_loads_the_verified_file(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        HeadTailKeypointDetector,
        "from_checkpoint",
        classmethod(lambda cls, path, **kw: seen.update(path=Path(path), kw=kw) or "detector"),
    )
    got = HeadTailKeypointDetector.from_store(
        LocalDirStore(_bundle(tmp_path / "b")),
        cache_dir=tmp_path / "c",
        manifest=_manifest(),
        device="cpu",
    )
    assert got == "detector"
    assert seen["path"] == tmp_path / "c" / sut.MODEL_NAME / "simcc-sam-e12" / "best_length.pt"
    assert seen["kw"] == {"device": "cpu"}


def test_from_store_refuses_a_wrong_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        HeadTailKeypointDetector,
        "from_checkpoint",
        classmethod(lambda cls, path, **kw: pytest.fail("loaded unverified weights")),
    )
    bundle = _bundle(tmp_path / "b", b"X" * len(BLOB))
    with pytest.raises(ModelIntegrityError):
        HeadTailKeypointDetector.from_store(
            LocalDirStore(bundle), cache_dir=tmp_path / "c", manifest=_manifest()
        )


def test_the_checkpoint_is_pinned_in_the_shipped_manifest():
    """The selected checkpoint (run `simcc_sam`, epoch 12), at the Hugging
    Face commit it was published in. Measured from the Hub's LFS record and
    checked against the file core downloaded and loaded (2026-10-23)."""
    from fishsense_core import models  # pylint: disable=import-outside-toplevel

    ref = models.builtin_manifest().resolve(sut.MODEL_NAME)
    assert (ref.version, ref.filename) == ("simcc-sam-e12", "simcc_sam_best_length.pt")
    assert ref.sha256 == "8918034a0f84f45eb2b8224a369781468ae63497ff06b009bed958ebb82b0110"
    assert ref.size == 142032463
    assert ref.origins["huggingface"] == {
        "repo": "ucsde4e/fishsense-headtail-keypoint",
        "revision": "4a45ac41734662351a334f4992c6fede7ff859f0",
    }


def test_an_offline_miss_does_not_reach_for_the_hub(tmp_path):
    """store=None with a cold cache raises at once; the Hub is only used
    when a caller hands over a HuggingFaceStore."""
    from fishsense_core.models import ModelUnavailable  # pylint: disable=import-outside-toplevel

    with pytest.raises(ModelUnavailable, match="headtail-keypoints/simcc-sam-e12"):
        HeadTailKeypointDetector.from_store(None, cache_dir=tmp_path)


def test_importable_without_torch(monkeypatch):
    """The crop rule and the decode are numpy; the extra loads on first use."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "timm", None)
    monkeypatch.setattr(sut, "_SIMCC", None)  # forget a class an earlier test built
    assert crop_from_box((0, 0, 10, 10), 100, 100) == (0, 0, 79, 79)
    with pytest.raises(ImportError, match="headtail-keypoints"):
        sut.build_model()
    with pytest.raises(ImportError, match="headtail-keypoints"):
        HeadTailKeypointDetector(types.SimpleNamespace(), device="cpu")
