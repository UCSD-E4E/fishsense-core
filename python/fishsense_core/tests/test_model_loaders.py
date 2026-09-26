"""LaserDetector.from_store and the SAM 3.1 loader, with stub stores and a
stub ``sam3``. No torch weights, no GPU, and no network needed."""

import hashlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("fishsense_core._native")

# pylint: disable=wrong-import-position
from fishsense_core import _laser_detector as ld
from fishsense_core import models
from fishsense_core.fish import sam3 as sam3_loader
from fishsense_core.models import LocalDirStore, Manifest, ModelIntegrityError

BLOB = b"stand-in checkpoint"


def _manifest(name: str, version: str, filename: str, body: bytes = BLOB) -> Manifest:
    return Manifest.parse(f"""
[[model]]
name = "{name}"
version = "{version}"
default = true
[[model.artifact]]
targets = ["server"]
filename = "{filename}"
sha256 = "{hashlib.sha256(body).hexdigest()}"
size = {len(body)}
""")


def _bundle(root: Path, name: str, version: str, filename: str, body: bytes = BLOB) -> Path:
    (root / name / version).mkdir(parents=True)
    (root / name / version / filename).write_bytes(body)
    return root


# ── laser ───────────────────────────────────────────────────────────────────


def test_checkpoint_hashes_come_from_the_manifest():
    laser = {
        r.sha256: r.filename
        for r in models.builtin_manifest().refs
        if r.name == ld.LASER_MODEL_NAME
    }
    assert ld.CHECKPOINT_SHA256 == laser
    # Every manifest checkpoint still has its encoder and bias calibration.
    assert set(laser.values()) <= set(ld.CHECKPOINT_ENCODERS)
    assert set(laser.values()) <= set(ld.CHECKPOINT_BIAS_OFFSETS)


def test_laser_from_store_loads_the_verified_file(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        ld.LaserDetector,
        "from_checkpoint",
        classmethod(lambda cls, path, **kw: seen.update(path=Path(path), kw=kw) or "detector"),
    )
    manifest = _manifest("laser-detector", "run3", "run3_epoch_021.pt")
    bundle = _bundle(tmp_path / "b", "laser-detector", "run3", "run3_epoch_021.pt")

    got = ld.LaserDetector.from_store(
        LocalDirStore(bundle), cache_dir=tmp_path / "c", manifest=manifest, device="cpu"
    )
    assert got == "detector"
    assert seen["path"] == tmp_path / "c" / "laser-detector" / "run3" / "run3_epoch_021.pt"
    assert seen["kw"] == {"device": "cpu"}


def test_laser_from_store_refuses_a_wrong_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ld.LaserDetector,
        "from_checkpoint",
        classmethod(lambda cls, path, **kw: pytest.fail("loaded unverified weights")),
    )
    manifest = _manifest("laser-detector", "run3", "run3_epoch_021.pt")
    bundle = _bundle(
        tmp_path / "b", "laser-detector", "run3", "run3_epoch_021.pt", b"X" * len(BLOB)
    )
    with pytest.raises(ModelIntegrityError):
        ld.LaserDetector.from_store(LocalDirStore(bundle), cache_dir=tmp_path / "c", manifest=manifest)


# ── SAM 3.1 ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_sam3(monkeypatch):
    """Install stub ``sam3`` and ``torch`` modules; return the call log."""
    log: dict = {"cuda": True}

    class Model:
        def eval(self):
            log["eval"] = True
            return self

    def build_sam3_image_model(*, checkpoint_path):
        log["checkpoint_path"] = checkpoint_path
        return Model()

    class Sam3Processor:
        def __init__(self, model):
            self.model = model

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: log["cuda"])
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = build_sam3_image_model
    processor = types.ModuleType("sam3.model.sam3_image_processor")
    processor.Sam3Processor = Sam3Processor
    for name, mod in {
        "torch": torch,
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model_builder": builder,
        "sam3.model.sam3_image_processor": processor,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return log


def test_sam3_load_segmenter_builds_from_the_verified_file(tmp_path, fake_sam3):
    manifest = _manifest("sam3", "3.1", "sam3.1_multiplex.pt")
    bundle = _bundle(tmp_path / "b", "sam3", "3.1", "sam3.1_multiplex.pt")

    seg = sam3_loader.load_segmenter(
        LocalDirStore(bundle), cache_dir=tmp_path / "c", manifest=manifest
    )
    # Same cache path fishsense-lite's checkpoint_cache produced, so the
    # `checkpoint` string it stamps on predictions is unchanged by the move.
    assert fake_sam3["checkpoint_path"] == str(
        tmp_path / "c" / "sam3" / "3.1" / "sam3.1_multiplex.pt"
    )
    assert fake_sam3["eval"] is True
    assert seg.model is not None


def test_sam3_without_a_gpu_fails_before_downloading(tmp_path, fake_sam3):
    fake_sam3["cuda"] = False

    class NeverCalled:
        def download_to(self, ref, dest):
            pytest.fail("fetched a 3.5 GB checkpoint on a host that cannot run it")

    with pytest.raises(sam3_loader.Sam3RequiresGpu):
        sam3_loader.load_segmenter(
            NeverCalled(),
            cache_dir=tmp_path,
            manifest=_manifest("sam3", "3.1", "sam3.1_multiplex.pt"),
        )


def test_sam3_is_not_in_the_shipped_manifest_yet(tmp_path, fake_sam3):
    # Pinned so adding the hash later is a deliberate, visible change.
    with pytest.raises(KeyError, match="sam3"):
        sam3_loader.load_segmenter(None, cache_dir=tmp_path)
