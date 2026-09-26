"""Tests for fishsense_core.models: manifest, fetch/verify/cache, stores.

Everything here runs against a small injected manifest and in-memory stores,
so it needs no network, no torch, and (except where marked) no `_native`.
"""

import hashlib
import socket
import threading
from pathlib import Path

import pytest

from fishsense_core import models
from fishsense_core.models import (
    MOBILE_COREML,
    SERVER,
    HuggingFaceStore,
    LocalDirStore,
    Manifest,
    ModelIntegrityError,
    ModelUnavailable,
    fetch,
    prefetch,
)

WEIGHTS = b"pretend these are weights"
OTHER = b"pretend these are the v2 weights"
MOBILE = b"quantized"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


MANIFEST = Manifest.parse(f"""
[[model]]
name = "m"
version = "1"
default = true
[model.origin.huggingface]
repo = "org/m"
revision = "abc123"
[[model.artifact]]
targets = ["server"]
filename = "m.pt"
sha256 = "{_sha(WEIGHTS)}"
size = {len(WEIGHTS)}
[[model.artifact]]
targets = ["mobile-coreml"]
filename = "m.onnx"
sha256 = "{_sha(MOBILE)}"
size = {len(MOBILE)}

[[model]]
name = "m"
version = "2"
[[model.artifact]]
targets = ["server"]
filename = "m2.pt"
sha256 = "{_sha(OTHER)}"
size = {len(OTHER)}

[[model]]
name = "server-only"
version = "1"
default = true
[[model.artifact]]
targets = ["server"]
filename = "s.pt"
sha256 = "{_sha(WEIGHTS)}"
size = {len(WEIGHTS)}
""")

CONTENT = {
    ("m", "1", "m.pt"): WEIGHTS,
    ("m", "1", "m.onnx"): MOBILE,
    ("m", "2", "m2.pt"): OTHER,
    ("server-only", "1", "s.pt"): WEIGHTS,
}


class FakeStore:
    """Serves CONTENT (or an override) and records every download."""

    def __init__(self, override: bytes | None = None, delay: threading.Event | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self._override = override
        self._delay = delay
        self._lock = threading.Lock()

    def download_to(self, ref, dest: Path) -> None:
        with self._lock:
            self.calls.append((ref.name, ref.version, ref.filename))
        if self._delay is not None:
            self._delay.wait(timeout=5)
        body = self._override if self._override is not None else CONTENT[
            (ref.name, ref.version, ref.filename)
        ]
        dest.write_bytes(body)


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a connection fails the test."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _fetch(store, cache, **kw):
    return fetch("m", store=store, cache_dir=cache, manifest=MANIFEST, **kw)


# ── resolve ─────────────────────────────────────────────────────────────────


def test_version_none_resolves_to_the_default():
    assert MANIFEST.resolve("m").version == "1"
    assert MANIFEST.resolve("m", "2").filename == "m2.pt"


def test_unknown_model_is_a_key_error_naming_the_known_ones():
    with pytest.raises(KeyError, match=r"m/1 \(server\)"):
        MANIFEST.resolve("nope")


def test_missing_target_is_an_error_not_a_fallback():
    # server-only has no mobile artifact; resolving must not hand back the
    # server file.
    with pytest.raises(KeyError):
        MANIFEST.resolve("server-only", target=MOBILE_COREML)


def test_id_is_name_version_and_short_hash():
    assert MANIFEST.resolve("m").id == f"m/1@{_sha(WEIGHTS)[:12]}"


# ── fetch ───────────────────────────────────────────────────────────────────


def test_fetch_downloads_when_absent(tmp_path):
    store = FakeStore()
    path = _fetch(store, tmp_path)
    assert path.read_bytes() == WEIGHTS
    assert store.calls == [("m", "1", "m.pt")]


def test_second_fetch_is_a_cache_hit(tmp_path):
    store = FakeStore()
    _fetch(store, tmp_path)
    _fetch(store, tmp_path)
    assert len(store.calls) == 1


def test_cache_layout_is_name_version_filename(tmp_path):
    # Same layout as fishsense-lite's checkpoint_cache, so switching the
    # worker over re-uses (and re-stamps) what is already on disk.
    assert _fetch(FakeStore(), tmp_path) == tmp_path / "m" / "1" / "m.pt"


def test_hash_mismatch_raises_and_leaves_nothing_behind(tmp_path):
    tampered = bytes([WEIGHTS[0] ^ 0xFF]) + WEIGHTS[1:]  # same size
    with pytest.raises(ModelIntegrityError, match="sha256"):
        _fetch(FakeStore(override=tampered), tmp_path)
    assert list((tmp_path / "m" / "1").iterdir()) == []


def test_size_mismatch_raises(tmp_path):
    with pytest.raises(ModelIntegrityError, match="bytes"):
        _fetch(FakeStore(override=WEIGHTS + b"x"), tmp_path)
    assert list((tmp_path / "m" / "1").iterdir()) == []


def test_file_without_a_stamp_is_fetched_again(tmp_path):
    # e.g. a file left by the old checkpoint_cache, which never verified.
    target = tmp_path / "m" / "1" / "m.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(WEIGHTS)
    store = FakeStore()
    _fetch(store, tmp_path)
    assert len(store.calls) == 1


def test_verify_cached_catches_a_tampered_file_with_a_good_stamp(tmp_path):
    store = FakeStore()
    path = _fetch(store, tmp_path)
    path.write_bytes(bytes([WEIGHTS[0] ^ 0xFF]) + WEIGHTS[1:])  # stamp untouched
    assert _fetch(store, tmp_path) == path  # default: trusts the stamp
    assert len(store.calls) == 1
    _fetch(store, tmp_path, verify_cached=True)
    assert len(store.calls) == 2
    assert path.read_bytes() == WEIGHTS


def test_concurrent_fetches_download_once(tmp_path):
    gate = threading.Event()
    store = FakeStore(delay=gate)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(_fetch(store, tmp_path)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(timeout=10)
    assert len(results) == 8
    assert len(store.calls) == 1


# ── offline ─────────────────────────────────────────────────────────────────


def test_no_store_and_empty_cache_fails_immediately(tmp_path, no_network):
    with pytest.raises(ModelUnavailable, match="prefetch"):
        _fetch(None, tmp_path)


def test_no_store_serves_a_verified_cache(tmp_path, no_network):
    path = _fetch(FakeStore(), tmp_path)
    assert _fetch(None, tmp_path) == path


def test_local_dir_store_reads_the_cache_layout_offline(tmp_path, no_network):
    bundle = tmp_path / "bundle"
    (bundle / "m" / "1").mkdir(parents=True)
    (bundle / "m" / "1" / "m.pt").write_bytes(WEIGHTS)
    path = _fetch(LocalDirStore(bundle), tmp_path / "cache")
    assert path.read_bytes() == WEIGHTS


def test_local_dir_store_missing_file_is_unavailable(tmp_path):
    with pytest.raises(ModelUnavailable):
        _fetch(LocalDirStore(tmp_path / "empty"), tmp_path / "cache")


def test_prefetch_then_bundle_round_trip(tmp_path, no_network):
    """The mobile bundling step: pull the pinned mobile artifacts into a
    directory while online, then load them from that directory with the
    network off."""
    bundle = tmp_path / "bundle"
    paths = prefetch(bundle, store=FakeStore(), target=MOBILE_COREML, manifest=MANIFEST)
    # Only models with a mobile artifact, and only their mobile file.
    assert paths == [bundle / "m" / "1" / "m.onnx"]

    got = fetch(
        "m",
        target=MOBILE_COREML,
        store=LocalDirStore(bundle),
        cache_dir=tmp_path / "device-cache",
        manifest=MANIFEST,
    )
    assert got.read_bytes() == MOBILE


# ── Hugging Face store ──────────────────────────────────────────────────────


def test_hugging_face_store_uses_the_pinned_revision(tmp_path, monkeypatch):
    seen = {}

    def fake_download(repo_id, filename, *, revision, local_dir):
        seen.update(repo_id=repo_id, filename=filename, revision=revision)
        out = Path(local_dir) / filename
        out.write_bytes(WEIGHTS)
        return str(out)

    monkeypatch.setattr(models, "_hf_hub_download", lambda: fake_download)
    _fetch(HuggingFaceStore(), tmp_path)
    assert seen == {"repo_id": "org/m", "filename": "m.pt", "revision": "abc123"}


def test_hugging_face_store_wrong_bytes_are_rejected(tmp_path, monkeypatch):
    def fake_download(repo_id, filename, *, revision, local_dir):
        out = Path(local_dir) / filename
        out.write_bytes(OTHER[: len(WEIGHTS)].ljust(len(WEIGHTS), b"-"))
        return str(out)

    monkeypatch.setattr(models, "_hf_hub_download", lambda: fake_download)
    with pytest.raises(ModelIntegrityError):
        _fetch(HuggingFaceStore(), tmp_path)


def test_hugging_face_store_needs_an_origin(tmp_path):
    with pytest.raises(ModelUnavailable, match="origin"):
        fetch("m", "2", store=HuggingFaceStore(), cache_dir=tmp_path, manifest=MANIFEST)


# ── the shipped manifest ────────────────────────────────────────────────────


def test_builtin_manifest_is_the_crate_file_byte_for_byte():
    native = pytest.importorskip("fishsense_core._native")
    crate_file = Path(__file__).parents[3] / "rust" / "fishsense-core" / "models.toml"
    if not crate_file.exists():
        pytest.skip("not running from a source checkout")
    assert native.models.manifest_toml() == crate_file.read_text()


def test_builtin_manifest_pins_the_laser_checkpoints():
    pytest.importorskip("fishsense_core._native")
    ref = models.builtin_manifest().resolve("laser-detector")
    assert ref.filename == "run3_epoch_021.pt"
    assert ref.origins["huggingface"]["repo"] == "ucsde4e/fishsense-laser-detector"
