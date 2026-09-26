"""Load model weights by name and version, verified against a pinned manifest.

fishsense-core is the one place models are identified. Every weight file it
knows is listed in ``models.toml`` (compiled into the Rust crate, read here
through ``_native``) with its sha256 and size, per *target* (``server``,
``mobile-coreml``). A caller asks for ``(name, version)``; ``version=None``
means the pinned default, so the model a result used follows from
``core_version``.

Where the bytes come from is **injected**, not decided here: :func:`fetch`
takes a :class:`WeightStore`, anything with
``download_to(ref, dest)``. Core ships the stores that need no credentials:

* :class:`LocalDirStore`, a directory in the cache layout. Used for bundles,
  offline machines, and tests.
* :class:`HuggingFaceStore`, the public mirror a manifest entry names, at its
  pinned revision.

A store that needs credentials (the Garage ``model-weights`` bucket, MLflow)
lives with whoever holds them, and core never builds one.

Whatever the store, nothing unverified is loaded: :func:`fetch` checks the size
and sha256 before it moves a download into place, and a mismatch raises
:class:`ModelIntegrityError`. With ``store=None`` it is purely offline, and a
cache miss raises :class:`ModelUnavailable` at once rather than reaching for
the network.

The cache layout is ``cache_dir/{name}/{version}/{filename}``, plus a
``{filename}.sha256`` stamp written after verification. A hit needs both, so a
multi-GB checkpoint is not re-hashed on every cold start; ``verify_cached=True``
re-hashes anyway.

``python -m fishsense_core.models prefetch --dest DIR`` fills ``DIR`` in that
layout (for example the models a mobile build bundles); :class:`LocalDirStore`
reads it back.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

__all__ = [
    "MOBILE_COREML",
    "SERVER",
    "HuggingFaceStore",
    "LocalDirStore",
    "Manifest",
    "ModelIntegrityError",
    "ModelRef",
    "ModelUnavailable",
    "WeightStore",
    "builtin_manifest",
    "fetch",
    "prefetch",
]

SERVER = "server"
MOBILE_COREML = "mobile-coreml"


class ModelIntegrityError(ValueError):
    """The bytes a store produced are not the file the manifest pins."""


class ModelUnavailable(RuntimeError):
    """The model cannot be had from here: not cached, and no store that has it."""


@dataclass(frozen=True)
class ModelRef:
    """One weight file: a resolved ``(name, version, target)``."""

    name: str
    version: str
    target: str
    filename: str
    sha256: str
    size: int
    #: Public places the bytes can be fetched from, e.g.
    #: ``{"huggingface": {"repo": ..., "revision": ...}}``. Stores may use
    #: these; the hash still decides.
    origins: Mapping[str, Mapping[str, str]] = field(default_factory=dict, compare=False)

    @property
    def id(self) -> str:
        """Provenance string, ``name/version@sha256[:12]``. It matches the Rust
        side (``ModelRef::id``), so a phone's measurement and a server's name a
        model the same way."""
        return f"{self.name}/{self.version}@{self.sha256[:12]}"


class Manifest:
    """Parsed ``models.toml``. The shipped one is validated by the Rust test
    suite; this parser checks only what it needs to resolve."""

    def __init__(self, refs: Sequence[ModelRef], defaults: Mapping[str, str]):
        self._refs = tuple(refs)
        self._defaults = dict(defaults)

    @classmethod
    def parse(cls, text: str) -> "Manifest":
        """Parse manifest TOML (the ``models.toml`` schema)."""
        refs: list[ModelRef] = []
        defaults: dict[str, str] = {}
        for model in tomllib.loads(text).get("model", []):
            name, version = model["name"], model["version"]
            if model.get("default"):
                defaults[name] = version
            origins = model.get("origin", {})
            for artifact in model.get("artifact", []):
                for target in artifact["targets"]:
                    refs.append(
                        ModelRef(
                            name=name,
                            version=version,
                            target=target,
                            filename=artifact["filename"],
                            sha256=artifact["sha256"].lower(),
                            size=int(artifact["size"]),
                            origins=origins,
                        )
                    )
        return cls(refs, defaults)

    @property
    def refs(self) -> tuple[ModelRef, ...]:
        """Every resolvable artifact."""
        return self._refs

    def resolve(self, name: str, version: str | None = None, target: str = SERVER) -> ModelRef:
        """The artifact for ``(name, version, target)``. ``version=None`` is the
        pinned default. Anything unknown is a ``KeyError`` that lists what is
        known. It never falls back to another version or target."""
        if version is None:
            version = self._defaults.get(name)
        for ref in self._refs:
            if (ref.name, ref.version, ref.target) == (name, version, target):
                return ref
        known = ", ".join(f"{r.name}/{r.version} ({r.target})" for r in self._refs)
        raise KeyError(
            f"unknown model {name}/{version or '<default>'} for target {target}; known: {known}"
        )


_BUILTIN: Manifest | None = None


def builtin_manifest() -> Manifest:
    """The manifest compiled into this build's ``_native``."""
    global _BUILTIN  # pylint: disable=global-statement
    if _BUILTIN is None:
        from fishsense_core import _native  # pylint: disable=import-outside-toplevel

        _BUILTIN = Manifest.parse(_native.models.manifest_toml())  # pylint: disable=c-extension-no-member
    return _BUILTIN


# ── stores ──────────────────────────────────────────────────────────────────


class WeightStore(Protocol):  # pylint: disable=too-few-public-methods
    """A backend that can produce a model file. Verification is
    :func:`fetch`'s job, not the store's."""

    def download_to(self, ref: ModelRef, dest: Path) -> None:
        """Write ``ref``'s bytes to ``dest`` (a temporary path in the
        destination directory), or raise."""


class LocalDirStore:  # pylint: disable=too-few-public-methods
    """Reads ``root/{name}/{version}/{filename}``, the layout :func:`fetch`
    and :func:`prefetch` write. It never touches the network."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)

    def download_to(self, ref: ModelRef, dest: Path) -> None:
        """Copy the file from ``root``; :class:`ModelUnavailable` if absent."""
        src = self.root / ref.name / ref.version / ref.filename
        if not src.is_file():
            raise ModelUnavailable(f"{ref.id}: not in {self.root} (looked for {src})")
        shutil.copyfile(src, dest)


def _hf_hub_download() -> Any:
    """``huggingface_hub.hf_hub_download``, imported only when used."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "HuggingFaceStore requires huggingface_hub. Install the optional "
            "extra: pip install 'fishsense_core[laser-detector]'"
        ) from exc
    return hf_hub_download


class HuggingFaceStore:  # pylint: disable=too-few-public-methods
    """Fetches from the Hugging Face repo and **pinned revision** a manifest
    entry names under ``[model.origin.huggingface]``."""

    def download_to(self, ref: ModelRef, dest: Path) -> None:
        """Download ``ref`` at its pinned revision into ``dest``."""
        origin = ref.origins.get("huggingface")
        if not origin:
            raise ModelUnavailable(f"{ref.id}: manifest names no huggingface origin")
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            got = _hf_hub_download()(
                origin["repo"], ref.filename, revision=origin["revision"], local_dir=tmp
            )
            shutil.move(got, dest)


# ── fetch ───────────────────────────────────────────────────────────────────

_LOCKS: dict[tuple[str, ...], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: tuple[str, ...]) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _check(path: Path, ref: ModelRef) -> None:
    size = path.stat().st_size
    if size != ref.size:
        raise ModelIntegrityError(f"{ref.id}: expected {ref.size} bytes, got {size}")
    with path.open("rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    if got != ref.sha256:
        raise ModelIntegrityError(f"{ref.id}: expected sha256 {ref.sha256}, got {got}")


def _stamp_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _cached(path: Path, ref: ModelRef, verify_cached: bool) -> bool:
    stamp = _stamp_path(path)
    try:
        if stamp.read_text().strip() != ref.sha256 or path.stat().st_size != ref.size:
            return False
    except FileNotFoundError:
        return False
    if verify_cached:
        try:
            _check(path, ref)
        except ModelIntegrityError:
            stamp.unlink(missing_ok=True)
            return False
    return True


def fetch(  # pylint: disable=too-many-arguments
    name: str,
    version: str | None = None,
    *,
    store: WeightStore | None,
    cache_dir: str | os.PathLike[str],
    target: str = SERVER,
    verify_cached: bool = False,
    manifest: Manifest | None = None,
) -> Path:
    """Return a local path to the verified weights for ``(name, version, target)``.

    Args:
        name: Manifest model name, e.g. ``"laser-detector"``.
        version: Manifest version; ``None`` is the pinned default.
        store: Where to get the bytes on a cache miss. ``None`` means offline:
            a miss raises :class:`ModelUnavailable` without any network access.
        cache_dir: Root of the ``{name}/{version}/{filename}`` cache.
        target: ``"server"`` (default) or ``"mobile-coreml"``.
        verify_cached: Re-hash a cache hit instead of trusting its stamp.
        manifest: Override the built-in manifest (tests).

    Raises:
        KeyError: ``(name, version, target)`` is not in the manifest.
        ModelIntegrityError: The store's bytes are not the pinned file.
        ModelUnavailable: Not cached, and ``store`` is ``None`` or lacks it.
    """
    ref = (manifest or builtin_manifest()).resolve(name, version, target)
    cache_dir = Path(cache_dir)
    path = cache_dir / ref.name / ref.version / ref.filename

    with _lock_for((str(cache_dir.resolve()), ref.name, ref.version, ref.filename)):
        if _cached(path, ref, verify_cached):
            return path
        if store is None:
            raise ModelUnavailable(
                f"{ref.id} ({ref.target}) is not in {cache_dir} and no store was "
                "given. Populate it while online with `python -m "
                f"fishsense_core.models prefetch --dest {cache_dir}`, or pass a store."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{ref.filename}.{os.getpid()}.{threading.get_ident()}.partial")
        try:
            store.download_to(ref, partial)
            _check(partial, ref)
            os.replace(partial, path)
            stamp_tmp = partial.with_suffix(".stamp")
            stamp_tmp.write_text(ref.sha256 + "\n")
            os.replace(stamp_tmp, _stamp_path(path))
        finally:
            partial.unlink(missing_ok=True)
    return path


def prefetch(
    dest: str | os.PathLike[str],
    *,
    store: WeightStore,
    target: str = SERVER,
    names: Sequence[str] | None = None,
    manifest: Manifest | None = None,
) -> list[Path]:
    """Fetch the default version of every model that has a ``target`` artifact
    (or only ``names``) into ``dest``, verified. Afterwards ``LocalDirStore(dest)``,
    or ``fetch(..., store=None, cache_dir=dest)``, serves them offline.

    This is the bundling step for an offline device: run it where there is
    network, and ship ``dest``.
    """
    manifest = manifest or builtin_manifest()
    wanted = []
    for ref in manifest.refs:
        if ref.target != target or (names is not None and ref.name not in names):
            continue
        try:
            default = manifest.resolve(ref.name, None, target)
        except KeyError:
            continue
        if default == ref and ref not in wanted:
            wanted.append(ref)
    return [
        fetch(r.name, r.version, store=store, cache_dir=dest, target=target, manifest=manifest)
        for r in wanted
    ]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fishsense_core.models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefetch", help="download and verify pinned models into a directory")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--target", default=SERVER, choices=[SERVER, MOBILE_COREML])
    p.add_argument("--name", action="append", dest="names", help="limit to this model (repeatable)")
    p.add_argument(
        "--from-dir",
        type=Path,
        help="read from a directory in the cache layout instead of Hugging Face",
    )
    sub.add_parser("list", help="print the manifest's artifacts")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for r in builtin_manifest().refs:
            print(f"{r.id}\t{r.target}\t{r.filename}\t{r.size}")
        return 0

    store: WeightStore = LocalDirStore(args.from_dir) if args.from_dir else HuggingFaceStore()
    for path in prefetch(args.dest, store=store, target=args.target, names=args.names):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
