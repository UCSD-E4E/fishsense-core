# Plan: move model loading into fishsense-core

Status: **the fishsense-core steps (2, 3, 3b) are built** on branch
`worktree-model-loader-plan`.
- **Still open:** the storage decision (fishsense-services PLAN.md §9.12: Garage
  `model-weights` vs MLflow). Only the worker-side adapter (step 4) depends on it.
- **Not done yet:** the fishsense-lite steps (4-6), which need a released core wheel;
  mobile step 3c, in its own repo; and the SAM 3.1 manifest entry, which waits on the
  step 1 hash.

Where the build departs from the text below:
- **No `embed-fishial` feature in this release.** Mobile builds with
  `default-features = false`, so a default-on feature would silently drop the embedded
  model the moment mobile bumps its pin. The gate lands together with mobile's bump
  (step 3c), as a `feat!`. Embedding stays unconditional for now, and
  `FishSegmentation::from_source` gives mobile the bundle path without the gate.
- **The build override is `FISHSENSE_FISHIAL_MODEL`,** not `FISHIAL_MODEL_PATH`. That
  name is already the rustc-env that `build.rs` emits for `include_bytes!`.
- **`WeightStore.download_to` takes the resolved `ModelRef`,** not
  `(name, version, filename)`. That lets `HuggingFaceStore` read the pinned origin from
  the manifest entry.
- **`prefetch` is Python** (`python -m fishsense_core.models prefetch`), not a Rust
  binary, because the stores are Python. Mobile's `rust-bridge/build.py` already runs
  under uv.
- **`LaserDetector.from_pretrained` is unchanged** and still defaults to `main`. The
  pinned path is the new `from_store(HuggingFaceStore(), ...)`.

Paths below are relative to `fishsense-lite` @ `a8b2c3bc` (main).
W = `services/fishsense-data-processing-workflow-worker/src/fishsense_data_processing_workflow_worker`,
S = `libs/fishsense-shared/src/fishsense_shared`.

## 1. Where things live today

Only SAM 3.1 goes through the `model-weights` bucket. The laser checkpoint is baked into
the image from Hugging Face `main`, with no revision pin. **Nothing anywhere verifies a
checksum.**

| What | Where |
|---|---|
| Laser checkpoint path (raw `os.environ`, not in Dynaconf) | `W/activities/predict_laser_image.py:36-38` — `E4EFS_LASER_DETECTOR__CHECKPOINT`, default `/e4efs/models/run3_epoch_021.pt` |
| Laser checkpoint fetch (build time, unpinned `main`, no hash) | `services/fishsense-data-processing-workflow-worker/Dockerfile:85-87` (`curl` from HF) |
| Laser load + singleton | `predict_laser_image.py:48-79` (`_load_detector` → `fishsense_core.laser.LaserDetector.from_checkpoint`; double-checked lock) |
| Laser behaviour version | `S/laser_predictor.py:52` `LASER_PREDICTOR_VERSION = 2`; tag `:55-65`; stamped with `checkpoint=basename(...)` at `predict_laser_image.py:225-227` |
| SAM 3.1 weight constants | `W/config.py:117-120` (Dynaconf defaults: `sam3.cache_dir`, `model_name="sam3"`, `model_version="3.1"`, `checkpoint_filename="sam3.1_multiplex.pt"`); prod copy in `deploy/k8s/data-worker/settings.toml:98-102` |
| SAM 3.1 code pin | worker `pyproject.toml:21` (`sam3 @ git+…@8e451d5`) |
| SAM 3.1 fetch + cache | `W/checkpoint_cache.py:49` `checkpoint_path`, `:56` `ensure_checkpoint` (per-key asyncio lock, `.partial` + `os.replace`, no size or hash check); called at `W/activities/predict_headtail_image.py:555-563` |
| SAM 3.1 load + singleton | `predict_headtail_image.py:97-171` (`_load_segmenter`, `get_segmenter`; raises non-retryable `NoGpuForSam3`) |
| Head/tail behaviour version | `S/headtail_predictor.py:70` `HEADTAIL_PREDICTOR_VERSION = 2`, `:98` fallback `-1`; stamped at `predict_headtail_image.py:570-573` with `checkpoint=str(full local cache path)` |
| Storage client | `S/object_store.py:98` `model_key`, `:160` `open_client`, `:196` `BaseObjectStoreClient`; `W/object_store.py:89` `download_model(name, version, filename) -> bytes` (whole object in memory) |
| Credentials / buckets | `W/config.py:76-97`; prod `deploy/k8s/data-worker/settings.toml:65-75` (`models_bucket = "model-weights"`); keys from Secret `fishsense-data-worker-secrets` (`deployment-gpu.yaml:136-145`, `deployment-gpu-cpu-fallback.yaml:98-107`) |
| Cache volume | `/e4efs/cache` is an **emptyDir** (12Gi on the GPU deployments), so every cold start downloads again |

Core already has half of the laser side: `python/fishsense_core/fishsense_core/_laser_detector.py`
has `DEFAULT_HF_REPO`/`DEFAULT_CHECKPOINT` (`:103-104`), a `CHECKPOINT_SHA256` identity table
(`:121-126`), `from_checkpoint` (`:650`) and `from_pretrained` (`:738`, HF, `revision="main"`).
For the laser model, the move is about **who fetches the file**, not about moving model code.

**The Rust side is what mobile runs.**
- `rust/fishsense-core/build.rs` downloads FishIAL at compile time from
  `https://huggingface.co/ccrutchf/fishial/resolve/main/fishial.onnx`. That URL points at
  `main`, not a pinned revision, and the download has no checksum and no local-path
  override.
- `src/fish/fish_segmentation.rs:23` embeds the result with `include_bytes!`.
- So at run time the crate needs no network, but *building* it does, and which bytes get
  embedded depends on whatever HF `main` holds on build day.
- **fishsense-mobile** (read @ `c97c867`) is a Flutter app, iOS first. It calls the
  crate through a Rust `staticlib` in `rust-bridge/`.
  - `rust-bridge/Cargo.toml:16` pins
    `fishsense-core = { git, tag = "fishsense_core-v2.0.0", default-features = false, features = ["coreml"] }`.
    So FishIAL reaches the phone embedded in the app binary, through core's `build.rs`.
    FishIAL is the only model mobile runs.
  - `rust-bridge/src/lib.rs:19-25` loads it once with
    `LazyLock` → `FishSegmentation::new()` → `load_model().expect(...)`. A load failure is
    a **panic** inside the FFI library, not an error the app can show.
  - The `photos` table (`lib/database.dart:82-122`, schema v7) has **no model or core
    version column**, and there is no sync code yet.
  - The ONNX Runtime and OpenCV xcframeworks come from `fishsense-mobile-thirdparty`
    `releases/latest` (`rust-bridge/build.py:27-29`), which is unpinned. That is outside
    this plan, but it is the same reproducibility gap for the runtime itself.
  - Building the app needs network access, but using it doesn't. Measuring on the phone
    already works offline today; what is missing is **pinning, verification and
    provenance**.

Out of scope: the retired slate `BoardMasker`.

## 2. API added to fishsense-core

New module `fishsense_core.models`. It is pure Python with no new base dependencies.

```python
@dataclass(frozen=True)
class ModelRef:            # one entry of the manifest
    name: str              # "sam3", "laser-detector"
    version: str           # "3.1", ...
    filename: str          # "sam3.1_multiplex.pt"
    sha256: str            # content hash, the actual identity
    size: int              # cheap sanity check before hashing

class WeightStore(Protocol):
    """Backend. Injected; core never builds one with credentials."""
    def download_to(self, name: str, version: str, filename: str, dest: Path) -> None: ...

def resolve(name: str, version: str | None = None) -> ModelRef        # None -> pinned default
def fetch(name, version=None, *, store: WeightStore, cache_dir: Path,
          verify_cached: bool = False) -> Path
```

- **Manifest: one file, owned by the Rust crate.** It lives at
  `rust/fishsense-core/models.toml`. Rust compiles it in with `include_str!`, and Python
  reads the same bytes through `_native.models.manifest_toml()` and parses them with
  `tomllib`. That leaves one copy, not two that can drift apart.
  - Each `(name, version)` lists one **artifact per target** (`server`, `mobile-coreml`,
    and so on), each with `filename`, `sha256` and `size`. This is PLAN §4.7's "one
    model_version, two builds".
  - Each name also has a `default` version.
  - `ModelRef` gains a `target` field, which defaults to `server` in Python.
  - Core pins the version: the model a result used follows from `core_version`, which is
    what PLAN §4.7's reproducibility needs.
  - `_laser_detector.CHECKPOINT_SHA256` gets derived from the manifest, so the laser
    hashes have a single source.
- **Verification.** `fetch` streams into `dest.partial`, checks the size, then the sha256,
  then does `os.replace`. On a mismatch it deletes the partial file and raises
  `ModelIntegrityError(expected, got)`. An unknown name or version raises `KeyError` naming
  the known ones. Nothing unverified is ever loaded.
- **Caching.** The cache layout is `cache_dir/{name}/{version}/{filename}`, the same as v1's
  `checkpoint_path`, so the switch does not move files. After the hash passes, `fetch`
  writes a `{filename}.sha256` stamp. A cache hit requires both the file and a stamp that
  matches the manifest, so a SAM-sized file is not hashed again on every start;
  `verify_cached=True` hashes it again anyway. There is a per-key `threading.Lock` in the
  process. Across processes, the atomic rename makes a duplicate download harmless. `fetch`
  is sync (core is sync), and the worker calls it through `asyncio.to_thread`, as it does
  today.
- **Stores shipped in core:** these hold no credentials.
  - `LocalDirStore(root)`: the same layout, read from a directory. Used for bundled or
    offline mobile use, for tests, and as the replacement for the laser env-var override.
  - `HuggingFaceStore(repo, revision)`: behind the existing `laser-detector` extra.
    `from_pretrained` keeps its signature and delegates to it, so it now gets hash-checked.
  - No S3 or MLflow store lives in core. `WeightStore` is a four-line protocol, and the
    backend adapter lives with whoever holds the credentials.
- **Model constructors** take a ref plus a store:
  - `LaserDetector.from_store(store, *, version=None, cache_dir, **kw)`, which calls
    `fetch` and then `from_checkpoint`.
  - `fishsense_core.fish.sam3.load_segmenter(store, *, version=None, cache_dir, device)`,
    which calls `fetch` and then `build_sam3_image_model` + `Sam3Processor`. The `sam3`
    import is lazy. The CUDA check moves in with it, as a plain `RuntimeError` subclass;
    the worker maps it to its non-retryable `ApplicationError`.
  - **`sam3` is not declared** as a core dependency or extra. It is git-only, and a bare
    git dependency in `Requires-Dist` is exactly what makes the current wheels
    uninstallable (see the `rectified` extra's comment in `pyproject.toml`). The worker
    keeps its `sam3` git pin. Core documents the commit it was tested against.

## 2b. Offline mobile (no Internet at run time)

**Rule:** on a phone, the registry is a **build-time** input only. At run time, core must
never make a network call, not even as a fallback. Every model the app runs ships in the
signed app bundle.

- **Which models are in scope.** Mobile runs the Rust crate through the CoreML execution
  provider, never the Python package. Today only FishIAL (ONNX) can run there. The laser
  `.pt` and the multi-GB SAM 3.1 checkpoint cannot, until the model-shrinking work in
  PLAN §4.7 produces `mobile-coreml` artifacts. The manifest has room for those artifacts
  now; the entries stay empty until they exist.
- **Rust API** (`src/models/`, adding only the `sha2` dependency):
  - `Manifest::builtin()`, which parses the manifest compiled into the crate.
  - `ModelSource::{Embedded, Bytes(&[u8]), Path(PathBuf)}`.
  - `verify(&ArtifactRef, &[u8]) -> Result<(), IntegrityError>`.
  - `FishSegmentation::from_source(ModelSource)`. `FishSegmentation::new()` stays and
    means `Embedded`.
  - Mobile hands in the path of a bundle resource. Core checks the bytes against the
    manifest entry for `(name, version, target)` and records which model it loaded.
- **Loaded models expose their identity.** Every loaded model can return
  `name/version/target@sha256[:12]`. Mobile writes that into each measurement, and
  the server reads the same string after the measurement syncs. Provenance then has the
  same format whether the measurement was made on a phone or on the server.
- **How a bundled path reaches the crate on mobile.**
  - The bundled directory is a Flutter asset, or an iOS bundle resource.
  - The platform side resolves it to a path once, at start-up, and passes it into the
    bridge's init call (`init_models(model_dir)`).
  - The bridge then builds a `LocalDirStore`-equivalent `ModelSource::Path` for each
    model.
  - Core never guesses platform paths. That keeps the same crate usable when Android
    lands.
- **Embedding vs. bundle resource.**
  - Embedding FishIAL today forces it into every consumer (the Python wheels included),
    and each new mobile model would grow the binary for the server too.
  - This plan keeps embedding as a default-on `embed-fishial` feature, so nothing
    breaks. Mobile can turn it off and ship the `.onnx` (later the `mobile-coreml` build)
    as a bundle resource instead. That is what PLAN §4.7's "consumers pull selectively"
    requires.
- **Building offline too.** `build.rs` pins an HF revision and checks the manifest
  sha256 after downloading. It also honours `FISHIAL_MODEL_PATH`, and when that is set it
  skips the network. Mobile CI and air-gapped builds then use a pre-fetched file, and a
  change to HF `main` fails the build rather than silently changing the embedded model.
- **Bundling step (mobile CI, which does have network).** A
  `fishsense-core-models prefetch --target mobile-coreml --dest <dir>` command pulls the
  pinned artifacts through the same `WeightStore` adapters and verifies them into
  `dir/{name}/{version}/{filename}`, the layout `LocalDirStore` reads. The app bundles
  that directory. I lean towards shipping this command as a small Rust binary, but a
  Python entry point would also work.
- **Hashing on the phone.** The iOS bundle is code-signed, so the full-hash check belongs
  in the bundling step. At run time, `verify` checks only the size by default. A full
  hash is opt-in, so a large model isn't re-hashed on every cold start.
- **Python offline.** This covers laptops in the field and the Python mobile tooling.
  - `fetch` against `LocalDirStore`, or against the cache alone (`store=None`), is a
    fully offline path.
  - If nothing is cached, it raises `ModelUnavailable(name, version, target)` straight
    away, never retries or hangs, and the message names the `prefetch` command.

## 3. What stays in the worker

- Credentials, the endpoint, bucket names (`models_bucket`, `models_prefix`), Dynaconf,
  and the boto3 client (`open_client`, `ObjectStoreClient`).
- A roughly 15-line adapter, `GarageWeightStore(ObjectStoreClient)`, that implements
  `download_to` on top of `model_key` and the models bucket. If MLflow is chosen, it
  becomes `MlflowWeightStore`, which maps `(name, version)` to a registered-model version
  and calls `mlflow.artifacts.download_artifacts`. Core is untouched either way.
- `cache_dir` (the emptyDir mount).
- The **behaviour** versions `LASER_PREDICTOR_VERSION` and `HEADTAIL_PREDICTOR_VERSION`,
  and the crop, gate and region constants. S's docstring says these version the
  behaviour, not the checkpoint, and the behaviour they cover (cropping, gating, the
  region polygon) is still worker code. They move when that logic moves (§5 of the
  services plan). The weight identity is what moves now.
- Temporal concerns: the per-process singletons, error mapping and role placement.

## 4. How the move lands (v1 green at every step)

Each step is one PR that can ship alone. Every consumer keeps working between steps.

1. **Ops, no code: pin the hashes and put both models in the bucket.**
   - Hash `model-weights/sam3/3.1/sam3.1_multiplex.pt`.
   - Upload the laser checkpoint to `model-weights/laser-detector/<version>/run3_epoch_021.pt`.
     Its sha256 must equal core's existing `bd3ab8f5…`.
   - Nothing reads the new object yet.
2. **core `feat(models)`**: the manifest, `ModelRef`, `WeightStore`, `fetch`,
   `LocalDirStore`, and `HuggingFaceStore` under `from_pretrained`. This is purely
   additive and is released as a minor bump. v1 is still on v4.0.0 and is unaffected.
3. **core `feat`**: `LaserDetector.from_store` and `fish.sam3.load_segmenter`. This is
   additive too, and `from_checkpoint`/`from_pretrained` stay.
3b. **core, Rust side for mobile.** This is independent of steps 4-6 and can land in
   parallel.
   - Add the `models.toml` manifest with its FishIAL entry.
   - Pin the HF revision and check the sha256 in `build.rs`, and honour
     `FISHIAL_MODEL_PATH`.
   - Add `src/models/`, `FishSegmentation::from_source`, and the `embed-fishial` feature
     (on by default).
   - The default build embeds the same bytes as before, and the new sha check proves it,
     so both the wheels and mobile's existing pin are unaffected.
3c. **fishsense-mobile** (its own repo, after 3b is released):
   - Bump the `rust-bridge` pin off `v2.0.0`.
   - Replace `load_model().expect(...)` with a `Result` returned over the method channel,
     so a failed load or verification shows as an error in the app instead of a crash.
   - Add a schema v8 column `model_id TEXT` (and `core_version TEXT`) to `photos`,
     filled from the loaded model's identity. Rows from before v8 stay `NULL`, meaning
     "FishIAL embedded by v2.0.0". That is knowable, so the migration could backfill it.
   - Optionally, turn off `embed-fishial` and bundle FishIAL through `prefetch`.
   - The laser and SAM `mobile-coreml` artifacts are added to the manifest when they
     exist. That needs a manifest change and no API change.
4. **lite, SAM 3.1 first:**
   - Bump the core pin and add `GarageWeightStore`.
   - Replace `ensure_checkpoint` + `_load_segmenter` with `load_segmenter(store, cache_dir=…)`.
   - The cache path is unchanged, so the stamped `checkpoint` string on
     `headtailprediction` stays byte-identical.
   - Keep the `[sam3]` Dynaconf keys for one release, with a validator that fails if they
     disagree with the core manifest.
   - SAM goes first because it is already on the bucket and needs no deploy change.
5. **lite, laser:**
   - `_load_detector` calls `LaserDetector.from_store(GarageWeightStore(...))`.
   - `E4EFS_LASER_DETECTOR__CHECKPOINT` becomes a `LocalDirStore` override for dev.
   - The stamped `checkpoint` stays `run3_epoch_021.pt`, so `LASER_PREDICTOR_VERSION` does
     **not** bump: the checkpoint and the behaviour are the same.
   - Before merging, confirm that the data worker's Garage key can **read `model-weights`**
     on the `gpu` **and** `gpu-cpu-fallback` Deployments; the fallback also runs the laser
     detector. Keep the Dockerfile `curl` in this PR, so a rollback is only a pin revert.
6. **lite cleanup:** remove the Dockerfile `curl` and `/e4efs/models`, `checkpoint_cache.py`
   and its tests, and the `[sam3]` config block.
7. **After §9.12 is decided:** only the adapter changes (or a second one is added), plus a
   manifest bump if artifacts are re-published. With MLflow, the manifest sha256 still
   gates what gets loaded.

## 5. Tests written first

**In core** (`python/fishsense_core/fishsense_core/models/test_*.py`). These tests use a
fake in-memory store and need no network and no torch:

- `fetch` downloads when the file is absent, and the store is called once with
  `(name, version, filename)`.
- A second `fetch` is a cache hit with zero store calls.
- The cache path is exactly `cache_dir/name/version/filename`. This pins v1
  compatibility; it is the same assertion as the worker's `test_checkpoint_cache.py:47`.
- **Checksum mismatch:** it raises `ModelIntegrityError`, and neither the final file nor
  a `.partial` is left behind.
- A size mismatch is rejected before hashing.
- A file cached without a stamp, or with a wrong stamp, is fetched again.
  `verify_cached=True` catches a tampered file whose stamp is still good.
- N threads fetching the same ref produce one download.
- An unknown name or version gives a `KeyError` that lists the known ones.
  `version=None` resolves to the manifest default.
- Manifest well-formedness: every sha is 64 hex characters, every size is > 0, and every
  name has a default.
- `_laser_detector.CHECKPOINT_SHA256` is derived from the manifest: `run3` maps to
  `bd3ab8f5…`.
- `LocalDirStore` fetches with sockets monkeypatched to raise, which proves the offline
  path.
- `HuggingFaceStore`: stub `hf_hub_download` and check that `revision` is passed through
  and that a wrong-hash download is rejected.

**Offline and mobile, in core:**

- **Rust:** the embedded FishIAL bytes hash to the manifest entry. This catches HF `main`
  drifting under the build. It needs no linear algebra, so it runs in the scratch crate
  too.
- **Rust:** `verify` rejects bytes that are wrong, and bytes of the wrong size.
  `from_source(Path)` loads from a local file.
- **Rust:** looking up an unknown `(name, version, target)` returns an error. It never
  falls back to another target.
- **CI:** `cargo tree -e normal --no-default-features --features coreml` contains no
  `reqwest`, `hyper` or `ureq`. This pins that the mobile build has no network code at
  run time; `reqwest` may only appear as a build dependency.
- **Python:** `_native.models.manifest_toml()` is byte-identical to the crate's
  `models.toml`. One source.
- **Python:** `fetch(store=None)` with an empty cache raises `ModelUnavailable` straight
  away. The test monkeypatches sockets to raise, so it also proves no network attempt is
  made.
- **`prefetch --target mobile-coreml` into a temp directory,** then a load of every
  artifact from that directory through `LocalDirStore` with the network off. This is the
  bundling round-trip.

**In lite** (before steps 4 and 5):

- `GarageWeightStore` reads from the models bucket and applies the models prefix. These
  are ports of `W/…/tests/test_object_store.py:125,146`.
- The stamped `checkpoint` is unchanged across the switch: the SAM 3.1 path string and the
  laser basename `run3_epoch_021.pt`.
- `test_version_history_names_the_pinned_checkpoint` (`test_checkpoint_cache.py:100-127`)
  is retargeted to the core manifest default for `sam3`.
- A missing CUDA still maps to the non-retryable `NoGpuForSam3`.

## 6. Decisions needed before code

1. **Storage backend (§9.12).** Garage `model-weights` or MLflow. This changes only the
   worker adapter and the step 1 publishing.
2. **The laser checkpoint's `version` label** in the bucket, e.g. `run3`. The existing
   `2` is the behaviour version, not the weights.
3. **Pin in core, or in config.** This plan pins in core, so changing a model means a core
   release, and `core_version` alone reproduces the model. The alternative keeps v1's
   config-driven version, validated against the manifest.
4. **Mobile packaging:** embed models in the binary, as today, or ship them as bundle
   resources.
   - This plan recommends bundle resources for any new model. FishIAL can stay embedded
     through `embed-fishial` until mobile's next core bump.
   - Either way, mobile's jump from v2.0.0 to v4+ brings in the breaking changes since
     then, among them non-landscape segmentation (#76). So it is its own PR in
     fishsense-mobile.
