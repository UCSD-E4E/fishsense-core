# Segmentation improvements — Phase 1 diagnosis + Phase 2 proposal

Context: `FishSegmentation::inference` produces degenerate masks on some
images. The user pre-labeled the failures into two categories
(`tiny_mask`, `spurious_blob`) and provided a 519-case fixture at
`/home/chris/Repos/school/e4e/fishsense/2026-04-19_fishsense-mobile/scripts/bug_report/fixture_segmentation/`.

---

## Phase 1 — diagnosis

### Category distribution (headline result)

```
n_total_cases     = 519
n_tiny_mask       =   0    ← empty category
n_spurious_blob   =   7    ← ~1.3 % of the fixture
```

The `tiny_mask` bucket is **empty** in the current fixture. Everything
below refers to `spurious_blob`.

### `spurious_blob` statistics (from `index.json`)

Each case has exactly `n_components = 2` — i.e., every failure is
"primary blob + one extra blob", never "many blobs".

| case | total_px | largest_px | 2nd_px | largest_fraction |
|------|----------|------------|--------|------------------|
| 01   | 179,795  | 149,386    | 30,409 | 0.831            |
| 02   | 200,472  | 176,246    | 24,226 | 0.879            |
| 03   | 155,876  | 137,523    | 18,353 | 0.882            |
| 04   | 151,806  | 129,642    | 22,164 | 0.854            |
| 05   | 255,686  | 175,997    | 79,689 | 0.688            |
| 06   |  58,888  |  31,135    | 27,753 | 0.529            |
| 07   | 491,733  | 396,846    | 94,887 | 0.807            |

Two size regimes:

- **"Small fin/object" secondary** (~18–30 k px): cases 01–04, 06
- **"Whole other fish" secondary** (~80–95 k px): cases 05, 07

### Direct mask inspection — are these 2 detections or 1 fragmented detection?

Every `mask.npy` contains **exactly two distinct positive values (1 and 2)**
— one per detection the code drew. This is forced by the code: each
detection writes its polygon with `color = (ind + 1)`, and within a
single detection only the single largest contour is drawn. So a
multi-blob output *must* correspond to ≥2 Mask R-CNN detections above
the score threshold. Confirmed.

### Rust diagnostic binary

Added an additive `FishSegmentation::inference_debug` method (new public
API — does not change `inference`) that returns per-detection stats
alongside the mask. Driver at
`rust/fishsense-core/examples/seg_debug.rs`. Run it with:

```
cargo run -p fishsense-core --example seg_debug -- /tmp/seg_debug_input/case_0?_rgb.npy
```

**Results on all 7 fixture cases, current code + embedded model:**

| case | n_det | det_0 score / mask_px / final_px | det_1 score / mask_px / final_px |
|------|-------|-----------------------------------|-----------------------------------|
| 01   | 2     | 1.000 /  9,396 /  30,723          | 1.000 / 45,478 / 148,328         |
| 02   | 1     | 0.990 / 51,813 / 169,478          | —                                |
| 03   | 1     | 1.000 / 42,064 / 137,841          | —                                |
| 04   | 1     | 1.000 / 38,780 / 127,071          | —                                |
| 05   | 2     | 1.000 / 54,211 / 177,489          | 0.965 / 24,268 /  78,997         |
| 06   | 2     | 1.000 /  9,783 /  31,875          | 0.706 /  8,848 /  28,777         |
| 07   | 2     | 1.000 / 122,129 / 400,389         | 0.998 / 28,576 /  92,722         |

Five things fall out of this:

1. **Only 4 of 7 fixture cases reproduce.** Cases 02, 03, 04 — all
   originally 2 components per `index.json` — now produce a single
   detection. This is **Phase 1 finding the user didn't describe**:
   the fixture is either non-deterministic to regenerate, or was
   captured against a different model/preprocessing path than the one
   this crate currently builds. The surviving primary matches the
   old primary (e.g. case_03 old largest 137,523 px ≈ new 137,841 px),
   so preprocessing is close; the *secondary* detections are what
   vanished. Likely cause: ONNX-Runtime NMS is sensitive to tie-breaking
   on detections that hover near the NMS IoU threshold, and
   `with_intra_threads(4)` + Level3 optimization are not bit-exact
   across runs/environments. Worth confirming by regenerating the
   fixture from *this* binary.
2. **Spurious detections are high-score, not marginal.** Of 4 reproducing
   cases, spurious scores are 1.000, 0.965, 0.706, 0.998. Lowering the
   score threshold does nothing — these already clear 0.3 by a wide
   margin. Raising the threshold removes at most the 0.706 case (case
   06) and buys very little, at the cost of recall on legitimate
   low-confidence fish elsewhere in the dataset.
3. **Score ≈ 1.000 saturates.** Four of the six reproducing detections
   report score = 1.000 exactly. Mask R-CNN classifier output saturates;
   ranking by score is uninformative once multiple detections are
   "max-confident". Area is the better discriminator.
4. **Primary detection = largest-area detection, in every case.** There
   is no case in the fixture where the intended fish is smaller than
   the spurious blob. Largest-area selection cleanly picks the right
   one.
5. **What the spurious detections actually are (from the RGB +
   visualization overlays):**
   - case 05 (top-of-frame blob, 78 k px, score 0.965): a second fish
     cropped at the top of the frame.
   - case 07 (upper-right blob, 93 k px, score 0.998): a second tuna,
     unambiguously present in the scene.
   - case 01 (45 k px pasted → 148 k final — **this is actually the
     primary**, not the spurious one; the spurious is the 9 k pasted
     blob at 30 k final): a gloved hand / small orange target near the
     fish head. High-confidence false positive on hand + orange stop.
   - case 06 (other detection, 0.706): appears to be the gloved hand
     obscuring the fish head — the model ends up with two overlapping
     detections of the same fish, one anchored on the body/tail and
     one on the head.

   So: ~half the "spurious" detections are *actually other fish in the
   scene* (the model is correct at the instance level; the downstream
   consumer is expecting "one-fish-per-image"), and ~half are genuine
   false positives on hands, gloves, and fish-like deck objects.

### Verification of the user's code description

Walking through `fish_segmentation.rs::convert_output_to_mask`:

| claim | code says | status |
|-------|-----------|--------|
| score threshold 0.3 | `SCORE_THRESHOLD: f32 = 0.3`, `if scores[ind] <= SCORE_THRESHOLD { continue; }` | ✅ |
| mask threshold 0.5 | `MASK_THRESHOLD: f32 = 0.5`, `if v > MASK_THRESHOLD { 255 } else { 0 }` | ✅ |
| contours with <10 vertices dropped | `if poly.len() < 10 { continue; }` | ✅ |
| "largest contour" kept | `contour_vec.sort_by_key(\|v\| Reverse(v.len()))` → `contours.first()` | ⚠️ **largest by vertex count, not area**. Correlated for Mask R-CNN output but not identical. In practice fine; flagging it for precision. |
| output encodes instance IDs | `color = (ind + 1) as i32`; one polygon per detection | ✅ |
| no NMS, no overlap merging, no min-instance-area filter | correct — none of those appear in `convert_output_to_mask` | ✅ |
| no "keep best instance" reduction | correct — every surviving detection is drawn | ✅ |

Additional things worth knowing that weren't in the user's description:

- If `do_inference` returns `Err(_)` (which the comment says happens
  when the model crashes on no-fish inputs), `inference()` **silently
  returns an all-zeros mask**. So "no detection" and "model crashed"
  are indistinguishable to callers. A real `NoDetection` result type
  would fix this.
- `bitmap_to_polygon` can return `FishNotFound` (when the pasted mask
  has zero contours — e.g. mask thresholded to all-zero); that's
  swallowed with `continue`. Benign.
- Only the single most-vertex-rich contour of each detection is drawn.
  Any other contours (disconnected fragments inside a single detection)
  are silently discarded. In practice Mask R-CNN instance masks are
  contiguous, so this is not a real problem.

### Answers to the user's numbered Phase-1 questions

1. **Counts per category**: 0 `tiny_mask`, 7 `spurious_blob`, out of 519.
2. **`tiny_mask` distribution**: empty — nothing to analyze.
3. **`tiny_mask` cause**: N/A (empty category). If `tiny_mask` ever
   appears in future fixtures, the diagnostic binary now distinguishes
   "0 detections above 0.3" from "1 tiny detection".
4. **`spurious_blob`**: always exactly 2 components. Largest-fraction
   ranges 0.53–0.88. Spurious detections are high-confidence
   (0.706–1.000). About half are real second fish in frame; about half
   are high-confidence false positives on gloves / orange deck
   furniture / fish heads confused by hand occlusion.
5. **Code vs. description**: verified accurate with one minor
   clarification (contour selection is by vertex count, not area).

---

## Phase 2 — proposal

### `tiny_mask` category

**Recommendation: do nothing for now.** The fixture has zero cases.
Writing code for a failure mode that isn't currently happening is
speculative. Keep the existing behavior (return all-zero mask on
inference error, empty mask if no detection clears 0.3).

One small improvement worth keeping on the table for later, if a
`tiny_mask` case ever appears: distinguish "no fish" from "model
errored" with a proper `Option<Array2<u8>>` or `Result` variant. This
is a legibility change, not a recall/accuracy change. Don't take it on
as part of this task.

### `spurious_blob` category

**Primary recommendation: add `inference_single(rgb) -> Option<Array2<u8>>`.**

Rationale, tied to Phase 1 findings:

- The user's downstream consumers (this mobile app, the fishsense
  pipeline) treat `mask > 0` as single-fish foreground. The bug is
  that `inference()` returns multi-instance output, not a single fish.
- Largest-by-area selection picks the correct fish in 100 % of the
  reproducing fixture cases. Score-based selection does not work
  (many detections saturate at score = 1.000).
- A new API is the right shape because the existing per-instance
  encoding is genuinely useful for callers who want to know about
  multiple fish (e.g. crowd scenes). The ground rules say not to
  change `inference`'s semantics, and we shouldn't.
- This is a pure post-processing change; no model retraining, no new
  dependencies.

**Proposed API:**

```rust
impl FishSegmentation {
    /// Runs the segmentation model and returns a single-instance binary
    /// mask (255 where fish, 0 elsewhere) of the largest-area fish
    /// detection. Returns `None` if no detection survives the score
    /// threshold. Use this when the downstream consumer expects
    /// one-fish-per-image; use [`inference`] when per-instance IDs
    /// matter.
    pub fn inference_single(
        &mut self,
        img: &Array3<u8>,
    ) -> Result<Option<Array2<u8>>, SegmentationError>;
}
```

**Implementation sketch** (in `convert_output_to_mask`-adjacent code,
not in `inference`):

1. Run the same pipeline as `inference` up through `do_inference`.
2. For each detection with `score > SCORE_THRESHOLD`, compute its
   thresholded pasted-mask pixel area (this is already done in
   `convert_output_to_mask_debug` as `mask_area_px`).
3. Keep only the detection with the **largest pasted-mask area** above
   a minimum-instance-area floor (suggested 5,000 px at 800×1058 model
   resolution — rules out pathological 1-vertex polygons and
   ~1 k-pixel spurious blobs without affecting legitimate small-fish
   detections; the smallest drawn detection in the fixture is 8,848 px
   at that resolution).
4. Rasterize that detection's polygon into a binary `Array2<u8>` (0 or
   255). No instance IDs.
5. Return `Ok(Some(mask))` or `Ok(None)` if no detection passed.

**PyO3 wrapper**: add `FishSegmentation.inference_single(img) ->
Optional[np.ndarray]` in `python/fishsense_core/src/fish/fish_segmentation.rs`,
mapping `None` to Python `None`.

**Expected residual failures after this fix:**

- Whenever the correct fish is *not* the largest detection. This never
  happens in the fixture; it would only happen in edge cases like a
  small foreground fish obscured by a large background fish. Accept
  this as a known limitation — it's a rank-by-area tradeoff and the
  fixture doesn't motivate anything more sophisticated.
- Multiple-fish scenes where the user wants the second fish. Those
  callers should keep using `inference`.
- Hand/glove false positives that happen to be *larger* than the real
  fish. Possible but none observed in the fixture.

### Alternatives considered and rejected

- **Post-hoc largest-or-best flag on `inference()`** — mixes API
  contracts (returns either instance-encoded or binary mask). A
  separate function is cleaner and doesn't risk surprising existing
  callers.
- **NMS / overlap merging at the detection level** — Mask R-CNN
  already runs per-class NMS internally. Our spurious detections in
  the fixture do *not* heavily overlap the primary (by bbox or mask
  IoU), so extra NMS would not help.
- **Min-instance-area filter on `inference`** — a conservative floor
  (≈5 k px) would catch genuinely tiny fragments but not the ~24 k px
  secondary in case 05 nor the ~28 k px secondary in case 06. Not
  useful alone.
- **Lowering/raising score threshold** — Phase 1 shows spurious
  detections are ≥0.706 and often 1.000. Threshold changes don't move
  the needle here.
- **Negative-example retraining** on gloves, hands, orange deck
  furniture, out-of-frame fish. This *is* the right long-term fix for
  the false-positive cases (01, 06) and would improve robustness on
  real deployments, but it is out of scope here and the user
  explicitly excluded retraining from this task. Flagging it for a
  separate task.

### Things I'd want from the user before Phase 3

1. **Sign-off on the API shape** (`inference_single` returning
   `Option<Array2<u8>>`, 0/255 binary mask, largest-by-area selection,
   5 k-px floor). Name, return type, and floor are all tunable.
2. **Resolve the fixture drift question**: I'd like to regenerate the
   7 spurious_blob cases from *this* build of the crate and confirm
   they still classify as failures. If only 4 of 7 are currently
   reproducible, the regression tests in Phase 3 should be based on
   what's actually reproducible — otherwise tests will be flaky.
   Possible options: (a) pin `with_intra_threads(1)` + `Level0` opt
   for tests to tame ORT non-determinism; (b) accept some flake and
   write tolerance into the assertions; (c) capture a fresh fixture.
   I'd recommend (a) for the test path only, with a normal-perf path
   for production. Happy to discuss.
3. **Are there additional failure modes you've seen in the wild that
   aren't in the fixture?** E.g. fish-on-fish occlusion, heavy
   underwater scenes, partial/cropped fish. If yes, send me even 5–10
   labeled examples and I'll incorporate them into the Phase 3
   regression tests before committing.
4. **Training data coverage, long-term**: if the false-positive cases
   (gloves/hands/orange targets) matter at deployment scale, a set of
   ~50–200 negative examples (photos of deck setup with no fish
   present, plus photos where a hand/glove is prominent next to the
   fish) would let us file a separate retraining task. You mentioned
   you might be able to generate or label more — worth doing.

---

**Stopping here per the Phase-2 instruction. Awaiting sign-off before
implementing.**
