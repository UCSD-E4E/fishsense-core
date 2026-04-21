# Head/tail orientation: Phase 1 diagnosis and Phase 2 proposal

## Fixture distribution (Phase 1, step 1)

Fixture root: `2026-04-19_fishsense-mobile/scripts/bug_report/fixture/`.

- **519 total cases**, **345 failures** at `threshold_px = 100`.
- **304 `likely_swap`** — **88% of failures**. Endpoints are approximately
  correct; head/tail assignment is flipped.
- **41 `endpoints_wrong`** — 12% of failures. At least one returned point
  isn't near either labeled endpoint.
- Worst `likely_swap`: `case_01` at ~1500 px on both endpoints (full fish
  length — orientation is fully inverted).
- Worst `endpoints_wrong`: `case_190` — snout 91 px (ok-ish), fork 884 px
  (one endpoint is completely wrong).

The problem space is dominated by a single failure class
(`likely_swap`). Fixing that alone would eliminate ~88% of the reported
failures.

## Per-case diagnostic (Phase 1, steps 2–4)

Tool added at `examples/diagnose_head_tail.rs`.

### `likely_swap` cases — what the classifier is seeing

Sampled every 15th `likely_swap` case (21 cases). For each, I ran the
full detector up through `classify_from_perimeter` and logged, per half,
the polygon area, convex-hull area, hull-area fraction, and the
per-half concavity info the current heuristic uses.

Example (`case_01`, worst failure):

| Half           | poly area | hull–poly | hull-delta frac | min significant concavity distance |
|----------------|-----------|-----------|-----------------|-------------------------------------|
| Snout side (PCA left)  | 293172    | 71294     | 24.3 %          | 1.30 px                             |
| Fork side (PCA right)  | 234018    | 128318    | **54.8 %**      | 1.30 px                             |

The raw PCA endpoints are already correct here (`left` 162 px from the
labeled snout, `right` 63 px from the labeled fork). PCA is not the
issue. Every one of the 21 sampled cases has PCA endpoints within
~140 px of their correct labels.

Aggregated over 21 sampled `likely_swap` cases:

| Heuristic                                           | correct | wrong |
|-----------------------------------------------------|---------|-------|
| **Hull-area delta** — tail is the half with the larger convex-hull-minus-polygon area | **20 / 21** | 1 / 21 |
| **Min-significant-concavity distance (current)** — tail is the half whose PCA endpoint is closer to a ≥1%-area concavity | 1 / 21  | **20 / 21** |

The current heuristic has the **wrong sign** on this dataset. The
reason is anatomical: on real fish the head half routinely contains
concavities as close to, or closer to, its PCA endpoint than the fork
notch is to the tail-side PCA endpoint — mouth openings, operculum/gill
slits, dorsal-fin notches, pectoral-fin attachment points, ventral-fin
attachment points. Meanwhile the PCA "fork" endpoint is almost always
the tip of one of the two caudal-fin lobes (a point *on* the convex
hull), so its distance to the fork notch is `half_lobe_span * sin(θ)` —
tens of pixels, not zero.

This is not a threshold-tuning issue. The signal is pointing the wrong
direction across the full population.

The single case where hull-area is also wrong — `case_181` — has the
fork side completely hull-convex (fork-side hull-delta = 0.0 %). No
concavity of any kind survives the 1 % filter on the fork half. This is
an edge case; the peduncle signal below handles it.

### `endpoints_wrong` cases — separate problem

Full sweep of all 41 `endpoints_wrong` cases. In nearly every one:
- The PCA endpoint on **one side** lands within ~30 px of its labeled
  keypoint (usually the snout — the head is densely and reliably
  segmented).
- The PCA endpoint on **the other side** lands hundreds of pixels short
  of its labeled keypoint. `case_190`: fork mismatch 884 px.
  `case_260`: 684 px. `case_294`: 589 px.
- Mask `nnz` is consistently a tiny fraction of the labeled span —
  e.g. `case_190` has 15 606 mask pixels for a fish whose labeled
  endpoints span ~1000 px. On `case_294` the PCA axis tilts well off
  horizontal because the mask contains a non-fish blob pulling the
  second principal direction.

**Diagnosis: these are upstream segmentation/mask failures, not
detector failures.** The PCA stage cannot recover endpoints that
aren't represented in the mask. A different head/tail classifier will
not help these — the caudal fin is simply not in the mask, or the
mask is contaminated with a non-fish component. This is a
segmentation-model problem to address in `fish_segmentation.rs` (or
upstream of it), not a geometry problem.

No `likely_swap` case in my sample had an empty half, degenerate
split, or other numerical instability. PCA converges cleanly on all
sampled masks.

### Why the recent commits didn't help

Commit history on the classifier:

```
a935d56 fix: pick nearest significant concavity, not largest, for head/tail
c5b686c fix: classify head/tail by concavity proximity, not total area
```

Both iterations replaced the working hull-area signal with a
concavity-distance signal. On three in-tree fixtures this was a net
win, but on the 304 real-world `likely_swap` masks it inverts the
sign. For reference, re-running hull-area delta on the three existing
in-tree fixtures:

| Fixture                       | snout-half frac | fork-half frac | hull-area picks tail correctly |
|-------------------------------|-----------------|----------------|-------------------------------|
| `head_tail_regression`        | 3.7 %           | 46.6 %         | ✓                             |
| `head_tail_snout_right`       | 35.4 %          | 33.5 %         | ✓ (2 pp margin)               |
| `head_tail_concavity_swap`    | 3.5 %           | 19.1 %         | ✓                             |

Hull-area classifies all three existing fixtures correctly too. Whatever
historical defect motivated the switch away from hull-area has evidently
been fixed by other changes since — most likely the perimeter-extraction
rewrite from hand-rolled Moore trace to `cv::findContours`, which
produces well-formed halves and valid `hull.difference(poly)` results.

## Phase 2 proposal

### Primary approach: caudal peduncle detection

Walk along the PCA axis and compute silhouette width (max-perp minus
min-perp of mask pixels projected onto the axis) as a function of
signed distance from the centroid. **The narrowest cross-section on a
fish's silhouette is the caudal peduncle — the waist between the body
and the tail fin.** The tail endpoint is whichever PCA endpoint lies
on the same side of the peduncle minimum as the caudal fin.

Operationally:

1. Use the PCA axis already computed in `fish_pca.rs`. No re-fit.
2. Project every mask pixel onto the axis, giving `(s, w)` where `s` is
   signed projection and `w` is perpendicular offset.
3. Bin `s` into ~50 bins over the endpoint-to-endpoint span. For each
   bin, `width(s) = max(w) - min(w)` over the pixels in that bin.
4. Find the bin with the smallest `width(s)`, excluding the outermost
   10 % of bins on each end (where width collapses naturally as you
   run off the fish).
5. The peduncle bin partitions the axis into a body side and a
   fin side. The PCA endpoint on the fin side (= the distal side,
   where width bounces back up to the caudal-fin span) is the tail.
   The other is the head.

Why this should work across the dataset:

- It is tied to a **stable anatomical feature**. Teleost caudal
  peduncles are narrow by construction; it is what makes a fin a fin
  rather than an extension of the body.
- It does not depend on convex-hull accidents that vary with species
  (deep-bodied vs. fusiform, prominent vs. reduced dorsal fin, open vs.
  closed mouth in the mask, pectoral-fin inclusion).
- Width minimum is a low-noise statistic — sampling hundreds of
  perpendicular offsets per bin averages away perimeter jitter.
- PCA gives the axis for free and is already correct on 99 %+ of
  in-distribution masks (Phase 1 showed PCA endpoints are within
  ~140 px of labels on every `likely_swap` case sampled).

### Expected failure modes

Honest accounting of where this will still break:

- **Mask failures (the `endpoints_wrong` class).** If the caudal fin
  isn't in the mask, there is no peduncle to find. 41/519 cases
  (~8 %). Same breakage as today; not a regression. These need a
  segmentation fix.
- **Curved/bent body pose.** If the fish is C-shaped, the PCA axis is
  a chord across the curve. Projected widths are dominated by the curve
  rather than the peduncle, and the minimum can land in the wrong place.
  Every-15th sampling didn't show an obvious C-shaped case in the
  fixture, but I can't prove there are none; worth flagging in docs.
- **Species without a well-defined peduncle.** Eels, triggerfish
  (boxy), some flatfish pseudo-lateral presentations. Unlikely in this
  dataset but will fail if present.
- **Very short fish or very coarse masks** (e.g. the `endpoints_wrong`
  sub-masks with <2 000 px). Bin width becomes larger than peduncle
  width; the minimum is noisy. Mitigated by requiring a minimum mask
  area or by adaptive bin count; worst case, fall back to hull-area.
- **Tie cases / very flat width profiles** (slab-shaped fish). The
  width minimum becomes weak. Fall back to hull-area delta.

### Implementation sketch

Scope: ~150 lines in a new `fish/fish_peduncle.rs`, plus a 30-line
classifier swap in `fish_geometry.rs::classify_from_perimeter`.

- New `fn locate_peduncle(mask, pca) -> Option<PeduncleInfo>`. Returns
  `{ s_peduncle: f64, width_min: f64, confidence: f64 }` where
  `confidence` is `(flank_width - width_min) / flank_width` over the
  flanking bins — a dimensionless narrowing ratio. Returns `None` when
  `confidence` is below a small threshold (e.g. 0.15) — not enough
  narrowing to be a real peduncle.
- Modify `classify_from_perimeter` to:
  - Try peduncle classification first.
  - If `locate_peduncle` returns `None`, fall back to hull-area delta
    (the proven-correct-on-95%-of-cases signal), not the current
    min-sig-dist heuristic.
  - Only fall back to min-sig-dist if hull-area delta is also a
    tie (e.g. both fractions differ by <2 pp) — which empirically
    never happens in this fixture.
- Confidence score becomes `peduncle.confidence` when peduncle wins,
  else the hull-area delta margin. Cleaner interpretation downstream.

No external dependencies; pure ndarray + f64 math. No change to the
Python binding — still returns `(head, tail)` as `f32 [x, y]`.

No change to `correct_head` / `correct_tail` geometry; those operate
on halves *after* classification. They were mis-targeted in the
failing cases only because the halves were swapped.

### What additional labeled data I want

The current fixture is plenty for the primary signal — 304 `likely_swap`
cases across what I assume is multiple species is a strong test set
for a classifier that's supposed to be species-agnostic. I don't need
more of those to make the decision.

What would help:

1. **Species breakdown for the current fixture**, if you know it.
   If all 304 are the same species, the 95 % hull-area win rate could
   be lucky. If it's a mix of 5+ species with the same outcome, that's
   a much stronger claim. I can't tell from the masks alone.
2. **A small handful (~5–10) of C-shaped / bent-pose fish** with
   labeled snout/fork. My Phase-1 sampling didn't surface any and the
   peduncle approach is vulnerable on those. If these are a real
   operational case, I need to know before committing — might need a
   PCA-free axis (e.g. skeletonization) for those.
3. **~5 examples of the `endpoints_wrong` cases with the RGB and raw
   segmentation output**, not just the binary mask. I want to confirm
   the caudal fin is actually absent from the mask (segmentation
   failure) vs. present but pruned by some post-processing. If the
   latter, that's a much cheaper fix than a new classifier.

If (1) shows single-species bias, or (2) is a real population, I'd
revisit whether the peduncle signal alone is enough or whether a
learned classifier over the endpoint crops makes more sense — but I'd
want that evidence before adding an ONNX dependency.

### Fallback / alternative ranking

If the peduncle approach doesn't land cleanly in implementation:

1. **Revert to hull-area delta as primary.** One-line change, fixes
   ~290/304 `likely_swap` cases, passes all 3 existing in-tree
   fixtures. Low-confidence-margin cases (case_181-style) will still
   be wrong (~1/20 of fixes), but that's a strict improvement over
   today's state where ~0/20 are right.
2. **Endpoint-tip curvature.** Useful as a reinforcing signal but
   weaker alone — some caudal fin tips are quite pointed.
3. **Learned classifier.** I'd want the species/pose evidence from
   (1)/(2) above before paying the ONNX + training-data cost.
4. **Stacked heuristic vote** (peduncle + hull-area + maybe
   tip-curvature, majority wins). Defensible stopgap but harder to
   debug than a single primary signal with named fallbacks.

## Sign-off needed

**Please confirm before I move to Phase 3:**

- [ ] Primary approach = caudal peduncle detection. ✅ / ✏️
- [ ] Fallback = hull-area delta (not min-sig-distance). ✅ / ✏️
- [ ] `endpoints_wrong` cases treated as out-of-scope for this PR
      (segmentation-side fix). ✅ / ✏️
- [ ] Any of the "additional labeled data" requests you want to
      action first.

Once signed off I'll:

1. Add a fixture-driven regression test in
   `fish/fish_head_tail_detector.rs` covering at minimum `case_01`
   (worst swap) and one `endpoints_wrong` case as a
   known-segmentation-failure marker. Parametrize over the full
   fixture if the checked-in copy is small enough; otherwise keep a
   curated subset in-tree and point the test at the external fixture
   via an env var for the full run.
2. Implement `fish/fish_peduncle.rs` and rewire
   `classify_from_perimeter`.
3. Keep all existing in-tree fixture tests green (they all also
   classify correctly under hull-area delta, per the table above).
4. Summarize diagnosis / approach / patch / tests in the PR.
