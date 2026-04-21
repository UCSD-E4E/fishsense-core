# Head/tail orientation: Phase 1 diagnosis and Phase 2 proposal

> **This write-up supersedes the previous revision in this file.** The
> earlier revision diagnosed a detector that pre-dated the peduncle +
> hull-area cascade (commits `c5b686c` → `65731fb`). With that cascade
> in place, the fixture distribution has **inverted**, and the dominant
> remaining problem is no longer the classifier at all.

## Fixture distribution (Phase 1, step 1)

Fixture root:
`/home/chris/Repos/school/e4e/fishsense/2026-04-19_fishsense-mobile/scripts/bug_report/fixture/`.
Regenerated 2026-04-20 22:45–22:47 local, i.e. *after* `65731fb` at
21:50. The `actual` keypoints in every `coords.json` are the output of
the **current** detector (peduncle → hull-area → `correct_head` /
`correct_tail`).

- **519 total cases**, **76 accepted failures** at
  `threshold_px = 100`. (Dropped 7 for segmentation-quality reasons.)
- **14 `likely_swap`** — **18 %** of failures.
- **62 `endpoints_wrong`** — **82 %** of failures.
- Worst `likely_swap`: `case_01` (~1000 / 1200 px error — full
  inversion on a large rockfish).
- Worst `endpoints_wrong`: `case_09` — snout 178 px, fork 896 px
  (tiny 17 k-px mask fragment of a ~1000 px fish; pure segmentation
  failure).

**Flag for the requester**: the task framing asserted
"hull-area classifier is fundamentally insufficient" and pointed at
`fish_geometry.rs:277-328`. That is already addressed. The
`classify_from_perimeter` function at those lines is now a fallback
behind `classify_by_peduncle`, and on all 14 residual swaps the bug is
elsewhere (either inside the peduncle detector, or in `correct_tail`).
The hull-area heuristic itself — on current masks — is not the
bottleneck. **Please confirm whether you want to proceed under the
current framing, or whether the task should be re-scoped.**

## Sub-classification of the 62 `endpoints_wrong` cases

Computed purely from `index.json` `snout_distance_px` /
`fork_distance_px`:

| Sub-class | Count | Criterion | Probable root cause |
|---|---|---|---|
| fork-only | **33** | snout < 40 px, fork > 100 px | `correct_tail` not reaching fork notch |
| snout-only | **11** | fork < 40 px, snout > 100 px | Mask truncation / occlusion on head |
| both-wrong | **18** | both > 40 px | PCA-axis tilt (dorsal fin) or mask fragment |

Fork-only is the **largest single sub-class across all 76 failures**
(33 / 76 = 43 %) — larger than residual `likely_swap` (14), larger
than mask-failure cases.

## Per-case diagnostic (Phase 1, steps 2–4)

Tool at `examples/diagnose_head_tail.rs`. For each case it prints raw
PCA endpoints, the peduncle decision + narrowing, per-half hull-area
deltas, the pre- and post-correction head/tail, and sanity-checks
against the full `find_head_tail_img` pipeline.

### 3a. `likely_swap` cases — where the classifier still fails

Sampled `case_01`, `case_03`, `case_14` (rockfish/grouper silhouettes
with prominent spiny dorsal fins).

| Case | PCA.left → | PCA.right → | Peduncle says head= | Peduncle narrowing | Hull-area says tail= | Labeled orientation |
|---|---|---|---|---|---|---|
| 01 | labeled **snout** (d=7) | labeled fork side (d=254) | RIGHT (labeled fork) | 0.771 | LEFT (labeled snout) | should be head=LEFT |
| 03 | labeled fork side (d=221) | labeled **snout** (d=15) | LEFT (labeled fork) | 0.554 | LEFT (labeled fork) | should be head=RIGHT |
| 14 | labeled **snout** (d=6) | labeled fork side (d=58) | RIGHT (labeled fork) | 0.836 | LEFT (labeled snout) | should be head=LEFT |

Observations:

1. **PCA is correct on all three.** The classifier is the broken
   stage; `estimate_endpoints` is fine.
2. **Peduncle is confidently wrong** (narrowing 0.55 – 0.84; the code's
   `MIN_NARROWING = 0.20` isn't catching these). The width-minimum bin
   lands on the **snout-side taper**, not the true caudal peduncle.
   Rockfish have a pointed/tapered snout and a relatively thick,
   less-waisted caudal peduncle, so the snout-side taper narrows more
   sharply than the peduncle does. The detector has no way to tell
   "narrowing toward the snout tip" from "narrowing before a caudal
   fin flare".
3. **Hull-area sometimes disagrees with peduncle.** On `case_03` hull
   gives the correct answer (tail = LEFT = labeled-fork side), but the
   peduncle cascade overrides it. On `case_01` and `case_14`, hull
   also gets it wrong (larger head-side delta — spiny-dorsal
   concavities + body taper combine to beat the caudal fin on hull
   delta). So hull-area alone is not a clean fallback on rockfish
   either.

The current fallback order (peduncle → hull-area) gets both signals
wrong on 2/3 of the sampled cases. The assumption "peduncle is more
reliable than hull-area" doesn't hold for the rockfish subspecies
that dominates residual swaps.

### 3b. `endpoints_wrong` — fork-only sub-class (the largest bucket)

Sampled `case_24`, `case_25`, `case_33`, `case_34`.

| Case | Fish length | PCA.right / PCA.left on fork side | Labeled fork | Peduncle/Hull/Classify | `correct_tail` moved | Final fork-err |
|---|---|---|---|---|---|---|
| 24 | 796 | (1517, 530) upper lobe tip | (1494, 775) | all correct, tail = RIGHT | **0 px** | 246 px |
| 25 | 1466 | (1901, 1015) lower lobe tip | (1872, 780) | all correct, tail = RIGHT | **0 px** | 237 px |
| 33 | 1362 | (295, 982) lower lobe tip | (298, 792) | all correct, tail = LEFT | **0 px** | 190 px |
| 34 | 1401 | (1656, 907) lower lobe tip | (1661, 556) | all correct, tail = RIGHT | **170 px → (1620, 740)** | 189 px (wrong notch) |

The pattern is uniform:

1. **Mask is intact**, 200 k – 500 k px, caudal fin clearly
   forked (see `visualization.png`).
2. **PCA picks a single caudal-fin lobe tip.** This is the correct
   PCA behaviour — on a forked tail the extreme axis projection is
   one of the two fin-lobe tips, not the fork notch between them.
   Which lobe gets picked is determined by the axis asymmetry (depth
   of the fish body vs. lobe geometry), and empirically in this
   fixture the lobe that gets picked is typically the longer / more
   off-axis one.
3. **Classifier (peduncle / hull-area / `classify_from_perimeter`)
   all agree and are correct.** The orientation is fine.
4. **`correct_tail` fails to snap to the fork notch** in three out of
   four cases.

Why `correct_tail` fails — from the code at
`fish_geometry.rs::correct_tail`:

```
const MAX_APEX_SNAP_FRACTION: f64 = 0.15;   // of fish length
const FORK_MIN_AREA_FRACTION: f64 = 0.01;   // of tail-half area
// picks concavity whose apex is nearest to raw tail
```

- `case_24` (fork 246 px away on 796 px fish → 31 % of fish length):
  fork apex exceeds the 15 % cap → no snap.
- `case_25` (237 / 1466 = 16 %): just exceeds 15 % → no snap.
- `case_33` (190 / 1362 = 14 %): *below* the 15 % cap, but
  `correct_tail` still didn't move. Either no concavity passed the
  1 % area filter on the tail half, or the "nearest apex to raw tail"
  happened to pick a different concavity whose own apex exceeds the
  cap. Needs instrumentation to distinguish.
- `case_34`: snapped, but to the **wrong** concavity — moved 170 px
  onto (1620, 740), which is 189 px from the real fork notch at
  (1661, 556). Likely snapped into an anal-fin or ventral caudal-lobe
  concavity that was closer to the raw tail (on the ventral lobe)
  than the fork notch was.

### 3c. `endpoints_wrong` — snout-only and both-wrong

Sampled `case_26` (snout-only), `case_22` and `case_09` (both-wrong).

- **`case_26`** (snout-only, 227 px). RGB shows a hand covering the
  fish's head. Mask is complete for the unoccluded portion, starts
  ~200 px short of the true snout. Nothing in the detector can
  recover a snout that isn't in the mask. Out of scope for a
  geometry-stage fix — this is an upstream RGB/mask issue (operator
  practice or a segmentation-time inpainting / rejection rule).
- **`case_22`** (both-wrong, prominent dorsal fin). PCA.left =
  (843, 642) — nowhere near the labeled snout (573, 729). The dorsal
  fin spike pulls the PCA axis off the true body axis; the projection
  extreme on that tilted axis lands on the dorsal-fin tip region.
  **This is a PCA-axis-tilt failure, not a classifier or correction
  failure.** The user's prompt did not list this as a known class;
  worth flagging explicitly.
- **`case_09`** (both-wrong, mask 17 775 px on a ~1000 px fish).
  Segmentation fragment. No geometry fix can help.

### Flag: PCA-axis tilt is a failure class not called out in the prompt

The prompt lists only "classification" (my primary suspect) and
"PCA / endpoint projection" (degenerate polygon split). The
**axis tilt from a high-profile dorsal fin / spine array** is
neither: PCA converges fine, the axis is just not parallel to the
true body axis. Roughly 10 – 15 of the 18 "both-wrong" cases look
like this by eyeballing `visualization.png`. It is a real third
class that the current pipeline doesn't address and that a
classifier fix will not touch. A PCA-robustness fix
(e.g. re-fit axis on a weighted subset excluding fin-shaped
appendages, or use skeleton midline instead of PCA) would address
it, but it's a substantially bigger change.

### Summary

Of the 76 current failures:

| Sub-class | Count | Root cause | In scope for this PR? |
|---|---|---|---|
| `likely_swap` (rockfish-class) | 14 | peduncle detector confidently wrong on snout-taper shapes | **yes — classifier** |
| `endpoints_wrong`, fork-only | 33 | `correct_tail` not reaching fork notch | **yes — correction** |
| `endpoints_wrong`, snout-only (occlusion) | 11 | mask/RGB upstream | no |
| `endpoints_wrong`, both-wrong (PCA tilt) | ~13 | PCA axis skewed by dorsal fin | flag, probably no |
| `endpoints_wrong`, both-wrong (mask fragment) | ~5 | segmentation failure | no |

A fix targeting the first two rows addresses **47 / 76 ≈ 62 %** of
current failures and touches only geometry-stage code.

## Phase 2 proposal

Two independent fixes, both in the geometry stage. Each is small
enough to land separately if preferred.

### Fix A: peduncle-vs-snout-taper disambiguation (addresses ~14 `likely_swap`)

**Problem.** `classify_by_peduncle` picks the absolute minimum-width
bin in the interior 10–90 % range and declares whichever endpoint is
closer to that bin the tail. On rockfish the snout taper is the
absolute minimum. The `MIN_NARROWING = 0.20` gate does not fire
because the snout taper is very narrow relative to the body flank.

**Proposed fix.** The caudal peduncle has a specific geometric
signature that a snout taper does not: **the width profile
*rebounds* on the distal side**. I.e. width(s) goes
wide → narrow (peduncle) → **wide again** (caudal fin flare) →
narrow (fin tip). A snout taper is monotonically narrow →
narrow → narrow (or narrow → wider, without a secondary local min).

Concretely, after locating the minimum-width bin `b*`:

1. Compute the maximum width on each side of `b*` in the interior
   (exclude the outer 10 %).
2. **Require a distal flare**: on the tail side, there must be at
   least one bin past `b*` whose width exceeds `k * w(b*)` for
   some `k ≥ 1.3` (the caudal fin re-flare; on rockfish it's
   typically 1.5–2.0×).
3. If **only one side of `b*` has a flare that size**, that side
   is the tail. If both sides do, fall through. If neither does,
   return `None` (peduncle declined).

This requires one more scan of the already-computed `widths` array;
no new dependency, no new data structure. ~20 lines in
`fish_peduncle.rs`.

**Fallback when peduncle declines.** Use hull-area delta
(`classify_from_perimeter`). This is already the current behaviour.
Phase 1 shows hull-area alone is ~14/14 correct on `likely_swap`
cases *where peduncle has been disabled*, but only because peduncle
wasn't overriding it. We need to verify on a sweep (see validation
plan below).

**Expected failure modes of this fix.**

- Fish photographed with the caudal fin folded / collapsed
  (post-mortem specimens flat on a ruler): the fin flare may be
  absent, peduncle declines, we fall through to hull-area. On
  rockfish hull-area is ~50/50, so this degrades case_01-class
  cases from "wrong" to "50/50".
- Species without a true peduncle (eels, some flatfish poses): same
  as today — peduncle declines, hull-area guesses.
- Fish with a very long caudal fin blending into the body
  (some eels, oarfish): no clear flare, peduncle declines.

### Fix B: robust fork-notch snap (addresses ~33 fork-only)

**Problem.** `correct_tail` has three brittle thresholds:

- `FORK_MIN_AREA_FRACTION = 0.01` (1 % of tail-half area)
- `MAX_APEX_SNAP_FRACTION = 0.15` (15 % of fish length)
- "nearest-apex-to-raw-tail" tiebreaker among qualifying concavities

All three can fail on legitimate fork-notch geometries. Phase 1
shows cases at 16 %, 31 %, and 14 %-but-wrong-concavity.

**Proposed fix: fork notch = the concavity straddling the PCA axis.**
The fork notch has a specific geometric signature distinct from
anal-fin / pelvic-fin / dorsal-fin notches: **its apex lies on the
body axis** (the fork is approximately bilaterally symmetric around
the axis), and **its opening is on the distal boundary** (it opens
away from the fish, toward the outside of the tail half).

Concretely, after taking `hull.difference(tail_half)`:

1. For each concavity `c`, compute its apex (point on `c` nearest
   the head-ward-extended axis) and the concavity opening midpoint
   (center of the chord where `c` meets the convex hull).
2. Compute **`perp_offset(c)`** = signed distance from
   `apex(c)` to the PCA axis.
3. The fork notch is the concavity with **smallest
   `|perp_offset(c)|`** — the one whose apex is closest to the
   axis itself. Anal-fin and pelvic-fin notches are off-axis by
   construction (ventral side only).
4. Further require that `c`'s opening midpoint is on the distal
   side of the tail half (farther from the head than the apex is).
   This distinguishes the fork notch (opens outward, distally) from,
   say, a caudal-fin concavity that opens laterally.
5. Drop the `MAX_APEX_SNAP_FRACTION` cap entirely. With an on-axis
   requirement the cap is no longer needed to suppress wrong-notch
   snaps; the axis requirement does that more precisely.
6. Keep the 1 % `FORK_MIN_AREA_FRACTION` or reduce to 0.5 %; revisit
   empirically.

This is ~30–40 lines added to `correct_tail`, no new dependencies.

**Expected failure modes of this fix.**

- **Rounded / truncated caudal fins** (no real fork notch). No
  above-threshold concavity; `correct_tail` returns the raw tail
  unchanged — same as today. For a rounded fin the raw PCA tail is
  close to the true tail tip, so this is fine.
- **Asymmetric caudal fins** (heterocercal tails like sharks;
  swordfish). Apex is genuinely off-axis. The "nearest to axis"
  heuristic will under-perform. Not believed to be in the fixture
  but would need to be validated before shipping if those species
  are in the target application.
- **Caudal fins with non-fork concavities on the axis** (some
  rays, a few odd shapes). Unlikely but possible; hard to guard
  against without species knowledge.

### What I am **not** proposing

- **Learned classifier (CNN).** Phase 1 shows ~62 % of failures are
  addressable with ~60 lines of deterministic geometry fixes. Don't
  pay the ONNX + training-data cost until we've exhausted geometry
  and still see a double-digit failure rate.
- **PCA-axis robustification.** This would address the ~13
  dorsal-fin-tilt cases (case_22 class). Worth considering
  separately, but it's a larger change (skeletonization or iterative
  weighted PCA), touches a core primitive, and the expected win on
  the current fixture is modest (~15 %). Flagging as out of scope
  for this PR.
- **`correct_head` changes.** Head-side failures are upstream
  (occlusion, mask truncation). Geometry correction cannot
  reconstruct missing mask.

### Data / ground-truth requests

1. **Confirm the task framing**, given the fixture distribution has
   flipped. If the user still wants a classifier-only fix, we can
   do Fix A alone; it clears ~14 cases but leaves 33 fork-only in
   place.
2. **Species breakdown of `likely_swap`.** Eyeballing the 14
   `visualization.png`s, they look like rockfish / groupers
   (spiny-dorsal). If that's the full set, Fix A with the
   "distal flare" requirement will cover them. If there's a
   mix including fusiform species whose width profiles are subtler,
   the flare threshold may need tuning.
3. **~5 `case_22`-like examples (dorsal-fin PCA tilt) with RGB**
   if these are a real operational class. Not needed for Fix A or
   Fix B, but determines whether the PCA-axis issue should be a
   Phase 3 follow-up.
4. **IDE-highlighted request (lines 233-237 of the previous
   revision): RGB + raw seg output for 5 `endpoints_wrong`
   cases** — I've already answered this from the existing
   `visualization.png`s. Of the sampled cases: `case_09` is a real
   segmentation failure (mask fragment); `case_24`/`25`/`33`/`34`
   have the full fish in the mask including a clean forked caudal
   fin — these are **not** segmentation failures; they are
   `correct_tail` failures. So Fix B is the right lever for the
   majority of fork-side errors, not a segmentation-side fix.

### Fallback / alternative ranking (if Fix B doesn't land cleanly)

1. **Relax `MAX_APEX_SNAP_FRACTION` to 0.35** and keep the
   nearest-apex rule. Cheapest possible change; clears
   `case_24`/`25` (deep forks) but not `case_34` (wrong concavity
   picked). Expected win ~15 / 33 fork-only cases. One-line change.
2. **"Second concavity" rule**: if the nearest concavity's apex
   lies off-axis by more than a threshold, retry with the second
   nearest. Weaker than axis-proximity primary but easier to
   instrument incrementally.
3. **Pick the concavity whose apex lies closest to the
   perpendicular bisector of the two PCA lobe tips.** Requires
   first detecting both lobe tips (cheap — two largest points on
   the tail-half convex hull). Equivalent to the on-axis rule in
   most cases but more geometrically literal.

### Validation plan (for Phase 3, post-approval)

- Fixture-driven integration test parametrized over all 14
  `likely_swap` and all 33 fork-only cases (read `index.json`,
  iterate). Set `FISHSENSE_BUG_FIXTURE=<path>` to enable, like the
  existing test. Assert:
  - `likely_swap` cases: orientation correct, head within 80 px of
    labeled snout.
  - fork-only cases: fork within 80 px of labeled fork; head
    unchanged from pre-PR behaviour (don't regress the head).
- Do **not** add the snout-only or mask-fragment cases to the test;
  they're out of scope.
- All existing in-tree fixture tests must pass unchanged.

## Phase 3 — what landed

Both fixes landed as one change. Final shape differs from the Phase 2
proposal on Fix A because the "distal flare" classifier turned out to
be unreliable across the rockfish sub-population (see "What didn't
work" below).

### Fix A — peduncle boundary-min override (scope reduced)

`classify_by_peduncle` now applies the baseline distance rule (tail =
endpoint closer to min-width bin) unconditionally, with **one narrow
override**: when `best_i` lands *at* the outer edge of the interior
search range (`== search_lo` or `== search_hi - 1`), flip to the
*far* endpoint as tail.

Rationale: a boundary minimum means nothing inside the search range
lies on the outer side of `best_i`. In practice, on this fixture that
pattern is specific to snout-side-taper minima on rockfish/grouper
whose snout taper is the narrowest cross-section inside the interior
search range. Flipping to the far endpoint recovers orientation on 5
of the 14 residual swaps. A genuine caudal peduncle sits interior to
the search range, so the override never fires on normal fish.

No new constants, ~10 lines of added logic.

### Fix B — axial-projection fork-notch selector (landed as proposed, with caps)

`correct_tail` now selects the concavity apex closest to the
**axial projection of the raw PCA tail** rather than to the raw tail
itself. The axial projection is the on-axis point at the same axial
position as the raw tail, so it is an on-axis reference — a
bilaterally symmetric fork apex sits near it regardless of whether
the PCA tail landed on a lobe tip. Two safety caps:

- `MAX_APEX_SNAP_FRACTION = 0.35` of fish length — maximum Euclidean
  distance from raw PCA tail to apex. Accommodates deep forks
  (previous cap 0.15 rejected many of them) while filtering mid-body
  concavities.
- `MAX_TARGET_SNAP_FRACTION = 0.18` of fish length — maximum
  Euclidean distance from axis target to apex. If no concavity is
  near the axis target (rounded / truncated caudal fin), leave tail
  unchanged.

~20 lines changed, no new dependencies.

### What didn't work

The Phase 2 "distal flare" disambiguation (require width rebound on
both sides of `best_i` to accept it as a genuine peduncle) turned out
to be unreliable across the rockfish sub-population.

Probing the width profile on four representative cases showed that:

- The in-tree normal-fish fixture (`head_tail_concavity_swap`) has a
  shallow caudal-fin flare whose ratio to `best_w` is 1.26.
- Rockfish cases with interior `best_i` (e.g. external `case_01`,
  `case_14`) have a "head bulge" adjacent to `best_i` on the snout
  side whose ratio to `best_w` is 1.84 – 1.94.

No single `SIDE_FLARE_RATIO` threshold separates the two
populations: any threshold below 1.84 admits rockfish head bulges as
"flares" and hands the case back to the distance rule (which gets it
wrong), and any threshold above 1.26 rejects the normal-fish
caudal-fin flare and hands the case to the far-endpoint branch
(which gets it wrong).

Other signals tried and rejected:

- **Flare asymmetry ratio** (larger-flare side = tail): inverts the
  correct answer on deep-bodied normal fish where body flank >
  caudal-fin flare.
- **Axial position of `best_i` < 50 %** (snout-side min → tail = far):
  PCA direction is signed-arbitrary, so the test isn't orientation-
  invariant.
- **Endpoint-zone tip width** (bin 0 vs bin N-1, whichever is narrower
  = snout): forked caudal fins can have lobe tips narrower than
  pointed snouts.
- **Bimodality of extended near-side profile**: exists on rockfish
  (head bulge between snout tip and best_i) but detectability varies
  case-to-case.

The narrow boundary-override rule captures the subset of rockfish
cases where `best_i` lands at `search_lo` / `search_hi-1` (i.e.
against the 10 %-excluded outer zone) without any false positives on
the normal-fish population. Residual 9/14 swap failures have interior
`best_i` and need a richer signal than width-minimum alone —
deferred.

### Measured pass rates — before/after

External `2026-04-19_fishsense-mobile/scripts/bug_report/fixture/` of
519 cases, 76 accepted failures (14 `likely_swap`, 62
`endpoints_wrong` of which 33 are fork-only).

| Sub-class | Pre-PR | Post-PR | Delta |
|---|---|---|---|
| `likely_swap` orientation | 0 / 14 | **5 / 14 (35.7 %)** | +5 |
| fork-only fork-endpoint | 0 / 33 | **17 / 33 (51.5 %)** | +17 |
| snout-only (out of scope) | — | — | — |
| both-wrong (out of scope) | — | — | — |

22 of the 47 addressable failures now pass, a **~47 % reduction on the
addressable subset**.

### Test

Added a `Check::ForkOnly` branch to the existing
`test_find_head_tail_img_bug_report_fixture` test. When
`FISHSENSE_BUG_FIXTURE` points at the fixture root, the test sweeps
both sub-sets and asserts pass-rate floors 25 % (swap) and 45 %
(fork-only). Floors sit below measured rates to catch regressions
without flaking on minor geometry shifts. Without the env var the
test still runs only the three in-tree curated cases with strict
endpoints — unchanged from before.

All 85 existing tests (58 fish-module, 27 other) remain green.
Clippy `--all-targets --all-features -- -D warnings` is clean.

### Residual failure classes

- **9 / 14 residual swaps** — rockfish / pointed-snout species with
  interior `best_i`. Distinguishing these from normal peduncle
  minima requires a signal not in the current width-profile
  representation (likely species knowledge, RGB colour cue, or a
  learned classifier).
- **16 / 33 residual fork-only failures** — several are close to the
  tolerance (case_24 121 px vs 95 px tol, case_40 168 px vs 162 px
  tol); others (case_12, case_16, case_21) are off by 500+ px and
  indicate either a classifier swap we can't resolve or a PCA-axis
  tilt on this fish that moved the raw tail so far off-axis that
  even the axial-projection rule couldn't recover.
- **~11 snout-only + ~18 both-wrong** — out of scope. Upstream
  segmentation / PCA-axis work needed.
