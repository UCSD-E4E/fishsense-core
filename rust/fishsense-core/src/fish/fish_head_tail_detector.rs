use ndarray::Array2;
use tracing::{debug, instrument};

use crate::{
    errors::FishSenseError,
    fish::{
        fish_geometry::{
            classify_orientation, compute_scale, correct_head, correct_tail, extract_perimeter,
            perpendicular_bisector, polygon_from_perimeter, split_polygon,
        },
        fish_pca::estimate_endpoints,
        fish_plane_fit::predict_keypoint_depths as plane_predict_depths,
    },
    spatial::types::{DepthMap, ImageCoord},
};
use geo::{Closest, ClosestPoint, Distance, Euclidean, Point};

/// Head/tail endpoints in image (mask) pixel space — the grid of the
/// mask passed to the detector.
pub struct HeadTailCoords {
    pub head: ImageCoord,
    pub tail: ImageCoord,
}

pub struct FishHeadTailDetector {}

impl FishHeadTailDetector {
    /// Two-stage pipeline: PCA → geometry refinement.
    ///
    /// 1. Estimates raw endpoints with PCA from the binary `mask`.
    /// 2. Classifies and corrects head/tail using polygon geometry.
    ///
    /// Returns `HeadTailCoords { head, tail }` in image (pixel) coordinates.
    /// Pair this with `predict_keypoint_depths` (or its detector-method
    /// alias) to get the per-keypoint depth via mask-bounded plane fit.
    #[instrument(skip(self, mask), fields(height = mask.dim().0, width = mask.dim().1))]
    pub fn find_head_tail_img(
        &self,
        mask: &Array2<u8>,
    ) -> Result<HeadTailCoords, FishSenseError> {
        // ── Stage 1: PCA ────────────────────────────────────────────────────
        let pca = estimate_endpoints(mask)?;
        let left = pca.left;
        let right = pca.right;
        debug!(
            left_x = left[0], left_y = left[1],
            right_x = right[0], right_y = right[1],
            "PCA endpoints estimated"
        );

        // ── Stage 2: Geometry refinement ───────────────────────────────────
        let perimeter = extract_perimeter(mask);
        if perimeter.len() < 3 {
            return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
                "fish mask perimeter has fewer than 3 points"
            )));
        }

        let classified = classify_orientation(mask, &perimeter, left, right)?;
        let head = classified.head;
        let tail = classified.tail;
        debug!(
            head_x = head[0], head_y = head[1],
            tail_x = tail[0], tail_y = tail[1],
            "head/tail classified from perimeter"
        );

        // Build the two polygon halves for correction.
        let scale = compute_scale(&perimeter, head, tail);
        let (perp_a, perp_b) = perpendicular_bisector(head, tail, scale);
        let poly = polygon_from_perimeter(&perimeter);
        let (half0, half1) = split_polygon(&poly, perp_a, perp_b);

        // Assign head/tail halves based on proximity.
        let head_pt = Point::new(head[0], head[1]);
        let d0: f64 = match half0.exterior().closest_point(&head_pt) {
            Closest::Intersection(q) | Closest::SinglePoint(q) => Euclidean::distance(head_pt, q),
            Closest::Indeterminate => f64::INFINITY,
        };
        let d1: f64 = match half1.exterior().closest_point(&head_pt) {
            Closest::Intersection(q) | Closest::SinglePoint(q) => Euclidean::distance(head_pt, q),
            Closest::Indeterminate => f64::INFINITY,
        };
        let (head_half, tail_half) = if d0 <= d1 {
            (half0, half1)
        } else {
            (half1, half0)
        };

        let head_corrected = correct_head(head, tail, &head_half, scale);
        let tail_corrected = correct_tail(head, tail, &tail_half, scale);
        debug!(
            head_x = head_corrected[0], head_y = head_corrected[1],
            tail_x = tail_corrected[0], tail_y = tail_corrected[1],
            "endpoints corrected via polygon geometry"
        );

        Ok(HeadTailCoords {
            head: ImageCoord(ndarray::array![
                head_corrected[0] as f32,
                head_corrected[1] as f32
            ]),
            tail: ImageCoord(ndarray::array![
                tail_corrected[0] as f32,
                tail_corrected[1] as f32
            ]),
        })
    }

    /// Predict the snout and fork depth via mask-bounded RANSAC plane
    /// fit in camera 3-D coordinates. See [`fish_plane_fit`] for the
    /// full algorithm and its empirical justification.
    ///
    /// `snout_xy` and `fork_xy` are in mask space (the same grid as
    /// `find_head_tail_img`'s output). `k_inv` is the 3×3 RGB-space
    /// inverse intrinsics matrix. Returns `(snout_depth_m, fork_depth_m)`
    /// in metres.
    pub fn predict_keypoint_depths(
        &self,
        depth_map: &DepthMap,
        mask: &Array2<u8>,
        k_inv: &Array2<f32>,
        snout_xy: &ImageCoord,
        fork_xy: &ImageCoord,
    ) -> Result<(f32, f32), FishSenseError> {
        plane_predict_depths(depth_map, mask, k_inv, snout_xy, fork_xy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ── find_head_tail_img unit tests ─────────────────────────────────────

    /// `find_head_tail_img` is sync and requires no depth map.
    /// Endpoints of a horizontal bar should be near the two extreme columns.
    #[test]
    fn test_find_head_tail_img_horizontal_bar() {
        let detector = FishHeadTailDetector {};

        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                mask[[r, c]] = 1;
            }
        }

        let result = detector.find_head_tail_img(&mask);
        assert!(result.is_ok(), "find_head_tail_img failed: {:?}", result.err());
        let coords = result.unwrap();

        let head_col = coords.head.0[0] as usize;
        let tail_col = coords.tail.0[0] as usize;
        let (min_col, max_col) = if head_col < tail_col {
            (head_col, tail_col)
        } else {
            (tail_col, head_col)
        };

        assert!(
            min_col <= 5,
            "one img endpoint should be near col 2, got cols {head_col} and {tail_col}"
        );
        assert!(
            max_col >= 55,
            "other img endpoint should be near col 57, got cols {head_col} and {tail_col}"
        );
    }

    /// Endpoints of a vertical bar should be near the two extreme rows.
    #[test]
    fn test_find_head_tail_img_vertical_bar() {
        let detector = FishHeadTailDetector {};

        // Vertical bar: cols 8..12, rows 2..58 in a 60×20 image.
        let mut mask = Array2::<u8>::zeros((60, 20));
        for r in 2..58 {
            for c in 8..12 {
                mask[[r, c]] = 1;
            }
        }

        let result = detector.find_head_tail_img(&mask);
        assert!(result.is_ok(), "find_head_tail_img failed: {:?}", result.err());
        let coords = result.unwrap();

        // y is index 1 in [x, y] ImageCoord.
        let head_row = coords.head.0[1] as usize;
        let tail_row = coords.tail.0[1] as usize;
        let (min_row, max_row) = if head_row < tail_row {
            (head_row, tail_row)
        } else {
            (tail_row, head_row)
        };

        assert!(
            min_row <= 5,
            "one img endpoint should be near row 2, got rows {head_row} and {tail_row}"
        );
        assert!(
            max_row >= 55,
            "other img endpoint should be near row 57, got rows {head_row} and {tail_row}"
        );
    }

    /// Returned image coordinates must lie within the mask bounding box.
    #[test]
    fn test_find_head_tail_img_coords_within_mask_bounding_box() {
        let detector = FishHeadTailDetector {};

        let mut mask = Array2::<u8>::zeros((20, 60));
        for r in 8..12 {
            for c in 2..58 {
                mask[[r, c]] = 1;
            }
        }

        let coords = detector.find_head_tail_img(&mask).unwrap();

        for (label, coord) in [("head", &coords.head), ("tail", &coords.tail)] {
            let x = coord.0[0] as usize;
            let y = coord.0[1] as usize;
            assert!(
                x < 60,
                "{label} x={x} is out of image width"
            );
            assert!(
                y < 20,
                "{label} y={y} is out of image height"
            );
        }
    }

    /// Empty mask → `find_head_tail_img` should return `Err`.
    #[test]
    fn test_find_head_tail_img_empty_mask_returns_err() {
        let detector = FishHeadTailDetector {};
        let mask = Array2::<u8>::zeros((20, 60));
        assert!(detector.find_head_tail_img(&mask).is_err());
    }

    /// Regression: on a real fish mask the detector used to return points that
    /// drifted far from the labelled snout/fork (see
    /// `tests/fixtures/head_tail_regression/`). Assert each returned endpoint
    /// lands within `TOL` of the labelled keypoint, with the correct
    /// head/tail orientation.
    #[test]
    fn test_find_head_tail_img_matches_labeled_fixture() {
        use ndarray_npy::read_npy;
        use std::path::PathBuf;

        const TOL_PX: f32 = 25.0;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/head_tail_regression");
        let mask: Array2<u8> = read_npy(fixture_dir.join("mask.npy"))
            .expect("mask.npy should load");

        // Labelled keypoints (from coords.json, [x, y] pixel coords).
        let snout = [550.9737_f32, 708.1848];
        let fork = [1533.3898_f32, 656.1633];

        let detector = FishHeadTailDetector {};
        let coords = detector
            .find_head_tail_img(&mask)
            .expect("detector should succeed on fixture mask");

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let head = [coords.head.0[0], coords.head.0[1]];
        let tail = [coords.tail.0[0], coords.tail.0[1]];

        let head_to_snout = dist(head, snout);
        let head_to_fork = dist(head, fork);
        assert!(
            head_to_snout < head_to_fork,
            "head should be closer to the snout than the fork: \
             head={head:?} snout={snout:?} fork={fork:?}"
        );

        assert!(
            head_to_snout <= TOL_PX,
            "head {head:?} is {head_to_snout:.1} px from labelled snout {snout:?} (tol {TOL_PX})"
        );
        let tail_to_fork = dist(tail, fork);
        assert!(
            tail_to_fork <= TOL_PX,
            "tail {tail:?} is {tail_to_fork:.1} px from labelled fork {fork:?} (tol {TOL_PX})"
        );
    }

    /// Regression: on a real fish mask that faces RIGHT (snout on the right
    /// end, fork on the left) the detector used to return points drifting off
    /// the head/tail axis — one near the dorsal-fin tip, one nowhere near
    /// either labelled endpoint. Mirrors the companion fixture
    /// `tests/fixtures/head_tail_regression/` (fish facing LEFT). Asserts
    /// each returned endpoint lands within `TOL` of its labelled keypoint,
    /// with the correct head/tail orientation.
    #[test]
    fn test_find_head_tail_img_snout_right_regression() {
        use ndarray_npy::read_npy;
        use std::path::PathBuf;

        const TOL_PX: f32 = 30.0;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/head_tail_snout_right");
        let mask: Array2<u8> = read_npy(fixture_dir.join("mask.npy"))
            .expect("mask.npy should load");

        // Labelled keypoints (from coords.json, [x, y] pixel coords).
        let snout = [1524.54_f32, 667.867];
        let fork = [349.432_f32, 805.949];

        let detector = FishHeadTailDetector {};
        let coords = detector
            .find_head_tail_img(&mask)
            .expect("detector should succeed on fixture mask");

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let head = [coords.head.0[0], coords.head.0[1]];
        let tail = [coords.tail.0[0], coords.tail.0[1]];

        let head_to_snout = dist(head, snout);
        let head_to_fork = dist(head, fork);
        assert!(
            head_to_snout < head_to_fork,
            "head should be closer to the snout than the fork: \
             head={head:?} snout={snout:?} fork={fork:?}"
        );

        assert!(
            head_to_snout <= TOL_PX,
            "head {head:?} is {head_to_snout:.1} px from labelled snout {snout:?} (tol {TOL_PX})"
        );
        let tail_to_fork = dist(tail, fork);
        assert!(
            tail_to_fork <= TOL_PX,
            "tail {tail:?} is {tail_to_fork:.1} px from labelled fork {fork:?} (tol {TOL_PX})"
        );
    }

    /// Regression: on a real fish mask the classifier used to return endpoint
    /// positions close to the labelled snout/fork but with head and tail
    /// *swapped* — the point returned as `head` lands on the fork and vice
    /// versa. Fixture at `tests/fixtures/head_tail_concavity_swap/` (fish
    /// facing LEFT: snout at the left end, fork at the right). Asserts
    /// each returned endpoint is within `TOL` of its labelled keypoint.
    #[test]
    fn test_find_head_tail_img_head_tail_not_swapped() {
        use ndarray_npy::read_npy;
        use std::path::PathBuf;

        const TOL_PX: f32 = 80.0;

        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/head_tail_concavity_swap");
        let mask: Array2<u8> =
            read_npy(fixture_dir.join("mask.npy")).expect("mask.npy should load");

        // Labelled keypoints (from coords.json, [x, y] pixel coords).
        let snout = [298.7084_f32, 907.3973];
        let fork = [1778.1605_f32, 817.2211];

        let detector = FishHeadTailDetector {};
        let coords = detector
            .find_head_tail_img(&mask)
            .expect("detector should succeed on fixture mask");

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let head = [coords.head.0[0], coords.head.0[1]];
        let tail = [coords.tail.0[0], coords.tail.0[1]];

        let head_to_snout = dist(head, snout);
        let head_to_fork = dist(head, fork);
        assert!(
            head_to_snout < head_to_fork,
            "head/tail swapped: head={head:?} is closer to fork {fork:?} ({head_to_fork:.1} px) \
             than to snout {snout:?} ({head_to_snout:.1} px)"
        );

        let tail_to_fork = dist(tail, fork);
        assert!(
            head_to_snout <= TOL_PX,
            "head {head:?} is {head_to_snout:.1} px from labelled snout {snout:?} (tol {TOL_PX})"
        );
        assert!(
            tail_to_fork <= TOL_PX,
            "tail {tail:?} is {tail_to_fork:.1} px from labelled fork {fork:?} (tol {TOL_PX})"
        );
    }

    /// Regression over a curated subset of the 2026-04-19 bug-report fixture.
    /// These three cases were the motivating examples for switching the
    /// head/tail classifier to the peduncle + hull-area-delta cascade;
    /// they are `likely_swap` failures (PCA endpoints approximately
    /// correct, orientation flipped under the previous classifier).
    ///
    /// Assertions (in-tree, curated cases):
    /// - Orientation: the returned head is strictly closer to the
    ///   labeled snout than to the labeled fork.
    /// - Endpoint proximity: each endpoint lies within
    ///   `max(80, 0.12 * fish_length)` pixels of its label.
    ///
    /// Set `FISHSENSE_BUG_FIXTURE=<path_to_fixture_root>` to sweep the
    /// full 519-case fixture. The sweep covers two disjoint sub-sets:
    ///
    /// - **`likely_swap`**: endpoints are approximately correct, only
    ///   the head/tail *label* is potentially flipped. Assert
    ///   orientation only; ≥25 % pass rate floor.
    /// - **Fork-only `endpoints_wrong`** (filtered as
    ///   `snout_distance_px < 40 && fork_distance_px > 100` in
    ///   `index.json`): head endpoint was already correct under the
    ///   pre-PR detector; this sub-set exists to validate that
    ///   `correct_tail` now reaches the fork notch. Assert fork
    ///   endpoint within `max(80, 0.12 * fish_length)` of label, and
    ///   orientation correct. ≥45 % pass rate floor.
    ///
    /// Floors are set below the current measured pass rates (swap
    /// 35.7 %, fork 51.5 %) so the test catches regressions without
    /// flaking on minor geometry shifts. The residual swap failures
    /// are rockfish-class snout-taper cases where the peduncle min is
    /// interior to the search range; disambiguating them needs a
    /// richer signal than width minima alone and is out of scope for
    /// this PR. The other `endpoints_wrong` sub-sets (snout-only
    /// occlusion, both-wrong mask fragments / PCA-axis tilt) are
    /// also out of scope — upstream mask / PCA failures that
    /// geometry-stage fixes cannot address.
    #[test]
    fn test_find_head_tail_img_bug_report_fixture() {
        use ndarray_npy::read_npy;
        use serde_json::Value;
        use std::path::PathBuf;

        let dist = |a: [f32; 2], b: [f32; 2]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        #[derive(Debug, Clone, Copy)]
        enum Check {
            OrientationOnly,
            StrictEndpoints,
            ForkOnly,
        }

        let run_case = |case_dir: &std::path::Path, check: Check| -> Result<(), String> {
            let mask: Array2<u8> = read_npy(case_dir.join("mask.npy"))
                .map_err(|e| format!("mask.npy load: {e}"))?;
            let coords_raw = std::fs::read_to_string(case_dir.join("coords.json"))
                .map_err(|e| format!("coords.json load: {e}"))?;
            let coords: Value = serde_json::from_str(&coords_raw)
                .map_err(|e| format!("coords.json parse: {e}"))?;
            let snout = [
                coords["expected"]["snout_xy"][0].as_f64().unwrap() as f32,
                coords["expected"]["snout_xy"][1].as_f64().unwrap() as f32,
            ];
            let fork = [
                coords["expected"]["fork_xy"][0].as_f64().unwrap() as f32,
                coords["expected"]["fork_xy"][1].as_f64().unwrap() as f32,
            ];
            let fish_length = dist(snout, fork);
            let endpoint_tol = 80.0_f32.max(0.12 * fish_length);

            let detector = FishHeadTailDetector {};
            let result = detector
                .find_head_tail_img(&mask)
                .map_err(|e| format!("detector err: {e:?}"))?;

            let head = [result.head.0[0], result.head.0[1]];
            let tail = [result.tail.0[0], result.tail.0[1]];
            let head_to_snout = dist(head, snout);
            let head_to_fork = dist(head, fork);
            let tail_to_fork = dist(tail, fork);

            if head_to_snout >= head_to_fork {
                return Err(format!(
                    "orientation: head {head:?} is {head_to_snout:.0} px from snout \
                     vs {head_to_fork:.0} px from fork — should be closer to snout"
                ));
            }
            match check {
                Check::OrientationOnly => {}
                Check::StrictEndpoints => {
                    if head_to_snout > endpoint_tol {
                        return Err(format!(
                            "head {head:?} is {head_to_snout:.0} px from snout \
                             (tol {endpoint_tol:.0} = max(80, 12% of {fish_length:.0} px fish))"
                        ));
                    }
                    if tail_to_fork > endpoint_tol {
                        return Err(format!(
                            "tail {tail:?} is {tail_to_fork:.0} px from fork \
                             (tol {endpoint_tol:.0} = max(80, 12% of {fish_length:.0} px fish))"
                        ));
                    }
                }
                Check::ForkOnly => {
                    if tail_to_fork > endpoint_tol {
                        return Err(format!(
                            "tail {tail:?} is {tail_to_fork:.0} px from fork \
                             (tol {endpoint_tol:.0} = max(80, 12% of {fish_length:.0} px fish))"
                        ));
                    }
                }
            }
            Ok(())
        };

        if let Ok(root) = std::env::var("FISHSENSE_BUG_FIXTURE") {
            let root = PathBuf::from(root);
            let index_raw = std::fs::read_to_string(root.join("index.json"))
                .expect("index.json at fixture root");
            let index: Value = serde_json::from_str(&index_raw).expect("index.json parse");
            let cases = index["cases"].as_array().expect("index.json cases array");

            let swap_cases: Vec<&str> = cases
                .iter()
                .filter(|c| c["failure_mode"].as_str() == Some("likely_swap"))
                .map(|c| c["case"].as_str().unwrap())
                .collect();

            // Fork-only: endpoints_wrong where snout is fine but fork is far.
            let fork_only_cases: Vec<&str> = cases
                .iter()
                .filter(|c| {
                    c["failure_mode"].as_str() == Some("endpoints_wrong")
                        && c["snout_distance_px"].as_f64().unwrap_or(f64::MAX) < 40.0
                        && c["fork_distance_px"].as_f64().unwrap_or(0.0) > 100.0
                })
                .map(|c| c["case"].as_str().unwrap())
                .collect();

            // ── Swap sub-set: orientation only ──
            let mut swap_passes = 0usize;
            let mut swap_failures: Vec<(String, String)> = Vec::new();
            for name in &swap_cases {
                match run_case(&root.join(name), Check::OrientationOnly) {
                    Ok(()) => swap_passes += 1,
                    Err(msg) => swap_failures.push((name.to_string(), msg)),
                }
            }
            let swap_total = swap_cases.len();
            let swap_rate = swap_passes as f64 / swap_total.max(1) as f64;

            // ── Fork-only sub-set: fork endpoint proximity ──
            let mut fork_passes = 0usize;
            let mut fork_failures: Vec<(String, String)> = Vec::new();
            for name in &fork_only_cases {
                match run_case(&root.join(name), Check::ForkOnly) {
                    Ok(()) => fork_passes += 1,
                    Err(msg) => fork_failures.push((name.to_string(), msg)),
                }
            }
            let fork_total = fork_only_cases.len();
            let fork_rate = fork_passes as f64 / fork_total.max(1) as f64;

            println!(
                "[bug_report_fixture] likely_swap orientation pass rate: \
                 {}/{} ({:.1}%)",
                swap_passes,
                swap_total,
                swap_rate * 100.0
            );
            println!(
                "[bug_report_fixture] fork-only endpoints_wrong fork-endpoint pass rate: \
                 {}/{} ({:.1}%)",
                fork_passes,
                fork_total,
                fork_rate * 100.0
            );

            // Floors are well below the measured rates (swap: 5/14 = 35.7%,
            // fork: 17/33 = 51.5%) so minor further regressions don't trip
            // the test; the goal is to catch substantial regressions. The
            // residual swap failures are rockfish-class snout-taper cases
            // where the peduncle min is interior (not at the boundary) —
            // those need a richer signal than width minima alone, out of
            // scope for this PR.
            const MIN_SWAP_RATE: f64 = 0.25;
            const MIN_FORK_RATE: f64 = 0.45;
            assert!(
                swap_rate >= MIN_SWAP_RATE,
                "likely_swap orientation pass rate {:.1}% ({}/{}) below floor {:.0}%. \
                 Sample failures: {:?}",
                swap_rate * 100.0,
                swap_passes,
                swap_total,
                MIN_SWAP_RATE * 100.0,
                swap_failures.iter().take(5).collect::<Vec<_>>()
            );
            assert!(
                fork_rate >= MIN_FORK_RATE,
                "fork-only fork-endpoint pass rate {:.1}% ({}/{}) below floor {:.0}%. \
                 Sample failures: {:?}",
                fork_rate * 100.0,
                fork_passes,
                fork_total,
                MIN_FORK_RATE * 100.0,
                fork_failures.iter().take(5).collect::<Vec<_>>()
            );
        } else {
            let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/bug_report");
            for name in ["case_01", "case_150", "case_273"] {
                run_case(&base.join(name), Check::StrictEndpoints)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            }
        }
    }

    // ── predict_keypoint_depths smoke test ──────────────────────────────
    //
    // The full algorithmic coverage lives in `fish::fish_plane_fit` —
    // this just wires through the detector method.

    /// Detector method delegates to `fish_plane_fit::predict_keypoint_depths`.
    /// Uniform depth in front of an identity camera → both keypoints get the
    /// scene depth back.
    #[test]
    fn predict_keypoint_depths_via_detector_uniform_plane() {
        let detector = FishHeadTailDetector {};
        let depth_value = 0.5_f32;
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth_map = DepthMap(Array2::<f32>::from_elem((40, 60), depth_value));
        let k_inv = ndarray::array![
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let snout = ImageCoord(ndarray::array![10.0_f32, 20.0]);
        let fork = ImageCoord(ndarray::array![50.0_f32, 20.0]);
        let (sd, fd) = detector
            .predict_keypoint_depths(&depth_map, &mask, &k_inv, &snout, &fork)
            .expect("detector should succeed on uniform plane");
        assert!((sd - depth_value).abs() < 1e-3);
        assert!((fd - depth_value).abs() < 1e-3);
    }
}
