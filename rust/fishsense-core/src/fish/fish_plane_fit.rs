//! Mask-bounded RANSAC plane fit in camera 3-D coordinates, used to
//! predict per-keypoint depth from a fish mask + depth map.
//!
//! Replaces the depth-equivalence flood-fill snap with a plane fit
//! that is robust to fish-on-board coplanarity, depth holes inside
//! the fish, and ARKit depth/RGB misregistration. The math model
//! matches the existing `plane_ransac` convention used by
//! `01_process.ipynb` for the scene-tilt diagnostic, but the output
//! is per-keypoint depth (the plane is evaluated along each
//! keypoint's camera ray) rather than a tilt angle.
//!
//! Empirical wins on which this Rust port is justified come from the
//! Python prototype `snap_mask_planefit` in
//! `08_snap_strategy_ab.ipynb` cell `048cde39`:
//!   * mobile bundle (true GT, n=25): median |err| 16.3 → 6.0 mm
//!   * CCFRP (board GT, n=444): median |err| 12.0 → 8.0 mm
//!   * 338 / 469 GT-bearing frames win (72 %)
//!   * ~60 % of systematic positive bias removed on both datasets
//!   * zero RANSAC failures across the 544-frame pool
//!
//! The Rust port matches the prototype's parameters:
//!   * residual threshold 0.005 m
//!   * 100 RANSAC trials, 3-point minimum samples
//!   * inlier least-squares refit on the best trial
//!
//! and uses a fixed-seed xorshift RNG so iOS / Python / CI runs
//! produce identical depths for identical inputs.

use ndarray::Array2;
use tracing::{debug, instrument, warn};

use crate::{
    errors::FishSenseError,
    spatial::types::{DepthMap, ImageCoord},
};

/// RANSAC inlier residual threshold, metres. Matches the Python
/// prototype `snap_mask_planefit` and is well above ARKit per-pixel
/// depth noise (~1 mm at 0.5 m) yet tight enough to reject the
/// support-surface vs. fish-body separation in coplanar scenes.
const RESIDUAL_THRESHOLD_M: f32 = 0.005;

/// Minimum number of mask-and-depth-valid points before RANSAC even
/// runs. Below this the keypoints fall through to local-median.
const MIN_POINTS: usize = 10;

/// Number of RANSAC trials. 100 is plenty for the typical 1k–5k point
/// fish mask: the probability of all 100 trials missing the dominant
/// plane is negligible for inlier ratios above ~50 %.
const RANSAC_TRIALS: usize = 100;

/// Fixed seed for the xorshift RNG that picks RANSAC sample indices.
/// "FSPF" in ASCII; arbitrary but pinned so plane-fit output is
/// reproducible across runs and platforms.
const RNG_SEED: u32 = 0x4653_5046;

/// Half-extent (in depth-grid pixels) of the local-median fallback
/// window. 2 → a 5×5 window, which is generous on a 256×192 ARKit
/// depth grid and trivial on the matched-resolution case.
const LOCAL_MEDIAN_RADIUS_DEPTH_PX: i64 = 2;

/// Predict snout and fork depth by fitting a plane through the
/// masked, depth-valid pixels (in camera 3-D coordinates) and
/// evaluating the plane along the camera ray to each keypoint.
///
/// `snout_xy` and `fork_xy` are in mask (RGB) space — same convention
/// as `find_head_tail_img`. `k_inv` is the 3×3 RGB-space inverse
/// intrinsics matrix. `depth_map` and `mask` may differ in resolution;
/// the plane fit iterates over the depth grid and projects each pixel
/// back into mask space to check the mask.
///
/// Returns `(snout_depth_m, fork_depth_m)` in metres.
///
/// Falls back per-keypoint to a small-window median of valid
/// (mask > 0 AND d > 0) depths when:
///   * `|P| < 10` (RANSAC isn't run at all)
///   * RANSAC fails to find a plane with at least 3 inliers
///   * the camera ray is parallel to the fitted plane
///     (`|1 - a·r₀ - b·r₁| < 1e-9`)
///   * the ray-plane intersection has non-positive depth
///
/// Returns `Err` only when the local-median fallback also fails
/// (no valid depth pixel in the window AND no usable plane) for at
/// least one keypoint — i.e. the inputs are badly degenerate
/// (empty mask, all-zero depth).
#[instrument(
    skip(depth_map, mask, k_inv, snout_xy, fork_xy),
    fields(
        depth_h = depth_map.0.dim().0,
        depth_w = depth_map.0.dim().1,
        mask_h = mask.dim().0,
        mask_w = mask.dim().1,
    ),
)]
pub fn predict_keypoint_depths(
    depth_map: &DepthMap,
    mask: &Array2<u8>,
    k_inv: &Array2<f32>,
    snout_xy: &ImageCoord,
    fork_xy: &ImageCoord,
) -> Result<(f32, f32), FishSenseError> {
    let (mask_h, mask_w) = mask.dim();
    let (depth_h, depth_w) = depth_map.0.dim();
    if mask_h == 0 || mask_w == 0 || depth_h == 0 || depth_w == 0 {
        return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
            "predict_keypoint_depths: empty mask or depth map"
        )));
    }
    if k_inv.shape() != [3, 3] {
        return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
            "k_inv must be 3x3, got {:?}",
            k_inv.shape()
        )));
    }

    let scale_x = depth_w as f32 / mask_w as f32;
    let scale_y = depth_h as f32 / mask_h as f32;

    let points = collect_mask_points(depth_map, mask, k_inv, scale_x, scale_y);
    debug!(n_points = points.len(), "collected masked depth points");

    let plane = if points.len() >= MIN_POINTS {
        ransac_plane(&points, RESIDUAL_THRESHOLD_M, RANSAC_TRIALS)
    } else {
        warn!(
            n_points = points.len(),
            min = MIN_POINTS,
            "fewer than MIN_POINTS valid masked pixels; skipping RANSAC"
        );
        None
    };

    let snout_d = evaluate_or_fallback(plane, k_inv, snout_xy, depth_map, mask, scale_x, scale_y);
    let fork_d = evaluate_or_fallback(plane, k_inv, fork_xy, depth_map, mask, scale_x, scale_y);

    match (snout_d, fork_d) {
        (Some(s), Some(f)) => {
            debug!(snout_d = s, fork_d = f, "predicted keypoint depths");
            Ok((s, f))
        }
        _ => Err(FishSenseError::AnyhowError(anyhow::anyhow!(
            "predict_keypoint_depths: both plane fit and local-median fallback \
             failed for at least one keypoint"
        ))),
    }
}

/// Iterate the depth grid, gather every (X, Y, Z) point whose
/// corresponding mask pixel is set and whose depth is positive and
/// finite. Output is in camera 3-D coordinates (metres).
fn collect_mask_points(
    depth_map: &DepthMap,
    mask: &Array2<u8>,
    k_inv: &Array2<f32>,
    scale_x: f32,
    scale_y: f32,
) -> Vec<[f32; 3]> {
    let (mask_h, mask_w) = mask.dim();
    let (depth_h, depth_w) = depth_map.0.dim();
    let mut points = Vec::new();
    for dv in 0..depth_h {
        for du in 0..depth_w {
            let d = depth_map.0[[dv, du]];
            if !(d > 0.0 && d.is_finite()) {
                continue;
            }
            let mu_f = du as f32 / scale_x;
            let mv_f = dv as f32 / scale_y;
            let mu = (mu_f.round() as usize).min(mask_w - 1);
            let mv = (mv_f.round() as usize).min(mask_h - 1);
            if mask[[mv, mu]] == 0 {
                continue;
            }
            let r = back_project_ray(k_inv, mu_f, mv_f);
            points.push([r[0] * d, r[1] * d, d]);
        }
    }
    points
}

/// `K⁻¹ · [u, v, 1]ᵀ` — the unit (depth-free) camera ray for a pixel.
fn back_project_ray(k_inv: &Array2<f32>, u: f32, v: f32) -> [f32; 3] {
    [
        k_inv[[0, 0]] * u + k_inv[[0, 1]] * v + k_inv[[0, 2]],
        k_inv[[1, 0]] * u + k_inv[[1, 1]] * v + k_inv[[1, 2]],
        k_inv[[2, 0]] * u + k_inv[[2, 1]] * v + k_inv[[2, 2]],
    ]
}

/// Evaluate the plane along the camera ray to `kp_xy`, falling back
/// to the local-median window when the plane is missing or the ray-
/// plane intersection is degenerate / non-positive.
fn evaluate_or_fallback(
    plane: Option<(f32, f32, f32)>,
    k_inv: &Array2<f32>,
    kp_xy: &ImageCoord,
    depth_map: &DepthMap,
    mask: &Array2<u8>,
    scale_x: f32,
    scale_y: f32,
) -> Option<f32> {
    let u = kp_xy.0[0];
    let v = kp_xy.0[1];
    if let Some((a, b, c)) = plane {
        // Z = aX + bY + c, with X = r₀·Z, Y = r₁·Z gives
        //   Z·(1 - a·r₀ - b·r₁) = c   →   Z = c / (1 - a·r₀ - b·r₁)
        let r = back_project_ray(k_inv, u, v);
        let denom = 1.0 - a * r[0] - b * r[1];
        if denom.abs() >= 1e-9 {
            let d = c / denom;
            if d > 0.0 && d.is_finite() {
                return Some(d);
            }
        }
        warn!(
            u, v, a, b, c, denom,
            "plane evaluation degenerate or non-positive; falling back to local median"
        );
    }
    local_median_depth(depth_map, mask, u, v, scale_x, scale_y)
}

/// Median of valid (mask > 0 AND d > 0) depths in a 5×5 window
/// around the keypoint's depth-grid pixel. `None` when the window
/// contains no valid sample.
fn local_median_depth(
    depth_map: &DepthMap,
    mask: &Array2<u8>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
) -> Option<f32> {
    let (depth_h, depth_w) = depth_map.0.dim();
    let (mask_h, mask_w) = mask.dim();
    let du_c = (u * scale_x).round() as i64;
    let dv_c = (v * scale_y).round() as i64;
    let mut samples: Vec<f32> = Vec::new();
    for ddv in -LOCAL_MEDIAN_RADIUS_DEPTH_PX..=LOCAL_MEDIAN_RADIUS_DEPTH_PX {
        for ddu in -LOCAL_MEDIAN_RADIUS_DEPTH_PX..=LOCAL_MEDIAN_RADIUS_DEPTH_PX {
            let du = du_c + ddu;
            let dv = dv_c + ddv;
            if du < 0 || dv < 0 || du >= depth_w as i64 || dv >= depth_h as i64 {
                continue;
            }
            let du = du as usize;
            let dv = dv as usize;
            let d = depth_map.0[[dv, du]];
            if !(d > 0.0 && d.is_finite()) {
                continue;
            }
            let mu = ((du as f32 / scale_x).round() as i64).clamp(0, mask_w as i64 - 1) as usize;
            let mv = ((dv as f32 / scale_y).round() as i64).clamp(0, mask_h as i64 - 1) as usize;
            if mask[[mv, mu]] == 0 {
                continue;
            }
            samples.push(d);
        }
    }
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    let med = if n.is_multiple_of(2) {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    } else {
        samples[n / 2]
    };
    Some(med)
}

/// RANSAC plane fit `Z = aX + bY + c` on a 3-D point cloud.
///
/// Picks 3 random points per trial, fits the exact plane, counts
/// inliers (residual ≤ `threshold`), keeps the trial with the most
/// inliers, and refits via least squares on that inlier set.
/// Returns `None` when no trial collects ≥ 3 inliers.
fn ransac_plane(
    points: &[[f32; 3]],
    threshold: f32,
    n_trials: usize,
) -> Option<(f32, f32, f32)> {
    if points.len() < 3 {
        return None;
    }
    let mut rng = XorShift32::new(RNG_SEED);
    let mut best: Option<(f32, f32, f32, usize)> = None;
    for _ in 0..n_trials {
        let i = rng.gen_range(points.len());
        let mut j = rng.gen_range(points.len());
        while j == i {
            j = rng.gen_range(points.len());
        }
        let mut k = rng.gen_range(points.len());
        while k == i || k == j {
            k = rng.gen_range(points.len());
        }
        let (a, b, c) = match plane_from_three(&points[i], &points[j], &points[k]) {
            Some(p) => p,
            None => continue,
        };
        let mut inliers = 0usize;
        for p in points {
            let z_pred = a * p[0] + b * p[1] + c;
            if (z_pred - p[2]).abs() <= threshold {
                inliers += 1;
            }
        }
        if inliers < 3 {
            continue;
        }
        if best.is_none_or(|(_, _, _, n)| inliers > n) {
            best = Some((a, b, c, inliers));
        }
    }
    let (a, b, c, _) = best?;
    let mut inlier_pts: Vec<[f32; 3]> = Vec::new();
    for p in points {
        let z_pred = a * p[0] + b * p[1] + c;
        if (z_pred - p[2]).abs() <= threshold {
            inlier_pts.push(*p);
        }
    }
    least_squares_plane(&inlier_pts).or(Some((a, b, c)))
}

/// Exact plane through three (X, Y, Z) points. Returns `None` when
/// the points project to a degenerate (X, Y) configuration (collinear
/// in image space) and the 3×3 system is singular.
fn plane_from_three(p1: &[f32; 3], p2: &[f32; 3], p3: &[f32; 3]) -> Option<(f32, f32, f32)> {
    use nalgebra::{Matrix3, Vector3};
    let m = Matrix3::new(
        p1[0], p1[1], 1.0,
        p2[0], p2[1], 1.0,
        p3[0], p3[1], 1.0,
    );
    let z = Vector3::new(p1[2], p2[2], p3[2]);
    let x = m.lu().solve(&z)?;
    if !(x[0].is_finite() && x[1].is_finite() && x[2].is_finite()) {
        return None;
    }
    Some((x[0], x[1], x[2]))
}

/// Least-squares plane fit `Z = aX + bY + c` via normal equations on
/// the inlier set. `points` must have at least 3 entries.
fn least_squares_plane(points: &[[f32; 3]]) -> Option<(f32, f32, f32)> {
    if points.len() < 3 {
        return None;
    }
    use nalgebra::{Matrix3, Vector3};
    let mut ata = Matrix3::<f32>::zeros();
    let mut atb = Vector3::<f32>::zeros();
    for p in points {
        let r = Vector3::new(p[0], p[1], 1.0);
        ata += r * r.transpose();
        atb += r * p[2];
    }
    let x = ata.lu().solve(&atb)?;
    if !(x[0].is_finite() && x[1].is_finite() && x[2].is_finite()) {
        return None;
    }
    Some((x[0], x[1], x[2]))
}

/// 32-bit xorshift RNG. Tiny, deterministic, sufficient for picking
/// RANSAC sample indices uniformly. Pinned seed makes plane-fit
/// output reproducible across iOS / Python / CI for identical inputs.
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Identity intrinsics: `K⁻¹·[u, v, 1] = [u, v, 1]` and the
    /// camera-3D X/Y axes coincide with image u/v scaled by depth.
    fn identity_k_inv() -> Array2<f32> {
        array![
            [1.0_f32, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Frontoparallel scene: a rectangular mask over a uniform-depth
    /// region. Both keypoints should recover the exact plane depth.
    #[test]
    fn frontoparallel_plane_returns_uniform_depth() {
        let depth_value = 0.5_f32;
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth = Array2::<f32>::from_elem((40, 60), depth_value);
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("plane fit must succeed on uniform depth");

        assert!(
            (sd - depth_value).abs() < 1e-3,
            "snout depth {sd} drifted from plane value {depth_value}"
        );
        assert!(
            (fd - depth_value).abs() < 1e-3,
            "fork depth {fd} drifted from plane value {depth_value}"
        );
    }

    /// Tilted plane: depth varies linearly across the image. With
    /// identity K_inv the camera rays grow as `[u, v, 1]`, so plane
    /// slopes (a, b) must be small enough that `1 - a·u - b·v` stays
    /// positive across the full image — otherwise the rendered depth
    /// map flips sign and the test setup is invalid.
    #[test]
    fn tilted_plane_recovered_at_keypoints() {
        // Plane in 3-D: Z = 0.005·X − 0.002·Y + 0.6 .
        // With u ∈ [0, 60), v ∈ [0, 40) and identity K_inv, the
        // denominator stays in [0.7, 1.08] → all rendered depths
        // positive and finite.
        let (a_true, b_true, c_true) = (0.005_f32, -0.002_f32, 0.6_f32);
        let k_inv = identity_k_inv();
        let (h, w) = (40usize, 60usize);
        let mut depth = Array2::<f32>::zeros((h, w));
        for v in 0..h {
            for u in 0..w {
                let r = back_project_ray(&k_inv, u as f32, v as f32);
                let denom = 1.0 - a_true * r[0] - b_true * r[1];
                let d = c_true / denom;
                depth[[v, u]] = d;
            }
        }
        let mut mask = Array2::<u8>::zeros((h, w));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &k_inv, &snout, &fork)
                .expect("tilted plane fit must succeed");

        // Expected: plug each ray into the same closed form.
        let snout_r = back_project_ray(&k_inv, 10.0, 20.0);
        let fork_r = back_project_ray(&k_inv, 50.0, 20.0);
        let snout_expected = c_true / (1.0 - a_true * snout_r[0] - b_true * snout_r[1]);
        let fork_expected = c_true / (1.0 - a_true * fork_r[0] - b_true * fork_r[1]);
        assert!(
            (sd - snout_expected).abs() < 1e-3,
            "snout {sd} should match closed-form {snout_expected}"
        );
        assert!(
            (fd - fork_expected).abs() < 1e-3,
            "fork {fd} should match closed-form {fork_expected}"
        );
    }

    /// LiDAR-style holes: scattered zero pixels inside the mask must
    /// not poison RANSAC. The plane fit should still recover the
    /// underlying depth.
    #[test]
    fn ransac_robust_to_internal_depth_holes() {
        let depth_value = 0.7_f32;
        let mut depth = Array2::<f32>::from_elem((40, 60), depth_value);
        // Punch holes inside the masked region.
        for &(r, c) in &[(15, 10), (16, 11), (20, 25), (25, 40), (28, 50)] {
            depth[[r, c]] = 0.0;
        }
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("hole-punctured plane fit must still succeed");
        assert!((sd - depth_value).abs() < 1e-3);
        assert!((fd - depth_value).abs() < 1e-3);
    }

    /// Mismatched grids: 2× downsampled depth map, plane still
    /// recovered. Pre-condition for the iOS path (1920×1440 RGB +
    /// 256×192 LiDAR depth).
    #[test]
    fn mismatched_grids_uniform_depth() {
        let depth_value = 0.42_f32;
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 16..24 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth = Array2::<f32>::from_elem((20, 30), depth_value);
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("mismatched-grid plane fit must succeed");
        assert!((sd - depth_value).abs() < 1e-3);
        assert!((fd - depth_value).abs() < 1e-3);
    }

    /// Determinism: identical inputs produce identical depths across
    /// repeated calls (seeded RNG, no platform randomness).
    #[test]
    fn predict_keypoint_depths_is_deterministic() {
        let mut depth = Array2::<f32>::zeros((40, 60));
        // Mild noise pattern so RANSAC trials genuinely pick different
        // 3-tuples each iteration.
        for v in 0..40 {
            for u in 0..60 {
                depth[[v, u]] = 0.5 + 0.001 * ((u + v) as f32).sin();
            }
        }
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let a = predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
            .unwrap();
        let b = predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
            .unwrap();
        assert_eq!(a, b, "RANSAC must be deterministic across calls");
    }

    /// `|P| < MIN_POINTS` triggers the per-keypoint local-median
    /// fallback. With a single masked pixel at known depth the
    /// fallback returns that depth.
    #[test]
    fn fewer_than_min_points_falls_back_to_local_median() {
        let mut mask = Array2::<u8>::zeros((10, 10));
        mask[[5, 5]] = 1;
        let mut depth = Array2::<f32>::zeros((10, 10));
        depth[[5, 5]] = 0.33;
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![5.0_f32, 5.0]);
        let fork = ImageCoord(array![5.0_f32, 5.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("local-median fallback must succeed when a valid sample exists");
        assert!((sd - 0.33).abs() < 1e-6);
        assert!((fd - 0.33).abs() < 1e-6);
    }

    /// Empty mask + zero depth → both plane fit and local median
    /// fail; function returns `Err`.
    #[test]
    fn empty_mask_returns_err() {
        let mask = Array2::<u8>::zeros((10, 10));
        let depth_map = DepthMap(Array2::<f32>::zeros((10, 10)));
        let snout = ImageCoord(array![5.0_f32, 5.0]);
        let fork = ImageCoord(array![5.0_f32, 5.0]);
        let result = predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork);
        assert!(result.is_err(), "empty inputs must return Err");
    }

    /// Bad K_inv shape → Err.
    #[test]
    fn rejects_non_3x3_k_inv() {
        let mask = Array2::<u8>::from_elem((4, 4), 1u8);
        let depth_map = DepthMap(Array2::<f32>::from_elem((4, 4), 1.0));
        let snout = ImageCoord(array![1.0_f32, 1.0]);
        let fork = ImageCoord(array![3.0_f32, 3.0]);
        let bad_k = Array2::<f32>::zeros((2, 3));
        let result = predict_keypoint_depths(&depth_map, &mask, &bad_k, &snout, &fork);
        assert!(result.is_err());
    }

    /// RANSAC must reject outliers and recover the dominant plane.
    /// 80 % of mask pixels lie on a uniform-depth plane; 20 % have
    /// 5× the depth (background bleed through a loose mask). Plane
    /// fit should still return the inlier depth at the keypoints,
    /// well within the residual threshold.
    #[test]
    fn ransac_rejects_background_bleed_outliers() {
        let inlier_depth = 0.5_f32;
        let outlier_depth = 2.5_f32;
        let (h, w) = (40usize, 60usize);
        let mut depth = Array2::<f32>::from_elem((h, w), inlier_depth);
        let mut mask = Array2::<u8>::zeros((h, w));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        // Inject ~20 % outlier depths inside the mask.
        for r in 10..30 {
            for c in 5..55 {
                if (r * 13 + c * 7) % 5 == 0 {
                    depth[[r, c]] = outlier_depth;
                }
            }
        }
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("RANSAC must succeed in the presence of outliers");
        // Plane should snap to the inlier value, not be pulled toward
        // the outliers (which would land somewhere between 0.5 and 2.5).
        assert!(
            (sd - inlier_depth).abs() < RESIDUAL_THRESHOLD_M,
            "snout depth {sd} should track inlier {inlier_depth}, not outlier {outlier_depth}"
        );
        assert!(
            (fd - inlier_depth).abs() < RESIDUAL_THRESHOLD_M,
            "fork depth {fd} should track inlier {inlier_depth}, not outlier {outlier_depth}"
        );
    }

    /// Realistic camera intrinsics (focal lengths and principal point
    /// on the order of an iPhone front camera). The fit must work
    /// independently of K_inv; identity is just a convenience.
    #[test]
    fn realistic_intrinsics_recover_plane() {
        // K_inv consistent with f≈1400 px, principal point ≈ image
        // centre for an 1920×1440 sensor.
        let fx_inv = 1.0 / 1400.0_f32;
        let fy_inv = 1.0 / 1400.0_f32;
        let cx = 960.0_f32;
        let cy = 720.0_f32;
        let k_inv = array![
            [fx_inv, 0.0, -cx * fx_inv],
            [0.0, fy_inv, -cy * fy_inv],
            [0.0, 0.0, 1.0],
        ];

        // Build a tilted plane in 3-D, render it into the depth map,
        // run the fit, and check the per-keypoint depth matches the
        // closed-form ray-plane intersection.
        let (a_true, b_true, c_true) = (0.05_f32, -0.02_f32, 0.55_f32);
        let (h, w) = (192usize, 256usize);
        let mut depth = Array2::<f32>::zeros((h, w));
        // The depth grid is 256×192 but K_inv is in mask-space (1920×1440),
        // so we render via the same back-projection convention used in
        // collect_mask_points: u_mask = u_depth / scale_x.
        let mask_w = 1920usize;
        let mask_h = 1440usize;
        let scale_x = w as f32 / mask_w as f32;
        let scale_y = h as f32 / mask_h as f32;
        for v in 0..h {
            for u in 0..w {
                let mu = u as f32 / scale_x;
                let mv = v as f32 / scale_y;
                let r = back_project_ray(&k_inv, mu, mv);
                let denom = 1.0 - a_true * r[0] - b_true * r[1];
                depth[[v, u]] = c_true / denom;
            }
        }
        let mut mask = Array2::<u8>::zeros((mask_h, mask_w));
        // Mask covers a fish-shaped region in the centre.
        for r in 600..900 {
            for c in 400..1500 {
                mask[[r, c]] = 1;
            }
        }
        let depth_map = DepthMap(depth);

        let snout = ImageCoord(array![450.0_f32, 720.0]);
        let fork = ImageCoord(array![1450.0_f32, 720.0]);
        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &k_inv, &snout, &fork)
                .expect("plane fit must succeed with realistic intrinsics");

        let snout_r = back_project_ray(&k_inv, 450.0, 720.0);
        let fork_r = back_project_ray(&k_inv, 1450.0, 720.0);
        let snout_expected = c_true / (1.0 - a_true * snout_r[0] - b_true * snout_r[1]);
        let fork_expected = c_true / (1.0 - a_true * fork_r[0] - b_true * fork_r[1]);
        assert!(
            (sd - snout_expected).abs() < 5e-3,
            "snout {sd} drifted from closed-form {snout_expected} (>5mm)"
        );
        assert!(
            (fd - fork_expected).abs() < 5e-3,
            "fork {fd} drifted from closed-form {fork_expected} (>5mm)"
        );
    }

    /// Anisotropic scaling: the depth grid is half-resolution in y
    /// but full-resolution in x. Ensures the per-axis scale factors
    /// don't get accidentally collapsed into a single number.
    #[test]
    fn anisotropic_scale_uniform_depth() {
        let depth_value = 0.6_f32;
        let mut mask = Array2::<u8>::zeros((40, 60));
        for r in 10..30 {
            for c in 5..55 {
                mask[[r, c]] = 1;
            }
        }
        let depth = Array2::<f32>::from_elem((20, 60), depth_value);
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![10.0_f32, 20.0]);
        let fork = ImageCoord(array![50.0_f32, 20.0]);

        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("anisotropic plane fit must succeed");
        assert!((sd - depth_value).abs() < 1e-3);
        assert!((fd - depth_value).abs() < 1e-3);
    }

    /// Local-median window: the keypoint's depth-grid pixel itself
    /// has invalid depth, but neighbouring mask pixels do. Median is
    /// taken over those valid samples only.
    #[test]
    fn local_median_skips_invalid_centre_pixel() {
        let mut mask = Array2::<u8>::zeros((10, 10));
        for r in 4..7 {
            for c in 4..7 {
                mask[[r, c]] = 1;
            }
        }
        let mut depth = Array2::<f32>::zeros((10, 10));
        // Centre pixel (5, 5) has d = 0 (invalid). 8 neighbours have
        // a known valid depth; median should equal that depth.
        for r in 4..7 {
            for c in 4..7 {
                if (r, c) != (5, 5) {
                    depth[[r, c]] = 0.42;
                }
            }
        }
        let depth_map = DepthMap(depth);

        // Force the local-median path by giving < MIN_POINTS valid
        // samples globally (8 < 10).
        let snout = ImageCoord(array![5.0_f32, 5.0]);
        let fork = ImageCoord(array![5.0_f32, 5.0]);
        let (sd, fd) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("local median should succeed");
        assert!((sd - 0.42).abs() < 1e-6);
        assert!((fd - 0.42).abs() < 1e-6);
    }

    /// Local-median across an even number of samples: returns the
    /// arithmetic mean of the two middle values.
    #[test]
    fn local_median_even_count_returns_mean_of_middle_two() {
        let mut mask = Array2::<u8>::zeros((10, 10));
        // Two masked pixels with distinct depths.
        mask[[5, 5]] = 1;
        mask[[5, 6]] = 1;
        let mut depth = Array2::<f32>::zeros((10, 10));
        depth[[5, 5]] = 0.4;
        depth[[5, 6]] = 0.6;
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![5.0_f32, 5.0]);
        let fork = ImageCoord(array![5.0_f32, 5.0]);
        let (sd, _) =
            predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork)
                .expect("local median (even count) should succeed");
        // Median of {0.4, 0.6} = 0.5.
        assert!((sd - 0.5).abs() < 1e-6, "expected 0.5, got {sd}");
    }

    /// Keypoint near the image edge: clamping in the fallback window
    /// must not panic, and a partial window of valid samples must
    /// still produce a sensible depth.
    #[test]
    fn keypoint_at_image_corner_does_not_panic() {
        let mut mask = Array2::<u8>::zeros((20, 20));
        mask[[0, 0]] = 1;
        let mut depth = Array2::<f32>::zeros((20, 20));
        depth[[0, 0]] = 0.55;
        let depth_map = DepthMap(depth);
        let snout = ImageCoord(array![0.0_f32, 0.0]);
        let fork = ImageCoord(array![19.0_f32, 19.0]);
        // Only one valid sample → < MIN_POINTS, both keypoints fall
        // through to local median. The fork keypoint's window
        // contains no valid samples, so it should fail.
        let result = predict_keypoint_depths(&depth_map, &mask, &identity_k_inv(), &snout, &fork);
        assert!(
            result.is_err(),
            "fork at (19, 19) has no valid samples in its window — must Err, not panic"
        );
    }

    /// `plane_from_three`: three points collinear in (X, Y) → returns
    /// `None` (under-determined system). Direct unit test.
    #[test]
    fn plane_from_three_rejects_collinear_xy() {
        // All three (X, Y) on the line Y = X.
        let p1 = [0.0_f32, 0.0, 1.0];
        let p2 = [1.0_f32, 1.0, 1.0];
        let p3 = [2.0_f32, 2.0, 1.0];
        let result = plane_from_three(&p1, &p2, &p3);
        assert!(
            result.is_none(),
            "collinear (X, Y) points should not yield a plane"
        );
    }

    /// `least_squares_plane`: closed-form recovery for points that
    /// exactly satisfy the plane equation. Uses a 2-D grid in (X, Y)
    /// so the columns of A = [X, Y, 1] are linearly independent and
    /// the normal-equations matrix has full rank.
    #[test]
    fn least_squares_plane_recovers_exact_plane() {
        // Plane Z = 0.3·X + 0.1·Y + 0.7
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for i in 0..5 {
            for j in 0..4 {
                let x = i as f32 * 0.2;
                let y = j as f32 * 0.3;
                let z = 0.3 * x + 0.1 * y + 0.7;
                pts.push([x, y, z]);
            }
        }
        let (a, b, c) =
            least_squares_plane(&pts).expect("least squares must succeed on a real plane");
        assert!((a - 0.3).abs() < 1e-4, "a: {a}");
        assert!((b - 0.1).abs() < 1e-4, "b: {b}");
        assert!((c - 0.7).abs() < 1e-4, "c: {c}");
    }

    /// XorShift32 produces a non-trivial sequence and the fixed seed
    /// makes that sequence reproducible. Pins the seed contract that
    /// guarantees deterministic plane-fit output.
    #[test]
    fn xorshift_is_seeded_and_reproducible() {
        let mut a = XorShift32::new(RNG_SEED);
        let mut b = XorShift32::new(RNG_SEED);
        let seq_a: Vec<u32> = (0..16).map(|_| a.next_u32()).collect();
        let seq_b: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
        assert_eq!(seq_a, seq_b, "same seed → same sequence");
        // Trivial smoke check: not all zero, not all the same.
        let unique = seq_a.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(unique > 8, "RNG output looks degenerate: {seq_a:?}");
    }
}
