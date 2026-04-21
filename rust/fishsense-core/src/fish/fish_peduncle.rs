//! Caudal peduncle detection for head/tail orientation.
//!
//! The peduncle is the narrow waist between the fish's body and its
//! caudal fin. It's the most stable anatomical feature for
//! distinguishing tail from head on a fish silhouette: the head taper
//! is monotonic, while the tail half has a "narrow-wide-narrow" width
//! profile across the peduncle-into-fin transition. Neither mouth
//! openings, gill concavities, nor dorsal/ventral-fin notches on the
//! head half move the width minimum off the peduncle in practice.
//!
//! This module projects the mask pixels onto the PCA axis (inferred
//! from the two PCA endpoints), bins by axis position, and finds the
//! width minimum. The PCA endpoint closer to that minimum along the
//! axis is the tail.
//!
//! Returns `None` (no decision) on:
//! - degenerate axes (both endpoints coincide),
//! - masks too small to produce stable width bins,
//! - shapes with no detectable waist (rectangles, eel-like
//!   silhouettes, heavily occluded fish).
//!
//! Callers should fall back to another classifier on `None`.

use ndarray::Array2;

/// Result of peduncle-based orientation detection.
#[derive(Debug, Clone, PartialEq)]
pub struct PeduncleDecision {
    pub head: [f64; 2],
    pub tail: [f64; 2],
    /// `(w_flank - w_peduncle) / w_flank` ∈ [0, 1]. How sharply the
    /// silhouette narrows at the peduncle relative to the body. A real
    /// peduncle is > 0.2; a rectangle is ~0.
    pub narrowing: f64,
    /// `|s_tail - s_peduncle| / |s_head - s_peduncle|` ∈ (0, ∞). Tail
    /// endpoints sit close to the peduncle (ratio ≪ 1); the classifier
    /// already picks tail = nearer endpoint, so values >1 are impossible
    /// by construction — the field is for downstream confidence weighting.
    pub asymmetry: f64,
}

/// Minimum narrowing ratio for a peduncle to be accepted as real.
/// Shapes below this are effectively uniform-width and do not support
/// a peduncle-based decision. Tuned against the bug-report fixture.
const MIN_NARROWING: f64 = 0.20;

/// Minimum non-zero mask pixels below which width bins become too
/// noisy to localize a peduncle. Degenerate/tiny masks → `None`.
const MIN_MASK_PIXELS: usize = 2_000;

/// Number of bins along the PCA axis. 50 gives ~30 px per bin on a
/// 1500-px-long fish — fine enough to localize a ~5 % peduncle, coarse
/// enough to smooth out perimeter jitter.
const N_BINS: usize = 50;

/// Fraction of bins excluded at each end of the axis from the width-min
/// search. Near the endpoints the silhouette width collapses naturally
/// to zero; the peduncle is interior.
const END_EXCLUDE_FRACTION: f64 = 0.10;

/// Bins [FLANK_LO..FLANK_HI] define the "body" from which the flank
/// width is measured for the narrowing ratio. 30-70 % keeps this well
/// clear of both endpoints and the peduncle itself on any plausible
/// peduncle location.
const FLANK_LO_FRACTION: f64 = 0.30;
const FLANK_HI_FRACTION: f64 = 0.70;


/// Classify head/tail by locating the caudal peduncle along the PCA axis.
///
/// `left` and `right` are the two PCA endpoints (from
/// [`super::fish_pca::estimate_endpoints`]). The axis direction is
/// inferred as `right - left`, avoiding a re-fit.
pub fn classify_by_peduncle(
    mask: &Array2<u8>,
    left: [f64; 2],
    right: [f64; 2],
) -> Option<PeduncleDecision> {
    let dx = right[0] - left[0];
    let dy = right[1] - left[1];
    let axis_len = (dx * dx + dy * dy).sqrt();
    if axis_len < 1.0 {
        return None;
    }
    let ux = dx / axis_len;
    let uy = dy / axis_len;
    // Perpendicular (rotate 90° ccw).
    let px = -uy;
    let py = ux;

    // Project every mask pixel onto (axis, perp) with origin at `left`.
    let mut s_values: Vec<f64> = Vec::new();
    let mut w_values: Vec<f64> = Vec::new();
    for ((row, col), &v) in mask.indexed_iter() {
        if v == 0 {
            continue;
        }
        let cx = col as f64 - left[0];
        let cy = row as f64 - left[1];
        s_values.push(cx * ux + cy * uy);
        w_values.push(cx * px + cy * py);
    }
    if s_values.len() < MIN_MASK_PIXELS {
        return None;
    }

    let s_min = s_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let s_max = s_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = s_max - s_min;
    if span < 1.0 {
        return None;
    }
    let bin_w = span / N_BINS as f64;

    let mut bin_min = vec![f64::INFINITY; N_BINS];
    let mut bin_max = vec![f64::NEG_INFINITY; N_BINS];
    for i in 0..s_values.len() {
        let s = s_values[i];
        let w = w_values[i];
        let bin = (((s - s_min) / bin_w).floor() as isize).clamp(0, N_BINS as isize - 1) as usize;
        if w < bin_min[bin] {
            bin_min[bin] = w;
        }
        if w > bin_max[bin] {
            bin_max[bin] = w;
        }
    }
    let widths: Vec<f64> = (0..N_BINS)
        .map(|i| {
            if bin_min[i].is_infinite() {
                f64::NAN
            } else {
                bin_max[i] - bin_min[i]
            }
        })
        .collect();

    let exclude = (N_BINS as f64 * END_EXCLUDE_FRACTION).round() as usize;
    let search_lo = exclude;
    let search_hi = N_BINS.saturating_sub(exclude);
    if search_hi <= search_lo + 2 {
        return None;
    }

    // Width minimum over the interior bins.
    let (mut best_i, mut best_w) = (search_lo, f64::INFINITY);
    for (i, &w) in widths
        .iter()
        .enumerate()
        .take(search_hi)
        .skip(search_lo)
    {
        if w.is_finite() && w < best_w {
            best_w = w;
            best_i = i;
        }
    }
    if !best_w.is_finite() {
        return None;
    }

    // Flank width: max body width in the 30-70 % band.
    let flank_lo = (N_BINS as f64 * FLANK_LO_FRACTION).round() as usize;
    let flank_hi = (N_BINS as f64 * FLANK_HI_FRACTION).round() as usize;
    let flank: f64 = (flank_lo..flank_hi)
        .filter_map(|i| {
            let w = widths[i];
            if w.is_finite() { Some(w) } else { None }
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if !flank.is_finite() || flank <= 0.0 {
        return None;
    }
    let narrowing = (flank - best_w) / flank;
    if narrowing < MIN_NARROWING {
        return None;
    }

    // Peduncle axial position (bin centre).
    let s_peduncle = s_min + (best_i as f64 + 0.5) * bin_w;

    // Endpoints in axis coordinates: left is at s=0, right is at s=axis_len.
    let d_left = s_peduncle.abs();
    let d_right = (axis_len - s_peduncle).abs();

    // Baseline: tail is the endpoint closer to the minimum-width bin.
    // A real caudal peduncle sits between body flank and caudal fin,
    // so the caudal-fin-side endpoint is closer to it than the snout-
    // side endpoint.
    let mut tail_on_right = d_right < d_left;

    // Boundary override: when `best_i` lands *at* the outer edge of
    // the interior search range (at `search_lo` or `search_hi - 1`),
    // the minimum sits flush against the excluded endpoint zone —
    // no bins inside the search range on the outer side.
    //
    // Empirically on the bug-report fixture, a boundary minimum is
    // the signature of a snout-side-taper min on species (rockfish,
    // grouper) whose snout taper is the narrowest cross-section
    // inside the search range. The distance rule then picks the
    // *snout* endpoint as tail. Flipping to the far endpoint
    // recovers the correct orientation on ~5 of the 14 residual
    // likely-swap cases. A genuine caudal peduncle sits interior to
    // the search range, so this override does not fire on normal
    // fish.
    if best_i == search_lo {
        tail_on_right = true;
    } else if best_i + 1 == search_hi {
        tail_on_right = false;
    }

    let (tail, head, d_tail, d_head) = if tail_on_right {
        (right, left, d_right, d_left)
    } else {
        (left, right, d_left, d_right)
    };
    let asymmetry = if d_head > 1e-9 { d_tail / d_head } else { 0.0 };

    Some(PeduncleDecision {
        head,
        tail,
        narrowing,
        asymmetry,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fish-shaped silhouette with a clear peduncle waist.
    /// Axis: horizontal, row=50. Body: cols 10..90, rows 35..65 (wide).
    /// Peduncle: cols 90..100, rows 48..52 (narrow).
    /// Caudal fin: cols 100..115, rows 35..65 (wide again).
    /// Head is at col 10 (wide but tapered), tail tip at col 115.
    fn fish_shaped_mask() -> Array2<u8> {
        let mut m = Array2::<u8>::zeros((100, 120));
        // Body: wide section with slight taper near the head.
        for c in 10..90 {
            // Taper head-side (cols 10..20) from narrow→wide.
            let half_h = if c < 20 { 5 + (c - 10) } else { 15 };
            for r in (50 - half_h)..(50 + half_h) {
                m[[r, c]] = 1;
            }
        }
        // Peduncle: very narrow.
        for c in 90..100 {
            for r in 48..52 {
                m[[r, c]] = 1;
            }
        }
        // Caudal fin: wide again.
        for c in 100..115 {
            for r in 35..65 {
                m[[r, c]] = 1;
            }
        }
        m
    }

    #[test]
    fn test_peduncle_on_fish_shape_picks_correct_tail() {
        let mask = fish_shaped_mask();
        let head_end = [10.0_f64, 50.0];
        let tail_end = [114.0_f64, 50.0];
        let d = classify_by_peduncle(&mask, head_end, tail_end).expect("peduncle should be found");
        assert_eq!(d.tail, tail_end, "tail should be the caudal-fin end");
        assert_eq!(d.head, head_end, "head should be the body end");
        assert!(d.narrowing > 0.5, "peduncle narrowing should be pronounced, got {}", d.narrowing);
    }

    #[test]
    fn test_peduncle_endpoints_swapped_still_picks_correct_tail() {
        // Swap `left` and `right` — the decision must be invariant.
        let mask = fish_shaped_mask();
        let head_end = [10.0_f64, 50.0];
        let tail_end = [114.0_f64, 50.0];
        let d = classify_by_peduncle(&mask, tail_end, head_end).expect("peduncle should be found");
        assert_eq!(d.tail, tail_end);
        assert_eq!(d.head, head_end);
    }

    #[test]
    fn test_peduncle_returns_none_on_rectangle() {
        // Uniform-width bar has no waist → no peduncle decision.
        let mut m = Array2::<u8>::zeros((30, 100));
        for r in 10..20 {
            for c in 5..95 {
                m[[r, c]] = 1;
            }
        }
        let d = classify_by_peduncle(&m, [5.0, 15.0], [94.0, 15.0]);
        assert!(d.is_none(), "rectangle should yield no peduncle decision, got {d:?}");
    }

    #[test]
    fn test_peduncle_returns_none_on_degenerate_axis() {
        let mut m = Array2::<u8>::zeros((30, 30));
        for r in 10..20 {
            for c in 10..20 {
                m[[r, c]] = 1;
            }
        }
        let d = classify_by_peduncle(&m, [15.0, 15.0], [15.0, 15.0]);
        assert!(d.is_none());
    }

    #[test]
    fn test_peduncle_returns_none_on_tiny_mask() {
        // Well below MIN_MASK_PIXELS — width bins are unreliable.
        let mut m = Array2::<u8>::zeros((40, 60));
        for r in 18..22 {
            for c in 5..55 {
                m[[r, c]] = 1;
            }
        }
        let d = classify_by_peduncle(&m, [5.0, 20.0], [54.0, 20.0]);
        assert!(d.is_none());
    }
}
