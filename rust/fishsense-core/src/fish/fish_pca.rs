//! PCA-based fish endpoint estimation.
//!
//! Given a binary segmentation mask, finds the two endpoints of the fish's
//! principal axis by:
//!   1. Collecting all non-zero pixel coordinates.
//!   2. Zero-centering the point cloud.
//!   3. Computing the 2×2 covariance matrix.
//!   4. Extracting the dominant eigenvector via symmetric eigendecomposition.
//!   5. Projecting every mask pixel onto that eigenvector and returning the
//!      pixels with the minimum and maximum projections as `left` and `right`.
//!
//! Coordinates are `[col, row]` (i.e. `[x, y]`) throughout.

use nalgebra::{Matrix2, SymmetricEigen};
use ndarray::Array2;

use crate::errors::FishSenseError;

/// The two raw PCA endpoints in `[col, row]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct PcaEndpoints {
    /// The endpoint with the smaller projection onto the principal axis.
    pub left: [f64; 2],
    /// The endpoint with the larger projection onto the principal axis.
    pub right: [f64; 2],
}

/// Estimates the two principal endpoints of a fish from its binary mask.
///
/// Returns `Err` if the mask is empty or so degenerate that a covariance
/// matrix cannot be formed (fewer than 2 non-zero pixels).
pub fn estimate_endpoints(mask: &Array2<u8>) -> Result<PcaEndpoints, FishSenseError> {
    // Collect non-zero pixel positions as [col, row].
    let points: Vec<[f64; 2]> = mask
        .indexed_iter()
        .filter(|(_, v)| **v != 0)
        .map(|((row, col), _)| [col as f64, row as f64])
        .collect();

    if points.len() < 2 {
        return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
            "mask has fewer than 2 non-zero pixels — cannot estimate PCA endpoints"
        )));
    }

    let n = points.len() as f64;

    // Compute centroid.
    let mean_col = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let mean_row = points.iter().map(|p| p[1]).sum::<f64>() / n;

    // Zero-centred coordinates.
    let centred: Vec<[f64; 2]> = points
        .iter()
        .map(|p| [p[0] - mean_col, p[1] - mean_row])
        .collect();

    // 2×2 covariance matrix (biased estimator — sufficient for PCA direction).
    let (mut cxx, mut cxy, mut cyy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for c in &centred {
        cxx += c[0] * c[0];
        cxy += c[0] * c[1];
        cyy += c[1] * c[1];
    }
    cxx /= n;
    cxy /= n;
    cyy /= n;

    // Symmetric eigendecomposition — eigenvalues are sorted ascending.
    let cov = Matrix2::new(cxx, cxy, cxy, cyy);
    let eigen = SymmetricEigen::new(cov);

    // Dominant eigenvector = column with the largest eigenvalue.
    // nalgebra sorts eigenvalues ascending, so the last column is dominant.
    let evals = eigen.eigenvalues;
    let evecs = eigen.eigenvectors;
    let dominant_col = if evals[0] >= evals[1] { 0 } else { 1 };
    let vx = evecs[(0, dominant_col)];
    let vy = evecs[(1, dominant_col)];

    // Project every mask pixel onto the dominant eigenvector and find extremes.
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;
    let mut left = points[0];
    let mut right = points[0];

    for p in &points {
        let proj = (p[0] - mean_col) * vx + (p[1] - mean_row) * vy;
        if proj < min_proj {
            min_proj = proj;
            left = *p;
        }
        if proj > max_proj {
            max_proj = proj;
            right = *p;
        }
    }

    Ok(PcaEndpoints { left, right })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a mask from a list of (row, col) pixel positions.
    fn mask_from_pixels(height: usize, width: usize, pixels: &[(usize, usize)]) -> Array2<u8> {
        let mut m = Array2::<u8>::zeros((height, width));
        for &(r, c) in pixels {
            m[[r, c]] = 1;
        }
        m
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_mask_returns_err() {
        let mask = Array2::<u8>::zeros((10, 10));
        assert!(estimate_endpoints(&mask).is_err());
    }

    #[test]
    fn test_single_pixel_returns_err() {
        let mask = mask_from_pixels(10, 10, &[(5, 5)]);
        assert!(estimate_endpoints(&mask).is_err());
    }

    // ── axis-aligned cases ────────────────────────────────────────────────

    /// A horizontal bar: pixels at row=5, cols 1..=8 in a 10×10 grid.
    /// Principal axis is horizontal, so left≈col 1 and right≈col 8.
    #[test]
    fn test_horizontal_bar_endpoints() {
        let pixels: Vec<(usize, usize)> = (1..=8).map(|c| (5, c)).collect();
        let mask = mask_from_pixels(10, 10, &pixels);
        let ep = estimate_endpoints(&mask).unwrap();

        // left should be the leftmost (smallest col), right the rightmost.
        assert!(
            (ep.left[0] - 1.0).abs() < 1.5,
            "left col should be near 1, got {}",
            ep.left[0]
        );
        assert!(
            (ep.right[0] - 8.0).abs() < 1.5,
            "right col should be near 8, got {}",
            ep.right[0]
        );
        // Both should be on row 5.
        assert_eq!(ep.left[1] as usize, 5);
        assert_eq!(ep.right[1] as usize, 5);
    }

    /// A vertical bar: pixels at col=5, rows 1..=8.
    /// Principal axis is vertical, so the extreme projections are at row 1
    /// and row 8; which one is "left" vs "right" doesn't matter — what matters
    /// is that the two row values are near 1 and 8.
    #[test]
    fn test_vertical_bar_endpoints() {
        let pixels: Vec<(usize, usize)> = (1..=8).map(|r| (r, 5)).collect();
        let mask = mask_from_pixels(10, 10, &pixels);
        let ep = estimate_endpoints(&mask).unwrap();

        let rows = [ep.left[1] as i32, ep.right[1] as i32];
        assert!(
            rows.contains(&1) || rows.iter().any(|&r| r <= 2),
            "one endpoint should be near row 1"
        );
        assert!(
            rows.contains(&8) || rows.iter().any(|&r| r >= 7),
            "one endpoint should be near row 8"
        );
    }

    /// A diagonal bar (y = x): pixels at (i, i) for i in 1..=8.
    /// The extreme projections should be at (1,1) and (8,8).
    #[test]
    fn test_diagonal_bar_endpoints() {
        let pixels: Vec<(usize, usize)> = (1..=8).map(|i| (i, i)).collect();
        let mask = mask_from_pixels(12, 12, &pixels);
        let ep = estimate_endpoints(&mask).unwrap();

        // Both endpoints should lie on the diagonal (col ≈ row).
        let left_on_diag = (ep.left[0] - ep.left[1]).abs() < 1.5;
        let right_on_diag = (ep.right[0] - ep.right[1]).abs() < 1.5;
        assert!(left_on_diag, "left should be on the diagonal");
        assert!(right_on_diag, "right should be on the diagonal");

        // One end near (1,1), other near (8,8).
        let cols = [ep.left[0] as i32, ep.right[0] as i32];
        assert!(
            cols.iter().any(|&c| c <= 2),
            "one endpoint should be near col 1"
        );
        assert!(
            cols.iter().any(|&c| c >= 7),
            "one endpoint should be near col 8"
        );
    }

    /// A two-pixel mask — minimal valid case.
    #[test]
    fn test_two_pixel_mask() {
        let mask = mask_from_pixels(10, 10, &[(2, 1), (2, 8)]);
        let ep = estimate_endpoints(&mask).unwrap();
        let cols: Vec<i32> = vec![ep.left[0] as i32, ep.right[0] as i32];
        assert!(cols.contains(&1) && cols.contains(&8));
    }
}
