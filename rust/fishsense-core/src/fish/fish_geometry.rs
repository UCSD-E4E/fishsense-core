//! Fish geometry helpers: perimeter extraction, polygon operations,
//! endpoint classification, and endpoint correction.
//!
//! This module is a Rust port of the Python `FishGeometry` class and the
//! classify/correct logic in `FishPointsOfInterestDetector`, replacing
//! Shapely operations with the `geo` crate.
//!
//! Coordinates are `[col, row]` (i.e. `[x, y]`) throughout.

use geo::{
    Area, BooleanOps, Closest, ClosestPoint, ConvexHull, Distance, Euclidean, LineString,
    MultiPolygon, Point, Polygon,
};
use opencv::core::{Mat, Point2i, Vector, CV_8UC1};
use opencv::imgproc::{find_contours_with_hierarchy, CHAIN_APPROX_NONE, RETR_EXTERNAL};
use std::ffi::c_void;

// ── private helpers ───────────────────────────────────────────────────────────

/// Distance from `pt` to the nearest point on `line`.
fn dist_pt_line(pt: Point<f64>, line: &LineString<f64>) -> f64 {
    match line.closest_point(&pt) {
        Closest::Intersection(q) | Closest::SinglePoint(q) => Euclidean::distance(pt, q),
        Closest::Indeterminate => f64::INFINITY,
    }
}
use ndarray::Array2;

use crate::errors::FishSenseError;

// ── public types ─────────────────────────────────────────────────────────────

/// Head and tail coordinates after classification and correction.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifiedEndpoints {
    pub head: [f64; 2],
    pub tail: [f64; 2],
    /// Confidence score in [0.5, 1.0]: how different the two halves were.
    pub confidence: f64,
}

// ── perimeter extraction ──────────────────────────────────────────────────

/// Extracts the ordered boundary of the largest external component of a
/// binary mask using `cv::findContours` (CHAIN_APPROX_NONE, RETR_EXTERNAL).
///
/// Returns pixels in `[col, row]` order, topologically ordered around the
/// contour so the resulting `LineString` forms a non-self-intersecting loop.
/// Returns an empty `Vec` when the mask contains no non-zero pixels.
///
/// The previous hand-rolled Moore trace fell back to scanline enumeration on
/// complex shapes (forked tails), which silently produced self-crossing
/// "polygons" whose geometric ops were nonsense — hence the regression to
/// `findContours`, which guarantees a valid ordering.
pub fn extract_perimeter(mask: &Array2<u8>) -> Vec<[f64; 2]> {
    if !mask.iter().any(|&v| v != 0) {
        return vec![];
    }

    let arr = mask.as_standard_layout();
    let (height, width) = arr.dim();
    let data_ptr = arr.as_ptr() as *mut c_void;

    // SAFETY: the Mat borrows `arr`'s storage; `arr` lives for this function
    // and is dropped after the Mat.
    let mat = match unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(height as i32, width as i32, CV_8UC1, data_ptr)
    } {
        Ok(m) => m,
        Err(_) => return vec![],
    };

    let mut contours: Vector<Vector<Point2i>> = Vector::new();
    let mut hierarchy: Vector<opencv::core::Vec4i> = Vector::new();
    if find_contours_with_hierarchy(
        &mat,
        &mut contours,
        &mut hierarchy,
        RETR_EXTERNAL,
        CHAIN_APPROX_NONE,
        Point2i::new(0, 0),
    )
    .is_err()
    {
        return vec![];
    }

    // Multiple disjoint components (e.g. segmentation noise) yield multiple
    // external contours; pick the longest, which corresponds to the fish.
    let Some(largest) = contours.iter().max_by_key(|c| c.len()) else {
        return vec![];
    };

    largest.iter().map(|p| [p.x as f64, p.y as f64]).collect()
}

// ── geometry helpers ──────────────────────────────────────────────────────

/// Fraction of a polygon half's area below which a concavity is treated as
/// mask noise. Used by both the head/tail classifier
/// ([`min_concavity_distance_to_point`]) and the tail corrector
/// ([`correct_tail`]).
const FORK_MIN_AREA_FRACTION: f64 = 0.01;

/// Builds a `geo::Polygon` from an ordered perimeter in `[col, row]` order.
pub fn polygon_from_perimeter(perimeter: &[[f64; 2]]) -> Polygon<f64> {
    let coords: Vec<(f64, f64)> = perimeter.iter().map(|p| (p[0], p[1])).collect();
    Polygon::new(LineString::from(coords), vec![])
}

/// Returns a line (as two endpoints) perpendicular to the segment `(a, b)`,
/// passing through the midpoint of `ab`, and extended by `scale` in each
/// direction along the perpendicular.
///
/// Output is `([col, row], [col, row])`.
pub fn perpendicular_bisector(a: [f64; 2], b: [f64; 2], scale: f64) -> ([f64; 2], [f64; 2]) {
    let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return ([mid[0], mid[1] - scale], [mid[0], mid[1] + scale]);
    }
    // Rotate 90°.
    let px = -dy / len;
    let py = dx / len;
    (
        [mid[0] + px * scale, mid[1] + py * scale],
        [mid[0] - px * scale, mid[1] - py * scale],
    )
}

/// Splits a polygon into two halves using a dividing line defined by two
/// points (`line_a`, `line_b`).  Returns the two largest resulting polygons.
pub fn split_polygon(
    poly: &Polygon<f64>,
    line_a: [f64; 2],
    line_b: [f64; 2],
) -> (Polygon<f64>, Polygon<f64>) {
    let far = 1e6_f64;
    let dx = line_b[0] - line_a[0];
    let dy = line_b[1] - line_a[1];
    // Normal pointing "left" of a→b.
    let nx = -dy;
    let ny = dx;

    let half_left = Polygon::new(
        LineString::from(vec![
            (line_a[0], line_a[1]),
            (line_b[0], line_b[1]),
            (line_b[0] + nx * far, line_b[1] + ny * far),
            (line_a[0] + nx * far, line_a[1] + ny * far),
            (line_a[0], line_a[1]),
        ]),
        vec![],
    );
    let half_right = Polygon::new(
        LineString::from(vec![
            (line_a[0], line_a[1]),
            (line_b[0], line_b[1]),
            (line_b[0] - nx * far, line_b[1] - ny * far),
            (line_a[0] - nx * far, line_a[1] - ny * far),
            (line_a[0], line_a[1]),
        ]),
        vec![],
    );

    let clip = |clipper: &Polygon<f64>| -> Polygon<f64> {
        let result: MultiPolygon<f64> = poly.intersection(clipper);
        result
            .0
            .into_iter()
            .max_by(|a, b| {
                a.unsigned_area()
                    .partial_cmp(&b.unsigned_area())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| Polygon::new(LineString::from(vec![(0.0, 0.0)]), vec![]))
    };

    (clip(&half_left), clip(&half_right))
}

/// Minimum distance from `point` to the boundary of any significant concavity
/// of `poly`.
///
/// "Concavity" here means a piece of `convex_hull(poly) - poly` — the regions
/// the hull bridges over. Concavities smaller than [`FORK_MIN_AREA_FRACTION`]
/// of `poly`'s area are treated as mask noise and excluded. Returns
/// `INFINITY` when no significant concavity exists (or the half is convex).
///
/// A caudal fork's tips sit on the hull, *at vertices of the fork-notch
/// concavity*, so for the fork-side half this distance is ≈ 0. A snout tip
/// sits on the hull but nowhere near any real concavity, so for the
/// snout-side half this distance is large (or infinite). This is the
/// tail-vs-head discriminator used by [`classify_from_perimeter`].
///
/// Previously this returned the distance to the *largest* concavity. On a
/// real fish the largest concavity in the fork-side hull-difference is the
/// broad bay around the caudal peduncle, not the fork notch — its apex can
/// sit ~100 px from the PCA fork endpoint. Meanwhile a small mask-noise
/// pocket near the snout can sit ~14 px from the PCA snout endpoint, so
/// "largest concavity" picked the wrong half as tail. Filtering by minimum
/// area and taking the minimum distance keeps the fork signal (any of the
/// fork side's several significant concavities is finite; a clean snout
/// half has none) while rejecting sub-threshold noise.
fn min_concavity_distance_to_point(poly: &Polygon<f64>, point: [f64; 2]) -> f64 {
    let hull: Polygon<f64> = poly.convex_hull();
    let diff: MultiPolygon<f64> = hull.difference(poly);
    let min_area = poly.unsigned_area() * FORK_MIN_AREA_FRACTION;
    let pt = Point::new(point[0], point[1]);
    diff.0
        .iter()
        .filter(|c| c.unsigned_area() >= min_area)
        .map(|c| dist_pt_line(pt, c.exterior()))
        .fold(f64::INFINITY, f64::min)
}

/// Nearest point on `poly`'s exterior boundary to `query`.
fn nearest_on_boundary(poly: &Polygon<f64>, query: Point<f64>) -> Point<f64> {
    match poly.exterior().closest_point(&query) {
        Closest::Intersection(p) | Closest::SinglePoint(p) => p,
        Closest::Indeterminate => query,
    }
}

// ── public pipeline functions ─────────────────────────────────────────────

/// Classifies which of `left`/`right` is the fish head vs. tail.
///
/// Extracts the perimeter from `mask`, builds the two polygon halves, and
/// assigns head/tail by comparing their convex-hull-difference areas
/// (larger difference → tail, more concave).
pub fn classify(
    mask: &Array2<u8>,
    left: [f64; 2],
    right: [f64; 2],
) -> Result<ClassifiedEndpoints, FishSenseError> {
    let perimeter = extract_perimeter(mask);
    if perimeter.len() < 3 {
        return Err(FishSenseError::AnyhowError(anyhow::anyhow!(
            "perimeter has fewer than 3 points — cannot classify endpoints"
        )));
    }
    classify_from_perimeter(&perimeter, left, right)
}

/// Core classification logic given an already-extracted perimeter.
///
/// Splits the polygon with the perpendicular bisector of `left`–`right`, then
/// asks of each half: how close does its PCA endpoint sit to its largest
/// concavity? A fork tip is a vertex of the fork-notch concavity, so the fork
/// side's distance is ≈ 0; a snout tip has no concavity nearby. The endpoint
/// with the smaller distance is the tail.
///
/// The previous heuristic compared the two halves' total convex-hull-minus-
/// polygon area. That fails on fish whose head half happens to contain more
/// concave area than the tail half — notably, fish with a prominent dorsal
/// fin — because the dorsal-fin concavity outweighs the caudal-fork notch.
/// Downstream `correct_tail` then pulls the "tail" onto the dorsal-fin notch.
pub fn classify_from_perimeter(
    perimeter: &[[f64; 2]],
    left: [f64; 2],
    right: [f64; 2],
) -> Result<ClassifiedEndpoints, FishSenseError> {
    let poly = polygon_from_perimeter(perimeter);

    let cols: Vec<f64> = perimeter.iter().map(|p| p[0]).collect();
    let rows: Vec<f64> = perimeter.iter().map(|p| p[1]).collect();
    let col_range = cols.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - cols.iter().cloned().fold(f64::INFINITY, f64::min);
    let row_range = rows.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - rows.iter().cloned().fold(f64::INFINITY, f64::min);
    let scale = col_range.max(row_range) * 2.0;

    let (perp_a, perp_b) = perpendicular_bisector(left, right, scale);
    let (half0, half1) = split_polygon(&poly, perp_a, perp_b);

    // Associate each PCA endpoint with the half whose exterior it lies on.
    // `left` and `right` are extremes along the same axis, so they land in
    // opposite halves; testing one endpoint is enough.
    let left_pt = Point::new(left[0], left[1]);
    let (left_half, right_half) = if dist_pt_line(left_pt, half0.exterior())
        <= dist_pt_line(left_pt, half1.exterior())
    {
        (&half0, &half1)
    } else {
        (&half1, &half0)
    };

    let d_left = min_concavity_distance_to_point(left_half, left);
    let d_right = min_concavity_distance_to_point(right_half, right);

    let (head, tail) = if d_left <= d_right {
        (right, left)
    } else {
        (left, right)
    };

    // Confidence: how one-sided the signal is. Saturates at 1.0 when one
    // endpoint is on a concavity and the other is far from any.
    let confidence = if d_left.is_finite() && d_right.is_finite() {
        let max_d = d_left.max(d_right);
        if max_d < 1e-12 {
            0.5
        } else {
            ((d_left - d_right).abs() / max_d / 2.0 + 0.5).clamp(0.5, 1.0)
        }
    } else if d_left.is_finite() || d_right.is_finite() {
        // One half is convex, the other has a concavity → strong signal.
        1.0
    } else {
        0.5
    };

    Ok(ClassifiedEndpoints {
        head,
        tail,
        confidence,
    })
}

/// Refines the head endpoint (port of `correct_head_coord`).
///
/// Splits `head_half` by a line perpendicular to the head→tail axis at the
/// head point, picks the sub-polygon facing away from the tail, and returns
/// the nearest point on its convex hull boundary to the extended head direction.
pub fn correct_head(
    head: [f64; 2],
    tail: [f64; 2],
    head_half: &Polygon<f64>,
    scale: f64,
) -> [f64; 2] {
    let dx = head[0] - tail[0];
    let dy = head[1] - tail[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return head;
    }
    let ux = dx / len;
    let uy = dy / len;

    let extended = Point::new(head[0] + ux * scale * 2.0, head[1] + uy * scale * 2.0);

    // Perpendicular line centred on the head point.
    let line_a = [head[0] - uy * scale, head[1] + ux * scale];
    let line_b = [head[0] + uy * scale, head[1] - ux * scale];

    let (sub0, sub1) = split_polygon(head_half, line_a, line_b);

    let d0: f64 = dist_pt_line(extended, sub0.exterior());
    let d1: f64 = dist_pt_line(extended, sub1.exterior());
    let chosen = if d0 <= d1 { &sub0 } else { &sub1 };

    let hull: Polygon<f64> = chosen.convex_hull();
    if hull.exterior().0.is_empty() {
        return head;
    }
    let nearest = nearest_on_boundary(&hull, extended);
    [nearest.x(), nearest.y()]
}

/// Refines the tail endpoint (port of `correct_tail_coord`).
///
/// Takes the convex-hull difference of `tail_half` (which yields every
/// concavity in the caudal-fin silhouette), drops concavities smaller than
/// [`FORK_MIN_AREA_FRACTION`] of the half's area, and snaps the tail to the
/// head-ward apex of the concavity whose apex lies nearest the raw tail —
/// the fork vertex for a forked caudal fin. When `tail_half` has no
/// significant concavities (rounded / pointed caudal fin) the tail is
/// returned unchanged.
///
/// Previously this function compared boundary-to-`extended` distances and
/// only applied the correction if the raw tail already lay inside a
/// concavity. For a forked caudal fin the PCA tail is the tip of one lobe
/// (on the convex hull, not inside any concavity) and a one-pixel
/// mask-noise pocket at that tip always beat the real fork notch on
/// boundary-distance — so the tail never moved to the fork.
pub fn correct_tail(
    head: [f64; 2],
    tail: [f64; 2],
    tail_half: &Polygon<f64>,
    scale: f64,
) -> [f64; 2] {
    let dx = tail[0] - head[0];
    let dy = tail[1] - head[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return tail;
    }
    let ux = dx / len;
    let uy = dy / len;

    let head_extended = Point::new(head[0] - ux * scale * 2.0, head[1] - uy * scale * 2.0);
    let tail_pt = Point::new(tail[0], tail[1]);

    let hull: Polygon<f64> = tail_half.convex_hull();
    let diff: MultiPolygon<f64> = hull.difference(tail_half);

    if diff.0.is_empty() {
        return tail;
    }

    let min_area = tail_half.unsigned_area() * FORK_MIN_AREA_FRACTION;

    let chosen = diff
        .0
        .iter()
        .filter(|p| p.unsigned_area() >= min_area)
        .min_by(|a, b| {
            let apex_a = nearest_on_boundary(a, head_extended);
            let apex_b = nearest_on_boundary(b, head_extended);
            let da = Euclidean::distance(tail_pt, apex_a);
            let db = Euclidean::distance(tail_pt, apex_b);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

    // A real fork notch is a small local feature of the caudal fin: its apex
    // sits within a few percent of the fish's length of the raw PCA tail (the
    // PCA tail itself is on the notch's boundary). If the best above-threshold
    // concavity's apex is far from the raw tail, no real fork notch passed
    // the area filter — don't snap onto an unrelated broad concavity.
    const MAX_APEX_SNAP_FRACTION: f64 = 0.15;
    let max_snap = len * MAX_APEX_SNAP_FRACTION;

    match chosen {
        Some(c) => {
            let apex = nearest_on_boundary(c, head_extended);
            if Euclidean::distance(tail_pt, apex) > max_snap {
                tail
            } else {
                [apex.x(), apex.y()]
            }
        }
        None => tail,
    }
}

/// Distance from the midpoint of `(a, b)` to the furthest perimeter point,
/// used to size perpendicular bisectors.
pub fn compute_scale(perimeter: &[[f64; 2]], a: [f64; 2], b: [f64; 2]) -> f64 {
    let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    perimeter
        .iter()
        .map(|p| {
            let dc = p[0] - mid[0];
            let dr = p[1] - mid[1];
            (dc * dc + dr * dr).sqrt()
        })
        .fold(0.0_f64, f64::max)
        .max(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn mask_from_pixels(height: usize, width: usize, pixels: &[(usize, usize)]) -> Array2<u8> {
        let mut m = Array2::<u8>::zeros((height, width));
        for &(r, c) in pixels {
            m[[r, c]] = 1;
        }
        m
    }

    fn rect_mask(height: usize, width: usize) -> Array2<u8> {
        Array2::<u8>::ones((height, width))
    }

    // ── perimeter extraction ──────────────────────────────────────────────

    #[test]
    fn test_perimeter_empty_mask_returns_empty() {
        let mask = Array2::<u8>::zeros((10, 10));
        assert!(extract_perimeter(&mask).is_empty());
    }

    #[test]
    fn test_perimeter_single_pixel() {
        let mask = mask_from_pixels(5, 5, &[(2, 2)]);
        let p = extract_perimeter(&mask);
        assert!(!p.is_empty());
        assert!(p.iter().any(|pt| pt[0] as usize == 2 && pt[1] as usize == 2));
    }

    #[test]
    fn test_perimeter_3x3_filled_square_excludes_interior() {
        let mut m = Array2::<u8>::zeros((5, 5));
        for r in 1..=3 {
            for c in 1..=3 {
                m[[r, c]] = 1;
            }
        }
        let p = extract_perimeter(&m);
        // Interior pixel (2,2) should NOT be in the perimeter.
        assert!(
            !p.iter()
                .any(|pt| pt[0] as usize == 2 && pt[1] as usize == 2),
            "interior pixel should not be in perimeter"
        );
        // At least the 8 border pixels should appear.
        assert!(p.len() >= 8, "expected ≥8 perimeter points, got {}", p.len());
    }

    /// Regression: a 4-non-zero-pixel mask (segmentation noise) used to send
    /// the Moore boundary trace into a sub-cycle that never returned to the
    /// seed, growing `boundary` without bound until the process was OOM-killed
    /// (observed on iOS, EXC_RESOURCE high watermark inside extract_perimeter).
    #[test]
    fn test_perimeter_four_pixel_noise_terminates_quickly() {
        let mask = mask_from_pixels(64, 64, &[(10, 10), (10, 11), (11, 10), (30, 40)]);
        let start = std::time::Instant::now();
        let p = extract_perimeter(&mask);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "extract_perimeter took {elapsed:?} on a 4-pixel mask"
        );
        assert!(!p.is_empty());
        assert!(p.len() <= 4);
    }

    #[test]
    fn test_perimeter_one_pixel_wide_line_terminates() {
        let pixels: Vec<(usize, usize)> = (5..25).map(|c| (10, c)).collect();
        let mask = mask_from_pixels(32, 32, &pixels);
        let start = std::time::Instant::now();
        let p = extract_perimeter(&mask);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "extract_perimeter took {elapsed:?} on a 1-pixel-wide line"
        );
        assert!(!p.is_empty());
    }

    #[test]
    fn test_perimeter_horizontal_bar_nonempty() {
        let pixels: Vec<(usize, usize)> = (0..5).map(|c| (0, c)).collect();
        let mask = mask_from_pixels(3, 7, &pixels);
        let p = extract_perimeter(&mask);
        assert!(!p.is_empty());
    }

    // ── perpendicular bisector ────────────────────────────────────────────

    #[test]
    fn test_perp_bisector_horizontal_line_is_vertical() {
        let (p1, p2) = perpendicular_bisector([0.0, 5.0], [10.0, 5.0], 10.0);
        // Both points share midpoint x=5.
        assert!((p1[0] - 5.0).abs() < 1e-9);
        assert!((p2[0] - 5.0).abs() < 1e-9);
        assert!(p1[1] != p2[1]);
    }

    #[test]
    fn test_perp_bisector_vertical_line_is_horizontal() {
        let (p1, p2) = perpendicular_bisector([5.0, 0.0], [5.0, 10.0], 10.0);
        assert!((p1[1] - 5.0).abs() < 1e-9);
        assert!((p2[1] - 5.0).abs() < 1e-9);
        assert!(p1[0] != p2[0]);
    }

    // ── classification ────────────────────────────────────────────────────

    #[test]
    fn test_classify_symmetric_rectangle_succeeds() {
        let mask = rect_mask(10, 30);
        let result = classify(&mask, [1.0, 5.0], [28.0, 5.0]);
        assert!(result.is_ok());
        let ep = result.unwrap();
        assert_ne!(ep.head, ep.tail);
        assert!(ep.confidence >= 0.5 && ep.confidence <= 1.0);
    }

    /// A mask with a concave notch on the right end: the right endpoint should
    /// be classified as the tail.
    #[test]
    fn test_classify_concave_right_end_is_tail() {
        let mut m = Array2::<u8>::zeros((20, 60));
        for r in 5..15 {
            for c in 2..58 {
                m[[r, c]] = 1;
            }
        }
        // Carve notch into right end.
        for depth in 0..5usize {
            m[[10, 57 - depth]] = 0;
            if 9usize.saturating_sub(depth) >= 1 {
                m[[9 - depth, 57]] = 0;
            }
            if 10 + depth < 19 {
                m[[10 + depth, 57]] = 0;
            }
        }

        let result = classify(&m, [2.0, 10.0], [57.0, 10.0]);
        assert!(result.is_ok());
        let ep = result.unwrap();
        assert!(
            ep.tail[0] > ep.head[0],
            "concave right end should be tail; head={:?} tail={:?}",
            ep.head,
            ep.tail
        );
    }

    // ── head correction ───────────────────────────────────────────────────

    #[test]
    fn test_correct_head_lands_near_boundary() {
        let m = rect_mask(10, 40);
        let perimeter = extract_perimeter(&m);
        let left = [1.0, 5.0];
        let right = [38.0, 5.0];
        let scale = compute_scale(&perimeter, left, right);
        let (perp_a, perp_b) = perpendicular_bisector(left, right, scale);
        let poly = polygon_from_perimeter(&perimeter);
        let (half0, half1) = split_polygon(&poly, perp_a, perp_b);

        let left_pt = Point::new(left[0], left[1]);
        let head_half =
            if dist_pt_line(left_pt, half0.exterior()) <= dist_pt_line(left_pt, half1.exterior())
            {
                half0
            } else {
                half1
            };

        let corrected = correct_head(left, right, &head_half, scale);
        let dist = dist_pt_line(Point::new(corrected[0], corrected[1]), head_half.exterior());
        assert!(dist < 2.0, "corrected head dist from boundary={dist}");
    }

    // ── tail correction ───────────────────────────────────────────────────

    #[test]
    fn test_correct_tail_convex_half_no_large_jump() {
        let m = rect_mask(10, 40);
        let perimeter = extract_perimeter(&m);
        let left = [1.0, 5.0];
        let right = [38.0, 5.0];
        let scale = compute_scale(&perimeter, left, right);
        let (perp_a, perp_b) = perpendicular_bisector(left, right, scale);
        let poly = polygon_from_perimeter(&perimeter);
        let (half0, half1) = split_polygon(&poly, perp_a, perp_b);

        let right_pt = Point::new(right[0], right[1]);
        let tail_half =
            if dist_pt_line(right_pt, half0.exterior()) <= dist_pt_line(right_pt, half1.exterior())
            {
                half0
            } else {
                half1
            };

        let corrected = correct_tail(left, right, &tail_half, scale);
        let moved = ((corrected[0] - right[0]).powi(2) + (corrected[1] - right[1]).powi(2)).sqrt();
        assert!(moved < scale, "tail moved {moved} which exceeds scale {scale}");
    }
}
