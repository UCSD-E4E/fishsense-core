//! Fish geometry helpers: perimeter extraction, polygon operations,
//! endpoint classification, and endpoint correction.
//!
//! This module is a Rust port of the Python `FishGeometry` class and the
//! classify/correct logic in `FishPointsOfInterestDetector`, replacing
//! Shapely operations with the `geo` crate.
//!
//! Coordinates are `[col, row]` (i.e. `[x, y]`) throughout.

use geo::{
    Area, BooleanOps, Closest, ClosestPoint, ConvexHull, Distance, Euclidean, Intersects,
    LineString, MultiPolygon, Point, Polygon,
};

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

/// Extracts an ordered list of boundary pixels from a binary mask using a
/// Moore neighbourhood boundary trace.
///
/// Returns pixels in `[col, row]` order. Returns an empty `Vec` when the mask
/// contains no non-zero pixels.
pub fn extract_perimeter(mask: &Array2<u8>) -> Vec<[f64; 2]> {
    let (height, width) = mask.dim();

    // Predicate: pixel (row, col) is on the 4-connected boundary.
    let in_bounds = |r: i32, c: i32| r >= 0 && r < height as i32 && c >= 0 && c < width as i32;
    let is_boundary = |row: usize, col: usize| -> bool {
        let r = row as i32;
        let c = col as i32;
        [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            .iter()
            .any(|&(nr, nc)| !in_bounds(nr, nc) || mask[[nr as usize, nc as usize]] == 0)
    };

    // Trivial collector used for empty, tiny, or pathological masks where the
    // Moore boundary trace either has nothing to do or cannot be trusted to
    // form a closed loop.
    let collect_all_boundary = || -> Vec<[f64; 2]> {
        mask.indexed_iter()
            .filter(|(_, v)| **v != 0)
            .filter(|((r, c), _)| is_boundary(*r, *c))
            .map(|((r, c), _)| [c as f64, r as f64])
            .collect()
    };

    // Count non-zero pixels up-front. This is the only quantity that can bound
    // the boundary-trace loop, and it lets us shortcut degenerate inputs
    // (noise from the segmentation model) before doing any tracing at all.
    let nnz = mask.iter().filter(|&&v| v != 0).count();
    if nnz == 0 {
        return vec![];
    }
    // For very small masks the Moore trace can enter a sub-cycle that never
    // returns to the seed; just enumerate boundary pixels directly.
    const TRACE_MIN_NNZ: usize = 8;
    if nnz < TRACE_MIN_NNZ {
        return collect_all_boundary();
    }

    // Find the first non-zero pixel as the seed.
    let (seed_row, seed_col) = mask
        .indexed_iter()
        .find(|(_, v)| **v != 0)
        .map(|((r, c), _)| (r, c))
        .expect("nnz > 0 guarantees a non-zero pixel exists");

    // 8-connected Moore neighbourhood, clockwise.
    const DIRS: [(i32, i32); 8] = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];

    let mut boundary: Vec<[f64; 2]> = Vec::new();
    let mut visited: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    let mut cur = (seed_row, seed_col);
    let mut dir_idx = 0usize;

    // Hard upper bound on iterations: each pixel can legitimately be visited
    // at most a handful of times during a Moore trace, so 8·nnz is generous
    // while still guaranteeing termination on pathological inputs.
    let max_iter = nnz.saturating_mul(8).saturating_add(16);
    let mut iter_count = 0usize;
    let mut aborted = false;

    loop {
        iter_count += 1;
        if iter_count > max_iter {
            aborted = true;
            break;
        }
        if !visited.insert(cur) && cur == (seed_row, seed_col) && !boundary.is_empty() {
            break;
        }
        if is_boundary(cur.0, cur.1) {
            boundary.push([cur.1 as f64, cur.0 as f64]);
        }

        let mut found = false;
        for i in 0..8 {
            let try_dir = (dir_idx + i) % 8;
            let (dr, dc) = DIRS[try_dir];
            let nr = cur.0 as i32 + dr;
            let nc = cur.1 as i32 + dc;
            if in_bounds(nr, nc) && mask[[nr as usize, nc as usize]] != 0 {
                cur = (nr as usize, nc as usize);
                dir_idx = try_dir;
                found = true;
                break;
            }
        }
        if !found {
            break;
        }
        if cur == (seed_row, seed_col) && boundary.len() > 1 {
            break;
        }
    }

    // Fallback for thin bars or aborted traces: collect all boundary pixels
    // directly. Discard whatever partial trace the loop accumulated.
    if aborted || boundary.len() < 3 {
        boundary = collect_all_boundary();
    }

    boundary
}

// ── geometry helpers ──────────────────────────────────────────────────────

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

/// Area of `convex_hull(poly) - poly`.
fn convex_difference_area(poly: &Polygon<f64>) -> f64 {
    let hull: Polygon<f64> = poly.convex_hull();
    let diff: MultiPolygon<f64> = hull.difference(poly);
    diff.0.iter().map(|p| p.unsigned_area()).sum()
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

    let area0 = convex_difference_area(&half0);
    let area1 = convex_difference_area(&half1);

    let (tail_half, head_half, tail_area, head_area) = if area0 >= area1 {
        (&half0, &half1, area0, area1)
    } else {
        (&half1, &half0, area1, area0)
    };

    let left_pt = Point::new(left[0], left[1]);

    let left_to_tail: f64 = dist_pt_line(left_pt, tail_half.exterior());
    let left_to_head: f64 = dist_pt_line(left_pt, head_half.exterior());

    let (head, tail) = if left_to_head <= left_to_tail {
        (left, right)
    } else {
        (right, left)
    };

    let max_area = tail_area.max(head_area);
    let confidence = if max_area < 1e-12 {
        0.5
    } else {
        ((tail_area - head_area) / max_area / 2.0 + 0.5).min(1.0)
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
/// Computes the convex-hull difference of `tail_half`, finds the piece closest
/// to the extended tail direction, and snaps to its boundary if the current
/// tail point lies inside it; otherwise leaves the tail unchanged.
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

    let extended = Point::new(tail[0] + ux * scale * 2.0, tail[1] + uy * scale * 2.0);
    let head_extended = Point::new(head[0] - ux * scale * 2.0, head[1] - uy * scale * 2.0);

    let hull: Polygon<f64> = tail_half.convex_hull();
    let diff: MultiPolygon<f64> = hull.difference(tail_half);

    if diff.0.is_empty() {
        return tail;
    }

    let closest_diff = diff
        .0
        .iter()
        .min_by(|a, b| {
            let da: f64 = dist_pt_line(extended, a.exterior());
            let db: f64 = dist_pt_line(extended, b.exterior());
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    let tail_pt = Point::new(tail[0], tail[1]);
    let tail_in_poly =
        closest_diff.intersects(&tail_pt) || closest_diff.exterior().intersects(&tail_pt);

    if tail_in_poly {
        let nearest = nearest_on_boundary(closest_diff, head_extended);
        [nearest.x(), nearest.y()]
    } else {
        tail
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
