//! Phase-1 segmentation diagnostic binary.
//!
//! Usage:
//!   cargo run -p fishsense-core --example seg_debug -- <rgb.npy> [<rgb.npy> ...]
//!
//! For each RGB NPY file (shape (H, W, 3), uint8), runs `inference_debug` and
//! prints per-detection stats: score, bbox, pasted-mask area, polygon vertex
//! count, whether the detection was drawn into the output, and — if dropped —
//! why. Also prints connected-component stats on the final mask.

use std::collections::BTreeMap;
use std::env;
use std::path::Path;

use fishsense_core::fish::fish_segmentation::FishSegmentation;
use ndarray::{Array2, Array3};

fn load_rgb(path: &Path) -> Array3<u8> {
    // ndarray-npy doesn't expose a plain NpyReader for typed arrays in this
    // version, so we adapt via NpzReader: the .npy is just a single-array NPZ.
    // Simpler: use the raw NPY header parser below.
    let bytes = std::fs::read(path).expect("read npy");
    parse_npy_u8_3d(&bytes)
}

/// Minimal NPY v1.0 / v2.0 parser for a uint8 3-D C-contiguous array.
fn parse_npy_u8_3d(bytes: &[u8]) -> Array3<u8> {
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not an NPY file");
    let major = bytes[6];
    let header_len_size = if major == 1 { 2 } else { 4 };
    let header_len = match major {
        1 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        2 | 3 => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        _ => panic!("unsupported NPY major={major}"),
    };
    let hdr_start = 8 + header_len_size;
    let header = std::str::from_utf8(&bytes[hdr_start..hdr_start + header_len]).unwrap();
    let data_start = hdr_start + header_len;
    assert!(header.contains("'descr': '|u1'") || header.contains("'descr': '<u1'") || header.contains("'descr': 'u1'"),
        "expected uint8 dtype, got header: {header}");
    assert!(!header.contains("'fortran_order': True"), "expected C-order");
    // extract shape: "(1440, 1920, 3)"
    let shape_start = header.find("'shape': (").unwrap() + "'shape': (".len();
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    assert_eq!(dims.len(), 3, "expected 3-D image, got shape {:?}", dims);
    let (h, w, c) = (dims[0], dims[1], dims[2]);
    let n = h * w * c;
    assert_eq!(bytes.len() - data_start, n, "byte count != h*w*c");
    let data = bytes[data_start..data_start + n].to_vec();
    Array3::from_shape_vec((h, w, c), data).unwrap()
}

fn count_connected_components(mask: &Array2<u8>) -> BTreeMap<u8, (usize, u32)> {
    // For each distinct positive value: (n_connected_components, total_px).
    let (h, w) = mask.dim();
    let mut label = vec![0i32; h * w];
    let mut per_value: BTreeMap<u8, (usize, u32)> = BTreeMap::new();
    let mut next_label = 1i32;
    for y in 0..h {
        for x in 0..w {
            let v = mask[[y, x]];
            if v == 0 || label[y * w + x] != 0 {
                continue;
            }
            next_label += 1;
            // BFS
            let mut stack = vec![(y, x)];
            let mut size: u32 = 0;
            while let Some((cy, cx)) = stack.pop() {
                if label[cy * w + cx] != 0 {
                    continue;
                }
                if mask[[cy, cx]] != v {
                    continue;
                }
                label[cy * w + cx] = next_label;
                size += 1;
                if cy > 0 {
                    stack.push((cy - 1, cx));
                }
                if cy + 1 < h {
                    stack.push((cy + 1, cx));
                }
                if cx > 0 {
                    stack.push((cy, cx - 1));
                }
                if cx + 1 < w {
                    stack.push((cy, cx + 1));
                }
            }
            let entry = per_value.entry(v).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += size;
        }
    }
    per_value
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: seg_debug <rgb.npy> [<rgb.npy> ...]");
        std::process::exit(2);
    }

    let mut seg = FishSegmentation::new();
    seg.load_model().expect("load_model");

    for p in &args {
        let path = Path::new(p);
        println!("\n== {} ==", path.display());
        let rgb = load_rgb(path);
        let (h, w, _) = rgb.dim();
        println!("   shape: {h}x{w}");

        let (mask, debugs) = seg.inference_debug(&rgb).expect("inference_debug");
        println!(
            "   detections above score={:.2}: {}",
            0.3_f32,
            debugs.len()
        );
        for d in &debugs {
            let (x1, y1, x2, y2) = d.bbox_xyxy;
            println!(
                "     idx={} score={:.3} bbox=({:.0},{:.0},{:.0},{:.0}) bbox_area={} pasted_mask_px={} poly_verts={} drawn={} drop={:?}",
                d.index,
                d.score,
                x1,
                y1,
                x2,
                y2,
                ((x2 - x1).max(0.0) * (y2 - y1).max(0.0)) as u32,
                d.mask_area_px,
                d.polygon_vertices,
                d.drawn,
                d.drop_reason,
            );
        }

        let ccs = count_connected_components(&mask);
        let total_pos: u32 = ccs.values().map(|(_, px)| px).sum();
        println!("   final mask: total_pos_px={} instances={}", total_pos, ccs.len());
        for (v, (n_cc, px)) in &ccs {
            println!("     instance_id={} n_connected_components={} px={}", v, n_cc, px);
        }
    }

}
