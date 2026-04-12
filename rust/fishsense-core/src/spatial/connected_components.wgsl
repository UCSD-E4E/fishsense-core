// GPU connected-components labeling via parallel union-find.
// Based on: https://doi.org/10.1145/3208040.3208041
//
// Three passes:
//   init   – every pixel points to itself (y_pred[i] = i)
//   merge  – each pixel merges with its right / down neighbour when
//            |depth[a] - depth[b]| <= epsilon  (4-connected)
//   flatten– every pixel is resolved to the root of its component

struct Parameters {
    width:   u32,
    height:  u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read>       parameters: Parameters;
@group(0) @binding(1) var<storage, read>       depth_map:  array<f32>;
@group(0) @binding(2) var<storage, read_write> y_pred:     array<atomic<u32>>;

fn pixel_idx(row: u32, col: u32) -> u32 {
    return row * parameters.width + col;
}

// Walk up the union-find tree to the root, halving the path as we go.
// In this forest every non-root satisfies y_pred[i] < i (we always
// point high-index nodes at low-index nodes), so the invariant
// atomicLoad(&y_pred[curr]) >= curr  iff  curr is a root.
fn find_root(idx: u32) -> u32 {
    var curr = atomicLoad(&y_pred[idx]);
    if (curr != idx) {
        var prev = idx;
        loop {
            let next = atomicLoad(&y_pred[curr]);
            if (next >= curr) { break; }      // curr is the root
            atomicStore(&y_pred[prev], next); // path halving: skip curr
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

// Merge the components of pixels a and b using compare-and-swap.
// Always links the higher-index root to the lower-index root so the
// invariant above is preserved.
fn union_trees(a: u32, b: u32) {
    var a_root = find_root(a);
    var b_root = find_root(b);

    loop {
        if (a_root == b_root) { break; }

        var ret: u32;
        if (a_root < b_root) {
            // Try to point b_root at a_root
            ret = atomicCompareExchangeWeak(&y_pred[b_root], b_root, a_root).old_value;
            if (ret != b_root) {
                b_root = ret; // lost the race — retry with updated root
            } else {
                break;
            }
        } else {
            // Try to point a_root at b_root
            ret = atomicCompareExchangeWeak(&y_pred[a_root], a_root, b_root).old_value;
            if (ret != a_root) {
                a_root = ret;
            } else {
                break;
            }
        }
    }
}

// Pass 1 — ECL-CC initialisation.
// Each pixel is pointed at the first neighbour in the adjacency list that
// has a smaller flat index AND is within epsilon.  For a 4-connected grid
// the only neighbours with smaller indices are "up" (idx − width) and
// "left" (idx − 1); we check "up" first because it has the smaller index.
// The pixel uses its own ID only when no such neighbour exists.
//
// This pre-links up/left edges in a single read-only pass so that the
// subsequent merge pass only needs to handle right/down edges, keeping
// every edge visited exactly once across the two passes.
@compute @workgroup_size(16, 16, 1)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= parameters.width || row >= parameters.height) { return; }

    let idx   = pixel_idx(row, col);
    let depth = depth_map[idx];

    if (row > 0u) {
        let up = pixel_idx(row - 1u, col);
        if (abs(depth - depth_map[up]) <= parameters.epsilon) {
            atomicStore(&y_pred[idx], up);
            return;
        }
    }
    if (col > 0u) {
        let left = pixel_idx(row, col - 1u);
        if (abs(depth - depth_map[left]) <= parameters.epsilon) {
            atomicStore(&y_pred[idx], left);
            return;
        }
    }
    atomicStore(&y_pred[idx], idx); // no smaller neighbour within epsilon
}

// Pass 2 — merge right/down neighbours within epsilon.
// Up/left edges were already handled by the init pass so each edge is
// still visited exactly once across both passes.
@compute @workgroup_size(16, 16, 1)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= parameters.width || row >= parameters.height) { return; }

    let idx   = pixel_idx(row, col);
    let depth = depth_map[idx];

    if (col + 1u < parameters.width) {
        let right = pixel_idx(row, col + 1u);
        if (abs(depth - depth_map[right]) <= parameters.epsilon) {
            union_trees(idx, right);
        }
    }

    if (row + 1u < parameters.height) {
        let down = pixel_idx(row + 1u, col);
        if (abs(depth - depth_map[down]) <= parameters.epsilon) {
            union_trees(idx, down);
        }
    }
}

// Pass 3 — flatten every path to its root.
// After this pass y_pred[idx] holds the component label for pixel idx;
// pixels in the same component share the same label (the root index).
@compute @workgroup_size(16, 16, 1)
fn flatten(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col < parameters.width && row < parameters.height) {
        let idx = pixel_idx(row, col);
        atomicStore(&y_pred[idx], find_root(idx));
    }
}
