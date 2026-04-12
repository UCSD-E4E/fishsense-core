use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use tracing::{debug, info, instrument, warn};
use wgpu::{util::DeviceExt, CommandEncoder, Device, ShaderModule};

use crate::{errors::FishSenseError, gpu::get_device_and_queue};

use super::types::DepthMap;

type Groups = Array2<u32>;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct ConnectedComponentsParameters {
    width:   u32,
    height:  u32,
    epsilon: f32,
}

/// Dispatches one named compute pass using a pre-built bind group and pipeline layout.
/// All three passes share the same layout so `init`/`flatten` (which don't read
/// `depth_map`) still accept the same 3-buffer bind group as `merge`.
#[allow(clippy::too_many_arguments)]
fn dispatch_pass(
    entry_point: &str,
    bind_group: &wgpu::BindGroup,
    pipeline_layout: &wgpu::PipelineLayout,
    width: u32,
    height: u32,
    shader_module: &ShaderModule,
    encoder: &mut CommandEncoder,
    device: &Device,
) {
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(pipeline_layout),
        module: shader_module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(entry_point),
        timestamp_writes: None,
    });
    cpass.set_pipeline(&pipeline);
    cpass.set_bind_group(0, bind_group, &[]);
    cpass.insert_debug_marker(entry_point);
    cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
}

/// GPU implementation of connected components.  Prefer [`connected_components`],
/// which falls back to [`connected_components_cpu`] automatically when no GPU
/// is available.
#[instrument(skip(depth_map), fields(width = tracing::field::Empty, height = tracing::field::Empty))]
pub async fn connected_components_gpu(
    depth_map: &DepthMap,
    epsilon: f32,
) -> Result<Groups, FishSenseError> {
    let (height, width) = depth_map.0.dim();
    tracing::Span::current()
        .record("width", width)
        .record("height", height);
    debug!(width, height, epsilon, "running GPU connected components");
    let pixel_count = (width * height) as u64;
    let y_pred_size = pixel_count * size_of::<u32>() as u64;

    let parameters = ConnectedComponentsParameters {
        width: width as u32,
        height: height as u32,
        epsilon,
    };

    let (device, queue) = get_device_and_queue().await?;
    let shader_module = device.create_shader_module(wgpu::include_wgsl!("connected_components.wgsl"));

    let parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parameters_buffer"),
        contents: bytemuck::bytes_of(&parameters),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let depth_map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("depth_map_buffer"),
        contents: bytemuck::cast_slice(
            depth_map.0.as_slice().context("depth map must be contiguous")?,
        ),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let y_pred_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y_pred_buffer"),
        size: y_pred_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let y_pred_cpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y_pred_cpu_buffer"),
        size: y_pred_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Build a shared bind group layout so that all three passes (init, merge,
    // flatten) accept the same 3-buffer group — even though init and flatten
    // don't read depth_map.  Using auto-derived layouts per entry point would
    // produce a 2-binding layout for init/flatten and a 3-binding one for
    // merge, causing a bind group mismatch at runtime.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cc_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cc_pipeline_layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cc_bind_group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: parameters_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: depth_map_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_pred_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    dispatch_pass("init",    &bind_group, &pipeline_layout, width as u32, height as u32, &shader_module, &mut encoder, &device);
    dispatch_pass("merge",   &bind_group, &pipeline_layout, width as u32, height as u32, &shader_module, &mut encoder, &device);
    dispatch_pass("flatten", &bind_group, &pipeline_layout, width as u32, height as u32, &shader_module, &mut encoder, &device);

    encoder.copy_buffer_to_buffer(&y_pred_buffer, 0, &y_pred_cpu_buffer, 0, y_pred_size);
    queue.submit(Some(encoder.finish()));

    // Bridge the map_async callback to the async caller via a sync channel.
    // device.poll(wait) guarantees the callback fires before poll returns.
    let y_pred_slice = y_pred_cpu_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    y_pred_slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|_| FishSenseError::UnknownError)?;
    rx.recv()
        .map_err(|_| FishSenseError::UnknownError)?
        .map_err(anyhow::Error::from)?;

    let y_pred_bytes = y_pred_slice.get_mapped_range();
    let y_pred_vec: Vec<u32> = bytemuck::cast_slice(&y_pred_bytes).to_vec();
    drop(y_pred_bytes);

    Array2::from_shape_vec((height, width), y_pred_vec)
        .map_err(|e| anyhow::anyhow!(e).into())
}

/// Walk the parent array to the root of `idx`, applying full path compression.
fn find_root_cpu(parent: &mut [u32], mut idx: u32) -> u32 {
    // Find the root (self-loop).
    let mut root = idx;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    // Compress: point every node on the path directly at the root.
    while parent[idx as usize] != root {
        let next = parent[idx as usize];
        parent[idx as usize] = root;
        idx = next;
    }
    root
}

/// Merge the components of pixels `a` and `b`, always linking the
/// higher-index root to the lower-index root to preserve the invariant
/// used by `find_root_cpu`.
fn union_cpu(parent: &mut [u32], a: u32, b: u32) {
    let a_root = find_root_cpu(parent, a);
    let b_root = find_root_cpu(parent, b);
    if a_root == b_root {
        return;
    }
    if a_root < b_root {
        parent[b_root as usize] = a_root;
    } else {
        parent[a_root as usize] = b_root;
    }
}

/// Labels every pixel in `depth_map` with its connected-component group.
///
/// Tries the GPU path first; if no hardware-backed adapter is available
/// (`CannotAcquireGpu`) it transparently falls back to [`connected_components_cpu`].
/// Which path was taken is logged at `info` level.
///
/// Returns an `Array2<u32>` of the same shape where pixels whose depth values
/// differ by at most `epsilon` and are 4-connected share the same non-zero label.
/// Labels are the flat index of each component's root pixel (scan order).
pub async fn connected_components(
    depth_map: &DepthMap,
    epsilon: f32,
) -> Result<Groups, FishSenseError> {
    match connected_components_gpu(depth_map, epsilon).await {
        Ok(result) => {
            info!("connected components: using GPU path");
            Ok(result)
        }
        Err(FishSenseError::CannotAcquireGpu) => {
            warn!("no GPU available, falling back to CPU for connected components");
            connected_components_cpu(depth_map, epsilon)
        }
        Err(e) => Err(e),
    }
}

/// CPU fallback for [`connected_components`].
///
/// Mirrors the three-pass ECL-CC algorithm from the GPU shader using a
/// sequential union-find with full path compression.  Use this when no
/// hardware-backed wGPU adapter is available (`DeviceType::Cpu` or
/// `CannotAcquireGpu`).
#[instrument(skip(depth_map), fields(width = tracing::field::Empty, height = tracing::field::Empty))]
pub fn connected_components_cpu(
    depth_map: &DepthMap,
    epsilon: f32,
) -> Result<Groups, FishSenseError> {
    let (height, width) = depth_map.0.dim();
    tracing::Span::current()
        .record("width", width)
        .record("height", height);
    debug!(width, height, epsilon, "running CPU connected components");
    let n = width * height;
    let depths = depth_map
        .0
        .as_slice()
        .context("depth map must be contiguous")?;

    let mut parent: Vec<u32> = (0..n as u32).collect();

    // Pass 1 — init: point each pixel at the first smaller-index 4-connected
    // neighbour within epsilon (up before left), mirroring the WGSL init pass.
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let depth = depths[idx];

            if row > 0 {
                let up = (row - 1) * width + col;
                if (depth - depths[up]).abs() <= epsilon {
                    parent[idx] = up as u32;
                    continue;
                }
            }
            if col > 0 {
                let left = row * width + (col - 1);
                if (depth - depths[left]).abs() <= epsilon {
                    parent[idx] = left as u32;
                }
            }
            // Otherwise parent[idx] stays as idx (self).
        }
    }

    // Pass 2 — merge: union right/down neighbours within epsilon.
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let depth = depths[idx];

            if col + 1 < width {
                let right = row * width + col + 1;
                if (depth - depths[right]).abs() <= epsilon {
                    union_cpu(&mut parent, idx as u32, right as u32);
                }
            }
            if row + 1 < height {
                let down = (row + 1) * width + col;
                if (depth - depths[down]).abs() <= epsilon {
                    union_cpu(&mut parent, idx as u32, down as u32);
                }
            }
        }
    }

    // Pass 3 — flatten: resolve every pixel to its component root.
    for i in 0..n {
        let root = find_root_cpu(&mut parent, i as u32);
        parent[i] = root;
    }

    Array2::from_shape_vec((height, width), parent).map_err(|e| anyhow::anyhow!(e).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const EPSILON: f32 = 0.005;

    /// Output label map has the same shape as the input depth map.
    #[tokio::test]
    async fn test_output_shape_matches_input() {
        let depth_map = DepthMap(Array2::<f32>::zeros((6, 8)));
        let labels = connected_components(&depth_map, EPSILON).await.unwrap();
        assert_eq!(labels.shape(), depth_map.0.shape());
    }

    /// A uniform depth map assigns the same label to every pixel.
    #[tokio::test]
    async fn test_uniform_map_single_label() {
        let depth_map = DepthMap(Array2::<f32>::from_elem((4, 4), 0.5));
        let labels = connected_components(&depth_map, EPSILON).await.unwrap();
        let first = labels[[0, 0]];
        assert!(labels.iter().all(|&l| l == first), "all pixels should share one label");
    }

    /// A single isolated pixel whose neighbours all differ by more than epsilon
    /// gets a unique label not shared by any of its 4-connected neighbours.
    #[tokio::test]
    async fn test_isolated_pixel_unique_label() {
        let mut data = Array2::<f32>::zeros((5, 5));
        data[[2, 2]] = 1.0; // isolated — neighbours are 0.0, diff = 1.0 > epsilon
        let depth_map = DepthMap(data);

        let labels = connected_components(&depth_map, EPSILON).await.unwrap();

        let isolated = labels[[2, 2]];
        assert_ne!(labels[[1, 2]], isolated, "above should differ");
        assert_ne!(labels[[3, 2]], isolated, "below should differ");
        assert_ne!(labels[[2, 1]], isolated, "left should differ");
        assert_ne!(labels[[2, 3]], isolated, "right should differ");
    }

    /// A 3×3 centre block at depth 0.5 surrounded by zeros — all 9 block pixels
    /// share one label and border pixels share a different label.
    #[tokio::test]
    async fn test_centre_block_uniform_label() {
        let mut data = Array2::<f32>::zeros((7, 7));
        for row in 2..=4usize {
            for col in 2..=4usize {
                data[[row, col]] = 0.5;
            }
        }
        let depth_map = DepthMap(data);

        let labels = connected_components(&depth_map, EPSILON).await.unwrap();

        let block_label = labels[[3, 3]];
        for row in 2..=4usize {
            for col in 2..=4usize {
                assert_eq!(labels[[row, col]], block_label, "[{row},{col}] should be in the block component");
            }
        }
        assert_ne!(labels[[0, 0]], block_label, "corner should not share the block label");
    }

    /// Two same-depth islands separated by a depth gap receive distinct labels.
    #[tokio::test]
    async fn test_two_islands_distinct_labels() {
        // Left island: cols 0–1, rows 0–4 at depth 0.5
        // Right island: cols 3–4, rows 0–4 at depth 0.5
        // Gap: col 2 at depth 0.0
        let mut data = Array2::<f32>::zeros((5, 5));
        for row in 0..5usize {
            data[[row, 0]] = 0.5;
            data[[row, 1]] = 0.5;
            data[[row, 3]] = 0.5;
            data[[row, 4]] = 0.5;
        }
        let depth_map = DepthMap(data);

        let labels = connected_components(&depth_map, EPSILON).await.unwrap();

        let left_label  = labels[[0, 0]];
        let right_label = labels[[0, 3]];
        let gap_label   = labels[[0, 2]];

        assert_ne!(left_label, right_label, "islands should have different labels");
        assert_ne!(left_label, gap_label,   "left island and gap should differ");
        assert_ne!(right_label, gap_label,  "right island and gap should differ");

        for row in 0..5usize {
            assert_eq!(labels[[row, 0]], left_label,  "[{row},0] should be in left island");
            assert_eq!(labels[[row, 1]], left_label,  "[{row},1] should be in left island");
            assert_eq!(labels[[row, 3]], right_label, "[{row},3] should be in right island");
            assert_eq!(labels[[row, 4]], right_label, "[{row},4] should be in right island");
        }
    }

    /// Adjacent pixels with depth within epsilon of each other share a label.
    #[tokio::test]
    async fn test_adjacent_within_epsilon_same_label() {
        let mut data = Array2::<f32>::zeros((3, 3));
        data[[1, 1]] = 0.003; // within EPSILON of 0.0 — should merge with neighbours
        let depth_map = DepthMap(data);

        let labels = connected_components(&depth_map, EPSILON).await.unwrap();

        let center = labels[[1, 1]];
        assert_eq!(labels[[0, 1]], center, "above should share label");
        assert_eq!(labels[[2, 1]], center, "below should share label");
        assert_eq!(labels[[1, 0]], center, "left should share label");
        assert_eq!(labels[[1, 2]], center, "right should share label");
    }

    // ── CPU path tests ───────────────────────────────────────────────────────

    /// Output label map has the same shape as the input depth map.
    #[test]
    fn cpu_output_shape_matches_input() {
        let depth_map = DepthMap(Array2::<f32>::zeros((6, 8)));
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();
        assert_eq!(labels.shape(), depth_map.0.shape());
    }

    /// A 1×1 map labels its single pixel.
    #[test]
    fn cpu_single_pixel() {
        let depth_map = DepthMap(Array2::<f32>::from_elem((1, 1), 0.5));
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();
        assert_eq!(labels[[0, 0]], 0);
    }

    /// A uniform depth map assigns the same label to every pixel.
    #[test]
    fn cpu_uniform_map_single_label() {
        let depth_map = DepthMap(Array2::<f32>::from_elem((4, 4), 0.5));
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();
        let first = labels[[0, 0]];
        assert!(labels.iter().all(|&l| l == first), "all pixels should share one label");
    }

    /// A single isolated pixel whose neighbours all differ by more than epsilon
    /// gets a unique label not shared by any of its 4-connected neighbours.
    #[test]
    fn cpu_isolated_pixel_unique_label() {
        let mut data = Array2::<f32>::zeros((5, 5));
        data[[2, 2]] = 1.0;
        let depth_map = DepthMap(data);
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        let isolated = labels[[2, 2]];
        assert_ne!(labels[[1, 2]], isolated, "above should differ");
        assert_ne!(labels[[3, 2]], isolated, "below should differ");
        assert_ne!(labels[[2, 1]], isolated, "left should differ");
        assert_ne!(labels[[2, 3]], isolated, "right should differ");
    }

    /// A 3×3 centre block at depth 0.5 surrounded by zeros — all 9 block pixels
    /// share one label and border pixels share a different label.
    #[test]
    fn cpu_centre_block_uniform_label() {
        let mut data = Array2::<f32>::zeros((7, 7));
        for row in 2..=4usize {
            for col in 2..=4usize {
                data[[row, col]] = 0.5;
            }
        }
        let depth_map = DepthMap(data);
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        let block_label = labels[[3, 3]];
        for row in 2..=4usize {
            for col in 2..=4usize {
                assert_eq!(labels[[row, col]], block_label, "[{row},{col}] should be in the block component");
            }
        }
        assert_ne!(labels[[0, 0]], block_label, "corner should not share the block label");
    }

    /// Two same-depth islands separated by a depth gap receive distinct labels.
    #[test]
    fn cpu_two_islands_distinct_labels() {
        let mut data = Array2::<f32>::zeros((5, 5));
        for row in 0..5usize {
            data[[row, 0]] = 0.5;
            data[[row, 1]] = 0.5;
            data[[row, 3]] = 0.5;
            data[[row, 4]] = 0.5;
        }
        let depth_map = DepthMap(data);
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        let left_label  = labels[[0, 0]];
        let right_label = labels[[0, 3]];
        let gap_label   = labels[[0, 2]];

        assert_ne!(left_label, right_label, "islands should have different labels");
        assert_ne!(left_label, gap_label,   "left island and gap should differ");
        assert_ne!(right_label, gap_label,  "right island and gap should differ");

        for row in 0..5usize {
            assert_eq!(labels[[row, 0]], left_label,  "[{row},0] should be in left island");
            assert_eq!(labels[[row, 1]], left_label,  "[{row},1] should be in left island");
            assert_eq!(labels[[row, 3]], right_label, "[{row},3] should be in right island");
            assert_eq!(labels[[row, 4]], right_label, "[{row},4] should be in right island");
        }
    }

    /// Adjacent pixels with depth within epsilon of each other share a label.
    #[test]
    fn cpu_adjacent_within_epsilon_same_label() {
        let mut data = Array2::<f32>::zeros((3, 3));
        data[[1, 1]] = 0.003;
        let depth_map = DepthMap(data);
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        let center = labels[[1, 1]];
        assert_eq!(labels[[0, 1]], center, "above should share label");
        assert_eq!(labels[[2, 1]], center, "below should share label");
        assert_eq!(labels[[1, 0]], center, "left should share label");
        assert_eq!(labels[[1, 2]], center, "right should share label");
    }

    /// A pixel exactly at the epsilon boundary merges with its neighbour;
    /// one just beyond does not.
    #[test]
    fn cpu_epsilon_boundary() {
        // Row of three pixels: depths [0.0, EPSILON, 2*EPSILON + tiny]
        // diff([0],[1]) = EPSILON     → exactly at boundary, should merge.
        // diff([1],[2]) = EPSILON + tiny → just over boundary, should not merge.
        let mut data = Array2::<f32>::zeros((1, 3));
        data[[0, 1]] = EPSILON;
        data[[0, 2]] = 2.0 * EPSILON + 0.001;
        let depth_map = DepthMap(data);
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        assert_eq!(labels[[0, 0]], labels[[0, 1]], "boundary pixels should merge");
        assert_ne!(labels[[0, 1]], labels[[0, 2]], "over-boundary pixels should not merge");
    }

    /// Every pixel at a unique depth produces N distinct labels.
    #[test]
    fn cpu_all_different_depths_distinct_labels() {
        let data = Array2::from_shape_fn((4, 4), |(r, c)| (r * 4 + c) as f32);
        let depth_map = DepthMap(data);
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        let mut seen = std::collections::HashSet::new();
        for &l in labels.iter() {
            seen.insert(l);
        }
        assert_eq!(seen.len(), 16, "each pixel should have a unique label");
    }

    /// A single-row strip is handled without off-by-one errors.
    #[test]
    fn cpu_single_row_strip() {
        // 5 pixels in a row all at depth 0.5 — one component.
        let depth_map = DepthMap(Array2::<f32>::from_elem((1, 5), 0.5));
        let labels = connected_components_cpu(&depth_map, EPSILON).unwrap();
        let first = labels[[0, 0]];
        assert!(labels.iter().all(|&l| l == first), "all strip pixels should share one label");
    }

    /// CPU and GPU produce identical label maps on a non-trivial input.
    #[tokio::test]
    async fn cpu_and_gpu_agree() {
        let mut data = Array2::<f32>::zeros((8, 8));
        // Two blobs and a gap to exercise the merge path in both implementations.
        for row in 0..4usize { for col in 0..3usize { data[[row, col]] = 0.5; } }
        for row in 0..4usize { for col in 5..8usize { data[[row, col]] = 0.5; } }
        let depth_map = DepthMap(data);

        let gpu_labels = connected_components_gpu(&depth_map, EPSILON).await.unwrap();
        let cpu_labels = connected_components_cpu(&depth_map, EPSILON).unwrap();

        assert_eq!(gpu_labels, cpu_labels, "GPU and CPU label maps must be identical");
    }
}
