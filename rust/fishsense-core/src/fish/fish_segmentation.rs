use std::cmp::{max, min};

use tracing::{debug, info, instrument, warn};

type InferenceOutputs = (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>);

use image::imageops::{resize, FilterType};
use image::{ImageBuffer, Luma};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point;
use ndarray::{s, Array2, Array3, ArrayD, IxDyn};

use crate::fish::fish_geometry::trace_outer_contours;
use ort::logging::LogLevel;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;
use thiserror::Error;

// The ONNX model is downloaded by build.rs and embedded at compile time.
// This means the binary is self-contained — no runtime network access is
// needed, which works correctly for both the Python wheel and the Flutter
// plugin.
static MODEL_BYTES: &[u8] = include_bytes!(env!("FISHIAL_MODEL_PATH"));

#[derive(Error, Debug)]
pub enum SegmentationError {
    #[error("image buffer → ndarray conversion failed: {0}")]
    CVToNDArrayError(String),
    #[error("fish not found in image")]
    FishNotFound,
    #[error("model has not been loaded — call load_model() first")]
    ModelLoadError,
    #[error("ndarray → image buffer conversion failed")]
    NDArrayToCVError,
    #[error("ORT error: {0}")]
    OrtErr(#[from] ort::Error),
    #[error("polygon not found after contour search")]
    PolyNotFound,
    #[error("ndarray shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
}

pub struct FishSegmentation {
    model_set: bool,
    model: Option<Session>,
    active_provider: Option<ActiveProvider>,
}

/// The execution provider that ORT registered for this session — useful for
/// telling whether a `cuda`-feature build actually got CUDA at runtime, since
/// ORT silently falls back to CPU when the CUDA libs can't be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveProvider {
    Cpu,
    Cuda,
    CoreMl,
}

impl ActiveProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            ActiveProvider::Cpu => "CPU",
            ActiveProvider::Cuda => "CUDA",
            ActiveProvider::CoreMl => "CoreML",
        }
    }
}

/// Diagnostic info per Mask R-CNN detection that survived the score threshold.
/// Produced by [`FishSegmentation::inference_debug`]; used by the Phase-1
/// segmentation-diagnostic binary. Not part of the stable API.
#[derive(Debug, Clone)]
pub struct DetectionDebug {
    pub index: usize,
    pub score: f32,
    pub bbox_xyxy: (f32, f32, f32, f32),
    pub mask_area_px: u32,
    pub polygon_vertices: usize,
    pub drawn: bool,
    pub drop_reason: Option<&'static str>,
}

impl FishSegmentation {
    pub(crate) const MIN_SIZE_TEST: usize = 800;
    pub(crate) const MAX_SIZE_TEST: usize = 1058;

    const SCORE_THRESHOLD: f32 = 0.3;
    const MASK_THRESHOLD: f32 = 0.5;
    /// Minimum pasted-mask area (at model resolution, in pixels) for a
    /// detection to be considered by [`inference_single`]. Guards against
    /// pathological tiny polygons without affecting real fish detections —
    /// the smallest drawn detection observed in the diagnostic fixture was
    /// ≈8,800 px.
    const MIN_SINGLE_INSTANCE_AREA_PX: u32 = 5_000;

    /// Creates a `FishSegmentation` that will use the model embedded at
    /// compile time by `build.rs`.  Call [`load_model`] before [`inference`].
    pub fn new() -> FishSegmentation {
        FishSegmentation {
            model_set: false,
            model: None,
            active_provider: None,
        }
    }

    /// Returns the execution provider that was registered when the session
    /// was created, or `None` if [`load_model`] hasn't been called yet.
    pub fn active_provider(&self) -> Option<ActiveProvider> {
        self.active_provider
    }

    /// Intra-op thread count for the ORT session.
    ///
    /// Defaults to `min(4, usable CPUs)`. Override with
    /// `FISHSENSE_ORT_INTRA_THREADS` — set it to `1` in environments where
    /// ORT's worker-thread pool misbehaves (it has intermittently deadlocked
    /// at thread-pool init on constrained CI runners; the ORT C call holds
    /// the GIL throughout, so that wedges the whole interpreter), or bump it
    /// up on a big host.
    fn intra_threads() -> usize {
        if let Some(n) = std::env::var("FISHSENSE_ORT_INTRA_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n >= 1)
        {
            return n;
        }
        std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(1)
    }

    fn build_session_options() -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        let builder = Session::builder()?.with_intra_threads(Self::intra_threads())?;

        // Silence ORT's per-kernel ERROR spam on the no-fish path. The upstream
        // FishIAL Mask R-CNN graph's mask head does `/Reshape_168`:
        // [N,1,56,56] → [N,1,-1], which ORT rejects when N == 0 — i.e. every
        // image where the detector finds nothing. We already catch the
        // resulting `ort::Error` and treat it as "no fish detected" (see
        // `do_inference` / `inference`), but ORT's C++ logger writes the
        // ExecuteKernel failure straight to fd 2 at ERROR severity *before* the
        // error propagates, bypassing Rust logging and any stderr redirection a
        // caller (e.g. an eval harness) set up. Raising the session log
        // severity to FATAL drops those expected, already-handled messages;
        // genuine problems still surface via the `tracing` macros below and via
        // the returned `Result`.
        let builder = builder.with_log_level(LogLevel::Fatal)?;

        // iOS enforces W^X, which blocks the runtime kernel fusion that
        // Level3 performs (causes EXC_BAD_ACCESS code=50). Level1 does only
        // ahead-of-time-safe optimizations.
        #[cfg(target_os = "ios")]
        let builder = builder.with_optimization_level(GraphOptimizationLevel::Level1)?;
        #[cfg(not(target_os = "ios"))]
        let builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;

        Ok(builder)
    }

    fn create_model() -> Result<(Session, ActiveProvider), ort::Error> {
        // Try accelerated EPs first with `error_on_failure` so registration
        // failures (e.g. missing CUDA libs at runtime) surface here and we
        // can fall back to CPU explicitly. ORT's default behaviour is to log
        // a warning and silently continue, which is exactly the silent-CPU
        // footgun this code path is fixing.

        #[cfg(feature = "cuda")]
        {
            let builder = Self::build_session_options()?;
            match builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .build()
                    .error_on_failure(),
            ]) {
                Ok(mut b) => {
                    info!("ORT registered CUDAExecutionProvider");
                    return b
                        .commit_from_memory(MODEL_BYTES)
                        .map(|s| (s, ActiveProvider::Cuda));
                }
                Err(e) => warn!("CUDA EP unavailable, falling back to CPU: {e}"),
            }
        }

        #[cfg(feature = "coreml")]
        {
            let builder = Self::build_session_options()?;
            match builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default()
                    .build()
                    .error_on_failure(),
            ]) {
                Ok(mut b) => {
                    info!("ORT registered CoreMLExecutionProvider");
                    return b
                        .commit_from_memory(MODEL_BYTES)
                        .map(|s| (s, ActiveProvider::CoreMl));
                }
                Err(e) => warn!("CoreML EP unavailable, falling back to CPU: {e}"),
            }
        }

        let mut builder = Self::build_session_options()?;
        builder
            .commit_from_memory(MODEL_BYTES)
            .map(|s| (s, ActiveProvider::Cpu))
    }

    #[instrument(skip(self))]
    pub fn load_model(&mut self) -> Result<(), SegmentationError> {
        if !self.model_set {
            debug!("loading embedded ONNX model");
            let (session, provider) = Self::create_model()?;
            self.model = Some(session);
            self.active_provider = Some(provider);
            self.model_set = true;
            info!(provider = provider.as_str(), "model loaded");
        } else {
            debug!("model already loaded, skipping");
        }
        Ok(())
    }

    fn get_model_mut(&mut self) -> Result<&mut Session, SegmentationError> {
        self.model.as_mut().ok_or(SegmentationError::ModelLoadError)
    }

    // ── polygon rasterization helper ─────────────────────────────────────

    /// Fills `poly` into `canvas` with intensity `value` — the pure-Rust
    /// equivalent of `cv::fillPoly` on a single-channel image.
    ///
    /// `imageproc::drawing::draw_polygon_mut` closes the ring implicitly and
    /// panics if the first and last vertices coincide, so a duplicated closing
    /// vertex (which contour traces never emit but polygon rescaling could
    /// round into existence) is trimmed first. Degenerate rings (<3 distinct
    /// vertices) enclose no area and are skipped.
    fn fill_polygon(canvas: &mut image::GrayImage, poly: &[Point<i32>], value: u8) {
        let pts: &[Point<i32>] = if poly.len() >= 2 && poly.first() == poly.last() {
            &poly[..poly.len() - 1]
        } else {
            poly
        };
        if pts.len() >= 3 {
            draw_polygon_mut(canvas, pts, Luma([value]));
        }
    }

    // ── Image pre-processing ─────────────────────────────────────────────

    pub(crate) fn pad_img(&self, img: &Array3<u8>) -> Array3<u8> {
        let (height, width, _) = img.dim();

        let mut pad_img = if height < width {
            Array3::zeros((Self::MIN_SIZE_TEST, Self::MAX_SIZE_TEST, 3))
        } else {
            Array3::zeros((Self::MAX_SIZE_TEST, Self::MIN_SIZE_TEST, 3))
        };

        pad_img.slice_mut(s![..height, ..width, ..]).assign(img);
        pad_img
    }

    pub(crate) fn resize_img(&self, img: &Array3<u8>) -> Result<Array3<u8>, SegmentationError> {
        let (height, width, _) = img.dim();

        let size = Self::MIN_SIZE_TEST as f32;
        let mut scale = size / min(height, width) as f32;

        let mut new_h: f32;
        let mut new_w: f32;
        if height < width {
            new_h = size;
            new_w = scale * width as f32;
        } else {
            new_h = scale * height as f32;
            new_w = size;
        }

        new_h = new_h.round();
        new_w = new_w.round();

        let max_side = max(new_h as usize, new_w as usize);
        if max_side > Self::MAX_SIZE_TEST {
            scale = Self::MAX_SIZE_TEST as f32 / max_side as f32;
            new_h *= scale;
            new_w *= scale;
        }

        let (rows, cols, ch) = img.dim();
        if ch != 3 {
            return Err(SegmentationError::NDArrayToCVError);
        }
        let arr = img.as_standard_layout();
        let raw: Vec<u8> = arr.iter().copied().collect();
        let buf: image::RgbImage = ImageBuffer::from_raw(cols as u32, rows as u32, raw)
            .ok_or(SegmentationError::NDArrayToCVError)?;
        let resized = resize(&buf, new_w as u32, new_h as u32, FilterType::Triangle);
        let (out_w, out_h) = (resized.width() as usize, resized.height() as usize);
        Array3::from_shape_vec((out_h, out_w, 3), resized.into_raw())
            .map_err(SegmentationError::ShapeError)
    }

    // ── Inference ────────────────────────────────────────────────────────

    fn do_inference(
        img: &Array3<f32>,
        model: &mut Session,
    ) -> Result<InferenceOutputs, ort::Error> {
        // Permute (H, W, C) → (C, H, W) and materialise a contiguous copy so
        // TensorRef::from_array_view doesn't reject a non-contiguous layout.
        let chw = img.view().permuted_axes([2, 0, 1]).as_standard_layout().into_owned();

        let input = TensorRef::from_array_view(chw.view())?;
        let outputs = model.run(ort::inputs!["argument_1.1" => input])?;

        // boxes=tensor18, classes=pred_classes, masks=5232, scores=2339
        let boxes = Self::extract_transposed(&outputs["tensor18"])?;
        let masks = Self::extract_transposed(&outputs["5232"])?;
        let scores = Self::extract_transposed(&outputs["2339"])?;

        Ok((boxes, masks, scores))
    }

    fn extract_transposed(value: &ort::value::DynValue) -> Result<ArrayD<f32>, ort::Error> {
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        let usize_shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&usize_shape), data.to_vec())
            .expect("ORT shape/data size mismatch");
        Ok(arr.t().into_owned())
    }

    fn do_paste_mask(
        &self,
        mask: &Array2<f32>,
        img_h: u32,
        img_w: u32,
    ) -> Result<Array2<f32>, SegmentationError> {
        let (h, w) = mask.dim();
        let arr = mask.as_standard_layout();
        let raw: Vec<f32> = arr.iter().copied().collect();
        let buf: ImageBuffer<Luma<f32>, Vec<f32>> =
            ImageBuffer::from_raw(w as u32, h as u32, raw)
                .ok_or(SegmentationError::NDArrayToCVError)?;
        let resized = resize(&buf, img_w, img_h, FilterType::Triangle);
        Array2::from_shape_vec((img_h as usize, img_w as usize), resized.into_raw())
            .map_err(SegmentationError::ShapeError)
    }

    /// Traces the outer contours of a binary bitmap ([`trace_outer_contours`],
    /// the pure-Rust stand-in for `cv::findContours` with RETR_CCOMP +
    /// CHAIN_APPROX_NONE), returning them ordered longest-first. Only outer
    /// boundaries are kept — the sole consumer takes the longest, which is
    /// always an outer contour. Returns `FishNotFound` when the bitmap has no
    /// foreground pixels.
    fn bitmap_to_polygon(
        &self,
        bitmap: &Array2<u8>,
    ) -> Result<Vec<Vec<Point<i32>>>, SegmentationError> {
        let mut polygons: Vec<Vec<Point<i32>>> = trace_outer_contours(bitmap)
            .into_iter()
            .map(|c| c.into_iter().map(|p| Point::new(p[0], p[1])).collect())
            .collect();

        if polygons.is_empty() {
            return Err(SegmentationError::FishNotFound);
        }

        polygons.sort_by_key(|p| std::cmp::Reverse(p.len()));
        Ok(polygons)
    }

    fn rescale_polygon(
        &self,
        poly: &[Point<i32>],
        start_x: u32,
        start_y: u32,
        width_scale: f32,
        height_scale: f32,
    ) -> Vec<Point<i32>> {
        poly.iter()
            .map(|p| {
                Point::new(
                    ((start_x as f32 + p.x as f32).ceil() * width_scale) as i32,
                    ((start_y as f32 + p.y as f32).ceil() * height_scale) as i32,
                )
            })
            .collect()
    }

    fn convert_output_to_mask(
        &self,
        boxes: &ArrayD<f32>,
        masks: &ArrayD<f32>,
        scores: &ArrayD<f32>,
        width_scale: f32,
        height_scale: f32,
        shape: (usize, usize, usize),
    ) -> Result<Array2<u8>, SegmentationError> {
        let mut masks_t = masks.clone();
        masks_t.swap_axes(3, 2);
        masks_t.swap_axes(2, 1);
        masks_t.swap_axes(1, 0);
        masks_t.swap_axes(1, 2);

        let mut complete_mask = image::GrayImage::new(shape.1 as u32, shape.0 as u32);

        let mask_count = scores.len();
        for ind in 0..mask_count {
            if scores[ind] <= Self::SCORE_THRESHOLD {
                continue;
            }

            let x1 = boxes[[0, ind]].ceil() as u32;
            let y1 = boxes[[1, ind]].ceil() as u32;
            let x2 = boxes[[2, ind]].floor() as u32;
            let y2 = boxes[[3, ind]].floor() as u32;
            let (mask_h, mask_w) = (y2 - y1 + 1, x2 - x1 + 1);

            let mask_2d = masks_t
                .slice(s![ind, .., .., 0])
                .mapv(|v| v)
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| SegmentationError::NDArrayToCVError)?;

            let np_mask = self
                .do_paste_mask(&mask_2d, mask_h, mask_w)?
                .mapv(|v| if v > Self::MASK_THRESHOLD { 255u8 } else { 0u8 });

            match self.bitmap_to_polygon(&np_mask) {
                Ok(contours) => {
                    if contours.is_empty() {
                        continue;
                    }
                    let poly = contours
                        .first()
                        .ok_or(SegmentationError::PolyNotFound)?;
                    if poly.len() < 10 {
                        continue;
                    }

                    let polygon_full =
                        self.rescale_polygon(poly, x1, y1, width_scale, height_scale);

                    let color = (ind + 1) as u8;
                    Self::fill_polygon(&mut complete_mask, &polygon_full, color);
                }
                Err(SegmentationError::FishNotFound) => continue,
                Err(e) => return Err(e),
            }
        }

        Array2::from_shape_vec((shape.0, shape.1), complete_mask.into_raw())
            .map_err(SegmentationError::ShapeError)
    }

    /// Same as [`convert_output_to_mask`] but also records per-detection debug info.
    /// Kept separate from the production path so the hot loop doesn't allocate
    /// a DetectionDebug per detection in normal use.
    fn convert_output_to_mask_debug(
        &self,
        boxes: &ArrayD<f32>,
        masks: &ArrayD<f32>,
        scores: &ArrayD<f32>,
        width_scale: f32,
        height_scale: f32,
        shape: (usize, usize, usize),
    ) -> Result<(Array2<u8>, Vec<DetectionDebug>), SegmentationError> {
        let mut masks_t = masks.clone();
        masks_t.swap_axes(3, 2);
        masks_t.swap_axes(2, 1);
        masks_t.swap_axes(1, 0);
        masks_t.swap_axes(1, 2);

        let mut complete_mask = image::GrayImage::new(shape.1 as u32, shape.0 as u32);

        let mut debugs: Vec<DetectionDebug> = Vec::new();
        let mask_count = scores.len();
        for ind in 0..mask_count {
            let score = scores[ind];
            if score <= Self::SCORE_THRESHOLD {
                continue;
            }

            let x1f = boxes[[0, ind]];
            let y1f = boxes[[1, ind]];
            let x2f = boxes[[2, ind]];
            let y2f = boxes[[3, ind]];
            let x1 = x1f.ceil() as u32;
            let y1 = y1f.ceil() as u32;
            let x2 = x2f.floor() as u32;
            let y2 = y2f.floor() as u32;
            let (mask_h, mask_w) = (y2 - y1 + 1, x2 - x1 + 1);

            let mask_2d = masks_t
                .slice(s![ind, .., .., 0])
                .mapv(|v| v)
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| SegmentationError::NDArrayToCVError)?;

            let np_mask = self
                .do_paste_mask(&mask_2d, mask_h, mask_w)?
                .mapv(|v| if v > Self::MASK_THRESHOLD { 255u8 } else { 0u8 });
            let mask_area_px: u32 = np_mask.iter().map(|&v| (v > 0) as u32).sum();

            let mut dbg = DetectionDebug {
                index: ind,
                score,
                bbox_xyxy: (x1f, y1f, x2f, y2f),
                mask_area_px,
                polygon_vertices: 0,
                drawn: false,
                drop_reason: None,
            };

            match self.bitmap_to_polygon(&np_mask) {
                Ok(contours) => {
                    if contours.is_empty() {
                        dbg.drop_reason = Some("empty_contours");
                        debugs.push(dbg);
                        continue;
                    }
                    let poly = contours
                        .first()
                        .ok_or(SegmentationError::PolyNotFound)?;
                    dbg.polygon_vertices = poly.len();
                    if poly.len() < 10 {
                        dbg.drop_reason = Some("too_few_vertices");
                        debugs.push(dbg);
                        continue;
                    }

                    let polygon_full =
                        self.rescale_polygon(poly, x1, y1, width_scale, height_scale);

                    let color = (ind + 1) as u8;
                    Self::fill_polygon(&mut complete_mask, &polygon_full, color);
                    dbg.drawn = true;
                    debugs.push(dbg);
                }
                Err(SegmentationError::FishNotFound) => {
                    dbg.drop_reason = Some("fish_not_found");
                    debugs.push(dbg);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        let complete = Array2::from_shape_vec((shape.0, shape.1), complete_mask.into_raw())
            .map_err(SegmentationError::ShapeError)?;
        Ok((complete, debugs))
    }

    /// Diagnostic-only: runs the same pipeline as [`inference`] but also
    /// returns per-detection info (score, bbox, mask area, whether it was drawn).
    /// Used by the Phase-1 segmentation diagnostic binary. Does not change the
    /// semantics of [`inference`].
    pub fn inference_debug(
        &mut self,
        img: &Array3<u8>,
    ) -> Result<(Array2<u8>, Vec<DetectionDebug>), SegmentationError> {
        let (orig_h, orig_w, _) = img.dim();
        let resized = self.resize_img(img)?;
        let padded = self.pad_img(&resized).mapv(|v| v as f32);
        let (new_h, new_w, _) = resized.dim();
        let width_scale = orig_w as f32 / new_w as f32;
        let height_scale = orig_h as f32 / new_h as f32;

        let model = self.get_model_mut()?;
        match Self::do_inference(&padded, model) {
            Ok((boxes, masks, scores)) => self.convert_output_to_mask_debug(
                &boxes,
                &masks,
                &scores,
                width_scale,
                height_scale,
                img.dim(),
            ),
            Err(_) => Ok((Array2::<u8>::zeros((orig_h, orig_w)), Vec::new())),
        }
    }

    /// Selects the single detection maximizing `score × thresholded-mask area`
    /// (among those passing score, contour-vertex, and minimum-area filters)
    /// and returns a binary 0/255 mask for that detection only.  Returns
    /// `None` if no detection passes all filters.
    ///
    /// Ranking by confidence-weighted area (rather than area alone) prevents a
    /// low-confidence over-large detection — e.g. a fish mask that bled into an
    /// occluding hand, or a barely-above-threshold blob — from overriding a
    /// high-confidence tight fish detection. Validated against human head/tail
    /// labels: score×area is Pareto-safe vs largest-area and never picks worse.
    fn build_single_instance_mask(
        &self,
        boxes: &ArrayD<f32>,
        masks: &ArrayD<f32>,
        scores: &ArrayD<f32>,
        width_scale: f32,
        height_scale: f32,
        shape: (usize, usize, usize),
    ) -> Result<Option<Array2<u8>>, SegmentationError> {
        let mut masks_t = masks.clone();
        masks_t.swap_axes(3, 2);
        masks_t.swap_axes(2, 1);
        masks_t.swap_axes(1, 0);
        masks_t.swap_axes(1, 2);

        // Keep only the best polygon's full-resolution point list, so we
        // rasterize exactly once at the end. `best` holds (score×area, points).
        let mut best: Option<(f32, Vec<Point<i32>>)> = None;

        let mask_count = scores.len();
        for ind in 0..mask_count {
            if scores[ind] <= Self::SCORE_THRESHOLD {
                continue;
            }

            let x1f = boxes[[0, ind]];
            let y1f = boxes[[1, ind]];
            let x2f = boxes[[2, ind]];
            let y2f = boxes[[3, ind]];
            let x1 = x1f.ceil() as u32;
            let y1 = y1f.ceil() as u32;
            let x2 = x2f.floor() as u32;
            let y2 = y2f.floor() as u32;
            let (mask_h, mask_w) = (y2 - y1 + 1, x2 - x1 + 1);

            let mask_2d = masks_t
                .slice(s![ind, .., .., 0])
                .mapv(|v| v)
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| SegmentationError::NDArrayToCVError)?;

            let np_mask = self
                .do_paste_mask(&mask_2d, mask_h, mask_w)?
                .mapv(|v| if v > Self::MASK_THRESHOLD { 255u8 } else { 0u8 });
            let area: u32 = np_mask.iter().map(|&v| (v > 0) as u32).sum();
            if area < Self::MIN_SINGLE_INSTANCE_AREA_PX {
                continue;
            }
            // Confidence-weighted area: a bigger mask only wins if its detection
            // score backs it up. Safe to skip the polygon step for any detection
            // that cannot beat the current best on this metric.
            let metric = scores[ind] * area as f32;
            if let Some((best_metric, _)) = best.as_ref()
                && metric <= *best_metric
            {
                continue;
            }

            let contours = match self.bitmap_to_polygon(&np_mask) {
                Ok(c) => c,
                Err(SegmentationError::FishNotFound) => continue,
                Err(e) => return Err(e),
            };
            if contours.is_empty() {
                continue;
            }
            let poly = contours.first().ok_or(SegmentationError::PolyNotFound)?;
            if poly.len() < 10 {
                continue;
            }

            let full_res: Vec<Point<i32>> = poly
                .iter()
                .map(|p| {
                    Point::new(
                        ((x1 as f32 + p.x as f32).ceil() * width_scale) as i32,
                        ((y1 as f32 + p.y as f32).ceil() * height_scale) as i32,
                    )
                })
                .collect();
            best = Some((metric, full_res));
        }

        let Some((_, polygon_full)) = best else {
            return Ok(None);
        };

        let mut out = image::GrayImage::new(shape.1 as u32, shape.0 as u32);
        Self::fill_polygon(&mut out, &polygon_full, 255);
        let out = Array2::from_shape_vec((shape.0, shape.1), out.into_raw())
            .map_err(SegmentationError::ShapeError)?;
        Ok(Some(out))
    }

    /// Runs segmentation and returns a single-instance binary mask (0 or 255)
    /// of the largest-area fish detection.  Returns `None` if no detection
    /// passes the score, contour-vertex, and minimum-area filters.  Use this
    /// when the downstream consumer expects one-fish-per-image; use
    /// [`inference`] when per-instance IDs matter.
    #[instrument(skip(self, img), fields(height = img.dim().0, width = img.dim().1))]
    pub fn inference_single(
        &mut self,
        img: &Array3<u8>,
    ) -> Result<Option<Array2<u8>>, SegmentationError> {
        let (orig_h, orig_w, _) = img.dim();

        let resized = self.resize_img(img)?;
        let padded = self.pad_img(&resized).mapv(|v| v as f32);
        let (new_h, new_w, _) = resized.dim();
        let width_scale = orig_w as f32 / new_w as f32;
        let height_scale = orig_h as f32 / new_h as f32;

        let model = self.get_model_mut()?;
        // Same contract as `inference`: ORT errors are treated as "no fish".
        match Self::do_inference(&padded, model) {
            Ok((boxes, masks, scores)) => self.build_single_instance_mask(
                &boxes,
                &masks,
                &scores,
                width_scale,
                height_scale,
                (orig_h, orig_w, 1),
            ),
            Err(ort_err) => {
                debug!("ORT inference error (likely no fish): {ort_err}");
                Ok(None)
            }
        }
    }

    #[instrument(skip(self, img), fields(height = img.dim().0, width = img.dim().1))]
    pub fn inference(&mut self, img: &Array3<u8>) -> Result<Array2<u8>, SegmentationError> {
        let (orig_h, orig_w, _) = img.dim();

        let resized = self.resize_img(img)?;
        let padded = self.pad_img(&resized).mapv(|v| v as f32);
        let (new_h, new_w, _) = resized.dim();
        debug!(orig_h, orig_w, resized_h = new_h, resized_w = new_w, "image pre-processed");

        let width_scale = orig_w as f32 / new_w as f32;
        let height_scale = orig_h as f32 / new_h as f32;

        let model = self.get_model_mut()?;

        // The ONNX model crashes (rather than returning an empty result) when
        // no fish are present.  Treat any ORT session-run error as "no fish
        // detected" and return an all-zero mask so callers handle it gracefully.
        match Self::do_inference(&padded, model) {
            Ok((boxes, masks, scores)) => {
                let n_detections = scores.iter().filter(|&&s| s > Self::SCORE_THRESHOLD).count();
                debug!(n_detections, "inference succeeded");
                self.convert_output_to_mask(
                    &boxes,
                    &masks,
                    &scores,
                    width_scale,
                    height_scale,
                    img.dim(),
                )
            }
            Err(ort_err) => {
                debug!("ORT inference error (likely no fish): {ort_err}");
                Ok(Array2::<u8>::zeros((orig_h, orig_w)))
            }
        }
    }
}

impl Default for FishSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn seg() -> FishSegmentation {
        FishSegmentation::new()
    }

    // ── pad_img ───────────────────────────────────────────────────────────

    /// Landscape image (height < width) → padded to (MIN, MAX, 3).
    #[test]
    fn test_pad_img_landscape_dims() {
        let padded = seg().pad_img(&Array3::<u8>::zeros((100, 200, 3)));
        assert_eq!(
            padded.dim(),
            (
                FishSegmentation::MIN_SIZE_TEST,
                FishSegmentation::MAX_SIZE_TEST,
                3
            )
        );
    }

    /// Portrait image (height > width) → padded to (MAX, MIN, 3).
    #[test]
    fn test_pad_img_portrait_dims() {
        let padded = seg().pad_img(&Array3::<u8>::zeros((200, 100, 3)));
        assert_eq!(
            padded.dim(),
            (
                FishSegmentation::MAX_SIZE_TEST,
                FishSegmentation::MIN_SIZE_TEST,
                3
            )
        );
    }

    /// Square image (height == width) — not strictly landscape, so portrait
    /// layout (MAX, MIN, 3).
    #[test]
    fn test_pad_img_square_uses_portrait_dims() {
        let padded = seg().pad_img(&Array3::<u8>::zeros((100, 100, 3)));
        assert_eq!(
            padded.dim(),
            (
                FishSegmentation::MAX_SIZE_TEST,
                FishSegmentation::MIN_SIZE_TEST,
                3
            )
        );
    }

    /// Original pixel values appear at the same (row, col, ch) position.
    #[test]
    fn test_pad_img_content_preserved() {
        let mut img: Array3<u8> = Array3::zeros((5, 10, 3));
        img[[2, 3, 1]] = 42;
        let padded = seg().pad_img(&img);
        assert_eq!(padded[[2, 3, 1]], 42);
    }

    /// Pixels outside the original extents are zero (not copied junk).
    #[test]
    fn test_pad_img_zeros_in_padded_region() {
        let img: Array3<u8> = Array3::from_elem((5, 10, 3), 255u8);
        let padded = seg().pad_img(&img);
        assert_eq!(padded[[6, 0, 0]], 0, "row below original should be zero");
        assert_eq!(padded[[0, 11, 0]], 0, "col right of original should be zero");
    }

    // ── resize_img ────────────────────────────────────────────────────────

    /// Landscape with aspect ratio ≤ MAX/MIN (≈1.32): the shorter side (height)
    /// is scaled to exactly MIN_SIZE_TEST without triggering the max-side clamp.
    /// 100×125 → scale=8 → 800×1000; max=1000 < 1058, so min side stays 800.
    #[test]
    fn test_resize_img_landscape_min_side_is_800() {
        let img: Array3<u8> = Array3::zeros((100, 125, 3));
        let resized = seg().resize_img(&img).expect("resize failed");
        let (h, w, _) = resized.dim();
        assert_eq!(h.min(w), FishSegmentation::MIN_SIZE_TEST);
    }

    /// Portrait with aspect ratio ≤ MAX/MIN (≈1.32): the shorter side (width)
    /// is scaled to exactly MIN_SIZE_TEST.
    /// 125×100 → scale=8 → 1000×800; max=1000 < 1058, so min side stays 800.
    #[test]
    fn test_resize_img_portrait_min_side_is_800() {
        let img: Array3<u8> = Array3::zeros((125, 100, 3));
        let resized = seg().resize_img(&img).expect("resize failed");
        let (h, w, _) = resized.dim();
        assert_eq!(h.min(w), FishSegmentation::MIN_SIZE_TEST);
    }

    /// Wide image (ratio > MAX/MIN): the max-side clamp kicks in and the longer
    /// side is bounded to MAX_SIZE_TEST.
    /// 100×200 (2:1) → 800×1600 → clamped to 529×1058; max=1058.
    #[test]
    fn test_resize_img_max_side_bounded() {
        let img: Array3<u8> = Array3::zeros((100, 200, 3));
        let resized = seg().resize_img(&img).expect("resize failed");
        let (h, w, _) = resized.dim();
        assert!(
            h.max(w) <= FishSegmentation::MAX_SIZE_TEST,
            "max side exceeded MAX_SIZE_TEST: got {}",
            h.max(w)
        );
    }

    /// Square image: both sides should become MIN_SIZE_TEST × MIN_SIZE_TEST.
    #[test]
    fn test_resize_img_square_becomes_800x800() {
        let img: Array3<u8> = Array3::zeros((100, 100, 3));
        let resized = seg().resize_img(&img).expect("resize failed");
        let (h, w, _) = resized.dim();
        assert_eq!(h, FishSegmentation::MIN_SIZE_TEST);
        assert_eq!(w, FishSegmentation::MIN_SIZE_TEST);
    }

    // ── inference (integration) ───────────────────────────────────────────

    /// Smoke test: load the embedded model and run inference on a blank image.
    /// A blank image contains no fish, so the model returns an all-zero mask.
    /// This verifies the full pipeline (model load → resize → pad → ORT → mask)
    /// without requiring any external fixture file.
    #[test]
    fn inference_smoke() {
        let img: Array3<u8> = Array3::zeros((480, 640, 3));
        let mut s = FishSegmentation::new();
        s.load_model().unwrap();
        let result = s.inference(&img).unwrap();
        assert_eq!(result.dim(), (480, 640), "output shape must match input H×W");
    }

    /// A no-fish image makes the FishIAL Mask R-CNN mask head fail its
    /// empty-batch `/Reshape_168` kernel. `inference_smoke` already covers the
    /// behavioural half of the fix (it returns `Ok` with an all-zero mask);
    /// this covers the other half — `build_session_options` raised the ORT
    /// session log severity to FATAL, so ORT must not write that kernel error
    /// to stderr. We redirect fd 2 around the call and assert it stays empty.
    #[test]
    fn inference_no_fish_is_silent_on_stderr() {
        use gag::BufferRedirect;
        use std::io::Read;

        let img: Array3<u8> = Array3::zeros((480, 640, 3));
        let mut s = FishSegmentation::new();
        s.load_model().unwrap();

        let mut captured = String::new();
        {
            let mut buf = BufferRedirect::stderr().expect("redirect stderr");
            let result = s.inference(&img).unwrap();
            assert_eq!(result.dim(), (480, 640));
            buf.read_to_string(&mut captured).unwrap();
        }
        assert!(
            captured.is_empty(),
            "ORT wrote to stderr on a no-fish image (session log severity not suppressed?):\n{captured}"
        );
    }

    /// Default-feature builds have no accelerator EP; `active_provider` must
    /// report CPU after `load_model` and `None` before it. This is the CI-
    /// friendly counterpart to the GPU-only check that CUDA actually engages.
    #[test]
    fn active_provider_defaults_to_cpu() {
        let mut s = FishSegmentation::new();
        assert_eq!(s.active_provider(), None, "should be None before load_model");
        s.load_model().unwrap();
        // Without any accelerator feature, the only path through create_model
        // is the CPU fallback.
        #[cfg(not(any(feature = "cuda", feature = "coreml")))]
        assert_eq!(s.active_provider(), Some(ActiveProvider::Cpu));
        // With cuda or coreml enabled the value depends on runtime EP
        // registration; just assert it's set to *something*.
        #[cfg(any(feature = "cuda", feature = "coreml"))]
        assert!(s.active_provider().is_some());
    }

    // ── segmentation regression against a committed fixture ───────────────
    //
    // Locks the production `inference()` path — the entry point the mobile app
    // uses (fishsense-mobile `rust-bridge/src/lib.rs` calls `inference`, never
    // `inference_single`) — against a golden mask generated from a committed
    // fish photo. The comparison is a foreground intersection-over-union, not
    // a bit-exact match: ORT CPU inference can jitter by a few boundary pixels
    // across BLAS builds / hardware (the reason the old NPZ test used a
    // tolerance too), but a real regression in the resize → detect → contour →
    // fill pipeline moves the IoU well below the threshold.

    fn fixture_path(rel: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(rel)
    }

    /// Decodes a committed JPEG into a BGR `Array3<u8>` — the channel order the
    /// FishIAL model expects (matching the mobile bridge, which feeds BGR).
    /// Returns `None` when the fixture is absent so callers can skip cleanly.
    fn load_bgr_fixture(rel: &str) -> Option<Array3<u8>> {
        let path = fixture_path(rel);
        if !path.exists() {
            return None;
        }
        let rgb = image::open(&path).expect("decode fixture jpeg").to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let arr = Array3::from_shape_vec((h, w, 3), rgb.into_raw()).expect("rgb buffer shape");
        // RGB → BGR: reverse the channel axis and materialise contiguously.
        Some(arr.slice(s![.., .., ..;-1]).to_owned())
    }

    /// Foreground (non-zero) intersection-over-union of two equal-shaped masks.
    fn foreground_iou(a: &Array2<u8>, b: &Array2<u8>) -> f64 {
        let (mut inter, mut union) = (0u64, 0u64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (fx, fy) = (x > 0, y > 0);
            if fx || fy {
                union += 1;
                if fx && fy {
                    inter += 1;
                }
            }
        }
        if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        }
    }

    /// Regression: the production `inference()` mask on the committed fish
    /// fixture stays within tolerance of the committed golden.
    ///
    /// After an *intentional* pipeline change, regenerate the golden with:
    ///   FISHSENSE_REGEN_GOLDEN=1 cargo test -p fishsense-core inference_matches_golden
    /// then commit `tests/fixtures/segmentation/mask.png`.
    #[test]
    fn inference_matches_golden() {
        let Some(img) = load_bgr_fixture("segmentation/rgb.jpg") else {
            eprintln!("skipping inference_matches_golden: fixture rgb.jpg absent");
            return;
        };

        let mut s = FishSegmentation::new();
        s.load_model().unwrap();
        let result = s.inference(&img).unwrap();

        let golden_path = fixture_path("segmentation/mask.png");
        if std::env::var_os("FISHSENSE_REGEN_GOLDEN").is_some() {
            let (h, w) = result.dim();
            let bin: Vec<u8> = result.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect();
            image::GrayImage::from_raw(w as u32, h as u32, bin)
                .expect("golden buffer")
                .save(&golden_path)
                .expect("write golden png");
            eprintln!(
                "regenerated golden: {} foreground px",
                result.iter().filter(|&&v| v > 0).count()
            );
            return;
        }

        if !golden_path.exists() {
            eprintln!("skipping inference_matches_golden: golden mask.png absent");
            return;
        }
        let golden = image::open(&golden_path).expect("decode golden").to_luma8();
        let (gw, gh) = (golden.width() as usize, golden.height() as usize);
        let golden = Array2::from_shape_vec((gh, gw), golden.into_raw()).expect("golden shape");

        assert_eq!(result.dim(), golden.dim(), "mask dims differ from golden");
        assert!(
            golden.iter().any(|&v| v > 0),
            "golden has no foreground — regenerate it"
        );
        let iou = foreground_iou(&result, &golden);
        assert!(
            iou >= 0.99,
            "inference() foreground IoU {iou:.4} vs golden below 0.99 — segmentation regressed"
        );
    }

    // ── inference_single ─────────────────────────────────────────────────
    //
    // Smoke test: a blank (all-zero) image produces no detections, so
    // `inference_single` must return `None` rather than an empty mask.
    // This is the legibility fix versus `inference`, which returns an
    // all-zero mask in the same scenario.
    #[test]
    fn inference_single_blank_returns_none() {
        let img: Array3<u8> = Array3::zeros((480, 640, 3));
        let mut s = FishSegmentation::new();
        s.load_model().unwrap();
        assert!(s.inference_single(&img).unwrap().is_none());
    }

    /// Counts the number of 4-connected components of non-zero pixels.
    fn count_ccs(mask: &Array2<u8>) -> usize {
        let (h, w) = mask.dim();
        let mut seen = vec![false; h * w];
        let mut n = 0;
        for y in 0..h {
            for x in 0..w {
                if mask[[y, x]] == 0 || seen[y * w + x] {
                    continue;
                }
                n += 1;
                let mut stack = vec![(y, x)];
                while let Some((cy, cx)) = stack.pop() {
                    if seen[cy * w + cx] || mask[[cy, cx]] == 0 {
                        continue;
                    }
                    seen[cy * w + cx] = true;
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
            }
        }
        n
    }

    /// Regression test: on the bundled NPZ, `inference_single` returns a
    /// binary mask (0/255) whose non-zero area matches the largest instance
    /// in the reference multi-instance output, with exactly one connected
    /// component.  Proves the single-instance API picks the intended fish.
    #[test]
    fn inference_single_npz_matches_largest_instance() {
        use ndarray_npy::NpzReader;
        use std::collections::BTreeMap;

        // Out-of-tree fixture (large real-image NPZ). Runs when present, skips
        // cleanly when not — no `#[ignore]`, no `-- --ignored` flag needed.
        let Ok(file) = std::fs::File::open("data/fish_segmentation.npz") else {
            eprintln!("skipping inference_single_npz_matches_largest_instance: fixture absent");
            return;
        };
        let mut npz = NpzReader::new(file).unwrap();
        let img8: Array3<u8> = npz.by_name("img8").unwrap();
        let truth: Array2<i32> = npz.by_name("segmentations").unwrap();

        let mut s = FishSegmentation::new();
        s.load_model().unwrap();
        let mask = s
            .inference_single(&img8)
            .unwrap()
            .expect("fixture has a fish; inference_single must return Some");

        assert_eq!(mask.dim(), truth.dim());
        assert!(
            mask.iter().all(|&v| v == 0 || v == 255),
            "mask must be binary 0/255"
        );
        assert_eq!(count_ccs(&mask), 1, "single-instance mask must have exactly one connected component");

        let mut counts: BTreeMap<i32, u64> = BTreeMap::new();
        for &v in truth.iter() {
            if v > 0 {
                *counts.entry(v).or_insert(0) += 1;
            }
        }
        let expected = *counts.values().max().expect("truth has at least one instance");
        let got: u64 = mask.iter().map(|&v| (v > 0) as u64).sum();
        let ratio = got as f64 / expected as f64;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "area ratio {ratio} (got {got} vs expected largest-instance {expected}) out of tolerance"
        );
    }

    /// Regression test for the spurious_blob class of bug. Fixture NPZ
    /// contains an RGB image plus the expected largest-instance and
    /// second-largest-instance pixel counts observed against the original
    /// multi-instance `inference()` output. Generate with (from the repo root):
    /// ```python
    /// import numpy as np
    /// from PIL import Image
    /// base = "/path/to/fixture_segmentation/spurious_blob/case_07"
    /// rgb = np.array(Image.open(f"{base}/rgb.png").convert("RGB"), dtype=np.uint8)
    /// np.savez("rust/fishsense-core/data/seg_spurious_blob_fixture.npz",
    ///          rgb=rgb,
    ///          expected_largest_px=np.array([396846], dtype=np.uint64),
    ///          expected_second_px=np.array([94887], dtype=np.uint64))
    /// ```
    #[test]
    fn inference_single_rejects_spurious_blob() {
        use ndarray::Array1;
        use ndarray_npy::NpzReader;

        // Out-of-tree fixture (see doc comment for generation). Runs when
        // present, skips cleanly when not.
        let Ok(file) = std::fs::File::open("data/seg_spurious_blob_fixture.npz") else {
            eprintln!("skipping inference_single_rejects_spurious_blob: fixture absent");
            return;
        };
        let mut npz = NpzReader::new(file).unwrap();
        let rgb: Array3<u8> = npz.by_name("rgb").unwrap();
        let largest_arr: Array1<u64> = npz.by_name("expected_largest_px").unwrap();
        let second_arr: Array1<u64> = npz.by_name("expected_second_px").unwrap();
        let expected_largest: u64 = largest_arr[0];
        let expected_second: u64 = second_arr[0];

        let mut s = FishSegmentation::new();
        s.load_model().unwrap();
        let mask = s
            .inference_single(&rgb)
            .unwrap()
            .expect("fixture image contains a primary fish");

        assert!(mask.iter().all(|&v| v == 0 || v == 255), "binary 0/255");
        assert_eq!(
            count_ccs(&mask),
            1,
            "single-instance mask must have exactly one connected component — spurious blob leaked"
        );

        let got: u64 = mask.iter().map(|&v| (v > 0) as u64).sum();
        // Area should match the primary instance within 10 %.
        let primary_ratio = got as f64 / expected_largest as f64;
        assert!(
            (0.9..=1.1).contains(&primary_ratio),
            "area {got} px not within 10 % of primary {expected_largest}"
        );
        // And should NOT match the two-instance total (primary + secondary).
        let combined = expected_largest + expected_second;
        assert!(
            got < combined * 95 / 100,
            "area {got} px is close to combined {combined} — inference_single did not drop the secondary"
        );
    }
}
