use std::cmp::{max, min};
use std::ffi::c_void;

use tracing::{debug, instrument};

type InferenceOutputs = (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>);

use ndarray::{s, Array2, Array3, ArrayD, Axis, IxDyn};
use opencv::core::{Mat, Point2i, Size, Vector, CV_8UC1, CV_8UC3, CV_32FC1};
use opencv::imgproc::{
    fill_poly, find_contours_with_hierarchy, resize_def, CHAIN_APPROX_NONE, LINE_8, RETR_CCOMP,
};
use opencv::prelude::MatTraitConst;
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
    #[error("CV → ndarray conversion failed: {0}")]
    CVToNDArrayError(String),
    #[error("fish not found in image")]
    FishNotFound,
    #[error("model has not been loaded — call load_model() first")]
    ModelLoadError,
    #[error("ndarray → CV conversion failed")]
    NDArrayToCVError,
    #[error("OpenCV error: {0}")]
    OpenCVError(#[from] opencv::Error),
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
        }
    }

    fn create_model() -> Result<Session, ort::Error> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;
        builder.commit_from_memory(MODEL_BYTES)
    }

    #[instrument(skip(self))]
    pub fn load_model(&mut self) -> Result<(), SegmentationError> {
        if !self.model_set {
            debug!("loading embedded ONNX model");
            self.model = Some(Self::create_model()?);
            self.model_set = true;
            debug!("model loaded");
        } else {
            debug!("model already loaded, skipping");
        }
        Ok(())
    }

    fn get_model_mut(&mut self) -> Result<&mut Session, SegmentationError> {
        self.model.as_mut().ok_or(SegmentationError::ModelLoadError)
    }

    // ── ndarray ↔ Mat helpers ────────────────────────────────────────────

    /// Wraps an `Array3<u8>` (rows, cols, ch) as a read-only OpenCV Mat.
    ///
    /// # Safety
    /// The returned Mat holds a raw pointer into `arr`'s data.  The caller
    /// must ensure `arr` outlives the Mat.
    unsafe fn array3_u8_as_mat(arr: &Array3<u8>) -> Result<Mat, SegmentationError> {
        let arr_c = arr.as_standard_layout();
        let (rows, cols, ch) = arr_c.dim();
        let type_code = match ch {
            1 => CV_8UC1,
            3 => CV_8UC3,
            _ => return Err(SegmentationError::NDArrayToCVError),
        };
        let data_ptr = arr_c.as_ptr() as *mut c_void;
        // SAFETY: data_ptr is valid for the lifetime of arr_c, which the
        // caller guarantees outlives the returned Mat.
        unsafe { Mat::new_rows_cols_with_data_unsafe_def(rows as i32, cols as i32, type_code, data_ptr) }
            .map_err(SegmentationError::OpenCVError)
    }

    /// Wraps an `Array3<f32>` (rows, cols, 1) as a read-only OpenCV Mat.
    ///
    /// # Safety
    /// Same lifetime contract as [`array3_u8_as_mat`].
    unsafe fn array3_f32_as_mat(arr: &Array3<f32>) -> Result<Mat, SegmentationError> {
        let arr_c = arr.as_standard_layout();
        let (rows, cols, ch) = arr_c.dim();
        if ch != 1 {
            return Err(SegmentationError::NDArrayToCVError);
        }
        let data_ptr = arr_c.as_ptr() as *mut c_void;
        // SAFETY: same as array3_u8_as_mat.
        unsafe { Mat::new_rows_cols_with_data_unsafe_def(rows as i32, cols as i32, CV_32FC1, data_ptr) }
            .map_err(SegmentationError::OpenCVError)
    }

    /// Copies an OpenCV `u8` Mat (CV_8UC1 or CV_8UC3) into an `Array3<u8>`.
    fn mat_to_array3_u8(mat: &Mat) -> Result<Array3<u8>, SegmentationError> {
        let rows = mat.rows() as usize;
        let cols = mat.cols() as usize;
        let ch = mat.channels() as usize;
        let total = rows * cols * ch;
        let slice = unsafe { std::slice::from_raw_parts(mat.data(), total) };
        Array3::from_shape_vec((rows, cols, ch), slice.to_vec())
            .map_err(SegmentationError::ShapeError)
    }

    /// Copies an OpenCV `f32` Mat (CV_32FC1) into an `Array3<f32>`.
    fn mat_to_array3_f32(mat: &Mat) -> Result<Array3<f32>, SegmentationError> {
        let rows = mat.rows() as usize;
        let cols = mat.cols() as usize;
        let total = rows * cols;
        let slice =
            unsafe { std::slice::from_raw_parts(mat.data() as *const f32, total) };
        Array3::from_shape_vec((rows, cols, 1), slice.to_vec())
            .map_err(SegmentationError::ShapeError)
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

        let mat = unsafe { Self::array3_u8_as_mat(img)? };
        let mut resized_cv = Mat::default();
        resize_def(&mat, &mut resized_cv, Size::new(new_w as i32, new_h as i32))?;
        Self::mat_to_array3_u8(&resized_cv)
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
        let mask3 = mask.clone().insert_axis(Axis(2));
        let mat = unsafe { Self::array3_f32_as_mat(&mask3)? };
        let mut resized_cv = Mat::default();
        resize_def(&mat, &mut resized_cv, Size::new(img_w as i32, img_h as i32))?;
        let resized3 = Self::mat_to_array3_f32(&resized_cv)?;
        Ok(resized3.remove_axis(Axis(2)))
    }

    fn bitmap_to_polygon(
        &self,
        bitmap: &Array2<u8>,
    ) -> Result<Vec<Vector<Point2i>>, SegmentationError> {
        let bitmap3 = bitmap.clone().insert_axis(Axis(2));
        let mat = unsafe { Self::array3_u8_as_mat(&bitmap3)? };

        let mut contours: Vector<Vector<Point2i>> = Vector::new();
        let mut hierarchy: Vector<opencv::core::Vec4i> = Vector::new();
        find_contours_with_hierarchy(
            &mat,
            &mut contours,
            &mut hierarchy,
            RETR_CCOMP,
            CHAIN_APPROX_NONE,
            Point2i::new(0, 0),
        )?;

        if hierarchy.is_empty() {
            return Err(SegmentationError::FishNotFound);
        }

        let mut contour_vec: Vec<Vector<Point2i>> = contours.iter().collect();
        contour_vec.sort_by_key(|v: &Vector<Point2i>| std::cmp::Reverse(v.len()));
        Ok(contour_vec)
    }

    fn rescale_polygon(
        &self,
        poly: &Vector<Point2i>,
        start_x: u32,
        start_y: u32,
        width_scale: f32,
        height_scale: f32,
    ) -> Vector<Point2i> {
        Vector::from_iter(poly.iter().map(|p| {
            Point2i::new(
                ((start_x as f32 + p.x as f32).ceil() * width_scale) as i32,
                ((start_y as f32 + p.y as f32).ceil() * height_scale) as i32,
            )
        }))
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

        let mut complete_mask_cv = Mat::new_rows_cols_with_default(
            shape.0 as i32,
            shape.1 as i32,
            CV_8UC1,
            0.into(),
        )?;

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

                    let color = (ind + 1) as i32;
                    fill_poly(
                        &mut complete_mask_cv,
                        &polygon_full,
                        (color, color, color).into(),
                        LINE_8,
                        0,
                        Point2i::new(0, 0),
                    )?;
                }
                Err(SegmentationError::FishNotFound) => continue,
                Err(e) => return Err(e),
            }
        }

        let complete3 = Self::mat_to_array3_u8(&complete_mask_cv)?;
        Ok(complete3.remove_axis(Axis(2)))
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

        let mut complete_mask_cv = Mat::new_rows_cols_with_default(
            shape.0 as i32,
            shape.1 as i32,
            CV_8UC1,
            0.into(),
        )?;

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

                    let color = (ind + 1) as i32;
                    fill_poly(
                        &mut complete_mask_cv,
                        &polygon_full,
                        (color, color, color).into(),
                        LINE_8,
                        0,
                        Point2i::new(0, 0),
                    )?;
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

        let complete3 = Self::mat_to_array3_u8(&complete_mask_cv)?;
        Ok((complete3.remove_axis(Axis(2)), debugs))
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

    /// Selects the single detection with the largest thresholded-mask area
    /// (among those passing score, contour-vertex, and minimum-area filters)
    /// and returns a binary 0/255 mask for that detection only.  Returns
    /// `None` if no detection passes all filters.
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
        // rasterize exactly once at the end.
        let mut best: Option<(u32, Vec<Point2i>)> = None;

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
            if let Some((best_area, _)) = best.as_ref()
                && area <= *best_area
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

            let full_res: Vec<Point2i> = poly
                .iter()
                .map(|p| {
                    Point2i::new(
                        ((x1 as f32 + p.x as f32).ceil() * width_scale) as i32,
                        ((y1 as f32 + p.y as f32).ceil() * height_scale) as i32,
                    )
                })
                .collect();
            best = Some((area, full_res));
        }

        let Some((_, full_res_poly)) = best else {
            return Ok(None);
        };

        let polygon_full: Vector<Point2i> = Vector::from_iter(full_res_poly);
        let mut out_cv = Mat::new_rows_cols_with_default(
            shape.0 as i32,
            shape.1 as i32,
            CV_8UC1,
            0.into(),
        )?;
        fill_poly(
            &mut out_cv,
            &polygon_full,
            (255, 255, 255).into(),
            LINE_8,
            0,
            Point2i::new(0, 0),
        )?;
        let out3 = Self::mat_to_array3_u8(&out_cv)?;
        Ok(Some(out3.remove_axis(Axis(2))))
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

    /// Pixel-accurate regression against the reference NPZ fixture.
    /// To run: place `data/fish_segmentation.npz` in the crate root, then
    ///   cargo test -p fishsense-core -- --ignored inference_npz
    #[test]
    #[ignore = "requires data/fish_segmentation.npz"]
    fn inference_npz() {
        use ndarray_npy::NpzReader;
        use ndarray_stats::DeviationExt;

        let mut npz =
            NpzReader::new(std::fs::File::open("data/fish_segmentation.npz").unwrap()).unwrap();
        let img8: Array3<u8> = npz.by_name("img8").unwrap();
        let truth: Array2<i32> = npz.by_name("segmentations").unwrap();

        let mut s = FishSegmentation::new();
        s.load_model().unwrap();
        let result = s.inference(&img8).unwrap().mapv(|v| v as i32);

        assert!(result.mean_abs_err(&truth).unwrap() < 2.0e-6);
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
    #[ignore = "requires data/fish_segmentation.npz"]
    fn inference_single_npz_matches_largest_instance() {
        use ndarray_npy::NpzReader;
        use std::collections::BTreeMap;

        let mut npz =
            NpzReader::new(std::fs::File::open("data/fish_segmentation.npz").unwrap()).unwrap();
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
    #[ignore = "requires data/seg_spurious_blob_fixture.npz (see test body for generation)"]
    fn inference_single_rejects_spurious_blob() {
        use ndarray::Array1;
        use ndarray_npy::NpzReader;

        let mut npz = NpzReader::new(
            std::fs::File::open("data/seg_spurious_blob_fixture.npz").unwrap(),
        )
        .unwrap();
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
