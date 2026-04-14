use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;

use crate::world_point_handler::WorldPointHandler;

pub struct FishLengthCalculator {
    pub world_point_handler: WorldPointHandler,
    pub image_height: usize,
    pub image_width: usize,
    pub depth_height: usize,
    pub depth_width: usize,
}

impl FishLengthCalculator {
    /// Samples `depth_map` at an image-space `(x, y)` coordinate.
    ///
    /// `coord` is `[x, y]` to match `WorldPointHandler` and `HeadTailCoords`.
    /// The depth map is indexed `[row, col]` = `[y, x]`. When the depth map
    /// resolution differs from the image resolution (e.g. ARKit LiDAR delivers
    /// ~256×192 depth alongside a ~1920×1440 RGB frame), the coordinate is
    /// rescaled and clamped into the depth grid.
    fn sample_depth(&self, depth_map: &Array2<f32>, coord: &Array1<f32>) -> f32 {
        let x = coord[0];
        let y = coord[1];
        let dx = ((x * self.depth_width as f32) / self.image_width as f32) as usize;
        let dy = ((y * self.depth_height as f32) / self.image_height as f32) as usize;
        let dx = dx.min(self.depth_width.saturating_sub(1));
        let dy = dy.min(self.depth_height.saturating_sub(1));
        depth_map[[dy, dx]]
    }

    /// Computes the 3-D distance between two image-space points using the
    /// depth map. `left_depth_coord` and `right_depth_coord` are `[x, y]`.
    pub fn calculate_fish_length(&self, depth_map: &Array2<f32>, left_depth_coord: &Array1<f32>, right_depth_coord: &Array1<f32>) -> f32 {
        let left_depth = self.sample_depth(depth_map, left_depth_coord);
        let right_depth = self.sample_depth(depth_map, right_depth_coord);

        let left_3d = self.world_point_handler.compute_world_point_from_depth(left_depth_coord, left_depth);
        let right_3d = self.world_point_handler.compute_world_point_from_depth(right_depth_coord, right_depth);

        (&left_3d - &right_3d).norm_l2()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::world_point_handler::WorldPointHandler;

    use super::FishLengthCalculator;

    #[test]
    fn calculate_fish_length() {
        let f_inv = 0.000_353_109_85_f32;
        let camera_intrinsics_inverted = array![[f_inv, 0f32, 0f32], [0f32, f_inv, 0f32], [0f32, 0f32, 1f32]];

        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted
        };

        let image_height = 3016;
        let image_width = 3987;
        let depth_map = Array2::from_elem((image_height, image_width), 0.535_531_04_f32);
        let fish_length_calcualtor = FishLengthCalculator {
            image_height,
            image_width,
            depth_height: image_height,
            depth_width: image_width,
            world_point_handler
        };

        let left = array![889.631_6_f32, 336.585_48_f32];
        let right = array![-355.368_4_f32, 395.585_48_f32];
        let fish_length = fish_length_calcualtor.calculate_fish_length(&depth_map, &left, &right);

        assert_eq!(fish_length, 0.23569532);
    }

    /// Same pixel for both endpoints → length should be zero.
    #[test]
    fn calculate_fish_length_same_point_is_zero() {
        let identity = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted: identity,
        };
        let depth_map = Array2::from_elem((10, 10), 1.0f32);
        let calc = FishLengthCalculator {
            image_height: 10,
            image_width: 10,
            depth_height: 10,
            depth_width: 10,
            world_point_handler,
        };
        let point = array![5.0f32, 5.0];
        let length = calc.calculate_fish_length(&depth_map, &point, &point);
        assert!(length.abs() < 1e-6, "expected 0.0, got {length}");
    }

    /// With identity intrinsics and unit depth, the 3-D distance equals the
    /// 2-D Euclidean distance between the two image coords. Coords are `[x, y]`,
    /// so left=[3,5] and right=[7,5] share row y=5; depth lookups are
    /// `depth_map[[5, 3]]` and `depth_map[[5, 7]]` (both = 1.0).
    /// Δx = 4, Δy = 0 → length = 4.
    #[test]
    fn calculate_fish_length_horizontal_separation() {
        let identity = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let world_point_handler = WorldPointHandler {
            camera_intrinsics_inverted: identity,
        };
        let depth_map = Array2::from_elem((10, 10), 1.0f32);
        let calc = FishLengthCalculator {
            image_height: 10,
            image_width: 10,
            depth_height: 10,
            depth_width: 10,
            world_point_handler,
        };
        // Both points share the same row (y=5) so row index is 5 for both.
        // left col=3 → depth_map[[5,3]]=1, right col=7 → depth_map[[5,7]]=1.
        let left = array![3.0f32, 5.0];
        let right = array![7.0f32, 5.0];
        let length = calc.calculate_fish_length(&depth_map, &left, &right);
        assert!((length - 4.0).abs() < 1e-5, "expected 4.0, got {length}");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Regression tests for two bugs found integrating with iOS / ARKit:
    //   1. depth-map resolution ≠ image resolution → out-of-bounds panic
    //      (non-unwinding across the FFI boundary, aborts the process).
    //   2. coord[0] was indexed as a row, contradicting WorldPointHandler
    //      and HeadTailCoords which both treat coord[0] as x.
    // ─────────────────────────────────────────────────────────────────────

    /// iOS regression: ARKit delivers ~256×192 depth alongside ~1920×1440 RGB.
    /// Image-space coords must be rescaled into the depth grid, not used as
    /// raw indices. Pre-fix this aborted the process.
    #[test]
    fn does_not_panic_when_depth_is_lower_resolution_than_image() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let calc = FishLengthCalculator {
            image_width: 1920,
            image_height: 1440,
            depth_width: 256,
            depth_height: 192,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity },
        };
        let depth_map = Array2::from_elem((192, 256), 1.0_f32);

        let head = array![689.0_f32, 401.0];
        let tail = array![855.0_f32, 1177.0];
        let length = calc.calculate_fish_length(&depth_map, &head, &tail);
        assert!(length.is_finite(), "expected finite length, got {length}");
    }

    /// With uniform depth, a downsampled depth map should give the same
    /// answer as a matched-resolution one.
    #[test]
    fn uniform_depth_is_resolution_invariant() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let head = array![100.0_f32, 200.0];
        let tail = array![900.0_f32, 200.0];

        let ref_calc = FishLengthCalculator {
            image_width: 1000,
            image_height: 800,
            depth_width: 1000,
            depth_height: 800,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity.clone() },
        };
        let ref_depth = Array2::from_elem((800, 1000), 2.0_f32);
        let reference = ref_calc.calculate_fish_length(&ref_depth, &head, &tail);

        let lo_calc = FishLengthCalculator {
            image_width: 1000,
            image_height: 800,
            depth_width: 250,
            depth_height: 200,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity },
        };
        let lo_depth = Array2::from_elem((200, 250), 2.0_f32);
        let downsampled = lo_calc.calculate_fish_length(&lo_depth, &head, &tail);

        assert!(
            (reference - downsampled).abs() < 1e-4,
            "matched and downsampled disagreed: ref={reference}, ds={downsampled}",
        );
    }

    /// Edge coordinates must clamp into the depth grid rather than overflow it.
    #[test]
    fn edge_coordinates_clamp_into_depth_map() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let calc = FishLengthCalculator {
            image_width: 1920,
            image_height: 1440,
            depth_width: 256,
            depth_height: 192,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity },
        };
        let depth_map = Array2::from_elem((192, 256), 1.5_f32);
        let corner = array![1919.0_f32, 1439.0];
        let origin = array![0.0_f32, 0.0];
        let length = calc.calculate_fish_length(&depth_map, &origin, &corner);
        assert!(length.is_finite());
    }

    /// `coord[0]` must be x, not row. Tall image (more rows than cols);
    /// depth varies only with x. A swapped index would either read the wrong
    /// cell or panic on the smaller dimension.
    #[test]
    fn coord_zero_is_x_not_row() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let image_width = 20_usize;
        let image_height = 100_usize;

        let mut depth_map = Array2::<f32>::zeros((image_height, image_width));
        for y in 0..image_height {
            for x in 0..image_width {
                depth_map[[y, x]] = x as f32 + 1.0;
            }
        }

        let calc = FishLengthCalculator {
            image_width,
            image_height,
            depth_width: image_width,
            depth_height: image_height,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity },
        };

        // Same row (y=50), different columns (x=5 vs x=15).
        // Correct: depths 6 and 16. Swapped: would index row 5 / row 15.
        let left = array![5.0_f32, 50.0];
        let right = array![15.0_f32, 50.0];
        let length = calc.calculate_fish_length(&depth_map, &left, &right);

        // Identity intrinsics: world = [x, y, 1] * depth.
        //   left_3d  = [ 30, 300,  6]
        //   right_3d = [240, 800, 16]
        //   diff     = [210, 500, 10]
        let expected = (210.0_f32.powi(2) + 500.0_f32.powi(2) + 10.0_f32.powi(2)).sqrt();
        assert!(
            (length - expected).abs() < 1e-3,
            "expected {expected}, got {length} — coord[0] is being treated as a row",
        );
    }

    /// Mirror: `coord[1]` must be y, not column. Wide image, depth varies
    /// only with y.
    #[test]
    fn coord_one_is_y_not_col() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let image_width = 100_usize;
        let image_height = 20_usize;

        let mut depth_map = Array2::<f32>::zeros((image_height, image_width));
        for y in 0..image_height {
            for x in 0..image_width {
                depth_map[[y, x]] = y as f32 + 1.0;
            }
        }

        let calc = FishLengthCalculator {
            image_width,
            image_height,
            depth_width: image_width,
            depth_height: image_height,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity },
        };

        let left = array![50.0_f32, 5.0];
        let right = array![50.0_f32, 15.0];
        let length = calc.calculate_fish_length(&depth_map, &left, &right);

        //   left_3d  = [300,  30,  6]
        //   right_3d = [800, 240, 16]
        //   diff     = [500, 210, 10]
        let expected = (500.0_f32.powi(2) + 210.0_f32.powi(2) + 10.0_f32.powi(2)).sqrt();
        assert!(
            (length - expected).abs() < 1e-3,
            "expected {expected}, got {length} — coord[1] is being treated as a column",
        );
    }

    /// End-to-end: non-square image, downsampled depth, asymmetric coords.
    /// If either bug regresses, this either panics or returns a wrong number.
    #[test]
    fn nonsquare_image_with_downsampled_depth_end_to_end() {
        let identity = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let calc = FishLengthCalculator {
            image_width: 1920,
            image_height: 1440,
            depth_width: 192,
            depth_height: 144,
            world_point_handler: WorldPointHandler { camera_intrinsics_inverted: identity },
        };
        let depth_map = Array2::from_elem((144, 192), 0.5_f32);

        let head = array![689.0_f32, 401.0];
        let tail = array![855.0_f32, 1177.0];
        let length = calc.calculate_fish_length(&depth_map, &head, &tail);

        //   head_3d = [344.5, 200.5, 0.5]
        //   tail_3d = [427.5, 588.5, 0.5]
        //   diff    = [83, 388, 0]
        let expected = (83.0_f32.powi(2) + 388.0_f32.powi(2)).sqrt();
        assert!(
            (length - expected).abs() < 1e-3,
            "expected {expected}, got {length}",
        );
    }
}