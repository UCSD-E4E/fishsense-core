use ndarray::{Array1, Array2};
use ndarray_linalg::Norm;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LaserCalibrationError {
    #[error("Centroid calculation failed.")]
    CentroidCalculationError,
}

pub fn calibrate_laser(points: &Array2<f32>) -> Result<(ndarray::Array1<f32>, ndarray::Array1<f32>), LaserCalibrationError> {
    let mut laser_orientation = Array1::<f32>::zeros(3);
    let &[n, _] = points.shape() else {
        panic!("Shape is not 2-dimensional!") // Panic is okay here since the dimensions are known at compile time
    };

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut v = &points.row(i) - &points.row(j);
            if v[2] < 0.0 {
                v = -v;
            }
            laser_orientation += &v;
        }
    }

    laser_orientation /= laser_orientation.norm();
    if laser_orientation[2] < 0.0 {
        laser_orientation = -laser_orientation;
    }

    let centroid_option = points.mean_axis(ndarray::Axis(0));
    if let Some(centroid) = centroid_option {
        let scale_factor = centroid[2] / laser_orientation[2];
        let mut laser_origin = &centroid - &(scale_factor * &laser_orientation);
        laser_origin[2] = 0.0; // Ensure laser origin is on the z=0 plane

        Ok((laser_origin, laser_orientation))
    } else {
        Err(LaserCalibrationError::CentroidCalculationError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f32 = 1e-5;

    fn assert_approx_eq(a: f32, b: f32, eps: f32) {
        assert!(
            (a - b).abs() < eps,
            "Expected {a} ≈ {b} (within {eps})"
        );
    }

    /// Points on the line z-axis: (0,0,1) and (0,0,2).
    /// The laser orientation should be (0,0,1) and the origin (0,0,0).
    #[test]
    fn test_vertical_line_two_points() {
        let points: Array2<f32> = array![[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];
        let (origin, orientation) = calibrate_laser(&points).unwrap();

        assert_approx_eq(origin[0], 0.0, EPSILON);
        assert_approx_eq(origin[1], 0.0, EPSILON);
        assert_approx_eq(origin[2], 0.0, EPSILON);

        assert_approx_eq(orientation[0], 0.0, EPSILON);
        assert_approx_eq(orientation[1], 0.0, EPSILON);
        assert_approx_eq(orientation[2], 1.0, EPSILON);
    }

    /// Points on the line (t, 0, t) for t=1,2,3 (45° in the xz-plane).
    /// Orientation should be (1/√2, 0, 1/√2) and origin (0,0,0).
    #[test]
    fn test_diagonal_line_three_points() {
        let points: Array2<f32> = array![
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0]
        ];
        let (origin, orientation) = calibrate_laser(&points).unwrap();

        let expected_component = std::f32::consts::FRAC_1_SQRT_2;
        assert_approx_eq(orientation[0], expected_component, EPSILON);
        assert_approx_eq(orientation[1], 0.0, EPSILON);
        assert_approx_eq(orientation[2], expected_component, EPSILON);

        assert_approx_eq(origin[0], 0.0, EPSILON);
        assert_approx_eq(origin[1], 0.0, EPSILON);
        assert_approx_eq(origin[2], 0.0, EPSILON);
    }

    /// The returned laser origin must always lie on the z=0 plane.
    #[test]
    fn test_origin_on_z0_plane() {
        let points: Array2<f32> = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let (origin, _orientation) = calibrate_laser(&points).unwrap();
        assert_approx_eq(origin[2], 0.0, EPSILON);
    }

    /// The returned laser orientation must be a unit vector.
    #[test]
    fn test_orientation_is_unit_vector() {
        let points: Array2<f32> = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let (_origin, orientation) = calibrate_laser(&points).unwrap();
        let norm: f32 = orientation.norm();
        assert_approx_eq(norm, 1.0, EPSILON);
    }

    /// The orientation z-component must always be positive (pointing forward).
    #[test]
    fn test_orientation_z_positive() {
        // Points on a line going in the negative-z direction from the camera's perspective.
        // The algorithm should flip the orientation to ensure z > 0.
        let points: Array2<f32> = array![[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]];
        let (_origin, orientation) = calibrate_laser(&points).unwrap();
        assert!(orientation[2] > 0.0, "Orientation z-component must be positive");
    }

    /// Non-collinear points: the origin must still lie on z=0 and the
    /// orientation must still be a unit vector with positive z.
    #[test]
    fn test_non_collinear_points() {
        let points: Array2<f32> = array![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0]
        ];
        let (origin, orientation) = calibrate_laser(&points).unwrap();
        assert_approx_eq(origin[2], 0.0, EPSILON);
        assert_approx_eq(orientation.norm(), 1.0, EPSILON);
        assert!(orientation[2] > 0.0);
    }
}
