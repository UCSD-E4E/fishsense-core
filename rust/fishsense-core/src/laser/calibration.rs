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
