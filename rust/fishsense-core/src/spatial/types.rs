use ndarray::{Array1, Array2};

pub struct ImageCoord(pub Array1<f32>);
pub struct DepthCoord(pub Array1<f32>);

pub struct DepthMap(pub Array2<f32>);