use thiserror::Error;

#[derive(Error, Debug)]
pub enum FishSenseError {
    #[error("A GPU could not be aquired.")]
    CannotAcquireGpu,
    #[error(transparent)]
    WgpuRequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error(transparent)]
    AnyhowError(#[from] anyhow::Error),
    #[error("An unknown error occurred")]
    UnknownError
}