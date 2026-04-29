use thiserror::Error;

#[derive(Error, Debug)]
pub enum FishSenseError {
    #[error(transparent)]
    AnyhowError(#[from] anyhow::Error),
}
