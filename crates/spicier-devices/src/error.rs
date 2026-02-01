//! Error types for spicier-devices.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid device parameter: {0}")]
    InvalidParameter(String),

    #[error("device {name} has invalid value: {value}")]
    InvalidValue { name: String, value: f64 },
}

pub type Result<T> = std::result::Result<T, Error>;
