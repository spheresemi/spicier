//! Error types for MPS backend operations.

use std::fmt;

/// MPS backend error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum MpsError {
    /// Device initialization failed.
    DeviceInit(String),
    /// No suitable GPU device found.
    NoDevice,
    /// Command queue creation failed.
    CommandQueue(String),
    /// Buffer creation failed.
    Buffer(String),
    /// Matrix descriptor creation failed.
    MatrixDescriptor(String),
    /// Invalid dimension or size.
    InvalidDimension(String),
    /// LU decomposition failed (e.g., singular matrix).
    Decomposition(String),
    /// Solve operation failed.
    Solve(String),
    /// GPU operation failed.
    Compute(String),
    /// Platform not supported (not macOS).
    UnsupportedPlatform,
}

impl fmt::Display for MpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MpsError::DeviceInit(msg) => write!(f, "MPS device initialization failed: {}", msg),
            MpsError::NoDevice => write!(f, "No Metal GPU device found"),
            MpsError::CommandQueue(msg) => write!(f, "Command queue creation failed: {}", msg),
            MpsError::Buffer(msg) => write!(f, "Buffer operation failed: {}", msg),
            MpsError::MatrixDescriptor(msg) => {
                write!(f, "Matrix descriptor creation failed: {}", msg)
            }
            MpsError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            MpsError::Decomposition(msg) => write!(f, "LU decomposition failed: {}", msg),
            MpsError::Solve(msg) => write!(f, "Solve operation failed: {}", msg),
            MpsError::Compute(msg) => write!(f, "GPU compute operation failed: {}", msg),
            MpsError::UnsupportedPlatform => {
                write!(f, "MPS backend is only supported on macOS")
            }
        }
    }
}

impl std::error::Error for MpsError {}

/// Result type for MPS operations.
pub type Result<T> = std::result::Result<T, MpsError>;
