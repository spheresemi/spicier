//! Error types for CUDA backend operations.

use std::fmt;

/// CUDA backend error type.
#[derive(Debug)]
pub enum CudaError {
    /// CUDA device initialization failed.
    DeviceInit(String),
    /// CUDA memory allocation failed.
    MemoryAlloc(String),
    /// cuBLAS operation failed.
    Cublas(String),
    /// Data transfer error (host<->device).
    Transfer(String),
    /// Invalid dimension or size.
    InvalidDimension(String),
    /// No CUDA device available.
    NoDevice,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::DeviceInit(msg) => write!(f, "CUDA device initialization failed: {}", msg),
            CudaError::MemoryAlloc(msg) => write!(f, "CUDA memory allocation failed: {}", msg),
            CudaError::Cublas(msg) => write!(f, "cuBLAS operation failed: {}", msg),
            CudaError::Transfer(msg) => write!(f, "CUDA data transfer failed: {}", msg),
            CudaError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            CudaError::NoDevice => write!(f, "No CUDA device available"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations.
pub type Result<T> = std::result::Result<T, CudaError>;
