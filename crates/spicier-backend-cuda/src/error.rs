//! Error types for CUDA backend operations.

use std::fmt;

/// CUDA backend error type.
#[derive(Debug)]
#[non_exhaustive]
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
    /// Some matrices in a batched operation were singular.
    SingularBatch {
        /// Indices of singular matrices in the batch.
        indices: Vec<usize>,
    },
    /// Batch size exceeds cuBLAS limit.
    BatchTooLarge {
        /// Requested batch size.
        size: usize,
        /// Maximum allowed batch size.
        max: usize,
    },
    /// Compute operation failed (cuSPARSE, cuSOLVER, etc.).
    Compute(String),
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
            CudaError::SingularBatch { indices } => {
                write!(f, "Singular matrices in batch at indices: {:?}", indices)
            }
            CudaError::BatchTooLarge { size, max } => {
                write!(f, "Batch size {} exceeds maximum {}", size, max)
            }
            CudaError::Compute(msg) => write!(f, "CUDA compute operation failed: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations.
pub type Result<T> = std::result::Result<T, CudaError>;
