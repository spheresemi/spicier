//! Error types for Metal/WebGPU backend operations.

use std::fmt;

/// Metal/WebGPU backend error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum WgpuError {
    /// GPU device initialization failed.
    DeviceInit(String),
    /// No suitable GPU adapter found.
    NoAdapter,
    /// Shader compilation failed.
    ShaderCompile(String),
    /// Buffer creation or mapping failed.
    Buffer(String),
    /// Compute pipeline creation failed.
    Pipeline(String),
    /// Invalid dimension or size.
    InvalidDimension(String),
    /// GPU operation failed.
    Compute(String),
    /// Buffer size exceeds GPU limits.
    BufferTooLarge {
        /// Required buffer size in bytes.
        required: u64,
        /// Maximum buffer size supported.
        max_buffer: u64,
    },
    /// Out of GPU memory.
    OutOfMemory(String),
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuError::DeviceInit(msg) => write!(f, "GPU device initialization failed: {}", msg),
            WgpuError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            WgpuError::ShaderCompile(msg) => write!(f, "Shader compilation failed: {}", msg),
            WgpuError::Buffer(msg) => write!(f, "GPU buffer operation failed: {}", msg),
            WgpuError::Pipeline(msg) => write!(f, "Compute pipeline creation failed: {}", msg),
            WgpuError::InvalidDimension(msg) => write!(f, "Invalid dimension: {}", msg),
            WgpuError::Compute(msg) => write!(f, "GPU compute operation failed: {}", msg),
            WgpuError::BufferTooLarge { required, max_buffer } => {
                write!(
                    f,
                    "Sweep requires {} bytes but max buffer size is {}. Use chunking or reduce sweep size.",
                    required, max_buffer
                )
            }
            WgpuError::OutOfMemory(msg) => write!(f, "Out of GPU memory: {}", msg),
        }
    }
}

impl std::error::Error for WgpuError {}

/// Result type for wgpu operations.
pub type Result<T> = std::result::Result<T, WgpuError>;
