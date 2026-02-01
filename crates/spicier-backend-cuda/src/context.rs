//! CUDA device context management.

use crate::error::{CudaError, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext as CudarCudaContext, CudaStream};
use std::sync::Arc;

/// CUDA context holding device and cuBLAS handle.
///
/// This is the main entry point for CUDA operations. Create one context
/// and reuse it for multiple operators to avoid reinitialization overhead.
pub struct CudaContext {
    /// CUDA context.
    pub(crate) ctx: Arc<CudarCudaContext>,
    /// Default stream.
    pub(crate) stream: Arc<CudaStream>,
    /// cuBLAS handle for BLAS operations.
    pub(crate) blas: CudaBlas,
}

impl CudaContext {
    /// Create a new CUDA context using device 0.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::NoDevice` if no CUDA device is available.
    /// Returns `CudaError::DeviceInit` if device initialization fails.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Create a new CUDA context using a specific device.
    pub fn with_device(device_id: usize) -> Result<Self> {
        let ctx = CudarCudaContext::new(device_id).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("no CUDA-capable device")
                || msg.contains("invalid device")
                || msg.contains("ordinal")
            {
                CudaError::NoDevice
            } else {
                CudaError::DeviceInit(msg)
            }
        })?;

        let stream = ctx.default_stream();

        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| CudaError::DeviceInit(format!("cuBLAS init failed: {}", e)))?;

        log::info!("CUDA context initialized on device {}", device_id);

        Ok(Self { ctx, stream, blas })
    }

    /// Check if CUDA is available on this system.
    pub fn is_available() -> bool {
        CudarCudaContext::new(0).is_ok()
    }

    /// Get the underlying CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get the underlying CUDA context.
    pub fn cuda_context(&self) -> &Arc<CudarCudaContext> {
        &self.ctx
    }
}
