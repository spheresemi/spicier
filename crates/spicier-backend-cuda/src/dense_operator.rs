//! Dense matrix operators backed by GPU memory using cuBLAS.
//!
//! Implements both [`ComplexOperator`] and [`RealOperator`] using
//! cuBLAS zgemv (complex) and dgemv (real) respectively.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use cudarc::cublas::sys::{
    cublasDgemv_v2, cublasOperation_t, cublasZgemv_v2, cuDoubleComplex,
};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits};
use num_complex::Complex64 as C64;
use spicier_solver::operator::{ComplexOperator, RealOperator};
use std::sync::Arc;

/// GPU-resident representation of a complex number for CUDA.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuComplex {
    pub re: f64,
    pub im: f64,
}

// Safety: GpuComplex is a plain struct of two f64s with same layout as cuDoubleComplex
unsafe impl DeviceRepr for GpuComplex {}
unsafe impl ValidAsZeroBits for GpuComplex {}

impl From<C64> for GpuComplex {
    fn from(c: C64) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl From<GpuComplex> for C64 {
    fn from(c: GpuComplex) -> Self {
        C64::new(c.re, c.im)
    }
}

// ============================================================================
// Complex dense operator (cuBLAS zgemv)
// ============================================================================

/// Dense NxN complex matrix operator with GPU acceleration.
///
/// Stores the matrix in GPU memory and uses cuBLAS zgemv for matvec.
pub struct CudaComplexDenseOperator {
    n: usize,
    gpu_matrix: CudaSlice<GpuComplex>,
    cpu_matrix: Vec<C64>,
    ctx: Arc<CudaContext>,
    cpu_threshold: usize,
}

impl CudaComplexDenseOperator {
    /// Build from an existing matrix.
    pub fn from_matrix(ctx: Arc<CudaContext>, matrix: Vec<C64>, n: usize) -> Result<Self> {
        if matrix.len() != n * n {
            return Err(CudaError::InvalidDimension(format!(
                "Matrix length {} doesn't match n*n = {}",
                matrix.len(),
                n * n
            )));
        }

        let gpu_data: Vec<GpuComplex> = matrix.iter().map(|&c| c.into()).collect();

        let gpu_matrix = ctx
            .stream
            .memcpy_stod(&gpu_data)
            .map_err(|e| CudaError::Transfer(format!("Host to device copy failed: {}", e)))?;

        log::debug!("Uploaded {}x{} complex matrix to GPU ({} bytes)", n, n, n * n * 16);

        Ok(Self {
            n,
            gpu_matrix,
            cpu_matrix: matrix,
            ctx,
            cpu_threshold: 64,
        })
    }

    /// Set the threshold below which CPU fallback is used.
    pub fn with_cpu_threshold(mut self, threshold: usize) -> Self {
        self.cpu_threshold = threshold;
        self
    }

    /// Get the matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    fn apply_cpu(&self, x: &[C64], y: &mut [C64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let mut sum = C64::new(0.0, 0.0);
            let row_start = i * self.n;
            for (j, xj) in x.iter().enumerate().take(self.n) {
                sum += self.cpu_matrix[row_start + j] * xj;
            }
            *yi = sum;
        }
    }

    fn apply_gpu(&self, x: &[C64], y: &mut [C64]) -> Result<()> {
        let x_gpu: Vec<GpuComplex> = x.iter().map(|&c| c.into()).collect();

        let d_x = self
            .ctx
            .stream
            .memcpy_stod(&x_gpu)
            .map_err(|e| CudaError::Transfer(format!("x upload failed: {}", e)))?;

        let mut d_y: CudaSlice<GpuComplex> = self
            .ctx
            .stream
            .alloc_zeros(self.n)
            .map_err(|e| CudaError::MemoryAlloc(format!("y alloc failed: {}", e)))?;

        let alpha = cuDoubleComplex { x: 1.0, y: 0.0 };
        let beta = cuDoubleComplex { x: 0.0, y: 0.0 };

        let stream = &self.ctx.stream;
        let (a_ptr, _a_guard) = self.gpu_matrix.device_ptr(stream);
        let (x_ptr, _x_guard) = d_x.device_ptr(stream);
        let (y_ptr, _y_guard) = d_y.device_ptr_mut(stream);

        unsafe {
            let status = cublasZgemv_v2(
                *self.ctx.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                self.n as i32,
                self.n as i32,
                &alpha as *const _,
                a_ptr as *const cuDoubleComplex,
                self.n as i32,
                x_ptr as *const cuDoubleComplex,
                1,
                &beta as *const _,
                y_ptr as *mut cuDoubleComplex,
                1,
            );

            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(CudaError::Cublas(format!("zgemv failed: {:?}", status)));
            }
        }

        drop(_a_guard);
        drop(_x_guard);
        drop(_y_guard);

        stream
            .synchronize()
            .map_err(|e| CudaError::Transfer(format!("sync failed: {}", e)))?;

        let y_gpu: Vec<GpuComplex> = self
            .ctx
            .stream
            .memcpy_dtov(&d_y)
            .map_err(|e| CudaError::Transfer(format!("y download failed: {}", e)))?;

        for (i, c) in y_gpu.iter().enumerate() {
            y[i] = C64::from(*c);
        }

        Ok(())
    }
}

impl ComplexOperator for CudaComplexDenseOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[C64], y: &mut [C64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);

        if self.n <= self.cpu_threshold {
            self.apply_cpu(x, y);
            return;
        }

        if let Err(e) = self.apply_gpu(x, y) {
            log::warn!("GPU complex apply failed, falling back to CPU: {}", e);
            self.apply_cpu(x, y);
        }
    }
}

// ============================================================================
// Real dense operator (cuBLAS dgemv)
// ============================================================================

/// Dense NxN real matrix operator with GPU acceleration.
///
/// Stores the matrix in GPU memory and uses cuBLAS dgemv for matvec.
pub struct CudaRealDenseOperator {
    n: usize,
    gpu_matrix: CudaSlice<f64>,
    cpu_matrix: Vec<f64>,
    ctx: Arc<CudaContext>,
    cpu_threshold: usize,
}

impl CudaRealDenseOperator {
    /// Build from an existing matrix.
    pub fn from_matrix(ctx: Arc<CudaContext>, matrix: Vec<f64>, n: usize) -> Result<Self> {
        if matrix.len() != n * n {
            return Err(CudaError::InvalidDimension(format!(
                "Matrix length {} doesn't match n*n = {}",
                matrix.len(),
                n * n
            )));
        }

        let gpu_matrix = ctx
            .stream
            .memcpy_stod(&matrix)
            .map_err(|e| CudaError::Transfer(format!("Host to device copy failed: {}", e)))?;

        log::debug!("Uploaded {}x{} real matrix to GPU ({} bytes)", n, n, n * n * 8);

        Ok(Self {
            n,
            gpu_matrix,
            cpu_matrix: matrix,
            ctx,
            cpu_threshold: 64,
        })
    }

    /// Set the threshold below which CPU fallback is used.
    pub fn with_cpu_threshold(mut self, threshold: usize) -> Self {
        self.cpu_threshold = threshold;
        self
    }

    /// Get the matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    fn apply_cpu(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let mut sum = 0.0;
            let row_start = i * self.n;
            for (j, xj) in x.iter().enumerate().take(self.n) {
                sum += self.cpu_matrix[row_start + j] * xj;
            }
            *yi = sum;
        }
    }

    fn apply_gpu(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        let d_x = self
            .ctx
            .stream
            .memcpy_stod(x)
            .map_err(|e| CudaError::Transfer(format!("x upload failed: {}", e)))?;

        let mut d_y: CudaSlice<f64> = self
            .ctx
            .stream
            .alloc_zeros(self.n)
            .map_err(|e| CudaError::MemoryAlloc(format!("y alloc failed: {}", e)))?;

        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;

        let stream = &self.ctx.stream;
        let (a_ptr, _a_guard) = self.gpu_matrix.device_ptr(stream);
        let (x_ptr, _x_guard) = d_x.device_ptr(stream);
        let (y_ptr, _y_guard) = d_y.device_ptr_mut(stream);

        unsafe {
            let status = cublasDgemv_v2(
                *self.ctx.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                self.n as i32,
                self.n as i32,
                &alpha as *const _,
                a_ptr as *const f64,
                self.n as i32,
                x_ptr as *const f64,
                1,
                &beta as *const _,
                y_ptr as *mut f64,
                1,
            );

            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(CudaError::Cublas(format!("dgemv failed: {:?}", status)));
            }
        }

        drop(_a_guard);
        drop(_x_guard);
        drop(_y_guard);

        stream
            .synchronize()
            .map_err(|e| CudaError::Transfer(format!("sync failed: {}", e)))?;

        let y_gpu: Vec<f64> = self
            .ctx
            .stream
            .memcpy_dtov(&d_y)
            .map_err(|e| CudaError::Transfer(format!("y download failed: {}", e)))?;

        y.copy_from_slice(&y_gpu);

        Ok(())
    }
}

impl RealOperator for CudaRealDenseOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);

        if self.n <= self.cpu_threshold {
            self.apply_cpu(x, y);
            return;
        }

        if let Err(e) = self.apply_gpu(x, y) {
            log::warn!("GPU real apply failed, falling back to CPU: {}", e);
            self.apply_cpu(x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_cuda_context() -> Option<Arc<CudaContext>> {
        std::panic::catch_unwind(CudaContext::new)
            .ok()
            .and_then(|result| result.ok())
            .map(Arc::new)
    }

    #[test]
    fn test_complex_from_matrix_dimension_check() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let matrix = vec![C64::new(1.0, 0.0); 9];
        assert!(CudaComplexDenseOperator::from_matrix(ctx.clone(), matrix.clone(), 3).is_ok());
        assert!(CudaComplexDenseOperator::from_matrix(ctx, matrix, 4).is_err());
    }

    #[test]
    fn test_real_from_matrix_dimension_check() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let matrix = vec![1.0; 9];
        assert!(CudaRealDenseOperator::from_matrix(ctx.clone(), matrix.clone(), 3).is_ok());
        assert!(CudaRealDenseOperator::from_matrix(ctx, matrix, 4).is_err());
    }

    #[test]
    fn test_complex_cpu_fallback() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let matrix = vec![
            C64::new(1.0, 0.0), C64::new(1.0, 1.0), C64::new(1.0, 2.0),
            C64::new(2.0, 0.0), C64::new(2.0, 1.0), C64::new(2.0, 2.0),
            C64::new(3.0, 0.0), C64::new(3.0, 1.0), C64::new(3.0, 2.0),
        ];
        let op = CudaComplexDenseOperator::from_matrix(ctx, matrix, 3)
            .unwrap()
            .with_cpu_threshold(100);

        let x = vec![C64::new(1.0, 0.0); 3];
        let mut y = vec![C64::new(0.0, 0.0); 3];
        op.apply(&x, &mut y);

        assert!((y[0].re - 3.0).abs() < 1e-10);
        assert!((y[0].im - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_cpu_fallback() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let op = CudaRealDenseOperator::from_matrix(ctx, matrix, 3)
            .unwrap()
            .with_cpu_threshold(100);

        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 15.0).abs() < 1e-10);
        assert!((y[2] - 24.0).abs() < 1e-10);
    }
}
