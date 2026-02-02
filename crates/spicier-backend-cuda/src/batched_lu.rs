//! GPU-accelerated batched LU factorization and solve.
//!
//! Provides efficient batched solving for Monte Carlo, corner analysis, and parameter sweeps
//! using cuBLAS batched LU operations (`cublasDgetrfBatched`, `cublasDgetrsBatched`).
//!
//! # Memory Layout
//!
//! cuBLAS batched operations require:
//! - An array of device pointers, each pointing to an N×N matrix in column-major order
//! - Contiguous storage for all matrices and vectors
//!
//! # Usage
//!
//! ```ignore
//! use spicier_backend_cuda::{CudaContext, CudaBatchedLuSolver};
//!
//! let ctx = CudaContext::new()?;
//! let solver = CudaBatchedLuSolver::new(&ctx);
//!
//! // matrices: batch_size matrices, each n×n in column-major order
//! // rhs: batch_size vectors, each of length n
//! let solutions = solver.solve_batch(&matrices, &rhs, n, batch_size)?;
//! ```

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use cudarc::cublas::sys::{
    cublasDgetrfBatched, cublasDgetrsBatched, cublasOperation_t, cublasStatus_t,
};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::sync::Arc;

/// Maximum batch size supported by cuBLAS batched operations.
pub const MAX_BATCH_SIZE: usize = 65535;

/// Minimum batch size for GPU to be worthwhile (kernel launch overhead).
pub const MIN_BATCH_SIZE: usize = 16;

/// Minimum matrix size for GPU to be worthwhile.
pub const MIN_MATRIX_SIZE: usize = 32;

/// Configuration for GPU batched sweep operations.
#[derive(Debug, Clone)]
pub struct GpuBatchedSweepConfig {
    /// Minimum batch size to use GPU (below this, CPU is used).
    pub min_batch_size: usize,
    /// Minimum matrix dimension to use GPU.
    pub min_matrix_size: usize,
    /// Maximum batch size per GPU launch (cuBLAS limit).
    pub max_batch_per_launch: usize,
}

impl Default for GpuBatchedSweepConfig {
    fn default() -> Self {
        Self {
            min_batch_size: MIN_BATCH_SIZE,
            min_matrix_size: MIN_MATRIX_SIZE,
            max_batch_per_launch: MAX_BATCH_SIZE,
        }
    }
}

impl GpuBatchedSweepConfig {
    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        matrix_size >= self.min_matrix_size && batch_size >= self.min_batch_size
    }
}

/// Contiguous GPU storage for N matrices.
///
/// Stores all matrices in a single contiguous buffer, plus an array of device
/// pointers (one per matrix) as required by cuBLAS batched APIs.
pub struct BatchedMatrices {
    /// Contiguous storage for all matrices (batch_size * n * n elements).
    #[allow(dead_code)]
    data: CudaSlice<f64>,
    /// Array of device pointers, each pointing to start of a matrix.
    pointers: CudaSlice<u64>,
    /// Matrix dimension (each matrix is n × n).
    n: usize,
    /// Number of matrices in the batch.
    batch_size: usize,
}

impl BatchedMatrices {
    /// Create batched matrices from host data.
    ///
    /// # Arguments
    /// * `ctx` - CUDA context
    /// * `matrices` - Flattened matrices in column-major order (batch_size * n * n elements)
    /// * `n` - Matrix dimension (each matrix is n × n)
    /// * `batch_size` - Number of matrices
    ///
    /// # Errors
    /// Returns error if dimensions don't match or memory allocation fails.
    pub fn from_host(
        ctx: &CudaContext,
        matrices: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let expected_len = batch_size * n * n;
        if matrices.len() != expected_len {
            return Err(CudaError::InvalidDimension(format!(
                "Expected {} elements ({} batches × {}×{}), got {}",
                expected_len,
                batch_size,
                n,
                n,
                matrices.len()
            )));
        }

        if batch_size > MAX_BATCH_SIZE {
            return Err(CudaError::BatchTooLarge {
                size: batch_size,
                max: MAX_BATCH_SIZE,
            });
        }

        // Upload matrix data to GPU
        let data = ctx
            .stream
            .memcpy_stod(matrices)
            .map_err(|e| CudaError::Transfer(format!("Matrix upload failed: {}", e)))?;

        // Get base device pointer
        let (base_ptr, _guard) = data.device_ptr(&ctx.stream);
        let base_addr = base_ptr;

        // Build array of pointers to each matrix
        let matrix_stride = (n * n) as u64 * std::mem::size_of::<f64>() as u64;
        let pointer_array: Vec<u64> = (0..batch_size)
            .map(|i| base_addr + (i as u64) * matrix_stride)
            .collect();

        // Upload pointer array to GPU
        let pointers = ctx
            .stream
            .memcpy_stod(&pointer_array)
            .map_err(|e| CudaError::Transfer(format!("Pointer array upload failed: {}", e)))?;

        drop(_guard);

        log::debug!(
            "Uploaded {} matrices ({}×{}) to GPU ({} bytes)",
            batch_size,
            n,
            n,
            expected_len * 8
        );

        Ok(Self {
            data,
            pointers,
            n,
            batch_size,
        })
    }

    /// Get the matrix dimension.
    pub fn matrix_size(&self) -> usize {
        self.n
    }

    /// Get the number of matrices.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Contiguous GPU storage for N RHS vectors.
pub struct BatchedVectors {
    /// Contiguous storage for all vectors (batch_size * n elements).
    data: CudaSlice<f64>,
    /// Array of device pointers, each pointing to start of a vector.
    pointers: CudaSlice<u64>,
    /// Vector length.
    #[allow(dead_code)]
    n: usize,
    /// Number of vectors in the batch.
    #[allow(dead_code)]
    batch_size: usize,
}

impl BatchedVectors {
    /// Create batched vectors from host data.
    pub fn from_host(
        ctx: &CudaContext,
        vectors: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let expected_len = batch_size * n;
        if vectors.len() != expected_len {
            return Err(CudaError::InvalidDimension(format!(
                "Expected {} elements ({} batches × {}), got {}",
                expected_len,
                batch_size,
                n,
                vectors.len()
            )));
        }

        // Upload vector data to GPU
        let data = ctx
            .stream
            .memcpy_stod(vectors)
            .map_err(|e| CudaError::Transfer(format!("Vector upload failed: {}", e)))?;

        // Get base device pointer
        let (base_ptr, _guard) = data.device_ptr(&ctx.stream);
        let base_addr = base_ptr;

        // Build array of pointers to each vector
        let vector_stride = n as u64 * std::mem::size_of::<f64>() as u64;
        let pointer_array: Vec<u64> = (0..batch_size)
            .map(|i| base_addr + (i as u64) * vector_stride)
            .collect();

        // Upload pointer array to GPU
        let pointers = ctx
            .stream
            .memcpy_stod(&pointer_array)
            .map_err(|e| CudaError::Transfer(format!("Pointer array upload failed: {}", e)))?;

        drop(_guard);

        Ok(Self {
            data,
            pointers,
            n,
            batch_size,
        })
    }

    /// Download solutions back to host.
    pub fn to_host(&self, ctx: &CudaContext) -> Result<Vec<f64>> {
        ctx.stream
            .memcpy_dtov(&self.data)
            .map_err(|e| CudaError::Transfer(format!("Solution download failed: {}", e)))
    }
}

/// GPU storage for pivot indices and info array.
pub struct BatchedPivots {
    /// Pivot indices: batch_size * n integers.
    pivots: CudaSlice<i32>,
    /// Info array: one integer per matrix (0 = success, >0 = singular).
    info: CudaSlice<i32>,
    /// Matrix dimension.
    #[allow(dead_code)]
    n: usize,
    /// Number of matrices.
    #[allow(dead_code)]
    batch_size: usize,
}

impl BatchedPivots {
    /// Allocate pivot and info storage.
    pub fn allocate(ctx: &CudaContext, n: usize, batch_size: usize) -> Result<Self> {
        let pivots: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(batch_size * n)
            .map_err(|e| CudaError::MemoryAlloc(format!("Pivot allocation failed: {}", e)))?;

        let info: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(batch_size)
            .map_err(|e| CudaError::MemoryAlloc(format!("Info allocation failed: {}", e)))?;

        Ok(Self {
            pivots,
            info,
            n,
            batch_size,
        })
    }

    /// Download info array to check for singular matrices.
    pub fn check_singularity(&self, ctx: &CudaContext) -> Result<Vec<usize>> {
        let info_host: Vec<i32> = ctx
            .stream
            .memcpy_dtov(&self.info)
            .map_err(|e| CudaError::Transfer(format!("Info download failed: {}", e)))?;

        let singular_indices: Vec<usize> = info_host
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v > 0 { Some(i) } else { None })
            .collect();

        Ok(singular_indices)
    }
}

/// Result of a batched LU solve operation.
#[derive(Debug)]
pub struct BatchedSolveResult {
    /// Solutions for each system (flattened: batch_size * n elements).
    pub solutions: Vec<f64>,
    /// Indices of matrices that were singular.
    pub singular_indices: Vec<usize>,
    /// Matrix dimension.
    pub n: usize,
    /// Number of systems solved.
    pub batch_size: usize,
}

impl BatchedSolveResult {
    /// Get the solution for a specific system.
    pub fn solution(&self, index: usize) -> Option<&[f64]> {
        if index >= self.batch_size {
            return None;
        }
        let start = index * self.n;
        let end = start + self.n;
        Some(&self.solutions[start..end])
    }

    /// Check if a specific system was singular.
    pub fn is_singular(&self, index: usize) -> bool {
        self.singular_indices.contains(&index)
    }

    /// Number of successfully solved systems.
    pub fn num_solved(&self) -> usize {
        self.batch_size - self.singular_indices.len()
    }
}

/// GPU-accelerated batched LU solver using cuBLAS.
pub struct CudaBatchedLuSolver {
    ctx: Arc<CudaContext>,
    config: GpuBatchedSweepConfig,
}

impl CudaBatchedLuSolver {
    /// Create a new batched LU solver.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            config: GpuBatchedSweepConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(ctx: Arc<CudaContext>, config: GpuBatchedSweepConfig) -> Self {
        Self { ctx, config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &GpuBatchedSweepConfig {
        &self.config
    }

    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        self.config.should_use_gpu(matrix_size, batch_size)
    }

    /// Solve a batch of linear systems Ax = b.
    ///
    /// # Arguments
    /// * `matrices` - Flattened matrices in column-major order (batch_size * n * n)
    /// * `rhs` - Flattened RHS vectors (batch_size * n)
    /// * `n` - Matrix/vector dimension
    /// * `batch_size` - Number of systems to solve
    ///
    /// # Returns
    /// Solutions and information about any singular systems.
    pub fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        if batch_size == 0 {
            return Ok(BatchedSolveResult {
                solutions: vec![],
                singular_indices: vec![],
                n,
                batch_size: 0,
            });
        }

        if batch_size > self.config.max_batch_per_launch {
            return Err(CudaError::BatchTooLarge {
                size: batch_size,
                max: self.config.max_batch_per_launch,
            });
        }

        // Upload matrices and RHS to GPU
        let gpu_matrices = BatchedMatrices::from_host(&self.ctx, matrices, n, batch_size)?;
        let gpu_rhs = BatchedVectors::from_host(&self.ctx, rhs, n, batch_size)?;
        let mut pivots = BatchedPivots::allocate(&self.ctx, n, batch_size)?;

        let n_i32 = n as i32;
        let batch_size_i32 = batch_size as i32;
        let stream = &self.ctx.stream;

        // Step 1: LU factorization
        {
            let (a_ptrs, _a_guard) = gpu_matrices.pointers.device_ptr(stream);
            let (pivot_ptr, _pivot_guard) = pivots.pivots.device_ptr_mut(stream);
            let (info_ptr, _info_guard) = pivots.info.device_ptr_mut(stream);

            let status = unsafe {
                cublasDgetrfBatched(
                    *self.ctx.blas.handle(),
                    n_i32,
                    a_ptrs as *const *mut f64,
                    n_i32,
                    pivot_ptr as *mut i32,
                    info_ptr as *mut i32,
                    batch_size_i32,
                )
            };

            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(CudaError::Cublas(format!(
                    "cublasDgetrfBatched failed: {:?}",
                    status
                )));
            }
        }

        // Synchronize before checking singularity
        stream
            .synchronize()
            .map_err(|e| CudaError::Transfer(format!("sync failed: {}", e)))?;

        let singular_indices = pivots.check_singularity(&self.ctx)?;
        if !singular_indices.is_empty() {
            log::warn!(
                "{} of {} matrices were singular",
                singular_indices.len(),
                batch_size
            );
        }

        // Step 2: Solve with factored matrices
        {
            let (a_ptrs, _a_guard) = gpu_matrices.pointers.device_ptr(stream);
            let (pivot_ptr, _pivot_guard) = pivots.pivots.device_ptr(stream);
            let (b_ptrs, _b_guard) = gpu_rhs.pointers.device_ptr(stream);

            let mut getrs_info: CudaSlice<i32> = self.ctx.stream.alloc_zeros(1).map_err(|e| {
                CudaError::MemoryAlloc(format!("getrs info allocation failed: {}", e))
            })?;
            let (getrs_info_ptr, _getrs_info_guard) = getrs_info.device_ptr_mut(stream);

            let status = unsafe {
                cublasDgetrsBatched(
                    *self.ctx.blas.handle(),
                    cublasOperation_t::CUBLAS_OP_N,
                    n_i32,
                    1,
                    a_ptrs as *const *const f64,
                    n_i32,
                    pivot_ptr as *const i32,
                    b_ptrs as *const *mut f64,
                    n_i32,
                    getrs_info_ptr as *mut i32,
                    batch_size_i32,
                )
            };

            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(CudaError::Cublas(format!(
                    "cublasDgetrsBatched failed: {:?}",
                    status
                )));
            }
        }

        // Synchronize and download solutions
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| CudaError::Transfer(format!("Synchronization failed: {}", e)))?;

        let solutions = gpu_rhs.to_host(&self.ctx)?;

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
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
    fn test_batched_matrices_layout() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let n = 2;
        let batch_size = 3;
        // 3 matrices, each 2×2, column-major:
        // Matrix 0: [[1,3],[2,4]]
        // Matrix 1: [[5,7],[6,8]]
        // Matrix 2: [[9,11],[10,12]]
        let matrices = vec![
            1.0, 2.0, 3.0, 4.0, // Matrix 0
            5.0, 6.0, 7.0, 8.0, // Matrix 1
            9.0, 10.0, 11.0, 12.0, // Matrix 2
        ];

        let batched = BatchedMatrices::from_host(&ctx, &matrices, n, batch_size);
        assert!(batched.is_ok());

        let batched = batched.unwrap();
        assert_eq!(batched.matrix_size(), 2);
        assert_eq!(batched.batch_size(), 3);
    }

    #[test]
    fn test_batched_lu_identity() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let solver = CudaBatchedLuSolver::new(ctx);
        let n = 2;
        let batch_size = 2;

        // Two 2×2 identity matrices in column-major order
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];

        // RHS vectors
        let rhs = vec![
            1.0, 2.0, // b0 = [1, 2]
            3.0, 4.0, // b1 = [3, 4]
        ];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        // For identity matrices, solution should equal RHS
        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-10);
        assert!((sol0[1] - 2.0).abs() < 1e-10);

        let sol1 = result.solution(1).unwrap();
        assert!((sol1[0] - 3.0).abs() < 1e-10);
        assert!((sol1[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_batched_lu_simple() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let solver = CudaBatchedLuSolver::new(ctx);
        let n = 2;
        let batch_size = 1;

        // Matrix: [[2, 1], [1, 3]] in column-major: [2, 1, 1, 3]
        // Solving Ax = b where b = [5, 5]
        // Solution should be x = [2, 1]
        let matrices = vec![2.0, 1.0, 1.0, 3.0];
        let rhs = vec![5.0, 5.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.singular_indices.is_empty());

        let sol = result.solution(0).unwrap();
        assert!(
            (sol[0] - 2.0).abs() < 1e-9,
            "x[0] = {} (expected 2.0)",
            sol[0]
        );
        assert!(
            (sol[1] - 1.0).abs() < 1e-9,
            "x[1] = {} (expected 1.0)",
            sol[1]
        );
    }

    #[test]
    fn test_batched_lu_singular_detection() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let solver = CudaBatchedLuSolver::new(ctx);
        let n = 2;
        let batch_size = 2;

        // Matrix 0: identity (non-singular)
        // Matrix 1: [[1, 2], [1, 2]] (singular - rows are identical)
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity
            1.0, 1.0, 2.0, 2.0, // Singular
        ];
        let rhs = vec![1.0, 2.0, 1.0, 2.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size);
        assert!(result.is_ok());

        let result = result.unwrap();
        // The singular matrix should be detected
        assert!(
            result.is_singular(1),
            "Matrix 1 should be detected as singular"
        );
        assert!(!result.is_singular(0), "Matrix 0 should not be singular");
    }

    #[test]
    fn test_config_thresholds() {
        let config = GpuBatchedSweepConfig::default();

        // Below both thresholds
        assert!(!config.should_use_gpu(16, 8));

        // Matrix size OK, batch too small
        assert!(!config.should_use_gpu(64, 8));

        // Batch OK, matrix too small
        assert!(!config.should_use_gpu(16, 32));

        // Both OK
        assert!(config.should_use_gpu(64, 32));
    }

    #[test]
    fn test_batch_too_large_error() {
        let ctx = match try_create_cuda_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Try to create matrices with batch size exceeding limit
        let matrices = vec![1.0; 4]; // Dummy data
        let result = BatchedMatrices::from_host(&ctx, &matrices, 2, MAX_BATCH_SIZE + 1);
        assert!(matches!(result, Err(CudaError::BatchTooLarge { .. })));
    }
}
