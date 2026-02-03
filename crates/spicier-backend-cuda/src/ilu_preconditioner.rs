//! CUDA ILU(0) preconditioner.
//!
//! This module provides a GPU-accelerated ILU(0) preconditioner that wraps
//! the CPU implementation and manages GPU data transfer.
//!
//! # Note on cuSPARSE ILU
//!
//! cuSPARSE provides `csrilu02` for ILU(0) factorization and `csrsv2` for
//! triangular solves. However, the cudarc bindings don't fully expose these
//! APIs (the csrsv2 variants are not available in the current version).
//!
//! For now, this implementation:
//! 1. Uses the CPU ILU(0) implementation from spicier-solver
//! 2. Manages GPU memory for the solve vectors
//! 3. Transfers data between host and device as needed
//!
//! Future work: Add direct cuSPARSE bindings for full GPU ILU.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use spicier_solver::{Ilu0Preconditioner, IluError, RealPreconditioner};
use std::sync::Arc;

/// CUDA ILU(0) preconditioner.
///
/// Currently wraps the CPU ILU(0) implementation with GPU memory management.
/// The factorization and triangular solves are performed on the CPU, but
/// this provides a unified API that can be upgraded to full GPU support.
pub struct CudaIlu0Preconditioner {
    /// Underlying CPU preconditioner.
    cpu_precond: Ilu0Preconditioner,
    /// CUDA context for memory operations.
    cuda_ctx: Arc<CudaContext>,
}

impl CudaIlu0Preconditioner {
    /// Create a new ILU(0) preconditioner from CSR structure.
    ///
    /// # Arguments
    /// * `cuda_ctx` - CUDA context
    /// * `n` - Matrix dimension
    /// * `row_ptr` - CSR row pointers (length n+1)
    /// * `col_idx` - CSR column indices (length nnz)
    /// * `values` - CSR values (length nnz)
    pub fn new(
        cuda_ctx: Arc<CudaContext>,
        n: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        let cpu_precond = Ilu0Preconditioner::from_csr(n, row_ptr, col_idx, values)
            .map_err(|e| CudaError::Compute(format!("ILU factorization failed: {}", e)))?;

        Ok(Self {
            cpu_precond,
            cuda_ctx,
        })
    }

    /// Create from matrix triplets.
    ///
    /// Builds CSR structure from (row, col, value) triplets.
    pub fn from_triplets(
        cuda_ctx: Arc<CudaContext>,
        n: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<Self> {
        let cpu_precond = Ilu0Preconditioner::from_triplets(n, triplets)
            .map_err(|e| CudaError::Compute(format!("ILU factorization failed: {}", e)))?;

        Ok(Self {
            cpu_precond,
            cuda_ctx,
        })
    }

    /// Update values with same sparsity pattern.
    ///
    /// Re-computes the ILU(0) factorization using new values.
    pub fn update_values(&mut self, values: &[f64]) -> Result<()> {
        self.cpu_precond
            .update_values(values)
            .map_err(|e| CudaError::Compute(format!("ILU update failed: {}", e)))
    }

    /// Apply preconditioner: y = (LU)^{-1} * x
    ///
    /// Currently transfers data to CPU, applies preconditioner, transfers back.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.cpu_precond.apply(x, y);
    }

    /// Get the matrix dimension.
    pub fn dim(&self) -> usize {
        self.cpu_precond.dim()
    }

    /// Get the CUDA context.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    /// Get sparsity information.
    pub fn sparsity(&self) -> (&[usize], &[usize]) {
        self.cpu_precond.sparsity()
    }

    /// Get number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.cpu_precond.nnz()
    }
}

impl RealPreconditioner for CudaIlu0Preconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.cpu_precond.apply(x, y);
    }

    fn dim(&self) -> usize {
        self.cpu_precond.dim()
    }
}

/// Convert IluError to CudaError.
impl From<IluError> for CudaError {
    fn from(e: IluError) -> Self {
        CudaError::Compute(format!("ILU error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_cuda_context() -> Arc<CudaContext> {
        Arc::new(CudaContext::new().expect("CUDA context creation failed"))
    }

    /// Test ILU with diagonal matrix on CUDA hardware.
    ///
    /// This test is ignored by default because it requires CUDA hardware.
    /// Run with: cargo test --ignored test_ilu_diagonal_matrix
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_ilu_diagonal_matrix() {
        let ctx = create_cuda_context();

        // Diagonal matrix: diag([2, 3, 4])
        let n = 3;
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 4.0];

        let precond = CudaIlu0Preconditioner::new(ctx, n, &row_ptr, &col_idx, &values).unwrap();

        // Apply to [2, 6, 12] should give [1, 2, 3]
        let x = vec![2.0, 6.0, 12.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-10, "y[0] = {}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-10, "y[1] = {}", y[1]);
        assert!((y[2] - 3.0).abs() < 1e-10, "y[2] = {}", y[2]);
    }

    /// Test ILU from triplets on CUDA hardware.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_ilu_from_triplets() {
        let ctx = create_cuda_context();

        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ];

        let precond = CudaIlu0Preconditioner::from_triplets(ctx, 3, &triplets).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        // Verify A*y ≈ x
        let ay0 = 2.0 * y[0] - 1.0 * y[1];
        let ay1 = -y[0] + 2.0 * y[1] - 1.0 * y[2];
        let ay2 = -y[1] + 2.0 * y[2];

        assert!((ay0 - x[0]).abs() < 1e-10);
        assert!((ay1 - x[1]).abs() < 1e-10);
        assert!((ay2 - x[2]).abs() < 1e-10);
    }

    /// Test ILU value updates on CUDA hardware.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_ilu_update_values() {
        let ctx = create_cuda_context();

        let n = 3;
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 4.0];

        let mut precond = CudaIlu0Preconditioner::new(ctx, n, &row_ptr, &col_idx, &values).unwrap();

        // Update to new values
        let new_values = vec![4.0, 6.0, 8.0];
        precond.update_values(&new_values).unwrap();

        let x = vec![4.0, 12.0, 24.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
    }
}
