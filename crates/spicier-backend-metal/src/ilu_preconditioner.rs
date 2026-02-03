//! GPU ILU(0) preconditioner for Metal/WebGPU backend.
//!
//! This module provides an ILU(0) preconditioner that integrates with the
//! Metal batched GMRES solver.
//!
//! # Implementation Notes
//!
//! ILU(0) factorization and triangular solves are inherently sequential within
//! each level of the dependency graph. While level-scheduled parallelism is
//! possible, the benefits are limited for typical MNA matrices.
//!
//! For batched sweeps (multiple RHS), we extract a shared ILU factor from the
//! baseline matrix structure and use it across all sweep points. This provides
//! a good approximation when sweep variations are small.
//!
//! The current implementation uses the CPU ILU(0) from spicier-solver for
//! the factorization and apply phases, with GPU buffer management for
//! integration with batched GMRES.

use crate::batched_spmv::BatchedCsrMatrix;
use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use spicier_solver::{Ilu0Preconditioner, IluError, RealPreconditioner};
use std::sync::Arc;

/// GPU ILU(0) preconditioner for Metal/WebGPU.
///
/// Wraps the CPU ILU(0) implementation and provides GPU buffer management
/// for integration with batched GMRES solvers.
pub struct GpuIlu0Preconditioner {
    /// Underlying CPU preconditioner.
    cpu_precond: Ilu0Preconditioner,
    /// WebGPU context.
    ctx: Arc<WgpuContext>,
    /// Matrix dimension.
    n: usize,
}

impl GpuIlu0Preconditioner {
    /// Create a new ILU(0) preconditioner from CSR structure.
    ///
    /// # Arguments
    /// * `ctx` - WebGPU context
    /// * `n` - Matrix dimension
    /// * `row_ptr` - CSR row pointers (length n+1)
    /// * `col_idx` - CSR column indices (length nnz)
    /// * `values` - CSR values (length nnz)
    pub fn new(
        ctx: Arc<WgpuContext>,
        n: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        let cpu_precond = Ilu0Preconditioner::from_csr(n, row_ptr, col_idx, values)
            .map_err(|e| WgpuError::Compute(format!("ILU factorization failed: {}", e)))?;

        Ok(Self {
            cpu_precond,
            ctx,
            n,
        })
    }

    /// Create from matrix triplets.
    pub fn from_triplets(
        ctx: Arc<WgpuContext>,
        n: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<Self> {
        let cpu_precond = Ilu0Preconditioner::from_triplets(n, triplets)
            .map_err(|e| WgpuError::Compute(format!("ILU factorization failed: {}", e)))?;

        Ok(Self {
            cpu_precond,
            ctx,
            n,
        })
    }

    /// Create from a batched CSR matrix structure using the first sweep point.
    ///
    /// Extracts the ILU factorization from sweep point 0 and uses it as a
    /// preconditioner for all sweep points. This works well when the sweep
    /// variations are small relative to the baseline matrix.
    pub fn from_batched_csr(
        ctx: Arc<WgpuContext>,
        structure: &BatchedCsrMatrix,
        values: &[f32],
        _num_sweeps: usize,
    ) -> Result<Self> {
        let n = structure.n;
        let nnz = structure.col_idx.len();

        // Convert row_ptr and col_idx to usize
        let row_ptr: Vec<usize> = structure.row_ptr.iter().map(|&x| x as usize).collect();
        let col_idx: Vec<usize> = structure.col_idx.iter().map(|&x| x as usize).collect();

        // Extract values from first sweep point, converting f32 to f64
        let values_f64: Vec<f64> = values[0..nnz].iter().map(|&x| x as f64).collect();

        Self::new(ctx, n, &row_ptr, &col_idx, &values_f64)
    }

    /// Update values with same sparsity pattern.
    pub fn update_values(&mut self, values: &[f64]) -> Result<()> {
        self.cpu_precond
            .update_values(values)
            .map_err(|e| WgpuError::Compute(format!("ILU update failed: {}", e)))
    }

    /// Apply preconditioner: y = (LU)^{-1} * x
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.cpu_precond.apply(x, y);
    }

    /// Apply preconditioner to f32 vectors (for GPU integration).
    ///
    /// Converts f32 to f64, applies ILU, converts back to f32.
    pub fn apply_f32(&self, x: &[f32], y: &mut [f32]) {
        let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let mut y_f64 = vec![0.0f64; y.len()];

        self.cpu_precond.apply(&x_f64, &mut y_f64);

        for (i, &val) in y_f64.iter().enumerate() {
            y[i] = val as f32;
        }
    }

    /// Apply preconditioner to batched vectors.
    ///
    /// Applies the same ILU preconditioner to multiple RHS vectors,
    /// one per sweep point.
    pub fn apply_batched(&self, x: &[f32], y: &mut [f32], num_sweeps: usize) {
        for sweep in 0..num_sweeps {
            let base = sweep * self.n;
            let x_slice = &x[base..base + self.n];
            let y_slice = &mut y[base..base + self.n];
            self.apply_f32(x_slice, y_slice);
        }
    }

    /// Get the matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Get the WebGPU context.
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.ctx
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

impl RealPreconditioner for GpuIlu0Preconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.cpu_precond.apply(x, y);
    }

    fn dim(&self) -> usize {
        self.n
    }
}

/// Convert IluError to WgpuError.
impl From<IluError> for WgpuError {
    fn from(e: IluError) -> Self {
        WgpuError::Compute(format!("ILU error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_context() -> Result<Arc<WgpuContext>> {
        Ok(Arc::new(WgpuContext::new()?))
    }

    #[test]
    fn test_ilu_diagonal_matrix() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        // Diagonal matrix: diag([2, 3, 4])
        let n = 3;
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 4.0];

        let precond = GpuIlu0Preconditioner::new(ctx, n, &row_ptr, &col_idx, &values).unwrap();

        // Apply to [2, 6, 12] should give [1, 2, 3]
        let x = vec![2.0, 6.0, 12.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-10, "y[0] = {}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-10, "y[1] = {}", y[1]);
        assert!((y[2] - 3.0).abs() < 1e-10, "y[2] = {}", y[2]);
    }

    #[test]
    fn test_ilu_f32_apply() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let n = 3;
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 4.0];

        let precond = GpuIlu0Preconditioner::new(ctx, n, &row_ptr, &col_idx, &values).unwrap();

        let x = vec![2.0f32, 6.0f32, 12.0f32];
        let mut y = vec![0.0f32; 3];
        precond.apply_f32(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-5, "y[0] = {}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-5, "y[1] = {}", y[1]);
        assert!((y[2] - 3.0).abs() < 1e-5, "y[2] = {}", y[2]);
    }

    #[test]
    fn test_ilu_batched_apply() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let n = 2;
        let row_ptr = vec![0, 1, 2];
        let col_idx = vec![0, 1];
        let values = vec![2.0, 4.0]; // diag([2, 4])

        let precond = GpuIlu0Preconditioner::new(ctx, n, &row_ptr, &col_idx, &values).unwrap();

        // Two sweep points
        // Sweep 0: x = [4, 8], expected y = [2, 2]
        // Sweep 1: x = [6, 12], expected y = [3, 3]
        let x = vec![4.0f32, 8.0f32, 6.0f32, 12.0f32];
        let mut y = vec![0.0f32; 4];
        precond.apply_batched(&x, &mut y, 2);

        assert!((y[0] - 2.0).abs() < 1e-5);
        assert!((y[1] - 2.0).abs() < 1e-5);
        assert!((y[2] - 3.0).abs() < 1e-5);
        assert!((y[3] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_ilu_from_triplets() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ];

        let precond = GpuIlu0Preconditioner::from_triplets(ctx, 3, &triplets).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        // Verify A*y ≈ x (ILU(0) is exact for tridiagonal)
        let ay0 = 2.0 * y[0] - 1.0 * y[1];
        let ay1 = -y[0] + 2.0 * y[1] - 1.0 * y[2];
        let ay2 = -y[1] + 2.0 * y[2];

        assert!((ay0 - x[0]).abs() < 1e-10);
        assert!((ay1 - x[1]).abs() < 1e-10);
        assert!((ay2 - x[2]).abs() < 1e-10);
    }
}
