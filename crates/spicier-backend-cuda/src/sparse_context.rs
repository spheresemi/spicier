//! CUDA sparse context for cuSPARSE and cuSOLVER sparse operations.
//!
//! Provides handle management for GPU sparse linear algebra operations,
//! including ILU(0) factorization and sparse triangular solves.

use crate::context::CudaContext;
use crate::error::{CudaError, Result};
use std::sync::Arc;

/// CUDA sparse context holding cuSPARSE and cuSOLVER sparse handles.
///
/// This context manages the GPU resources needed for sparse operations:
/// - cuSPARSE handle for sparse matrix operations (SpMV, ILU, triangular solve)
/// - cuSOLVER sparse handle for direct sparse solvers
pub struct CudaSparseContext {
    /// Base CUDA context.
    cuda_ctx: Arc<CudaContext>,
    /// cuSPARSE handle.
    cusparse_handle: cudarc::cusparse::sys::cusparseHandle_t,
    /// cuSOLVER sparse handle.
    cusolver_sp_handle: cudarc::cusolver::sys::cusolverSpHandle_t,
}

impl CudaSparseContext {
    /// Create a new sparse context from an existing CUDA context.
    pub fn new(cuda_ctx: Arc<CudaContext>) -> Result<Self> {
        // Create cuSPARSE handle
        let cusparse_handle = cudarc::cusparse::result::create()
            .map_err(|e| CudaError::DeviceInit(format!("cuSPARSE init failed: {:?}", e)))?;

        // Create cuSOLVER sparse handle
        let cusolver_sp_handle = create_cusolver_sp_handle()
            .map_err(|e| CudaError::DeviceInit(format!("cuSOLVER sparse init failed: {}", e)))?;

        log::info!("CUDA sparse context initialized");

        Ok(Self {
            cuda_ctx,
            cusparse_handle,
            cusolver_sp_handle,
        })
    }

    /// Get the underlying CUDA context.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    /// Get the cuSPARSE handle for sparse operations.
    pub fn cusparse_handle(&self) -> cudarc::cusparse::sys::cusparseHandle_t {
        self.cusparse_handle
    }

    /// Get the cuSOLVER sparse handle for sparse direct solvers.
    pub fn cusolver_sp_handle(&self) -> cudarc::cusolver::sys::cusolverSpHandle_t {
        self.cusolver_sp_handle
    }
}

impl Drop for CudaSparseContext {
    fn drop(&mut self) {
        // Destroy cuSPARSE handle
        unsafe {
            let _ = cudarc::cusparse::result::destroy(self.cusparse_handle);
        }
        // Destroy cuSOLVER sparse handle
        unsafe {
            let _ = destroy_cusolver_sp_handle(self.cusolver_sp_handle);
        }
    }
}

// cuSOLVER sparse handle creation/destruction helpers
// These are needed because cudarc doesn't expose these in the safe result module

fn create_cusolver_sp_handle() -> std::result::Result<cudarc::cusolver::sys::cusolverSpHandle_t, String> {
    use std::mem::MaybeUninit;
    let mut handle = MaybeUninit::uninit();
    let status = unsafe {
        cudarc::cusolver::sys::cusolverSpCreate(handle.as_mut_ptr())
    };
    if status != cudarc::cusolver::sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
        return Err(format!("cusolverSpCreate failed: {:?}", status));
    }
    Ok(unsafe { handle.assume_init() })
}

unsafe fn destroy_cusolver_sp_handle(
    handle: cudarc::cusolver::sys::cusolverSpHandle_t,
) -> std::result::Result<(), String> {
    let status = unsafe { cudarc::cusolver::sys::cusolverSpDestroy(handle) };
    if status != cudarc::cusolver::sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
        return Err(format!("cusolverSpDestroy failed: {:?}", status));
    }
    Ok(())
}

/// CSR (Compressed Sparse Row) matrix descriptor for GPU operations.
///
/// Represents a sparse matrix in CSR format ready for GPU operations.
/// The structure is stored on CPU; the values can be updated for
/// multiple solves with the same sparsity pattern.
#[derive(Clone)]
pub struct CsrMatrixDescriptor {
    /// Number of rows (and columns, for square matrices).
    pub n: usize,
    /// Number of non-zeros.
    pub nnz: usize,
    /// Row pointers (length n+1).
    pub row_ptr: Vec<i32>,
    /// Column indices (length nnz).
    pub col_idx: Vec<i32>,
}

impl CsrMatrixDescriptor {
    /// Create a new CSR matrix descriptor.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension (n×n)
    /// * `row_ptr` - Row pointers as usize (will be converted to i32)
    /// * `col_idx` - Column indices as usize (will be converted to i32)
    pub fn new(n: usize, row_ptr: &[usize], col_idx: &[usize]) -> Self {
        let row_ptr_i32: Vec<i32> = row_ptr.iter().map(|&x| x as i32).collect();
        let col_idx_i32: Vec<i32> = col_idx.iter().map(|&x| x as i32).collect();
        let nnz = col_idx.len();

        Self {
            n,
            nnz,
            row_ptr: row_ptr_i32,
            col_idx: col_idx_i32,
        }
    }

    /// Create from triplets.
    ///
    /// Builds CSR structure from (row, col, value) triplets.
    /// Duplicate entries at the same position are summed.
    pub fn from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> (Self, Vec<f64>) {
        use std::collections::BTreeMap;

        // Aggregate by (row, col) and sum duplicates
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for &(row, col, val) in triplets {
            *entries.entry((row, col)).or_insert(0.0) += val;
        }

        // Build CSR
        let mut row_ptr = vec![0i32; n + 1];
        let mut col_idx = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        let mut current_row = 0;
        for (&(row, col), &val) in &entries {
            while current_row <= row {
                row_ptr[current_row] = col_idx.len() as i32;
                current_row += 1;
            }
            col_idx.push(col as i32);
            values.push(val);
        }
        while current_row <= n {
            row_ptr[current_row] = col_idx.len() as i32;
            current_row += 1;
        }

        let nnz = values.len();

        (
            Self {
                n,
                nnz,
                row_ptr,
                col_idx,
            },
            values,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require CUDA hardware and will skip if unavailable

    #[test]
    fn test_csr_descriptor_from_triplets() {
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];

        let (desc, values) = CsrMatrixDescriptor::from_triplets(2, &triplets);

        assert_eq!(desc.n, 2);
        assert_eq!(desc.nnz, 4);
        assert_eq!(desc.row_ptr, vec![0, 2, 4]);
        assert_eq!(desc.col_idx, vec![0, 1, 0, 1]);
        assert_eq!(values, vec![4.0, 1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_csr_descriptor_duplicate_handling() {
        let triplets = vec![
            (0, 0, 2.0),
            (0, 0, 3.0), // Duplicate should sum to 5.0
            (1, 1, 1.0),
        ];

        let (desc, values) = CsrMatrixDescriptor::from_triplets(2, &triplets);

        assert_eq!(desc.nnz, 2);
        assert_eq!(values[0], 5.0);
    }

    /// Test sparse context creation on CUDA hardware.
    ///
    /// This test is ignored by default because it requires CUDA hardware.
    /// Run with: cargo test --ignored test_sparse_context_creation
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_sparse_context_creation() {
        let cuda_ctx = Arc::new(CudaContext::new().expect("CUDA context creation failed"));

        // Create sparse context
        let sparse_ctx = CudaSparseContext::new(cuda_ctx);
        assert!(sparse_ctx.is_ok(), "Failed to create sparse context: {:?}", sparse_ctx.err());
    }
}
