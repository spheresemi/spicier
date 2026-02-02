//! CUDA backend for Spicier GPU-accelerated operators.

pub mod batched_lu;
pub mod batched_sweep;
pub mod context;
pub mod dense_operator;
pub mod error;
pub mod ilu_preconditioner;
pub mod sparse_context;

pub use batched_lu::{
    BatchedMatrices, BatchedPivots, BatchedSolveResult, BatchedVectors, CudaBatchedLuSolver,
    GpuBatchedSweepConfig, MAX_BATCH_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
};
pub use batched_sweep::{GpuBatchedSweepResult, solve_batched_sweep_gpu};
pub use context::CudaContext;
pub use error::{CudaError, Result};
pub use ilu_preconditioner::CudaIlu0Preconditioner;
pub use sparse_context::{CsrMatrixDescriptor, CudaSparseContext};
