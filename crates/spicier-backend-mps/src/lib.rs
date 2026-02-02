//! Metal Performance Shaders backend for Spicier GPU-accelerated batched LU solving.
//!
//! This crate uses Apple's optimized Metal Performance Shaders (MPS) for
//! efficient batched LU decomposition and solve operations on Apple GPUs.
//!
//! MPS provides highly optimized kernels that significantly outperform custom
//! compute shaders for linear algebra operations.
//!
//! # Platform Support
//!
//! This crate only functions on macOS. On other platforms, all operations
//! return `MpsError::UnsupportedPlatform`.
//!
//! # Example
//!
//! ```ignore
//! use spicier_backend_mps::{MpsContext, MpsBatchedLuSolver};
//! use std::sync::Arc;
//!
//! let ctx = Arc::new(MpsContext::new()?);
//! let solver = MpsBatchedLuSolver::new(ctx)?;
//!
//! // Solve batch of 2x2 systems
//! let matrices = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix (column-major)
//! let rhs = vec![1.0, 2.0];
//! let result = solver.solve_batch(&matrices, &rhs, 2, 1)?;
//! ```

pub mod batched_lu;
pub mod context;
pub mod error;

pub use batched_lu::{
    BatchedSolveResult, MpsBatchConfig, MpsBatchedLuSolver, MAX_BATCH_SIZE, MIN_BATCH_SIZE,
    MIN_MATRIX_SIZE,
};
pub use context::MpsContext;
pub use error::{MpsError, Result};
