//! Unified GPU-accelerated batched sweep solving.
//!
//! This crate provides a unified API for GPU-accelerated batched LU solving
//! across different backends (CUDA, Metal). It enables efficient parallel
//! solving for Monte Carlo, corner analysis, and parameter sweeps.
//!
//! # Features
//!
//! - `cuda` - Enable CUDA backend (NVIDIA GPUs)
//! - `metal` - Enable Metal backend (Apple GPUs)
//! - `accelerate` - Enable Apple Accelerate framework (macOS optimized LAPACK)
//! - `faer` - Enable faer backend (high-performance SIMD CPU)
//! - `parallel` - Enable parallel CPU sweeps using rayon
//!
//! # Usage
//!
//! ```ignore
//! use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};
//! use spicier_solver::{
//!     DispatchConfig, ConvergenceCriteria, MonteCarloGenerator,
//!     ParameterVariation,
//! };
//!
//! let backend = BackendSelector::auto();
//! let config = DispatchConfig::default();
//! let generator = MonteCarloGenerator::new(100);
//! let variations = vec![ParameterVariation::new("R1", 1000.0)];
//!
//! let result = solve_batched_sweep_gpu(
//!     &backend,
//!     &factory,
//!     &generator,
//!     &variations,
//!     &ConvergenceCriteria::default(),
//!     &config,
//! )?;
//!
//! println!("Used GPU: {}", result.used_gpu);
//! println!("Backend: {:?}", result.backend_used);
//! ```

pub mod batch_layout;
pub mod convergence;
mod error;
pub mod rng;
mod solver;
pub mod statistics;
mod sweep;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "mps")]
mod mps;

#[cfg(feature = "faer")]
mod faer_solver;

#[cfg(feature = "faer")]
mod faer_sparse_solver;

#[cfg(feature = "accelerate")]
mod accelerate_solver;

#[cfg(feature = "parallel")]
mod parallel;

pub use error::{BatchedSweepError, Result};
pub use solver::{
    BackendSelector, BackendType, BatchedLuSolver, BatchedSolveResult, GpuBatchConfig,
    MAX_BATCH_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
};
pub use sweep::{GpuBatchedSweepResult, solve_batched_sweep_auto, solve_batched_sweep_gpu};

// Re-export key RNG types for convenience
pub use rng::{
    CUDA_RNG_CODE, GpuRngConfig, WGSL_RNG_CODE, gaussian, gaussian_f32, gaussian_scaled,
    gaussian_scaled_f32, generate_gaussian_parameters, generate_gaussian_parameters_f32, uniform,
    uniform_f32,
};

#[cfg(feature = "cuda")]
pub use cuda::CudaBatchedSolver;

#[cfg(feature = "metal")]
pub use metal::MetalBatchedSolver;

#[cfg(feature = "mps")]
pub use mps::MpsBatchedSolver;

#[cfg(feature = "faer")]
pub use faer_solver::FaerBatchedSolver;

#[cfg(feature = "faer")]
pub use faer_sparse_solver::{FaerSparseCachedBatchedSolver, FaerTripletBatchedSolver};

#[cfg(feature = "accelerate")]
pub use accelerate_solver::AccelerateBatchedSolver;

#[cfg(feature = "parallel")]
pub use parallel::{
    solve_batched_sweep_parallel, solve_batched_sweep_parallel_auto, ParallelSweepConfig,
};
