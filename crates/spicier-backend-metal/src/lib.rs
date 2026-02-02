//! Metal/WebGPU backend for Spicier GPU-accelerated operators.
//!
//! This crate provides GPU acceleration for circuit simulation:
//! - Batched LU solving for parameter sweeps
//! - Device evaluation kernels (MOSFET, diode, BJT) for massively parallel sweeps

mod batch_layout;
pub mod batched_lu;
pub mod context;
pub mod dense_operator;
pub mod device_eval;
pub mod error;

pub use batched_lu::{
    BatchedSolveResult, GpuBatchConfig, MAX_MATRIX_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
    MetalBatchedLuSolver,
};
pub use context::WgpuContext;
pub use device_eval::{GpuMosfetEvaluator, GpuMosfetParams, MosfetEvalResult};
pub use error::{Result, WgpuError};
