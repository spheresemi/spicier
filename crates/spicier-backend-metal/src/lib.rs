//! Metal/WebGPU backend for Spicier GPU-accelerated operators.
//!
//! This crate provides GPU acceleration for circuit simulation:
//! - Batched LU solving for parameter sweeps
//! - Device evaluation kernels (MOSFET, diode, BJT) for massively parallel sweeps
//! - Matrix assembly kernels for parallel stamping across sweep points
//! - Batched sparse matrix-vector multiplication for iterative solvers
//! - Batched GMRES iterative solver for parallel sweeps
//! - GPU-native Newton-Raphson solver for massively parallel circuit sweeps
//! - Memory management for large sweeps that exceed GPU buffer limits

mod batch_layout;
pub mod batched_gmres;
pub mod batched_lu;
pub mod batched_spmv;
pub mod buffer_pool;
pub mod chunked_sweep;
pub mod context;
pub mod dense_operator;
pub mod device_eval;
pub mod error;
pub mod gpu_newton;
pub mod ilu_preconditioner;
pub mod matrix_assembly;
pub mod memory;

pub use batched_gmres::{
    BatchedGmresConfig, BatchedGmresResult, GpuBatchedGmres, GpuBatchedVectorOps,
};
pub use batched_lu::{
    BatchedSolveResult, GpuBatchConfig, MAX_MATRIX_SIZE, MIN_BATCH_SIZE, MIN_MATRIX_SIZE,
    MetalBatchedLuSolver,
};
pub use batched_spmv::{BatchedCsrMatrix, GpuBatchedSpmv};
pub use buffer_pool::{BufferPool, BufferPoolStats};
pub use chunked_sweep::{
    ChunkContext, ChunkResult, ChunkedSweepConfig, ChunkedSweepExecutor,
    ChunkedSweepExecutorBuilder, ChunkedSweepResult,
};
pub use context::WgpuContext;
pub use device_eval::{
    BjtEvalResult, DiodeEvalResult, GpuBjtEvaluator, GpuBjtParams, GpuDiodeEvaluator,
    GpuDiodeParams, GpuMosfetEvaluator, GpuMosfetParams, MosfetEvalResult,
};
pub use error::{Result, WgpuError};
pub use gpu_newton::{
    BjtNodes, BjtStampLocations, DiodeDeviceInfo, DiodeNodes, DiodeStampLocations,
    GpuCircuitTopology, GpuNewtonRaphson, GpuNrConfig, GpuNrResult, MosfetDeviceInfo, MosfetNodes,
    MosfetStampLocations, VoltageLimitParams,
};
pub use ilu_preconditioner::GpuIlu0Preconditioner;
pub use matrix_assembly::{ConductanceStamp, CurrentStamp, GpuMatrixAssembler, GpuRhsAssembler};
pub use memory::{
    GpuMemoryCalculator, GpuMemoryCalculatorBuilder, GpuMemoryConfig, SweepMemoryRequirements,
};
