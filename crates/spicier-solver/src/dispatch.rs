//! Solver dispatch configuration and backend integration.
//!
//! Provides a unified configuration for selecting:
//! - Compute backend (CPU, CUDA, Metal)
//! - Solver strategy (Direct LU, Iterative GMRES, Auto)
//! - Size thresholds for dispatch decisions
//! - Preconditioner configuration (Jacobi, ILU(0))

use crate::backend::ComputeBackend;
use crate::gmres::GmresConfig;

/// Configuration for GPU batched sweep operations.
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Minimum batch size to use GPU (below this, CPU is used).
    pub min_batch_size: usize,
    /// Minimum matrix dimension to use GPU.
    pub min_matrix_size: usize,
    /// Maximum batch size per GPU launch (cuBLAS limit).
    pub max_batch_per_launch: usize,
}

impl Default for GpuBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 16,
            min_matrix_size: 32,
            max_batch_per_launch: 65535,
        }
    }
}

/// Preconditioner type for iterative solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreconditionerType {
    /// No preconditioning (identity).
    None,
    /// Jacobi (diagonal) preconditioning.
    #[default]
    Jacobi,
    /// ILU(0) incomplete LU preconditioning.
    Ilu0,
}

impl PreconditionerType {
    /// Parse from a string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "none" | "identity" => Some(Self::None),
            "jacobi" | "diag" | "diagonal" => Some(Self::Jacobi),
            "ilu" | "ilu0" | "ilu(0)" => Some(Self::Ilu0),
            _ => None,
        }
    }
}

/// Configuration for ILU preconditioner.
#[derive(Debug, Clone)]
pub struct IluConfig {
    /// Whether ILU is enabled (vs fallback to Jacobi).
    pub enabled: bool,
    /// Fill level (0 = ILU(0), maintains original sparsity pattern).
    /// Higher values allow more fill-in for better preconditioning.
    pub fill_level: usize,
    /// Size threshold above which ILU is preferred over Jacobi.
    /// Below this threshold, Jacobi is used for its simplicity.
    pub size_threshold: usize,
}

impl Default for IluConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fill_level: 0,
            size_threshold: 1000, // Use ILU for matrices >= 1000
        }
    }
}

/// Solver dispatch configuration.
///
/// Controls how the solver selects between different backends and algorithms
/// based on system size and available hardware.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Compute backend to use.
    pub backend: ComputeBackend,
    /// Solver strategy selection.
    pub strategy: SolverDispatchStrategy,
    /// Size threshold below which CPU is always used (even if GPU available).
    pub cpu_threshold: usize,
    /// Size threshold above which GMRES is preferred over direct LU.
    pub gmres_threshold: usize,
    /// Size threshold above which GPU sparse direct solve is used (CUDA only).
    /// For circuits with 50k+ nodes, sparse LU on GPU via cuSOLVER.
    pub sparse_direct_threshold: usize,
    /// GMRES configuration for iterative solving.
    pub gmres_config: GmresConfig,
    /// GPU batch configuration for sweep operations.
    pub gpu_batch_config: GpuBatchConfig,
    /// ILU preconditioner configuration.
    pub ilu_config: IluConfig,
    /// Preconditioner type for GMRES.
    pub preconditioner: PreconditionerType,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            strategy: SolverDispatchStrategy::Auto,
            cpu_threshold: 1000,           // < 1k nodes: always CPU
            gmres_threshold: 10_000,       // >= 10k nodes: prefer GMRES
            sparse_direct_threshold: 50_000, // >= 50k: GPU sparse direct (CUDA)
            gmres_config: GmresConfig::default(),
            gpu_batch_config: GpuBatchConfig::default(),
            ilu_config: IluConfig::default(),
            preconditioner: PreconditionerType::default(),
        }
    }
}

impl DispatchConfig {
    /// Create a CPU-only configuration.
    pub fn cpu() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        }
    }

    /// Create a configuration with CUDA backend.
    pub fn cuda(device_id: usize) -> Self {
        Self {
            backend: ComputeBackend::Cuda { device_id },
            ..Default::default()
        }
    }

    /// Create a configuration with Metal backend.
    pub fn metal(adapter_name: impl Into<String>) -> Self {
        Self {
            backend: ComputeBackend::Metal {
                adapter_name: adapter_name.into(),
            },
            ..Default::default()
        }
    }

    /// Set the solver strategy.
    pub fn with_strategy(mut self, strategy: SolverDispatchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the CPU threshold.
    pub fn with_cpu_threshold(mut self, threshold: usize) -> Self {
        self.cpu_threshold = threshold;
        self
    }

    /// Set the GMRES threshold.
    pub fn with_gmres_threshold(mut self, threshold: usize) -> Self {
        self.gmres_threshold = threshold;
        self
    }

    /// Set the GMRES configuration.
    pub fn with_gmres_config(mut self, config: GmresConfig) -> Self {
        self.gmres_config = config;
        self
    }

    /// Decide whether to use GPU for a given system size.
    pub fn use_gpu(&self, size: usize) -> bool {
        if size < self.cpu_threshold {
            return false;
        }
        !matches!(self.backend, ComputeBackend::Cpu)
    }

    /// Decide whether to use GMRES for a given system size.
    pub fn use_gmres(&self, size: usize) -> bool {
        match self.strategy {
            SolverDispatchStrategy::DirectLU => false,
            SolverDispatchStrategy::IterativeGmres => true,
            SolverDispatchStrategy::Auto => size >= self.gmres_threshold,
        }
    }

    /// Decide whether to use GPU for batched sweep operations.
    ///
    /// Returns true if:
    /// - A GPU backend is configured (CUDA or Metal)
    /// - Matrix size is above `gpu_batch_config.min_matrix_size`
    /// - Batch size is above `gpu_batch_config.min_batch_size`
    pub fn use_gpu_batch(&self, matrix_size: usize, batch_size: usize) -> bool {
        // Must have GPU backend
        if matches!(self.backend, ComputeBackend::Cpu) {
            return false;
        }

        // Check thresholds
        matrix_size >= self.gpu_batch_config.min_matrix_size
            && batch_size >= self.gpu_batch_config.min_batch_size
    }

    /// Get the GPU batch configuration.
    pub fn gpu_batch_config(&self) -> &GpuBatchConfig {
        &self.gpu_batch_config
    }

    /// Set the GPU batch configuration.
    pub fn with_gpu_batch_config(mut self, config: GpuBatchConfig) -> Self {
        self.gpu_batch_config = config;
        self
    }

    /// Set the ILU preconditioner configuration.
    pub fn with_ilu_config(mut self, config: IluConfig) -> Self {
        self.ilu_config = config;
        self
    }

    /// Set the preconditioner type.
    pub fn with_preconditioner(mut self, preconditioner: PreconditionerType) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    /// Set the sparse direct threshold.
    pub fn with_sparse_direct_threshold(mut self, threshold: usize) -> Self {
        self.sparse_direct_threshold = threshold;
        self
    }

    /// Decide whether to use GPU sparse direct solve (CUDA cuSOLVER).
    ///
    /// Returns true if:
    /// - Backend is CUDA
    /// - Size is above sparse_direct_threshold
    pub fn use_gpu_sparse_direct(&self, size: usize) -> bool {
        matches!(self.backend, ComputeBackend::Cuda { .. })
            && size >= self.sparse_direct_threshold
    }

    /// Select the appropriate preconditioner type for a given size.
    ///
    /// Uses ILU(0) for larger systems if enabled, otherwise Jacobi.
    pub fn select_preconditioner(&self, size: usize) -> PreconditionerType {
        if self.preconditioner != PreconditionerType::Jacobi {
            // Explicit override
            return self.preconditioner;
        }

        // Auto-select: use ILU for larger systems
        if self.ilu_config.enabled && size >= self.ilu_config.size_threshold {
            PreconditionerType::Ilu0
        } else {
            PreconditionerType::Jacobi
        }
    }

    /// Get a human-readable description of the dispatch decision for a size.
    pub fn describe(&self, size: usize) -> String {
        let backend = if self.use_gpu(size) {
            self.backend.name()
        } else {
            "CPU"
        };
        let solver = if self.use_gmres(size) {
            "GMRES"
        } else {
            "Direct LU"
        };
        format!("{} with {}", solver, backend)
    }
}

/// Solver strategy for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SolverDispatchStrategy {
    /// Automatically select based on system size.
    #[default]
    Auto,
    /// Always use direct LU factorization.
    DirectLU,
    /// Always use iterative GMRES.
    IterativeGmres,
}

impl SolverDispatchStrategy {
    /// Parse from a string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "lu" | "direct" | "directlu" => Some(Self::DirectLU),
            "gmres" | "iterative" => Some(Self::IterativeGmres),
            _ => None,
        }
    }
}

/// Result of a dispatched solve operation.
#[derive(Debug, Clone)]
pub struct DispatchedSolveInfo {
    /// Backend that was actually used.
    pub backend_used: String,
    /// Solver that was actually used.
    pub solver_used: String,
    /// Number of iterations (for iterative solvers).
    pub iterations: Option<usize>,
    /// Final residual (for iterative solvers).
    pub residual: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = DispatchConfig::default();
        assert_eq!(config.backend, ComputeBackend::Cpu);
        assert_eq!(config.cpu_threshold, 1000);
        assert_eq!(config.gmres_threshold, 10_000);
    }

    #[test]
    fn use_gpu_decision() {
        let cpu_config = DispatchConfig::cpu();
        assert!(!cpu_config.use_gpu(500));
        assert!(!cpu_config.use_gpu(5000));

        let cuda_config = DispatchConfig::cuda(0).with_cpu_threshold(1000);
        assert!(!cuda_config.use_gpu(500)); // Below threshold
        assert!(cuda_config.use_gpu(1500)); // Above threshold
    }

    #[test]
    fn use_gmres_decision() {
        let config = DispatchConfig::default();

        // Auto: uses threshold
        assert!(!config.use_gmres(5000));
        assert!(config.use_gmres(15000));

        // Force direct LU
        let lu_config = config
            .clone()
            .with_strategy(SolverDispatchStrategy::DirectLU);
        assert!(!lu_config.use_gmres(15000));

        // Force GMRES
        let gmres_config = config
            .clone()
            .with_strategy(SolverDispatchStrategy::IterativeGmres);
        assert!(gmres_config.use_gmres(500));
    }

    #[test]
    fn describe_output() {
        let config = DispatchConfig::cuda(0)
            .with_cpu_threshold(1000)
            .with_gmres_threshold(5000);

        // Small: CPU + LU
        assert_eq!(config.describe(500), "Direct LU with CPU");

        // Medium: GPU + LU
        assert_eq!(config.describe(2000), "Direct LU with CUDA");

        // Large: GPU + GMRES
        assert_eq!(config.describe(10000), "GMRES with CUDA");
    }

    #[test]
    fn strategy_from_name() {
        assert_eq!(
            SolverDispatchStrategy::from_name("auto"),
            Some(SolverDispatchStrategy::Auto)
        );
        assert_eq!(
            SolverDispatchStrategy::from_name("LU"),
            Some(SolverDispatchStrategy::DirectLU)
        );
        assert_eq!(
            SolverDispatchStrategy::from_name("gmres"),
            Some(SolverDispatchStrategy::IterativeGmres)
        );
        assert_eq!(SolverDispatchStrategy::from_name("invalid"), None);
    }

    #[test]
    fn use_gpu_batch_decision() {
        // CPU backend: never use GPU batch
        let cpu_config = DispatchConfig::cpu();
        assert!(!cpu_config.use_gpu_batch(64, 32));
        assert!(!cpu_config.use_gpu_batch(100, 100));

        // CUDA backend with default thresholds
        let cuda_config = DispatchConfig::cuda(0);
        // Default: min_matrix_size = 32, min_batch_size = 16

        // Below both thresholds
        assert!(!cuda_config.use_gpu_batch(16, 8));

        // Matrix size OK, batch too small
        assert!(!cuda_config.use_gpu_batch(64, 8));

        // Batch OK, matrix too small
        assert!(!cuda_config.use_gpu_batch(16, 32));

        // Both OK
        assert!(cuda_config.use_gpu_batch(64, 32));
        assert!(cuda_config.use_gpu_batch(100, 100));
    }

    #[test]
    fn custom_gpu_batch_config() {
        let config = DispatchConfig::cuda(0).with_gpu_batch_config(GpuBatchConfig {
            min_batch_size: 8,
            min_matrix_size: 16,
            max_batch_per_launch: 1000,
        });

        // Now lower thresholds
        assert!(config.use_gpu_batch(16, 8));
        assert!(!config.use_gpu_batch(15, 8));
        assert!(!config.use_gpu_batch(16, 7));
    }
}
