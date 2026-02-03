//! Unified batched LU solver trait and backend selection.

use crate::error::Result;

/// Maximum batch size supported by most GPU backends.
pub const MAX_BATCH_SIZE: usize = 65535;

/// Minimum batch size for GPU to be worthwhile (kernel launch overhead).
pub const MIN_BATCH_SIZE: usize = 16;

/// Minimum matrix size for GPU to be worthwhile.
pub const MIN_MATRIX_SIZE: usize = 32;

/// Configuration for GPU batched operations.
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Minimum batch size to use GPU (below this, CPU is used).
    pub min_batch_size: usize,
    /// Minimum matrix dimension to use GPU.
    pub min_matrix_size: usize,
    /// Maximum batch size per GPU launch.
    pub max_batch_per_launch: usize,
}

impl Default for GpuBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: MIN_BATCH_SIZE,
            min_matrix_size: MIN_MATRIX_SIZE,
            max_batch_per_launch: MAX_BATCH_SIZE,
        }
    }
}

impl GpuBatchConfig {
    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        matrix_size >= self.min_matrix_size && batch_size >= self.min_batch_size
    }
}

/// Result of a batched LU solve operation.
#[derive(Debug, Clone)]
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

/// Type of backend being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend (wgpu-based).
    Metal,
    /// Apple Metal Performance Shaders backend (MPS - optimized kernels).
    Mps,
    /// Apple Accelerate framework (optimized LAPACK for macOS).
    Accelerate,
    /// Faer high-performance CPU (SIMD-optimized).
    Faer,
    /// CPU fallback (pure Rust nalgebra, no GPU).
    Cpu,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Mps => write!(f, "MPS"),
            BackendType::Accelerate => write!(f, "Accelerate"),
            BackendType::Faer => write!(f, "Faer"),
            BackendType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Unified trait for batched LU solvers across different GPU backends.
pub trait BatchedLuSolver: Send + Sync {
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
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult>;

    /// Check if GPU should be used for the given problem size.
    fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool;

    /// Get the backend type.
    fn backend_type(&self) -> BackendType;

    /// Get the configuration.
    fn config(&self) -> &GpuBatchConfig;
}

/// Selector for choosing the appropriate GPU backend.
#[derive(Debug, Clone, Default)]
pub struct BackendSelector {
    /// Preferred backend type.
    pub preferred: Option<BackendType>,
    /// Configuration for batch operations.
    pub config: GpuBatchConfig,
}

impl BackendSelector {
    /// Create a new backend selector with automatic detection.
    pub fn auto() -> Self {
        Self::default()
    }

    /// Create a selector that prefers CUDA.
    #[cfg(feature = "cuda")]
    pub fn prefer_cuda() -> Self {
        Self {
            preferred: Some(BackendType::Cuda),
            config: GpuBatchConfig::default(),
        }
    }

    /// Create a selector that prefers Metal.
    #[cfg(feature = "metal")]
    pub fn prefer_metal() -> Self {
        Self {
            preferred: Some(BackendType::Metal),
            config: GpuBatchConfig::default(),
        }
    }

    /// Create a selector that prefers MPS (Metal Performance Shaders).
    #[cfg(feature = "mps")]
    pub fn prefer_mps() -> Self {
        Self {
            preferred: Some(BackendType::Mps),
            config: GpuBatchConfig::default(),
        }
    }

    /// Create a selector that prefers Faer (high-performance SIMD CPU).
    #[cfg(feature = "faer")]
    pub fn prefer_faer() -> Self {
        Self {
            preferred: Some(BackendType::Faer),
            config: GpuBatchConfig::default(),
        }
    }

    /// Create a selector that prefers Apple Accelerate (macOS only).
    #[cfg(feature = "accelerate")]
    pub fn prefer_accelerate() -> Self {
        Self {
            preferred: Some(BackendType::Accelerate),
            config: GpuBatchConfig::default(),
        }
    }

    /// Create a selector that forces CPU fallback.
    pub fn cpu_only() -> Self {
        Self {
            preferred: Some(BackendType::Cpu),
            config: GpuBatchConfig::default(),
        }
    }

    /// Set custom configuration.
    pub fn with_config(mut self, config: GpuBatchConfig) -> Self {
        self.config = config;
        self
    }

    /// Try to create a batched solver for the selected backend.
    pub fn create_solver(&self) -> Result<Box<dyn BatchedLuSolver>> {
        // If CPU is explicitly requested, return CPU solver
        if self.preferred == Some(BackendType::Cpu) {
            return Ok(Box::new(CpuBatchedSolver::new(self.config.clone())));
        }

        // Try preferred backend first
        if let Some(preferred) = self.preferred {
            if let Some(solver) = self.try_create_backend(preferred) {
                return Ok(solver);
            }
        }

        // Auto-detect: try backends in order of preference
        #[cfg(feature = "cuda")]
        if let Some(solver) = self.try_create_backend(BackendType::Cuda) {
            return Ok(solver);
        }

        // Prefer MPS over Metal (wgpu) on macOS - MPS has optimized kernels
        #[cfg(feature = "mps")]
        if let Some(solver) = self.try_create_backend(BackendType::Mps) {
            return Ok(solver);
        }

        #[cfg(feature = "metal")]
        if let Some(solver) = self.try_create_backend(BackendType::Metal) {
            return Ok(solver);
        }

        // Prefer Accelerate on macOS (highly optimized LAPACK)
        #[cfg(feature = "accelerate")]
        if let Some(solver) = self.try_create_backend(BackendType::Accelerate) {
            return Ok(solver);
        }

        // Prefer Faer (high-performance SIMD) over plain CPU
        #[cfg(feature = "faer")]
        if let Some(solver) = self.try_create_backend(BackendType::Faer) {
            return Ok(solver);
        }

        // Fall back to pure Rust CPU
        Ok(Box::new(CpuBatchedSolver::new(self.config.clone())))
    }

    fn try_create_backend(&self, backend: BackendType) -> Option<Box<dyn BatchedLuSolver>> {
        match backend {
            #[cfg(feature = "cuda")]
            BackendType::Cuda => match crate::cuda::CudaBatchedSolver::new(self.config.clone()) {
                Ok(solver) => Some(Box::new(solver)),
                Err(e) => {
                    log::debug!("CUDA backend unavailable: {}", e);
                    None
                }
            },
            #[cfg(not(feature = "cuda"))]
            BackendType::Cuda => None,

            #[cfg(feature = "metal")]
            BackendType::Metal => {
                match crate::metal::MetalBatchedSolver::new(self.config.clone()) {
                    Ok(solver) => Some(Box::new(solver)),
                    Err(e) => {
                        log::debug!("Metal backend unavailable: {}", e);
                        None
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            BackendType::Metal => None,

            #[cfg(feature = "mps")]
            BackendType::Mps => match crate::mps::MpsBatchedSolver::new(self.config.clone()) {
                Ok(solver) => Some(Box::new(solver)),
                Err(e) => {
                    log::debug!("MPS backend unavailable: {}", e);
                    None
                }
            },
            #[cfg(not(feature = "mps"))]
            BackendType::Mps => None,

            #[cfg(feature = "faer")]
            BackendType::Faer => Some(Box::new(crate::faer_solver::FaerBatchedSolver::new(
                self.config.clone(),
            ))),
            #[cfg(not(feature = "faer"))]
            BackendType::Faer => None,

            #[cfg(feature = "accelerate")]
            BackendType::Accelerate => Some(Box::new(
                crate::accelerate_solver::AccelerateBatchedSolver::new(self.config.clone()),
            )),
            #[cfg(not(feature = "accelerate"))]
            BackendType::Accelerate => None,

            BackendType::Cpu => Some(Box::new(CpuBatchedSolver::new(self.config.clone()))),
        }
    }
}

/// CPU fallback solver using nalgebra's LU decomposition.
pub struct CpuBatchedSolver {
    config: GpuBatchConfig,
}

impl CpuBatchedSolver {
    /// Create a new CPU batched solver.
    pub fn new(config: GpuBatchConfig) -> Self {
        Self { config }
    }
}

impl BatchedLuSolver for CpuBatchedSolver {
    fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        use nalgebra::{DMatrix, DVector};

        let expected_matrix_len = batch_size * n * n;
        let expected_rhs_len = batch_size * n;

        if matrices.len() != expected_matrix_len {
            return Err(crate::error::BatchedSweepError::InvalidDimension(format!(
                "Expected {} matrix elements, got {}",
                expected_matrix_len,
                matrices.len()
            )));
        }

        if rhs.len() != expected_rhs_len {
            return Err(crate::error::BatchedSweepError::InvalidDimension(format!(
                "Expected {} RHS elements, got {}",
                expected_rhs_len,
                rhs.len()
            )));
        }

        let mut solutions = Vec::with_capacity(expected_rhs_len);
        let mut singular_indices = Vec::new();

        for i in 0..batch_size {
            // Extract matrix (column-major order)
            let mat_start = i * n * n;
            let mat_data = &matrices[mat_start..mat_start + n * n];
            let matrix = DMatrix::from_column_slice(n, n, mat_data);

            // Extract RHS
            let rhs_start = i * n;
            let rhs_data = &rhs[rhs_start..rhs_start + n];
            let b = DVector::from_column_slice(rhs_data);

            // Solve using LU decomposition
            let lu = nalgebra::linalg::LU::new(matrix);
            match lu.solve(&b) {
                Some(solution) => {
                    solutions.extend(solution.iter());
                }
                None => {
                    // Singular matrix - use zeros
                    solutions.extend(std::iter::repeat_n(0.0, n));
                    singular_indices.push(i);
                }
            }
        }

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
    }

    fn should_use_gpu(&self, _matrix_size: usize, _batch_size: usize) -> bool {
        false // CPU solver never uses GPU
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_solver_identity() {
        let solver = CpuBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 2;

        // Two 2×2 identity matrices in column-major order
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];

        let rhs = vec![
            1.0, 2.0, // b0 = [1, 2]
            3.0, 4.0, // b1 = [3, 4]
        ];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-10);
        assert!((sol0[1] - 2.0).abs() < 1e-10);

        let sol1 = result.solution(1).unwrap();
        assert!((sol1[0] - 3.0).abs() < 1e-10);
        assert!((sol1[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_solver_singular() {
        let solver = CpuBatchedSolver::new(GpuBatchConfig::default());

        let n = 2;
        let batch_size = 2;

        // Matrix 0: identity (non-singular)
        // Matrix 1: [[1, 2], [1, 2]] (singular)
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity
            1.0, 1.0, 2.0, 2.0, // Singular
        ];
        let rhs = vec![1.0, 2.0, 1.0, 2.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.is_singular(1));
        assert!(!result.is_singular(0));
    }

    #[test]
    fn test_backend_selector_cpu() {
        let selector = BackendSelector::cpu_only();
        let solver = selector.create_solver().unwrap();
        assert_eq!(solver.backend_type(), BackendType::Cpu);
    }

    #[test]
    fn test_config_thresholds() {
        let config = GpuBatchConfig::default();

        assert!(!config.should_use_gpu(16, 8));
        assert!(!config.should_use_gpu(64, 8));
        assert!(!config.should_use_gpu(16, 32));
        assert!(config.should_use_gpu(64, 32));
    }

    #[cfg(feature = "accelerate")]
    #[test]
    fn test_backend_selector_accelerate() {
        let selector = BackendSelector::prefer_accelerate();
        let solver = selector.create_solver().unwrap();
        assert_eq!(solver.backend_type(), BackendType::Accelerate);
    }
}
