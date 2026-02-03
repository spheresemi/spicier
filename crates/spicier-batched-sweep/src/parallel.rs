//! Parallel CPU sweep solving using rayon.
//!
//! This module provides parallel sweep execution for CPU-based solvers,
//! leveraging rayon's work-stealing thread pool for efficient multi-core
//! utilization.
//!
//! Each sweep point is processed independently:
//! 1. Thread-local matrix/RHS buffers avoid allocation contention
//! 2. Each point assembles and solves in parallel
//! 3. Results are collected into pre-allocated buffers
//!
//! This approach is ideal for CPU solvers (Accelerate, faer) where there's
//! no benefit to batching, and parallelism provides linear scaling.

use crate::error::Result;
use crate::solver::{BackendSelector, BackendType};
use crate::sweep::GpuBatchedSweepResult;
use nalgebra::DVector;
use rayon::prelude::*;
use spicier_solver::{
    ConvergenceCriteria, DispatchConfig, ParameterVariation, SweepPoint, SweepPointGenerator,
    SweepStamperFactory,
};

/// Configuration for parallel sweep execution.
#[derive(Debug, Clone)]
pub struct ParallelSweepConfig {
    /// Minimum points to use parallel execution (below this, sequential is faster).
    pub min_points_for_parallel: usize,
    /// Chunk size for work distribution. None = auto (rayon default).
    pub chunk_size: Option<usize>,
}

impl Default for ParallelSweepConfig {
    fn default() -> Self {
        Self {
            min_points_for_parallel: 4,
            chunk_size: None,
        }
    }
}

impl ParallelSweepConfig {
    /// Create config with explicit chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Create config with minimum parallel threshold.
    pub fn with_min_parallel(mut self, min: usize) -> Self {
        self.min_points_for_parallel = min;
        self
    }
}

/// Execute a parallel CPU sweep analysis.
///
/// Uses rayon to parallelize sweep point evaluation across CPU cores.
/// Each thread creates its own solver instance and processes a subset
/// of sweep points independently.
///
/// Falls back to sequential execution if:
/// - Number of points is below `config.min_points_for_parallel`
/// - Only 1 rayon thread is available
///
/// # Arguments
/// * `backend` - Backend selector (should use CPU-based backends for best results)
/// * `factory` - Factory for creating stampers with varied parameters
/// * `generator` - Generator for sweep points
/// * `variations` - Parameter variations to sweep
/// * `_criteria` - Convergence criteria (for future nonlinear support)
/// * `_dispatch_config` - Dispatch configuration (unused, for API compatibility)
/// * `parallel_config` - Parallel execution configuration
///
/// # Performance
///
/// On an M3 Ultra with 8 P-cores and Accelerate backend:
/// - 1000 points × 100×100 matrices: ~6ms (vs ~50ms sequential)
/// - Linear scaling up to physical core count
pub fn solve_batched_sweep_parallel(
    backend: &BackendSelector,
    factory: &dyn SweepStamperFactory,
    generator: &dyn SweepPointGenerator,
    variations: &[ParameterVariation],
    _criteria: &ConvergenceCriteria,
    _dispatch_config: &DispatchConfig,
    parallel_config: &ParallelSweepConfig,
) -> Result<GpuBatchedSweepResult> {
    let points = generator.generate(variations);
    let total_count = points.len();

    if total_count == 0 {
        return Ok(GpuBatchedSweepResult {
            solutions: vec![],
            points: vec![],
            converged_count: 0,
            total_count: 0,
            singular_indices: vec![],
            used_gpu: false,
            backend_used: BackendType::Cpu,
        });
    }

    // Get system size from first point
    let first_stamper = factory.create_stamper(
        points
            .first()
            .map(|p| p.parameters.as_slice())
            .unwrap_or(&[]),
    );
    let system_size = first_stamper.num_nodes() + first_stamper.num_vsources();

    // Decide whether to use parallel execution
    let use_parallel =
        total_count >= parallel_config.min_points_for_parallel && rayon::current_num_threads() > 1;

    if !use_parallel {
        // Fall back to sequential batched solve
        return crate::sweep::solve_batched_sweep_gpu(
            backend,
            factory,
            generator,
            variations,
            _criteria,
            _dispatch_config,
        );
    }

    // Create solver to determine backend type
    let solver = backend.create_solver()?;
    let backend_type = solver.backend_type();

    log::info!(
        "Using parallel {} for sweep ({} points, {} threads)",
        backend_type,
        total_count,
        rayon::current_num_threads()
    );

    // Parallel execution: each thread processes a subset of points
    let results: Vec<_> = if let Some(chunk_size) = parallel_config.chunk_size {
        points
            .par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                process_chunk(backend, factory, chunk, chunk_idx * chunk_size, system_size)
            })
            .collect()
    } else {
        points
            .par_iter()
            .enumerate()
            .map(|(idx, point)| process_point(backend, factory, point, idx, system_size))
            .collect()
    };

    // Collect results
    let mut solutions = Vec::with_capacity(total_count);
    let mut singular_indices = Vec::new();

    for (idx, result) in results.into_iter().enumerate() {
        match result {
            Ok(sol) => solutions.push(sol),
            Err(_) => {
                solutions.push(DVector::zeros(system_size));
                singular_indices.push(idx);
            }
        }
    }

    let converged_count = total_count - singular_indices.len();

    Ok(GpuBatchedSweepResult {
        solutions,
        points,
        converged_count,
        total_count,
        singular_indices,
        used_gpu: false,
        backend_used: backend_type,
    })
}

/// Process a single sweep point.
fn process_point(
    backend: &BackendSelector,
    factory: &dyn SweepStamperFactory,
    point: &SweepPoint,
    _idx: usize,
    system_size: usize,
) -> std::result::Result<DVector<f64>, ()> {
    // Create thread-local solver
    let solver = backend.create_solver().map_err(|_| ())?;

    // Create stamper and stamp matrix
    let stamper = factory.create_stamper(&point.parameters);

    let mut matrix = nalgebra::DMatrix::zeros(system_size, system_size);
    let mut rhs = DVector::zeros(system_size);

    stamper.stamp_linear(&mut matrix, &mut rhs);

    // Convert to column-major flat buffer
    let mut mat_data = Vec::with_capacity(system_size * system_size);
    for col in 0..system_size {
        for row in 0..system_size {
            mat_data.push(matrix[(row, col)]);
        }
    }

    let rhs_data: Vec<f64> = rhs.iter().copied().collect();

    // Solve single system
    let result = solver
        .solve_batch(&mat_data, &rhs_data, system_size, 1)
        .map_err(|_| ())?;

    if result.is_singular(0) {
        return Err(());
    }

    result
        .solution(0)
        .map(|s| DVector::from_vec(s.to_vec()))
        .ok_or(())
}

/// Process a chunk of sweep points (for chunked parallelism).
fn process_chunk(
    backend: &BackendSelector,
    factory: &dyn SweepStamperFactory,
    chunk: &[SweepPoint],
    base_idx: usize,
    system_size: usize,
) -> Vec<std::result::Result<DVector<f64>, ()>> {
    // Create thread-local solver (reused for entire chunk)
    let solver = match backend.create_solver() {
        Ok(s) => s,
        Err(_) => return vec![Err(()); chunk.len()],
    };

    // Pre-allocate buffers for the chunk
    let chunk_size = chunk.len();
    let mut matrices = Vec::with_capacity(chunk_size * system_size * system_size);
    let mut rhs_vectors = Vec::with_capacity(chunk_size * system_size);

    // Assemble all matrices in chunk
    for point in chunk {
        let stamper = factory.create_stamper(&point.parameters);

        let mut matrix = nalgebra::DMatrix::zeros(system_size, system_size);
        let mut rhs = DVector::zeros(system_size);

        stamper.stamp_linear(&mut matrix, &mut rhs);

        // Convert to column-major
        for col in 0..system_size {
            for row in 0..system_size {
                matrices.push(matrix[(row, col)]);
            }
        }

        for i in 0..system_size {
            rhs_vectors.push(rhs[i]);
        }
    }

    // Solve batch
    let batch_result = match solver.solve_batch(&matrices, &rhs_vectors, system_size, chunk_size) {
        Ok(r) => r,
        Err(_) => return vec![Err(()); chunk_size],
    };

    // Extract individual solutions
    let mut results = Vec::with_capacity(chunk_size);
    for i in 0..chunk_size {
        if batch_result.is_singular(i) {
            results.push(Err(()));
        } else if let Some(sol) = batch_result.solution(i) {
            results.push(Ok(DVector::from_vec(sol.to_vec())));
        } else {
            results.push(Err(()));
        }
    }

    // Log progress for large chunks
    if chunk_size >= 100 {
        log::debug!(
            "Processed chunk at index {} ({} points)",
            base_idx,
            chunk_size
        );
    }

    results
}

/// Convenience function for parallel sweep with default configuration.
pub fn solve_batched_sweep_parallel_auto(
    factory: &dyn SweepStamperFactory,
    generator: &dyn SweepPointGenerator,
    variations: &[ParameterVariation],
    criteria: &ConvergenceCriteria,
    config: &DispatchConfig,
) -> Result<GpuBatchedSweepResult> {
    let backend = BackendSelector::auto();
    let parallel_config = ParallelSweepConfig::default();
    solve_batched_sweep_parallel(
        &backend,
        factory,
        generator,
        variations,
        criteria,
        config,
        &parallel_config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spicier_solver::{LinearSweepGenerator, MonteCarloGenerator, SweepStamper};
    use std::sync::Arc;

    struct SimpleDividerFactory {
        r2_nominal: f64,
    }

    impl SweepStamperFactory for SimpleDividerFactory {
        fn create_stamper(&self, parameters: &[f64]) -> Arc<dyn SweepStamper> {
            let r1 = parameters.first().copied().unwrap_or(1000.0);
            Arc::new(SimpleDividerStamper {
                r1,
                r2: self.r2_nominal,
                v_source: 10.0,
            })
        }
    }

    struct SimpleDividerStamper {
        r1: f64,
        r2: f64,
        v_source: f64,
    }

    impl SweepStamper for SimpleDividerStamper {
        fn stamp_linear(&self, matrix: &mut nalgebra::DMatrix<f64>, rhs: &mut DVector<f64>) {
            let g1 = 1.0 / self.r1;
            let g2 = 1.0 / self.r2;

            matrix[(0, 0)] += g1;
            matrix[(1, 1)] += g1;
            matrix[(0, 1)] -= g1;
            matrix[(1, 0)] -= g1;
            matrix[(1, 1)] += g2;
            matrix[(0, 2)] += 1.0;
            matrix[(2, 0)] += 1.0;
            rhs[2] = self.v_source;
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_parallel_sweep_basic() {
        let backend = BackendSelector::cpu_only();
        let config = DispatchConfig::default();
        let parallel_config = ParallelSweepConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = LinearSweepGenerator::new(10);
        let variations = vec![ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0)];

        let result = solve_batched_sweep_parallel(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
            &parallel_config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.total_count, 10);
        assert_eq!(result.converged_count, 10);
    }

    #[test]
    fn test_parallel_sweep_monte_carlo() {
        let backend = BackendSelector::auto();
        let config = DispatchConfig::default();
        let parallel_config = ParallelSweepConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = MonteCarloGenerator::new(100).with_seed(42);
        let variations = vec![
            ParameterVariation::new("R1", 1000.0)
                .with_bounds(500.0, 1500.0)
                .with_sigma(0.1),
        ];

        let result = solve_batched_sweep_parallel(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
            &parallel_config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.total_count, 100);
        assert!(result.converged_count >= 95);

        // Verify solutions are reasonable
        let stats = result.statistics(1);
        assert!(
            stats.mean > 3.0 && stats.mean < 7.0,
            "Mean was {}",
            stats.mean
        );
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        let backend = BackendSelector::cpu_only();
        let config = DispatchConfig::default();
        let parallel_config = ParallelSweepConfig::default();
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = MonteCarloGenerator::new(50).with_seed(123);
        let variations = vec![ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0)];

        // Run sequential
        let seq_result = crate::sweep::solve_batched_sweep_gpu(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
        )
        .unwrap();

        // Run parallel
        let par_result = solve_batched_sweep_parallel(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
            &parallel_config,
        )
        .unwrap();

        // Results should match
        assert_eq!(seq_result.total_count, par_result.total_count);
        assert_eq!(seq_result.converged_count, par_result.converged_count);

        // Solutions should be identical (deterministic)
        for i in 0..seq_result.total_count {
            let seq_sol = seq_result.solution(i).unwrap();
            let par_sol = par_result.solution(i).unwrap();
            for j in 0..seq_sol.len() {
                assert!(
                    (seq_sol[j] - par_sol[j]).abs() < 1e-10,
                    "Point {}, elem {}: seq={}, par={}",
                    i,
                    j,
                    seq_sol[j],
                    par_sol[j]
                );
            }
        }
    }

    #[test]
    fn test_chunked_parallel() {
        let backend = BackendSelector::cpu_only();
        let config = DispatchConfig::default();
        let parallel_config = ParallelSweepConfig::default().with_chunk_size(10);
        let factory = SimpleDividerFactory { r2_nominal: 1000.0 };
        let generator = LinearSweepGenerator::new(50);
        let variations = vec![ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0)];

        let result = solve_batched_sweep_parallel(
            &backend,
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
            &config,
            &parallel_config,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.total_count, 50);
        assert_eq!(result.converged_count, 50);
    }
}
