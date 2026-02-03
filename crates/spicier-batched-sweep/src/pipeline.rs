//! Pipelined batched sweep execution.
//!
//! This module provides double-buffered pipelining to overlap CPU matrix
//! assembly with GPU (or parallel CPU) solving. While one batch is being
//! solved, the next batch is being assembled.
//!
//! # Pipeline Architecture
//!
//! ```text
//! Time -->
//!
//! CPU:  [Assemble B0] [Assemble B1] [Assemble B2] [Assemble B3] ...
//! GPU:                [Solve B0   ] [Solve B1   ] [Solve B2   ] ...
//!                     ^--- overlap ---^
//! ```
//!
//! # Example
//!
//! ```ignore
//! use spicier_batched_sweep::pipeline::{PipelinedSweep, PipelineConfig};
//!
//! let config = PipelineConfig {
//!     chunk_size: 1000,  // Process 1000 points per batch
//!     num_buffers: 2,    // Double buffering
//! };
//!
//! let result = PipelinedSweep::new(config, solver)
//!     .execute(factory, &points)?;
//! ```

use crate::error::Result;
use crate::solver::{BackendType, BatchedLuSolver, BatchedSolveResult};
use nalgebra::DVector;
use spicier_solver::{SweepPoint, SweepStamperFactory};

/// Configuration for pipelined sweep execution.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Number of sweep points per chunk/batch.
    pub chunk_size: usize,
    /// Number of buffers (2 = double buffering, 3 = triple buffering).
    pub num_buffers: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            num_buffers: 2,
        }
    }
}

impl PipelineConfig {
    /// Create config for a specific chunk size.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }

    /// Calculate optimal chunk size based on problem parameters.
    ///
    /// Aims to balance:
    /// - Large enough chunks for GPU efficiency
    /// - Small enough for good overlap
    pub fn auto(system_size: usize, total_points: usize) -> Self {
        // Heuristic: aim for ~1ms assembly time per chunk
        // Larger matrices need smaller chunks for balance
        let chunk_size = if system_size < 50 {
            2000
        } else if system_size < 100 {
            1000
        } else {
            500
        };

        // But don't make chunks larger than total / 4 (need at least 4 chunks for pipelining)
        let chunk_size = chunk_size.min(total_points / 4).max(100);

        Self {
            chunk_size,
            num_buffers: 2,
        }
    }
}

/// Pre-allocated buffer for a batch of matrices and RHS vectors.
#[derive(Debug)]
struct BatchBuffer {
    /// Flattened matrices in column-major order.
    matrices: Vec<f64>,
    /// Flattened RHS vectors.
    rhs: Vec<f64>,
    /// System size (n × n matrices, n-element RHS).
    system_size: usize,
    /// Number of systems in this batch.
    batch_size: usize,
    /// Starting index in the overall sweep.
    start_index: usize,
}

impl BatchBuffer {
    fn new(system_size: usize, max_batch_size: usize) -> Self {
        Self {
            matrices: vec![0.0; max_batch_size * system_size * system_size],
            rhs: vec![0.0; max_batch_size * system_size],
            system_size,
            batch_size: 0,
            start_index: 0,
        }
    }

    fn clear(&mut self) {
        self.batch_size = 0;
        self.start_index = 0;
    }

    fn matrices_slice(&self) -> &[f64] {
        &self.matrices[..self.batch_size * self.system_size * self.system_size]
    }

    fn rhs_slice(&self) -> &[f64] {
        &self.rhs[..self.batch_size * self.system_size]
    }
}

/// Pipelined sweep executor.
pub struct PipelinedSweep<'a> {
    config: PipelineConfig,
    solver: &'a dyn BatchedLuSolver,
}

impl<'a> PipelinedSweep<'a> {
    /// Create a new pipelined sweep executor.
    pub fn new(config: PipelineConfig, solver: &'a dyn BatchedLuSolver) -> Self {
        Self { config, solver }
    }

    /// Execute the pipelined sweep.
    ///
    /// Returns solutions in the same order as input points.
    pub fn execute(
        &self,
        factory: &dyn SweepStamperFactory,
        points: &[SweepPoint],
        system_size: usize,
    ) -> Result<PipelinedSweepResult> {
        let total_count = points.len();
        let chunk_size = self.config.chunk_size.min(total_count);

        // For small sweeps, just use simple sequential execution
        if total_count <= chunk_size || total_count < 100 {
            return self.execute_sequential(factory, points, system_size);
        }

        // For larger sweeps, use pipelined execution
        self.execute_pipelined(factory, points, system_size)
    }

    /// Simple sequential execution (no pipelining).
    fn execute_sequential(
        &self,
        factory: &dyn SweepStamperFactory,
        points: &[SweepPoint],
        system_size: usize,
    ) -> Result<PipelinedSweepResult> {
        let total_count = points.len();

        // Assemble all matrices
        let mut buffer = BatchBuffer::new(system_size, total_count);
        assemble_batch(factory, points, 0, total_count, &mut buffer);

        // Solve
        let batch_result = self.solver.solve_batch(
            buffer.matrices_slice(),
            buffer.rhs_slice(),
            system_size,
            total_count,
        )?;

        // Convert to output format
        let solutions = convert_solutions(&batch_result, system_size);

        Ok(PipelinedSweepResult {
            solutions,
            singular_indices: batch_result.singular_indices,
            total_count,
            backend_used: self.solver.backend_type(),
            chunks_processed: 1,
        })
    }

    /// Pipelined execution with overlapped assembly and solve.
    fn execute_pipelined(
        &self,
        factory: &dyn SweepStamperFactory,
        points: &[SweepPoint],
        system_size: usize,
    ) -> Result<PipelinedSweepResult> {
        let total_count = points.len();
        let chunk_size = self.config.chunk_size;
        let num_chunks = total_count.div_ceil(chunk_size);

        // Pre-allocate solution storage
        let mut all_solutions: Vec<Option<DVector<f64>>> = vec![None; total_count];
        let mut all_singular: Vec<usize> = Vec::new();

        // Process chunks with simple overlap:
        // While solving chunk K, assemble chunk K+1
        let mut current_chunk = 0;
        let mut pending_result: Option<BatchedSolveResult> = None;
        let mut pending_start: usize = 0;

        // Assemble first chunk
        let mut buffer_a = BatchBuffer::new(system_size, chunk_size);
        let mut buffer_b = BatchBuffer::new(system_size, chunk_size);

        let start = current_chunk * chunk_size;
        let end = (start + chunk_size).min(total_count);
        assemble_batch(factory, points, start, end, &mut buffer_a);
        current_chunk += 1;

        // Pipeline loop
        loop {
            // Start solve on current buffer
            let solve_result = self.solver.solve_batch(
                buffer_a.matrices_slice(),
                buffer_a.rhs_slice(),
                system_size,
                buffer_a.batch_size,
            )?;

            // Store previous result if any
            if let Some(prev_result) = pending_result.take() {
                store_results(
                    &prev_result,
                    pending_start,
                    system_size,
                    &mut all_solutions,
                    &mut all_singular,
                );
            }

            pending_result = Some(solve_result);
            pending_start = buffer_a.start_index;

            // Check if we have more chunks to assemble
            if current_chunk < num_chunks {
                // Assemble next chunk into buffer_b while solve completes
                let start = current_chunk * chunk_size;
                let end = (start + chunk_size).min(total_count);
                assemble_batch(factory, points, start, end, &mut buffer_b);
                current_chunk += 1;

                // Swap buffers
                std::mem::swap(&mut buffer_a, &mut buffer_b);
            } else {
                // No more chunks to assemble
                break;
            }
        }

        // Store final result
        if let Some(prev_result) = pending_result.take() {
            store_results(
                &prev_result,
                pending_start,
                system_size,
                &mut all_solutions,
                &mut all_singular,
            );
        }

        // Convert Option<DVector> to DVector (use zeros for None)
        let solutions: Vec<DVector<f64>> = all_solutions
            .into_iter()
            .map(|opt| opt.unwrap_or_else(|| DVector::zeros(system_size)))
            .collect();

        Ok(PipelinedSweepResult {
            solutions,
            singular_indices: all_singular,
            total_count,
            backend_used: self.solver.backend_type(),
            chunks_processed: num_chunks,
        })
    }
}

/// Assemble a batch of matrices from sweep points.
fn assemble_batch(
    factory: &dyn SweepStamperFactory,
    points: &[SweepPoint],
    start: usize,
    end: usize,
    buffer: &mut BatchBuffer,
) {
    buffer.clear();
    buffer.start_index = start;
    buffer.batch_size = end - start;

    let system_size = buffer.system_size;

    for (local_idx, point) in points[start..end].iter().enumerate() {
        let stamper = factory.create_stamper(&point.parameters);

        let mut matrix = nalgebra::DMatrix::zeros(system_size, system_size);
        let mut rhs = DVector::zeros(system_size);

        stamper.stamp_linear(&mut matrix, &mut rhs);

        // Copy to buffer in column-major order
        let mat_offset = local_idx * system_size * system_size;
        for col in 0..system_size {
            for row in 0..system_size {
                buffer.matrices[mat_offset + col * system_size + row] = matrix[(row, col)];
            }
        }

        // Copy RHS
        let rhs_offset = local_idx * system_size;
        for i in 0..system_size {
            buffer.rhs[rhs_offset + i] = rhs[i];
        }
    }
}

/// Store batch results into the overall solution array.
fn store_results(
    batch_result: &BatchedSolveResult,
    start_index: usize,
    _system_size: usize,
    all_solutions: &mut [Option<DVector<f64>>],
    all_singular: &mut Vec<usize>,
) {
    for i in 0..batch_result.batch_size {
        let global_idx = start_index + i;
        if let Some(sol_slice) = batch_result.solution(i) {
            all_solutions[global_idx] = Some(DVector::from_vec(sol_slice.to_vec()));
        }
    }

    // Map singular indices to global
    for &local_idx in &batch_result.singular_indices {
        all_singular.push(start_index + local_idx);
    }
}

/// Convert BatchedSolveResult to Vec<DVector>.
fn convert_solutions(result: &BatchedSolveResult, system_size: usize) -> Vec<DVector<f64>> {
    (0..result.batch_size)
        .map(|i| {
            result
                .solution(i)
                .map(|s| DVector::from_vec(s.to_vec()))
                .unwrap_or_else(|| DVector::zeros(system_size))
        })
        .collect()
}

/// Result of a pipelined sweep execution.
#[derive(Debug)]
pub struct PipelinedSweepResult {
    /// Solutions for each sweep point.
    pub solutions: Vec<DVector<f64>>,
    /// Indices of singular systems.
    pub singular_indices: Vec<usize>,
    /// Total number of points processed.
    pub total_count: usize,
    /// Backend used for solving.
    pub backend_used: BackendType,
    /// Number of chunks processed.
    pub chunks_processed: usize,
}

impl PipelinedSweepResult {
    /// Get the solution at a specific point.
    pub fn solution(&self, index: usize) -> Option<&DVector<f64>> {
        self.solutions.get(index)
    }

    /// Check if a specific system was singular.
    pub fn is_singular(&self, index: usize) -> bool {
        self.singular_indices.contains(&index)
    }

    /// Number of successfully solved systems.
    pub fn converged_count(&self) -> usize {
        self.total_count - self.singular_indices.len()
    }
}

/// Timing statistics for pipeline analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineTimings {
    /// Total wall clock time.
    pub total_ms: f64,
    /// Time spent assembling matrices.
    pub assembly_ms: f64,
    /// Time spent solving.
    pub solve_ms: f64,
    /// Time saved by overlapping (estimated).
    pub overlap_savings_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{CpuBatchedSolver, GpuBatchConfig};
    use spicier_solver::SweepStamper;
    use std::sync::Arc;

    struct TestStamperFactory {
        system_size: usize,
    }

    impl SweepStamperFactory for TestStamperFactory {
        fn create_stamper(&self, parameters: &[f64]) -> Arc<dyn SweepStamper> {
            let value = parameters.first().copied().unwrap_or(1.0);
            Arc::new(TestStamper {
                system_size: self.system_size,
                value,
            })
        }
    }

    struct TestStamper {
        system_size: usize,
        value: f64,
    }

    impl SweepStamper for TestStamper {
        fn stamp_linear(&self, matrix: &mut nalgebra::DMatrix<f64>, rhs: &mut DVector<f64>) {
            // Simple diagonal matrix
            for i in 0..self.system_size {
                matrix[(i, i)] = self.value;
                rhs[i] = self.value * (i as f64 + 1.0);
            }
        }

        fn num_nodes(&self) -> usize {
            self.system_size
        }

        fn num_vsources(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.num_buffers, 2);
    }

    #[test]
    fn test_pipeline_config_auto() {
        let config = PipelineConfig::auto(50, 10000);
        assert!(config.chunk_size >= 100);
        assert!(config.chunk_size <= 2500); // max 10000/4
    }

    #[test]
    fn test_sequential_execution() {
        let solver = CpuBatchedSolver::new(GpuBatchConfig::default());
        let config = PipelineConfig::with_chunk_size(100);
        let pipeline = PipelinedSweep::new(config, &solver);

        let factory = TestStamperFactory { system_size: 3 };
        let points: Vec<SweepPoint> = (0..10)
            .map(|i| SweepPoint {
                parameters: vec![1.0 + i as f64 * 0.1],
            })
            .collect();

        let result = pipeline.execute(&factory, &points, 3).unwrap();

        assert_eq!(result.total_count, 10);
        assert_eq!(result.converged_count(), 10);
        assert!(result.singular_indices.is_empty());

        // Check first solution: diagonal matrix with value 1.0
        // A = diag(1,1,1), b = [1,2,3] -> x = [1,2,3]
        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-10);
        assert!((sol0[1] - 2.0).abs() < 1e-10);
        assert!((sol0[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipelined_execution() {
        let solver = CpuBatchedSolver::new(GpuBatchConfig::default());
        let config = PipelineConfig::with_chunk_size(25); // Force chunking
        let pipeline = PipelinedSweep::new(config, &solver);

        let factory = TestStamperFactory { system_size: 3 };
        let points: Vec<SweepPoint> = (0..100)
            .map(|i| SweepPoint {
                parameters: vec![1.0 + i as f64 * 0.01],
            })
            .collect();

        let result = pipeline.execute(&factory, &points, 3).unwrap();

        assert_eq!(result.total_count, 100);
        assert_eq!(result.converged_count(), 100);
        assert_eq!(result.chunks_processed, 4); // 100 / 25 = 4 chunks

        // Verify solutions are correct
        // Solution should be [1, 2, 3] since A*x = b where A=diag(v), b=[v,2v,3v] -> x=[1,2,3]
        for (i, sol) in result.solutions.iter().enumerate() {
            assert!(
                (sol[0] - 1.0).abs() < 1e-10,
                "Point {}: sol[0]={}, expected 1.0",
                i,
                sol[0]
            );
        }
    }

    #[test]
    fn test_pipeline_handles_remainder() {
        let solver = CpuBatchedSolver::new(GpuBatchConfig::default());
        let config = PipelineConfig::with_chunk_size(30);
        let pipeline = PipelinedSweep::new(config, &solver);

        let factory = TestStamperFactory { system_size: 2 };
        // 100 points with chunk_size=30 -> chunks of 30, 30, 30, 10
        let points: Vec<SweepPoint> = (0..100)
            .map(|_| SweepPoint {
                parameters: vec![2.0],
            })
            .collect();

        let result = pipeline.execute(&factory, &points, 2).unwrap();

        assert_eq!(result.total_count, 100);
        assert_eq!(result.converged_count(), 100);
        assert_eq!(result.chunks_processed, 4);
    }
}
