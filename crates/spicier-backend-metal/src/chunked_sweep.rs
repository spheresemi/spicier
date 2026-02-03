//! Chunked sweep execution for large parameter sweeps.
//!
//! This module provides functionality to process large sweeps that exceed
//! GPU memory limits by breaking them into manageable chunks.
//!
//! # Problem
//!
//! For very large sweeps (10k+ points) with complex circuits:
//! - 10k sweeps × 100×100 sparse (10k nnz) = 400MB values alone
//! - This exceeds typical wgpu 256MB buffer limits
//!
//! # Solution
//!
//! Process the sweep in chunks that fit within GPU memory, aggregating
//! results as we go.

use crate::buffer_pool::BufferPool;
use crate::context::WgpuContext;
use crate::error::Result;
use crate::memory::{GpuMemoryCalculator, GpuMemoryConfig};
use std::ops::Range;
use std::sync::Arc;

/// Configuration for chunked sweep execution.
#[derive(Clone, Debug)]
pub struct ChunkedSweepConfig {
    /// Chunk size (auto-calculated from memory limits if None).
    pub chunk_size: Option<usize>,
    /// Enable streaming mode: compute statistics on-GPU without storing all solutions.
    pub streaming: bool,
    /// Pre-warm buffer pool before processing.
    pub prewarm_pool: bool,
}

impl Default for ChunkedSweepConfig {
    fn default() -> Self {
        Self {
            chunk_size: None,
            streaming: false,
            prewarm_pool: true,
        }
    }
}

impl ChunkedSweepConfig {
    /// Create config for streaming mode (compute stats, don't store solutions).
    pub fn streaming() -> Self {
        Self {
            streaming: true,
            ..Default::default()
        }
    }

    /// Create config with a specific chunk size.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size: Some(chunk_size),
            ..Default::default()
        }
    }
}

/// Context passed to the chunk processor callback.
pub struct ChunkContext<'a> {
    /// Index of this chunk (0-based).
    pub chunk_idx: usize,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Range of sweep indices in this chunk.
    pub sweep_range: Range<usize>,
    /// Buffer pool for temporary allocations.
    pub buffer_pool: &'a mut BufferPool,
}

impl<'a> ChunkContext<'a> {
    /// Number of sweeps in this chunk.
    pub fn chunk_size(&self) -> usize {
        self.sweep_range.len()
    }

    /// Whether this is the first chunk.
    pub fn is_first(&self) -> bool {
        self.chunk_idx == 0
    }

    /// Whether this is the last chunk.
    pub fn is_last(&self) -> bool {
        self.chunk_idx == self.total_chunks - 1
    }

    /// Progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        (self.chunk_idx + 1) as f64 / self.total_chunks as f64
    }
}

/// Result from processing a single chunk.
#[derive(Clone, Debug)]
pub struct ChunkResult {
    /// Solutions for this chunk (num_sweeps_in_chunk × num_nodes).
    pub solutions: Vec<f32>,
    /// Whether each sweep in the chunk converged.
    pub converged: Vec<bool>,
    /// Number of iterations for each sweep in the chunk.
    pub iterations: Vec<u32>,
    /// Final residual norms for this chunk.
    pub residuals: Vec<f32>,
}

/// Result from a complete chunked sweep.
#[derive(Clone, Debug)]
pub struct ChunkedSweepResult {
    /// All solutions concatenated (num_sweeps × num_nodes).
    /// Only present if not in streaming mode.
    pub solutions: Option<Vec<f32>>,
    /// Whether each sweep converged.
    pub converged: Vec<bool>,
    /// Number of iterations for each sweep.
    pub iterations: Vec<u32>,
    /// Final residual norms.
    pub residuals: Vec<f32>,
    /// Number of chunks processed.
    pub num_chunks: usize,
    /// Total elapsed time.
    pub elapsed: std::time::Duration,
}

/// Executor for chunked sweep operations.
pub struct ChunkedSweepExecutor {
    ctx: Arc<WgpuContext>,
    buffer_pool: BufferPool,
    memory_calc: GpuMemoryCalculator,
    config: ChunkedSweepConfig,
}

impl ChunkedSweepExecutor {
    /// Create a new chunked sweep executor.
    pub fn new(ctx: Arc<WgpuContext>, config: ChunkedSweepConfig) -> Self {
        let memory_config = GpuMemoryConfig::from_context(&ctx);
        let memory_calc = GpuMemoryCalculator::new(memory_config);
        let buffer_pool = BufferPool::new(ctx.clone());

        Self {
            ctx,
            buffer_pool,
            memory_calc,
            config,
        }
    }

    /// Create with custom memory configuration.
    pub fn with_memory_config(
        ctx: Arc<WgpuContext>,
        config: ChunkedSweepConfig,
        memory_config: GpuMemoryConfig,
    ) -> Self {
        let memory_calc = GpuMemoryCalculator::new(memory_config);
        let buffer_pool = BufferPool::new(ctx.clone());

        Self {
            ctx,
            buffer_pool,
            memory_calc,
            config,
        }
    }

    /// Get the memory calculator.
    pub fn memory_calculator(&self) -> &GpuMemoryCalculator {
        &self.memory_calc
    }

    /// Get the buffer pool (for manual management).
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Get mutable access to the buffer pool.
    pub fn buffer_pool_mut(&mut self) -> &mut BufferPool {
        &mut self.buffer_pool
    }

    /// Calculate chunk size for a sweep.
    pub fn calculate_chunk_size(&self, num_sweeps: usize, nnz: usize, num_nodes: usize) -> usize {
        self.config
            .chunk_size
            .unwrap_or_else(|| self.memory_calc.chunk_size(num_sweeps, nnz, num_nodes))
    }

    /// Check if chunking is needed for a sweep.
    pub fn needs_chunking(&self, num_sweeps: usize, nnz: usize, num_nodes: usize) -> bool {
        let chunk_size = self.calculate_chunk_size(num_sweeps, nnz, num_nodes);
        chunk_size < num_sweeps
    }

    /// Execute a sweep in chunks.
    ///
    /// The `process_chunk` callback is called for each chunk and should
    /// perform the actual GPU computation.
    ///
    /// # Arguments
    /// * `num_sweeps` - Total number of sweep points
    /// * `num_nodes` - Number of nodes in the circuit
    /// * `nnz` - Number of non-zeros per matrix
    /// * `process_chunk` - Callback to process each chunk
    ///
    /// # Returns
    /// Combined results from all chunks
    pub fn execute<F>(
        &mut self,
        num_sweeps: usize,
        num_nodes: usize,
        nnz: usize,
        mut process_chunk: F,
    ) -> Result<ChunkedSweepResult>
    where
        F: FnMut(&mut ChunkContext) -> Result<ChunkResult>,
    {
        let start_time = std::time::Instant::now();
        let chunk_size = self.calculate_chunk_size(num_sweeps, nnz, num_nodes);
        let num_chunks = num_sweeps.div_ceil(chunk_size);

        // Pre-warm buffer pool if configured
        if self.config.prewarm_pool && num_chunks > 1 {
            self.prewarm_for_sweep(chunk_size, nnz, num_nodes);
        }

        // Allocate result storage
        let mut all_solutions = if self.config.streaming {
            None
        } else {
            Some(Vec::with_capacity(num_sweeps * num_nodes))
        };
        let mut all_converged = Vec::with_capacity(num_sweeps);
        let mut all_iterations = Vec::with_capacity(num_sweeps);
        let mut all_residuals = Vec::with_capacity(num_sweeps);

        // Process each chunk
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(num_sweeps);
            let sweep_range = start..end;

            let mut context = ChunkContext {
                chunk_idx,
                total_chunks: num_chunks,
                sweep_range,
                buffer_pool: &mut self.buffer_pool,
            };

            let chunk_result = process_chunk(&mut context)?;

            // Aggregate results
            if let Some(ref mut solutions) = all_solutions {
                solutions.extend(chunk_result.solutions);
            }
            all_converged.extend(chunk_result.converged);
            all_iterations.extend(chunk_result.iterations);
            all_residuals.extend(chunk_result.residuals);
        }

        Ok(ChunkedSweepResult {
            solutions: all_solutions,
            converged: all_converged,
            iterations: all_iterations,
            residuals: all_residuals,
            num_chunks,
            elapsed: start_time.elapsed(),
        })
    }

    /// Pre-warm the buffer pool for a sweep.
    fn prewarm_for_sweep(&mut self, chunk_size: usize, nnz: usize, num_nodes: usize) {
        let f32_size = std::mem::size_of::<f32>() as u64;

        // Typical buffers needed per chunk
        let csr_size = chunk_size as u64 * nnz as u64 * f32_size;
        let rhs_size = chunk_size as u64 * num_nodes as u64 * f32_size;
        let sol_size = chunk_size as u64 * num_nodes as u64 * f32_size;

        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        self.buffer_pool.prewarm(&[
            (csr_size, storage_usage, 1), // CSR values
            (rhs_size, storage_usage, 1), // RHS vector
            (sol_size, storage_usage, 2), // Solutions and prev_solutions
        ]);
    }

    /// Clear the buffer pool.
    pub fn clear_pool(&mut self) {
        self.buffer_pool.clear();
    }

    /// Get the context (for access to device/queue).
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.ctx
    }
}

/// Builder for ChunkedSweepExecutor.
pub struct ChunkedSweepExecutorBuilder {
    config: ChunkedSweepConfig,
    memory_config: Option<GpuMemoryConfig>,
}

impl ChunkedSweepExecutorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: ChunkedSweepConfig::default(),
            memory_config: None,
        }
    }

    /// Set chunk size explicitly.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = Some(size);
        self
    }

    /// Enable streaming mode.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.config.streaming = enabled;
        self
    }

    /// Set whether to pre-warm the buffer pool.
    pub fn prewarm_pool(mut self, enabled: bool) -> Self {
        self.config.prewarm_pool = enabled;
        self
    }

    /// Set custom memory configuration.
    pub fn memory_config(mut self, config: GpuMemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    /// Build the executor.
    pub fn build(self, ctx: Arc<WgpuContext>) -> ChunkedSweepExecutor {
        match self.memory_config {
            Some(mem_config) => {
                ChunkedSweepExecutor::with_memory_config(ctx, self.config, mem_config)
            }
            None => ChunkedSweepExecutor::new(ctx, self.config),
        }
    }
}

impl Default for ChunkedSweepExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to determine if a sweep needs chunking and execute appropriately.
pub fn execute_with_chunking<F, G>(
    ctx: Arc<WgpuContext>,
    num_sweeps: usize,
    num_nodes: usize,
    nnz: usize,
    single_batch: F,
    chunked: G,
) -> Result<ChunkedSweepResult>
where
    F: FnOnce() -> Result<ChunkResult>,
    G: FnOnce(&mut ChunkedSweepExecutor) -> Result<ChunkedSweepResult>,
{
    let memory_calc = GpuMemoryCalculator::from_context(&ctx);

    if memory_calc.fits_single_allocation(num_sweeps, nnz, num_nodes) {
        // Process as single batch
        let start = std::time::Instant::now();
        let result = single_batch()?;
        Ok(ChunkedSweepResult {
            solutions: Some(result.solutions),
            converged: result.converged,
            iterations: result.iterations,
            residuals: result.residuals,
            num_chunks: 1,
            elapsed: start.elapsed(),
        })
    } else {
        // Use chunked execution
        let mut executor = ChunkedSweepExecutor::new(ctx, ChunkedSweepConfig::default());
        chunked(&mut executor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_context() {
        // Can't create a real ChunkContext without a BufferPool, but we can test the logic
        let range = 100..200;
        assert_eq!(range.len(), 100);
    }

    #[test]
    fn test_chunked_sweep_config_default() {
        let config = ChunkedSweepConfig::default();
        assert!(config.chunk_size.is_none());
        assert!(!config.streaming);
        assert!(config.prewarm_pool);
    }

    #[test]
    fn test_chunked_sweep_config_streaming() {
        let config = ChunkedSweepConfig::streaming();
        assert!(config.streaming);
    }

    #[test]
    fn test_chunked_sweep_config_with_chunk_size() {
        let config = ChunkedSweepConfig::with_chunk_size(500);
        assert_eq!(config.chunk_size, Some(500));
    }
}
