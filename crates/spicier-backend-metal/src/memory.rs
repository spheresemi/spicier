//! GPU memory management utilities for large sweep operations.
//!
//! This module provides tools for managing GPU memory when running large
//! parameter sweeps that may exceed GPU buffer limits.
//!
//! # Memory Limits
//!
//! WebGPU/wgpu typically enforces a 256MB maximum buffer size. For large sweeps,
//! the working set can easily exceed this:
//!
//! - 10k sweeps × 100×100 sparse matrix (10k nnz) = 400MB values alone
//!
//! This module calculates optimal chunk sizes to stay within limits.

use crate::context::WgpuContext;

/// Default maximum buffer size (256 MB).
pub const DEFAULT_MAX_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

/// GPU memory configuration.
#[derive(Clone, Debug)]
pub struct GpuMemoryConfig {
    /// Maximum buffer size (query from device or use 256MB default).
    pub max_buffer_size: u64,
    /// Target memory usage ratio (0.8 = 80% of max to leave headroom).
    pub target_usage_ratio: f64,
    /// Minimum chunk size (don't chunk below this many sweeps).
    pub min_chunk_size: usize,
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            target_usage_ratio: 0.8,
            min_chunk_size: 16,
        }
    }
}

impl GpuMemoryConfig {
    /// Create config from a WgpuContext, querying device limits.
    pub fn from_context(ctx: &WgpuContext) -> Self {
        Self {
            max_buffer_size: ctx.max_buffer_size(),
            ..Default::default()
        }
    }

    /// Create config with a custom max buffer size.
    pub fn with_max_buffer_size(max_buffer_size: u64) -> Self {
        Self {
            max_buffer_size,
            ..Default::default()
        }
    }
}

/// Calculator for GPU memory requirements and chunk sizes.
#[derive(Clone, Debug)]
pub struct GpuMemoryCalculator {
    config: GpuMemoryConfig,
}

impl GpuMemoryCalculator {
    /// Create a new memory calculator with the given config.
    pub fn new(config: GpuMemoryConfig) -> Self {
        Self { config }
    }

    /// Create a memory calculator from a WgpuContext.
    pub fn from_context(ctx: &WgpuContext) -> Self {
        Self::new(GpuMemoryConfig::from_context(ctx))
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(GpuMemoryConfig::default())
    }

    /// Get the effective max buffer size (with headroom applied).
    pub fn effective_max_size(&self) -> u64 {
        (self.config.max_buffer_size as f64 * self.config.target_usage_ratio) as u64
    }

    /// Calculate memory needed per sweep point.
    ///
    /// This accounts for:
    /// - CSR values: nnz × 4 bytes (f32)
    /// - RHS vector: num_nodes × 4 bytes (f32)
    /// - Solution vector: num_nodes × 4 bytes (f32)
    /// - Previous solution: num_nodes × 4 bytes (f32)
    /// - GMRES Krylov basis: num_nodes × max_krylov × 4 bytes
    /// - GMRES Hessenberg: max_krylov × max_krylov × 4 bytes (amortized)
    /// - Device eval results: proportional to device count
    pub fn memory_per_sweep(&self, nnz: usize, num_nodes: usize, max_krylov: usize) -> u64 {
        let f32_size = std::mem::size_of::<f32>() as u64;

        // Core solver buffers per sweep
        let csr_values = nnz as u64 * f32_size;
        let rhs = num_nodes as u64 * f32_size;
        let solution = num_nodes as u64 * f32_size;
        let prev_solution = num_nodes as u64 * f32_size;

        // GMRES buffers per sweep
        let krylov_basis = num_nodes as u64 * max_krylov as u64 * f32_size;
        let residual = num_nodes as u64 * f32_size;

        // Preconditioner diagonal (Jacobi)
        let precond = num_nodes as u64 * f32_size;

        // Total per sweep (conservative estimate)
        csr_values + rhs + solution + prev_solution + krylov_basis + residual + precond
    }

    /// Calculate total memory needed for a sweep.
    ///
    /// # Arguments
    /// * `num_sweeps` - Number of sweep points
    /// * `nnz` - Number of non-zeros in the sparse matrix
    /// * `num_nodes` - Number of nodes in the circuit
    ///
    /// # Returns
    /// Total bytes needed (approximate upper bound)
    pub fn total_memory_needed(&self, num_sweeps: usize, nnz: usize, num_nodes: usize) -> u64 {
        // Default GMRES Krylov dimension
        let max_krylov = 30;
        self.total_memory_needed_with_krylov(num_sweeps, nnz, num_nodes, max_krylov)
    }

    /// Calculate total memory with specific Krylov dimension.
    pub fn total_memory_needed_with_krylov(
        &self,
        num_sweeps: usize,
        nnz: usize,
        num_nodes: usize,
        max_krylov: usize,
    ) -> u64 {
        let per_sweep = self.memory_per_sweep(nnz, num_nodes, max_krylov);
        per_sweep * num_sweeps as u64
    }

    /// Check if sweep fits in a single buffer allocation.
    ///
    /// # Arguments
    /// * `num_sweeps` - Number of sweep points
    /// * `nnz` - Number of non-zeros per matrix
    ///
    /// # Returns
    /// `true` if the sweep can be processed in a single batch
    pub fn fits_single_buffer(&self, num_sweeps: usize, nnz: usize) -> bool {
        // The largest single buffer is typically the CSR values
        let csr_buffer_size = num_sweeps as u64 * nnz as u64 * std::mem::size_of::<f32>() as u64;
        csr_buffer_size <= self.effective_max_size()
    }

    /// Check if sweep fits in a single allocation for all buffers.
    pub fn fits_single_allocation(
        &self,
        num_sweeps: usize,
        nnz: usize,
        num_nodes: usize,
    ) -> bool {
        let total = self.total_memory_needed(num_sweeps, nnz, num_nodes);
        total <= self.effective_max_size()
    }

    /// Calculate optimal chunk size for a sweep.
    ///
    /// Returns the maximum number of sweep points that can be processed
    /// in a single GPU batch while staying within memory limits.
    ///
    /// # Arguments
    /// * `num_sweeps` - Total number of sweep points
    /// * `nnz` - Number of non-zeros per matrix
    /// * `num_nodes` - Number of nodes in the circuit
    ///
    /// # Returns
    /// Optimal chunk size (may be >= num_sweeps if everything fits)
    pub fn chunk_size(&self, num_sweeps: usize, nnz: usize, num_nodes: usize) -> usize {
        self.chunk_size_with_krylov(num_sweeps, nnz, num_nodes, 30)
    }

    /// Calculate chunk size with specific Krylov dimension.
    pub fn chunk_size_with_krylov(
        &self,
        num_sweeps: usize,
        nnz: usize,
        num_nodes: usize,
        max_krylov: usize,
    ) -> usize {
        let per_sweep = self.memory_per_sweep(nnz, num_nodes, max_krylov);
        if per_sweep == 0 {
            return num_sweeps;
        }

        let effective_max = self.effective_max_size();
        let max_sweeps = (effective_max / per_sweep) as usize;

        // Ensure at least min_chunk_size
        let chunk = max_sweeps.max(self.config.min_chunk_size);

        // Cap at total sweeps
        chunk.min(num_sweeps)
    }

    /// Calculate number of chunks needed for a sweep.
    pub fn num_chunks(&self, num_sweeps: usize, nnz: usize, num_nodes: usize) -> usize {
        let chunk = self.chunk_size(num_sweeps, nnz, num_nodes);
        (num_sweeps + chunk - 1) / chunk
    }

    /// Get the configured max buffer size.
    pub fn max_buffer_size(&self) -> u64 {
        self.config.max_buffer_size
    }

    /// Get the config.
    pub fn config(&self) -> &GpuMemoryConfig {
        &self.config
    }
}

/// Builder for GpuMemoryCalculator with fluent API.
#[derive(Clone, Debug)]
pub struct GpuMemoryCalculatorBuilder {
    config: GpuMemoryConfig,
}

impl GpuMemoryCalculatorBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: GpuMemoryConfig::default(),
        }
    }

    /// Set maximum buffer size.
    pub fn max_buffer_size(mut self, size: u64) -> Self {
        self.config.max_buffer_size = size;
        self
    }

    /// Set target usage ratio (0.0 to 1.0).
    pub fn target_usage_ratio(mut self, ratio: f64) -> Self {
        self.config.target_usage_ratio = ratio.clamp(0.1, 1.0);
        self
    }

    /// Set minimum chunk size.
    pub fn min_chunk_size(mut self, size: usize) -> Self {
        self.config.min_chunk_size = size.max(1);
        self
    }

    /// Build the calculator.
    pub fn build(self) -> GpuMemoryCalculator {
        GpuMemoryCalculator::new(self.config)
    }
}

impl Default for GpuMemoryCalculatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory requirements for a sweep operation.
#[derive(Clone, Debug)]
pub struct SweepMemoryRequirements {
    /// Total memory needed (bytes).
    pub total_bytes: u64,
    /// Memory per sweep point (bytes).
    pub bytes_per_sweep: u64,
    /// Whether it fits in a single allocation.
    pub fits_single: bool,
    /// Recommended chunk size.
    pub chunk_size: usize,
    /// Number of chunks needed.
    pub num_chunks: usize,
}

impl GpuMemoryCalculator {
    /// Get complete memory requirements for a sweep.
    pub fn requirements(
        &self,
        num_sweeps: usize,
        nnz: usize,
        num_nodes: usize,
    ) -> SweepMemoryRequirements {
        let bytes_per_sweep = self.memory_per_sweep(nnz, num_nodes, 30);
        let total_bytes = self.total_memory_needed(num_sweeps, nnz, num_nodes);
        let fits_single = self.fits_single_allocation(num_sweeps, nnz, num_nodes);
        let chunk_size = self.chunk_size(num_sweeps, nnz, num_nodes);
        let num_chunks = self.num_chunks(num_sweeps, nnz, num_nodes);

        SweepMemoryRequirements {
            total_bytes,
            bytes_per_sweep,
            fits_single,
            chunk_size,
            num_chunks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_calculator_default() {
        let calc = GpuMemoryCalculator::with_defaults();
        assert_eq!(calc.max_buffer_size(), DEFAULT_MAX_BUFFER_SIZE);
    }

    #[test]
    fn test_memory_per_sweep() {
        let calc = GpuMemoryCalculator::with_defaults();

        // 100 nodes, 1000 nnz, 30 Krylov
        let per_sweep = calc.memory_per_sweep(1000, 100, 30);

        // Should be > 0 and reasonable
        assert!(per_sweep > 0);
        // CSR values alone: 1000 * 4 = 4KB
        // Solutions: 100 * 4 * 2 = 800B
        // Krylov basis: 100 * 30 * 4 = 12KB
        // Total should be in the 20-50KB range per sweep
        assert!(per_sweep > 15_000);
        assert!(per_sweep < 100_000);
    }

    #[test]
    fn test_fits_single_buffer_small() {
        let calc = GpuMemoryCalculator::with_defaults();

        // Small sweep: 100 sweeps, 1000 nnz
        // CSR buffer: 100 * 1000 * 4 = 400KB - should fit
        assert!(calc.fits_single_buffer(100, 1000));
    }

    #[test]
    fn test_fits_single_buffer_large() {
        let calc = GpuMemoryCalculator::with_defaults();

        // Large sweep: 100,000 sweeps, 10,000 nnz
        // CSR buffer: 100,000 * 10,000 * 4 = 4GB - should NOT fit
        assert!(!calc.fits_single_buffer(100_000, 10_000));
    }

    #[test]
    fn test_chunk_size_calculation() {
        let calc = GpuMemoryCalculator::with_defaults();

        // Parameters that would need chunking
        let num_sweeps = 10_000;
        let nnz = 10_000;
        let num_nodes = 100;

        let chunk = calc.chunk_size(num_sweeps, nnz, num_nodes);

        // Chunk should be less than total sweeps
        assert!(chunk < num_sweeps);
        // Chunk should be at least min_chunk_size
        assert!(chunk >= calc.config().min_chunk_size);
        // Chunk should fit in memory
        assert!(calc.fits_single_allocation(chunk, nnz, num_nodes));
    }

    #[test]
    fn test_chunk_size_when_fits() {
        let calc = GpuMemoryCalculator::with_defaults();

        // Small sweep that should fit in one chunk
        let num_sweeps = 100;
        let nnz = 500;
        let num_nodes = 50;

        let chunk = calc.chunk_size(num_sweeps, nnz, num_nodes);

        // Should return full sweep size
        assert_eq!(chunk, num_sweeps);
    }

    #[test]
    fn test_num_chunks() {
        let calc = GpuMemoryCalculator::with_defaults();

        // Parameters that would need chunking
        let num_sweeps = 10_000;
        let nnz = 10_000;
        let num_nodes = 100;

        let num_chunks = calc.num_chunks(num_sweeps, nnz, num_nodes);
        let chunk_size = calc.chunk_size(num_sweeps, nnz, num_nodes);

        // num_chunks * chunk_size should cover all sweeps
        assert!(num_chunks * chunk_size >= num_sweeps);
        // But not too wastefully
        assert!((num_chunks - 1) * chunk_size < num_sweeps);
    }

    #[test]
    fn test_requirements() {
        let calc = GpuMemoryCalculator::with_defaults();

        let reqs = calc.requirements(1000, 5000, 100);

        assert!(reqs.total_bytes > 0);
        assert!(reqs.bytes_per_sweep > 0);
        assert!(reqs.chunk_size >= 1);
        assert!(reqs.num_chunks >= 1);
    }

    #[test]
    fn test_builder() {
        let calc = GpuMemoryCalculatorBuilder::new()
            .max_buffer_size(128 * 1024 * 1024)
            .target_usage_ratio(0.9)
            .min_chunk_size(32)
            .build();

        assert_eq!(calc.max_buffer_size(), 128 * 1024 * 1024);
        assert!((calc.config().target_usage_ratio - 0.9).abs() < 0.001);
        assert_eq!(calc.config().min_chunk_size, 32);
    }

    #[test]
    fn test_effective_max_size() {
        let calc = GpuMemoryCalculatorBuilder::new()
            .max_buffer_size(100)
            .target_usage_ratio(0.5)
            .build();

        assert_eq!(calc.effective_max_size(), 50);
    }
}
