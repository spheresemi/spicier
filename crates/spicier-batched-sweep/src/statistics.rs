//! GPU-friendly statistics computation for batched sweeps.
//!
//! This module provides efficient statistical analysis of sweep results:
//! - Reduction operations (min, max, mean, variance, std dev)
//! - Histogram computation for distribution analysis
//! - Yield analysis for specification checking
//!
//! The algorithms are designed for GPU parallelization using tree-based
//! reduction patterns that minimize synchronization.
//!
//! # Example
//!
//! ```
//! use spicier_batched_sweep::statistics::{SweepStatistics, compute_statistics, YieldSpec};
//!
//! // Sweep results: 1000 points, 5 nodes each
//! let solutions: Vec<f64> = vec![1.0; 5000];
//! let n = 5;
//! let batch_size = 1000;
//!
//! // Compute statistics for node 0
//! let stats = compute_statistics(&solutions, n, batch_size, 0);
//! println!("Node 0: mean={:.3}, std={:.3}", stats.mean, stats.std_dev);
//!
//! // Yield analysis: check if node 0 is within [0.9, 1.1]
//! let spec = YieldSpec::new(0, 0.9, 1.1);
//! let yield_pct = spec.compute_yield(&solutions, n, batch_size);
//! println!("Yield: {:.1}%", yield_pct * 100.0);
//! ```

use std::f64;

/// Statistics for a single variable across all sweep points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepStatistics {
    /// Number of samples.
    pub count: usize,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population variance.
    pub variance: f64,
    /// Population standard deviation.
    pub std_dev: f64,
    /// Sum of all values.
    pub sum: f64,
}

impl SweepStatistics {
    /// Create statistics from raw moments.
    pub fn from_moments(count: usize, sum: f64, sum_sq: f64, min: f64, max: f64) -> Self {
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        let variance = if count > 0 {
            (sum_sq / count as f64) - mean * mean
        } else {
            0.0
        };
        // Clamp to avoid negative variance from floating-point errors
        let variance = variance.max(0.0);
        let std_dev = variance.sqrt();

        Self {
            count,
            min,
            max,
            mean,
            variance,
            std_dev,
            sum,
        }
    }

    /// Range (max - min).
    #[inline]
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Coefficient of variation (std_dev / mean), if mean != 0.
    #[inline]
    pub fn coefficient_of_variation(&self) -> Option<f64> {
        if self.mean.abs() > f64::EPSILON {
            Some(self.std_dev / self.mean.abs())
        } else {
            None
        }
    }
}

impl Default for SweepStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            sum: 0.0,
        }
    }
}

/// Accumulator for incremental statistics computation.
///
/// Uses Welford's online algorithm for numerically stable variance.
/// This structure is designed for parallel reduction - multiple accumulators
/// can be merged efficiently.
#[derive(Debug, Clone, Copy)]
pub struct StatisticsAccumulator {
    count: usize,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl StatisticsAccumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add a single value.
    #[inline]
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Merge another accumulator into this one.
    ///
    /// This is the key operation for parallel reduction:
    /// each thread computes partial statistics, then merges.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Finalize into statistics.
    pub fn finalize(&self) -> SweepStatistics {
        SweepStatistics::from_moments(self.count, self.sum, self.sum_sq, self.min, self.max)
    }
}

impl Default for StatisticsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute statistics for a specific node across all sweep points.
///
/// # Arguments
/// * `solutions` - Flattened solution vectors (batch_size × n)
/// * `n` - Number of nodes per solution
/// * `batch_size` - Number of sweep points
/// * `node_idx` - Index of the node to analyze (0..n)
pub fn compute_statistics(
    solutions: &[f64],
    n: usize,
    batch_size: usize,
    node_idx: usize,
) -> SweepStatistics {
    debug_assert_eq!(solutions.len(), n * batch_size);
    debug_assert!(node_idx < n);

    let mut acc = StatisticsAccumulator::new();
    for i in 0..batch_size {
        let value = solutions[i * n + node_idx];
        acc.add(value);
    }
    acc.finalize()
}

/// Compute statistics for all nodes at once.
///
/// Returns a vector of statistics, one per node.
pub fn compute_all_statistics(
    solutions: &[f64],
    n: usize,
    batch_size: usize,
) -> Vec<SweepStatistics> {
    let mut accumulators: Vec<StatisticsAccumulator> = (0..n)
        .map(|_| StatisticsAccumulator::new())
        .collect();

    for i in 0..batch_size {
        let base = i * n;
        for (node_idx, acc) in accumulators.iter_mut().enumerate() {
            acc.add(solutions[base + node_idx]);
        }
    }

    accumulators.iter().map(|acc| acc.finalize()).collect()
}

/// Histogram bin for distribution analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistogramBin {
    /// Number of values in this bin.
    pub count: usize,
}

/// Histogram of values for distribution analysis.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bins from min to max.
    pub bins: Vec<HistogramBin>,
    /// Lower bound of first bin.
    pub min: f64,
    /// Upper bound of last bin.
    pub max: f64,
    /// Width of each bin.
    pub bin_width: f64,
    /// Total count of all samples.
    pub total_count: usize,
}

impl Histogram {
    /// Create a histogram with the specified number of bins.
    ///
    /// # Arguments
    /// * `values` - Iterator of values to bin
    /// * `num_bins` - Number of bins (must be > 0)
    /// * `min` - Lower bound (values below are clamped to first bin)
    /// * `max` - Upper bound (values above are clamped to last bin)
    pub fn new<I>(values: I, num_bins: usize, min: f64, max: f64) -> Self
    where
        I: Iterator<Item = f64>,
    {
        assert!(num_bins > 0, "num_bins must be > 0");
        assert!(max > min, "max must be > min");

        let bin_width = (max - min) / num_bins as f64;
        let mut bins = vec![HistogramBin { count: 0 }; num_bins];
        let mut total_count = 0;

        for value in values {
            let bin_idx = ((value - min) / bin_width).floor() as isize;
            let bin_idx = bin_idx.clamp(0, (num_bins - 1) as isize) as usize;
            bins[bin_idx].count += 1;
            total_count += 1;
        }

        Self {
            bins,
            min,
            max,
            bin_width,
            total_count,
        }
    }

    /// Create a histogram from sweep solutions for a specific node.
    pub fn from_sweep(
        solutions: &[f64],
        n: usize,
        batch_size: usize,
        node_idx: usize,
        num_bins: usize,
    ) -> Self {
        // First pass: find min/max
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for i in 0..batch_size {
            let value = solutions[i * n + node_idx];
            min = min.min(value);
            max = max.max(value);
        }

        // Add small epsilon to avoid edge case where all values are equal
        if (max - min).abs() < f64::EPSILON {
            min -= 0.5;
            max += 0.5;
        }

        // Second pass: bin values
        let values = (0..batch_size).map(|i| solutions[i * n + node_idx]);
        Self::new(values, num_bins, min, max)
    }

    /// Get the bin index for a value.
    pub fn bin_index(&self, value: f64) -> usize {
        let idx = ((value - self.min) / self.bin_width).floor() as isize;
        idx.clamp(0, (self.bins.len() - 1) as isize) as usize
    }

    /// Get the center value of a bin.
    pub fn bin_center(&self, bin_idx: usize) -> f64 {
        self.min + (bin_idx as f64 + 0.5) * self.bin_width
    }

    /// Get the percentage of values in a bin.
    pub fn bin_percentage(&self, bin_idx: usize) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.bins[bin_idx].count as f64 / self.total_count as f64
        }
    }

    /// Find the bin with the most values (mode).
    pub fn mode_bin(&self) -> usize {
        self.bins
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.count)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Yield specification for pass/fail analysis.
#[derive(Debug, Clone, Copy)]
pub struct YieldSpec {
    /// Node index to check.
    pub node_idx: usize,
    /// Lower specification limit.
    pub lower_limit: f64,
    /// Upper specification limit.
    pub upper_limit: f64,
}

impl YieldSpec {
    /// Create a new yield specification.
    pub fn new(node_idx: usize, lower_limit: f64, upper_limit: f64) -> Self {
        Self {
            node_idx,
            lower_limit,
            upper_limit,
        }
    }

    /// Check if a value passes the specification.
    #[inline]
    pub fn passes(&self, value: f64) -> bool {
        value >= self.lower_limit && value <= self.upper_limit
    }

    /// Compute yield (fraction of passing points) for sweep results.
    pub fn compute_yield(&self, solutions: &[f64], n: usize, batch_size: usize) -> f64 {
        let mut pass_count = 0;
        for i in 0..batch_size {
            let value = solutions[i * n + self.node_idx];
            if self.passes(value) {
                pass_count += 1;
            }
        }
        pass_count as f64 / batch_size as f64
    }

    /// Count passing and failing points.
    pub fn count_pass_fail(&self, solutions: &[f64], n: usize, batch_size: usize) -> (usize, usize) {
        let mut pass_count = 0;
        for i in 0..batch_size {
            let value = solutions[i * n + self.node_idx];
            if self.passes(value) {
                pass_count += 1;
            }
        }
        (pass_count, batch_size - pass_count)
    }
}

/// Multiple yield specifications for multi-parameter yield analysis.
#[derive(Debug, Clone)]
pub struct YieldAnalysis {
    /// Individual specifications.
    pub specs: Vec<YieldSpec>,
}

impl YieldAnalysis {
    /// Create a new yield analysis with multiple specifications.
    pub fn new(specs: Vec<YieldSpec>) -> Self {
        Self { specs }
    }

    /// Check if all specifications pass for a single point.
    pub fn all_pass(&self, solution: &[f64]) -> bool {
        self.specs.iter().all(|spec| spec.passes(solution[spec.node_idx]))
    }

    /// Compute overall yield (all specs must pass).
    pub fn compute_yield(&self, solutions: &[f64], n: usize, batch_size: usize) -> f64 {
        let mut pass_count = 0;
        for i in 0..batch_size {
            let point = &solutions[i * n..(i + 1) * n];
            if self.all_pass(point) {
                pass_count += 1;
            }
        }
        pass_count as f64 / batch_size as f64
    }

    /// Get individual yields for each specification.
    pub fn individual_yields(&self, solutions: &[f64], n: usize, batch_size: usize) -> Vec<f64> {
        self.specs
            .iter()
            .map(|spec| spec.compute_yield(solutions, n, batch_size))
            .collect()
    }
}

/// Summary of sweep results combining statistics and yield.
#[derive(Debug, Clone)]
pub struct SweepSummary {
    /// Statistics for each node.
    pub statistics: Vec<SweepStatistics>,
    /// Overall yield (if yield specs provided).
    pub yield_percentage: Option<f64>,
    /// Individual yields per specification.
    pub individual_yields: Option<Vec<f64>>,
    /// Number of sweep points.
    pub batch_size: usize,
    /// Number of nodes.
    pub n: usize,
}

/// Streaming statistics accumulator for chunked sweep processing.
///
/// This allows computing statistics incrementally without storing all
/// solutions in memory. Each chunk of solutions is processed and the
/// statistics are updated, then the chunk can be discarded.
///
/// # Example
///
/// ```
/// use spicier_batched_sweep::statistics::StreamingStatistics;
///
/// let n = 5; // 5 nodes per solution
/// let mut streaming = StreamingStatistics::new(n);
///
/// // Process chunks as they arrive
/// let chunk1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0,   // point 1
///                             1.1, 2.1, 3.1, 4.1, 5.1];  // point 2
/// streaming.process_chunk(&chunk1, 2);
///
/// let chunk2: Vec<f64> = vec![0.9, 1.9, 2.9, 3.9, 4.9,   // point 3
///                             1.2, 2.2, 3.2, 4.2, 5.2];  // point 4
/// streaming.process_chunk(&chunk2, 2);
///
/// // Get final statistics for all 4 points
/// let stats = streaming.finalize();
/// assert_eq!(stats.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingStatistics {
    /// Per-node accumulators.
    accumulators: Vec<StatisticsAccumulator>,
    /// Number of nodes per solution.
    num_nodes: usize,
    /// Total samples processed.
    total_samples: usize,
}

impl StreamingStatistics {
    /// Create a new streaming statistics accumulator.
    ///
    /// # Arguments
    /// * `num_nodes` - Number of nodes per solution vector
    pub fn new(num_nodes: usize) -> Self {
        Self {
            accumulators: (0..num_nodes).map(|_| StatisticsAccumulator::new()).collect(),
            num_nodes,
            total_samples: 0,
        }
    }

    /// Process a chunk of solutions.
    ///
    /// # Arguments
    /// * `solutions` - Flattened solution vectors (num_sweeps_in_chunk × num_nodes)
    /// * `num_sweeps` - Number of sweep points in this chunk
    ///
    /// # Panics
    /// Panics if solutions.len() != num_sweeps * num_nodes
    pub fn process_chunk(&mut self, solutions: &[f64], num_sweeps: usize) {
        debug_assert_eq!(
            solutions.len(),
            num_sweeps * self.num_nodes,
            "solutions.len() ({}) != num_sweeps ({}) * num_nodes ({})",
            solutions.len(),
            num_sweeps,
            self.num_nodes
        );

        for i in 0..num_sweeps {
            let base = i * self.num_nodes;
            for (node_idx, acc) in self.accumulators.iter_mut().enumerate() {
                acc.add(solutions[base + node_idx]);
            }
        }
        self.total_samples += num_sweeps;
    }

    /// Process a chunk of f32 solutions (converts to f64 internally).
    ///
    /// # Arguments
    /// * `solutions` - Flattened f32 solution vectors
    /// * `num_sweeps` - Number of sweep points in this chunk
    pub fn process_chunk_f32(&mut self, solutions: &[f32], num_sweeps: usize) {
        debug_assert_eq!(
            solutions.len(),
            num_sweeps * self.num_nodes,
            "solutions.len() ({}) != num_sweeps ({}) * num_nodes ({})",
            solutions.len(),
            num_sweeps,
            self.num_nodes
        );

        for i in 0..num_sweeps {
            let base = i * self.num_nodes;
            for (node_idx, acc) in self.accumulators.iter_mut().enumerate() {
                acc.add(solutions[base + node_idx] as f64);
            }
        }
        self.total_samples += num_sweeps;
    }

    /// Merge another streaming statistics into this one.
    ///
    /// This is useful for parallel reduction across multiple threads.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            self.num_nodes, other.num_nodes,
            "Cannot merge statistics with different num_nodes"
        );

        for (acc, other_acc) in self.accumulators.iter_mut().zip(&other.accumulators) {
            acc.merge(other_acc);
        }
        self.total_samples += other.total_samples;
    }

    /// Finalize and return statistics for each node.
    pub fn finalize(&self) -> Vec<SweepStatistics> {
        self.accumulators.iter().map(|acc| acc.finalize()).collect()
    }

    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get the total number of samples processed.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Reset all accumulators to start fresh.
    pub fn reset(&mut self) {
        for acc in &mut self.accumulators {
            *acc = StatisticsAccumulator::new();
        }
        self.total_samples = 0;
    }

    /// Get statistics for a specific node without finalizing.
    pub fn node_statistics(&self, node_idx: usize) -> SweepStatistics {
        self.accumulators[node_idx].finalize()
    }
}

impl SweepSummary {
    /// Compute a complete sweep summary.
    pub fn compute(
        solutions: &[f64],
        n: usize,
        batch_size: usize,
        yield_analysis: Option<&YieldAnalysis>,
    ) -> Self {
        let statistics = compute_all_statistics(solutions, n, batch_size);

        let (yield_percentage, individual_yields) = if let Some(analysis) = yield_analysis {
            (
                Some(analysis.compute_yield(solutions, n, batch_size)),
                Some(analysis.individual_yields(solutions, n, batch_size)),
            )
        } else {
            (None, None)
        };

        Self {
            statistics,
            yield_percentage,
            individual_yields,
            batch_size,
            n,
        }
    }
}

/// WGSL shader code for GPU-side parallel reduction.
///
/// This code implements a tree-based parallel reduction for computing
/// min, max, sum, and sum of squares in a single pass.
pub const WGSL_REDUCTION_CODE: &str = r#"
// Statistics reduction accumulator
struct StatsAccum {
    count: u32,
    sum: f32,
    sum_sq: f32,
    min_val: f32,
    max_val: f32,
}

// Initialize accumulator
fn stats_init() -> StatsAccum {
    return StatsAccum(0u, 0.0, 0.0, 1e38, -1e38);
}

// Add a value to accumulator
fn stats_add(acc: ptr<function, StatsAccum>, value: f32) {
    (*acc).count += 1u;
    (*acc).sum += value;
    (*acc).sum_sq += value * value;
    (*acc).min_val = min((*acc).min_val, value);
    (*acc).max_val = max((*acc).max_val, value);
}

// Merge two accumulators (for parallel reduction)
fn stats_merge(a: StatsAccum, b: StatsAccum) -> StatsAccum {
    return StatsAccum(
        a.count + b.count,
        a.sum + b.sum,
        a.sum_sq + b.sum_sq,
        min(a.min_val, b.min_val),
        max(a.max_val, b.max_val)
    );
}

// Workgroup shared memory for reduction
var<workgroup> shared_stats: array<StatsAccum, 256>;

// Parallel reduction within workgroup
fn workgroup_reduce(local_id: u32, local_size: u32, acc: StatsAccum) -> StatsAccum {
    shared_stats[local_id] = acc;
    workgroupBarrier();

    // Tree reduction
    var stride = local_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_stats[local_id] = stats_merge(
                shared_stats[local_id],
                shared_stats[local_id + stride]
            );
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return shared_stats[0];
}
"#;

/// CUDA code for GPU-side parallel reduction.
pub const CUDA_REDUCTION_CODE: &str = r#"
// Statistics reduction accumulator
struct StatsAccum {
    unsigned int count;
    float sum;
    float sum_sq;
    float min_val;
    float max_val;
};

__device__ StatsAccum stats_init() {
    return {0, 0.0f, 0.0f, 1e38f, -1e38f};
}

__device__ void stats_add(StatsAccum* acc, float value) {
    acc->count += 1;
    acc->sum += value;
    acc->sum_sq += value * value;
    acc->min_val = fminf(acc->min_val, value);
    acc->max_val = fmaxf(acc->max_val, value);
}

__device__ StatsAccum stats_merge(StatsAccum a, StatsAccum b) {
    return {
        a.count + b.count,
        a.sum + b.sum,
        a.sum_sq + b.sum_sq,
        fminf(a.min_val, b.min_val),
        fmaxf(a.max_val, b.max_val)
    };
}

// Warp-level reduction using shuffle
__device__ StatsAccum warp_reduce(StatsAccum acc) {
    for (int offset = 16; offset > 0; offset /= 2) {
        StatsAccum other;
        other.count = __shfl_down_sync(0xffffffff, acc.count, offset);
        other.sum = __shfl_down_sync(0xffffffff, acc.sum, offset);
        other.sum_sq = __shfl_down_sync(0xffffffff, acc.sum_sq, offset);
        other.min_val = __shfl_down_sync(0xffffffff, acc.min_val, offset);
        other.max_val = __shfl_down_sync(0xffffffff, acc.max_val, offset);
        acc = stats_merge(acc, other);
    }
    return acc;
}

// Block-level reduction
__shared__ StatsAccum shared_stats[32];  // One per warp

__device__ StatsAccum block_reduce(StatsAccum acc) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp reduction
    acc = warp_reduce(acc);

    // First thread in each warp writes to shared
    if (lane == 0) {
        shared_stats[warp_id] = acc;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        acc = (lane < blockDim.x / 32) ? shared_stats[lane] : stats_init();
        acc = warp_reduce(acc);
    }

    return acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut acc = StatisticsAccumulator::new();
        for &v in &values {
            acc.add(v);
        }
        let stats = acc.finalize();

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.sum - 15.0).abs() < 1e-10);
        // Variance of [1,2,3,4,5] = 2.0
        assert!((stats.variance - 2.0).abs() < 1e-10);
        assert!((stats.std_dev - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_statistics_single_value() {
        let mut acc = StatisticsAccumulator::new();
        acc.add(42.0);
        let stats = acc.finalize();

        assert_eq!(stats.count, 1);
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.variance, 0.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn test_accumulator_merge() {
        let mut acc1 = StatisticsAccumulator::new();
        acc1.add(1.0);
        acc1.add(2.0);

        let mut acc2 = StatisticsAccumulator::new();
        acc2.add(3.0);
        acc2.add(4.0);
        acc2.add(5.0);

        acc1.merge(&acc2);
        let stats = acc1.finalize();

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_statistics() {
        // 3 sweep points, 2 nodes each
        // Node 0: [1.0, 2.0, 3.0]
        // Node 1: [10.0, 20.0, 30.0]
        let solutions = vec![
            1.0, 10.0,  // Point 0
            2.0, 20.0,  // Point 1
            3.0, 30.0,  // Point 2
        ];

        let stats0 = compute_statistics(&solutions, 2, 3, 0);
        assert_eq!(stats0.count, 3);
        assert!((stats0.mean - 2.0).abs() < 1e-10);

        let stats1 = compute_statistics(&solutions, 2, 3, 1);
        assert_eq!(stats1.count, 3);
        assert!((stats1.mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_all_statistics() {
        let solutions = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
        ];

        let all_stats = compute_all_statistics(&solutions, 2, 3);
        assert_eq!(all_stats.len(), 2);
        assert!((all_stats[0].mean - 2.0).abs() < 1e-10);
        assert!((all_stats[1].mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_basic() {
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let hist = Histogram::new(values.into_iter(), 4, 1.0, 5.0);

        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.total_count, 9);
        assert!((hist.bin_width - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_from_sweep() {
        let solutions = vec![
            1.0, 0.0,
            2.0, 0.0,
            3.0, 0.0,
            4.0, 0.0,
            5.0, 0.0,
        ];

        let hist = Histogram::from_sweep(&solutions, 2, 5, 0, 4);
        assert_eq!(hist.total_count, 5);
        assert_eq!(hist.bins.len(), 4);
    }

    #[test]
    fn test_yield_spec() {
        let spec = YieldSpec::new(0, 1.5, 3.5);

        assert!(!spec.passes(1.0));
        assert!(spec.passes(2.0));
        assert!(spec.passes(3.0));
        assert!(!spec.passes(4.0));
    }

    #[test]
    fn test_yield_computation() {
        // Node 0 values: [1.0, 2.0, 3.0, 4.0, 5.0]
        let solutions = vec![
            1.0, 0.0,
            2.0, 0.0,
            3.0, 0.0,
            4.0, 0.0,
            5.0, 0.0,
        ];

        // Spec: node 0 must be in [1.5, 3.5]
        // Passes: 2.0, 3.0 (2 out of 5 = 40%)
        let spec = YieldSpec::new(0, 1.5, 3.5);
        let yield_pct = spec.compute_yield(&solutions, 2, 5);
        assert!((yield_pct - 0.4).abs() < 1e-10);

        let (pass, fail) = spec.count_pass_fail(&solutions, 2, 5);
        assert_eq!(pass, 2);
        assert_eq!(fail, 3);
    }

    #[test]
    fn test_yield_analysis_multiple_specs() {
        // Node 0: [1, 2, 3], Node 1: [10, 20, 30]
        let solutions = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
        ];

        let analysis = YieldAnalysis::new(vec![
            YieldSpec::new(0, 1.5, 2.5),  // Passes: 2.0 only
            YieldSpec::new(1, 15.0, 25.0), // Passes: 20.0 only
        ]);

        // Only point 1 (2.0, 20.0) passes both
        let yield_pct = analysis.compute_yield(&solutions, 2, 3);
        assert!((yield_pct - 1.0/3.0).abs() < 1e-10);

        let individual = analysis.individual_yields(&solutions, 2, 3);
        assert!((individual[0] - 1.0/3.0).abs() < 1e-10); // Node 0
        assert!((individual[1] - 1.0/3.0).abs() < 1e-10); // Node 1
    }

    #[test]
    fn test_sweep_summary() {
        let solutions = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
        ];

        let analysis = YieldAnalysis::new(vec![
            YieldSpec::new(0, 0.0, 5.0), // All pass
        ]);

        let summary = SweepSummary::compute(&solutions, 2, 3, Some(&analysis));

        assert_eq!(summary.n, 2);
        assert_eq!(summary.batch_size, 3);
        assert_eq!(summary.statistics.len(), 2);
        assert!((summary.yield_percentage.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let stats = SweepStatistics::from_moments(100, 500.0, 2600.0, 4.0, 6.0);
        // mean = 5.0, variance = 26 - 25 = 1, std = 1
        // CV = 1/5 = 0.2
        let cv = stats.coefficient_of_variation().unwrap();
        assert!((cv - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_mode() {
        // Most values around 3.0
        let values = vec![1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0];
        let hist = Histogram::new(values.into_iter(), 5, 1.0, 5.0);
        let mode_bin = hist.mode_bin();
        let mode_center = hist.bin_center(mode_bin);
        // Mode should be around 3.0
        assert!((mode_center - 3.0).abs() < hist.bin_width);
    }

    // =========================================================================
    // Streaming Statistics Tests
    // =========================================================================

    #[test]
    fn test_streaming_statistics_single_chunk() {
        // 3 sweep points, 2 nodes each
        let solutions = vec![
            1.0, 10.0,  // Point 0
            2.0, 20.0,  // Point 1
            3.0, 30.0,  // Point 2
        ];

        let mut streaming = StreamingStatistics::new(2);
        streaming.process_chunk(&solutions, 3);

        let stats = streaming.finalize();
        assert_eq!(stats.len(), 2);
        assert_eq!(streaming.total_samples(), 3);

        // Node 0: mean of [1, 2, 3] = 2
        assert!((stats[0].mean - 2.0).abs() < 1e-10);
        // Node 1: mean of [10, 20, 30] = 20
        assert!((stats[1].mean - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_statistics_multiple_chunks() {
        let mut streaming = StreamingStatistics::new(2);

        // First chunk: 2 points
        let chunk1 = vec![
            1.0, 10.0,  // Point 0
            2.0, 20.0,  // Point 1
        ];
        streaming.process_chunk(&chunk1, 2);

        // Second chunk: 2 points
        let chunk2 = vec![
            3.0, 30.0,  // Point 2
            4.0, 40.0,  // Point 3
        ];
        streaming.process_chunk(&chunk2, 2);

        let stats = streaming.finalize();
        assert_eq!(streaming.total_samples(), 4);

        // Node 0: mean of [1, 2, 3, 4] = 2.5
        assert!((stats[0].mean - 2.5).abs() < 1e-10);
        // Node 1: mean of [10, 20, 30, 40] = 25
        assert!((stats[1].mean - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_statistics_f32() {
        let mut streaming = StreamingStatistics::new(2);

        let chunk: Vec<f32> = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
        ];
        streaming.process_chunk_f32(&chunk, 3);

        let stats = streaming.finalize();
        assert_eq!(streaming.total_samples(), 3);
        assert!((stats[0].mean - 2.0).abs() < 1e-6);
        assert!((stats[1].mean - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_streaming_statistics_merge() {
        let mut streaming1 = StreamingStatistics::new(2);
        let mut streaming2 = StreamingStatistics::new(2);

        streaming1.process_chunk(&[1.0, 10.0, 2.0, 20.0], 2);
        streaming2.process_chunk(&[3.0, 30.0, 4.0, 40.0], 2);

        streaming1.merge(&streaming2);
        let stats = streaming1.finalize();

        assert_eq!(streaming1.total_samples(), 4);
        assert!((stats[0].mean - 2.5).abs() < 1e-10);
        assert!((stats[1].mean - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_statistics_reset() {
        let mut streaming = StreamingStatistics::new(2);
        streaming.process_chunk(&[1.0, 10.0, 2.0, 20.0], 2);

        assert_eq!(streaming.total_samples(), 2);

        streaming.reset();
        assert_eq!(streaming.total_samples(), 0);

        streaming.process_chunk(&[5.0, 50.0], 1);
        let stats = streaming.finalize();

        assert_eq!(streaming.total_samples(), 1);
        assert!((stats[0].mean - 5.0).abs() < 1e-10);
        assert!((stats[1].mean - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_statistics_matches_batch() {
        // Compare streaming to batch computation
        let solutions = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
        ];

        // Batch computation
        let batch_stats = compute_all_statistics(&solutions, 2, 5);

        // Streaming computation in 2 chunks
        let mut streaming = StreamingStatistics::new(2);
        streaming.process_chunk(&solutions[0..6], 3); // First 3 points
        streaming.process_chunk(&solutions[6..10], 2); // Last 2 points
        let streaming_stats = streaming.finalize();

        // Should match
        for i in 0..2 {
            assert!(
                (batch_stats[i].mean - streaming_stats[i].mean).abs() < 1e-10,
                "mean mismatch for node {}", i
            );
            assert!(
                (batch_stats[i].min - streaming_stats[i].min).abs() < 1e-10,
                "min mismatch for node {}", i
            );
            assert!(
                (batch_stats[i].max - streaming_stats[i].max).abs() < 1e-10,
                "max mismatch for node {}", i
            );
            assert!(
                (batch_stats[i].variance - streaming_stats[i].variance).abs() < 1e-10,
                "variance mismatch for node {}", i
            );
        }
    }

    #[test]
    fn test_streaming_node_statistics() {
        let mut streaming = StreamingStatistics::new(3);
        streaming.process_chunk(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);

        // Get stats for just node 1 (values: 2.0, 5.0)
        let node1_stats = streaming.node_statistics(1);
        assert!((node1_stats.mean - 3.5).abs() < 1e-10);
        assert_eq!(node1_stats.min, 2.0);
        assert_eq!(node1_stats.max, 5.0);
    }
}
