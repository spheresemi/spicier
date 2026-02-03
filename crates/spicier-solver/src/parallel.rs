//! Parallel matrix assembly utilities.
//!
//! Provides thread-safe mechanisms for parallel stamping into sparse matrices.
//! Uses thread-local accumulation with final merge to avoid lock contention.

use std::sync::Mutex;
use std::thread;

/// Thread-local triplet accumulator for parallel matrix assembly.
///
/// Each thread accumulates triplets into its own Vec, then all are merged
/// at the end. This avoids lock contention during the hot stamping loop.
#[derive(Debug)]
pub struct ParallelTripletAccumulator {
    /// Thread-local triplet buffers.
    buffers: Mutex<Vec<Vec<(usize, usize, f64)>>>,
    /// Number of threads to use.
    num_threads: usize,
}

impl ParallelTripletAccumulator {
    /// Create a new accumulator with the specified number of threads.
    pub fn new(num_threads: usize) -> Self {
        let buffers: Vec<Vec<(usize, usize, f64)>> =
            (0..num_threads).map(|_| Vec::with_capacity(1024)).collect();
        Self {
            buffers: Mutex::new(buffers),
            num_threads,
        }
    }

    /// Create an accumulator using all available CPU cores.
    pub fn with_available_parallelism() -> Self {
        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::new(num_threads)
    }

    /// Get the number of threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Get a mutable reference to a thread's buffer.
    ///
    /// # Safety
    /// Caller must ensure thread_id < num_threads and no concurrent access
    /// to the same thread_id.
    pub fn get_buffer(
        &self,
        thread_id: usize,
    ) -> impl std::ops::DerefMut<Target = Vec<(usize, usize, f64)>> + '_ {
        struct BufferGuard<'a> {
            buffers: std::sync::MutexGuard<'a, Vec<Vec<(usize, usize, f64)>>>,
            thread_id: usize,
        }

        impl std::ops::Deref for BufferGuard<'_> {
            type Target = Vec<(usize, usize, f64)>;
            fn deref(&self) -> &Self::Target {
                &self.buffers[self.thread_id]
            }
        }

        impl std::ops::DerefMut for BufferGuard<'_> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.buffers[self.thread_id]
            }
        }

        BufferGuard {
            buffers: self.buffers.lock().unwrap(),
            thread_id,
        }
    }

    /// Clear all buffers for reuse.
    pub fn clear(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        for buf in buffers.iter_mut() {
            buf.clear();
        }
    }

    /// Merge all thread-local buffers into a single triplet list.
    pub fn merge(&self) -> Vec<(usize, usize, f64)> {
        let buffers = self.buffers.lock().unwrap();
        let total_len: usize = buffers.iter().map(|b| b.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for buf in buffers.iter() {
            result.extend_from_slice(buf);
        }
        result
    }

    /// Merge all buffers and extend an existing triplet list.
    pub fn merge_into(&self, target: &mut Vec<(usize, usize, f64)>) {
        let buffers = self.buffers.lock().unwrap();
        for buf in buffers.iter() {
            target.extend_from_slice(buf);
        }
    }
}

impl Default for ParallelTripletAccumulator {
    fn default() -> Self {
        Self::with_available_parallelism()
    }
}

/// Stamp a conductance into a triplet buffer.
#[inline]
pub fn stamp_conductance_triplets(
    triplets: &mut Vec<(usize, usize, f64)>,
    node_pos: Option<usize>,
    node_neg: Option<usize>,
    conductance: f64,
) {
    match (node_pos, node_neg) {
        (Some(i), Some(j)) => {
            triplets.push((i, i, conductance));
            triplets.push((j, j, conductance));
            triplets.push((i, j, -conductance));
            triplets.push((j, i, -conductance));
        }
        (Some(i), None) => {
            triplets.push((i, i, conductance));
        }
        (None, Some(j)) => {
            triplets.push((j, j, conductance));
        }
        (None, None) => {}
    }
}

/// Stamp a current source into RHS buffer.
///
/// Note: RHS stamping is typically done sequentially since it's just O(n) additions.
#[inline]
pub fn stamp_current_source_rhs(
    rhs: &mut [f64],
    node_pos: Option<usize>,
    node_neg: Option<usize>,
    current: f64,
) {
    if let Some(i) = node_pos {
        rhs[i] -= current;
    }
    if let Some(j) = node_neg {
        rhs[j] += current;
    }
}

/// Parallel range utility for splitting work across threads.
pub fn parallel_ranges(total: usize, num_threads: usize) -> Vec<(usize, usize)> {
    if num_threads == 0 || total == 0 {
        return vec![];
    }
    let chunk_size = total.div_ceil(num_threads);
    (0..num_threads)
        .map(|i| {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(total);
            (start, end)
        })
        .filter(|(start, end)| start < end)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_accumulator_basic() {
        let acc = ParallelTripletAccumulator::new(2);

        // Thread 0 stamps
        {
            let mut buf = acc.get_buffer(0);
            buf.push((0, 0, 1.0));
            buf.push((0, 1, 2.0));
        }

        // Thread 1 stamps
        {
            let mut buf = acc.get_buffer(1);
            buf.push((1, 0, 3.0));
            buf.push((1, 1, 4.0));
        }

        let merged = acc.merge();
        assert_eq!(merged.len(), 4);
        assert!(merged.contains(&(0, 0, 1.0)));
        assert!(merged.contains(&(0, 1, 2.0)));
        assert!(merged.contains(&(1, 0, 3.0)));
        assert!(merged.contains(&(1, 1, 4.0)));
    }

    #[test]
    fn test_parallel_accumulator_clear() {
        let acc = ParallelTripletAccumulator::new(2);

        {
            let mut buf = acc.get_buffer(0);
            buf.push((0, 0, 1.0));
        }

        acc.clear();

        let merged = acc.merge();
        assert!(merged.is_empty());
    }

    #[test]
    fn test_stamp_conductance_triplets() {
        let mut triplets = Vec::new();

        // Both nodes non-ground
        stamp_conductance_triplets(&mut triplets, Some(0), Some(1), 0.001);
        assert_eq!(triplets.len(), 4);

        triplets.clear();

        // One node grounded
        stamp_conductance_triplets(&mut triplets, Some(0), None, 0.001);
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0], (0, 0, 0.001));
    }

    #[test]
    fn test_parallel_ranges() {
        // 10 items, 3 threads
        let ranges = parallel_ranges(10, 3);
        assert_eq!(ranges, vec![(0, 4), (4, 8), (8, 10)]);

        // 10 items, 10 threads
        let ranges = parallel_ranges(10, 10);
        assert_eq!(ranges.len(), 10);

        // 3 items, 10 threads
        let ranges = parallel_ranges(3, 10);
        assert_eq!(ranges.len(), 3);

        // Edge cases
        assert!(parallel_ranges(0, 4).is_empty());
        assert!(parallel_ranges(10, 0).is_empty());
    }

    #[test]
    fn test_stamp_current_source_rhs() {
        let mut rhs = vec![0.0; 3];

        stamp_current_source_rhs(&mut rhs, Some(0), Some(1), 1e-3);
        assert!((rhs[0] - (-1e-3)).abs() < 1e-15);
        assert!((rhs[1] - 1e-3).abs() < 1e-15);
        assert!((rhs[2] - 0.0).abs() < 1e-15);
    }
}
