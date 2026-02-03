//! Buffer pooling for GPU memory reuse.
//!
//! This module provides a buffer pool that pre-allocates and reuses GPU buffers
//! to avoid repeated allocation overhead during chunked sweep operations.
//!
//! # Usage
//!
//! ```ignore
//! let mut pool = BufferPool::new(ctx.clone());
//!
//! // Acquire a buffer
//! let buffer = pool.acquire(1024 * 1024, wgpu::BufferUsages::STORAGE);
//!
//! // ... use buffer ...
//!
//! // Return buffer for reuse
//! pool.release(buffer);
//! ```

use crate::context::WgpuContext;
use std::sync::Arc;

/// Size buckets for buffer pooling (in bytes).
const BUCKET_SIZES: [u64; 6] = [
    1024,              // 1 KB
    64 * 1024,         // 64 KB
    1024 * 1024,       // 1 MB
    16 * 1024 * 1024,  // 16 MB
    64 * 1024 * 1024,  // 64 MB
    256 * 1024 * 1024, // 256 MB
];

/// A pooled buffer with metadata.
struct PooledBuffer {
    /// The wgpu buffer.
    buffer: wgpu::Buffer,
    /// Actual size of the buffer.
    size: u64,
    /// Usage flags the buffer was created with.
    usage: wgpu::BufferUsages,
}

/// Buffer pool for reusing GPU buffers.
///
/// Buffers are organized into size buckets for efficient lookup.
/// When a buffer is requested, the pool returns an existing buffer
/// of sufficient size if available, or creates a new one.
pub struct BufferPool {
    ctx: Arc<WgpuContext>,
    /// Available buffers organized by bucket index.
    buckets: Vec<Vec<PooledBuffer>>,
    /// Statistics: total allocations.
    total_allocations: usize,
    /// Statistics: reused allocations.
    reused_allocations: usize,
}

impl BufferPool {
    /// Create a new buffer pool.
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        Self {
            ctx,
            buckets: (0..BUCKET_SIZES.len()).map(|_| Vec::new()).collect(),
            total_allocations: 0,
            reused_allocations: 0,
        }
    }

    /// Find the bucket index for a given size.
    fn bucket_index(size: u64) -> usize {
        for (i, &bucket_size) in BUCKET_SIZES.iter().enumerate() {
            if size <= bucket_size {
                return i;
            }
        }
        BUCKET_SIZES.len() - 1
    }

    /// Get the bucket size for a given index.
    fn bucket_size(index: usize) -> u64 {
        BUCKET_SIZES[index.min(BUCKET_SIZES.len() - 1)]
    }

    /// Acquire a buffer of at least `size` bytes with the specified usage.
    ///
    /// If a suitable buffer exists in the pool, it is returned.
    /// Otherwise, a new buffer is allocated.
    ///
    /// # Arguments
    /// * `size` - Minimum size in bytes
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    /// A buffer of at least the requested size
    pub fn acquire(&mut self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.total_allocations += 1;

        let bucket_idx = Self::bucket_index(size);

        // Look for a suitable buffer in this bucket or larger
        for idx in bucket_idx..self.buckets.len() {
            // Find a buffer with matching or compatible usage
            if let Some(pos) = self.buckets[idx]
                .iter()
                .position(|b| b.size >= size && b.usage.contains(usage))
            {
                self.reused_allocations += 1;
                return self.buckets[idx].remove(pos).buffer;
            }
        }

        // No suitable buffer found, allocate a new one
        let alloc_size = Self::bucket_size(bucket_idx).max(size);
        self.allocate_buffer(alloc_size, usage)
    }

    /// Acquire a buffer with a specific label.
    pub fn acquire_labeled(
        &mut self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        // For labeled buffers, we always allocate new to preserve the label
        // This is mainly for debugging purposes
        self.total_allocations += 1;
        self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Allocate a new buffer.
    fn allocate_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Release a buffer back to the pool for reuse.
    ///
    /// The buffer's size and usage are preserved for future matching.
    ///
    /// # Arguments
    /// * `buffer` - The buffer to release
    pub fn release(&mut self, buffer: wgpu::Buffer) {
        let size = buffer.size();
        let usage = buffer.usage();
        let bucket_idx = Self::bucket_index(size);

        self.buckets[bucket_idx].push(PooledBuffer {
            buffer,
            size,
            usage,
        });
    }

    /// Pre-warm the pool with expected buffer sizes.
    ///
    /// This is useful when you know in advance what buffers you'll need,
    /// allowing allocation to happen upfront rather than during processing.
    ///
    /// # Arguments
    /// * `specs` - List of (size, usage, count) tuples
    pub fn prewarm(&mut self, specs: &[(u64, wgpu::BufferUsages, usize)]) {
        for &(size, usage, count) in specs {
            let bucket_idx = Self::bucket_index(size);
            let alloc_size = Self::bucket_size(bucket_idx).max(size);

            for _ in 0..count {
                let buffer = self.allocate_buffer(alloc_size, usage);
                self.buckets[bucket_idx].push(PooledBuffer {
                    buffer,
                    size: alloc_size,
                    usage,
                });
            }
        }
    }

    /// Clear all pooled buffers, freeing GPU memory.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }

    /// Get the total number of buffers currently in the pool.
    pub fn pooled_count(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// Get the total memory used by pooled buffers.
    pub fn pooled_memory(&self) -> u64 {
        self.buckets
            .iter()
            .flat_map(|b| b.iter())
            .map(|b| b.size)
            .sum()
    }

    /// Get allocation statistics.
    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            total_allocations: self.total_allocations,
            reused_allocations: self.reused_allocations,
            pooled_buffers: self.pooled_count(),
            pooled_memory: self.pooled_memory(),
        }
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.total_allocations = 0;
        self.reused_allocations = 0;
    }
}

/// Buffer pool statistics.
#[derive(Clone, Debug, Default)]
pub struct BufferPoolStats {
    /// Total number of buffer acquisitions.
    pub total_allocations: usize,
    /// Number of acquisitions satisfied from pool.
    pub reused_allocations: usize,
    /// Number of buffers currently in pool.
    pub pooled_buffers: usize,
    /// Total memory in pooled buffers (bytes).
    pub pooled_memory: u64,
}

impl BufferPoolStats {
    /// Get the reuse ratio (0.0 to 1.0).
    pub fn reuse_ratio(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.reused_allocations as f64 / self.total_allocations as f64
        }
    }
}

/// A scoped buffer that automatically returns to the pool when dropped.
///
/// This is useful for ensuring buffers are returned even in error paths.
pub struct ScopedBuffer<'a> {
    pool: &'a mut BufferPool,
    buffer: Option<wgpu::Buffer>,
}

impl<'a> ScopedBuffer<'a> {
    /// Create a new scoped buffer.
    pub fn new(pool: &'a mut BufferPool, size: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer = pool.acquire(size, usage);
        Self {
            pool,
            buffer: Some(buffer),
        }
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().unwrap()
    }

    /// Take ownership of the buffer without returning it to the pool.
    pub fn take(mut self) -> wgpu::Buffer {
        self.buffer.take().unwrap()
    }
}

impl<'a> Drop for ScopedBuffer<'a> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

impl<'a> std::ops::Deref for ScopedBuffer<'a> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_index() {
        assert_eq!(BufferPool::bucket_index(100), 0); // 100B -> 1KB bucket
        assert_eq!(BufferPool::bucket_index(1024), 0); // 1KB -> 1KB bucket
        assert_eq!(BufferPool::bucket_index(2000), 1); // 2KB -> 64KB bucket
        assert_eq!(BufferPool::bucket_index(100_000), 2); // 100KB -> 1MB bucket
        assert_eq!(BufferPool::bucket_index(10_000_000), 3); // 10MB -> 16MB bucket
        assert_eq!(BufferPool::bucket_index(50_000_000), 4); // 50MB -> 64MB bucket
        assert_eq!(BufferPool::bucket_index(100_000_000), 5); // 100MB -> 256MB bucket
    }

    #[test]
    fn test_bucket_size() {
        assert_eq!(BufferPool::bucket_size(0), 1024);
        assert_eq!(BufferPool::bucket_size(1), 64 * 1024);
        assert_eq!(BufferPool::bucket_size(2), 1024 * 1024);
    }

    #[test]
    fn test_stats_reuse_ratio() {
        let stats = BufferPoolStats {
            total_allocations: 100,
            reused_allocations: 75,
            pooled_buffers: 0,
            pooled_memory: 0,
        };
        assert!((stats.reuse_ratio() - 0.75).abs() < 0.001);

        let empty_stats = BufferPoolStats::default();
        assert_eq!(empty_stats.reuse_ratio(), 0.0);
    }
}
