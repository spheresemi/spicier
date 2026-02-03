//! Memory layout optimization for GPU batched operations.
//!
//! This module provides utilities for efficient GPU memory layout:
//! - Alignment padding for coalesced memory access
//! - Efficient f64↔f32 conversion with layout transformation
//! - Contiguous batch storage with proper stride calculations
//!
//! # GPU Memory Access Patterns
//!
//! GPUs achieve best performance when threads in a warp/wavefront access
//! consecutive memory addresses (coalesced access). For NVIDIA, warp size
//! is 32; for AMD/Apple, wavefront size is typically 32 or 64.
//!
//! # Example
//!
//! ```
//! use spicier_batched_sweep::batch_layout::{BatchLayout, pack_matrices_f32};
//!
//! let n = 10;  // 10×10 matrices
//! let batch_size = 100;
//!
//! // Create layout with warp-aligned rows
//! let layout = BatchLayout::new(n, batch_size);
//! assert!(layout.padded_row_stride() >= n);
//! assert_eq!(layout.padded_row_stride() % 32, 0);
//!
//! // Pack f64 column-major matrices into f32 row-major with padding
//! let matrices_f64: Vec<f64> = vec![1.0; batch_size * n * n];
//! let packed = pack_matrices_f32(&matrices_f64, n, batch_size, &layout);
//! assert_eq!(packed.len(), layout.total_matrix_elements());
//! ```

/// GPU warp/wavefront size for memory alignment.
/// 32 is optimal for NVIDIA and common for Apple Silicon.
pub const WARP_SIZE: usize = 32;

/// Minimum alignment for GPU buffers (in bytes).
/// 256 bytes is required by many GPU APIs.
pub const BUFFER_ALIGNMENT: usize = 256;

/// Describes the memory layout for a batch of matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLayout {
    /// Original matrix dimension (n×n).
    pub n: usize,
    /// Number of matrices in the batch.
    pub batch_size: usize,
    /// Padded row stride (elements per row including padding).
    /// Aligned to WARP_SIZE for coalesced access.
    padded_row_stride: usize,
    /// Padded matrix size (elements per matrix including row padding).
    padded_matrix_size: usize,
}

impl BatchLayout {
    /// Create a new batch layout with optimal alignment.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension (n×n matrices)
    /// * `batch_size` - Number of matrices in the batch
    pub fn new(n: usize, batch_size: usize) -> Self {
        let padded_row_stride = Self::align_to_warp(n);
        let padded_matrix_size = padded_row_stride * n;

        Self {
            n,
            batch_size,
            padded_row_stride,
            padded_matrix_size,
        }
    }

    /// Create a layout without padding (for comparison/testing).
    pub fn unpadded(n: usize, batch_size: usize) -> Self {
        Self {
            n,
            batch_size,
            padded_row_stride: n,
            padded_matrix_size: n * n,
        }
    }

    /// Align a size to the next multiple of WARP_SIZE.
    #[inline]
    pub fn align_to_warp(size: usize) -> usize {
        size.div_ceil(WARP_SIZE) * WARP_SIZE
    }

    /// Get the padded row stride.
    #[inline]
    pub fn padded_row_stride(&self) -> usize {
        self.padded_row_stride
    }

    /// Get the padded matrix size (elements per matrix).
    #[inline]
    pub fn padded_matrix_size(&self) -> usize {
        self.padded_matrix_size
    }

    /// Total elements needed for all matrices.
    #[inline]
    pub fn total_matrix_elements(&self) -> usize {
        self.padded_matrix_size * self.batch_size
    }

    /// Total elements needed for all RHS vectors.
    #[inline]
    pub fn total_rhs_elements(&self) -> usize {
        // RHS vectors can use unpadded n (they're accessed per-workgroup, not coalesced)
        self.n * self.batch_size
    }

    /// Get the offset for matrix element [batch_idx][row][col].
    #[inline]
    pub fn matrix_offset(&self, batch_idx: usize, row: usize, col: usize) -> usize {
        batch_idx * self.padded_matrix_size + row * self.padded_row_stride + col
    }

    /// Get the offset for RHS element [batch_idx][i].
    #[inline]
    pub fn rhs_offset(&self, batch_idx: usize, i: usize) -> usize {
        batch_idx * self.n + i
    }

    /// Check if padding is actually being used.
    #[inline]
    pub fn has_padding(&self) -> bool {
        self.padded_row_stride > self.n
    }

    /// Get the amount of padding per row (in elements).
    #[inline]
    pub fn row_padding(&self) -> usize {
        self.padded_row_stride - self.n
    }
}

/// Pack f64 column-major matrices into f32 row-major format with alignment padding.
///
/// This function performs three operations in a single pass:
/// 1. f64 → f32 conversion
/// 2. Column-major → row-major transpose
/// 3. Row padding for GPU alignment
///
/// # Arguments
/// * `matrices` - Input matrices in column-major order (batch_size × n × n)
/// * `n` - Matrix dimension
/// * `batch_size` - Number of matrices
/// * `layout` - Target memory layout
///
/// # Returns
/// Packed f32 data in row-major order with padding
pub fn pack_matrices_f32(
    matrices: &[f64],
    n: usize,
    batch_size: usize,
    layout: &BatchLayout,
) -> Vec<f32> {
    debug_assert_eq!(matrices.len(), batch_size * n * n);
    debug_assert_eq!(layout.n, n);
    debug_assert_eq!(layout.batch_size, batch_size);

    let mut packed = vec![0.0f32; layout.total_matrix_elements()];

    for batch_idx in 0..batch_size {
        let src_offset = batch_idx * n * n;
        for row in 0..n {
            for col in 0..n {
                // Source: column-major [col * n + row]
                // Dest: row-major with padding [row * stride + col]
                let src_idx = src_offset + col * n + row;
                let dst_idx = layout.matrix_offset(batch_idx, row, col);
                packed[dst_idx] = matrices[src_idx] as f32;
            }
        }
        // Padding elements remain as 0.0
    }

    packed
}

/// Pack f64 RHS vectors into f32 format.
///
/// # Arguments
/// * `rhs` - Input RHS vectors (batch_size × n)
/// * `n` - Vector dimension
/// * `batch_size` - Number of vectors
///
/// # Returns
/// Packed f32 data
pub fn pack_rhs_f32(rhs: &[f64], n: usize, batch_size: usize) -> Vec<f32> {
    debug_assert_eq!(rhs.len(), batch_size * n);
    rhs.iter().map(|&v| v as f32).collect()
}

/// Unpack f32 solutions back to f64 format.
///
/// # Arguments
/// * `solutions_f32` - Solutions in f32 format (batch_size × n)
/// * `n` - Vector dimension
/// * `batch_size` - Number of vectors
///
/// # Returns
/// Solutions in f64 format
pub fn unpack_solutions_f64(solutions_f32: &[f32], n: usize, batch_size: usize) -> Vec<f64> {
    debug_assert_eq!(solutions_f32.len(), batch_size * n);
    solutions_f32.iter().map(|&v| v as f64).collect()
}

/// Information about padding overhead.
#[derive(Debug, Clone, Copy)]
pub struct PaddingStats {
    /// Useful elements per matrix.
    pub useful_elements: usize,
    /// Total elements per matrix (including padding).
    pub total_elements: usize,
    /// Padding overhead ratio (0.0 = no overhead, 1.0 = 100% overhead).
    pub overhead_ratio: f64,
    /// Total bytes wasted on padding (for f32 data).
    pub wasted_bytes: usize,
}

impl BatchLayout {
    /// Calculate padding statistics.
    pub fn padding_stats(&self) -> PaddingStats {
        let useful_elements = self.n * self.n;
        let total_elements = self.padded_matrix_size;
        let padding_elements = total_elements - useful_elements;

        PaddingStats {
            useful_elements,
            total_elements,
            overhead_ratio: padding_elements as f64 / useful_elements as f64,
            wasted_bytes: padding_elements * self.batch_size * std::mem::size_of::<f32>(),
        }
    }
}

/// Shader constants for use in WGSL/CUDA code generation.
#[derive(Debug, Clone, Copy)]
pub struct ShaderLayoutConstants {
    /// Matrix dimension.
    pub n: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Padded row stride.
    pub row_stride: u32,
    /// Padded matrix size.
    pub matrix_stride: u32,
}

impl From<&BatchLayout> for ShaderLayoutConstants {
    fn from(layout: &BatchLayout) -> Self {
        Self {
            n: layout.n as u32,
            batch_size: layout.batch_size as u32,
            row_stride: layout.padded_row_stride as u32,
            matrix_stride: layout.padded_matrix_size as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_warp() {
        assert_eq!(BatchLayout::align_to_warp(1), 32);
        assert_eq!(BatchLayout::align_to_warp(31), 32);
        assert_eq!(BatchLayout::align_to_warp(32), 32);
        assert_eq!(BatchLayout::align_to_warp(33), 64);
        assert_eq!(BatchLayout::align_to_warp(64), 64);
        assert_eq!(BatchLayout::align_to_warp(100), 128);
    }

    #[test]
    fn test_batch_layout_creation() {
        let layout = BatchLayout::new(10, 100);
        assert_eq!(layout.n, 10);
        assert_eq!(layout.batch_size, 100);
        assert_eq!(layout.padded_row_stride(), 32); // 10 rounded up to 32
        assert_eq!(layout.padded_matrix_size(), 32 * 10); // 320 elements per matrix
        assert!(layout.has_padding());
        assert_eq!(layout.row_padding(), 22); // 32 - 10
    }

    #[test]
    fn test_batch_layout_no_padding_needed() {
        let layout = BatchLayout::new(32, 50);
        assert_eq!(layout.padded_row_stride(), 32);
        assert!(!layout.has_padding());
        assert_eq!(layout.row_padding(), 0);
    }

    #[test]
    fn test_batch_layout_unpadded() {
        let layout = BatchLayout::unpadded(10, 100);
        assert_eq!(layout.padded_row_stride(), 10);
        assert!(!layout.has_padding());
    }

    #[test]
    fn test_matrix_offset() {
        let layout = BatchLayout::new(10, 3);
        // First matrix, first element
        assert_eq!(layout.matrix_offset(0, 0, 0), 0);
        // First matrix, element [1][2]
        assert_eq!(layout.matrix_offset(0, 1, 2), 32 + 2); // row_stride=32
        // Second matrix, first element
        assert_eq!(layout.matrix_offset(1, 0, 0), 320); // matrix_size=320
    }

    #[test]
    fn test_pack_matrices_identity() {
        // 2×2 identity matrix in column-major: [1, 0, 0, 1]
        let matrices = vec![1.0, 0.0, 0.0, 1.0];
        let layout = BatchLayout::new(2, 1);

        let packed = pack_matrices_f32(&matrices, 2, 1, &layout);

        // Row-major with padding to 32: [1, 0, 0...0 (30 zeros), 0, 1, 0...0 (30 zeros)]
        assert_eq!(packed.len(), 32 * 2); // 2 rows × 32 stride
        assert_eq!(packed[0], 1.0); // [0][0]
        assert_eq!(packed[1], 0.0); // [0][1]
        assert_eq!(packed[32], 0.0); // [1][0]
        assert_eq!(packed[33], 1.0); // [1][1]
    }

    #[test]
    fn test_pack_matrices_transpose() {
        // 2×2 matrix [[1, 2], [3, 4]] in column-major: [1, 3, 2, 4]
        let matrices = vec![1.0, 3.0, 2.0, 4.0];
        let layout = BatchLayout::unpadded(2, 1);

        let packed = pack_matrices_f32(&matrices, 2, 1, &layout);

        // Row-major: [1, 2, 3, 4]
        assert_eq!(packed, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pack_matrices_batch() {
        // Two 2×2 matrices
        // M0 = [[1, 2], [3, 4]] column-major: [1, 3, 2, 4]
        // M1 = [[5, 6], [7, 8]] column-major: [5, 7, 6, 8]
        let matrices = vec![
            1.0, 3.0, 2.0, 4.0, // M0
            5.0, 7.0, 6.0, 8.0, // M1
        ];
        let layout = BatchLayout::unpadded(2, 2);

        let packed = pack_matrices_f32(&matrices, 2, 2, &layout);

        // Row-major: [1, 2, 3, 4, 5, 6, 7, 8]
        assert_eq!(packed, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_padding_stats() {
        let layout = BatchLayout::new(10, 100);
        let stats = layout.padding_stats();

        assert_eq!(stats.useful_elements, 100); // 10×10
        assert_eq!(stats.total_elements, 320); // 32×10
        assert!((stats.overhead_ratio - 2.2).abs() < 0.01); // 220/100 = 2.2
        assert_eq!(stats.wasted_bytes, 220 * 100 * 4); // 220 padding elements × 100 matrices × 4 bytes
    }

    #[test]
    fn test_rhs_packing() {
        let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let packed = pack_rhs_f32(&rhs, 3, 2);
        assert_eq!(packed, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_solution_unpacking() {
        let solutions_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let unpacked = unpack_solutions_f64(&solutions_f32, 2, 2);
        assert_eq!(unpacked, vec![1.0f64, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_shader_constants() {
        let layout = BatchLayout::new(10, 100);
        let constants: ShaderLayoutConstants = (&layout).into();

        assert_eq!(constants.n, 10);
        assert_eq!(constants.batch_size, 100);
        assert_eq!(constants.row_stride, 32);
        assert_eq!(constants.matrix_stride, 320);
    }
}
