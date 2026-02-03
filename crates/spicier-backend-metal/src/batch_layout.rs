//! Memory layout utilities for GPU batched operations.
//!
//! Provides alignment and padding for optimal GPU memory access patterns.

/// GPU warp/wavefront size for memory alignment.
pub const WARP_SIZE: usize = 32;

/// Describes the memory layout for a batch of matrices.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct BatchLayout {
    /// Original matrix dimension (n×n).
    pub n: usize,
    /// Number of matrices in the batch.
    pub batch_size: usize,
    /// Padded row stride (elements per row including padding).
    padded_row_stride: usize,
    /// Padded matrix size (elements per matrix including row padding).
    padded_matrix_size: usize,
}

impl BatchLayout {
    /// Create a new batch layout with optimal alignment.
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

    /// Create a layout without padding.
    #[allow(dead_code)]
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
        (size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE
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

    /// Get the offset for matrix element [batch_idx][row][col].
    #[inline]
    pub fn matrix_offset(&self, batch_idx: usize, row: usize, col: usize) -> usize {
        batch_idx * self.padded_matrix_size + row * self.padded_row_stride + col
    }

    /// Check if padding is actually being used.
    #[allow(dead_code)]
    #[inline]
    pub fn has_padding(&self) -> bool {
        self.padded_row_stride > self.n
    }
}

/// Pack f64 column-major matrices into f32 row-major format with alignment padding.
pub fn pack_matrices_f32(
    matrices: &[f64],
    n: usize,
    batch_size: usize,
    layout: &BatchLayout,
) -> Vec<f32> {
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
    }

    packed
}

/// Pack f64 RHS vectors into f32 format.
pub fn pack_rhs_f32(rhs: &[f64]) -> Vec<f32> {
    rhs.iter().map(|&v| v as f32).collect()
}

/// Unpack f32 solutions back to f64 format.
pub fn unpack_solutions_f64(solutions_f32: &[f32]) -> Vec<f64> {
    solutions_f32.iter().map(|&v| v as f64).collect()
}
