//! SIMD-accelerated f64 dot product and matrix-vector multiplication.
//!
//! Same SIMD dispatch pattern as the complex variants, but for real-valued
//! operations used in DC and transient analysis.

use crate::capability::SimdCapability;

/// Compute real dot product: `sum(a[i] * b[i])` for i in 0..n.
///
/// Uses runtime-dispatched SIMD when available.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn real_dot_product(a: &[f64], b: &[f64], capability: SimdCapability) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");

    match capability {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdCapability::Avx512 => unsafe { real_dot_avx512(a, b) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdCapability::Avx2 => unsafe { real_dot_avx2(a, b) },
        SimdCapability::Scalar => real_dot_scalar(a, b),
    }
}

/// Scalar implementation of real dot product.
#[inline]
pub fn real_dot_scalar(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum += ai * bi;
    }
    sum
}

/// Compute real matrix-vector product: y = A * x
///
/// # Panics
///
/// Panics if vector sizes don't match matrix dimension.
#[inline]
pub fn real_matvec(
    matrix: &[f64],
    n: usize,
    x: &[f64],
    y: &mut [f64],
    capability: SimdCapability,
) {
    assert_eq!(matrix.len(), n * n, "Matrix size mismatch");
    assert_eq!(x.len(), n, "Input vector size mismatch");
    assert_eq!(y.len(), n, "Output vector size mismatch");

    for (i, yi) in y.iter_mut().enumerate() {
        let row_start = i * n;
        let row = &matrix[row_start..row_start + n];
        *yi = real_dot_product(row, x, capability);
    }
}

/// Scalar implementation of real matrix-vector product.
#[inline]
pub fn real_matvec_scalar(matrix: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    assert_eq!(matrix.len(), n * n, "Matrix size mismatch");
    assert_eq!(x.len(), n, "Input vector size mismatch");
    assert_eq!(y.len(), n, "Output vector size mismatch");

    for (i, yi) in y.iter_mut().enumerate() {
        let mut sum = 0.0;
        let row_start = i * n;
        for j in 0..n {
            sum += matrix[row_start + j] * x[j];
        }
        *yi = sum;
    }
}

// ============================================================================
// AVX2 Implementation
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn real_dot_avx2(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let simd_len = n / 4 * 4; // Process 4 f64 per iteration
    let mut acc = _mm256_setzero_pd();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut i = 0;
    while i < simd_len {
        let a_vec = _mm256_loadu_pd(a_ptr.add(i));
        let b_vec = _mm256_loadu_pd(b_ptr.add(i));
        acc = _mm256_fmadd_pd(a_vec, b_vec, acc);
        i += 4;
    }

    // Horizontal sum of 4 f64
    let high = _mm256_extractf128_pd(acc, 1);
    let low = _mm256_castpd256_pd128(acc);
    let sum_128 = _mm_add_pd(low, high);
    let high_64 = _mm_unpackhi_pd(sum_128, sum_128);
    let sum_64 = _mm_add_sd(sum_128, high_64);

    let mut result = 0.0f64;
    _mm_store_sd(&mut result, sum_64);

    // Scalar tail
    for j in simd_len..n {
        result += a[j] * b[j];
    }
    result
}

// ============================================================================
// AVX-512 Implementation
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn real_dot_avx512(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let simd_len = n / 8 * 8; // Process 8 f64 per iteration
    let mut acc = _mm512_setzero_pd();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut i = 0;
    while i < simd_len {
        let a_vec = _mm512_loadu_pd(a_ptr.add(i));
        let b_vec = _mm512_loadu_pd(b_ptr.add(i));
        acc = _mm512_fmadd_pd(a_vec, b_vec, acc);
        i += 8;
    }

    // Reduce 512 -> 256 -> 128 -> scalar
    let sum256_low = _mm512_castpd512_pd256(acc);
    let sum256_high = _mm512_extractf64x4_pd(acc, 1);
    let sum256 = _mm256_add_pd(sum256_low, sum256_high);

    let sum128_low = _mm256_castpd256_pd128(sum256);
    let sum128_high = _mm256_extractf128_pd(sum256, 1);
    let sum128 = _mm_add_pd(sum128_low, sum128_high);

    let high_64 = _mm_unpackhi_pd(sum128, sum128);
    let sum_64 = _mm_add_sd(sum128, high_64);

    let mut result = 0.0f64;
    _mm_store_sd(&mut result, sum_64);

    // Scalar tail
    for j in simd_len..n {
        result += a[j] * b[j];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_dot_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = real_dot_scalar(&a, &b);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn simd_vs_scalar_consistency() {
        let cap = SimdCapability::detect();

        for size in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 100, 127, 128, 256] {
            let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
            let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.3).collect();

            let scalar_result = real_dot_scalar(&a, &b);
            let simd_result = real_dot_product(&a, &b, cap);

            let diff = (scalar_result - simd_result).abs();
            let scale = scalar_result.abs().max(1.0);
            assert!(
                diff <= scale * 1e-12,
                "Size {}: Scalar {} vs SIMD {} (cap={:?})",
                size,
                scalar_result,
                simd_result,
                cap
            );
        }
    }

    #[test]
    fn empty_vectors() {
        let cap = SimdCapability::detect();
        let empty: Vec<f64> = vec![];
        let result = real_dot_product(&empty, &empty, cap);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn matvec_identity() {
        let cap = SimdCapability::detect();
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 7.0];
        let mut y = vec![0.0; 2];

        real_matvec(&identity, 2, &x, &mut y, cap);
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn matvec_simd_vs_scalar() {
        let cap = SimdCapability::detect();

        for n in [1, 2, 3, 4, 8, 15, 16, 31, 32, 50, 64, 100] {
            let matrix: Vec<f64> = (0..n * n)
                .map(|i| (i as f64 * 0.1).sin())
                .collect();
            let x: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.3).cos())
                .collect();

            let mut y_scalar = vec![0.0; n];
            let mut y_simd = vec![0.0; n];

            real_matvec_scalar(&matrix, n, &x, &mut y_scalar);
            real_matvec(&matrix, n, &x, &mut y_simd, cap);

            for i in 0..n {
                let diff = (y_scalar[i] - y_simd[i]).abs();
                let scale = y_scalar[i].abs().max(1.0);
                assert!(
                    diff <= scale * 1e-12,
                    "n={}, row {}: Scalar {} vs SIMD {}",
                    n,
                    i,
                    y_scalar[i],
                    y_simd[i]
                );
            }
        }
    }

    #[test]
    fn matvec_3x3() {
        let cap = SimdCapability::detect();
        // [[1,2,3],[4,5,6],[7,8,9]] * [1,1,1] = [6,15,24]
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];

        real_matvec(&matrix, 3, &x, &mut y, cap);
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 15.0).abs() < 1e-10);
        assert!((y[2] - 24.0).abs() < 1e-10);
    }
}
