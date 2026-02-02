//! SIMD-accelerated complex dot product and matrix-vector multiplication.
//!
//! Computes `sum(a[i] * b[i])` — the non-conjugate complex dot product.

use crate::capability::SimdCapability;
use num_complex::Complex64 as C64;

/// Compute complex dot product: `sum(a[i] * b[i])` for i in 0..n.
///
/// Uses runtime-dispatched SIMD when available.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn complex_dot_product(a: &[C64], b: &[C64], capability: SimdCapability) -> C64 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");

    match capability {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdCapability::Avx512 => unsafe { complex_dot_avx512(a, b) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdCapability::Avx2 => unsafe { complex_dot_avx2(a, b) },
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        SimdCapability::Accelerate => complex_dot_scalar(a, b), // TODO: implement with vDSP
        SimdCapability::Scalar => complex_dot_scalar(a, b),
    }
}

/// Scalar implementation of complex dot product.
#[inline]
pub fn complex_dot_scalar(a: &[C64], b: &[C64]) -> C64 {
    let mut sum = C64::new(0.0, 0.0);
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum += ai * bi;
    }
    sum
}

/// Compute complex matrix-vector product: y = A * x
///
/// Uses SIMD-accelerated dot products for each row.
///
/// # Panics
///
/// Panics if vector sizes don't match matrix dimension.
#[inline]
pub fn complex_matvec(
    matrix: &[C64],
    n: usize,
    x: &[C64],
    y: &mut [C64],
    capability: SimdCapability,
) {
    assert_eq!(matrix.len(), n * n, "Matrix size mismatch");
    assert_eq!(x.len(), n, "Input vector size mismatch");
    assert_eq!(y.len(), n, "Output vector size mismatch");

    for (i, yi) in y.iter_mut().enumerate() {
        let row_start = i * n;
        let row = &matrix[row_start..row_start + n];
        *yi = complex_dot_product(row, x, capability);
    }
}

/// Scalar implementation of matrix-vector product.
#[inline]
pub fn complex_matvec_scalar(matrix: &[C64], n: usize, x: &[C64], y: &mut [C64]) {
    assert_eq!(matrix.len(), n * n, "Matrix size mismatch");
    assert_eq!(x.len(), n, "Input vector size mismatch");
    assert_eq!(y.len(), n, "Output vector size mismatch");

    for (i, yi) in y.iter_mut().enumerate() {
        let mut sum = C64::new(0.0, 0.0);
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
unsafe fn complex_dot_avx2(a: &[C64], b: &[C64]) -> C64 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let simd_len = n / 2 * 2;
    let mut acc = _mm256_setzero_pd();

    let a_ptr = a.as_ptr() as *const f64;
    let b_ptr = b.as_ptr() as *const f64;

    let mut i = 0;
    while i < simd_len {
        let a_vec = _mm256_loadu_pd(a_ptr.add(i * 2));
        let b_vec = _mm256_loadu_pd(b_ptr.add(i * 2));

        let a_re = _mm256_shuffle_pd(a_vec, a_vec, 0b0000);
        let a_im = _mm256_shuffle_pd(a_vec, a_vec, 0b1111);
        let b_swap = _mm256_shuffle_pd(b_vec, b_vec, 0b0101);

        let prod1 = _mm256_mul_pd(a_re, b_vec);
        let prod2 = _mm256_mul_pd(a_im, b_swap);

        let result = _mm256_addsub_pd(prod1, prod2);
        acc = _mm256_add_pd(acc, result);
        i += 2;
    }

    let high = _mm256_extractf128_pd(acc, 1);
    let low = _mm256_castpd256_pd128(acc);
    let sum_128 = _mm_add_pd(low, high);

    let mut result_arr = [0.0f64; 2];
    _mm_storeu_pd(result_arr.as_mut_ptr(), sum_128);

    let mut sum = C64::new(result_arr[0], result_arr[1]);
    for j in simd_len..n {
        sum += a[j] * b[j];
    }
    sum
}

// ============================================================================
// AVX-512 Implementation
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn complex_dot_avx512(a: &[C64], b: &[C64]) -> C64 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let simd_len = n / 4 * 4;
    let mut acc = _mm512_setzero_pd();

    let a_ptr = a.as_ptr() as *const f64;
    let b_ptr = b.as_ptr() as *const f64;

    let neg_mask = _mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);

    let mut i = 0;
    while i < simd_len {
        let a_vec = _mm512_loadu_pd(a_ptr.add(i * 2));
        let b_vec = _mm512_loadu_pd(b_ptr.add(i * 2));

        let a_re = _mm512_shuffle_pd(a_vec, a_vec, 0b00000000);
        let a_im = _mm512_shuffle_pd(a_vec, a_vec, 0b11111111);
        let b_swap = _mm512_shuffle_pd(b_vec, b_vec, 0b01010101);

        let prod1 = _mm512_mul_pd(a_re, b_vec);
        let prod2 = _mm512_mul_pd(a_im, b_swap);

        let prod2_signed = _mm512_mul_pd(prod2, neg_mask);
        let result = _mm512_add_pd(prod1, prod2_signed);

        acc = _mm512_add_pd(acc, result);
        i += 4;
    }

    let sum256_low = _mm512_castpd512_pd256(acc);
    let sum256_high = _mm512_extractf64x4_pd(acc, 1);
    let sum256 = _mm256_add_pd(sum256_low, sum256_high);

    let sum128_low = _mm256_castpd256_pd128(sum256);
    let sum128_high = _mm256_extractf128_pd(sum256, 1);
    let sum128 = _mm_add_pd(sum128_low, sum128_high);

    let mut result_arr = [0.0f64; 2];
    _mm_storeu_pd(result_arr.as_mut_ptr(), sum128);

    let mut sum = C64::new(result_arr[0], result_arr[1]);
    for j in simd_len..n {
        sum += a[j] * b[j];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: C64, b: C64, rel_tol: f64, abs_tol: f64) -> bool {
        let diff = (a - b).norm();
        let max_norm = a.norm().max(b.norm());
        diff <= abs_tol || diff <= rel_tol * max_norm
    }

    #[test]
    fn scalar_dot_product_basic() {
        let a = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let b = vec![C64::new(5.0, 6.0), C64::new(7.0, 8.0)];

        // (1+2i)(5+6i) = -7+16i
        // (3+4i)(7+8i) = -11+52i
        // Sum = -18+68i
        let expected = C64::new(-18.0, 68.0);
        let result = complex_dot_scalar(&a, &b);
        assert!(
            (result - expected).norm() < 1e-10,
            "Expected {:?}, got {:?}",
            expected,
            result
        );
    }

    #[test]
    fn simd_vs_scalar_consistency() {
        let cap = SimdCapability::detect();

        for size in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 100, 127, 128] {
            let a: Vec<C64> = (0..size)
                .map(|i| C64::new(i as f64 * 0.1, i as f64 * 0.2))
                .collect();
            let b: Vec<C64> = (0..size)
                .map(|i| C64::new((size - i) as f64 * 0.3, i as f64 * 0.05))
                .collect();

            let scalar_result = complex_dot_scalar(&a, &b);
            let simd_result = complex_dot_product(&a, &b, cap);

            assert!(
                approx_eq(scalar_result, simd_result, 1e-12, 1e-14),
                "Size {}: Scalar {:?} vs SIMD {:?} ({:?})",
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
        let empty: Vec<C64> = vec![];
        let result = complex_dot_product(&empty, &empty, cap);
        assert_eq!(result, C64::new(0.0, 0.0));
    }

    #[test]
    fn matvec_identity() {
        let cap = SimdCapability::detect();
        let identity = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        let x = vec![C64::new(3.0, 4.0), C64::new(5.0, 6.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];

        complex_matvec(&identity, 2, &x, &mut y, cap);
        assert!((y[0] - x[0]).norm() < 1e-10);
        assert!((y[1] - x[1]).norm() < 1e-10);
    }

    #[test]
    fn matvec_simd_vs_scalar() {
        let cap = SimdCapability::detect();

        for n in [1, 2, 3, 4, 8, 15, 16, 31, 32, 50, 64, 100] {
            let matrix: Vec<C64> = (0..n * n)
                .map(|i| C64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
                .collect();
            let x: Vec<C64> = (0..n)
                .map(|i| C64::new((i as f64 * 0.3).cos(), (i as f64 * 0.4).sin()))
                .collect();

            let mut y_scalar = vec![C64::new(0.0, 0.0); n];
            let mut y_simd = vec![C64::new(0.0, 0.0); n];

            complex_matvec_scalar(&matrix, n, &x, &mut y_scalar);
            complex_matvec(&matrix, n, &x, &mut y_simd, cap);

            for i in 0..n {
                assert!(
                    approx_eq(y_scalar[i], y_simd[i], 1e-12, 1e-14),
                    "n={}, row {}: Scalar {:?} vs SIMD {:?}",
                    n,
                    i,
                    y_scalar[i],
                    y_simd[i]
                );
            }
        }
    }
}
