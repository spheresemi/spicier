//! SIMD-accelerated conjugate dot product for complex vectors.
//!
//! Computes `<a, b> = sum_i conj(a[i]) * b[i]` — the standard complex inner product.

use crate::capability::SimdCapability;
use num_complex::Complex64 as C64;

/// Compute the conjugate dot product (complex inner product):
///
///   `<a, b> = sum_i conj(a[i]) * b[i]`
///
/// Uses runtime-dispatched SIMD when available on x86/x86_64.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn complex_conjugate_dot_product(a: &[C64], b: &[C64], capability: SimdCapability) -> C64 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");

    match capability {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdCapability::Avx512 => {
            // SAFETY: AVX-512 availability verified via is_x86_feature_detected in detect()
            unsafe { conjugate_dot_avx512(a, b) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdCapability::Avx2 => {
            // SAFETY: AVX2+FMA availability verified via is_x86_feature_detected in detect()
            unsafe { conjugate_dot_avx2(a, b) }
        }
        #[cfg(all(target_os = "macos", feature = "accelerate"))]
        SimdCapability::Accelerate => conjugate_dot_scalar(a, b), // TODO: implement with vDSP
        SimdCapability::Scalar => conjugate_dot_scalar(a, b),
    }
}

/// Scalar implementation of conjugate dot product.
#[inline]
pub fn conjugate_dot_scalar(a: &[C64], b: &[C64]) -> C64 {
    let mut sum = C64::new(0.0, 0.0);
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum += ai.conj() * bi;
    }
    sum
}

// ============================================================================
// AVX2 Implementation
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn conjugate_dot_avx2(a: &[C64], b: &[C64]) -> C64 {
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

        // For conjugate: negate prod2, then addsub gives conjugate pattern
        let neg_prod2 = _mm256_sub_pd(_mm256_setzero_pd(), prod2);
        let result = _mm256_addsub_pd(prod1, neg_prod2);

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
        sum += a[j].conj() * b[j];
    }
    sum
}

// ============================================================================
// AVX-512 Implementation
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn conjugate_dot_avx512(a: &[C64], b: &[C64]) -> C64 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let simd_len = n / 4 * 4;
    let mut acc = _mm512_setzero_pd();

    let a_ptr = a.as_ptr() as *const f64;
    let b_ptr = b.as_ptr() as *const f64;

    let conj_sign = _mm512_set_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    let mut i = 0;
    while i < simd_len {
        let a_vec = _mm512_loadu_pd(a_ptr.add(i * 2));
        let b_vec = _mm512_loadu_pd(b_ptr.add(i * 2));

        let a_re = _mm512_shuffle_pd(a_vec, a_vec, 0b00000000);
        let a_im = _mm512_shuffle_pd(a_vec, a_vec, 0b11111111);
        let b_swap = _mm512_shuffle_pd(b_vec, b_vec, 0b01010101);

        let prod1 = _mm512_mul_pd(a_re, b_vec);
        let prod2 = _mm512_mul_pd(a_im, b_swap);

        let prod2_signed = _mm512_mul_pd(prod2, conj_sign);
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
        sum += a[j].conj() * b[j];
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
    fn scalar_conjugate_dot_basic() {
        let a = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
        let b = vec![C64::new(5.0, 6.0), C64::new(7.0, 8.0)];

        // conj(1+2i)*(5+6i) = (1-2i)(5+6i) = 17-4i
        // conj(3+4i)*(7+8i) = (3-4i)(7+8i) = 53-4i
        // Sum = 70-8i
        let expected = C64::new(70.0, -8.0);
        let result = conjugate_dot_scalar(&a, &b);
        assert!(
            (result - expected).norm() < 1e-10,
            "Expected {:?}, got {:?}",
            expected,
            result
        );
    }

    #[test]
    fn conjugate_dot_self_is_real() {
        let v = vec![C64::new(1.0, 2.0), C64::new(3.0, -1.0), C64::new(0.5, 0.7)];
        let cap = SimdCapability::detect();
        let result = complex_conjugate_dot_product(&v, &v, cap);

        assert!(result.im.abs() < 1e-14);
        assert!(result.re > 0.0);
    }

    #[test]
    fn simd_vs_scalar_consistency() {
        let cap = SimdCapability::detect();

        for size in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 100, 127, 128] {
            let a: Vec<C64> = (0..size)
                .map(|i| C64::new(i as f64 * 0.1, -(i as f64) * 0.2))
                .collect();
            let b: Vec<C64> = (0..size)
                .map(|i| C64::new((size - i) as f64 * 0.3, i as f64 * 0.05))
                .collect();

            let scalar_result = conjugate_dot_scalar(&a, &b);
            let simd_result = complex_conjugate_dot_product(&a, &b, cap);

            assert!(
                approx_eq(scalar_result, simd_result, 1e-12, 1e-14),
                "Size {}: Scalar {:?} vs SIMD {:?} (cap={:?})",
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
        let result = complex_conjugate_dot_product(&empty, &empty, cap);
        assert_eq!(result, C64::new(0.0, 0.0));
    }

    #[test]
    fn pure_real_vectors() {
        let cap = SimdCapability::detect();
        let a = vec![C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(3.0, 0.0)];
        let b = vec![C64::new(4.0, 0.0), C64::new(5.0, 0.0), C64::new(6.0, 0.0)];
        let expected = C64::new(32.0, 0.0);
        let result = complex_conjugate_dot_product(&a, &b, cap);
        assert!(
            (result - expected).norm() < 1e-10,
            "Expected {:?}, got {:?}",
            expected,
            result
        );
    }
}
