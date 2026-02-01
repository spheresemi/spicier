//! Dense matrix operators backed by flat Vec storage.
//!
//! Implements both [`RealOperator`] and [`ComplexOperator`] using
//! SIMD-accelerated matrix-vector products from `spicier-simd`.

use num_complex::Complex64 as C64;
use spicier_simd::{SimdCapability, complex_matvec, real_matvec};
use spicier_solver::operator::{ComplexOperator, RealOperator};

/// Dense NxN real (f64) matrix operator.
///
/// Uses SIMD-accelerated matrix-vector multiplication when available.
pub struct RealDenseOperator {
    n: usize,
    /// Row-major storage: data[i*n + j] = A[i][j].
    data: Vec<f64>,
    /// Cached SIMD capability.
    simd_cap: SimdCapability,
}

impl RealDenseOperator {
    /// Build from pre-computed matrix data (row-major).
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != n * n`.
    pub fn from_data(n: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            n * n,
            "Data length {} doesn't match {}x{} matrix",
            data.len(),
            n,
            n
        );
        let simd_cap = SimdCapability::detect();
        Self { n, data, simd_cap }
    }

    /// Get matrix entry (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Get the SIMD capability being used.
    pub fn simd_capability(&self) -> SimdCapability {
        self.simd_cap
    }

    /// Get the raw matrix data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

impl RealOperator for RealDenseOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        real_matvec(&self.data, self.n, x, y, self.simd_cap);
    }
}

/// Dense NxN complex (C64) matrix operator.
///
/// Uses SIMD-accelerated matrix-vector multiplication when available.
pub struct ComplexDenseOperator {
    n: usize,
    /// Row-major storage: data[i*n + j] = A[i][j].
    data: Vec<C64>,
    /// Cached SIMD capability.
    simd_cap: SimdCapability,
}

impl ComplexDenseOperator {
    /// Build from pre-computed matrix data (row-major).
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != n * n`.
    pub fn from_data(n: usize, data: Vec<C64>) -> Self {
        assert_eq!(
            data.len(),
            n * n,
            "Data length {} doesn't match {}x{} matrix",
            data.len(),
            n,
            n
        );
        let simd_cap = SimdCapability::detect();
        Self { n, data, simd_cap }
    }

    /// Get matrix entry (i, j).
    pub fn get(&self, i: usize, j: usize) -> C64 {
        self.data[i * self.n + j]
    }

    /// Get the SIMD capability being used.
    pub fn simd_capability(&self) -> SimdCapability {
        self.simd_cap
    }

    /// Get the raw matrix data.
    pub fn data(&self) -> &[C64] {
        &self.data
    }
}

impl ComplexOperator for ComplexDenseOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[C64], y: &mut [C64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);
        complex_matvec(&self.data, self.n, x, y, self.simd_cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Real operator tests ----

    #[test]
    fn real_identity() {
        let n = 3;
        let mut data = vec![0.0; n * n];
        data[0] = 1.0;
        data[4] = 1.0;
        data[8] = 1.0;
        let op = RealDenseOperator::from_data(n, data);

        let x = vec![2.0, 3.0, 4.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        assert!((y[0] - 2.0).abs() < 1e-15);
        assert!((y[1] - 3.0).abs() < 1e-15);
        assert!((y[2] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn real_3x3() {
        // [[1,2,3],[4,5,6],[7,8,9]] * [1,1,1] = [6,15,24]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let op = RealDenseOperator::from_data(3, data);

        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 15.0).abs() < 1e-10);
        assert!((y[2] - 24.0).abs() < 1e-10);
    }

    #[test]
    fn real_dim() {
        let op = RealDenseOperator::from_data(4, vec![0.0; 16]);
        assert_eq!(op.dim(), 4);
    }

    #[test]
    fn real_get() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let op = RealDenseOperator::from_data(2, data);
        assert_eq!(op.get(0, 0), 1.0);
        assert_eq!(op.get(0, 1), 2.0);
        assert_eq!(op.get(1, 0), 3.0);
        assert_eq!(op.get(1, 1), 4.0);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn real_wrong_size() {
        RealDenseOperator::from_data(3, vec![0.0; 10]);
    }

    // ---- Complex operator tests ----

    #[test]
    fn complex_identity() {
        let n = 2;
        let data = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        let op = ComplexDenseOperator::from_data(n, data);

        let x = vec![C64::new(3.0, 4.0), C64::new(5.0, 6.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op.apply(&x, &mut y);

        assert!((y[0] - x[0]).norm() < 1e-10);
        assert!((y[1] - x[1]).norm() < 1e-10);
    }

    #[test]
    fn complex_matvec_test() {
        // [[1+0i, 1+1i], [2+0i, 2+1i]] * [1+0i, 1+0i] = [2+1i, 4+1i]
        let data = vec![
            C64::new(1.0, 0.0),
            C64::new(1.0, 1.0),
            C64::new(2.0, 0.0),
            C64::new(2.0, 1.0),
        ];
        let op = ComplexDenseOperator::from_data(2, data);

        let x = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op.apply(&x, &mut y);

        assert!((y[0] - C64::new(2.0, 1.0)).norm() < 1e-10);
        assert!((y[1] - C64::new(4.0, 1.0)).norm() < 1e-10);
    }

    #[test]
    fn complex_dim() {
        let op = ComplexDenseOperator::from_data(3, vec![C64::new(0.0, 0.0); 9]);
        assert_eq!(op.dim(), 3);
    }

    #[test]
    fn complex_get() {
        let data = vec![
            C64::new(1.0, 2.0),
            C64::new(3.0, 4.0),
            C64::new(5.0, 6.0),
            C64::new(7.0, 8.0),
        ];
        let op = ComplexDenseOperator::from_data(2, data);
        assert_eq!(op.get(0, 0), C64::new(1.0, 2.0));
        assert_eq!(op.get(1, 1), C64::new(7.0, 8.0));
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn complex_wrong_size() {
        ComplexDenseOperator::from_data(3, vec![C64::new(0.0, 0.0); 10]);
    }

    // ---- Trait object tests ----

    #[test]
    fn real_as_trait_object() {
        let data = vec![2.0, 0.0, 0.0, 3.0];
        let op = RealDenseOperator::from_data(2, data);
        let op_ref: &dyn RealOperator = &op;

        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        op_ref.apply(&x, &mut y);

        assert!((y[0] - 2.0).abs() < 1e-15);
        assert!((y[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn complex_as_trait_object() {
        let data = vec![
            C64::new(2.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(3.0, 0.0),
        ];
        let op = ComplexDenseOperator::from_data(2, data);
        let op_ref: &dyn ComplexOperator = &op;

        let x = vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op_ref.apply(&x, &mut y);

        assert!((y[0] - C64::new(2.0, 0.0)).norm() < 1e-15);
        assert!((y[1] - C64::new(3.0, 0.0)).norm() < 1e-15);
    }
}
