//! Backend-agnostic operator traits for matrix-vector products.
//!
//! These traits abstract the linear operator (matvec) so that CPU, CUDA, Metal,
//! and other backends can provide implementations. Two separate traits handle
//! real-valued (DC/transient) and complex-valued (AC) operations.

use num_complex::Complex64 as C64;

/// A linear operator that computes y = A * x for real (f64) vectors.
///
/// Used for DC operating point and transient analysis where the MNA system
/// is real-valued.
pub trait RealOperator: Send + Sync {
    /// Dimension of the operator (N x N).
    fn dim(&self) -> usize;

    /// Apply the operator: y = A * x.
    ///
    /// `x` and `y` are f64 vectors of length `dim()`.
    fn apply(&self, x: &[f64], y: &mut [f64]);
}

/// A linear operator that computes y = A * x for complex (C64) vectors.
///
/// Used for AC small-signal analysis and iterative solvers like GMRES.
pub trait ComplexOperator: Send + Sync {
    /// Dimension of the operator (N x N).
    fn dim(&self) -> usize;

    /// Apply the operator: y = A * x.
    ///
    /// `x` and `y` are complex vectors of length `dim()`.
    fn apply(&self, x: &[C64], y: &mut [C64]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple diagonal real operator for testing.
    struct DiagReal {
        diag: Vec<f64>,
    }

    impl RealOperator for DiagReal {
        fn dim(&self) -> usize {
            self.diag.len()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) {
            for i in 0..self.diag.len() {
                y[i] = self.diag[i] * x[i];
            }
        }
    }

    /// Simple diagonal complex operator for testing.
    struct DiagComplex {
        diag: Vec<C64>,
    }

    impl ComplexOperator for DiagComplex {
        fn dim(&self) -> usize {
            self.diag.len()
        }

        fn apply(&self, x: &[C64], y: &mut [C64]) {
            for i in 0..self.diag.len() {
                y[i] = self.diag[i] * x[i];
            }
        }
    }

    #[test]
    fn real_operator_basic() {
        let op = DiagReal {
            diag: vec![2.0, 3.0, 4.0],
        };
        assert_eq!(op.dim(), 3);

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        assert!((y[0] - 2.0).abs() < 1e-15);
        assert!((y[1] - 6.0).abs() < 1e-15);
        assert!((y[2] - 12.0).abs() < 1e-15);
    }

    #[test]
    fn complex_operator_basic() {
        let op = DiagComplex {
            diag: vec![C64::new(1.0, 1.0), C64::new(2.0, 0.0)],
        };
        assert_eq!(op.dim(), 2);

        let x = vec![C64::new(1.0, 0.0), C64::new(0.0, 1.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        op.apply(&x, &mut y);

        // (1+i)*1 = 1+i
        assert!((y[0] - C64::new(1.0, 1.0)).norm() < 1e-15);
        // 2*(i) = 2i
        assert!((y[1] - C64::new(0.0, 2.0)).norm() < 1e-15);
    }

    #[test]
    fn operator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DiagReal>();
        assert_send_sync::<DiagComplex>();
    }
}
