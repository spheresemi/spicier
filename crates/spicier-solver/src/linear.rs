//! Linear system solvers.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::error::{Error, Result};

/// Solve a linear system Ax = b using LU decomposition.
pub fn solve_dense(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>> {
    if a.nrows() != a.ncols() {
        return Err(Error::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }
    if a.nrows() != b.len() {
        return Err(Error::DimensionMismatch {
            expected: a.nrows(),
            actual: b.len(),
        });
    }

    a.clone().lu().solve(b).ok_or(Error::SingularMatrix)
}

/// Solve a complex linear system Ax = b using LU decomposition.
pub fn solve_complex(
    a: &DMatrix<Complex<f64>>,
    b: &DVector<Complex<f64>>,
) -> Result<DVector<Complex<f64>>> {
    if a.nrows() != a.ncols() {
        return Err(Error::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }
    if a.nrows() != b.len() {
        return Err(Error::DimensionMismatch {
            expected: a.nrows(),
            actual: b.len(),
        });
    }

    a.clone().lu().solve(b).ok_or(Error::SingularMatrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_solve_simple() {
        // 2x + y = 5
        // x + 3y = 6
        // Solution: x = 1.8, y = 1.4
        let a = dmatrix![2.0, 1.0; 1.0, 3.0];
        let b = dvector![5.0, 6.0];

        let x = solve_dense(&a, &b).unwrap();

        assert!((x[0] - 1.8).abs() < 1e-10);
        assert!((x[1] - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix() {
        let a = dmatrix![1.0, 2.0; 2.0, 4.0]; // Singular (row 2 = 2 * row 1)
        let b = dvector![1.0, 2.0];

        let result = solve_dense(&a, &b);
        assert!(matches!(result, Err(Error::SingularMatrix)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = dmatrix![1.0, 2.0; 3.0, 4.0];
        let b = dvector![1.0, 2.0, 3.0];

        let result = solve_dense(&a, &b);
        assert!(matches!(result, Err(Error::DimensionMismatch { .. })));
    }
}
