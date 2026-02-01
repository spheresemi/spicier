//! Linear system solvers.

use faer::prelude::*;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::error::{Error, Result};

/// Systems with this many or more variables use the sparse solver path.
pub const SPARSE_THRESHOLD: usize = 50;

// ============================================================================
// Cached Sparse LU Solver (Real)
// ============================================================================

/// Cached sparse LU solver for real systems.
///
/// Caches the symbolic factorization (elimination tree, fill-in pattern) so that
/// repeated solves with the same sparsity pattern only require numeric factorization.
/// This provides significant speedup for Newton-Raphson iterations and transient
/// timesteps where the matrix structure is fixed.
pub struct CachedSparseLu {
    symbolic: SymbolicLu<usize>,
    size: usize,
}

impl CachedSparseLu {
    /// Create a new cached solver from the sparsity pattern.
    ///
    /// The symbolic factorization is computed once and reused for all subsequent solves.
    pub fn new(size: usize, triplets: &[(usize, usize, f64)]) -> Result<Self> {
        // Build sparse matrix to extract symbolic structure
        let faer_triplets: Vec<_> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();

        let sparse_mat = SparseColMat::<usize, f64>::try_new_from_triplets(size, size, &faer_triplets)
            .map_err(|_| Error::SingularMatrix)?;

        // Compute symbolic factorization
        let symbolic = SymbolicLu::try_new(sparse_mat.symbolic())
            .map_err(|e| Error::SolverError(format!("Symbolic factorization failed: {:?}", e)))?;

        Ok(Self { symbolic, size })
    }

    /// Create from an existing symbolic sparse matrix structure.
    pub fn from_symbolic(symbolic_mat: SymbolicSparseColMat<usize>) -> Result<Self> {
        let size = symbolic_mat.nrows();
        let symbolic = SymbolicLu::try_new(symbolic_mat.as_ref())
            .map_err(|e| Error::SolverError(format!("Symbolic factorization failed: {:?}", e)))?;

        Ok(Self { symbolic, size })
    }

    /// Solve Ax = b using the cached symbolic factorization.
    ///
    /// Only numeric factorization is performed, reusing the cached elimination tree.
    pub fn solve(&self, triplets: &[(usize, usize, f64)], rhs: &DVector<f64>) -> Result<DVector<f64>> {
        if self.size != rhs.len() {
            return Err(Error::DimensionMismatch {
                expected: self.size,
                actual: rhs.len(),
            });
        }

        // Build sparse matrix with new values
        let faer_triplets: Vec<_> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();

        let sparse_mat =
            SparseColMat::<usize, f64>::try_new_from_triplets(self.size, self.size, &faer_triplets)
                .map_err(|_| Error::SingularMatrix)?;

        // Numeric factorization using cached symbolic
        let lu = Lu::try_new_with_symbolic(self.symbolic.clone(), sparse_mat.as_ref())
            .map_err(|_| Error::SingularMatrix)?;

        // Convert RHS and solve
        let faer_rhs = Col::<f64>::from_fn(self.size, |i| rhs[i]);
        let faer_x = lu.solve(&faer_rhs);

        Ok(DVector::from_fn(self.size, |i, _| faer_x[i]))
    }

    /// Get the system size.
    pub fn size(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Cached Sparse LU Solver (Complex)
// ============================================================================

/// Cached sparse LU solver for complex systems.
///
/// Same as [`CachedSparseLu`] but for complex-valued matrices (AC analysis).
pub struct CachedSparseLuComplex {
    symbolic: SymbolicLu<usize>,
    size: usize,
}

impl CachedSparseLuComplex {
    /// Create a new cached solver from the sparsity pattern.
    pub fn new(size: usize, triplets: &[(usize, usize, Complex<f64>)]) -> Result<Self> {
        let faer_triplets: Vec<_> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, c64::new(v.re, v.im)))
            .collect();

        let sparse_mat =
            SparseColMat::<usize, c64>::try_new_from_triplets(size, size, &faer_triplets)
                .map_err(|_| Error::SingularMatrix)?;

        let symbolic = SymbolicLu::try_new(sparse_mat.symbolic())
            .map_err(|e| Error::SolverError(format!("Symbolic factorization failed: {:?}", e)))?;

        Ok(Self { symbolic, size })
    }

    /// Solve Ax = b using the cached symbolic factorization.
    pub fn solve(
        &self,
        triplets: &[(usize, usize, Complex<f64>)],
        rhs: &DVector<Complex<f64>>,
    ) -> Result<DVector<Complex<f64>>> {
        if self.size != rhs.len() {
            return Err(Error::DimensionMismatch {
                expected: self.size,
                actual: rhs.len(),
            });
        }

        let faer_triplets: Vec<_> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, c64::new(v.re, v.im)))
            .collect();

        let sparse_mat =
            SparseColMat::<usize, c64>::try_new_from_triplets(self.size, self.size, &faer_triplets)
                .map_err(|_| Error::SingularMatrix)?;

        let lu = Lu::try_new_with_symbolic(self.symbolic.clone(), sparse_mat.as_ref())
            .map_err(|_| Error::SingularMatrix)?;

        let faer_rhs = Col::<c64>::from_fn(self.size, |i| c64::new(rhs[i].re, rhs[i].im));
        let faer_x = lu.solve(&faer_rhs);

        Ok(DVector::from_fn(self.size, |i, _| {
            Complex::new(faer_x[i].re, faer_x[i].im)
        }))
    }

    /// Get the system size.
    pub fn size(&self) -> usize {
        self.size
    }
}

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

/// Solve a sparse linear system Ax = b using sparse LU decomposition.
///
/// The matrix is constructed from triplets `(row, col, value)`. Duplicate entries
/// at the same position are summed automatically by faer.
pub fn solve_sparse(
    size: usize,
    triplets: &[(usize, usize, f64)],
    rhs: &DVector<f64>,
) -> Result<DVector<f64>> {
    if size != rhs.len() {
        return Err(Error::DimensionMismatch {
            expected: size,
            actual: rhs.len(),
        });
    }

    // Convert triplets to faer format
    let faer_triplets: Vec<_> = triplets
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, v))
        .collect();

    let sparse_mat = SparseColMat::<usize, f64>::try_new_from_triplets(
        size,
        size,
        &faer_triplets,
    )
    .map_err(|_| Error::SingularMatrix)?;

    let lu = sparse_mat.sp_lu().map_err(|_| Error::SingularMatrix)?;

    // Convert nalgebra DVector to faer Col
    let faer_rhs = Col::<f64>::from_fn(size, |i| rhs[i]);

    let faer_x = lu.solve(&faer_rhs);

    // Convert back to nalgebra DVector
    Ok(DVector::from_fn(size, |i, _| faer_x[i]))
}

/// Solve a sparse complex linear system Ax = b using sparse LU decomposition.
pub fn solve_sparse_complex(
    size: usize,
    triplets: &[(usize, usize, Complex<f64>)],
    rhs: &DVector<Complex<f64>>,
) -> Result<DVector<Complex<f64>>> {
    if size != rhs.len() {
        return Err(Error::DimensionMismatch {
            expected: size,
            actual: rhs.len(),
        });
    }

    // Convert triplets to faer format using c64
    let faer_triplets: Vec<_> = triplets
        .iter()
        .map(|&(r, c, v)| Triplet::new(r, c, c64::new(v.re, v.im)))
        .collect();

    let sparse_mat = SparseColMat::<usize, c64>::try_new_from_triplets(
        size,
        size,
        &faer_triplets,
    )
    .map_err(|_| Error::SingularMatrix)?;

    let lu = sparse_mat.sp_lu().map_err(|_| Error::SingularMatrix)?;

    // Convert nalgebra DVector<Complex> to faer Col<c64>
    let faer_rhs = Col::<c64>::from_fn(size, |i| c64::new(rhs[i].re, rhs[i].im));

    let faer_x = lu.solve(&faer_rhs);

    // Convert back to nalgebra DVector<Complex>
    Ok(DVector::from_fn(size, |i, _| {
        Complex::new(faer_x[i].re, faer_x[i].im)
    }))
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

    #[test]
    fn test_solve_sparse_simple() {
        // Same system as test_solve_simple:
        // 2x + y = 5
        // x + 3y = 6
        // Solution: x = 1.8, y = 1.4
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let b = dvector![5.0, 6.0];

        let x = solve_sparse(2, &triplets, &b).unwrap();

        assert!((x[0] - 1.8).abs() < 1e-10, "x[0] = {} (expected 1.8)", x[0]);
        assert!((x[1] - 1.4).abs() < 1e-10, "x[1] = {} (expected 1.4)", x[1]);
    }

    #[test]
    fn test_solve_sparse_complex_simple() {
        // (2+i)x + y = 5+i
        // x + (3-i)y = 6
        let triplets = vec![
            (0, 0, Complex::new(2.0, 1.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, -1.0)),
        ];
        let b = dvector![Complex::new(5.0, 1.0), Complex::new(6.0, 0.0)];

        let x = solve_sparse_complex(2, &triplets, &b).unwrap();

        // Verify by computing Ax and comparing to b
        let ax0 = Complex::new(2.0, 1.0) * x[0] + Complex::new(1.0, 0.0) * x[1];
        let ax1 = Complex::new(1.0, 0.0) * x[0] + Complex::new(3.0, -1.0) * x[1];

        assert!((ax0 - Complex::new(5.0, 1.0)).norm() < 1e-10, "Ax[0] mismatch");
        assert!((ax1 - Complex::new(6.0, 0.0)).norm() < 1e-10, "Ax[1] mismatch");
    }

    #[test]
    fn test_solve_sparse_matches_dense() {
        // Build a 20x20 diagonally dominant system and verify sparse == dense
        let size = 20;
        let a = DMatrix::from_fn(size, size, |i, j| {
            if i == j {
                (size as f64) + 1.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            }
        });
        let b = DVector::from_fn(size, |i, _| (i + 1) as f64);

        // Build triplets from the dense matrix
        let mut triplets = Vec::new();
        for i in 0..size {
            for j in 0..size {
                let v = a[(i, j)];
                if v.abs() > 1e-15 {
                    triplets.push((i, j, v));
                }
            }
        }

        let x_dense = solve_dense(&a, &b).unwrap();
        let x_sparse = solve_sparse(size, &triplets, &b).unwrap();

        for i in 0..size {
            assert!(
                (x_dense[i] - x_sparse[i]).abs() < 1e-10,
                "Mismatch at [{}]: dense={}, sparse={}",
                i,
                x_dense[i],
                x_sparse[i]
            );
        }
    }

    #[test]
    fn test_solve_sparse_dimension_mismatch() {
        let triplets = vec![(0, 0, 1.0)];
        let b = dvector![1.0, 2.0];

        let result = solve_sparse(1, &triplets, &b);
        assert!(matches!(result, Err(Error::DimensionMismatch { .. })));
    }

    #[test]
    fn test_solve_sparse_with_duplicate_triplets() {
        // Duplicates at the same position should be summed
        // A = [[3, 1], [1, 3]] with (0,0) split as 2.0 + 1.0
        let triplets = vec![
            (0, 0, 2.0),
            (0, 0, 1.0), // duplicate: summed to 3.0
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let b = dvector![4.0, 4.0];

        // A = [[3,1],[1,3]], b = [4,4] → x = [1,1]
        let x = solve_sparse(2, &triplets, &b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] = {} (expected 1.0)", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-10, "x[1] = {} (expected 1.0)", x[1]);
    }

    // ========================================================================
    // Cached Sparse LU Tests
    // ========================================================================

    #[test]
    fn test_cached_sparse_lu_simple() {
        // Same system as test_solve_sparse_simple
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let b = dvector![5.0, 6.0];

        let cached = CachedSparseLu::new(2, &triplets).unwrap();
        let x = cached.solve(&triplets, &b).unwrap();

        assert!((x[0] - 1.8).abs() < 1e-10, "x[0] = {} (expected 1.8)", x[0]);
        assert!((x[1] - 1.4).abs() < 1e-10, "x[1] = {} (expected 1.4)", x[1]);
    }

    #[test]
    fn test_cached_sparse_lu_reuse() {
        // Create cached solver with initial sparsity pattern
        let triplets1 = vec![
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let b1 = dvector![5.0, 6.0];

        let cached = CachedSparseLu::new(2, &triplets1).unwrap();

        // First solve
        let x1 = cached.solve(&triplets1, &b1).unwrap();
        assert!((x1[0] - 1.8).abs() < 1e-10);
        assert!((x1[1] - 1.4).abs() < 1e-10);

        // Second solve with different values but same sparsity pattern
        let triplets2 = vec![
            (0, 0, 4.0),  // changed
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 4.0),  // changed
        ];
        let b2 = dvector![10.0, 10.0];

        // A = [[4,1],[1,4]], b = [10,10] → x = [2,2]
        let x2 = cached.solve(&triplets2, &b2).unwrap();
        assert!((x2[0] - 2.0).abs() < 1e-10, "x2[0] = {} (expected 2.0)", x2[0]);
        assert!((x2[1] - 2.0).abs() < 1e-10, "x2[1] = {} (expected 2.0)", x2[1]);
    }

    #[test]
    fn test_cached_sparse_lu_matches_uncached() {
        // Build a 20x20 system and verify cached == uncached
        let size = 20;
        let mut triplets = Vec::new();
        for i in 0..size {
            for j in 0..size {
                let v = if i == j {
                    (size as f64) + 1.0
                } else {
                    1.0 / ((i as f64 - j as f64).abs() + 1.0)
                };
                if v.abs() > 1e-15 {
                    triplets.push((i, j, v));
                }
            }
        }
        let b = DVector::from_fn(size, |i, _| (i + 1) as f64);

        let x_uncached = solve_sparse(size, &triplets, &b).unwrap();

        let cached = CachedSparseLu::new(size, &triplets).unwrap();
        let x_cached = cached.solve(&triplets, &b).unwrap();

        for i in 0..size {
            assert!(
                (x_uncached[i] - x_cached[i]).abs() < 1e-10,
                "Mismatch at [{}]: uncached={}, cached={}",
                i,
                x_uncached[i],
                x_cached[i]
            );
        }
    }

    #[test]
    fn test_cached_sparse_lu_complex_simple() {
        let triplets = vec![
            (0, 0, Complex::new(2.0, 1.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, -1.0)),
        ];
        let b = dvector![Complex::new(5.0, 1.0), Complex::new(6.0, 0.0)];

        let cached = CachedSparseLuComplex::new(2, &triplets).unwrap();
        let x = cached.solve(&triplets, &b).unwrap();

        // Verify by computing Ax and comparing to b
        let ax0 = Complex::new(2.0, 1.0) * x[0] + Complex::new(1.0, 0.0) * x[1];
        let ax1 = Complex::new(1.0, 0.0) * x[0] + Complex::new(3.0, -1.0) * x[1];

        assert!((ax0 - Complex::new(5.0, 1.0)).norm() < 1e-10, "Ax[0] mismatch");
        assert!((ax1 - Complex::new(6.0, 0.0)).norm() < 1e-10, "Ax[1] mismatch");
    }

    #[test]
    fn test_cached_sparse_lu_complex_reuse() {
        // Test reusing complex cached solver with different values
        let triplets1 = vec![
            (0, 0, Complex::new(2.0, 0.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(2.0, 0.0)),
        ];
        let b1 = dvector![Complex::new(3.0, 0.0), Complex::new(3.0, 0.0)];

        let cached = CachedSparseLuComplex::new(2, &triplets1).unwrap();

        // First solve: A = [[2,1],[1,2]], b = [3,3] → x = [1,1]
        let x1 = cached.solve(&triplets1, &b1).unwrap();
        assert!((x1[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
        assert!((x1[1] - Complex::new(1.0, 0.0)).norm() < 1e-10);

        // Second solve with different values
        let triplets2 = vec![
            (0, 0, Complex::new(3.0, 1.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 0, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(3.0, -1.0)),
        ];
        let b2 = dvector![Complex::new(4.0, 1.0), Complex::new(4.0, -1.0)];

        let x2 = cached.solve(&triplets2, &b2).unwrap();

        // Verify Ax2 = b2
        let ax0 = triplets2[0].2 * x2[0] + triplets2[1].2 * x2[1];
        let ax1 = triplets2[2].2 * x2[0] + triplets2[3].2 * x2[1];
        assert!((ax0 - b2[0]).norm() < 1e-10, "Ax2[0] mismatch");
        assert!((ax1 - b2[1]).norm() < 1e-10, "Ax2[1] mismatch");
    }
}
