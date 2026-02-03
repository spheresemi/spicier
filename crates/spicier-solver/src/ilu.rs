//! ILU(0) Incomplete LU Preconditioner.
//!
//! ILU(0) computes an approximate LU factorization maintaining the original
//! sparsity pattern. For MNA matrices (diagonally dominant, local structure),
//! ILU(0) provides dramatically better preconditioning than Jacobi while
//! remaining efficient to compute and apply.
//!
//! # Algorithm
//!
//! ILU(0) factorization follows the standard LU algorithm but only stores
//! entries at positions where the original matrix has nonzeros:
//!
//! ```text
//! For k = 1 to n-1:
//!   For i = k+1 to n where A(i,k) != 0:
//!     A(i,k) = A(i,k) / A(k,k)
//!     For j = k+1 to n where A(k,j) != 0:
//!       If A(i,j) exists in pattern:
//!         A(i,j) = A(i,j) - A(i,k) * A(k,j)
//! ```
//!
//! The L factor (with unit diagonal) is stored below the diagonal,
//! and U (including diagonal) is stored on and above the diagonal.

use num_complex::Complex64 as C64;

use crate::preconditioner::{ComplexPreconditioner, RealPreconditioner};

// ============================================================================
// Real ILU(0) Preconditioner
// ============================================================================

/// ILU(0) preconditioner for real-valued linear systems.
///
/// Computes an incomplete LU factorization that maintains the original
/// sparsity pattern. Much more effective than Jacobi for non-diagonal
/// dominated systems while still being efficient.
///
/// # Construction
///
/// From CSR (Compressed Sparse Row) format:
/// - `row_ptr[i]` gives the starting index in col_idx/values for row i
/// - `col_idx[k]` gives the column index for entry k
/// - `values[k]` gives the value for entry k
///
/// # Usage
///
/// ```rust,ignore
/// use spicier_solver::Ilu0Preconditioner;
///
/// // CSR format for a 3x3 matrix
/// let row_ptr = vec![0, 2, 4, 6];
/// let col_idx = vec![0, 1, 0, 1, 1, 2];
/// let values = vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0];
///
/// let precond = Ilu0Preconditioner::from_csr(3, &row_ptr, &col_idx, &values)?;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let mut y = vec![0.0; 3];
/// precond.apply(&x, &mut y);  // y = (L*U)^{-1} * x
/// ```
pub struct Ilu0Preconditioner {
    /// Row pointers (size: n+1).
    row_ptr: Vec<usize>,
    /// Column indices for each non-zero.
    col_idx: Vec<usize>,
    /// Combined L (below diagonal) and U (on/above diagonal) values.
    lu_values: Vec<f64>,
    /// Index of diagonal entry for each row (for fast access).
    diag_idx: Vec<usize>,
    /// Matrix dimension.
    size: usize,
}

impl Ilu0Preconditioner {
    /// Create from CSR (Compressed Sparse Row) format.
    ///
    /// Performs ILU(0) factorization maintaining the original sparsity pattern.
    ///
    /// # Arguments
    ///
    /// * `size` - Matrix dimension (n×n)
    /// * `row_ptr` - Row pointers (length n+1)
    /// * `col_idx` - Column indices for each non-zero
    /// * `values` - Non-zero values
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Zero pivot encountered during factorization
    /// - Diagonal entry missing from sparsity pattern
    #[allow(clippy::needless_range_loop)]
    pub fn from_csr(
        size: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
    ) -> Result<Self, IluError> {
        if row_ptr.len() != size + 1 {
            return Err(IluError::InvalidStructure(
                "row_ptr length must be size + 1".into(),
            ));
        }

        let nnz = row_ptr[size];
        if col_idx.len() != nnz || values.len() != nnz {
            return Err(IluError::InvalidStructure(
                "col_idx and values length must match nnz".into(),
            ));
        }

        // Find diagonal indices
        let mut diag_idx = vec![0; size];
        for i in 0..size {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];
            let mut found = false;
            for k in row_start..row_end {
                if col_idx[k] == i {
                    diag_idx[i] = k;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(IluError::MissingDiagonal(i));
            }
        }

        // Copy values for in-place factorization
        let mut lu_values = values.to_vec();

        // Perform ILU(0) factorization
        Self::factorize_ilu0(size, row_ptr, col_idx, &diag_idx, &mut lu_values)?;

        Ok(Self {
            row_ptr: row_ptr.to_vec(),
            col_idx: col_idx.to_vec(),
            lu_values,
            diag_idx,
            size,
        })
    }

    /// Create from matrix triplets.
    ///
    /// Builds CSR structure from (row, col, value) triplets and performs
    /// ILU(0) factorization. Duplicate entries at the same position are summed.
    ///
    /// # Arguments
    ///
    /// * `size` - Matrix dimension (n×n)
    /// * `triplets` - (row, col, value) entries
    pub fn from_triplets(size: usize, triplets: &[(usize, usize, f64)]) -> Result<Self, IluError> {
        let (row_ptr, col_idx, values) = triplets_to_csr(size, triplets);
        Self::from_csr(size, &row_ptr, &col_idx, &values)
    }

    /// Update values with same sparsity pattern.
    ///
    /// Re-computes the ILU(0) factorization using new values but the
    /// same sparsity structure. More efficient than creating a new
    /// preconditioner when solving multiple systems with the same pattern.
    ///
    /// # Arguments
    ///
    /// * `values` - New non-zero values (must match original nnz)
    pub fn update_values(&mut self, values: &[f64]) -> Result<(), IluError> {
        if values.len() != self.lu_values.len() {
            return Err(IluError::InvalidStructure(
                "values length must match original nnz".into(),
            ));
        }

        // Copy new values
        self.lu_values.copy_from_slice(values);

        // Re-factorize
        Self::factorize_ilu0(
            self.size,
            &self.row_ptr,
            &self.col_idx,
            &self.diag_idx,
            &mut self.lu_values,
        )
    }

    /// Perform ILU(0) factorization in-place.
    ///
    /// After this, lu_values contains:
    /// - L entries (below diagonal, with implicit unit diagonal)
    /// - U entries (on and above diagonal)
    #[allow(clippy::needless_range_loop)]
    fn factorize_ilu0(
        size: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        diag_idx: &[usize],
        lu_values: &mut [f64],
    ) -> Result<(), IluError> {
        // Build a map from (row, col) to index for O(1) lookup
        let mut col_to_idx: Vec<std::collections::HashMap<usize, usize>> =
            vec![Default::default(); size];
        for i in 0..size {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];
            for k in row_start..row_end {
                col_to_idx[i].insert(col_idx[k], k);
            }
        }

        // ILU(0) factorization following standard algorithm
        for i in 1..size {
            let row_i_start = row_ptr[i];
            let row_i_end = row_ptr[i + 1];

            // For each column k < i where A(i,k) != 0
            for k_ptr in row_i_start..row_i_end {
                let k = col_idx[k_ptr];
                if k >= i {
                    break; // Only process lower triangular part
                }

                // A(i,k) = A(i,k) / A(k,k)
                let a_kk = lu_values[diag_idx[k]];
                if a_kk.abs() < 1e-30 {
                    return Err(IluError::ZeroPivot(k));
                }
                lu_values[k_ptr] /= a_kk;
                let a_ik = lu_values[k_ptr];

                // For each column j > k where A(k,j) != 0
                let row_k_start = row_ptr[k];
                let row_k_end = row_ptr[k + 1];
                for j_ptr in row_k_start..row_k_end {
                    let j = col_idx[j_ptr];
                    if j <= k {
                        continue; // Only process upper triangular part of row k
                    }

                    // If A(i,j) exists in pattern, update it
                    if let Some(&ij_ptr) = col_to_idx[i].get(&j) {
                        let a_kj = lu_values[j_ptr];
                        lu_values[ij_ptr] -= a_ik * a_kj;
                    }
                    // If A(i,j) doesn't exist, ILU(0) drops the fill-in
                }
            }
        }

        Ok(())
    }

    /// Apply the preconditioner: y = (L*U)^{-1} * x
    ///
    /// Solves L*z = x (forward substitution), then U*y = z (back substitution).
    fn apply_impl(&self, x: &[f64], y: &mut [f64]) {
        let n = self.size;

        // Forward substitution: L * z = x
        // L has unit diagonal (implicit), stored below diagonal
        // z is stored temporarily in y
        y.copy_from_slice(x);

        for i in 0..n {
            let row_start = self.row_ptr[i];
            let diag = self.diag_idx[i];

            // Subtract L(i,j) * z(j) for j < i
            for k in row_start..diag {
                let j = self.col_idx[k];
                let l_ij = self.lu_values[k];
                y[i] -= l_ij * y[j];
            }
            // z[i] is now correct (L has unit diagonal)
        }

        // Back substitution: U * y = z
        for i in (0..n).rev() {
            let diag = self.diag_idx[i];
            let row_end = self.row_ptr[i + 1];

            // Subtract U(i,j) * y(j) for j > i
            for k in (diag + 1)..row_end {
                let j = self.col_idx[k];
                let u_ij = self.lu_values[k];
                y[i] -= u_ij * y[j];
            }

            // Divide by diagonal
            let u_ii = self.lu_values[diag];
            if u_ii.abs() > 1e-30 {
                y[i] /= u_ii;
            }
        }
    }

    /// Get the sparsity pattern information.
    pub fn sparsity(&self) -> (&[usize], &[usize]) {
        (&self.row_ptr, &self.col_idx)
    }

    /// Get the number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.lu_values.len()
    }
}

impl RealPreconditioner for Ilu0Preconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.size);
        assert_eq!(y.len(), self.size);
        self.apply_impl(x, y);
    }

    fn dim(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Complex ILU(0) Preconditioner
// ============================================================================

/// ILU(0) preconditioner for complex-valued linear systems.
///
/// Same algorithm as real ILU(0) but with complex arithmetic.
pub struct ComplexIlu0Preconditioner {
    /// Row pointers (size: n+1).
    row_ptr: Vec<usize>,
    /// Column indices for each non-zero.
    col_idx: Vec<usize>,
    /// Combined L (below diagonal) and U (on/above diagonal) values.
    lu_values: Vec<C64>,
    /// Index of diagonal entry for each row.
    diag_idx: Vec<usize>,
    /// Matrix dimension.
    size: usize,
}

impl ComplexIlu0Preconditioner {
    /// Create from CSR format with complex values.
    #[allow(clippy::needless_range_loop)]
    pub fn from_csr(
        size: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[C64],
    ) -> Result<Self, IluError> {
        if row_ptr.len() != size + 1 {
            return Err(IluError::InvalidStructure(
                "row_ptr length must be size + 1".into(),
            ));
        }

        let nnz = row_ptr[size];
        if col_idx.len() != nnz || values.len() != nnz {
            return Err(IluError::InvalidStructure(
                "col_idx and values length must match nnz".into(),
            ));
        }

        // Find diagonal indices
        let mut diag_idx = vec![0; size];
        for i in 0..size {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];
            let mut found = false;
            for k in row_start..row_end {
                if col_idx[k] == i {
                    diag_idx[i] = k;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(IluError::MissingDiagonal(i));
            }
        }

        // Copy values for in-place factorization
        let mut lu_values = values.to_vec();

        // Perform ILU(0) factorization
        Self::factorize_ilu0(size, row_ptr, col_idx, &diag_idx, &mut lu_values)?;

        Ok(Self {
            row_ptr: row_ptr.to_vec(),
            col_idx: col_idx.to_vec(),
            lu_values,
            diag_idx,
            size,
        })
    }

    /// Create from matrix triplets.
    pub fn from_triplets(size: usize, triplets: &[(usize, usize, C64)]) -> Result<Self, IluError> {
        let (row_ptr, col_idx, values) = triplets_to_csr_complex(size, triplets);
        Self::from_csr(size, &row_ptr, &col_idx, &values)
    }

    /// Update values with same sparsity pattern.
    pub fn update_values(&mut self, values: &[C64]) -> Result<(), IluError> {
        if values.len() != self.lu_values.len() {
            return Err(IluError::InvalidStructure(
                "values length must match original nnz".into(),
            ));
        }

        self.lu_values.copy_from_slice(values);

        Self::factorize_ilu0(
            self.size,
            &self.row_ptr,
            &self.col_idx,
            &self.diag_idx,
            &mut self.lu_values,
        )
    }

    /// Perform ILU(0) factorization in-place.
    #[allow(clippy::needless_range_loop)]
    fn factorize_ilu0(
        size: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        diag_idx: &[usize],
        lu_values: &mut [C64],
    ) -> Result<(), IluError> {
        let mut col_to_idx: Vec<std::collections::HashMap<usize, usize>> =
            vec![Default::default(); size];
        for i in 0..size {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];
            for k in row_start..row_end {
                col_to_idx[i].insert(col_idx[k], k);
            }
        }

        for i in 1..size {
            let row_i_start = row_ptr[i];
            let row_i_end = row_ptr[i + 1];

            for k_ptr in row_i_start..row_i_end {
                let k = col_idx[k_ptr];
                if k >= i {
                    break;
                }

                let a_kk = lu_values[diag_idx[k]];
                if a_kk.norm() < 1e-30 {
                    return Err(IluError::ZeroPivot(k));
                }
                lu_values[k_ptr] /= a_kk;
                let a_ik = lu_values[k_ptr];

                let row_k_start = row_ptr[k];
                let row_k_end = row_ptr[k + 1];
                for j_ptr in row_k_start..row_k_end {
                    let j = col_idx[j_ptr];
                    if j <= k {
                        continue;
                    }

                    if let Some(&ij_ptr) = col_to_idx[i].get(&j) {
                        let a_kj = lu_values[j_ptr];
                        lu_values[ij_ptr] -= a_ik * a_kj;
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_impl(&self, x: &[C64], y: &mut [C64]) {
        let n = self.size;

        // Forward substitution: L * z = x
        y.copy_from_slice(x);

        for i in 0..n {
            let row_start = self.row_ptr[i];
            let diag = self.diag_idx[i];

            for k in row_start..diag {
                let j = self.col_idx[k];
                let l_ij = self.lu_values[k];
                y[i] -= l_ij * y[j];
            }
        }

        // Back substitution: U * y = z
        for i in (0..n).rev() {
            let diag = self.diag_idx[i];
            let row_end = self.row_ptr[i + 1];

            for k in (diag + 1)..row_end {
                let j = self.col_idx[k];
                let u_ij = self.lu_values[k];
                y[i] -= u_ij * y[j];
            }

            let u_ii = self.lu_values[diag];
            if u_ii.norm() > 1e-30 {
                y[i] /= u_ii;
            }
        }
    }

    /// Get the sparsity pattern information.
    pub fn sparsity(&self) -> (&[usize], &[usize]) {
        (&self.row_ptr, &self.col_idx)
    }

    /// Get the number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.lu_values.len()
    }
}

impl ComplexPreconditioner for ComplexIlu0Preconditioner {
    fn apply(&self, x: &[C64], y: &mut [C64]) {
        assert_eq!(x.len(), self.size);
        assert_eq!(y.len(), self.size);
        self.apply_impl(x, y);
    }

    fn dim(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors from ILU factorization.
#[derive(Debug, Clone)]
pub enum IluError {
    /// Zero or near-zero pivot encountered.
    ZeroPivot(usize),
    /// Diagonal entry missing from sparsity pattern.
    MissingDiagonal(usize),
    /// Invalid matrix structure.
    InvalidStructure(String),
}

impl std::fmt::Display for IluError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IluError::ZeroPivot(idx) => write!(f, "Zero pivot at row {}", idx),
            IluError::MissingDiagonal(idx) => {
                write!(f, "Missing diagonal entry at row {}", idx)
            }
            IluError::InvalidStructure(msg) => write!(f, "Invalid structure: {}", msg),
        }
    }
}

impl std::error::Error for IluError {}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert triplets to CSR format.
fn triplets_to_csr(
    size: usize,
    triplets: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    use std::collections::BTreeMap;

    // First, aggregate by (row, col) and sum duplicates
    let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for &(row, col, val) in triplets {
        *entries.entry((row, col)).or_insert(0.0) += val;
    }

    // Build CSR
    let mut row_ptr = vec![0; size + 1];
    let mut col_idx = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());

    let mut current_row = 0;
    for (&(row, col), &val) in &entries {
        while current_row <= row {
            row_ptr[current_row] = col_idx.len();
            current_row += 1;
        }
        col_idx.push(col);
        values.push(val);
    }
    while current_row <= size {
        row_ptr[current_row] = col_idx.len();
        current_row += 1;
    }

    (row_ptr, col_idx, values)
}

/// Convert complex triplets to CSR format.
fn triplets_to_csr_complex(
    size: usize,
    triplets: &[(usize, usize, C64)],
) -> (Vec<usize>, Vec<usize>, Vec<C64>) {
    use std::collections::BTreeMap;

    let mut entries: BTreeMap<(usize, usize), C64> = BTreeMap::new();
    for &(row, col, val) in triplets {
        *entries.entry((row, col)).or_insert(C64::new(0.0, 0.0)) += val;
    }

    let mut row_ptr = vec![0; size + 1];
    let mut col_idx = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());

    let mut current_row = 0;
    for (&(row, col), &val) in &entries {
        while current_row <= row {
            row_ptr[current_row] = col_idx.len();
            current_row += 1;
        }
        col_idx.push(col);
        values.push(val);
    }
    while current_row <= size {
        row_ptr[current_row] = col_idx.len();
        current_row += 1;
    }

    (row_ptr, col_idx, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilu0_diagonal_matrix() {
        // Diagonal matrix: ILU(0) should give exact factorization
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 4.0];

        let precond = Ilu0Preconditioner::from_csr(3, &row_ptr, &col_idx, &values).unwrap();

        // Apply to [2, 6, 12] should give [1, 2, 3]
        let x = vec![2.0, 6.0, 12.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-12, "y[0] = {}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-12, "y[1] = {}", y[1]);
        assert!((y[2] - 3.0).abs() < 1e-12, "y[2] = {}", y[2]);
    }

    #[test]
    fn ilu0_tridiagonal_matrix() {
        // Tridiagonal: [-1, 2, -1] pattern
        // | 2 -1  0 |
        // |-1  2 -1 |
        // | 0 -1  2 |
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];

        let precond = Ilu0Preconditioner::from_csr(3, &row_ptr, &col_idx, &values).unwrap();

        // Test: apply to a vector and verify it improves
        let x = vec![1.0, 0.0, 1.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        // For a tridiagonal system, ILU(0) is exact (no fill-in dropped)
        // So y should be exact solution to A*y = x
        // Verify by computing A*y and comparing to x
        let ay0 = 2.0 * y[0] - 1.0 * y[1];
        let ay1 = -y[0] + 2.0 * y[1] - 1.0 * y[2];
        let ay2 = -y[1] + 2.0 * y[2];

        assert!(
            (ay0 - x[0]).abs() < 1e-12,
            "A*y[0] = {}, expected {}",
            ay0,
            x[0]
        );
        assert!(
            (ay1 - x[1]).abs() < 1e-12,
            "A*y[1] = {}, expected {}",
            ay1,
            x[1]
        );
        assert!(
            (ay2 - x[2]).abs() < 1e-12,
            "A*y[2] = {}, expected {}",
            ay2,
            x[2]
        );
    }

    #[test]
    fn ilu0_from_triplets() {
        // Same tridiagonal matrix from triplets
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ];

        let precond = Ilu0Preconditioner::from_triplets(3, &triplets).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        // Verify A*y = x
        let ay0 = 2.0 * y[0] - 1.0 * y[1];
        let ay1 = -y[0] + 2.0 * y[1] - 1.0 * y[2];
        let ay2 = -y[1] + 2.0 * y[2];

        assert!((ay0 - x[0]).abs() < 1e-12);
        assert!((ay1 - x[1]).abs() < 1e-12);
        assert!((ay2 - x[2]).abs() < 1e-12);
    }

    #[test]
    fn ilu0_handles_duplicate_triplets() {
        // Diagonal with duplicate entries that should sum
        let triplets = vec![
            (0, 0, 1.0),
            (0, 0, 1.0), // Duplicate: should sum to 2.0
            (1, 1, 3.0),
            (2, 2, 4.0),
        ];

        let precond = Ilu0Preconditioner::from_triplets(3, &triplets).unwrap();

        let x = vec![4.0, 9.0, 16.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        assert!((y[0] - 2.0).abs() < 1e-12); // 4/2 = 2
        assert!((y[1] - 3.0).abs() < 1e-12); // 9/3 = 3
        assert!((y[2] - 4.0).abs() < 1e-12); // 16/4 = 4
    }

    #[test]
    fn ilu0_update_values() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![2.0, 3.0, 4.0];

        let mut precond = Ilu0Preconditioner::from_csr(3, &row_ptr, &col_idx, &values).unwrap();

        // Update to new diagonal values
        let new_values = vec![4.0, 6.0, 8.0];
        precond.update_values(&new_values).unwrap();

        let x = vec![4.0, 12.0, 24.0];
        let mut y = vec![0.0; 3];
        precond.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn ilu0_missing_diagonal_error() {
        // Matrix missing diagonal entry at row 1
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 0, 2]; // Row 1 has col 0 instead of col 1
        let values = vec![1.0, 1.0, 1.0];

        let result = Ilu0Preconditioner::from_csr(3, &row_ptr, &col_idx, &values);
        assert!(matches!(result, Err(IluError::MissingDiagonal(1))));
    }

    #[test]
    fn ilu0_zero_pivot_error() {
        // Matrix with zero pivot at row 0 that is accessed during elimination of row 1
        // | 0  1 |
        // | 1  2 |
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let values = vec![0.0, 1.0, 1.0, 2.0]; // Zero diagonal at row 0

        let result = Ilu0Preconditioner::from_csr(2, &row_ptr, &col_idx, &values);
        assert!(matches!(result, Err(IluError::ZeroPivot(0))));
    }

    #[test]
    fn ilu0_preconditioner_dim() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let precond = Ilu0Preconditioner::from_triplets(3, &triplets).unwrap();
        assert_eq!(precond.dim(), 3);
    }

    #[test]
    fn complex_ilu0_diagonal_matrix() {
        let row_ptr = vec![0, 1, 2];
        let col_idx = vec![0, 1];
        let values = vec![C64::new(2.0, 0.0), C64::new(0.0, 4.0)]; // 2 and 4i

        let precond = ComplexIlu0Preconditioner::from_csr(2, &row_ptr, &col_idx, &values).unwrap();

        let x = vec![C64::new(4.0, 0.0), C64::new(0.0, 8.0)];
        let mut y = vec![C64::new(0.0, 0.0); 2];
        precond.apply(&x, &mut y);

        // y[0] = 4/2 = 2, y[1] = 8i/(4i) = 2
        assert!((y[0] - C64::new(2.0, 0.0)).norm() < 1e-12);
        assert!((y[1] - C64::new(2.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn complex_ilu0_from_triplets() {
        let triplets = vec![(0, 0, C64::new(2.0, 1.0)), (1, 1, C64::new(3.0, -1.0))];

        let precond = ComplexIlu0Preconditioner::from_triplets(2, &triplets).unwrap();
        assert_eq!(precond.dim(), 2);
    }

    #[test]
    fn ilu0_sparsity_info() {
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)];
        let precond = Ilu0Preconditioner::from_triplets(2, &triplets).unwrap();

        let (row_ptr, col_idx) = precond.sparsity();
        assert_eq!(row_ptr.len(), 3);
        assert_eq!(col_idx.len(), 4);
        assert_eq!(precond.nnz(), 4);
    }

    #[test]
    fn ilu0_improves_gmres_convergence() {
        use crate::gmres::{GmresConfig, solve_gmres_real, solve_gmres_real_preconditioned};
        use crate::preconditioner::JacobiPreconditioner;
        use crate::sparse_operator::SparseRealOperator;

        // Create a diagonally dominant 10x10 tridiagonal matrix
        // This is a classic test case for iterative solvers
        let n = 10;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 4.0)); // Main diagonal
            if i > 0 {
                triplets.push((i, i - 1, -1.0)); // Lower diagonal
            }
            if i < n - 1 {
                triplets.push((i, i + 1, -1.0)); // Upper diagonal
            }
        }

        // Create operator from triplets
        let op = SparseRealOperator::from_triplets(n, &triplets)
            .expect("Failed to create sparse operator");

        // RHS vector
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();

        let config = GmresConfig {
            max_iter: 100,
            tol: 1e-10,
            restart: 30,
        };

        // Solve without preconditioning
        let result_none = solve_gmres_real(&op, &b, &config);

        // Solve with Jacobi preconditioning
        let jacobi = JacobiPreconditioner::from_triplets(n, &triplets);
        let result_jacobi = solve_gmres_real_preconditioned(&op, &jacobi, &b, &config);

        // Solve with ILU(0) preconditioning
        let ilu = Ilu0Preconditioner::from_triplets(n, &triplets).unwrap();
        let result_ilu = solve_gmres_real_preconditioned(&op, &ilu, &b, &config);

        // All should converge
        assert!(
            result_none.converged,
            "Unpreconditioned GMRES did not converge"
        );
        assert!(
            result_jacobi.converged,
            "Jacobi-preconditioned GMRES did not converge"
        );
        assert!(
            result_ilu.converged,
            "ILU-preconditioned GMRES did not converge"
        );

        // ILU should be as good or better than Jacobi
        // (For tridiagonal, ILU(0) is exact, so it converges in 1 iteration)
        assert!(
            result_ilu.iterations <= result_jacobi.iterations,
            "ILU ({} iters) should be <= Jacobi ({} iters)",
            result_ilu.iterations,
            result_jacobi.iterations
        );

        // Verify solutions are close
        for i in 0..n {
            assert!(
                (result_ilu.x[i] - result_jacobi.x[i]).abs() < 1e-6,
                "Solutions differ at index {}: ILU={}, Jacobi={}",
                i,
                result_ilu.x[i],
                result_jacobi.x[i]
            );
        }
    }

    #[test]
    fn ilu0_4x4_with_fill_dropping() {
        // A 4x4 matrix where ILU(0) must drop fill-in
        // This tests the actual ILU(0) behavior vs full LU
        //
        // | 4  1  0  1 |
        // | 1  4  1  0 |
        // | 0  1  4  1 |
        // | 1  0  1  4 |
        //
        // ILU(0) cannot store fill at (2,0), (3,1), (0,3), (1,2) etc.
        // but it approximates well for diagonally dominant matrices
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 1.0),
            (0, 3, 1.0),
            (1, 0, 1.0),
            (1, 1, 4.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
            (2, 3, 1.0),
            (3, 0, 1.0),
            (3, 2, 1.0),
            (3, 3, 4.0),
        ];

        let precond = Ilu0Preconditioner::from_triplets(4, &triplets).unwrap();

        // Test that applying the preconditioner produces reasonable results
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        precond.apply(&x, &mut y);

        // y should be a reasonable approximation to A^{-1} * x
        // Verify the preconditioner doesn't produce NaN or wildly wrong values
        for yi in &y {
            assert!(yi.is_finite(), "ILU produced non-finite value");
        }
    }
}
