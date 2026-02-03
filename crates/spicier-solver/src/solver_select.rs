//! Automatic solver selection based on system characteristics.
//!
//! Provides heuristics for choosing between direct (LU) and iterative (GMRES)
//! solvers based on system size, sparsity, and user preferences.

use crate::error::Result;
use crate::gmres::{GmresConfig, RealGmresResult, solve_gmres_real};
use crate::linear::{SPARSE_THRESHOLD, solve_sparse};
use crate::sparse_operator::SparseRealOperator;
use nalgebra::DVector;

/// Solver selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SolverStrategy {
    /// Automatically select based on system size (default).
    #[default]
    Auto,
    /// Always use direct LU factorization.
    DirectLU,
    /// Always use iterative GMRES.
    IterativeGmres,
}

impl SolverStrategy {
    /// Parse from string (for CLI).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "lu" | "direct" => Some(Self::DirectLU),
            "gmres" | "iterative" => Some(Self::IterativeGmres),
            _ => None,
        }
    }

    /// Get the strategy name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DirectLU => "direct (LU)",
            Self::IterativeGmres => "iterative (GMRES)",
        }
    }
}

impl std::fmt::Display for SolverStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for solver selection.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Solver selection strategy.
    pub strategy: SolverStrategy,
    /// System size threshold for switching from LU to GMRES in auto mode.
    /// Systems with size >= this value will use GMRES.
    pub gmres_threshold: usize,
    /// GMRES configuration (used when GMRES is selected).
    pub gmres_config: GmresConfig,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            strategy: SolverStrategy::Auto,
            // 10k nodes is a reasonable threshold where GMRES starts to outperform
            // direct LU due to memory and fill-in considerations.
            // For MNA matrices, this corresponds to circuits with ~10k nodes.
            gmres_threshold: 10_000,
            gmres_config: GmresConfig::default(),
        }
    }
}

impl SolverConfig {
    /// Create a config that always uses direct LU.
    pub fn direct_lu() -> Self {
        Self {
            strategy: SolverStrategy::DirectLU,
            ..Default::default()
        }
    }

    /// Create a config that always uses GMRES.
    pub fn gmres() -> Self {
        Self {
            strategy: SolverStrategy::IterativeGmres,
            ..Default::default()
        }
    }

    /// Create a config with a custom GMRES threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            gmres_threshold: threshold,
            ..Default::default()
        }
    }
}

/// Result of a solver selection solve.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Solution vector.
    pub solution: DVector<f64>,
    /// Which solver was used.
    pub solver_used: SolverStrategy,
    /// Number of iterations (only for GMRES).
    pub iterations: Option<usize>,
    /// Final residual (only for GMRES).
    pub residual: Option<f64>,
}

/// Solve a real linear system with automatic solver selection.
///
/// For small/medium systems, uses direct sparse LU factorization.
/// For very large systems, uses iterative GMRES.
///
/// # Arguments
/// * `size` - System dimension
/// * `triplets` - Sparse matrix in triplet format (row, col, value)
/// * `rhs` - Right-hand side vector
/// * `config` - Solver configuration
pub fn solve_auto(
    size: usize,
    triplets: &[(usize, usize, f64)],
    rhs: &DVector<f64>,
    config: &SolverConfig,
) -> Result<SolveResult> {
    let use_gmres = match config.strategy {
        SolverStrategy::DirectLU => false,
        SolverStrategy::IterativeGmres => true,
        SolverStrategy::Auto => size >= config.gmres_threshold,
    };

    if use_gmres {
        solve_with_gmres(size, triplets, rhs, &config.gmres_config)
    } else {
        solve_with_lu(size, triplets, rhs)
    }
}

/// Solve using direct sparse LU factorization.
fn solve_with_lu(
    size: usize,
    triplets: &[(usize, usize, f64)],
    rhs: &DVector<f64>,
) -> Result<SolveResult> {
    let solution = if size >= SPARSE_THRESHOLD {
        solve_sparse(size, triplets, rhs)?
    } else {
        // For very small systems, build dense and solve
        use crate::linear::solve_dense;
        use nalgebra::DMatrix;

        let mut matrix = DMatrix::zeros(size, size);
        for &(row, col, value) in triplets {
            matrix[(row, col)] += value;
        }
        solve_dense(&matrix, rhs)?
    };

    Ok(SolveResult {
        solution,
        solver_used: SolverStrategy::DirectLU,
        iterations: None,
        residual: None,
    })
}

/// Solve using iterative GMRES.
fn solve_with_gmres(
    size: usize,
    triplets: &[(usize, usize, f64)],
    rhs: &DVector<f64>,
    gmres_config: &GmresConfig,
) -> Result<SolveResult> {
    let op = SparseRealOperator::from_triplets(size, triplets)
        .ok_or(crate::error::Error::SingularMatrix)?;

    let rhs_slice: Vec<f64> = rhs.iter().copied().collect();
    let result: RealGmresResult = solve_gmres_real(&op, &rhs_slice, gmres_config);

    if !result.converged {
        // GMRES didn't converge - fall back to direct LU
        return solve_with_lu(size, triplets, rhs);
    }

    Ok(SolveResult {
        solution: DVector::from_vec(result.x),
        solver_used: SolverStrategy::IterativeGmres,
        iterations: Some(result.iterations),
        residual: Some(result.residual),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diagonal_system(n: usize) -> (Vec<(usize, usize, f64)>, DVector<f64>) {
        let triplets: Vec<_> = (0..n).map(|i| (i, i, (i + 1) as f64)).collect();
        let rhs = DVector::from_fn(n, |i, _| (i + 1) as f64);
        (triplets, rhs)
    }

    #[test]
    fn test_solver_strategy_from_name() {
        assert_eq!(
            SolverStrategy::from_name("auto"),
            Some(SolverStrategy::Auto)
        );
        assert_eq!(
            SolverStrategy::from_name("lu"),
            Some(SolverStrategy::DirectLU)
        );
        assert_eq!(
            SolverStrategy::from_name("direct"),
            Some(SolverStrategy::DirectLU)
        );
        assert_eq!(
            SolverStrategy::from_name("gmres"),
            Some(SolverStrategy::IterativeGmres)
        );
        assert_eq!(
            SolverStrategy::from_name("iterative"),
            Some(SolverStrategy::IterativeGmres)
        );
        assert_eq!(SolverStrategy::from_name("invalid"), None);
    }

    #[test]
    fn test_solver_strategy_display() {
        assert_eq!(format!("{}", SolverStrategy::Auto), "auto");
        assert_eq!(format!("{}", SolverStrategy::DirectLU), "direct (LU)");
        assert_eq!(
            format!("{}", SolverStrategy::IterativeGmres),
            "iterative (GMRES)"
        );
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.strategy, SolverStrategy::Auto);
        assert_eq!(config.gmres_threshold, 10_000);
    }

    #[test]
    fn test_solve_auto_small_system_uses_lu() {
        let (triplets, rhs) = make_diagonal_system(10);
        let config = SolverConfig::default();

        let result = solve_auto(10, &triplets, &rhs, &config).unwrap();

        assert_eq!(result.solver_used, SolverStrategy::DirectLU);
        assert!(result.iterations.is_none());

        // Check solution: x[i] = 1.0 for all i
        for xi in result.solution.iter() {
            assert!((xi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_solve_auto_force_lu() {
        let (triplets, rhs) = make_diagonal_system(100);
        let config = SolverConfig::direct_lu();

        let result = solve_auto(100, &triplets, &rhs, &config).unwrap();

        assert_eq!(result.solver_used, SolverStrategy::DirectLU);
    }

    #[test]
    fn test_solve_auto_force_gmres() {
        let (triplets, rhs) = make_diagonal_system(100);
        let config = SolverConfig::gmres();

        let result = solve_auto(100, &triplets, &rhs, &config).unwrap();

        assert_eq!(result.solver_used, SolverStrategy::IterativeGmres);
        assert!(result.iterations.is_some());
        assert!(result.residual.is_some());

        // Check solution
        for xi in result.solution.iter() {
            assert!((xi - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_solve_auto_threshold() {
        let config = SolverConfig::with_threshold(50);

        // Below threshold: LU
        let (triplets, rhs) = make_diagonal_system(40);
        let result = solve_auto(40, &triplets, &rhs, &config).unwrap();
        assert_eq!(result.solver_used, SolverStrategy::DirectLU);

        // At/above threshold: GMRES
        let (triplets, rhs) = make_diagonal_system(50);
        let result = solve_auto(50, &triplets, &rhs, &config).unwrap();
        assert_eq!(result.solver_used, SolverStrategy::IterativeGmres);
    }

    #[test]
    fn test_solve_spd_system() {
        // 2x2 SPD system: [[4, 1], [1, 3]] * [1, 1] = [5, 4]
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let rhs = DVector::from_vec(vec![5.0, 4.0]);

        // Test with LU
        let config = SolverConfig::direct_lu();
        let result = solve_auto(2, &triplets, &rhs, &config).unwrap();
        assert!((result.solution[0] - 1.0).abs() < 1e-10);
        assert!((result.solution[1] - 1.0).abs() < 1e-10);

        // Test with GMRES
        let config = SolverConfig::gmres();
        let result = solve_auto(2, &triplets, &rhs, &config).unwrap();
        assert!((result.solution[0] - 1.0).abs() < 1e-6);
        assert!((result.solution[1] - 1.0).abs() < 1e-6);
    }
}
