//! Error types for spicier-solver.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("singular matrix")]
    SingularMatrix,

    #[error("convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("invalid matrix dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("solver error: {0}")]
    SolverError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
