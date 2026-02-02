//! Error types for the validation crate.

use std::path::PathBuf;

use thiserror::Error;

/// Result type for validation operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during validation.
#[derive(Debug, Error)]
pub enum Error {
    /// ngspice is not installed or not found in PATH.
    #[error("ngspice not found: {0}")]
    NgspiceNotFound(String),

    /// ngspice execution failed.
    #[error("ngspice execution failed: {0}")]
    NgspiceExecutionFailed(String),

    /// ngspice timed out.
    #[error("ngspice timed out after {0} seconds")]
    NgspiceTimeout(u64),

    /// Failed to parse ngspice rawfile.
    #[error("failed to parse rawfile: {0}")]
    RawfileParseError(String),

    /// Rawfile format not supported.
    #[error("unsupported rawfile format: {0}")]
    UnsupportedRawfileFormat(String),

    /// Spicier parsing failed.
    #[error("spicier parse error: {0}")]
    SpicierParseError(#[from] spicier_parser::Error),

    /// Spicier solver failed.
    #[error("spicier solver error: {0}")]
    SpicierSolverError(#[from] spicier_solver::Error),

    /// Variable not found in results.
    #[error("variable not found: {0}")]
    VariableNotFound(String),

    /// Comparison failed (results differ beyond tolerance).
    #[error("comparison failed: {0}")]
    ComparisonFailed(String),

    /// Analysis type mismatch.
    #[error("analysis type mismatch: expected {expected}, got {actual}")]
    AnalysisTypeMismatch { expected: String, actual: String },

    /// No analysis command in netlist.
    #[error("no analysis command found in netlist")]
    NoAnalysisCommand,

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Golden data file not found.
    #[error("golden data file not found: {path}")]
    GoldenDataNotFound { path: PathBuf },

    /// Invalid golden data format.
    #[error("invalid golden data format: {0}")]
    InvalidGoldenData(String),

    /// Temp file error.
    #[error("temp file error: {0}")]
    TempFile(String),

    /// Unsupported analysis type.
    #[error("unsupported analysis type: {0}")]
    UnsupportedAnalysis(String),
}
