//! Error types for spicier-core.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("duplicate node: {0}")]
    DuplicateNode(String),

    #[error("invalid circuit: {0}")]
    InvalidCircuit(String),

    #[error("matrix error: {0}")]
    MatrixError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
