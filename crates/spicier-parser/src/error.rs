//! Error types for spicier-parser.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("unknown element type: {0}")]
    UnknownElement(String),

    #[error("invalid value: {0}")]
    InvalidValue(String),

    #[error("missing node: {0}")]
    MissingNode(String),
}

pub type Result<T> = std::result::Result<T, Error>;
