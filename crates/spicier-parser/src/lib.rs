//! SPICE netlist parser for Spicier.
//!
//! This crate provides parsing of SPICE netlists into circuit representations.
//!
//! # Example
//!
//! ```
//! use spicier_parser::parse;
//!
//! let netlist = parse(r#"
//! Voltage Divider
//! V1 1 0 10
//! R1 1 2 1k
//! R2 2 0 1k
//! .end
//! "#).unwrap();
//!
//! assert_eq!(netlist.num_devices(), 3);
//! ```

pub mod error;
pub mod lexer;
pub mod parser;

pub use error::{Error, Result};
pub use parser::{AcSweepType, AnalysisCommand, InitialCondition, ParseResult, parse, parse_full};
