//! Cross-simulator validation tool for comparing spicier with ngspice.
//!
//! This crate provides infrastructure for:
//! - Running circuits through ngspice and capturing results
//! - Running the same circuits through spicier
//! - Comparing results with configurable tolerances
//! - Generating validation reports
//!
//! # Status
//!
//! This crate is under development.

pub mod error;

mod compare;
mod golden;
mod ngspice;
// mod spicier;  // TODO: Fix API compatibility

pub use error::{Error as ValidationError, Result as ValidationResult};
