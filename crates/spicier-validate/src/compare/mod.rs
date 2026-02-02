//! Comparison logic for validating simulation results.
//!
//! This module provides functionality to compare results from ngspice and spicier,
//! with configurable tolerances for different analysis types.

// TODO: Re-enable when spicier runner module API is complete
// pub mod ac;
// pub mod dc;
pub mod report;
pub mod tolerances;
// pub mod transient;

// pub use ac::compare_ac;
// pub use dc::compare_dc_op;
pub use report::{ComparisonReport, ComparisonSummary, VariableComparison, WorstPointInfo};
pub use tolerances::{
    AcTolerances, ComparisonConfig, DcTolerances, TransientTolerances, relative_error,
    values_match,
};
// pub use transient::compare_transient;
