//! Linear and nonlinear solvers for Spicier.
//!
//! This crate provides:
//! - Linear system solvers (dense and sparse, real and complex)
//! - DC operating point analysis
//! - AC small-signal frequency-domain analysis
//! - Transient time-domain analysis
//! - Newton-Raphson iteration for nonlinear circuits

pub mod ac;
pub mod dc;
pub mod error;
pub mod linear;
pub mod newton;
pub mod transient;

pub use ac::{
    AcParams, AcResult, AcStamper, AcSweepType, ComplexMna, generate_frequencies, solve_ac,
};
pub use dc::{DcSolution, DcSweepParams, DcSweepResult, DcSweepStamper, solve_dc, solve_dc_sweep};
pub use error::{Error, Result};
pub use newton::{ConvergenceCriteria, NonlinearStamper, NrResult, solve_newton_raphson};
pub use transient::{
    CapacitorState, InductorState, IntegrationMethod, TransientParams, TransientResult,
    TransientStamper, solve_transient,
};
