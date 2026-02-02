//! Linear and nonlinear solvers for Spicier.
//!
//! This crate provides:
//! - Linear system solvers (dense and sparse, real and complex)
//! - DC operating point analysis
//! - AC small-signal frequency-domain analysis
//! - Transient time-domain analysis
//! - Newton-Raphson iteration for nonlinear circuits

pub mod ac;
pub mod backend;
pub mod batched_newton;
pub mod dc;
pub mod dispatch;
pub mod error;
pub mod gmres;
pub mod linear;
pub mod newton;
pub mod operator;
pub mod parallel;
pub mod preconditioner;
pub mod solver_select;
pub mod sparse_operator;
pub mod sweep;
pub mod transient;

pub use ac::{
    AcParams, AcResult, AcStamper, AcSweepType, ComplexMna, generate_frequencies, solve_ac,
    solve_ac_dispatched,
};
pub use dc::{
    DcSolution, DcSweepParams, DcSweepResult, DcSweepStamper, solve_dc, solve_dc_dispatched,
    solve_dc_sweep, solve_dc_sweep_dispatched,
};
pub use error::{Error, Result};
pub use linear::{CachedSparseLu, CachedSparseLuComplex};
pub use newton::{ConvergenceCriteria, NonlinearStamper, NrResult, solve_newton_raphson};
pub use batched_newton::{
    BatchedNonlinearDevices, LinearStamper, solve_batched_newton_raphson,
};
pub use transient::{
    AdaptiveTransientParams, AdaptiveTransientResult, CapacitorState, InductorState,
    InitialConditions, IntegrationMethod, TransientParams, TransientResult, TransientStamper,
    solve_transient, solve_transient_adaptive, solve_transient_dispatched,
};
pub use backend::ComputeBackend;
pub use gmres::{
    GmresConfig, GmresResult, RealGmresResult, solve_gmres, solve_gmres_preconditioned,
    solve_gmres_real, solve_gmres_real_preconditioned,
};
pub use operator::{ComplexOperator, RealOperator};
pub use preconditioner::{
    ComplexJacobiPreconditioner, ComplexPreconditioner, IdentityPreconditioner,
    JacobiPreconditioner, RealPreconditioner,
};
pub use solver_select::{SolveResult, SolverConfig, SolverStrategy, solve_auto};
pub use sparse_operator::{SparseComplexOperator, SparseRealOperator};
pub use parallel::{
    ParallelTripletAccumulator, parallel_ranges, stamp_conductance_triplets,
    stamp_current_source_rhs,
};
pub use sweep::{
    BatchedSweepResult, CornerGenerator, LinearSweepGenerator, MonteCarloGenerator,
    ParameterVariation, SweepPoint, SweepPointGenerator, SweepStamper, SweepStamperFactory,
    SweepStatistics, solve_batched_sweep,
};
pub use dispatch::{DispatchConfig, DispatchedSolveInfo, SolverDispatchStrategy};
