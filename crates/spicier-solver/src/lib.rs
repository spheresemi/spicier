//! Linear and nonlinear solvers for Spicier.
//!
//! This crate provides simulation engines for circuit analysis:
//!
//! - **DC Analysis** - Operating point and parameter sweeps
//! - **AC Analysis** - Small-signal frequency response
//! - **Transient Analysis** - Time-domain simulation
//! - **Newton-Raphson** - Nonlinear circuit convergence
//!
//! # Analysis Types
//!
//! ## DC Operating Point
//!
//! Find the steady-state (DC) solution for a circuit:
//!
//! ```rust
//! use spicier_core::mna::MnaSystem;
//! use spicier_solver::solve_dc;
//!
//! // Create MNA system for voltage divider
//! let mut mna = MnaSystem::new(2, 1);
//! mna.stamp_voltage_source(Some(0), None, 0, 10.0);  // V1 = 10V
//! mna.stamp_conductance(Some(0), Some(1), 1e-3);      // R1 = 1kΩ
//! mna.stamp_conductance(Some(1), None, 1e-3);         // R2 = 1kΩ
//!
//! let solution = solve_dc(&mna).expect("solve failed");
//! let v1 = solution.voltage(spicier_core::NodeId::new(1));
//! let v2 = solution.voltage(spicier_core::NodeId::new(2));
//! assert!((v1 - 10.0).abs() < 1e-9);
//! assert!((v2 - 5.0).abs() < 1e-9);  // Voltage divider: 10V / 2
//! ```
//!
//! ## AC Small-Signal Analysis
//!
//! Compute frequency response using the [`AcStamper`] trait:
//!
//! ```rust
//! use num_complex::Complex;
//! use spicier_solver::{AcParams, AcStamper, AcSweepType, ComplexMna, solve_ac};
//!
//! struct RcLowpass;
//!
//! impl AcStamper for RcLowpass {
//!     fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
//!         // V1 = 1V AC source at node 0
//!         mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));
//!         // R = 1kΩ from node 0 to node 1
//!         mna.stamp_conductance(Some(0), Some(1), 1e-3);
//!         // C = 159nF from node 1 to ground (f_c ≈ 1kHz)
//!         let yc = Complex::new(0.0, omega * 159e-9);
//!         mna.stamp_admittance(Some(1), None, yc);
//!     }
//!     fn num_nodes(&self) -> usize { 2 }
//!     fn num_vsources(&self) -> usize { 1 }
//! }
//!
//! let params = AcParams {
//!     sweep_type: AcSweepType::Decade,
//!     num_points: 10,
//!     fstart: 100.0,
//!     fstop: 10000.0,
//! };
//!
//! let result = solve_ac(&RcLowpass, &params).expect("AC solve failed");
//! let mag_db = result.magnitude_db(1);  // Get magnitude in dB at node 1
//! ```
//!
//! ## Transient Analysis
//!
//! Time-domain simulation with capacitors and inductors:
//!
//! ```rust
//! use nalgebra::DVector;
//! use spicier_core::mna::MnaSystem;
//! use spicier_solver::{
//!     TransientParams, TransientStamper, CapacitorState,
//!     IntegrationMethod, solve_transient,
//! };
//!
//! struct RcCircuit;
//!
//! impl TransientStamper for RcCircuit {
//!     fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
//!         mna.stamp_voltage_source(Some(0), None, 0, 5.0);  // V1 = 5V
//!         mna.stamp_conductance(Some(0), Some(1), 1e-3);     // R = 1kΩ
//!     }
//!     fn num_nodes(&self) -> usize { 2 }
//!     fn num_vsources(&self) -> usize { 1 }
//! }
//!
//! let dc = DVector::from_vec(vec![5.0, 0.0]);  // Initial condition
//! let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];  // C = 1µF
//!
//! let params = TransientParams {
//!     tstop: 5e-3,  // 5ms
//!     tstep: 1e-4,  // 100µs
//!     method: IntegrationMethod::Trapezoidal,
//! };
//!
//! let result = solve_transient(&RcCircuit, &mut caps, &mut vec![], &params, &dc)
//!     .expect("transient solve failed");
//! ```
//!
//! # Solver Selection
//!
//! The crate automatically selects between:
//! - **Dense LU** - For small circuits (< 100 nodes)
//! - **Sparse LU** - For medium circuits (100-10000 nodes)
//! - **GMRES** - For large circuits (> 10000 nodes)
//!
//! Use [`DispatchConfig`] to customize solver selection.
//!
//! # Convergence Aids
//!
//! For difficult nonlinear circuits:
//! - [`solve_with_source_stepping`] - Gradually ramp sources
//! - [`solve_with_gmin_stepping`] - Add minimum conductance

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
pub use newton::{
    ConvergenceCriteria, GminSteppingParams, GminSteppingResult, NonlinearStamper, NrResult,
    ScaledNonlinearStamper, SourceSteppingParams, SourceSteppingResult, solve_newton_raphson,
    solve_with_gmin_stepping, solve_with_source_stepping,
};
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
