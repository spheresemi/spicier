//! # Spicier
//!
//! A high-performance SPICE circuit simulator written in Rust.
//!
//! Spicier provides a complete circuit simulation toolkit including:
//! - SPICE netlist parsing
//! - DC operating point and sweep analysis
//! - AC small-signal frequency response
//! - Transient time-domain simulation
//! - Support for linear and nonlinear devices
//!
//! ## Quick Start
//!
//! ```rust
//! use spicier::prelude::*;
//!
//! // Parse a simple voltage divider
//! let netlist = "Voltage Divider\nV1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n.OP\n.END\n";
//!
//! let result = spicier::parse_full(netlist).unwrap();
//! println!("Circuit has {} nodes", result.netlist.num_nodes());
//! ```
//!
//! ## Running Simulations
//!
//! ```rust,ignore
//! use spicier::prelude::*;
//!
//! // Parse netlist
//! let result = spicier::parse(netlist)?;
//!
//! // Run DC operating point
//! let dc = spicier::solve_dc(&stamper)?;
//! println!("V(2) = {} V", dc.voltage(1));
//!
//! // Run AC analysis
//! let ac_params = AcParams {
//!     sweep_type: AcSweepType::Decade,
//!     num_points: 10,
//!     fstart: 1.0,
//!     fstop: 1e6,
//! };
//! let ac = spicier::solve_ac(&ac_stamper, &ac_params)?;
//!
//! // Run transient analysis
//! let tran_params = TransientParams::new(1e-6, 1e-3);
//! let tran = spicier::solve_transient(&tran_stamper, caps, inds, &tran_params, &dc)?;
//! ```
//!
//! ## Features
//!
//! - `cuda` - Enable CUDA GPU acceleration (requires NVIDIA GPU)
//! - `metal` - Enable Metal/WebGPU acceleration (macOS/cross-platform)
//! - `full` - Enable all optional features

// Re-export core crates
pub use spicier_core as core;
pub use spicier_devices as devices;
pub use spicier_parser as parser;
pub use spicier_simd as simd;
pub use spicier_solver as solver;

// Re-export CPU backend
pub use spicier_backend_cpu as backend_cpu;

// Conditional re-exports for GPU backends
#[cfg(feature = "cuda")]
pub use spicier_backend_cuda as backend_cuda;

#[cfg(feature = "metal")]
pub use spicier_backend_metal as backend_metal;

// ============================================================================
// Convenient re-exports from spicier_core
// ============================================================================

pub use spicier_core::{
    AcDeviceInfo,
    // Circuit representation
    Circuit,
    // Errors
    Error as CoreError,
    // Netlist
    Netlist,
    Node,
    NodeId,
    // Stamper traits
    Stamper,
    TransientDeviceInfo,
};

// MNA system (exported from submodule)
pub use spicier_core::mna::MnaSystem;

// ============================================================================
// Convenient re-exports from spicier_parser
// ============================================================================

pub use spicier_parser::{
    AcSweepType as ParserAcSweepType,
    // Analysis commands
    AnalysisCommand,
    // Errors
    Error as ParseError,
    // Result types
    ParseResult,
    // Main parse function
    parse,
    parse_full,
};

// ============================================================================
// Convenient re-exports from spicier_solver
// ============================================================================

pub use spicier_solver::{
    AcParams,
    AcResult,
    AcSweepType,
    AdaptiveTransientParams,
    AdaptiveTransientResult,
    ComplexOperator,
    // Backends
    ComputeBackend,
    ConvergenceCriteria,
    DcSolution,
    DcSweepParams,
    DcSweepResult,
    DispatchConfig,
    // Errors
    Error as SolverError,
    // GMRES
    GmresConfig,
    IntegrationMethod,
    // Operators
    RealOperator,
    SolverConfig,
    // Solver selection
    SolverStrategy,
    TransientParams,
    TransientResult,
    // AC analysis
    solve_ac,
    solve_ac_dispatched,
    // DC analysis
    solve_dc,
    // Dispatched solvers
    solve_dc_dispatched,
    // DC sweep
    solve_dc_sweep,
    // Newton-Raphson
    solve_newton_raphson,
    // Transient analysis
    solve_transient,
    // Adaptive transient
    solve_transient_adaptive,
    solve_transient_dispatched,
};

// ============================================================================
// Convenient re-exports from spicier_devices
// ============================================================================

pub use spicier_devices::{
    BehavioralCurrentSource,
    // Behavioral
    BehavioralVoltageSource,
    Capacitor,
    Cccs,
    Ccvs,
    CurrentSource,
    // Semiconductors
    Diode,
    DiodeParams,
    // Errors
    Error as DeviceError,
    Inductor,
    Mosfet,
    MosfetParams,
    MosfetType,
    // Passive elements
    Resistor,
    Vccs,
    // Controlled sources
    Vcvs,
    // Sources
    VoltageSource,
    // Waveforms
    Waveform,
};

// ============================================================================
// Re-export commonly used external types
// ============================================================================

/// Re-export of nalgebra's dynamic vector type.
pub use nalgebra::DVector;

/// Re-export of nalgebra's dynamic matrix type.
pub use nalgebra::DMatrix;

/// Re-export of num_complex's Complex type.
pub use num_complex::Complex;

// ============================================================================
// Prelude module for convenient imports
// ============================================================================

/// Prelude module containing commonly used types and traits.
///
/// ```rust
/// use spicier::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use crate::{Circuit, MnaSystem, Netlist, Node, NodeId, Stamper};

    // Parser
    pub use crate::{AnalysisCommand, ParseResult, parse, parse_full};

    // Solver - DC
    pub use crate::{DcSolution, solve_dc};

    // Solver - AC
    pub use crate::{AcParams, AcResult, AcSweepType, solve_ac};

    // Solver - Transient
    pub use crate::{IntegrationMethod, TransientParams, TransientResult, solve_transient};

    // Solver - Newton-Raphson
    pub use crate::{ConvergenceCriteria, solve_newton_raphson};

    // Devices
    pub use crate::{
        Capacitor, CurrentSource, Diode, Inductor, Mosfet, MosfetType, Resistor, VoltageSource,
        Waveform,
    };

    // Common external types
    pub use crate::{Complex, DMatrix, DVector};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_circuit() {
        let netlist = "Test\nV1 1 0 DC 5\nR1 1 0 1k\n.op\n.end\n";
        let result = parse(netlist);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_full_with_analysis() {
        let netlist = "Test\nV1 1 0 DC 5\nR1 1 0 1k\n.op\n.end\n";
        let result = parse_full(netlist).unwrap();
        assert!(!result.analyses.is_empty());
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        // Verify types are accessible
        let _: NodeId = NodeId::GROUND;
        let r = Resistor::new("R1", NodeId::new(1), NodeId::new(2), 1000.0);
        assert_eq!(r.resistance, 1000.0);
    }
}
