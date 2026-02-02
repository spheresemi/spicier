//! Device models and MNA stamps for Spicier.
//!
//! This crate provides device models for:
//! - Passive elements: R, C, L
//! - Sources: V, I (independent) with time-varying waveforms
//! - Nonlinear devices: Diode, MOSFET
//! - Behavioral sources: B (arbitrary expressions)
//! - Batched device evaluation with SIMD-friendly SoA layout

pub mod batch;
pub mod behavioral;
pub mod controlled;
pub mod diode;
pub mod error;
pub mod expression;
pub mod mosfet;
pub mod passive;
pub mod sources;
pub mod stamp;
pub mod waveforms;

// Re-export batch types
pub use batch::{BatchMosfetType, DiodeBatch, MosfetBatch, SIMD_LANES_AVX2, round_up_to_simd};

// Re-export behavioral sources
pub use behavioral::{BehavioralCurrentSource, BehavioralVoltageSource};

// Re-export controlled sources
pub use controlled::{Cccs, Ccvs, Vccs, Vcvs};

// Re-export diode
pub use diode::{Diode, DiodeParams};

// Re-export error types
pub use error::{Error, Result};

// Re-export expression types
pub use expression::{EvalContext, Expr, parse_expression};

// Re-export MOSFET
pub use mosfet::{Mosfet, MosfetParams, MosfetRegion, MosfetType};

// Re-export passive elements
pub use passive::{Capacitor, Inductor, Resistor};

// Re-export sources
pub use sources::{CurrentSource, VoltageSource};

// Re-export stamp trait
pub use stamp::Stamp;

// Re-export waveforms
pub use waveforms::Waveform;
