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

pub use batch::{BatchMosfetType, DiodeBatch, MosfetBatch, round_up_to_simd, SIMD_LANES_AVX2};
pub use behavioral::{BehavioralCurrentSource, BehavioralVoltageSource};
pub use error::{Error, Result};
pub use expression::{Expr, EvalContext, parse_expression};
pub use stamp::Stamp;
pub use waveforms::Waveform;
