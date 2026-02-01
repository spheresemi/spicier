//! Device models and MNA stamps for Spicier.
//!
//! This crate provides device models for:
//! - Passive elements: R, C, L
//! - Sources: V, I (independent)
//! - Nonlinear devices: Diode, MOSFET (future)

pub mod diode;
pub mod error;
pub mod mosfet;
pub mod passive;
pub mod sources;
pub mod stamp;

pub use error::{Error, Result};
pub use stamp::Stamp;
