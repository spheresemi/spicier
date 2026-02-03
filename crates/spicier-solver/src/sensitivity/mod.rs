//! Sensitivity analysis for circuit simulation.
//!
//! This module provides sensitivity analysis using forward finite differences.
//! It computes how circuit outputs (voltages, currents) change with respect
//! to parameter variations.
//!
//! # Features
//!
//! - **DC Sensitivity** - Operating point sensitivity to parameter changes
//! - **AC Sensitivity** - Frequency response sensitivity
//! - **Normalized Sensitivity** - Dimensionless relative sensitivities
//!
//! # Example
//!
//! ```ignore
//! use spicier_solver::sensitivity::{
//!     SensitivityConfig, SensitivityParam, compute_dc_sensitivity
//! };
//!
//! // Define what parameters to vary
//! let config = SensitivityConfig {
//!     delta_ratio: 1e-6,
//!     params: vec![
//!         SensitivityParam::Resistance { name: "R1".to_string(), value: 1000.0 },
//!     ],
//!     outputs: vec!["V(2)".to_string()],
//! };
//!
//! // Compute sensitivities
//! let results = compute_dc_sensitivity(&stamper, &config)?;
//!
//! for r in &results {
//!     println!("{}: dVout/d{} = {:.4}", r.output, r.param_name(), r.value);
//!     println!("  Normalized: {:.4}", r.normalized);
//! }
//! ```

mod ac;
mod config;
mod dc;

pub use ac::{
    AcSensitivityResult, AcSensitivityStamper, compute_ac_sensitivity, compute_ac_sensitivity_sweep,
};
pub use config::{SensitivityConfig, SensitivityOutput, SensitivityParam};
pub use dc::{DcSensitivityResult, DcSensitivityStamper, compute_dc_sensitivity};
