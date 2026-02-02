//! Noise analysis for circuit simulation.
//!
//! This module provides small-signal noise analysis, computing the noise
//! contributed by each device to a specified output node.
//!
//! # Noise Sources
//!
//! - **Thermal noise** (Johnson-Nyquist): Resistors generate voltage noise
//!   with spectral density Sv = 4kTR (V²/Hz)
//! - **Shot noise**: Current through PN junctions generates noise with
//!   spectral density Si = 2qI (A²/Hz)
//! - **Flicker noise** (1/f): MOSFETs and BJTs exhibit low-frequency noise
//!   with spectral density proportional to 1/f
//!
//! # Analysis Method
//!
//! 1. Compute DC operating point to determine bias currents
//! 2. For each frequency point:
//!    - Compute AC transfer function from each noise source to output
//!    - Multiply by noise spectral density
//!    - Sum squared contributions (noise adds in power, not amplitude)
//! 3. Report total output noise, input-referred noise, and per-device contributions
//!
//! # Example
//!
//! ```ignore
//! use spicier_solver::noise::{NoiseConfig, NoiseResult, compute_noise};
//!
//! let config = NoiseConfig {
//!     output_node: 2,        // V(2) is the output
//!     input_source: Some(0), // V1 is the input (for input-referred noise)
//!     fstart: 1.0,
//!     fstop: 1e6,
//!     num_points: 10,
//!     sweep_type: NoiseSweepType::Decade,
//!     temperature: 300.0,    // 27°C
//! };
//!
//! let result = compute_noise(&stamper, &config)?;
//! println!("Total output noise at 1kHz: {} V/√Hz", result.output_noise_at(1000.0));
//! ```

mod sources;
mod analysis;

pub use sources::{NoiseSource, NoiseSourceType};
pub use analysis::{
    NoiseConfig, NoiseContribution, NoiseResult, NoiseSweepType, NoiseStamper,
    compute_noise,
};
