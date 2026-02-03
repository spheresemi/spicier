//! Spectral analysis for transient waveforms.
//!
//! This module provides FFT-based spectral analysis and total harmonic distortion (THD)
//! computation for transient simulation results.
//!
//! # Features
//!
//! - **FFT Analysis** - Compute magnitude, phase, and power spectral density
//! - **Window Functions** - Hanning, Hamming, Blackman, Rectangular
//! - **THD Computation** - Extract harmonics and compute THD percentage
//!
//! # Example
//!
//! ```ignore
//! use spicier_solver::spectral::{compute_fft, compute_thd, SpectralConfig, WindowFunction};
//!
//! // Analyze a transient result at node 1
//! let config = SpectralConfig::default();
//! let spectrum = compute_fft(&transient_result, 1, &config)?;
//!
//! // Get the dominant frequency
//! let peak_idx = spectrum.magnitude.iter()
//!     .enumerate()
//!     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//!     .map(|(i, _)| i)
//!     .unwrap();
//! let dominant_freq = spectrum.frequencies[peak_idx];
//!
//! // Compute THD for a 1kHz fundamental
//! let thd = compute_thd(&transient_result, 1, 1000.0, 10)?;
//! println!("THD: {:.2}%", thd.thd_percent);
//! ```

mod fft;
mod thd;
mod window;

pub use fft::{
    SpectralConfig, SpectralResult, compute_fft, compute_fft_from_samples, resample_uniform,
};
pub use thd::{HarmonicInfo, ThdResult, compute_thd, compute_thd_from_samples};
pub use window::WindowFunction;
