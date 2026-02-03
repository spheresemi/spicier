//! Total Harmonic Distortion (THD) computation.
//!
//! THD measures the ratio of harmonic content to the fundamental frequency,
//! expressed as a percentage:
//!
//! THD = sqrt(V2² + V3² + ... + Vn²) / V1 × 100%
//!
//! where V1 is the fundamental magnitude and V2..Vn are harmonic magnitudes.

use super::fft::{SpectralConfig, compute_fft, compute_fft_from_samples};
use super::window::WindowFunction;
use crate::transient::TransientResult;

/// Information about a single harmonic.
#[derive(Debug, Clone)]
pub struct HarmonicInfo {
    /// Harmonic number (1 = fundamental, 2 = second harmonic, etc.)
    pub harmonic_number: usize,
    /// Frequency in Hz.
    pub frequency: f64,
    /// Magnitude (linear).
    pub magnitude: f64,
    /// Magnitude in dB.
    pub magnitude_db: f64,
    /// Phase in radians.
    pub phase: f64,
    /// Magnitude relative to fundamental (percentage).
    pub relative_percent: f64,
}

/// Result of THD analysis.
#[derive(Debug, Clone)]
pub struct ThdResult {
    /// THD as a percentage.
    pub thd_percent: f64,
    /// THD in dB (20*log10(THD/100)).
    pub thd_db: f64,
    /// Fundamental frequency in Hz.
    pub fundamental_freq: f64,
    /// Fundamental magnitude.
    pub fundamental_magnitude: f64,
    /// Information about each harmonic (including fundamental).
    pub harmonics: Vec<HarmonicInfo>,
    /// Total RMS of all harmonics (excluding fundamental).
    pub harmonic_rms: f64,
    /// Number of harmonics included in calculation.
    pub num_harmonics: usize,
}

/// Compute THD from a transient waveform.
///
/// # Arguments
/// * `waveform` - Transient simulation result
/// * `node_idx` - Index of the node to analyze (0-based)
/// * `fundamental_freq` - Expected fundamental frequency in Hz
/// * `num_harmonics` - Number of harmonics to analyze (excluding fundamental)
///
/// # Returns
/// THD result with harmonic breakdown
pub fn compute_thd(
    waveform: &TransientResult,
    node_idx: usize,
    fundamental_freq: f64,
    num_harmonics: usize,
) -> ThdResult {
    // Use Hanning window for best frequency resolution without excessive leakage
    let config = SpectralConfig {
        window: WindowFunction::Hanning,
        fft_size: None,
        zero_padding: true,
    };

    let spectrum = compute_fft(waveform, node_idx, &config);
    compute_thd_from_spectrum(&spectrum, fundamental_freq, num_harmonics)
}

/// Compute THD from raw samples.
///
/// # Arguments
/// * `samples` - Uniformly sampled signal
/// * `sample_rate` - Sample rate in Hz
/// * `fundamental_freq` - Expected fundamental frequency in Hz
/// * `num_harmonics` - Number of harmonics to analyze (excluding fundamental)
pub fn compute_thd_from_samples(
    samples: &[f64],
    sample_rate: f64,
    fundamental_freq: f64,
    num_harmonics: usize,
) -> ThdResult {
    let config = SpectralConfig {
        window: WindowFunction::Hanning,
        fft_size: None,
        zero_padding: true,
    };

    let spectrum = compute_fft_from_samples(samples, sample_rate, &config);
    compute_thd_from_spectrum(&spectrum, fundamental_freq, num_harmonics)
}

/// Find the bin index closest to a target frequency.
fn find_bin_for_frequency(frequencies: &[f64], target_freq: f64) -> Option<usize> {
    if frequencies.is_empty() || target_freq < 0.0 {
        return None;
    }

    let mut best_idx = 0;
    let mut best_diff = (frequencies[0] - target_freq).abs();

    for (i, &f) in frequencies.iter().enumerate() {
        let diff = (f - target_freq).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }

    // Verify we're reasonably close to the target
    let freq_resolution = if frequencies.len() > 1 {
        frequencies[1] - frequencies[0]
    } else {
        1.0
    };

    // Return best match regardless of how close it is
    let _ = freq_resolution; // Used for potential future threshold check
    Some(best_idx)
}

/// Search for the actual peak near an expected frequency.
///
/// This handles cases where the actual frequency differs slightly from expected.
fn find_peak_near_frequency(
    frequencies: &[f64],
    magnitudes: &[f64],
    target_freq: f64,
    search_range: f64,
) -> Option<usize> {
    if frequencies.is_empty() {
        return None;
    }

    let mut best_idx = None;
    let mut best_mag = 0.0;

    for (i, (&freq, &mag)) in frequencies.iter().zip(magnitudes.iter()).enumerate() {
        if (freq - target_freq).abs() <= search_range && mag > best_mag {
            best_mag = mag;
            best_idx = Some(i);
        }
    }

    best_idx
}

/// Compute THD from a pre-computed spectrum.
fn compute_thd_from_spectrum(
    spectrum: &super::fft::SpectralResult,
    fundamental_freq: f64,
    num_harmonics: usize,
) -> ThdResult {
    if spectrum.frequencies.is_empty() || fundamental_freq <= 0.0 {
        return ThdResult {
            thd_percent: 0.0,
            thd_db: f64::NEG_INFINITY,
            fundamental_freq,
            fundamental_magnitude: 0.0,
            harmonics: Vec::new(),
            harmonic_rms: 0.0,
            num_harmonics: 0,
        };
    }

    // Frequency resolution for peak search
    let freq_resolution = if spectrum.frequencies.len() > 1 {
        spectrum.frequencies[1] - spectrum.frequencies[0]
    } else {
        fundamental_freq * 0.01
    };
    let search_range = freq_resolution * 2.0;

    let mut harmonics = Vec::with_capacity(num_harmonics + 1);

    // Find fundamental (harmonic 1)
    let fund_idx = find_peak_near_frequency(
        &spectrum.frequencies,
        &spectrum.magnitude,
        fundamental_freq,
        search_range,
    );

    let (fundamental_magnitude, fund_phase, actual_fund_freq) = match fund_idx {
        Some(idx) => (
            spectrum.magnitude[idx],
            spectrum.phase[idx],
            spectrum.frequencies[idx],
        ),
        None => {
            // Fall back to expected frequency bin
            let idx = find_bin_for_frequency(&spectrum.frequencies, fundamental_freq).unwrap_or(0);
            (
                spectrum.magnitude.get(idx).copied().unwrap_or(0.0),
                spectrum.phase.get(idx).copied().unwrap_or(0.0),
                spectrum
                    .frequencies
                    .get(idx)
                    .copied()
                    .unwrap_or(fundamental_freq),
            )
        }
    };

    // Fundamental harmonic info
    harmonics.push(HarmonicInfo {
        harmonic_number: 1,
        frequency: actual_fund_freq,
        magnitude: fundamental_magnitude,
        magnitude_db: if fundamental_magnitude > 1e-20 {
            20.0 * fundamental_magnitude.log10()
        } else {
            -400.0
        },
        phase: fund_phase,
        relative_percent: 100.0,
    });

    // Find higher harmonics
    let mut harmonic_sum_sq = 0.0;
    let mut found_harmonics = 0;

    for n in 2..=(num_harmonics + 1) {
        let target_freq = fundamental_freq * n as f64;

        // Skip if beyond Nyquist
        if target_freq > spectrum.sample_rate / 2.0 {
            break;
        }

        let (mag, phase_val, actual_freq) = match find_peak_near_frequency(
            &spectrum.frequencies,
            &spectrum.magnitude,
            target_freq,
            search_range,
        ) {
            Some(idx) => (
                spectrum.magnitude[idx],
                spectrum.phase[idx],
                spectrum.frequencies[idx],
            ),
            None => {
                // Fall back to expected bin
                match find_bin_for_frequency(&spectrum.frequencies, target_freq) {
                    Some(idx) => (
                        spectrum.magnitude.get(idx).copied().unwrap_or(0.0),
                        spectrum.phase.get(idx).copied().unwrap_or(0.0),
                        spectrum
                            .frequencies
                            .get(idx)
                            .copied()
                            .unwrap_or(target_freq),
                    ),
                    None => (0.0, 0.0, target_freq),
                }
            }
        };

        let relative_percent = if fundamental_magnitude > 1e-20 {
            (mag / fundamental_magnitude) * 100.0
        } else {
            0.0
        };

        harmonics.push(HarmonicInfo {
            harmonic_number: n,
            frequency: actual_freq,
            magnitude: mag,
            magnitude_db: if mag > 1e-20 {
                20.0 * mag.log10()
            } else {
                -400.0
            },
            phase: phase_val,
            relative_percent,
        });

        harmonic_sum_sq += mag * mag;
        found_harmonics += 1;
    }

    let harmonic_rms = harmonic_sum_sq.sqrt();

    let thd_percent = if fundamental_magnitude > 1e-20 {
        (harmonic_rms / fundamental_magnitude) * 100.0
    } else {
        0.0
    };

    let thd_db = if thd_percent > 1e-20 {
        20.0 * (thd_percent / 100.0).log10()
    } else {
        f64::NEG_INFINITY
    };

    ThdResult {
        thd_percent,
        thd_db,
        fundamental_freq: actual_fund_freq,
        fundamental_magnitude,
        harmonics,
        harmonic_rms,
        num_harmonics: found_harmonics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn create_signal_with_harmonics(
        fundamental: f64,
        harmonic_amplitudes: &[f64], // [fund, 2nd, 3rd, ...]
        sample_rate: f64,
        duration: f64,
    ) -> Vec<f64> {
        let n = (sample_rate * duration) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                harmonic_amplitudes
                    .iter()
                    .enumerate()
                    .map(|(h, &amp)| amp * (2.0 * PI * fundamental * (h + 1) as f64 * t).sin())
                    .sum()
            })
            .collect()
    }

    #[test]
    fn test_pure_sine_zero_thd() {
        // Pure sine wave should have THD ≈ 0%
        let samples = create_signal_with_harmonics(1000.0, &[1.0], 44100.0, 0.1);
        let result = compute_thd_from_samples(&samples, 44100.0, 1000.0, 10);

        assert!(
            result.thd_percent < 1.0,
            "Pure sine THD {} should be < 1%",
            result.thd_percent
        );
    }

    #[test]
    fn test_known_harmonics() {
        // Signal with known harmonic content:
        // Fundamental: 1.0
        // 2nd harmonic: 0.1 (10%)
        // 3rd harmonic: 0.05 (5%)
        // Expected THD = sqrt(0.1² + 0.05²) / 1.0 × 100 = sqrt(0.0125) × 100 ≈ 11.18%
        let samples = create_signal_with_harmonics(1000.0, &[1.0, 0.1, 0.05], 44100.0, 0.1);
        let result = compute_thd_from_samples(&samples, 44100.0, 1000.0, 10);

        let expected_thd = (0.1_f64.powi(2) + 0.05_f64.powi(2)).sqrt() * 100.0;
        assert!(
            (result.thd_percent - expected_thd).abs() < 2.0,
            "THD {} should be near {}%",
            result.thd_percent,
            expected_thd
        );

        // Check that we found the correct number of significant harmonics
        assert!(result.harmonics.len() >= 3);
        assert_eq!(result.harmonics[0].harmonic_number, 1);
        assert_eq!(result.harmonics[1].harmonic_number, 2);
    }

    #[test]
    fn test_square_wave_thd() {
        // Square wave has harmonics at odd multiples: 1, 3, 5, 7, ...
        // Amplitude of nth harmonic = 4/(n*π)
        // THD should be quite high (~48%)
        let fundamental = 1000.0;
        let sample_rate = 100000.0;
        let n = (sample_rate * 0.1) as usize;

        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let phase = 2.0 * PI * fundamental * t;
                if phase.sin() >= 0.0 { 1.0 } else { -1.0 }
            })
            .collect();

        let result = compute_thd_from_samples(&samples, sample_rate, fundamental, 20);

        // Square wave THD should be roughly 48%
        assert!(
            result.thd_percent > 30.0 && result.thd_percent < 60.0,
            "Square wave THD {} should be roughly 48%",
            result.thd_percent
        );
    }

    #[test]
    fn test_harmonic_relative_percentages() {
        // Create signal where 2nd harmonic is exactly 10% of fundamental
        let samples = create_signal_with_harmonics(1000.0, &[1.0, 0.1], 44100.0, 0.1);
        let result = compute_thd_from_samples(&samples, 44100.0, 1000.0, 5);

        // Second harmonic should be ~10%
        if result.harmonics.len() >= 2 {
            let h2_percent = result.harmonics[1].relative_percent;
            assert!(
                (h2_percent - 10.0).abs() < 2.0,
                "2nd harmonic relative {} should be ~10%",
                h2_percent
            );
        }
    }

    #[test]
    fn test_thd_db_conversion() {
        // 10% THD = -20 dB
        let samples = create_signal_with_harmonics(1000.0, &[1.0, 0.1], 44100.0, 0.1);
        let result = compute_thd_from_samples(&samples, 44100.0, 1000.0, 5);

        // THD in dB should be approximately 20*log10(THD/100)
        let expected_db = 20.0 * (result.thd_percent / 100.0).log10();
        assert!(
            (result.thd_db - expected_db).abs() < 1.0,
            "THD dB {} should match {}",
            result.thd_db,
            expected_db
        );
    }
}
