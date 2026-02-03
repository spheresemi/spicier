//! Batch spectral analysis for Monte Carlo sweeps.
//!
//! This module provides parallel FFT/THD computation for analyzing
//! multiple transient waveforms from sweep simulations.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use spicier_solver::spectral::{SpectralConfig, ThdResult, compute_thd};
use spicier_solver::transient::TransientResult;

/// Statistics from batch THD analysis.
#[derive(Debug, Clone)]
pub struct ThdStatistics {
    /// Mean THD percentage.
    pub mean_thd: f64,
    /// Standard deviation of THD.
    pub std_thd: f64,
    /// Minimum THD percentage.
    pub min_thd: f64,
    /// Maximum THD percentage.
    pub max_thd: f64,
    /// Median THD percentage.
    pub median_thd: f64,
    /// Percentage of samples meeting the THD spec (yield).
    pub yield_percent: f64,
    /// Number of samples analyzed.
    pub num_samples: usize,
    /// THD spec used for yield calculation.
    pub thd_spec: f64,
}

/// Compute batch THD for multiple transient waveforms.
///
/// This function is optimized for analyzing Monte Carlo sweep results
/// where we need THD for many waveforms.
///
/// # Arguments
/// * `waveforms` - Vector of transient simulation results
/// * `node_idx` - Index of the node to analyze (0-based)
/// * `fundamental_freq` - Expected fundamental frequency in Hz
/// * `num_harmonics` - Number of harmonics to analyze (excluding fundamental)
///
/// # Returns
/// Vector of THD results for each waveform.
#[cfg(not(feature = "parallel"))]
pub fn compute_batch_thd(
    waveforms: &[TransientResult],
    node_idx: usize,
    fundamental_freq: f64,
    num_harmonics: usize,
) -> Vec<ThdResult> {
    waveforms
        .iter()
        .map(|w| compute_thd(w, node_idx, fundamental_freq, num_harmonics))
        .collect()
}

/// Compute batch THD for multiple transient waveforms (parallel version).
#[cfg(feature = "parallel")]
pub fn compute_batch_thd(
    waveforms: &[TransientResult],
    node_idx: usize,
    fundamental_freq: f64,
    num_harmonics: usize,
) -> Vec<ThdResult> {
    waveforms
        .par_iter()
        .map(|w| compute_thd(w, node_idx, fundamental_freq, num_harmonics))
        .collect()
}

/// Compute batch THD with statistics.
///
/// Analyzes all waveforms and computes statistical summary.
///
/// # Arguments
/// * `waveforms` - Vector of transient simulation results
/// * `node_idx` - Index of the node to analyze
/// * `fundamental_freq` - Expected fundamental frequency in Hz
/// * `num_harmonics` - Number of harmonics to analyze
/// * `thd_spec` - Maximum acceptable THD percentage for yield calculation
///
/// # Returns
/// Tuple of (individual results, statistics)
pub fn compute_batch_thd_with_stats(
    waveforms: &[TransientResult],
    node_idx: usize,
    fundamental_freq: f64,
    num_harmonics: usize,
    thd_spec: f64,
) -> (Vec<ThdResult>, ThdStatistics) {
    let results = compute_batch_thd(waveforms, node_idx, fundamental_freq, num_harmonics);
    let stats = compute_thd_statistics(&results, thd_spec);
    (results, stats)
}

/// Compute statistics from THD results.
pub fn compute_thd_statistics(results: &[ThdResult], thd_spec: f64) -> ThdStatistics {
    if results.is_empty() {
        return ThdStatistics {
            mean_thd: 0.0,
            std_thd: 0.0,
            min_thd: 0.0,
            max_thd: 0.0,
            median_thd: 0.0,
            yield_percent: 100.0,
            num_samples: 0,
            thd_spec,
        };
    }

    let thd_values: Vec<f64> = results.iter().map(|r| r.thd_percent).collect();
    let n = thd_values.len() as f64;

    // Mean
    let mean_thd = thd_values.iter().sum::<f64>() / n;

    // Standard deviation
    let variance = thd_values
        .iter()
        .map(|x| (x - mean_thd).powi(2))
        .sum::<f64>()
        / n;
    let std_thd = variance.sqrt();

    // Min/max
    let min_thd = thd_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_thd = thd_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Median
    let mut sorted = thd_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_thd = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Yield (percentage meeting spec)
    let passing = thd_values.iter().filter(|&&t| t <= thd_spec).count();
    let yield_percent = (passing as f64 / n) * 100.0;

    ThdStatistics {
        mean_thd,
        std_thd,
        min_thd,
        max_thd,
        median_thd,
        yield_percent,
        num_samples: results.len(),
        thd_spec,
    }
}

/// Batch FFT analysis result for a single node across multiple waveforms.
#[derive(Debug, Clone)]
pub struct BatchSpectralResult {
    /// Frequency bins (same for all waveforms).
    pub frequencies: Vec<f64>,
    /// Mean magnitude spectrum across all waveforms.
    pub mean_magnitude: Vec<f64>,
    /// Standard deviation of magnitude at each frequency.
    pub std_magnitude: Vec<f64>,
    /// Minimum magnitude at each frequency.
    pub min_magnitude: Vec<f64>,
    /// Maximum magnitude at each frequency.
    pub max_magnitude: Vec<f64>,
    /// Number of waveforms analyzed.
    pub num_samples: usize,
}

/// Compute batch FFT statistics for multiple waveforms.
///
/// This function computes FFT for each waveform and then calculates
/// statistical summaries across the ensemble.
///
/// # Arguments
/// * `waveforms` - Vector of transient simulation results
/// * `node_idx` - Index of the node to analyze
/// * `config` - Spectral analysis configuration
///
/// # Returns
/// Batch spectral result with statistical summaries.
#[cfg(not(feature = "parallel"))]
pub fn compute_batch_fft_stats(
    waveforms: &[TransientResult],
    node_idx: usize,
    config: &SpectralConfig,
) -> BatchSpectralResult {
    use spicier_solver::spectral::compute_fft;

    if waveforms.is_empty() {
        return BatchSpectralResult {
            frequencies: Vec::new(),
            mean_magnitude: Vec::new(),
            std_magnitude: Vec::new(),
            min_magnitude: Vec::new(),
            max_magnitude: Vec::new(),
            num_samples: 0,
        };
    }

    // Compute all spectra
    let spectra: Vec<_> = waveforms
        .iter()
        .map(|w| compute_fft(w, node_idx, config))
        .collect();

    compute_batch_stats_from_spectra(&spectra)
}

/// Compute batch FFT statistics (parallel version).
#[cfg(feature = "parallel")]
pub fn compute_batch_fft_stats(
    waveforms: &[TransientResult],
    node_idx: usize,
    config: &SpectralConfig,
) -> BatchSpectralResult {
    use spicier_solver::spectral::compute_fft;

    if waveforms.is_empty() {
        return BatchSpectralResult {
            frequencies: Vec::new(),
            mean_magnitude: Vec::new(),
            std_magnitude: Vec::new(),
            min_magnitude: Vec::new(),
            max_magnitude: Vec::new(),
            num_samples: 0,
        };
    }

    // Compute all spectra in parallel
    let spectra: Vec<_> = waveforms
        .par_iter()
        .map(|w| compute_fft(w, node_idx, config))
        .collect();

    compute_batch_stats_from_spectra(&spectra)
}

/// Compute statistics from a collection of spectra.
fn compute_batch_stats_from_spectra(
    spectra: &[spicier_solver::spectral::SpectralResult],
) -> BatchSpectralResult {
    if spectra.is_empty() {
        return BatchSpectralResult {
            frequencies: Vec::new(),
            mean_magnitude: Vec::new(),
            std_magnitude: Vec::new(),
            min_magnitude: Vec::new(),
            max_magnitude: Vec::new(),
            num_samples: 0,
        };
    }

    let num_bins = spectra[0].frequencies.len();
    let n = spectra.len() as f64;

    let frequencies = spectra[0].frequencies.clone();

    let mut mean_magnitude = vec![0.0; num_bins];
    let mut min_magnitude = vec![f64::INFINITY; num_bins];
    let mut max_magnitude = vec![f64::NEG_INFINITY; num_bins];

    // First pass: compute sum, min, max
    for spectrum in spectra {
        for (i, &mag) in spectrum.magnitude.iter().enumerate() {
            if i < num_bins {
                mean_magnitude[i] += mag;
                min_magnitude[i] = min_magnitude[i].min(mag);
                max_magnitude[i] = max_magnitude[i].max(mag);
            }
        }
    }

    // Convert sum to mean
    for m in &mut mean_magnitude {
        *m /= n;
    }

    // Second pass: compute variance
    let mut std_magnitude = vec![0.0; num_bins];
    for spectrum in spectra {
        for (i, &mag) in spectrum.magnitude.iter().enumerate() {
            if i < num_bins {
                std_magnitude[i] += (mag - mean_magnitude[i]).powi(2);
            }
        }
    }

    // Convert variance to std dev
    for s in &mut std_magnitude {
        *s = (*s / n).sqrt();
    }

    BatchSpectralResult {
        frequencies,
        mean_magnitude,
        std_magnitude,
        min_magnitude,
        max_magnitude,
        num_samples: spectra.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use spicier_solver::transient::TimePoint;
    use std::f64::consts::PI;

    fn create_test_waveform(freq: f64, amplitude: f64, num_periods: f64) -> TransientResult {
        let sample_rate = freq * 100.0; // 100 samples per period
        let duration = num_periods / freq;
        let num_points = (duration * sample_rate) as usize;

        let points: Vec<TimePoint> = (0..num_points)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let v = amplitude * (2.0 * PI * freq * t).sin();
                TimePoint {
                    time: t,
                    solution: DVector::from_vec(vec![v]),
                }
            })
            .collect();

        TransientResult {
            points,
            num_nodes: 1,
        }
    }

    #[test]
    fn test_batch_thd_statistics() {
        // Create waveforms with varying THD
        // Pure sines should have THD ≈ 0
        let waveforms: Vec<TransientResult> = (0..10)
            .map(|i| {
                let amp = 1.0 + 0.01 * i as f64; // Slight variation
                create_test_waveform(1000.0, amp, 10.0)
            })
            .collect();

        let (results, stats) = compute_batch_thd_with_stats(&waveforms, 0, 1000.0, 5, 10.0);

        assert_eq!(results.len(), 10);
        assert_eq!(stats.num_samples, 10);

        // Pure sines should have low THD
        assert!(
            stats.mean_thd < 5.0,
            "Mean THD {} should be low for pure sines",
            stats.mean_thd
        );

        // All should pass a 10% spec
        assert!(
            stats.yield_percent > 90.0,
            "Yield {} should be high for pure sines",
            stats.yield_percent
        );
    }

    #[test]
    fn test_thd_yield_calculation() {
        let results = vec![
            ThdResult {
                thd_percent: 1.0,
                thd_db: -40.0,
                fundamental_freq: 1000.0,
                fundamental_magnitude: 1.0,
                harmonics: vec![],
                harmonic_rms: 0.01,
                num_harmonics: 5,
            },
            ThdResult {
                thd_percent: 5.0,
                thd_db: -26.0,
                fundamental_freq: 1000.0,
                fundamental_magnitude: 1.0,
                harmonics: vec![],
                harmonic_rms: 0.05,
                num_harmonics: 5,
            },
            ThdResult {
                thd_percent: 15.0,
                thd_db: -16.5,
                fundamental_freq: 1000.0,
                fundamental_magnitude: 1.0,
                harmonics: vec![],
                harmonic_rms: 0.15,
                num_harmonics: 5,
            },
        ];

        // With 10% spec, 2 of 3 should pass
        let stats = compute_thd_statistics(&results, 10.0);
        assert!((stats.yield_percent - 66.67).abs() < 1.0);

        // With 1% spec, only 1 should pass
        let stats_strict = compute_thd_statistics(&results, 1.0);
        assert!((stats_strict.yield_percent - 33.33).abs() < 1.0);
    }

    #[test]
    fn test_batch_fft_stats() {
        // Create similar waveforms
        let waveforms: Vec<TransientResult> = (0..5)
            .map(|_| create_test_waveform(1000.0, 1.0, 10.0))
            .collect();

        let config = SpectralConfig::default();
        let result = compute_batch_fft_stats(&waveforms, 0, &config);

        assert_eq!(result.num_samples, 5);
        assert!(!result.frequencies.is_empty());

        // For identical waveforms, std should be near zero
        let max_std = result.std_magnitude.iter().cloned().fold(0.0, f64::max);
        assert!(
            max_std < 0.01,
            "Std deviation {} should be near zero for identical waveforms",
            max_std
        );
    }

    #[test]
    fn test_empty_batch() {
        let waveforms: Vec<TransientResult> = Vec::new();

        let results = compute_batch_thd(&waveforms, 0, 1000.0, 5);
        assert!(results.is_empty());

        let stats = compute_thd_statistics(&[], 10.0);
        assert_eq!(stats.num_samples, 0);
        assert_eq!(stats.yield_percent, 100.0); // No failures = 100% yield
    }
}
