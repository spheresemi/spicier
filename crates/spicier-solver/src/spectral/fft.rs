//! FFT computation and spectral analysis.

use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

use crate::transient::TransientResult;

use super::window::WindowFunction;

/// Configuration for spectral analysis.
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// Window function to apply before FFT.
    pub window: WindowFunction,
    /// FFT size. If None, uses next power of 2 >= signal length.
    pub fft_size: Option<usize>,
    /// Whether to zero-pad to the FFT size.
    pub zero_padding: bool,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            window: WindowFunction::Hanning,
            fft_size: None,
            zero_padding: true,
        }
    }
}

/// Result of spectral analysis.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Frequency bins in Hz.
    pub frequencies: Vec<f64>,
    /// Magnitude of each frequency bin: |X(f)|.
    pub magnitude: Vec<f64>,
    /// Magnitude in dB: 20*log10(|X(f)|).
    pub magnitude_db: Vec<f64>,
    /// Phase in radians: arg(X(f)).
    pub phase: Vec<f64>,
    /// Power spectral density (magnitude squared).
    pub power_spectral_density: Vec<f64>,
    /// Sample rate used for the analysis (Hz).
    pub sample_rate: f64,
    /// FFT size used.
    pub fft_size: usize,
}

impl SpectralResult {
    /// Find the frequency with the maximum magnitude.
    pub fn peak_frequency(&self) -> Option<f64> {
        self.magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| self.frequencies[i])
    }

    /// Find the magnitude at a specific frequency (nearest bin).
    pub fn magnitude_at(&self, freq: f64) -> Option<f64> {
        let bin = self.frequency_to_bin(freq)?;
        self.magnitude.get(bin).copied()
    }

    /// Find the magnitude in dB at a specific frequency (nearest bin).
    pub fn magnitude_db_at(&self, freq: f64) -> Option<f64> {
        let bin = self.frequency_to_bin(freq)?;
        self.magnitude_db.get(bin).copied()
    }

    /// Convert a frequency to the nearest bin index.
    fn frequency_to_bin(&self, freq: f64) -> Option<usize> {
        if self.frequencies.is_empty() || freq < 0.0 {
            return None;
        }
        let bin_width = self.sample_rate / self.fft_size as f64;
        let bin = (freq / bin_width).round() as usize;
        if bin < self.frequencies.len() {
            Some(bin)
        } else {
            None
        }
    }
}

/// Resample a waveform to uniform time steps using linear interpolation.
///
/// This is necessary for FFT analysis since the FFT assumes uniform sampling.
///
/// # Arguments
/// * `waveform` - The transient result to resample
/// * `node_idx` - Index of the node to extract
/// * `sample_rate` - Desired sample rate in Hz (None = auto from waveform)
///
/// # Returns
/// Tuple of (samples, actual_sample_rate)
pub fn resample_uniform(
    waveform: &TransientResult,
    node_idx: usize,
    sample_rate: Option<f64>,
) -> (Vec<f64>, f64) {
    if waveform.points.is_empty() {
        return (Vec::new(), 0.0);
    }

    let t_start = waveform.points.first().unwrap().time;
    let t_stop = waveform.points.last().unwrap().time;
    let duration = t_stop - t_start;

    if duration <= 0.0 {
        return (Vec::new(), 0.0);
    }

    // Determine sample rate
    let sr = sample_rate.unwrap_or_else(|| {
        // Use average sample rate from original data, at least 2x Nyquist of any content
        let n = waveform.points.len();
        if n > 1 {
            (n - 1) as f64 / duration
        } else {
            1.0
        }
    });

    let num_samples = (duration * sr).ceil() as usize + 1;
    let dt = if num_samples > 1 {
        duration / (num_samples - 1) as f64
    } else {
        duration
    };

    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = t_start + i as f64 * dt;
        if let Some(v) = waveform.voltage_at(node_idx, t) {
            samples.push(v);
        } else if let Some(last) = samples.last() {
            samples.push(*last);
        } else {
            samples.push(0.0);
        }
    }

    (samples, sr)
}

/// Next power of 2 greater than or equal to n.
fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

/// Compute FFT of a transient waveform.
///
/// # Arguments
/// * `waveform` - Transient simulation result
/// * `node_idx` - Index of the node to analyze (0-based)
/// * `config` - Spectral analysis configuration
///
/// # Returns
/// Spectral result with frequencies, magnitudes, phases, and PSD.
#[allow(clippy::needless_range_loop)]
pub fn compute_fft(
    waveform: &TransientResult,
    node_idx: usize,
    config: &SpectralConfig,
) -> SpectralResult {
    // Resample to uniform time steps
    let (samples, sample_rate) = resample_uniform(waveform, node_idx, None);

    if samples.is_empty() {
        return SpectralResult {
            frequencies: Vec::new(),
            magnitude: Vec::new(),
            magnitude_db: Vec::new(),
            phase: Vec::new(),
            power_spectral_density: Vec::new(),
            sample_rate: 0.0,
            fft_size: 0,
        };
    }

    // Determine FFT size
    let fft_size = config
        .fft_size
        .unwrap_or_else(|| next_power_of_2(samples.len()));

    // Apply window function
    let windowed = config.window.apply(&samples);
    let coherent_gain = config.window.coherent_gain(windowed.len());

    // Prepare FFT input (zero-pad if needed)
    let mut fft_input: Vec<Complex<f64>> = windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();

    if config.zero_padding && fft_input.len() < fft_size {
        fft_input.resize(fft_size, Complex::new(0.0, 0.0));
    }

    let actual_fft_size = fft_input.len();

    // Create FFT planner and perform FFT
    let mut planner = FftPlanner::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(actual_fft_size);
    fft.process(&mut fft_input);

    // Extract positive frequencies only (DC to Nyquist)
    let n_positive = actual_fft_size / 2 + 1;
    let freq_resolution = sample_rate / actual_fft_size as f64;

    let mut frequencies = Vec::with_capacity(n_positive);
    let mut magnitude = Vec::with_capacity(n_positive);
    let mut magnitude_db = Vec::with_capacity(n_positive);
    let mut phase = Vec::with_capacity(n_positive);
    let mut power_spectral_density = Vec::with_capacity(n_positive);

    // Normalization factor: use original window length (not zero-padded size)
    // This gives correct amplitude for sinusoids regardless of zero-padding
    let norm_factor = windowed.len() as f64 * coherent_gain;

    for i in 0..n_positive {
        frequencies.push(i as f64 * freq_resolution);

        // Normalize by original window length and correct for window coherent gain
        let normalized = fft_input[i] / norm_factor;
        let unscaled_mag = normalized.norm();

        // Scale by 2 for single-sided spectrum magnitude (except DC and Nyquist)
        // This gives the correct amplitude reading for sinusoids
        let scale = if i == 0 || (actual_fft_size % 2 == 0 && i == n_positive - 1) {
            1.0
        } else {
            2.0
        };

        let mag = unscaled_mag * scale;
        magnitude.push(mag);

        // Convert to dB (with floor to avoid -inf)
        let db = if mag > 1e-20 {
            20.0 * mag.log10()
        } else {
            -400.0 // Floor at -400 dB
        };
        magnitude_db.push(db);

        phase.push(normalized.arg());

        // PSD uses unscaled magnitude squared times scale factor
        // This preserves Parseval's theorem: sum(|x|^2) = sum(PSD)
        power_spectral_density.push(unscaled_mag * unscaled_mag * scale);
    }

    SpectralResult {
        frequencies,
        magnitude,
        magnitude_db,
        phase,
        power_spectral_density,
        sample_rate,
        fft_size: actual_fft_size,
    }
}

/// Compute FFT from raw samples (for direct use without TransientResult).
///
/// # Arguments
/// * `samples` - Uniformly sampled signal
/// * `sample_rate` - Sample rate in Hz
/// * `config` - Spectral analysis configuration
#[allow(clippy::needless_range_loop)]
pub fn compute_fft_from_samples(
    samples: &[f64],
    sample_rate: f64,
    config: &SpectralConfig,
) -> SpectralResult {
    if samples.is_empty() || sample_rate <= 0.0 {
        return SpectralResult {
            frequencies: Vec::new(),
            magnitude: Vec::new(),
            magnitude_db: Vec::new(),
            phase: Vec::new(),
            power_spectral_density: Vec::new(),
            sample_rate: 0.0,
            fft_size: 0,
        };
    }

    let fft_size = config
        .fft_size
        .unwrap_or_else(|| next_power_of_2(samples.len()));

    // Apply window
    let windowed = config.window.apply(samples);
    let coherent_gain = config.window.coherent_gain(windowed.len());

    // Prepare FFT input
    let mut fft_input: Vec<Complex<f64>> = windowed.iter().map(|&x| Complex::new(x, 0.0)).collect();

    if config.zero_padding && fft_input.len() < fft_size {
        fft_input.resize(fft_size, Complex::new(0.0, 0.0));
    }

    let actual_fft_size = fft_input.len();

    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(actual_fft_size);
    fft.process(&mut fft_input);

    // Extract results
    let n_positive = actual_fft_size / 2 + 1;
    let freq_resolution = sample_rate / actual_fft_size as f64;

    // Normalization factor: use original window length (not zero-padded size)
    let norm_factor = windowed.len() as f64 * coherent_gain;

    let mut frequencies = Vec::with_capacity(n_positive);
    let mut magnitude = Vec::with_capacity(n_positive);
    let mut magnitude_db = Vec::with_capacity(n_positive);
    let mut phase = Vec::with_capacity(n_positive);
    let mut power_spectral_density = Vec::with_capacity(n_positive);

    for i in 0..n_positive {
        frequencies.push(i as f64 * freq_resolution);

        let normalized = fft_input[i] / norm_factor;
        let unscaled_mag = normalized.norm();

        let scale = if i == 0 || (actual_fft_size % 2 == 0 && i == n_positive - 1) {
            1.0
        } else {
            2.0
        };

        let mag = unscaled_mag * scale;
        magnitude.push(mag);

        let db = if mag > 1e-20 {
            20.0 * mag.log10()
        } else {
            -400.0
        };
        magnitude_db.push(db);

        phase.push(normalized.arg());
        // PSD uses unscaled magnitude squared times scale factor
        power_spectral_density.push(unscaled_mag * unscaled_mag * scale);
    }

    SpectralResult {
        frequencies,
        magnitude,
        magnitude_db,
        phase,
        power_spectral_density,
        sample_rate,
        fft_size: actual_fft_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn create_sinusoid(freq: f64, amplitude: f64, sample_rate: f64, duration: f64) -> Vec<f64> {
        let n = (sample_rate * duration) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                amplitude * (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_fft_sinusoid_peak() {
        // Create a 1kHz sinusoid sampled at 10kHz for 0.1s (1000 samples = exact integer periods)
        let samples = create_sinusoid(1000.0, 1.0, 10000.0, 0.1);
        let config = SpectralConfig {
            window: WindowFunction::Rectangular,
            zero_padding: false,
            fft_size: None,
        };

        let result = compute_fft_from_samples(&samples, 10000.0, &config);

        // Find peak frequency
        let peak = result.peak_frequency().unwrap();
        assert!(
            (peak - 1000.0).abs() < 20.0,
            "Peak frequency {} Hz should be near 1000 Hz",
            peak
        );

        // Peak magnitude should be near 1.0 (amplitude)
        let peak_mag = result.magnitude_at(1000.0).unwrap();
        assert!(
            (peak_mag - 1.0).abs() < 0.1,
            "Peak magnitude {} should be near 1.0",
            peak_mag
        );
    }

    #[test]
    fn test_parseval_theorem() {
        // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
        // For real signals with single-sided spectrum, we need to account for the 2x factor
        let samples = create_sinusoid(500.0, 2.0, 8000.0, 0.128);
        let config = SpectralConfig {
            window: WindowFunction::Rectangular,
            zero_padding: false,
            fft_size: None,
        };

        let result = compute_fft_from_samples(&samples, 8000.0, &config);

        // Time domain energy
        let time_energy: f64 = samples.iter().map(|x| x * x).sum();
        let time_energy_avg = time_energy / samples.len() as f64;

        // Frequency domain energy (Parseval)
        let freq_energy: f64 = result.power_spectral_density.iter().sum();

        // These should be approximately equal (within numerical precision)
        let ratio = freq_energy / time_energy_avg;
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "Parseval ratio {} should be near 1.0",
            ratio
        );
    }

    #[test]
    fn test_dc_component() {
        // Constant signal should have only DC component
        let samples = vec![5.0; 1024];
        let config = SpectralConfig {
            window: WindowFunction::Rectangular,
            zero_padding: false,
            fft_size: None,
        };

        let result = compute_fft_from_samples(&samples, 1000.0, &config);

        // DC magnitude should be 5.0
        assert!(
            (result.magnitude[0] - 5.0).abs() < 0.01,
            "DC magnitude {} should be 5.0",
            result.magnitude[0]
        );

        // All other components should be near zero
        for i in 1..result.magnitude.len() {
            assert!(
                result.magnitude[i] < 0.01,
                "Non-DC magnitude[{}] = {} should be near 0",
                i,
                result.magnitude[i]
            );
        }
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(1000), 1024);
        assert_eq!(next_power_of_2(1024), 1024);
        assert_eq!(next_power_of_2(1025), 2048);
    }

    #[test]
    fn test_empty_signal() {
        let samples: Vec<f64> = Vec::new();
        let config = SpectralConfig::default();
        let result = compute_fft_from_samples(&samples, 1000.0, &config);

        assert!(result.frequencies.is_empty());
        assert!(result.magnitude.is_empty());
    }
}
