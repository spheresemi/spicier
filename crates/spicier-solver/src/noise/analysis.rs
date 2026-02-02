//! Noise analysis computation.

use std::f64::consts::PI;
use num_complex::Complex;

use crate::ac::ComplexMna;
use crate::error::Result;
use crate::linear::{SPARSE_THRESHOLD, solve_complex, solve_sparse_complex};

use super::sources::NoiseSource;

/// Frequency sweep type for noise analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSweepType {
    /// Linear frequency spacing.
    Linear,
    /// Logarithmic spacing per decade.
    Decade,
    /// Logarithmic spacing per octave.
    Octave,
}

/// Configuration for noise analysis.
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Output node index (0-based, excluding ground).
    pub output_node: usize,
    /// Optional reference node for differential output (None for single-ended).
    pub output_ref_node: Option<usize>,
    /// Input source index (for computing input-referred noise). None if not needed.
    pub input_source_idx: Option<usize>,
    /// Start frequency in Hz.
    pub fstart: f64,
    /// Stop frequency in Hz.
    pub fstop: f64,
    /// Number of points (total for linear, per decade/octave for log).
    pub num_points: usize,
    /// Frequency sweep type.
    pub sweep_type: NoiseSweepType,
    /// Temperature in Kelvin (default 300K = 27°C).
    pub temperature: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            output_node: 0,
            output_ref_node: None,
            input_source_idx: None,
            fstart: 1.0,
            fstop: 1e6,
            num_points: 10,
            sweep_type: NoiseSweepType::Decade,
            temperature: 300.0,
        }
    }
}

/// Noise contribution from a single source.
#[derive(Debug, Clone)]
pub struct NoiseContribution {
    /// Name of the noise source.
    pub source_name: String,
    /// Output noise voltage squared (V²/Hz) at each frequency.
    pub output_noise_sq: Vec<f64>,
    /// Percentage contribution to total noise at each frequency.
    pub contribution_percent: Vec<f64>,
}

/// Result of noise analysis.
#[derive(Debug, Clone)]
pub struct NoiseResult {
    /// Frequency points in Hz.
    pub frequencies: Vec<f64>,
    /// Total output noise voltage density (V/√Hz) at each frequency.
    pub output_noise: Vec<f64>,
    /// Total output noise voltage squared (V²/Hz) at each frequency.
    pub output_noise_sq: Vec<f64>,
    /// Input-referred noise voltage density (V/√Hz) at each frequency.
    /// Only computed if input_source_idx was specified.
    pub input_noise: Vec<f64>,
    /// Per-source noise contributions.
    pub contributions: Vec<NoiseContribution>,
    /// Equivalent input noise resistance (Ohms) at each frequency.
    /// Rn = Sv_input / (4kT)
    pub equiv_input_noise_resistance: Vec<f64>,
    /// Temperature used for analysis (Kelvin).
    pub temperature: f64,
}

impl NoiseResult {
    /// Get total output noise at a specific frequency (interpolated).
    pub fn output_noise_at(&self, frequency: f64) -> Option<f64> {
        interpolate_log(&self.frequencies, &self.output_noise, frequency)
    }

    /// Get input-referred noise at a specific frequency (interpolated).
    pub fn input_noise_at(&self, frequency: f64) -> Option<f64> {
        if self.input_noise.is_empty() {
            return None;
        }
        interpolate_log(&self.frequencies, &self.input_noise, frequency)
    }

    /// Compute integrated noise over a frequency range.
    ///
    /// # Arguments
    /// * `fstart` - Start frequency in Hz
    /// * `fstop` - Stop frequency in Hz
    ///
    /// # Returns
    /// Integrated RMS noise voltage in Volts
    pub fn integrated_noise(&self, fstart: f64, fstop: f64) -> f64 {
        if self.frequencies.len() < 2 {
            return 0.0;
        }

        let mut integral = 0.0;
        for i in 0..self.frequencies.len() - 1 {
            let f1 = self.frequencies[i];
            let f2 = self.frequencies[i + 1];

            // Skip if outside range
            if f2 < fstart || f1 > fstop {
                continue;
            }

            // Clip to range
            let f1_clipped = f1.max(fstart);
            let f2_clipped = f2.min(fstop);

            // Trapezoidal integration of noise power
            let sv1 = self.output_noise_sq[i];
            let sv2 = self.output_noise_sq[i + 1];

            // For 1/f noise, use log-average; for white noise, use linear
            // Simple trapezoidal rule here
            integral += 0.5 * (sv1 + sv2) * (f2_clipped - f1_clipped);
        }

        integral.sqrt()
    }

    /// Get the dominant noise contributor at a specific frequency.
    pub fn dominant_contributor_at(&self, frequency: f64) -> Option<&str> {
        let freq_idx = self.frequencies
            .iter()
            .position(|&f| f >= frequency)?;

        self.contributions
            .iter()
            .max_by(|a, b| {
                a.contribution_percent.get(freq_idx)
                    .unwrap_or(&0.0)
                    .partial_cmp(b.contribution_percent.get(freq_idx).unwrap_or(&0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.source_name.as_str())
    }
}

/// Trait for circuits that support noise analysis.
pub trait NoiseStamper {
    /// Stamp the AC small-signal circuit at a given frequency.
    ///
    /// # Arguments
    /// * `mna` - Complex MNA system to stamp
    /// * `omega` - Angular frequency (2π × frequency in Hz)
    fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64);

    /// Get the noise sources in the circuit.
    ///
    /// This should be called after DC operating point is computed,
    /// as noise currents depend on DC bias conditions.
    fn noise_sources(&self) -> Vec<NoiseSource>;

    /// Number of nodes (excluding ground).
    fn num_nodes(&self) -> usize;

    /// Number of voltage sources / branch currents.
    fn num_vsources(&self) -> usize;

    /// Get the AC gain from input source to output.
    ///
    /// # Arguments
    /// * `omega` - Angular frequency
    /// * `input_source_idx` - Index of the input voltage source
    /// * `output_node` - Output node index
    /// * `output_ref_node` - Optional reference node for differential output
    ///
    /// # Returns
    /// Complex gain H(jω) = Vout / Vin
    fn input_gain(
        &self,
        omega: f64,
        input_source_idx: usize,
        output_node: usize,
        output_ref_node: Option<usize>,
    ) -> Result<Complex<f64>>;
}

/// Generate frequency points for noise analysis.
fn generate_frequencies(config: &NoiseConfig) -> Vec<f64> {
    let mut frequencies = Vec::new();

    match config.sweep_type {
        NoiseSweepType::Linear => {
            let step = (config.fstop - config.fstart) / (config.num_points.max(1) - 1) as f64;
            for i in 0..config.num_points {
                frequencies.push(config.fstart + i as f64 * step);
            }
        }
        NoiseSweepType::Decade => {
            let decades = (config.fstop / config.fstart).log10();
            let total_points = (decades * config.num_points as f64).ceil() as usize + 1;
            let log_step = (config.fstop.log10() - config.fstart.log10()) / (total_points - 1) as f64;
            for i in 0..total_points {
                let log_f = config.fstart.log10() + i as f64 * log_step;
                frequencies.push(10.0_f64.powf(log_f));
            }
        }
        NoiseSweepType::Octave => {
            let octaves = (config.fstop / config.fstart).log2();
            let total_points = (octaves * config.num_points as f64).ceil() as usize + 1;
            let log_step = (config.fstop.ln() - config.fstart.ln()) / (total_points - 1) as f64;
            for i in 0..total_points {
                let log_f = config.fstart.ln() + i as f64 * log_step;
                frequencies.push(log_f.exp());
            }
        }
    }

    frequencies
}

/// Solve a complex MNA system.
fn solve_complex_mna(mna: &ComplexMna) -> Result<Vec<Complex<f64>>> {
    let size = mna.size();

    if size >= SPARSE_THRESHOLD {
        let result = solve_sparse_complex(size, &mna.triplets, mna.rhs())?;
        Ok(result.iter().copied().collect())
    } else {
        let matrix = mna.to_dense_matrix();
        let result = solve_complex(&matrix, mna.rhs())?;
        Ok(result.iter().copied().collect())
    }
}

/// Compute noise analysis.
///
/// # Arguments
/// * `stamper` - Circuit stamper that provides AC stamps and noise sources
/// * `config` - Noise analysis configuration
///
/// # Returns
/// Noise analysis results with output noise, input-referred noise, and contributions
pub fn compute_noise(
    stamper: &dyn NoiseStamper,
    config: &NoiseConfig,
) -> Result<NoiseResult> {
    let frequencies = generate_frequencies(config);
    let noise_sources = stamper.noise_sources();
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();

    let mut output_noise_sq = vec![0.0; frequencies.len()];
    let mut contributions: Vec<NoiseContribution> = noise_sources
        .iter()
        .map(|s| NoiseContribution {
            source_name: s.name.clone(),
            output_noise_sq: vec![0.0; frequencies.len()],
            contribution_percent: vec![0.0; frequencies.len()],
        })
        .collect();

    // For each frequency point
    for (freq_idx, &freq) in frequencies.iter().enumerate() {
        let omega = 2.0 * PI * freq;

        // Build the nominal AC small-signal system
        let mut mna = ComplexMna::new(num_nodes, num_vsources);
        stamper.stamp_ac(&mut mna, omega);

        // For each noise source, compute its contribution to output noise
        for (source_idx, source) in noise_sources.iter().enumerate() {
            // Get the noise current spectral density
            let si = source.current_spectral_density(freq, config.temperature);

            if si <= 0.0 {
                continue;
            }

            // Compute transfer function from noise source to output
            // Inject unit current at source nodes, solve for output voltage
            let mut mna_with_source = mna.clone();

            // Inject 1A current at the noise source location
            if let Some(node_pos) = source.node_pos {
                if node_pos < num_nodes {
                    mna_with_source.add_rhs(node_pos, Complex::new(1.0, 0.0));
                }
            }
            if let Some(node_neg) = source.node_neg {
                if node_neg < num_nodes {
                    mna_with_source.add_rhs(node_neg, Complex::new(-1.0, 0.0));
                }
            }

            // Solve for node voltages
            let solution = solve_complex_mna(&mna_with_source)?;

            // Extract output voltage (transfer impedance)
            let v_out = if let Some(ref_node) = config.output_ref_node {
                let v_pos = solution.get(config.output_node)
                    .copied()
                    .unwrap_or(Complex::new(0.0, 0.0));
                let v_neg = solution.get(ref_node)
                    .copied()
                    .unwrap_or(Complex::new(0.0, 0.0));
                v_pos - v_neg
            } else {
                solution.get(config.output_node)
                    .copied()
                    .unwrap_or(Complex::new(0.0, 0.0))
            };

            // Transfer impedance magnitude squared
            let zt_sq = v_out.norm_sqr();

            // Output noise voltage squared: Sv_out = |Zt|² × Si
            let sv_out = zt_sq * si;

            contributions[source_idx].output_noise_sq[freq_idx] = sv_out;
            output_noise_sq[freq_idx] += sv_out;
        }

        // Compute contribution percentages
        let total = output_noise_sq[freq_idx];
        if total > 0.0 {
            for contrib in &mut contributions {
                contrib.contribution_percent[freq_idx] =
                    100.0 * contrib.output_noise_sq[freq_idx] / total;
            }
        }
    }

    // Compute output noise voltage density (V/√Hz)
    let output_noise: Vec<f64> = output_noise_sq.iter().map(|&sv| sv.sqrt()).collect();

    // Compute input-referred noise if input source specified
    let input_noise = if let Some(input_idx) = config.input_source_idx {
        let mut input_noise_vec = Vec::with_capacity(frequencies.len());
        for (freq_idx, &freq) in frequencies.iter().enumerate() {
            let omega = 2.0 * PI * freq;

            // Get gain from input to output
            if let Ok(gain) = stamper.input_gain(
                omega,
                input_idx,
                config.output_node,
                config.output_ref_node,
            ) {
                let gain_mag = gain.norm();
                if gain_mag > 1e-20 {
                    // Input-referred noise = output noise / |gain|
                    input_noise_vec.push(output_noise[freq_idx] / gain_mag);
                } else {
                    input_noise_vec.push(0.0);
                }
            } else {
                input_noise_vec.push(0.0);
            }
        }
        input_noise_vec
    } else {
        Vec::new()
    };

    // Compute equivalent input noise resistance
    let boltzmann_4kt = 4.0 * super::sources::BOLTZMANN * config.temperature;
    let equiv_input_noise_resistance: Vec<f64> = if !input_noise.is_empty() {
        input_noise.iter()
            .map(|&vn| {
                let sv = vn * vn;
                sv / boltzmann_4kt
            })
            .collect()
    } else {
        output_noise.iter()
            .map(|&vn| {
                let sv = vn * vn;
                sv / boltzmann_4kt
            })
            .collect()
    };

    Ok(NoiseResult {
        frequencies,
        output_noise,
        output_noise_sq,
        input_noise,
        contributions,
        equiv_input_noise_resistance,
        temperature: config.temperature,
    })
}

/// Linear interpolation with log-frequency handling.
fn interpolate_log(frequencies: &[f64], values: &[f64], target_freq: f64) -> Option<f64> {
    if frequencies.is_empty() || values.is_empty() {
        return None;
    }

    if target_freq <= frequencies[0] {
        return Some(values[0]);
    }
    if target_freq >= *frequencies.last()? {
        return Some(*values.last()?);
    }

    // Find bracketing indices
    for i in 0..frequencies.len() - 1 {
        if frequencies[i] <= target_freq && target_freq <= frequencies[i + 1] {
            // Log interpolation
            let log_f = target_freq.log10();
            let log_f1 = frequencies[i].log10();
            let log_f2 = frequencies[i + 1].log10();
            let t = (log_f - log_f1) / (log_f2 - log_f1);

            // Linear interpolation in log-frequency space
            return Some(values[i] + t * (values[i + 1] - values[i]));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_frequencies_decade() {
        let config = NoiseConfig {
            fstart: 1.0,
            fstop: 1000.0,
            num_points: 10,
            sweep_type: NoiseSweepType::Decade,
            ..Default::default()
        };

        let freqs = generate_frequencies(&config);

        // Should span 3 decades with 10 points per decade
        assert!(freqs.len() > 20);
        assert!((freqs[0] - 1.0).abs() < 1e-10);
        assert!((freqs.last().unwrap() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_generate_frequencies_linear() {
        let config = NoiseConfig {
            fstart: 100.0,
            fstop: 1000.0,
            num_points: 10,
            sweep_type: NoiseSweepType::Linear,
            ..Default::default()
        };

        let freqs = generate_frequencies(&config);

        assert_eq!(freqs.len(), 10);
        assert!((freqs[0] - 100.0).abs() < 1e-10);
        assert!((freqs[9] - 1000.0).abs() < 1e-10);

        // Check uniform spacing
        let step = (1000.0 - 100.0) / 9.0;
        for i in 0..10 {
            let expected = 100.0 + i as f64 * step;
            assert!((freqs[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_log() {
        let frequencies = vec![1.0, 10.0, 100.0, 1000.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];

        // Exact points
        assert!((interpolate_log(&frequencies, &values, 1.0).unwrap() - 1.0).abs() < 1e-10);
        assert!((interpolate_log(&frequencies, &values, 100.0).unwrap() - 3.0).abs() < 1e-10);

        // Interpolated point (geometric mean of 10 and 100 is ~31.6)
        let mid = interpolate_log(&frequencies, &values, 31.6227766).unwrap();
        assert!((mid - 2.5).abs() < 0.1);
    }
}
