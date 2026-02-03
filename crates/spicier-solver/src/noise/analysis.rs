//! Noise analysis computation.

use num_complex::Complex;
use std::f64::consts::PI;

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
    /// Source resistance in Ohms (for noise figure calculation).
    /// If None, noise figure is not computed.
    pub source_resistance: Option<f64>,
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
            source_resistance: None,
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
    /// Noise figure in dB at each frequency.
    /// NF = 10 * log10(F) where F = Sv_input / Sv_source.
    /// Only computed if source_resistance was specified.
    pub noise_figure_db: Vec<f64>,
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
        let freq_idx = self.frequencies.iter().position(|&f| f >= frequency)?;

        self.contributions
            .iter()
            .max_by(|a, b| {
                a.contribution_percent
                    .get(freq_idx)
                    .unwrap_or(&0.0)
                    .partial_cmp(b.contribution_percent.get(freq_idx).unwrap_or(&0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.source_name.as_str())
    }

    /// Get noise figure in dB at a specific frequency (interpolated).
    ///
    /// Returns None if noise figure was not computed (no source resistance specified).
    pub fn noise_figure_at(&self, frequency: f64) -> Option<f64> {
        if self.noise_figure_db.is_empty() {
            return None;
        }
        interpolate_log(&self.frequencies, &self.noise_figure_db, frequency)
    }

    /// Get the minimum noise figure and the frequency at which it occurs.
    ///
    /// Returns (frequency, NF_dB) or None if noise figure was not computed.
    pub fn min_noise_figure(&self) -> Option<(f64, f64)> {
        if self.noise_figure_db.is_empty() {
            return None;
        }
        self.noise_figure_db
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, &nf)| (self.frequencies[idx], nf))
    }

    /// Get noise figure statistics: (min, max, average) in dB.
    ///
    /// Returns None if noise figure was not computed.
    pub fn noise_figure_stats(&self) -> Option<(f64, f64, f64)> {
        if self.noise_figure_db.is_empty() {
            return None;
        }
        let min = self
            .noise_figure_db
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .noise_figure_db
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg = self.noise_figure_db.iter().sum::<f64>() / self.noise_figure_db.len() as f64;
        Some((min, max, avg))
    }

    /// Export noise spectral density to CSV format.
    ///
    /// The CSV includes frequency, output noise, input noise (if available),
    /// equivalent input noise resistance, noise figure (if available), and
    /// per-source contributions.
    ///
    /// # Returns
    /// CSV string with header row and data
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Build header
        let mut headers = vec![
            "Frequency(Hz)",
            "OutputNoise(V/rtHz)",
            "OutputNoiseSq(V^2/Hz)",
        ];
        if !self.input_noise.is_empty() {
            headers.push("InputNoise(V/rtHz)");
        }
        headers.push("EquivNoiseR(Ohms)");
        if !self.noise_figure_db.is_empty() {
            headers.push("NoiseFigure(dB)");
        }
        for contrib in &self.contributions {
            headers.push(&contrib.source_name);
        }
        csv.push_str(&headers.join(","));
        csv.push('\n');

        // Write data rows
        for i in 0..self.frequencies.len() {
            let mut row = vec![
                format!("{:.6e}", self.frequencies[i]),
                format!("{:.6e}", self.output_noise[i]),
                format!("{:.6e}", self.output_noise_sq[i]),
            ];
            if !self.input_noise.is_empty() {
                row.push(format!("{:.6e}", self.input_noise[i]));
            }
            row.push(format!("{:.6e}", self.equiv_input_noise_resistance[i]));
            if !self.noise_figure_db.is_empty() {
                row.push(format!("{:.4}", self.noise_figure_db[i]));
            }
            for contrib in &self.contributions {
                // Output contribution percentage
                row.push(format!("{:.2}", contrib.contribution_percent[i]));
            }
            csv.push_str(&row.join(","));
            csv.push('\n');
        }

        csv
    }

    /// Export noise spectral density to a file.
    ///
    /// # Arguments
    /// * `path` - File path to write to
    ///
    /// # Returns
    /// Result indicating success or IO error
    pub fn write_csv(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_csv())
    }

    /// Export detailed noise contributions to CSV format.
    ///
    /// This format provides the noise power (V²/Hz) from each source at each frequency,
    /// suitable for detailed noise budget analysis.
    pub fn contributions_to_csv(&self) -> String {
        let mut csv = String::new();

        // Header: Frequency, Source1_Sq, Source1_%, Source2_Sq, Source2_%, ...
        let mut headers = vec!["Frequency(Hz)".to_string()];
        for contrib in &self.contributions {
            headers.push(format!("{}_V2Hz", contrib.source_name));
            headers.push(format!("{}_%", contrib.source_name));
        }
        headers.push("Total_V2Hz".to_string());
        csv.push_str(&headers.join(","));
        csv.push('\n');

        // Data rows
        for i in 0..self.frequencies.len() {
            let mut row = vec![format!("{:.6e}", self.frequencies[i])];
            for contrib in &self.contributions {
                row.push(format!("{:.6e}", contrib.output_noise_sq[i]));
                row.push(format!("{:.2}", contrib.contribution_percent[i]));
            }
            row.push(format!("{:.6e}", self.output_noise_sq[i]));
            csv.push_str(&row.join(","));
            csv.push('\n');
        }

        csv
    }

    /// Export noise contributions to a file.
    pub fn write_contributions_csv(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        std::fs::write(path, self.contributions_to_csv())
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
            let log_step =
                (config.fstop.log10() - config.fstart.log10()) / (total_points - 1) as f64;
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
pub fn compute_noise(stamper: &dyn NoiseStamper, config: &NoiseConfig) -> Result<NoiseResult> {
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
                let v_pos = solution
                    .get(config.output_node)
                    .copied()
                    .unwrap_or(Complex::new(0.0, 0.0));
                let v_neg = solution
                    .get(ref_node)
                    .copied()
                    .unwrap_or(Complex::new(0.0, 0.0));
                v_pos - v_neg
            } else {
                solution
                    .get(config.output_node)
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
            if let Ok(gain) =
                stamper.input_gain(omega, input_idx, config.output_node, config.output_ref_node)
            {
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
        input_noise
            .iter()
            .map(|&vn| {
                let sv = vn * vn;
                sv / boltzmann_4kt
            })
            .collect()
    } else {
        output_noise
            .iter()
            .map(|&vn| {
                let sv = vn * vn;
                sv / boltzmann_4kt
            })
            .collect()
    };

    // Compute noise figure if source resistance is specified
    // NF = 10 * log10(F) where F = Sv_input / Sv_source = Rn / Rs
    let noise_figure_db = if let Some(rs) = config.source_resistance {
        if rs > 0.0 && !equiv_input_noise_resistance.is_empty() {
            equiv_input_noise_resistance
                .iter()
                .map(|&rn| {
                    let f = rn / rs;
                    if f > 0.0 { 10.0 * f.log10() } else { 0.0 }
                })
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Ok(NoiseResult {
        frequencies,
        output_noise,
        output_noise_sq,
        input_noise,
        contributions,
        equiv_input_noise_resistance,
        noise_figure_db,
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
        for (i, freq) in freqs.iter().enumerate() {
            let expected = 100.0 + i as f64 * step;
            assert!((freq - expected).abs() < 1e-10);
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

    #[test]
    fn test_noise_figure_calculation() {
        // Create a mock NoiseResult to test noise figure methods
        let result = NoiseResult {
            frequencies: vec![100.0, 1000.0, 10000.0],
            output_noise: vec![1e-8, 1e-8, 1e-8],
            output_noise_sq: vec![1e-16, 1e-16, 1e-16],
            input_noise: vec![1e-9, 1e-9, 1e-9],
            contributions: vec![],
            // Rn = Sv / 4kT = 1e-18 / (4 * 1.38e-23 * 300) = 60.4 ohms
            equiv_input_noise_resistance: vec![60.4, 60.4, 60.4],
            // NF = 10 * log10(Rn / Rs) with Rs = 50 ohms
            // NF = 10 * log10(60.4 / 50) = 10 * log10(1.208) = 0.82 dB
            noise_figure_db: vec![0.82, 0.82, 0.82],
            temperature: 300.0,
        };

        // Test noise_figure_at
        let nf = result.noise_figure_at(1000.0).unwrap();
        assert!(
            (nf - 0.82).abs() < 0.01,
            "NF at 1kHz = {} (expected 0.82)",
            nf
        );

        // Test min_noise_figure
        let (freq, min_nf) = result.min_noise_figure().unwrap();
        assert!((min_nf - 0.82).abs() < 0.01);
        assert!((100.0..=10000.0).contains(&freq));

        // Test noise_figure_stats
        let (min, max, avg) = result.noise_figure_stats().unwrap();
        assert!((min - 0.82).abs() < 0.01);
        assert!((max - 0.82).abs() < 0.01);
        assert!((avg - 0.82).abs() < 0.01);
    }

    #[test]
    fn test_noise_figure_empty_when_no_source_resistance() {
        let result = NoiseResult {
            frequencies: vec![100.0, 1000.0],
            output_noise: vec![1e-8, 1e-8],
            output_noise_sq: vec![1e-16, 1e-16],
            input_noise: vec![],
            contributions: vec![],
            equiv_input_noise_resistance: vec![50.0, 50.0],
            noise_figure_db: vec![], // Empty when source_resistance not specified
            temperature: 300.0,
        };

        assert!(result.noise_figure_at(1000.0).is_none());
        assert!(result.min_noise_figure().is_none());
        assert!(result.noise_figure_stats().is_none());
    }

    #[test]
    fn test_noise_csv_export() {
        let result = NoiseResult {
            frequencies: vec![100.0, 1000.0, 10000.0],
            output_noise: vec![1e-8, 2e-8, 3e-8],
            output_noise_sq: vec![1e-16, 4e-16, 9e-16],
            input_noise: vec![1e-9, 2e-9, 3e-9],
            contributions: vec![
                NoiseContribution {
                    source_name: "R1".to_string(),
                    output_noise_sq: vec![5e-17, 2e-16, 4.5e-16],
                    contribution_percent: vec![50.0, 50.0, 50.0],
                },
                NoiseContribution {
                    source_name: "R2".to_string(),
                    output_noise_sq: vec![5e-17, 2e-16, 4.5e-16],
                    contribution_percent: vec![50.0, 50.0, 50.0],
                },
            ],
            equiv_input_noise_resistance: vec![60.0, 60.0, 60.0],
            noise_figure_db: vec![0.8, 0.8, 0.8],
            temperature: 300.0,
        };

        let csv = result.to_csv();

        // Check header
        assert!(csv.contains("Frequency(Hz)"));
        assert!(csv.contains("OutputNoise(V/rtHz)"));
        assert!(csv.contains("InputNoise(V/rtHz)"));
        assert!(csv.contains("NoiseFigure(dB)"));
        assert!(csv.contains("R1"));
        assert!(csv.contains("R2"));

        // Check data rows exist
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows

        // Check first data row contains frequency
        assert!(lines[1].contains("1.000000e2"));
    }

    #[test]
    fn test_noise_contributions_csv_export() {
        let result = NoiseResult {
            frequencies: vec![100.0, 1000.0],
            output_noise: vec![1e-8, 1e-8],
            output_noise_sq: vec![1e-16, 1e-16],
            input_noise: vec![],
            contributions: vec![
                NoiseContribution {
                    source_name: "R1".to_string(),
                    output_noise_sq: vec![6e-17, 6e-17],
                    contribution_percent: vec![60.0, 60.0],
                },
                NoiseContribution {
                    source_name: "D1_shot".to_string(),
                    output_noise_sq: vec![4e-17, 4e-17],
                    contribution_percent: vec![40.0, 40.0],
                },
            ],
            equiv_input_noise_resistance: vec![50.0, 50.0],
            noise_figure_db: vec![],
            temperature: 300.0,
        };

        let csv = result.contributions_to_csv();

        // Check header
        assert!(csv.contains("Frequency(Hz)"));
        assert!(csv.contains("R1_V2Hz"));
        assert!(csv.contains("R1_%"));
        assert!(csv.contains("D1_shot_V2Hz"));
        assert!(csv.contains("Total_V2Hz"));

        // Check data
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 data rows
    }
}
