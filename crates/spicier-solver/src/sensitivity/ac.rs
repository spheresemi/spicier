//! AC (small-signal) sensitivity analysis using forward finite differences.
//!
//! Computes how AC responses (magnitude, phase) change with parameter variations.

use num_complex::Complex;
use std::f64::consts::PI;

use crate::ac::{AcParams, ComplexMna};
use crate::error::Result;
use crate::linear::{SPARSE_THRESHOLD, solve_complex, solve_sparse_complex};

use super::config::{SensitivityConfig, SensitivityOutput, SensitivityParam};

/// Trait for circuits that support AC sensitivity analysis.
pub trait AcSensitivityStamper {
    /// Stamp the circuit with nominal parameter values at a given frequency.
    ///
    /// # Arguments
    /// * `mna` - Complex MNA system to stamp
    /// * `omega` - Angular frequency (2π × frequency in Hz)
    fn stamp_ac_nominal(&self, mna: &mut ComplexMna, omega: f64);

    /// Stamp the circuit with a perturbed parameter value at a given frequency.
    ///
    /// # Arguments
    /// * `mna` - Complex MNA system to stamp
    /// * `omega` - Angular frequency
    /// * `param` - The parameter being perturbed (with new value)
    fn stamp_ac_perturbed(&self, mna: &mut ComplexMna, omega: f64, param: &SensitivityParam);

    /// Number of nodes (excluding ground).
    fn num_nodes(&self) -> usize;

    /// Number of voltage sources / branch currents.
    fn num_vsources(&self) -> usize;
}

/// Result of AC sensitivity analysis.
#[derive(Debug, Clone)]
pub struct AcSensitivityResult {
    /// The parameter that was varied.
    pub param: SensitivityParam,
    /// The output that was measured.
    pub output: SensitivityOutput,
    /// Frequency in Hz.
    pub frequency: f64,
    /// Sensitivity of magnitude: d|H|/dParam.
    pub magnitude_sensitivity: f64,
    /// Sensitivity of magnitude in dB: d(20*log10|H|)/dParam.
    pub magnitude_db_sensitivity: f64,
    /// Sensitivity of phase: d(arg(H))/dParam (radians).
    pub phase_sensitivity: f64,
    /// Complex sensitivity: dH/dParam.
    pub complex_sensitivity: Complex<f64>,
    /// Nominal complex output value.
    pub nominal_value: Complex<f64>,
    /// Nominal parameter value.
    pub nominal_param: f64,
}

impl AcSensitivityResult {
    /// Get the parameter name.
    pub fn param_name(&self) -> String {
        self.param.name()
    }

    /// Get the output name.
    pub fn output_name(&self) -> &str {
        self.output.name()
    }
}

/// Extract an output value from a complex solution vector.
fn extract_output_complex(
    solution: &[Complex<f64>],
    output: &SensitivityOutput,
    num_nodes: usize,
) -> Complex<f64> {
    match output {
        SensitivityOutput::Voltage { node_idx, .. } => solution
            .get(*node_idx)
            .copied()
            .unwrap_or(Complex::new(0.0, 0.0)),
        SensitivityOutput::Current { branch_idx, .. } => solution
            .get(num_nodes + *branch_idx)
            .copied()
            .unwrap_or(Complex::new(0.0, 0.0)),
        SensitivityOutput::VoltageDiff {
            node_pos, node_neg, ..
        } => {
            let vp = solution
                .get(*node_pos)
                .copied()
                .unwrap_or(Complex::new(0.0, 0.0));
            let vn = solution
                .get(*node_neg)
                .copied()
                .unwrap_or(Complex::new(0.0, 0.0));
            vp - vn
        }
    }
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

/// Compute AC sensitivity at a single frequency.
///
/// # Arguments
/// * `stamper` - Circuit stamper that can apply parameter perturbations
/// * `frequency` - Frequency in Hz
/// * `config` - Sensitivity configuration
///
/// # Returns
/// Vector of sensitivity results for each parameter-output combination.
pub fn compute_ac_sensitivity(
    stamper: &dyn AcSensitivityStamper,
    frequency: f64,
    config: &SensitivityConfig,
) -> Result<Vec<AcSensitivityResult>> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let omega = 2.0 * PI * frequency;

    // Solve nominal AC operating point
    let mut mna_nominal = ComplexMna::new(num_nodes, num_vsources);
    stamper.stamp_ac_nominal(&mut mna_nominal, omega);
    let nominal_solution = solve_complex_mna(&mna_nominal)?;

    // Extract nominal output values
    let nominal_outputs: Vec<Complex<f64>> = config
        .outputs
        .iter()
        .map(|o| extract_output_complex(&nominal_solution, o, num_nodes))
        .collect();

    let mut results = Vec::with_capacity(config.params.len() * config.outputs.len());

    // For each parameter
    for param in &config.params {
        let nominal_value = param.value();
        let delta = config.compute_delta(nominal_value);
        let perturbed_value = nominal_value + delta;
        let perturbed_param = param.with_value(perturbed_value);

        // Solve perturbed AC operating point
        let mut mna_perturbed = ComplexMna::new(num_nodes, num_vsources);
        stamper.stamp_ac_perturbed(&mut mna_perturbed, omega, &perturbed_param);
        let perturbed_solution = solve_complex_mna(&mna_perturbed)?;

        // Compute sensitivities for each output
        for (i, output) in config.outputs.iter().enumerate() {
            let nominal_output = nominal_outputs[i];
            let perturbed_output = extract_output_complex(&perturbed_solution, output, num_nodes);

            // Complex sensitivity: dH/dParam
            let complex_sensitivity = (perturbed_output - nominal_output) / delta;

            // Magnitude sensitivity
            let nominal_mag = nominal_output.norm();
            let perturbed_mag = perturbed_output.norm();
            let magnitude_sensitivity = (perturbed_mag - nominal_mag) / delta;

            // Magnitude dB sensitivity
            let magnitude_db_sensitivity = if nominal_mag > 1e-20 {
                // d(20*log10(|H|))/dParam = (20/ln(10)) * (1/|H|) * d|H|/dParam
                (20.0 / 10.0_f64.ln()) * magnitude_sensitivity / nominal_mag
            } else {
                0.0
            };

            // Phase sensitivity
            let nominal_phase = nominal_output.arg();
            let perturbed_phase = perturbed_output.arg();
            // Handle phase wrap-around
            let mut phase_diff = perturbed_phase - nominal_phase;
            if phase_diff > PI {
                phase_diff -= 2.0 * PI;
            } else if phase_diff < -PI {
                phase_diff += 2.0 * PI;
            }
            let phase_sensitivity = phase_diff / delta;

            results.push(AcSensitivityResult {
                param: param.clone(),
                output: output.clone(),
                frequency,
                magnitude_sensitivity,
                magnitude_db_sensitivity,
                phase_sensitivity,
                complex_sensitivity,
                nominal_value: nominal_output,
                nominal_param: nominal_value,
            });
        }
    }

    Ok(results)
}

/// Compute AC sensitivity across multiple frequencies.
///
/// # Arguments
/// * `stamper` - Circuit stamper
/// * `ac_params` - AC sweep parameters
/// * `config` - Sensitivity configuration
///
/// # Returns
/// Vector of sensitivity results at each frequency point.
pub fn compute_ac_sensitivity_sweep(
    stamper: &dyn AcSensitivityStamper,
    ac_params: &AcParams,
    config: &SensitivityConfig,
) -> Result<Vec<Vec<AcSensitivityResult>>> {
    // Generate frequency points
    let frequencies = crate::ac::generate_frequencies(ac_params);

    let mut all_results = Vec::with_capacity(frequencies.len());

    for freq in frequencies {
        let freq_results = compute_ac_sensitivity(stamper, freq, config)?;
        all_results.push(freq_results);
    }

    Ok(all_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ac::AcSweepType;
    use crate::sensitivity::SensitivityOutput;

    /// RC lowpass filter for AC sensitivity testing.
    /// Transfer function: H(s) = 1 / (1 + sRC)
    struct RcLowpassStamper {
        r: f64,
        c: f64,
    }

    impl AcSensitivityStamper for RcLowpassStamper {
        fn stamp_ac_nominal(&self, mna: &mut ComplexMna, omega: f64) {
            // V1 = 1V AC at node 0
            mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));
            // R from node 0 to node 1
            mna.stamp_conductance(Some(0), Some(1), 1.0 / self.r);
            // C from node 1 to ground: Y = jωC
            let yc = Complex::new(0.0, omega * self.c);
            mna.stamp_admittance(Some(1), None, yc);
        }

        fn stamp_ac_perturbed(&self, mna: &mut ComplexMna, omega: f64, param: &SensitivityParam) {
            let r = match param {
                SensitivityParam::Resistance { name, value } if name == "R" => *value,
                _ => self.r,
            };
            let c = match param {
                SensitivityParam::Capacitance { name, value } if name == "C" => *value,
                _ => self.c,
            };

            mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));
            mna.stamp_conductance(Some(0), Some(1), 1.0 / r);
            let yc = Complex::new(0.0, omega * c);
            mna.stamp_admittance(Some(1), None, yc);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_rc_lowpass_sensitivity() {
        // RC lowpass: R=1k, C=159nF → fc ≈ 1kHz
        let stamper = RcLowpassStamper {
            r: 1000.0,
            c: 159e-9,
        };

        let config = SensitivityConfig {
            delta_ratio: 1e-6,
            delta_min: 1e-15,
            params: vec![
                SensitivityParam::Resistance {
                    name: "R".to_string(),
                    value: 1000.0,
                },
                SensitivityParam::Capacitance {
                    name: "C".to_string(),
                    value: 159e-9,
                },
            ],
            outputs: vec![SensitivityOutput::voltage_named(1, "Vout")],
        };

        // Test at fc (should have -3dB gain)
        let fc = 1.0 / (2.0 * PI * 1000.0 * 159e-9);
        let results = compute_ac_sensitivity(&stamper, fc, &config).unwrap();

        assert_eq!(results.len(), 2);

        // At fc, magnitude should be ~0.707 (-3dB)
        let sr = results.iter().find(|r| r.param_name() == "R").unwrap();
        assert!(
            (sr.nominal_value.norm() - 0.707).abs() < 0.05,
            "Nominal magnitude {} should be ~0.707",
            sr.nominal_value.norm()
        );

        // Sensitivity to R should be negative (increasing R decreases bandwidth)
        // and sensitivity to C should also be negative
        let sc = results.iter().find(|r| r.param_name() == "C").unwrap();

        // Both sensitivities should have similar signs at cutoff
        // (they both affect the RC time constant similarly)
        assert!(
            sr.magnitude_sensitivity * sc.magnitude_sensitivity > 0.0
                || sr.magnitude_sensitivity.abs() < 1e-10
                || sc.magnitude_sensitivity.abs() < 1e-10,
            "R and C sensitivities should have same sign at fc"
        );
    }

    #[test]
    fn test_ac_sensitivity_sweep() {
        let stamper = RcLowpassStamper {
            r: 1000.0,
            c: 159e-9,
        };

        let config = SensitivityConfig::new(
            vec![SensitivityParam::Resistance {
                name: "R".to_string(),
                value: 1000.0,
            }],
            vec![SensitivityOutput::voltage(1)],
        );

        // 100Hz to 10kHz is 2 decades, 5 points per decade = 10 points + 1 = 11 total
        let ac_params = AcParams {
            fstart: 100.0,
            fstop: 10000.0,
            num_points: 5, // Points per decade
            sweep_type: AcSweepType::Decade,
        };

        let results = compute_ac_sensitivity_sweep(&stamper, &ac_params, &config).unwrap();

        // Should have 11 frequency points (2 decades × 5 + 1)
        assert_eq!(results.len(), 11);

        // At low frequencies, sensitivity should be small (far from cutoff)
        // At high frequencies, sensitivity should be larger
        let low_freq_sens = results[0][0].magnitude_sensitivity.abs();
        let high_freq_sens = results[results.len() - 1][0].magnitude_sensitivity.abs();

        // Sensitivity at high frequencies should generally be larger or similar
        // (The exact relationship depends on where we are relative to fc)
        assert!(
            high_freq_sens >= low_freq_sens * 0.1,
            "High freq sensitivity {} should be comparable to low {}",
            high_freq_sens,
            low_freq_sens
        );
    }
}
