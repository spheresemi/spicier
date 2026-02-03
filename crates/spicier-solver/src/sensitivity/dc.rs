//! DC sensitivity analysis using forward finite differences.
//!
//! This module computes how DC operating point outputs change with respect
//! to parameter variations using the formula:
//!
//! dOutput/dParam ≈ (Output(param + δ) - Output(param)) / δ

use spicier_core::mna::MnaSystem;

use crate::dc::solve_dc;
use crate::error::Result;

use super::config::{SensitivityConfig, SensitivityOutput, SensitivityParam};

/// Trait for circuits that support DC sensitivity analysis.
///
/// Implementors must be able to stamp the circuit with modified parameter values.
pub trait DcSensitivityStamper {
    /// Stamp the circuit with nominal parameter values.
    fn stamp_nominal(&self, mna: &mut MnaSystem);

    /// Stamp the circuit with a perturbed parameter value.
    ///
    /// # Arguments
    /// * `mna` - MNA system to stamp
    /// * `param` - The parameter being perturbed (with new value)
    fn stamp_perturbed(&self, mna: &mut MnaSystem, param: &SensitivityParam);

    /// Number of nodes (excluding ground).
    fn num_nodes(&self) -> usize;

    /// Number of voltage sources / branch currents.
    fn num_vsources(&self) -> usize;
}

/// Result of DC sensitivity analysis for a single parameter-output pair.
#[derive(Debug, Clone)]
pub struct DcSensitivityResult {
    /// The parameter that was varied.
    pub param: SensitivityParam,
    /// The output that was measured.
    pub output: SensitivityOutput,
    /// Sensitivity value: dOutput/dParam.
    pub value: f64,
    /// Normalized sensitivity: (dOutput/dParam) * (Param/Output).
    /// This is dimensionless and allows comparison across different parameters.
    pub normalized: f64,
    /// Nominal output value.
    pub nominal_output: f64,
    /// Nominal parameter value.
    pub nominal_param: f64,
}

impl DcSensitivityResult {
    /// Get the parameter name.
    pub fn param_name(&self) -> String {
        self.param.name()
    }

    /// Get the output name.
    pub fn output_name(&self) -> &str {
        self.output.name()
    }
}

/// Extract an output value from a solution vector.
fn extract_output(solution: &[f64], output: &SensitivityOutput, num_nodes: usize) -> f64 {
    match output {
        SensitivityOutput::Voltage { node_idx, .. } => {
            solution.get(*node_idx).copied().unwrap_or(0.0)
        }
        SensitivityOutput::Current { branch_idx, .. } => solution
            .get(num_nodes + *branch_idx)
            .copied()
            .unwrap_or(0.0),
        SensitivityOutput::VoltageDiff {
            node_pos, node_neg, ..
        } => {
            let vp = solution.get(*node_pos).copied().unwrap_or(0.0);
            let vn = solution.get(*node_neg).copied().unwrap_or(0.0);
            vp - vn
        }
    }
}

/// Compute DC sensitivity analysis.
///
/// Uses forward finite differences to compute dOutput/dParam for all
/// combinations of parameters and outputs specified in the config.
///
/// # Arguments
/// * `stamper` - Circuit stamper that can apply parameter perturbations
/// * `config` - Sensitivity configuration
///
/// # Returns
/// Vector of sensitivity results for each parameter-output combination.
pub fn compute_dc_sensitivity(
    stamper: &dyn DcSensitivityStamper,
    config: &SensitivityConfig,
) -> Result<Vec<DcSensitivityResult>> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();

    // Solve nominal operating point
    let mut mna_nominal = MnaSystem::new(num_nodes, num_vsources);
    stamper.stamp_nominal(&mut mna_nominal);
    let nominal_solution = solve_dc(&mna_nominal)?;

    // Extract nominal output values
    let nominal_solution_vec: Vec<f64> = nominal_solution
        .node_voltages
        .iter()
        .chain(nominal_solution.branch_currents.iter())
        .copied()
        .collect();

    let nominal_outputs: Vec<f64> = config
        .outputs
        .iter()
        .map(|o| extract_output(&nominal_solution_vec, o, num_nodes))
        .collect();

    let mut results = Vec::with_capacity(config.params.len() * config.outputs.len());

    // For each parameter
    for param in &config.params {
        let nominal_value = param.value();
        let delta = config.compute_delta(nominal_value);
        let perturbed_value = nominal_value + delta;
        let perturbed_param = param.with_value(perturbed_value);

        // Solve perturbed operating point
        let mut mna_perturbed = MnaSystem::new(num_nodes, num_vsources);
        stamper.stamp_perturbed(&mut mna_perturbed, &perturbed_param);
        let perturbed_solution = solve_dc(&mna_perturbed)?;

        let perturbed_solution_vec: Vec<f64> = perturbed_solution
            .node_voltages
            .iter()
            .chain(perturbed_solution.branch_currents.iter())
            .copied()
            .collect();

        // Compute sensitivities for each output
        for (i, output) in config.outputs.iter().enumerate() {
            let nominal_output = nominal_outputs[i];
            let perturbed_output = extract_output(&perturbed_solution_vec, output, num_nodes);

            // Forward difference: dOutput/dParam = (f(x+h) - f(x)) / h
            let sensitivity = (perturbed_output - nominal_output) / delta;

            // Normalized sensitivity: (dY/dX) * (X/Y)
            let normalized = if nominal_output.abs() > 1e-20 {
                sensitivity * (nominal_value / nominal_output)
            } else {
                0.0
            };

            results.push(DcSensitivityResult {
                param: param.clone(),
                output: output.clone(),
                value: sensitivity,
                normalized,
                nominal_output,
                nominal_param: nominal_value,
            });
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Voltage divider stamper for sensitivity testing.
    /// Circuit: V1 -- R1 -- node1 -- R2 -- GND
    struct VoltageDividerStamper {
        v1: f64,
        r1: f64,
        r2: f64,
    }

    impl DcSensitivityStamper for VoltageDividerStamper {
        fn stamp_nominal(&self, mna: &mut MnaSystem) {
            // V1 at node 0
            mna.stamp_voltage_source(Some(0), None, 0, self.v1);
            // R1 from node 0 to node 1
            mna.stamp_conductance(Some(0), Some(1), 1.0 / self.r1);
            // R2 from node 1 to ground
            mna.stamp_conductance(Some(1), None, 1.0 / self.r2);
        }

        fn stamp_perturbed(&self, mna: &mut MnaSystem, param: &SensitivityParam) {
            let v1 = match param {
                SensitivityParam::VoltageSource { name, value } if name == "V1" => *value,
                _ => self.v1,
            };
            let r1 = match param {
                SensitivityParam::Resistance { name, value } if name == "R1" => *value,
                _ => self.r1,
            };
            let r2 = match param {
                SensitivityParam::Resistance { name, value } if name == "R2" => *value,
                _ => self.r2,
            };

            mna.stamp_voltage_source(Some(0), None, 0, v1);
            mna.stamp_conductance(Some(0), Some(1), 1.0 / r1);
            mna.stamp_conductance(Some(1), None, 1.0 / r2);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_voltage_divider_sensitivity() {
        // Voltage divider: V1=10V, R1=1k, R2=1k
        // Vout = V1 * R2/(R1+R2) = 10 * 1k/2k = 5V
        let stamper = VoltageDividerStamper {
            v1: 10.0,
            r1: 1000.0,
            r2: 1000.0,
        };

        let config = SensitivityConfig {
            delta_ratio: 1e-6,
            delta_min: 1e-12,
            params: vec![
                SensitivityParam::VoltageSource {
                    name: "V1".to_string(),
                    value: 10.0,
                },
                SensitivityParam::Resistance {
                    name: "R1".to_string(),
                    value: 1000.0,
                },
                SensitivityParam::Resistance {
                    name: "R2".to_string(),
                    value: 1000.0,
                },
            ],
            outputs: vec![SensitivityOutput::voltage_named(1, "Vout")],
        };

        let results = compute_dc_sensitivity(&stamper, &config).unwrap();

        assert_eq!(results.len(), 3);

        // Find sensitivity to V1
        let sv1 = results.iter().find(|r| r.param_name() == "V1").unwrap();
        // dVout/dV1 = R2/(R1+R2) = 0.5
        assert!(
            (sv1.value - 0.5).abs() < 1e-6,
            "dVout/dV1 = {} (expected 0.5)",
            sv1.value
        );
        // Normalized: 0.5 * (10/5) = 1.0
        assert!(
            (sv1.normalized - 1.0).abs() < 1e-6,
            "Normalized dVout/dV1 = {} (expected 1.0)",
            sv1.normalized
        );

        // Find sensitivity to R1
        let sr1 = results.iter().find(|r| r.param_name() == "R1").unwrap();
        // dVout/dR1 = -V1 * R2 / (R1+R2)^2 = -10 * 1000 / 4e6 = -2.5e-3
        let expected_sr1 = -10.0 * 1000.0 / (2000.0_f64.powi(2));
        assert!(
            (sr1.value - expected_sr1).abs() < 1e-8,
            "dVout/dR1 = {} (expected {})",
            sr1.value,
            expected_sr1
        );

        // Find sensitivity to R2
        let sr2 = results.iter().find(|r| r.param_name() == "R2").unwrap();
        // dVout/dR2 = V1 * R1 / (R1+R2)^2 = 10 * 1000 / 4e6 = 2.5e-3
        let expected_sr2 = 10.0 * 1000.0 / (2000.0_f64.powi(2));
        assert!(
            (sr2.value - expected_sr2).abs() < 1e-8,
            "dVout/dR2 = {} (expected {})",
            sr2.value,
            expected_sr2
        );
    }

    #[test]
    fn test_sensitivity_symmetry() {
        // For equal R1 and R2, |dVout/dR1| should equal |dVout/dR2|
        let stamper = VoltageDividerStamper {
            v1: 5.0,
            r1: 2000.0,
            r2: 2000.0,
        };

        let config = SensitivityConfig {
            delta_ratio: 1e-6,
            delta_min: 1e-12,
            params: vec![
                SensitivityParam::Resistance {
                    name: "R1".to_string(),
                    value: 2000.0,
                },
                SensitivityParam::Resistance {
                    name: "R2".to_string(),
                    value: 2000.0,
                },
            ],
            outputs: vec![SensitivityOutput::voltage(1)],
        };

        let results = compute_dc_sensitivity(&stamper, &config).unwrap();

        let sr1 = results.iter().find(|r| r.param_name() == "R1").unwrap();
        let sr2 = results.iter().find(|r| r.param_name() == "R2").unwrap();

        // |dVout/dR1| == |dVout/dR2|
        assert!(
            (sr1.value.abs() - sr2.value.abs()).abs() < 1e-9,
            "Magnitude should be equal: {} vs {}",
            sr1.value.abs(),
            sr2.value.abs()
        );

        // Signs should be opposite
        assert!(
            sr1.value * sr2.value < 0.0,
            "Signs should be opposite: {} and {}",
            sr1.value,
            sr2.value
        );
    }
}
