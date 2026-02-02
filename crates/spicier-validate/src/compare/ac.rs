//! AC analysis comparison.

use crate::compare::report::{ComparisonReport, VariableComparison, WorstPointInfo};
use crate::compare::tolerances::AcTolerances;
use crate::ngspice::types::NgspiceAc;
use crate::spicier::SpicierAc;

/// Compare AC analysis results.
pub fn compare_ac(
    ngspice: &NgspiceAc,
    spicier: &SpicierAc,
    tolerances: &AcTolerances,
    variables: Option<&[String]>,
) -> ComparisonReport {
    let mut report = ComparisonReport::new("AC Analysis");

    // Get list of variables to compare
    let var_names: Vec<String> = match variables {
        Some(vars) => vars.to_vec(),
        None => ngspice
            .values
            .keys()
            .filter(|k| k.to_lowercase().starts_with("v("))
            .cloned()
            .collect(),
    };

    for name in var_names {
        // Get ngspice magnitude in dB
        let ng_mag = match ngspice.magnitude_db(&name) {
            Some(v) => v,
            None => {
                report.add_comparison(VariableComparison {
                    name: name.clone(),
                    passed: false,
                    expected: "N/A".to_string(),
                    actual: "not found in ngspice".to_string(),
                    error: "missing".to_string(),
                    worst_point: None,
                });
                continue;
            }
        };

        let ng_phase = ngspice.phase_deg(&name).unwrap_or_default();

        // Get spicier results
        let sp_mag = match spicier.magnitude_db(&name) {
            Some(v) => v,
            None => {
                report.add_comparison(VariableComparison {
                    name: name.clone(),
                    passed: false,
                    expected: format!("{} points", ng_mag.len()),
                    actual: "not found in spicier".to_string(),
                    error: "missing".to_string(),
                    worst_point: None,
                });
                continue;
            }
        };

        let sp_phase = spicier.phase_deg(&name).unwrap_or_default();

        // Compare at each frequency point
        // Find worst case deviation
        let mut worst_mag_error = 0.0;
        let mut worst_mag_point: Option<WorstPointInfo> = None;
        let mut worst_phase_error = 0.0;
        let mut worst_phase_point: Option<WorstPointInfo> = None;
        let mut all_passed = true;

        // Align frequency points between ngspice and spicier
        for (i, &freq) in ngspice.frequencies.iter().enumerate() {
            // Find closest frequency in spicier results
            let (j, &(sp_freq, sp_mag_val)) = sp_mag
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.0 - freq).abs();
                    let db = (b.0 - freq).abs();
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap_or((0, &(0.0, 0.0)));

            // Skip if frequencies are too different (more than 1% relative)
            if (sp_freq - freq).abs() / freq > 0.01 {
                continue;
            }

            let ng_mag_val = ng_mag[i];
            let mag_error = (sp_mag_val - ng_mag_val).abs();

            if mag_error > tolerances.magnitude_db {
                all_passed = false;
            }

            if mag_error > worst_mag_error {
                worst_mag_error = mag_error;
                worst_mag_point = Some(WorstPointInfo {
                    at: freq,
                    expected: ng_mag_val,
                    actual: sp_mag_val,
                    error: mag_error,
                });
            }

            // Compare phase
            if i < ng_phase.len() && j < sp_phase.len() {
                let ng_phase_val = ng_phase[i];
                let sp_phase_val = sp_phase[j].1;

                // Handle phase wrapping
                let mut phase_error = (sp_phase_val - ng_phase_val).abs();
                if phase_error > 180.0 {
                    phase_error = 360.0 - phase_error;
                }

                if phase_error > tolerances.phase_deg {
                    all_passed = false;
                }

                if phase_error > worst_phase_error {
                    worst_phase_error = phase_error;
                    worst_phase_point = Some(WorstPointInfo {
                        at: freq,
                        expected: ng_phase_val,
                        actual: sp_phase_val,
                        error: phase_error,
                    });
                }
            }
        }

        // Create comparison entry for magnitude
        report.add_comparison(VariableComparison {
            name: format!("{}|mag", name),
            passed: all_passed || worst_mag_error <= tolerances.magnitude_db,
            expected: format!("{} points", ng_mag.len()),
            actual: format!("{} points", sp_mag.len()),
            error: format!("{:.3} dB max", worst_mag_error),
            worst_point: worst_mag_point,
        });

        // Create comparison entry for phase
        report.add_comparison(VariableComparison {
            name: format!("{}|phase", name),
            passed: all_passed || worst_phase_error <= tolerances.phase_deg,
            expected: format!("{} points", ng_phase.len()),
            actual: format!("{} points", sp_phase.len()),
            error: format!("{:.3}° max", worst_phase_error),
            worst_point: worst_phase_point,
        });
    }

    report.finalize();
    report
}
