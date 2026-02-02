//! Transient analysis comparison.

use crate::compare::report::{ComparisonReport, VariableComparison, WorstPointInfo};
use crate::compare::tolerances::{TransientTolerances, values_match};
use crate::ngspice::types::NgspiceTransient;
use crate::spicier::SpicierTransient;

/// Compare transient analysis results.
pub fn compare_transient(
    ngspice: &NgspiceTransient,
    spicier: &SpicierTransient,
    tolerances: &TransientTolerances,
    variables: Option<&[String]>,
) -> ComparisonReport {
    let mut report = ComparisonReport::new("Transient Analysis");

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
        // Get ngspice values
        let ng_values = match ngspice.values.get(&name) {
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

        // Get spicier waveform
        let sp_waveform = match spicier.voltage(&name) {
            Some(v) => v,
            None => {
                report.add_comparison(VariableComparison {
                    name: name.clone(),
                    passed: false,
                    expected: format!("{} points", ng_values.len()),
                    actual: "not found in spicier".to_string(),
                    error: "missing".to_string(),
                    worst_point: None,
                });
                continue;
            }
        };

        // Compare at each ngspice timepoint using interpolation from spicier
        let mut worst_error = 0.0;
        let mut worst_point: Option<WorstPointInfo> = None;
        let mut all_passed = true;
        let mut total_squared_error = 0.0;
        let mut count = 0;

        for (i, &time) in ngspice.times.iter().enumerate() {
            let ng_value = ng_values[i];

            // Get spicier value at this time (interpolated)
            let sp_value = match spicier.voltage_at(&name, time) {
                Some(v) => v,
                None => {
                    // Time outside range - find closest
                    sp_waveform
                        .iter()
                        .min_by(|a, b| {
                            let da = (a.0 - time).abs();
                            let db = (b.0 - time).abs();
                            da.partial_cmp(&db).unwrap()
                        })
                        .map(|&(_, v)| v)
                        .unwrap_or(0.0)
                }
            };

            let passed = values_match(
                ng_value,
                sp_value,
                tolerances.voltage_abs,
                tolerances.voltage_rel,
            );

            if !passed {
                all_passed = false;
            }

            let error = (sp_value - ng_value).abs();
            total_squared_error += error * error;
            count += 1;

            if error > worst_error {
                worst_error = error;
                worst_point = Some(WorstPointInfo {
                    at: time,
                    expected: ng_value,
                    actual: sp_value,
                    error,
                });
            }
        }

        let rms_error = if count > 0 {
            (total_squared_error / count as f64).sqrt()
        } else {
            0.0
        };

        report.add_comparison(VariableComparison {
            name: name.clone(),
            passed: all_passed,
            expected: format!("{} points", ngspice.times.len()),
            actual: format!("{} points", sp_waveform.len()),
            error: format!("{:.3e} max, {:.3e} RMS", worst_error, rms_error),
            worst_point,
        });
    }

    report.finalize();
    report
}
