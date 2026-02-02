//! DC operating point comparison.

use crate::compare::report::{ComparisonReport, VariableComparison};
use crate::compare::tolerances::{DcTolerances, values_match};
use crate::ngspice::types::NgspiceDcOp;
use crate::spicier::SpicierDcOp;

/// Compare DC operating point results.
pub fn compare_dc_op(
    ngspice: &NgspiceDcOp,
    spicier: &SpicierDcOp,
    tolerances: &DcTolerances,
    variables: Option<&[String]>,
) -> ComparisonReport {
    let mut report = ComparisonReport::new("DC Operating Point");

    // Get list of variables to compare
    let var_names: Vec<String> = match variables {
        Some(vars) => vars.to_vec(),
        None => ngspice.values.keys().cloned().collect(),
    };

    for name in var_names {
        let ng_value = match ngspice.voltage(&name) {
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

        // Determine if this is a voltage or current variable
        let is_current = name.to_lowercase().starts_with("i(");

        let sp_value = if is_current {
            spicier.current(&name)
        } else {
            spicier.voltage(&name)
        };

        let sp_value = match sp_value {
            Some(v) => v,
            None => {
                report.add_comparison(VariableComparison {
                    name: name.clone(),
                    passed: false,
                    expected: format!("{:.9e}", ng_value),
                    actual: "not found in spicier".to_string(),
                    error: "missing".to_string(),
                    worst_point: None,
                });
                continue;
            }
        };

        // Select appropriate tolerances
        let (abs_tol, rel_tol) = if is_current {
            (tolerances.current_abs, tolerances.current_rel)
        } else {
            (tolerances.voltage_abs, tolerances.voltage_rel)
        };

        let passed = values_match(ng_value, sp_value, abs_tol, rel_tol);
        let error = sp_value - ng_value;
        let rel_error = if ng_value.abs() > 1e-15 {
            error.abs() / ng_value.abs()
        } else {
            error.abs()
        };

        report.add_comparison(VariableComparison {
            name: name.clone(),
            passed,
            expected: format!("{:.9e}", ng_value),
            actual: format!("{:.9e}", sp_value),
            error: format!("{:.3e} ({:.2}%)", error, rel_error * 100.0),
            worst_point: None,
        });
    }

    report.finalize();
    report
}
