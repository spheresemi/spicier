//! Cross-simulator validation tool for spicier.
//!
//! This crate provides functionality to compare simulation results between
//! spicier and ngspice, helping validate the correctness of spicier's
//! circuit simulation implementations.
//!
//! # Features
//!
//! - Run netlists through both ngspice and spicier
//! - Compare DC operating point, AC frequency response, and transient results
//! - Configurable tolerances for different analysis types
//! - Generate comparison reports in text or JSON format
//! - Support for golden reference data validation
//!
//! # Example
//!
//! ```ignore
//! use spicier_validate::{compare_simulators, ComparisonConfig, NgspiceConfig};
//!
//! let netlist = r#"
//! Voltage Divider
//! V1 1 0 DC 10
//! R1 1 2 1k
//! R2 2 0 1k
//! .op
//! .end
//! "#;
//!
//! let config = ComparisonConfig::default();
//! let report = compare_simulators(netlist, &config)?;
//!
//! if report.passed {
//!     println!("All results match within tolerance!");
//! } else {
//!     println!("Mismatches found:\n{}", report.to_text());
//! }
//! ```

pub mod compare;
pub mod error;
pub mod golden;
pub mod ngspice;
pub mod spicier;

// Re-export commonly used types
pub use compare::{
    AcTolerances, ComparisonConfig, ComparisonReport, ComparisonSummary, DcTolerances,
    TransientTolerances, VariableComparison, WorstPointInfo, compare_ac, compare_dc_op,
    compare_transient, relative_error, values_match,
};

pub use error::{Error, Result};

pub use golden::{
    AcPoint, AcSweepParams, GoldenAcTolerances, GoldenAnalysis, GoldenCircuit, GoldenDataFile,
    GoldenDcTolerances, GoldenTranTolerances, TranParams, TranPoint, load_golden_directory,
    load_golden_file,
};

pub use ngspice::{
    AnalysisType, NgspiceAc, NgspiceConfig, NgspiceDcOp, NgspiceResult, NgspiceTransient,
    RawVariable, RawfileData, RawfileHeader, is_ngspice_available, ngspice_version, parse_rawfile,
    run_ngspice,
};

pub use spicier::{SpicierAc, SpicierDcOp, SpicierResult, SpicierTransient, run_spicier};

/// Compare a netlist through both ngspice and spicier.
pub fn compare_simulators(netlist: &str, config: &ComparisonConfig) -> Result<ComparisonReport> {
    let ngspice_config = NgspiceConfig::default();

    if !is_ngspice_available(&ngspice_config) {
        return Err(Error::NgspiceNotFound(
            "ngspice not found in PATH".to_string(),
        ));
    }

    let ng_rawfile = run_ngspice(netlist, &ngspice_config)?;
    let ng_result = NgspiceResult::from_rawfile(&ng_rawfile);
    let sp_result = run_spicier(netlist)?;

    match (&ng_result, &sp_result) {
        (NgspiceResult::DcOp(ng), SpicierResult::DcOp(sp)) => Ok(compare_dc_op(
            ng,
            sp,
            &config.dc,
            config.variables.as_deref(),
        )),
        (NgspiceResult::Ac(ng), SpicierResult::Ac(sp)) => {
            Ok(compare_ac(ng, sp, &config.ac, config.variables.as_deref()))
        }
        (NgspiceResult::Transient(ng), SpicierResult::Transient(sp)) => Ok(compare_transient(
            ng,
            sp,
            &config.transient,
            config.variables.as_deref(),
        )),
        _ => Err(Error::AnalysisTypeMismatch {
            expected: format!("{:?}", ng_result),
            actual: format!("{:?}", sp_result),
        }),
    }
}

/// Validate spicier results against golden reference data.
pub fn validate_against_golden(circuit: &GoldenCircuit) -> Result<ComparisonReport> {
    use num_complex::Complex;
    use std::collections::HashMap;

    let sp_result = run_spicier(&circuit.netlist)?;

    match (&circuit.analysis, &sp_result) {
        (
            GoldenAnalysis::DcOp {
                results,
                tolerances,
            },
            SpicierResult::DcOp(sp),
        ) => {
            let ng = NgspiceDcOp {
                values: results.clone(),
            };
            let tol = DcTolerances {
                voltage_abs: tolerances.voltage,
                voltage_rel: 1e-6,
                current_abs: tolerances.current,
                current_rel: 1e-6,
            };
            Ok(compare_dc_op(&ng, sp, &tol, None))
        }
        (
            GoldenAnalysis::Ac {
                results,
                tolerances,
                node,
                ..
            },
            SpicierResult::Ac(sp),
        ) => {
            let frequencies: Vec<f64> = results.iter().map(|p| p.freq).collect();
            let mut values: HashMap<String, Vec<Complex<f64>>> = HashMap::new();
            let complex_vals: Vec<Complex<f64>> = results
                .iter()
                .map(|p| {
                    let mag = 10.0_f64.powf(p.mag_db / 20.0);
                    let phase_rad = p.phase_deg.to_radians();
                    Complex::from_polar(mag, phase_rad)
                })
                .collect();
            values.insert(node.clone(), complex_vals);

            let ng = NgspiceAc {
                frequencies,
                values,
            };
            let tol = AcTolerances {
                magnitude_db: tolerances.mag_db,
                phase_deg: tolerances.phase_deg,
            };
            Ok(compare_ac(&ng, sp, &tol, Some(std::slice::from_ref(node))))
        }
        (
            GoldenAnalysis::Tran {
                results,
                tolerances,
                node,
                ..
            },
            SpicierResult::Transient(sp),
        ) => {
            let times: Vec<f64> = results.iter().map(|p| p.time).collect();
            let vals: Vec<f64> = results.iter().map(|p| p.value).collect();
            let mut values: HashMap<String, Vec<f64>> = HashMap::new();
            values.insert(node.clone(), vals);

            let ng = NgspiceTransient { times, values };
            let tol = TransientTolerances {
                voltage_abs: tolerances.voltage,
                voltage_rel: 1e-3,
                time_shift: 0.0,
            };
            Ok(compare_transient(&ng, sp, &tol, Some(std::slice::from_ref(node))))
        }
        _ => Err(Error::AnalysisTypeMismatch {
            expected: "golden analysis type".to_string(),
            actual: "spicier result type".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ComparisonConfig::default();
        assert!(config.dc.voltage_abs > 0.0);
        assert!(config.ac.magnitude_db > 0.0);
        assert!(config.transient.voltage_abs > 0.0);
    }

    #[test]
    fn test_values_match() {
        assert!(values_match(1.0, 1.0, 1e-6, 1e-4));
        assert!(values_match(0.0, 1e-7, 1e-6, 1e-4));
        assert!(values_match(1.0, 1.0001, 1e-9, 1e-3));
        assert!(!values_match(1.0, 1.01, 1e-9, 1e-4));
    }

    #[test]
    #[ignore]
    fn test_compare_voltage_divider() {
        let netlist = "Voltage Divider\nV1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k\n.op\n.end\n";

        let config = ComparisonConfig::default();
        let report = compare_simulators(netlist, &config).unwrap();
        assert!(report.passed, "Report:\n{}", report.to_text());
    }
}
