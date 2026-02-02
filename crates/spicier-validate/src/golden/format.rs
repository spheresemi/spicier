//! Golden data file format types.
//!
//! These types match the existing golden data format used in spicier-parser tests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A golden data file containing multiple test circuits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenDataFile {
    /// Generator (e.g., "ngspice-43").
    pub generator: String,
    /// When this file was generated.
    pub generated_at: String,
    /// Description of this test suite.
    pub description: String,
    /// List of test circuits.
    pub circuits: Vec<GoldenCircuit>,
}

/// A single test circuit with expected results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenCircuit {
    /// Test name.
    pub name: String,
    /// Description of what this test verifies.
    pub description: String,
    /// SPICE netlist content.
    pub netlist: String,
    /// Analysis type and expected results.
    pub analysis: GoldenAnalysis,
}

/// Analysis type with expected results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GoldenAnalysis {
    /// DC operating point analysis.
    #[serde(rename = "dc_op")]
    DcOp {
        /// Expected results (variable name -> value).
        results: HashMap<String, f64>,
        /// Tolerances for comparison.
        tolerances: GoldenDcTolerances,
    },

    /// AC frequency sweep analysis.
    #[serde(rename = "ac")]
    Ac {
        /// AC sweep parameters.
        sweep: AcSweepParams,
        /// Node being measured (e.g., "V(2)").
        node: String,
        /// Expected results at each frequency.
        results: Vec<AcPoint>,
        /// Tolerances for comparison.
        tolerances: GoldenAcTolerances,
    },

    /// Transient analysis.
    #[serde(rename = "tran")]
    Tran {
        /// Transient parameters.
        params: TranParams,
        /// Node being measured.
        node: String,
        /// Expected results at each time point.
        results: Vec<TranPoint>,
        /// Tolerances for comparison.
        tolerances: GoldenTranTolerances,
    },
}

/// Tolerances for DC analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenDcTolerances {
    /// Voltage tolerance (V).
    pub voltage: f64,
    /// Current tolerance (A).
    #[serde(default)]
    pub current: f64,
}

/// Tolerances for AC analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenAcTolerances {
    /// Magnitude tolerance (dB).
    pub mag_db: f64,
    /// Phase tolerance (degrees).
    pub phase_deg: f64,
}

/// Tolerances for transient analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenTranTolerances {
    /// Voltage tolerance (V).
    pub voltage: f64,
}

/// AC sweep parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcSweepParams {
    /// Sweep type: "lin", "dec", or "oct".
    #[serde(rename = "type")]
    pub sweep_type: String,
    /// Number of points (total or per decade/octave).
    pub points: usize,
    /// Start frequency (Hz).
    pub fstart: f64,
    /// Stop frequency (Hz).
    pub fstop: f64,
}

/// A single AC frequency point result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcPoint {
    /// Frequency (Hz).
    pub freq: f64,
    /// Magnitude (dB).
    pub mag_db: f64,
    /// Phase (degrees).
    pub phase_deg: f64,
}

/// Transient parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranParams {
    /// Time step (s).
    pub tstep: f64,
    /// Stop time (s).
    pub tstop: f64,
    /// Start time (s).
    #[serde(default)]
    pub tstart: f64,
    /// Use initial conditions.
    #[serde(default)]
    pub uic: bool,
}

/// A single transient time point result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranPoint {
    /// Time (s).
    pub time: f64,
    /// Value.
    pub value: f64,
}
