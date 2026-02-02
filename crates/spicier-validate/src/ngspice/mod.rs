//! ngspice integration module.
//!
//! This module provides functionality to run netlists through ngspice
//! and parse the results.

pub mod rawfile;
pub mod runner;
pub mod types;

pub use rawfile::parse_rawfile;
pub use runner::{NgspiceConfig, is_ngspice_available, ngspice_version, run_ngspice};
pub use types::{
    AnalysisType, NgspiceAc, NgspiceDcOp, NgspiceResult, NgspiceTransient, RawVariable,
    RawfileData, RawfileHeader,
};
