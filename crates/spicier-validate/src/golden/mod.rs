//! Golden data management.
//!
//! This module provides functionality for loading and using golden reference
//! data files to validate spicier simulation results.

pub mod format;

use std::path::Path;

use crate::error::{Error, Result};

pub use format::{
    AcPoint, AcSweepParams, GoldenAcTolerances, GoldenAnalysis, GoldenCircuit, GoldenDataFile,
    GoldenDcTolerances, GoldenTranTolerances, TranParams, TranPoint,
};

/// Load a golden data file from disk.
pub fn load_golden_file(path: &Path) -> Result<GoldenDataFile> {
    if !path.exists() {
        return Err(Error::GoldenDataNotFound {
            path: path.to_path_buf(),
        });
    }

    let content = std::fs::read_to_string(path)?;
    let data: GoldenDataFile =
        serde_json::from_str(&content).map_err(|e| Error::InvalidGoldenData(e.to_string()))?;

    Ok(data)
}

/// Load all golden data files from a directory.
pub fn load_golden_directory(dir: &Path) -> Result<Vec<GoldenDataFile>> {
    if !dir.is_dir() {
        return Err(Error::GoldenDataNotFound {
            path: dir.to_path_buf(),
        });
    }

    let mut files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "json") {
            match load_golden_file(&path) {
                Ok(data) => files.push(data),
                Err(e) => {
                    // Log warning but continue with other files
                    eprintln!("Warning: Failed to load {}: {}", path.display(), e);
                }
            }
        }
    }

    Ok(files)
}
