//! Tolerance configurations for comparison.

use serde::{Deserialize, Serialize};

/// Tolerances for DC analysis comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcTolerances {
    /// Absolute voltage tolerance (V).
    pub voltage_abs: f64,
    /// Relative voltage tolerance (fraction).
    pub voltage_rel: f64,
    /// Absolute current tolerance (A).
    pub current_abs: f64,
    /// Relative current tolerance (fraction).
    pub current_rel: f64,
}

impl Default for DcTolerances {
    fn default() -> Self {
        Self {
            voltage_abs: 1e-6,
            voltage_rel: 1e-4,
            current_abs: 1e-9,
            current_rel: 1e-4,
        }
    }
}

/// Tolerances for AC analysis comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcTolerances {
    /// Magnitude tolerance (dB).
    pub magnitude_db: f64,
    /// Phase tolerance (degrees).
    pub phase_deg: f64,
}

impl Default for AcTolerances {
    fn default() -> Self {
        Self {
            magnitude_db: 0.1,
            phase_deg: 1.0,
        }
    }
}

/// Tolerances for transient analysis comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransientTolerances {
    /// Absolute voltage tolerance (V).
    pub voltage_abs: f64,
    /// Relative voltage tolerance (fraction).
    pub voltage_rel: f64,
    /// Maximum time shift tolerance (s).
    pub time_shift: f64,
}

impl Default for TransientTolerances {
    fn default() -> Self {
        Self {
            voltage_abs: 1e-4,
            voltage_rel: 1e-3,
            time_shift: 0.0,
        }
    }
}

/// Complete comparison configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// DC analysis tolerances.
    pub dc: DcTolerances,
    /// AC analysis tolerances.
    pub ac: AcTolerances,
    /// Transient analysis tolerances.
    pub transient: TransientTolerances,
    /// Specific variables to compare (None = compare all).
    #[serde(default)]
    pub variables: Option<Vec<String>>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            dc: DcTolerances::default(),
            ac: AcTolerances::default(),
            transient: TransientTolerances::default(),
            variables: None,
        }
    }
}

impl ComparisonConfig {
    /// Create with custom DC tolerances.
    pub fn with_dc_tolerances(mut self, tol: DcTolerances) -> Self {
        self.dc = tol;
        self
    }

    /// Create with custom AC tolerances.
    pub fn with_ac_tolerances(mut self, tol: AcTolerances) -> Self {
        self.ac = tol;
        self
    }

    /// Create with custom transient tolerances.
    pub fn with_transient_tolerances(mut self, tol: TransientTolerances) -> Self {
        self.transient = tol;
        self
    }

    /// Limit comparison to specific variables.
    pub fn with_variables(mut self, vars: Vec<String>) -> Self {
        self.variables = Some(vars);
        self
    }
}

/// Check if two values match within absolute and relative tolerances.
pub fn values_match(expected: f64, actual: f64, abs_tol: f64, rel_tol: f64) -> bool {
    let abs_diff = (expected - actual).abs();

    // Check absolute tolerance
    if abs_diff <= abs_tol {
        return true;
    }

    // Check relative tolerance (relative to expected value)
    if expected.abs() > 0.0 {
        let rel_diff = abs_diff / expected.abs();
        if rel_diff <= rel_tol {
            return true;
        }
    }

    false
}

/// Calculate relative error between two values.
pub fn relative_error(expected: f64, actual: f64) -> f64 {
    if expected.abs() < 1e-15 {
        if actual.abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (actual - expected).abs() / expected.abs()
    }
}
