//! Comparison report generation.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Comparison result for a single variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableComparison {
    /// Variable name (e.g., "V(1)", "I(V1)").
    pub name: String,
    /// Whether this variable passed comparison.
    pub passed: bool,
    /// Expected value (for DC) or description.
    pub expected: String,
    /// Actual value (for DC) or description.
    pub actual: String,
    /// Error value or description.
    pub error: String,
    /// Worst case deviation point (for AC/transient).
    pub worst_point: Option<WorstPointInfo>,
}

/// Information about the worst deviation point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorstPointInfo {
    /// Independent variable value (frequency or time).
    pub at: f64,
    /// Expected value at this point.
    pub expected: f64,
    /// Actual value at this point.
    pub actual: f64,
    /// Error at this point.
    pub error: f64,
}

/// Summary statistics for a comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Total number of variables compared.
    pub total_variables: usize,
    /// Number of variables that passed.
    pub passed_variables: usize,
    /// Number of variables that failed.
    pub failed_variables: usize,
    /// Maximum error observed.
    pub max_error: f64,
    /// Average error observed.
    pub avg_error: f64,
}

/// Complete comparison report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Path to the netlist file (if from file).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub netlist_path: Option<PathBuf>,
    /// Analysis type.
    pub analysis_type: String,
    /// Whether overall comparison passed.
    pub passed: bool,
    /// Per-variable comparison results.
    pub comparisons: Vec<VariableComparison>,
    /// Overall summary statistics.
    pub summary: ComparisonSummary,
}

impl ComparisonReport {
    /// Create a new report.
    pub fn new(analysis_type: &str) -> Self {
        Self {
            netlist_path: None,
            analysis_type: analysis_type.to_string(),
            passed: true,
            comparisons: Vec::new(),
            summary: ComparisonSummary {
                total_variables: 0,
                passed_variables: 0,
                failed_variables: 0,
                max_error: 0.0,
                avg_error: 0.0,
            },
        }
    }

    /// Add a variable comparison result.
    pub fn add_comparison(&mut self, comp: VariableComparison) {
        if !comp.passed {
            self.passed = false;
            self.summary.failed_variables += 1;
        } else {
            self.summary.passed_variables += 1;
        }
        self.summary.total_variables += 1;
        self.comparisons.push(comp);
    }

    /// Finalize the report by computing summary statistics.
    pub fn finalize(&mut self) {
        if self.comparisons.is_empty() {
            return;
        }

        let mut total_error = 0.0_f64;
        let mut max_error = 0.0_f64;
        let mut count = 0;

        for comp in &self.comparisons {
            if let Ok(err) = comp.error.parse::<f64>() {
                total_error += err.abs();
                max_error = max_error.max(err.abs());
                count += 1;
            }
        }

        self.summary.max_error = max_error;
        self.summary.avg_error = if count > 0 {
            total_error / count as f64
        } else {
            0.0
        };
    }

    /// Format as human-readable text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "Comparison Report: {} Analysis\n",
            self.analysis_type
        ));
        out.push_str(&format!("Status: {}\n", if self.passed { "PASS" } else { "FAIL" }));
        out.push_str(&format!(
            "Variables: {}/{} passed\n\n",
            self.summary.passed_variables, self.summary.total_variables
        ));

        for comp in &self.comparisons {
            let status = if comp.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!(
                "  {}: {} [{}]\n    Expected: {}\n    Actual:   {}\n    Error:    {}\n",
                comp.name, status, comp.error, comp.expected, comp.actual, comp.error
            ));

            if let Some(ref worst) = comp.worst_point {
                out.push_str(&format!(
                    "    Worst at: {:.6e} (expected={:.6e}, actual={:.6e}, error={:.6e})\n",
                    worst.at, worst.expected, worst.actual, worst.error
                ));
            }
        }

        if !self.passed {
            out.push_str("\nFailed variables:\n");
            for comp in &self.comparisons {
                if !comp.passed {
                    out.push_str(&format!("  - {}\n", comp.name));
                }
            }
        }

        out
    }
}
