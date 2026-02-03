//! Measurement evaluation for .MEAS statements.
//!
//! This module provides evaluation of measurements against simulation results.

use std::collections::HashMap;

use spicier_parser::{MeasureAnalysis, MeasureType, Measurement, StatFunc, TriggerType};

use crate::ac::AcResult;
use crate::dc::{DcSolution, DcSweepResult};
use crate::transient::TransientResult;

/// Error type for measurement evaluation.
#[derive(Debug, Clone)]
pub enum MeasureError {
    /// The measurement type is not valid for this analysis.
    InvalidForAnalysis,
    /// Expression could not be evaluated.
    ExpressionError(String),
    /// Trigger condition was not found.
    TriggerNotFound,
    /// No data points available.
    NoData,
}

impl std::fmt::Display for MeasureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasureError::InvalidForAnalysis => write!(f, "invalid for this analysis type"),
            MeasureError::ExpressionError(e) => write!(f, "expression error: {}", e),
            MeasureError::TriggerNotFound => write!(f, "trigger condition not found"),
            MeasureError::NoData => write!(f, "no data points"),
        }
    }
}

impl std::error::Error for MeasureError {}

/// Result of evaluating a measurement.
#[derive(Debug, Clone)]
pub struct MeasureResult {
    /// Name of the measurement.
    pub name: String,
    /// The measured value (None if measurement failed).
    pub value: Option<f64>,
    /// Error message if measurement failed.
    pub error: Option<String>,
}

/// Evaluates measurements against simulation results.
pub struct MeasureEvaluator;

impl MeasureEvaluator {
    /// Evaluate a TRAN measurement against transient results.
    pub fn eval_tran(
        meas: &Measurement,
        result: &TransientResult,
        node_map: &HashMap<String, usize>,
    ) -> MeasureResult {
        if !matches!(meas.analysis, MeasureAnalysis::Tran) {
            return MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some("not a TRAN measurement".to_string()),
            };
        }

        match Self::eval_tran_inner(meas, result, node_map) {
            Ok(value) => MeasureResult {
                name: meas.name.clone(),
                value: Some(value),
                error: None,
            },
            Err(e) => MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some(e.to_string()),
            },
        }
    }

    fn eval_tran_inner(
        meas: &Measurement,
        result: &TransientResult,
        node_map: &HashMap<String, usize>,
    ) -> Result<f64, MeasureError> {
        if result.points.is_empty() {
            return Err(MeasureError::NoData);
        }

        match &meas.measure_type {
            MeasureType::TrigTarg {
                trig_expr,
                trig_val,
                trig_type,
                targ_expr,
                targ_val,
                targ_type,
            } => {
                let trig_values = Self::extract_waveform(result, trig_expr, node_map)?;
                let targ_values = Self::extract_waveform(result, targ_expr, node_map)?;
                let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

                let trig_time = Self::find_crossing(&times, &trig_values, *trig_val, trig_type)?;
                let targ_time = Self::find_crossing(&times, &targ_values, *targ_val, targ_type)?;

                Ok(targ_time - trig_time)
            }
            MeasureType::FindWhen {
                find_expr,
                when_expr,
                when_val,
                when_type,
            } => {
                let find_values = Self::extract_waveform(result, find_expr, node_map)?;
                let when_values = Self::extract_waveform(result, when_expr, node_map)?;
                let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

                let when_time = Self::find_crossing(&times, &when_values, *when_val, when_type)?;

                // Interpolate find_values at when_time
                Self::interpolate_at_time(&times, &find_values, when_time)
            }
            MeasureType::FindAt {
                find_expr,
                at_value,
            } => {
                let find_values = Self::extract_waveform(result, find_expr, node_map)?;
                let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

                Self::interpolate_at_time(&times, &find_values, *at_value)
            }
            MeasureType::Statistic {
                func,
                expr,
                from,
                to,
            } => {
                let values = Self::extract_waveform(result, expr, node_map)?;
                let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

                let t_start = from.unwrap_or(0.0);
                let t_end = to.unwrap_or_else(|| times.last().copied().unwrap_or(0.0));

                // Filter to range
                let (filtered_times, filtered_values): (Vec<f64>, Vec<f64>) = times
                    .iter()
                    .zip(values.iter())
                    .filter(|(t, _)| **t >= t_start && **t <= t_end)
                    .map(|(t, v)| (*t, *v))
                    .unzip();

                if filtered_values.is_empty() {
                    return Err(MeasureError::NoData);
                }

                Self::eval_statistic(func, &filtered_times, &filtered_values)
            }
            _ => Err(MeasureError::InvalidForAnalysis),
        }
    }

    /// Evaluate a DC operating point measurement.
    ///
    /// DC operating point measurements are limited to simple value extraction
    /// since there's only a single data point.
    pub fn eval_dc(
        meas: &Measurement,
        solution: &DcSolution,
        node_map: &HashMap<String, usize>,
    ) -> MeasureResult {
        if !matches!(meas.analysis, MeasureAnalysis::Dc) {
            return MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some("not a DC measurement".to_string()),
            };
        }

        match Self::eval_dc_inner(meas, solution, node_map) {
            Ok(value) => MeasureResult {
                name: meas.name.clone(),
                value: Some(value),
                error: None,
            },
            Err(e) => MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some(e.to_string()),
            },
        }
    }

    fn eval_dc_inner(
        meas: &Measurement,
        solution: &DcSolution,
        node_map: &HashMap<String, usize>,
    ) -> Result<f64, MeasureError> {
        // DC operating point only supports simple value extraction
        match &meas.measure_type {
            MeasureType::FindAt { find_expr, .. } => {
                // For DC OP, ignore at_value and just return the value
                Self::extract_dc_value(solution, find_expr, node_map)
            }
            MeasureType::Statistic { func, expr, .. } => {
                // For single point, all statistics return the same value
                let value = Self::extract_dc_value(solution, expr, node_map)?;
                match func {
                    StatFunc::Avg | StatFunc::Rms | StatFunc::Min | StatFunc::Max => Ok(value),
                    StatFunc::Pp | StatFunc::Integ => Ok(0.0), // Single point: PP=0, integral=0
                    _ => Err(MeasureError::InvalidForAnalysis),
                }
            }
            _ => Err(MeasureError::InvalidForAnalysis),
        }
    }

    /// Evaluate a DC sweep measurement.
    pub fn eval_dc_sweep(
        meas: &Measurement,
        result: &DcSweepResult,
        node_map: &HashMap<String, usize>,
    ) -> MeasureResult {
        if !matches!(meas.analysis, MeasureAnalysis::Dc) {
            return MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some("not a DC measurement".to_string()),
            };
        }

        match Self::eval_dc_sweep_inner(meas, result, node_map) {
            Ok(value) => MeasureResult {
                name: meas.name.clone(),
                value: Some(value),
                error: None,
            },
            Err(e) => MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some(e.to_string()),
            },
        }
    }

    fn eval_dc_sweep_inner(
        meas: &Measurement,
        result: &DcSweepResult,
        node_map: &HashMap<String, usize>,
    ) -> Result<f64, MeasureError> {
        if result.solutions.is_empty() {
            return Err(MeasureError::NoData);
        }

        match &meas.measure_type {
            MeasureType::FindAt {
                find_expr,
                at_value,
            } => {
                // Extract waveform across sweep
                let values = Self::extract_dc_sweep_waveform(result, find_expr, node_map)?;
                let sweep_values = &result.sweep_values;

                // Interpolate at the specified sweep value
                Self::interpolate_at_time(sweep_values, &values, *at_value)
            }
            MeasureType::FindWhen {
                find_expr,
                when_expr,
                when_val,
                when_type,
            } => {
                let find_values = Self::extract_dc_sweep_waveform(result, find_expr, node_map)?;
                let when_values = Self::extract_dc_sweep_waveform(result, when_expr, node_map)?;
                let sweep_values = &result.sweep_values;

                let when_sweep =
                    Self::find_crossing(sweep_values, &when_values, *when_val, when_type)?;
                Self::interpolate_at_time(sweep_values, &find_values, when_sweep)
            }
            MeasureType::TrigTarg {
                trig_expr,
                trig_val,
                trig_type,
                targ_expr,
                targ_val,
                targ_type,
            } => {
                let trig_values = Self::extract_dc_sweep_waveform(result, trig_expr, node_map)?;
                let targ_values = Self::extract_dc_sweep_waveform(result, targ_expr, node_map)?;
                let sweep_values = &result.sweep_values;

                let trig_point =
                    Self::find_crossing(sweep_values, &trig_values, *trig_val, trig_type)?;
                let targ_point =
                    Self::find_crossing(sweep_values, &targ_values, *targ_val, targ_type)?;

                Ok(targ_point - trig_point)
            }
            MeasureType::Statistic {
                func,
                expr,
                from,
                to,
            } => {
                let values = Self::extract_dc_sweep_waveform(result, expr, node_map)?;
                let sweep_values = &result.sweep_values;

                let sv_start = from.unwrap_or_else(|| sweep_values.first().copied().unwrap_or(0.0));
                let sv_end = to.unwrap_or_else(|| sweep_values.last().copied().unwrap_or(0.0));

                // Filter to range
                let (filtered_sweep, filtered_values): (Vec<f64>, Vec<f64>) = sweep_values
                    .iter()
                    .zip(values.iter())
                    .filter(|(sv, _)| **sv >= sv_start && **sv <= sv_end)
                    .map(|(sv, v)| (*sv, *v))
                    .unzip();

                if filtered_values.is_empty() {
                    return Err(MeasureError::NoData);
                }

                Self::eval_statistic(func, &filtered_sweep, &filtered_values)
            }
            _ => Err(MeasureError::InvalidForAnalysis),
        }
    }

    /// Evaluate an AC measurement.
    pub fn eval_ac(
        meas: &Measurement,
        result: &AcResult,
        node_map: &HashMap<String, usize>,
    ) -> MeasureResult {
        if !matches!(meas.analysis, MeasureAnalysis::Ac) {
            return MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some("not an AC measurement".to_string()),
            };
        }

        match Self::eval_ac_inner(meas, result, node_map) {
            Ok(value) => MeasureResult {
                name: meas.name.clone(),
                value: Some(value),
                error: None,
            },
            Err(e) => MeasureResult {
                name: meas.name.clone(),
                value: None,
                error: Some(e.to_string()),
            },
        }
    }

    fn eval_ac_inner(
        meas: &Measurement,
        result: &AcResult,
        node_map: &HashMap<String, usize>,
    ) -> Result<f64, MeasureError> {
        if result.points.is_empty() {
            return Err(MeasureError::NoData);
        }

        match &meas.measure_type {
            MeasureType::FindAt {
                find_expr,
                at_value,
            } => {
                // Extract magnitude waveform across frequency
                let values = Self::extract_ac_waveform(result, find_expr, node_map)?;
                let freqs: Vec<f64> = result.points.iter().map(|p| p.frequency).collect();

                // Interpolate at the specified frequency (using log scale for frequency)
                Self::interpolate_log_freq(&freqs, &values, *at_value)
            }
            MeasureType::FindWhen {
                find_expr,
                when_expr,
                when_val,
                when_type,
            } => {
                let find_values = Self::extract_ac_waveform(result, find_expr, node_map)?;
                let when_values = Self::extract_ac_waveform(result, when_expr, node_map)?;
                let freqs: Vec<f64> = result.points.iter().map(|p| p.frequency).collect();

                let when_freq = Self::find_crossing(&freqs, &when_values, *when_val, when_type)?;
                Self::interpolate_log_freq(&freqs, &find_values, when_freq)
            }
            MeasureType::TrigTarg {
                trig_expr,
                trig_val,
                trig_type,
                targ_expr,
                targ_val,
                targ_type,
            } => {
                let trig_values = Self::extract_ac_waveform(result, trig_expr, node_map)?;
                let targ_values = Self::extract_ac_waveform(result, targ_expr, node_map)?;
                let freqs: Vec<f64> = result.points.iter().map(|p| p.frequency).collect();

                let trig_freq = Self::find_crossing(&freqs, &trig_values, *trig_val, trig_type)?;
                let targ_freq = Self::find_crossing(&freqs, &targ_values, *targ_val, targ_type)?;

                // Return ratio of frequencies (useful for bandwidth measurements)
                Ok(targ_freq / trig_freq)
            }
            MeasureType::Statistic {
                func,
                expr,
                from,
                to,
            } => {
                let values = Self::extract_ac_waveform(result, expr, node_map)?;
                let freqs: Vec<f64> = result.points.iter().map(|p| p.frequency).collect();

                let f_start = from.unwrap_or_else(|| freqs.first().copied().unwrap_or(0.0));
                let f_end = to.unwrap_or_else(|| freqs.last().copied().unwrap_or(0.0));

                // Filter to range
                let (filtered_freqs, filtered_values): (Vec<f64>, Vec<f64>) = freqs
                    .iter()
                    .zip(values.iter())
                    .filter(|(f, _)| **f >= f_start && **f <= f_end)
                    .map(|(f, v)| (*f, *v))
                    .unzip();

                if filtered_values.is_empty() {
                    return Err(MeasureError::NoData);
                }

                Self::eval_statistic(func, &filtered_freqs, &filtered_values)
            }
            _ => Err(MeasureError::InvalidForAnalysis),
        }
    }

    /// Extract a single DC value for an expression.
    fn extract_dc_value(
        solution: &DcSolution,
        expr: &str,
        node_map: &HashMap<String, usize>,
    ) -> Result<f64, MeasureError> {
        let expr_upper = expr.to_uppercase();
        if expr_upper.starts_with("V(") && expr_upper.ends_with(')') {
            let node_name = &expr[2..expr.len() - 1];
            if let Some(&idx) = node_map.get(&node_name.to_uppercase()) {
                if idx < solution.num_nodes {
                    return Ok(solution.node_voltages[idx]);
                }
            } else if let Some(&idx) = node_map.get(node_name) {
                if idx < solution.num_nodes {
                    return Ok(solution.node_voltages[idx]);
                }
            }
            Err(MeasureError::ExpressionError(format!(
                "unknown node: {}",
                node_name
            )))
        } else {
            Err(MeasureError::ExpressionError(format!(
                "unsupported expression: {}",
                expr
            )))
        }
    }

    /// Extract waveform values from DC sweep results.
    fn extract_dc_sweep_waveform(
        result: &DcSweepResult,
        expr: &str,
        node_map: &HashMap<String, usize>,
    ) -> Result<Vec<f64>, MeasureError> {
        let expr_upper = expr.to_uppercase();
        if expr_upper.starts_with("V(") && expr_upper.ends_with(')') {
            let node_name = &expr[2..expr.len() - 1];
            let idx = if let Some(&idx) = node_map.get(&node_name.to_uppercase()) {
                idx
            } else if let Some(&idx) = node_map.get(node_name) {
                idx
            } else {
                return Err(MeasureError::ExpressionError(format!(
                    "unknown node: {}",
                    node_name
                )));
            };

            Ok(result
                .solutions
                .iter()
                .map(|sol| {
                    if idx < sol.num_nodes {
                        sol.node_voltages[idx]
                    } else {
                        0.0
                    }
                })
                .collect())
        } else {
            Err(MeasureError::ExpressionError(format!(
                "unsupported expression: {}",
                expr
            )))
        }
    }

    /// Extract magnitude waveform from AC results.
    ///
    /// For expressions like V(node), returns magnitude in linear scale.
    /// For VM(node), returns magnitude in linear scale.
    /// For VDB(node), returns magnitude in dB.
    /// For VP(node), returns phase in degrees.
    fn extract_ac_waveform(
        result: &AcResult,
        expr: &str,
        node_map: &HashMap<String, usize>,
    ) -> Result<Vec<f64>, MeasureError> {
        let expr_upper = expr.to_uppercase();

        // Parse the expression type
        let (node_name, value_type) = if expr_upper.starts_with("VDB(") && expr_upper.ends_with(')')
        {
            (&expr[4..expr.len() - 1], "db")
        } else if expr_upper.starts_with("VP(") && expr_upper.ends_with(')') {
            (&expr[3..expr.len() - 1], "phase")
        } else if expr_upper.starts_with("VM(") && expr_upper.ends_with(')') {
            (&expr[3..expr.len() - 1], "mag")
        } else if expr_upper.starts_with("V(") && expr_upper.ends_with(')') {
            (&expr[2..expr.len() - 1], "mag") // Default to magnitude
        } else {
            return Err(MeasureError::ExpressionError(format!(
                "unsupported AC expression: {}",
                expr
            )));
        };

        let idx = if let Some(&idx) = node_map.get(&node_name.to_uppercase()) {
            idx
        } else if let Some(&idx) = node_map.get(node_name) {
            idx
        } else {
            return Err(MeasureError::ExpressionError(format!(
                "unknown node: {}",
                node_name
            )));
        };

        Ok(result
            .points
            .iter()
            .map(|p| {
                let v = p.solution[idx];
                match value_type {
                    "db" => 20.0 * v.norm().log10(),
                    "phase" => v.arg() * 180.0 / std::f64::consts::PI,
                    _ => v.norm(), // magnitude
                }
            })
            .collect())
    }

    /// Interpolate value at a frequency using log-scale interpolation.
    fn interpolate_log_freq(
        freqs: &[f64],
        values: &[f64],
        target_freq: f64,
    ) -> Result<f64, MeasureError> {
        if freqs.is_empty() {
            return Err(MeasureError::NoData);
        }

        if target_freq <= freqs[0] {
            return Ok(values[0]);
        }
        if target_freq >= *freqs.last().unwrap() {
            return Ok(*values.last().unwrap());
        }

        // Find interval and interpolate in log-frequency space
        for i in 0..freqs.len() - 1 {
            let f0 = freqs[i];
            let f1 = freqs[i + 1];

            if target_freq >= f0 && target_freq <= f1 {
                let log_f0 = f0.ln();
                let log_f1 = f1.ln();
                let log_target = target_freq.ln();
                let alpha = (log_target - log_f0) / (log_f1 - log_f0);
                return Ok(values[i] * (1.0 - alpha) + values[i + 1] * alpha);
            }
        }

        Err(MeasureError::NoData)
    }

    /// Extract waveform values for an expression (currently supports simple V(node) only).
    fn extract_waveform(
        result: &TransientResult,
        expr: &str,
        node_map: &HashMap<String, usize>,
    ) -> Result<Vec<f64>, MeasureError> {
        // Parse simple V(node) expression
        let expr_upper = expr.to_uppercase();
        if expr_upper.starts_with("V(") && expr_upper.ends_with(')') {
            let node_name = &expr[2..expr.len() - 1];
            if let Some(&idx) = node_map.get(&node_name.to_uppercase()) {
                Ok(result.points.iter().map(|p| p.solution[idx]).collect())
            } else if let Some(&idx) = node_map.get(node_name) {
                Ok(result.points.iter().map(|p| p.solution[idx]).collect())
            } else {
                Err(MeasureError::ExpressionError(format!(
                    "unknown node: {}",
                    node_name
                )))
            }
        } else {
            Err(MeasureError::ExpressionError(format!(
                "unsupported expression: {}",
                expr
            )))
        }
    }

    /// Find the time when values cross a threshold.
    fn find_crossing(
        times: &[f64],
        values: &[f64],
        threshold: f64,
        trigger: &TriggerType,
    ) -> Result<f64, MeasureError> {
        let n = match trigger {
            TriggerType::Rise(n) | TriggerType::Fall(n) | TriggerType::Cross(n) => *n,
        };

        let mut count = 0;

        for i in 0..values.len() - 1 {
            let v0 = values[i];
            let v1 = values[i + 1];
            let t0 = times[i];
            let t1 = times[i + 1];

            let is_crossing = match trigger {
                TriggerType::Rise(_) => v0 < threshold && v1 >= threshold,
                TriggerType::Fall(_) => v0 > threshold && v1 <= threshold,
                TriggerType::Cross(_) => {
                    (v0 < threshold && v1 >= threshold) || (v0 > threshold && v1 <= threshold)
                }
            };

            if is_crossing {
                count += 1;
                if count == n {
                    // Linear interpolation to find exact crossing time
                    if (v1 - v0).abs() < 1e-30 {
                        return Ok(t0);
                    }
                    let alpha = (threshold - v0) / (v1 - v0);
                    return Ok(t0 + alpha * (t1 - t0));
                }
            }
        }

        Err(MeasureError::TriggerNotFound)
    }

    /// Interpolate value at a specific time.
    fn interpolate_at_time(times: &[f64], values: &[f64], time: f64) -> Result<f64, MeasureError> {
        if times.is_empty() {
            return Err(MeasureError::NoData);
        }

        // Handle boundary cases
        if time <= times[0] {
            return Ok(values[0]);
        }
        if time >= *times.last().unwrap() {
            return Ok(*values.last().unwrap());
        }

        // Find interval
        for i in 0..times.len() - 1 {
            let t0 = times[i];
            let t1 = times[i + 1];

            if time >= t0 && time <= t1 {
                let alpha = (time - t0) / (t1 - t0);
                return Ok(values[i] * (1.0 - alpha) + values[i + 1] * alpha);
            }
        }

        Err(MeasureError::NoData)
    }

    /// Evaluate a statistical function over values.
    fn eval_statistic(func: &StatFunc, times: &[f64], values: &[f64]) -> Result<f64, MeasureError> {
        if values.is_empty() {
            return Err(MeasureError::NoData);
        }

        match func {
            StatFunc::Avg => {
                let sum: f64 = values.iter().sum();
                Ok(sum / values.len() as f64)
            }
            StatFunc::Rms => {
                let sum_sq: f64 = values.iter().map(|v| v * v).sum();
                Ok((sum_sq / values.len() as f64).sqrt())
            }
            StatFunc::Min => Ok(values.iter().copied().fold(f64::INFINITY, f64::min)),
            StatFunc::Max => Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
            StatFunc::Pp => {
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                Ok(max - min)
            }
            StatFunc::Integ => {
                // Trapezoidal integration
                if times.len() < 2 {
                    return Ok(0.0);
                }
                let mut integral = 0.0;
                for i in 0..times.len() - 1 {
                    let dt = times[i + 1] - times[i];
                    let avg_val = (values[i] + values[i + 1]) / 2.0;
                    integral += dt * avg_val;
                }
                Ok(integral)
            }
            _ => Err(MeasureError::InvalidForAnalysis),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient::TimePoint;
    use nalgebra::DVector;

    fn make_test_result() -> TransientResult {
        // Create a simple ramp from 0 to 1 over 10 time points
        let points: Vec<TimePoint> = (0..10)
            .map(|i| {
                let t = i as f64 * 0.1;
                let v = t; // Simple ramp
                TimePoint {
                    time: t,
                    solution: DVector::from_vec(vec![v]),
                }
            })
            .collect();

        TransientResult {
            points,
            num_nodes: 1,
        }
    }

    #[test]
    fn test_measure_max() {
        let result = make_test_result();
        let mut node_map = HashMap::new();
        node_map.insert("1".to_string(), 0);

        let meas = Measurement {
            name: "vmax".to_string(),
            analysis: MeasureAnalysis::Tran,
            measure_type: MeasureType::Statistic {
                func: StatFunc::Max,
                expr: "V(1)".to_string(),
                from: None,
                to: None,
            },
        };

        let mr = MeasureEvaluator::eval_tran(&meas, &result, &node_map);
        assert!(mr.value.is_some());
        assert!((mr.value.unwrap() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_measure_min() {
        let result = make_test_result();
        let mut node_map = HashMap::new();
        node_map.insert("1".to_string(), 0);

        let meas = Measurement {
            name: "vmin".to_string(),
            analysis: MeasureAnalysis::Tran,
            measure_type: MeasureType::Statistic {
                func: StatFunc::Min,
                expr: "V(1)".to_string(),
                from: None,
                to: None,
            },
        };

        let mr = MeasureEvaluator::eval_tran(&meas, &result, &node_map);
        assert!(mr.value.is_some());
        assert!((mr.value.unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure_avg() {
        let result = make_test_result();
        let mut node_map = HashMap::new();
        node_map.insert("1".to_string(), 0);

        let meas = Measurement {
            name: "vavg".to_string(),
            analysis: MeasureAnalysis::Tran,
            measure_type: MeasureType::Statistic {
                func: StatFunc::Avg,
                expr: "V(1)".to_string(),
                from: None,
                to: None,
            },
        };

        let mr = MeasureEvaluator::eval_tran(&meas, &result, &node_map);
        assert!(mr.value.is_some());
        // Average of 0, 0.1, 0.2, ..., 0.9 = 0.45
        assert!((mr.value.unwrap() - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_measure_find_at() {
        let result = make_test_result();
        let mut node_map = HashMap::new();
        node_map.insert("1".to_string(), 0);

        let meas = Measurement {
            name: "vmid".to_string(),
            analysis: MeasureAnalysis::Tran,
            measure_type: MeasureType::FindAt {
                find_expr: "V(1)".to_string(),
                at_value: 0.45, // time = 0.45
            },
        };

        let mr = MeasureEvaluator::eval_tran(&meas, &result, &node_map);
        assert!(mr.value.is_some());
        // At t=0.45, V should be 0.45 (linear ramp)
        assert!((mr.value.unwrap() - 0.45).abs() < 1e-10);
    }
}
