//! Configuration types for sensitivity analysis.

/// A parameter that can be varied for sensitivity analysis.
#[derive(Debug, Clone)]
pub enum SensitivityParam {
    /// Resistance value (Ohms).
    Resistance {
        /// Component name (e.g., "R1").
        name: String,
        /// Nominal resistance value.
        value: f64,
    },
    /// Capacitance value (Farads).
    Capacitance {
        /// Component name (e.g., "C1").
        name: String,
        /// Nominal capacitance value.
        value: f64,
    },
    /// Inductance value (Henries).
    Inductance {
        /// Component name (e.g., "L1").
        name: String,
        /// Nominal inductance value.
        value: f64,
    },
    /// Voltage source value (Volts).
    VoltageSource {
        /// Component name (e.g., "V1").
        name: String,
        /// Nominal voltage value.
        value: f64,
    },
    /// Current source value (Amps).
    CurrentSource {
        /// Component name (e.g., "I1").
        name: String,
        /// Nominal current value.
        value: f64,
    },
    /// Generic device parameter.
    DeviceParam {
        /// Device name.
        device_name: String,
        /// Parameter name.
        param: String,
        /// Nominal value.
        value: f64,
    },
    /// Model parameter (affects all devices using the model).
    ModelParam {
        /// Model name.
        model_name: String,
        /// Parameter name.
        param: String,
        /// Nominal value.
        value: f64,
    },
}

impl SensitivityParam {
    /// Get the parameter name for display.
    pub fn name(&self) -> String {
        match self {
            SensitivityParam::Resistance { name, .. } => name.clone(),
            SensitivityParam::Capacitance { name, .. } => name.clone(),
            SensitivityParam::Inductance { name, .. } => name.clone(),
            SensitivityParam::VoltageSource { name, .. } => name.clone(),
            SensitivityParam::CurrentSource { name, .. } => name.clone(),
            SensitivityParam::DeviceParam {
                device_name, param, ..
            } => {
                format!("{}.{}", device_name, param)
            }
            SensitivityParam::ModelParam {
                model_name, param, ..
            } => {
                format!("{}.{}", model_name, param)
            }
        }
    }

    /// Get the nominal value.
    pub fn value(&self) -> f64 {
        match self {
            SensitivityParam::Resistance { value, .. } => *value,
            SensitivityParam::Capacitance { value, .. } => *value,
            SensitivityParam::Inductance { value, .. } => *value,
            SensitivityParam::VoltageSource { value, .. } => *value,
            SensitivityParam::CurrentSource { value, .. } => *value,
            SensitivityParam::DeviceParam { value, .. } => *value,
            SensitivityParam::ModelParam { value, .. } => *value,
        }
    }

    /// Create a new param with a modified value.
    pub fn with_value(&self, new_value: f64) -> Self {
        match self {
            SensitivityParam::Resistance { name, .. } => SensitivityParam::Resistance {
                name: name.clone(),
                value: new_value,
            },
            SensitivityParam::Capacitance { name, .. } => SensitivityParam::Capacitance {
                name: name.clone(),
                value: new_value,
            },
            SensitivityParam::Inductance { name, .. } => SensitivityParam::Inductance {
                name: name.clone(),
                value: new_value,
            },
            SensitivityParam::VoltageSource { name, .. } => SensitivityParam::VoltageSource {
                name: name.clone(),
                value: new_value,
            },
            SensitivityParam::CurrentSource { name, .. } => SensitivityParam::CurrentSource {
                name: name.clone(),
                value: new_value,
            },
            SensitivityParam::DeviceParam {
                device_name, param, ..
            } => SensitivityParam::DeviceParam {
                device_name: device_name.clone(),
                param: param.clone(),
                value: new_value,
            },
            SensitivityParam::ModelParam {
                model_name, param, ..
            } => SensitivityParam::ModelParam {
                model_name: model_name.clone(),
                param: param.clone(),
                value: new_value,
            },
        }
    }
}

/// Output specification for sensitivity analysis.
#[derive(Debug, Clone)]
pub enum SensitivityOutput {
    /// Node voltage: V(node_idx).
    Voltage {
        /// Node index (0-based, excluding ground).
        node_idx: usize,
        /// Display name (e.g., "V(out)").
        name: String,
    },
    /// Branch current: I(branch_idx).
    Current {
        /// Branch index (0-based).
        branch_idx: usize,
        /// Display name (e.g., "I(V1)").
        name: String,
    },
    /// Voltage difference: V(node_pos) - V(node_neg).
    VoltageDiff {
        /// Positive node index.
        node_pos: usize,
        /// Negative node index.
        node_neg: usize,
        /// Display name.
        name: String,
    },
}

impl SensitivityOutput {
    /// Create a voltage output.
    pub fn voltage(node_idx: usize) -> Self {
        SensitivityOutput::Voltage {
            node_idx,
            name: format!("V({})", node_idx + 1), // 1-based for display
        }
    }

    /// Create a voltage output with custom name.
    pub fn voltage_named(node_idx: usize, name: impl Into<String>) -> Self {
        SensitivityOutput::Voltage {
            node_idx,
            name: name.into(),
        }
    }

    /// Create a current output.
    pub fn current(branch_idx: usize) -> Self {
        SensitivityOutput::Current {
            branch_idx,
            name: format!("I({})", branch_idx),
        }
    }

    /// Create a current output with custom name.
    pub fn current_named(branch_idx: usize, name: impl Into<String>) -> Self {
        SensitivityOutput::Current {
            branch_idx,
            name: name.into(),
        }
    }

    /// Get the display name.
    pub fn name(&self) -> &str {
        match self {
            SensitivityOutput::Voltage { name, .. } => name,
            SensitivityOutput::Current { name, .. } => name,
            SensitivityOutput::VoltageDiff { name, .. } => name,
        }
    }
}

/// Configuration for sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Relative perturbation for finite differences.
    /// Default is 1e-6 (one part per million).
    pub delta_ratio: f64,
    /// Minimum absolute perturbation (used when parameter is near zero).
    pub delta_min: f64,
    /// Parameters to compute sensitivity for.
    pub params: Vec<SensitivityParam>,
    /// Outputs to compute sensitivity of.
    pub outputs: Vec<SensitivityOutput>,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            delta_ratio: 1e-6,
            delta_min: 1e-12,
            params: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

impl SensitivityConfig {
    /// Create a new configuration with the given parameters and outputs.
    pub fn new(params: Vec<SensitivityParam>, outputs: Vec<SensitivityOutput>) -> Self {
        Self {
            params,
            outputs,
            ..Default::default()
        }
    }

    /// Compute the perturbation delta for a given parameter value.
    pub fn compute_delta(&self, value: f64) -> f64 {
        let abs_delta = (value.abs() * self.delta_ratio).max(self.delta_min);
        if value >= 0.0 { abs_delta } else { -abs_delta }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_name() {
        let r = SensitivityParam::Resistance {
            name: "R1".to_string(),
            value: 1000.0,
        };
        assert_eq!(r.name(), "R1");

        let dp = SensitivityParam::DeviceParam {
            device_name: "M1".to_string(),
            param: "vth0".to_string(),
            value: 0.5,
        };
        assert_eq!(dp.name(), "M1.vth0");
    }

    #[test]
    fn test_param_with_value() {
        let r = SensitivityParam::Resistance {
            name: "R1".to_string(),
            value: 1000.0,
        };
        let r2 = r.with_value(2000.0);
        assert_eq!(r2.value(), 2000.0);
        assert_eq!(r2.name(), "R1");
    }

    #[test]
    fn test_compute_delta() {
        let config = SensitivityConfig::default();

        // Normal case: delta = value * delta_ratio
        let delta = config.compute_delta(1000.0);
        assert!((delta - 1e-3).abs() < 1e-10);

        // Near zero: uses delta_min
        let delta = config.compute_delta(1e-15);
        assert!((delta - config.delta_min).abs() < 1e-20);

        // Negative value
        let delta = config.compute_delta(-1000.0);
        assert!((delta - (-1e-3)).abs() < 1e-10);
    }

    #[test]
    fn test_output_constructors() {
        let v = SensitivityOutput::voltage(0);
        assert_eq!(v.name(), "V(1)");

        let v_named = SensitivityOutput::voltage_named(0, "Vout");
        assert_eq!(v_named.name(), "Vout");

        let i = SensitivityOutput::current(0);
        assert_eq!(i.name(), "I(0)");
    }
}
