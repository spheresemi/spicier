//! Diode device model using the Shockley equation.

use spicier_core::mna::MnaSystem;
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// Diode model parameters.
#[derive(Debug, Clone)]
pub struct DiodeParams {
    /// Saturation current (A). Default: 1e-14.
    pub is: f64,
    /// Emission coefficient. Default: 1.0.
    pub n: f64,
    /// Series resistance (ohms). Default: 0.0.
    pub rs: f64,
    /// Junction capacitance (F). Default: 0.0.
    pub cj0: f64,
    /// Junction potential (V). Default: 1.0.
    pub vj: f64,
    /// Breakdown voltage (V). Default: f64::INFINITY.
    pub bv: f64,
}

impl Default for DiodeParams {
    fn default() -> Self {
        Self {
            is: 1e-14,
            n: 1.0,
            rs: 0.0,
            cj0: 0.0,
            vj: 1.0,
            bv: f64::INFINITY,
        }
    }
}

/// Thermal voltage at a given temperature.
pub fn thermal_voltage(temp_k: f64) -> f64 {
    const K_BOLTZMANN: f64 = 1.380649e-23;
    const Q_ELECTRON: f64 = 1.602176634e-19;
    K_BOLTZMANN * temp_k / Q_ELECTRON
}

/// A diode element.
#[derive(Debug, Clone)]
pub struct Diode {
    /// Device name (e.g., "D1").
    pub name: String,
    /// Anode node.
    pub node_pos: NodeId,
    /// Cathode node.
    pub node_neg: NodeId,
    /// Model parameters.
    pub params: DiodeParams,
}

impl Diode {
    /// Create a new diode with default parameters.
    pub fn new(name: impl Into<String>, node_pos: NodeId, node_neg: NodeId) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            params: DiodeParams::default(),
        }
    }

    /// Create a new diode with specified parameters.
    pub fn with_params(
        name: impl Into<String>,
        node_pos: NodeId,
        node_neg: NodeId,
        params: DiodeParams,
    ) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            params,
        }
    }

    /// Evaluate diode current and conductance at a given voltage.
    ///
    /// Returns (current, conductance) where:
    /// - current = Is * (exp(Vd / (n * Vt)) - 1)
    /// - conductance = dI/dV = Is / (n * Vt) * exp(Vd / (n * Vt))
    pub fn evaluate(&self, vd: f64) -> (f64, f64) {
        let vt = thermal_voltage(300.15); // Room temperature (27°C)
        let nvt = self.params.n * vt;

        // Limit voltage to prevent overflow in exp()
        let vd_limited = limit_voltage(vd, nvt);

        let exp_term = (vd_limited / nvt).exp();
        let id = self.params.is * (exp_term - 1.0);
        let gd = self.params.is * exp_term / nvt;

        // Ensure minimum conductance for numerical stability
        let gd = gd.max(1e-12);

        (id, gd)
    }

    /// Stamp the linearized diode model into the MNA system.
    ///
    /// At operating point Vd0, the diode is represented as:
    /// - A conductance Gd = dI/dV(Vd0)
    /// - A current source Ieq = Id(Vd0) - Gd * Vd0
    pub fn stamp_nonlinear(&self, mna: &mut MnaSystem, vd: f64) {
        let (id, gd) = self.evaluate(vd);
        let ieq = id - gd * vd;

        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);

        // Stamp conductance
        mna.stamp_conductance(i, j, gd);

        // Stamp equivalent current source
        mna.stamp_current_source(i, j, ieq);
    }
}

/// Voltage limiting to prevent numerical overflow.
///
/// Limits the step in diode voltage to prevent exp() overflow
/// while still allowing convergence.
fn limit_voltage(vd: f64, nvt: f64) -> f64 {
    let vcrit = nvt * (nvt / (std::f64::consts::SQRT_2 * 1e-14)).ln();

    if vd > vcrit {
        // Limit using log compression
        let arg = (vd - vcrit) / nvt;
        vcrit + nvt * (1.0 + arg.ln_1p())
    } else {
        vd
    }
}

fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

impl Stamp for Diode {
    fn stamp(&self, mna: &mut MnaSystem) {
        // For initial DC guess, use Gmin shunt
        let gmin = 1e-12;
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_conductance(i, j, gmin);
    }
}

impl Element for Diode {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }
}

impl Stamper for Diode {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diode_forward_bias() {
        let d = Diode::new("D1", NodeId::new(1), NodeId::GROUND);

        // At 0.7V forward bias, current should be significant
        let (id, gd) = d.evaluate(0.7);
        assert!(id > 0.0, "Forward current should be positive: {}", id);
        assert!(gd > 0.0, "Forward conductance should be positive: {}", gd);
    }

    #[test]
    fn test_diode_reverse_bias() {
        let d = Diode::new("D1", NodeId::new(1), NodeId::GROUND);

        // At -1V reverse bias, current should be very small (≈ -Is)
        let (id, _gd) = d.evaluate(-1.0);
        assert!(id < 0.0, "Reverse current should be negative: {}", id);
        assert!(id.abs() < 1e-12, "Reverse current should be ≈ -Is: {}", id);
    }

    #[test]
    fn test_diode_zero_bias() {
        let d = Diode::new("D1", NodeId::new(1), NodeId::GROUND);

        let (id, _gd) = d.evaluate(0.0);
        assert!(id.abs() < 1e-15, "Zero-bias current should be ≈ 0: {}", id);
    }

    #[test]
    fn test_thermal_voltage() {
        let vt = thermal_voltage(300.15);
        // At room temperature, Vt ≈ 25.85 mV
        assert!(
            (vt - 0.02585).abs() < 0.001,
            "Vt = {} (expected ≈ 0.02585)",
            vt
        );
    }

    #[test]
    fn test_voltage_limiting() {
        let nvt = 0.02585;
        // Very large voltage should be limited but not explode
        let limited = limit_voltage(100.0, nvt);
        assert!(limited < 100.0, "Should be limited: {}", limited);
        assert!(limited > 0.0, "Should be positive: {}", limited);
    }
}
