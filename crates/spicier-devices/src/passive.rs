//! Passive device models: Resistor, Capacitor, Inductor.

use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// Convert a NodeId to an MNA matrix index (None for ground).
fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

/// A resistor element.
#[derive(Debug, Clone)]
pub struct Resistor {
    /// Device name (e.g., "R1").
    pub name: String,
    /// Positive terminal node.
    pub node_pos: NodeId,
    /// Negative terminal node.
    pub node_neg: NodeId,
    /// Resistance value in ohms.
    pub resistance: f64,
}

impl Resistor {
    /// Create a new resistor.
    pub fn new(
        name: impl Into<String>,
        node_pos: NodeId,
        node_neg: NodeId,
        resistance: f64,
    ) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            resistance,
        }
    }

    /// Get the conductance (1/R).
    pub fn conductance(&self) -> f64 {
        1.0 / self.resistance
    }
}

impl Stamp for Resistor {
    fn stamp(&self, mna: &mut MnaSystem) {
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_conductance(i, j, self.conductance());
    }
}

impl Element for Resistor {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }
}

impl Stamper for Resistor {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Resistor {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            conductance: self.conductance(),
        }
    }
}

/// Capacitor model parameters for `.MODEL` definitions.
///
/// Supports voltage-dependent capacitance, temperature coefficients,
/// geometry scaling, and parasitic resistance. Used for SKY130 MIM
/// capacitors and similar PDK models.
///
/// Capacitance evaluation:
/// ```text
/// C0 = c_per_area * W * L   (if geometry specified)
/// C0 = c_base               (if no geometry)
/// C(V,T) = C0 * (1 + VC1*V + VC2*V²) * (1 + TC1*ΔT + TC2*ΔT²)
/// ```
#[derive(Debug, Clone)]
pub struct CapacitorParams {
    /// Base capacitance per unit area (F/m²). Used with W and L.
    pub c_per_area: f64,
    /// Fixed base capacitance (F). Used when no geometry is specified.
    pub c_base: f64,
    /// Linear voltage coefficient (1/V).
    pub vc1: f64,
    /// Quadratic voltage coefficient (1/V²).
    pub vc2: f64,
    /// Linear temperature coefficient (1/°C).
    pub tc1: f64,
    /// Quadratic temperature coefficient (1/°C²).
    pub tc2: f64,
    /// Series resistance (Ω). 0 means no series resistance.
    pub rs: f64,
    /// Parallel (leakage) resistance (Ω). 0 means no leakage.
    pub rp: f64,
    /// Width (m).
    pub w: f64,
    /// Length (m).
    pub l: f64,
    /// Nominal temperature (K).
    pub tnom: f64,
}

impl Default for CapacitorParams {
    fn default() -> Self {
        Self {
            c_per_area: 0.0,
            c_base: 0.0,
            vc1: 0.0,
            vc2: 0.0,
            tc1: 0.0,
            tc2: 0.0,
            rs: 0.0,
            rp: 0.0,
            w: 0.0,
            l: 0.0,
            tnom: 300.15, // 27°C in K
        }
    }
}

impl CapacitorParams {
    /// Compute the base capacitance C0 from geometry or fixed value.
    pub fn base_capacitance(&self) -> f64 {
        if self.w > 0.0 && self.l > 0.0 && self.c_per_area > 0.0 {
            self.c_per_area * self.w * self.l
        } else {
            self.c_base
        }
    }

    /// Evaluate capacitance at given voltage and temperature.
    ///
    /// `v` is the voltage across the capacitor (V).
    /// `temp` is the operating temperature (K).
    pub fn capacitance_at(&self, v: f64, temp: f64) -> f64 {
        let c0 = self.base_capacitance();
        let vc_factor = 1.0 + self.vc1 * v + self.vc2 * v * v;
        let dt = temp - self.tnom;
        let tc_factor = 1.0 + self.tc1 * dt + self.tc2 * dt * dt;
        c0 * vc_factor * tc_factor
    }
}

/// A capacitor element.
///
/// Supports both simple fixed-value capacitors and model-based capacitors
/// with voltage/temperature-dependent capacitance and parasitic resistance.
#[derive(Debug, Clone)]
pub struct Capacitor {
    /// Device name (e.g., "C1").
    pub name: String,
    /// Positive terminal node.
    pub node_pos: NodeId,
    /// Negative terminal node.
    pub node_neg: NodeId,
    /// Capacitance value in farads (for fixed-value capacitors).
    pub capacitance: f64,
    /// Optional model parameters (for model-based capacitors).
    pub params: Option<CapacitorParams>,
}

impl Capacitor {
    /// Create a new fixed-value capacitor.
    pub fn new(
        name: impl Into<String>,
        node_pos: NodeId,
        node_neg: NodeId,
        capacitance: f64,
    ) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            capacitance,
            params: None,
        }
    }

    /// Create a capacitor with model parameters.
    pub fn with_params(
        name: impl Into<String>,
        node_pos: NodeId,
        node_neg: NodeId,
        params: CapacitorParams,
    ) -> Self {
        let capacitance = params.base_capacitance();
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            capacitance,
            params: Some(params),
        }
    }

    /// Get the effective capacitance at given voltage and temperature.
    ///
    /// For fixed-value capacitors, returns the constant capacitance.
    /// For model-based capacitors, evaluates C(V,T).
    pub fn effective_capacitance(&self, voltage: f64, temp: f64) -> f64 {
        match &self.params {
            Some(p) => p.capacitance_at(voltage, temp),
            None => self.capacitance,
        }
    }
}

impl Stamp for Capacitor {
    fn stamp(&self, mna: &mut MnaSystem) {
        // For DC analysis, capacitor is open circuit (no stamp).
        // Transient analysis will use companion model.

        // Stamp parallel leakage resistance if present.
        if let Some(ref p) = self.params {
            if p.rp > 0.0 {
                let i = node_to_index(self.node_pos);
                let j = node_to_index(self.node_neg);
                mna.stamp_conductance(i, j, 1.0 / p.rp);
            }
        }
    }
}

impl Element for Capacitor {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }
}

impl Stamper for Capacitor {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Capacitor {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            capacitance: self.capacitance,
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::Capacitor {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            capacitance: self.capacitance,
        }
    }
}

/// An inductor element.
#[derive(Debug, Clone)]
pub struct Inductor {
    /// Device name (e.g., "L1").
    pub name: String,
    /// Positive terminal node.
    pub node_pos: NodeId,
    /// Negative terminal node.
    pub node_neg: NodeId,
    /// Inductance value in henries.
    pub inductance: f64,
    /// Index of the current variable for this inductor.
    pub current_index: usize,
}

impl Inductor {
    /// Create a new inductor.
    pub fn new(
        name: impl Into<String>,
        node_pos: NodeId,
        node_neg: NodeId,
        inductance: f64,
        current_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            inductance,
            current_index,
        }
    }
}

impl Stamp for Inductor {
    fn stamp(&self, mna: &mut MnaSystem) {
        // For DC analysis, inductor is short circuit (0V voltage source).
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_voltage_source(i, j, self.current_index, 0.0);
    }
}

impl Element for Inductor {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }

    fn num_current_vars(&self) -> usize {
        1
    }
}

impl Stamper for Inductor {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn num_current_vars(&self) -> usize {
        1
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn branch_index(&self) -> Option<usize> {
        Some(self.current_index)
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Inductor {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            inductance: self.inductance,
            branch_idx: self.current_index,
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::Inductor {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            inductance: self.inductance,
            branch_index: self.current_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resistor_stamp() {
        let mut mna = MnaSystem::new(2, 0);
        let r = Resistor::new("R1", NodeId::new(1), NodeId::new(2), 1000.0);

        Stamp::stamp(&r, &mut mna);
        let matrix = mna.to_dense_matrix();

        let g = 0.001; // 1/1000
        assert!((matrix[(0, 0)] - g).abs() < 1e-10);
        assert!((matrix[(1, 1)] - g).abs() < 1e-10);
        assert!((matrix[(0, 1)] + g).abs() < 1e-10);
        assert!((matrix[(1, 0)] + g).abs() < 1e-10);
    }

    #[test]
    fn test_resistor_to_ground() {
        let mut mna = MnaSystem::new(1, 0);
        let r = Resistor::new("R1", NodeId::new(1), NodeId::GROUND, 100.0);

        Stamp::stamp(&r, &mut mna);
        let matrix = mna.to_dense_matrix();

        let g = 0.01;
        assert!((matrix[(0, 0)] - g).abs() < 1e-10);
    }

    #[test]
    fn test_inductor_dc_stamp() {
        let mut mna = MnaSystem::new(2, 1);
        let l = Inductor::new("L1", NodeId::new(1), NodeId::new(2), 1e-3, 0);

        Stamp::stamp(&l, &mut mna);
        let matrix = mna.to_dense_matrix();

        // Should stamp as 0V voltage source
        assert_eq!(matrix[(0, 2)], 1.0);
        assert_eq!(matrix[(2, 0)], 1.0);
        assert_eq!(matrix[(1, 2)], -1.0);
        assert_eq!(matrix[(2, 1)], -1.0);
        assert_eq!(mna.rhs()[2], 0.0);
    }

    // CapacitorParams tests

    #[test]
    fn test_capacitor_params_default() {
        let cp = CapacitorParams::default();
        assert_eq!(cp.c_per_area, 0.0);
        assert_eq!(cp.c_base, 0.0);
        assert_eq!(cp.vc1, 0.0);
        assert_eq!(cp.vc2, 0.0);
        assert_eq!(cp.tc1, 0.0);
        assert_eq!(cp.tc2, 0.0);
        assert_eq!(cp.rs, 0.0);
        assert_eq!(cp.rp, 0.0);
        assert!((cp.tnom - 300.15).abs() < 0.01);
    }

    #[test]
    fn test_capacitor_params_base_capacitance_geometry() {
        let cp = CapacitorParams {
            c_per_area: 2e-3, // 2 fF/um^2 = 2e-3 F/m^2
            w: 10e-6,
            l: 10e-6,
            ..Default::default()
        };
        // C0 = 2e-3 * 10e-6 * 10e-6 = 200e-15 = 200 fF
        assert!((cp.base_capacitance() - 200e-15).abs() < 1e-18);
    }

    #[test]
    fn test_capacitor_params_base_capacitance_fixed() {
        let cp = CapacitorParams {
            c_base: 10e-12,
            ..Default::default()
        };
        assert!((cp.base_capacitance() - 10e-12).abs() < 1e-18);
    }

    #[test]
    fn test_capacitor_params_voltage_dependence() {
        let cp = CapacitorParams {
            c_base: 10e-12,
            vc1: 0.01,   // 1% per volt
            vc2: 0.001,  // 0.1% per volt^2
            ..Default::default()
        };

        // At 0V: C = 10pF * (1 + 0) = 10pF
        let c0 = cp.capacitance_at(0.0, 300.15);
        assert!((c0 - 10e-12).abs() < 1e-18);

        // At 1V: C = 10pF * (1 + 0.01 + 0.001) = 10pF * 1.011 = 10.11pF
        let c1 = cp.capacitance_at(1.0, 300.15);
        assert!((c1 - 10.11e-12).abs() < 1e-16);

        // At 2V: C = 10pF * (1 + 0.02 + 0.004) = 10pF * 1.024 = 10.24pF
        let c2 = cp.capacitance_at(2.0, 300.15);
        assert!((c2 - 10.24e-12).abs() < 1e-16);
    }

    #[test]
    fn test_capacitor_params_temperature_dependence() {
        let cp = CapacitorParams {
            c_base: 10e-12,
            tc1: 1e-4,    // 100 ppm/°C
            tc2: 1e-6,    // 1 ppm/°C^2
            tnom: 300.15,  // 27°C
            ..Default::default()
        };

        // At nominal temp: C = 10pF (no change)
        let c_nom = cp.capacitance_at(0.0, 300.15);
        assert!((c_nom - 10e-12).abs() < 1e-18);

        // At +100°C (400.15K): dT=100, factor = 1 + 0.01 + 0.01 = 1.02
        let c_hot = cp.capacitance_at(0.0, 400.15);
        let expected = 10e-12 * (1.0 + 1e-4 * 100.0 + 1e-6 * 10000.0);
        assert!((c_hot - expected).abs() < 1e-18);
    }

    #[test]
    fn test_capacitor_params_combined_v_and_t() {
        let cp = CapacitorParams {
            c_per_area: 2e-3,
            w: 10e-6,
            l: 10e-6,
            vc1: 0.01,
            tc1: 1e-4,
            tnom: 300.15,
            ..Default::default()
        };

        // C0 = 200fF, V=1V, T=350.15K (dT=50)
        // vc_factor = 1 + 0.01*1 = 1.01
        // tc_factor = 1 + 1e-4*50 = 1.005
        // C = 200fF * 1.01 * 1.005 = 200fF * 1.01505 = 203.01fF
        let c = cp.capacitance_at(1.0, 350.15);
        let expected = 200e-15 * 1.01 * 1.005;
        assert!((c - expected).abs() < 1e-20);
    }

    #[test]
    fn test_capacitor_fixed_value() {
        let c = Capacitor::new("C1", NodeId::new(1), NodeId::GROUND, 10e-12);
        assert_eq!(c.capacitance, 10e-12);
        assert!(c.params.is_none());

        // effective_capacitance ignores voltage/temp for fixed
        assert_eq!(c.effective_capacitance(1.0, 400.0), 10e-12);
    }

    #[test]
    fn test_capacitor_with_params() {
        let cp = CapacitorParams {
            c_base: 10e-12,
            vc1: 0.01,
            ..Default::default()
        };
        let c = Capacitor::with_params("C1", NodeId::new(1), NodeId::GROUND, cp);

        // base capacitance stored
        assert!((c.capacitance - 10e-12).abs() < 1e-18);
        assert!(c.params.is_some());

        // effective_capacitance uses voltage
        let c_at_1v = c.effective_capacitance(1.0, 300.15);
        assert!((c_at_1v - 10.1e-12).abs() < 1e-16);
    }

    #[test]
    fn test_capacitor_dc_stamp_no_params() {
        // Fixed capacitor: DC stamp is no-op (open circuit)
        let mut mna = MnaSystem::new(2, 0);
        let c = Capacitor::new("C1", NodeId::new(1), NodeId::new(2), 10e-12);

        Stamp::stamp(&c, &mut mna);
        let matrix = mna.to_dense_matrix();

        assert_eq!(matrix[(0, 0)], 0.0);
        assert_eq!(matrix[(1, 1)], 0.0);
    }

    #[test]
    fn test_capacitor_dc_stamp_with_leakage() {
        // Model capacitor with parallel leakage resistance
        let cp = CapacitorParams {
            c_base: 10e-12,
            rp: 1e9, // 1 GOhm leakage
            ..Default::default()
        };
        let c = Capacitor::with_params("C1", NodeId::new(1), NodeId::new(2), cp);

        let mut mna = MnaSystem::new(2, 0);
        Stamp::stamp(&c, &mut mna);
        let matrix = mna.to_dense_matrix();

        let g = 1.0 / 1e9;
        assert!((matrix[(0, 0)] - g).abs() < 1e-20);
        assert!((matrix[(1, 1)] - g).abs() < 1e-20);
        assert!((matrix[(0, 1)] + g).abs() < 1e-20);
        assert!((matrix[(1, 0)] + g).abs() < 1e-20);
    }

    #[test]
    fn test_capacitor_backward_compat() {
        // Ensure the basic API hasn't changed
        let c = Capacitor::new("C1", NodeId::new(1), NodeId::GROUND, 1e-6);
        assert_eq!(c.name, "C1");
        assert_eq!(c.node_pos, NodeId::new(1));
        assert_eq!(c.node_neg, NodeId::GROUND);
        assert_eq!(c.capacitance, 1e-6);

        let nodes = Element::nodes(&c);
        assert_eq!(nodes.len(), 2);
    }
}
