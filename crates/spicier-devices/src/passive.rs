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

/// A capacitor element.
#[derive(Debug, Clone)]
pub struct Capacitor {
    /// Device name (e.g., "C1").
    pub name: String,
    /// Positive terminal node.
    pub node_pos: NodeId,
    /// Negative terminal node.
    pub node_neg: NodeId,
    /// Capacitance value in farads.
    pub capacitance: f64,
}

impl Capacitor {
    /// Create a new capacitor.
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
        }
    }
}

impl Stamp for Capacitor {
    fn stamp(&self, _mna: &mut MnaSystem) {
        // For DC analysis, capacitor is open circuit (no stamp).
        // Transient analysis will use companion model.
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
}
