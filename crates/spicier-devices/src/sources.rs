//! Independent source models: Voltage and Current sources.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::AcDeviceInfo;
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

/// An independent voltage source.
#[derive(Debug, Clone)]
pub struct VoltageSource {
    /// Device name (e.g., "V1").
    pub name: String,
    /// Positive terminal node.
    pub node_pos: NodeId,
    /// Negative terminal node.
    pub node_neg: NodeId,
    /// DC voltage value in volts.
    pub voltage: f64,
    /// Index of the current variable for this source.
    pub current_index: usize,
}

impl VoltageSource {
    /// Create a new voltage source.
    pub fn new(
        name: impl Into<String>,
        node_pos: NodeId,
        node_neg: NodeId,
        voltage: f64,
        current_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            voltage,
            current_index,
        }
    }
}

impl Stamp for VoltageSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_voltage_source(i, j, self.current_index, self.voltage);
    }
}

impl Element for VoltageSource {
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

impl Stamper for VoltageSource {
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
        AcDeviceInfo::VoltageSource {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            branch_idx: self.current_index,
            ac_mag: 1.0, // Default AC magnitude
        }
    }

    fn is_source(&self) -> bool {
        true
    }

    fn stamp_nonlinear_scaled(
        &self,
        mna: &mut MnaSystem,
        _solution: &DVector<f64>,
        source_factor: f64,
    ) {
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_voltage_source(i, j, self.current_index, self.voltage * source_factor);
    }
}

/// An independent current source.
#[derive(Debug, Clone)]
pub struct CurrentSource {
    /// Device name (e.g., "I1").
    pub name: String,
    /// Positive terminal node (current flows out).
    pub node_pos: NodeId,
    /// Negative terminal node (current flows in).
    pub node_neg: NodeId,
    /// DC current value in amperes.
    pub current: f64,
}

impl CurrentSource {
    /// Create a new current source.
    pub fn new(name: impl Into<String>, node_pos: NodeId, node_neg: NodeId, current: f64) -> Self {
        Self {
            name: name.into(),
            node_pos,
            node_neg,
            current,
        }
    }
}

impl Stamp for CurrentSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Current flows from node_pos to node_neg (out of pos, into neg)
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_current_source(i, j, self.current);
    }
}

impl Element for CurrentSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }
}

impl Stamper for CurrentSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::CurrentSource {
            node_pos: node_to_index(self.node_pos),
            node_neg: node_to_index(self.node_neg),
            ac_mag: 0.0, // No AC by default
        }
    }

    fn is_source(&self) -> bool {
        true
    }

    fn stamp_nonlinear_scaled(
        &self,
        mna: &mut MnaSystem,
        _solution: &DVector<f64>,
        source_factor: f64,
    ) {
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_current_source(i, j, self.current * source_factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voltage_source_stamp() {
        let mut mna = MnaSystem::new(2, 1);
        let v = VoltageSource::new("V1", NodeId::new(1), NodeId::GROUND, 5.0, 0);

        Stamp::stamp(&v, &mut mna);
        let matrix = mna.to_dense_matrix();

        // Check coupling
        assert_eq!(matrix[(0, 2)], 1.0);
        assert_eq!(matrix[(2, 0)], 1.0);
        // Check voltage value
        assert_eq!(mna.rhs()[2], 5.0);
    }

    #[test]
    fn test_current_source_stamp() {
        let mut mna = MnaSystem::new(2, 0);
        let i = CurrentSource::new("I1", NodeId::GROUND, NodeId::new(1), 0.001);

        Stamp::stamp(&i, &mut mna);

        // 1mA into node 1
        assert_eq!(mna.rhs()[0], 0.001);
    }
}
