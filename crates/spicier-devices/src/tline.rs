//! Transmission line device model using lumped LC approximation.
//!
//! This module provides a lossless transmission line model approximated
//! as a cascade of N LC sections. The transmission line is specified by:
//! - Z0: Characteristic impedance (Ohms)
//! - TD: Propagation delay (seconds)
//! - NL: Number of LC sections (optional, auto-calculated if not specified)
//!
//! For a lossless line with Z0 and TD, each LC section has:
//! - L_section = Z0 × TD / N
//! - C_section = TD / (Z0 × N)
//!
//! The lumped model is accurate up to frequency f_max where N ≥ 10 × TD × f_max.

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

/// Default number of LC sections when not specified.
const DEFAULT_NUM_SECTIONS: usize = 10;

/// Minimum number of LC sections.
const MIN_SECTIONS: usize = 2;

/// A lossless transmission line element.
///
/// The transmission line connects two ports (port1 and port2), each with
/// positive and negative terminals. It is characterized by its characteristic
/// impedance Z0 and propagation delay TD.
///
/// Internally, it is modeled as a cascade of LC pi-sections for DC, AC, and
/// transient analysis.
#[derive(Debug, Clone)]
pub struct TransmissionLine {
    /// Device name (e.g., "T1").
    pub name: String,
    /// Port 1 positive terminal node.
    pub port1_pos: NodeId,
    /// Port 1 negative terminal node.
    pub port1_neg: NodeId,
    /// Port 2 positive terminal node.
    pub port2_pos: NodeId,
    /// Port 2 negative terminal node.
    pub port2_neg: NodeId,
    /// Characteristic impedance in ohms.
    pub z0: f64,
    /// Propagation delay in seconds.
    pub td: f64,
    /// Number of LC sections for the lumped approximation.
    pub num_sections: usize,
    /// Internal node IDs for the LC ladder (created during expansion).
    /// These are the nodes between LC sections.
    internal_nodes: Vec<NodeId>,
    /// Base index for the inductor branch currents.
    pub current_base_index: usize,
}

impl TransmissionLine {
    /// Create a new transmission line with automatic section count.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        port1_pos: NodeId,
        port1_neg: NodeId,
        port2_pos: NodeId,
        port2_neg: NodeId,
        z0: f64,
        td: f64,
        current_base_index: usize,
    ) -> Self {
        Self::with_sections(
            name,
            port1_pos,
            port1_neg,
            port2_pos,
            port2_neg,
            z0,
            td,
            DEFAULT_NUM_SECTIONS,
            current_base_index,
        )
    }

    /// Create a new transmission line with specified number of sections.
    #[allow(clippy::too_many_arguments)]
    pub fn with_sections(
        name: impl Into<String>,
        port1_pos: NodeId,
        port1_neg: NodeId,
        port2_pos: NodeId,
        port2_neg: NodeId,
        z0: f64,
        td: f64,
        num_sections: usize,
        current_base_index: usize,
    ) -> Self {
        let num_sections = num_sections.max(MIN_SECTIONS);
        Self {
            name: name.into(),
            port1_pos,
            port1_neg,
            port2_pos,
            port2_neg,
            z0,
            td,
            num_sections,
            internal_nodes: Vec::new(),
            current_base_index,
        }
    }

    /// Set the internal nodes for the LC ladder.
    ///
    /// This must be called after the transmission line is created but before
    /// stamping. The internal nodes connect the cascaded LC sections.
    pub fn set_internal_nodes(&mut self, nodes: Vec<NodeId>) {
        self.internal_nodes = nodes;
    }

    /// Get the inductance per section (L = Z0 * TD / N).
    pub fn inductance_per_section(&self) -> f64 {
        self.z0 * self.td / self.num_sections as f64
    }

    /// Get the capacitance per section (C = TD / (Z0 * N)).
    pub fn capacitance_per_section(&self) -> f64 {
        self.td / (self.z0 * self.num_sections as f64)
    }

    /// Get the number of internal nodes needed (N-1 for N sections).
    pub fn num_internal_nodes(&self) -> usize {
        self.num_sections - 1
    }

    /// Get the number of inductor current variables needed (one per section).
    pub fn num_inductors(&self) -> usize {
        self.num_sections
    }
}

impl Stamp for TransmissionLine {
    fn stamp(&self, mna: &mut MnaSystem) {
        // For DC analysis, inductors are short circuits (0V voltage sources)
        // and capacitors are open circuits (no contribution).
        //
        // This means the transmission line acts as a short circuit at DC,
        // which is the correct behavior for a lossless line.
        //
        // We stamp each inductor as a 0V voltage source.

        let l_value = self.inductance_per_section();
        let _c_value = self.capacitance_per_section();

        // Get node indices for ports (using positive terminals for the series path)
        let port1_idx = node_to_index(self.port1_pos);
        let port2_idx = node_to_index(self.port2_pos);
        let neg_idx = node_to_index(self.port1_neg); // Assume port1_neg == port2_neg for simplicity

        // Build the chain of internal nodes
        // Section i connects node[i-1] to node[i] via inductor
        // with shunt capacitor at node[i]
        //
        // Port1+ --L1-- int1 --L2-- int2 --...-- intN-1 --LN-- Port2+
        //              |          |                     |
        //              C1         C2                   CN
        //              |          |                     |
        // Port1- ------+----------+-----...-------------+------ Port2-

        // For DC: inductors are shorts, capacitors are open
        // So we just stamp the inductor DC model (0V voltage source)

        for i in 0..self.num_sections {
            let branch_idx = self.current_base_index + i;

            // Determine the nodes this inductor connects
            let (left_node, right_node) = if i == 0 {
                // First section: port1_pos to internal_node[0] (or port2 if N=1)
                if self.num_sections == 1 {
                    (port1_idx, port2_idx)
                } else {
                    let int_idx = node_to_index(self.internal_nodes[0]);
                    (port1_idx, int_idx)
                }
            } else if i == self.num_sections - 1 {
                // Last section: internal_node[N-2] to port2_pos
                let int_idx = node_to_index(self.internal_nodes[i - 1]);
                (int_idx, port2_idx)
            } else {
                // Middle sections: internal_node[i-1] to internal_node[i]
                let left_int_idx = node_to_index(self.internal_nodes[i - 1]);
                let right_int_idx = node_to_index(self.internal_nodes[i]);
                (left_int_idx, right_int_idx)
            };

            // Stamp the inductor as a 0V voltage source for DC
            // v_left - v_right = 0 (short circuit)
            mna.stamp_voltage_source(left_node, right_node, branch_idx, 0.0);

            // Capacitors don't contribute at DC (open circuit), so no stamp needed

            // Note: For actual transient/AC analysis, we would need to stamp
            // the dynamic companion models, but that's handled by transient_info()
            // and ac_info() returning the component values.
            let _ = (l_value, neg_idx);
        }
    }
}

impl Element for TransmissionLine {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        let mut nodes = vec![
            self.port1_pos,
            self.port1_neg,
            self.port2_pos,
            self.port2_neg,
        ];
        nodes.extend(self.internal_nodes.iter().copied());
        nodes
    }

    fn num_current_vars(&self) -> usize {
        // One current variable per inductor (one per section)
        self.num_sections
    }
}

impl Stamper for TransmissionLine {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn num_current_vars(&self) -> usize {
        self.num_sections
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::TransmissionLine {
            port1_pos: node_to_index(self.port1_pos),
            port1_neg: node_to_index(self.port1_neg),
            port2_pos: node_to_index(self.port2_pos),
            port2_neg: node_to_index(self.port2_neg),
            z0: self.z0,
            td: self.td,
            num_sections: self.num_sections,
            internal_nodes: self
                .internal_nodes
                .iter()
                .map(|n| node_to_index(*n))
                .collect(),
            current_base_index: self.current_base_index,
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::TransmissionLine {
            port1_pos: node_to_index(self.port1_pos),
            port1_neg: node_to_index(self.port1_neg),
            port2_pos: node_to_index(self.port2_pos),
            port2_neg: node_to_index(self.port2_neg),
            z0: self.z0,
            td: self.td,
            num_sections: self.num_sections,
            internal_nodes: self
                .internal_nodes
                .iter()
                .map(|n| node_to_index(*n))
                .collect(),
            current_base_index: self.current_base_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transmission_line_creation() {
        let tl = TransmissionLine::new(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            0,
        );

        assert_eq!(tl.name, "T1");
        assert_eq!(tl.z0, 50.0);
        assert_eq!(tl.td, 1e-9);
        assert_eq!(tl.num_sections, DEFAULT_NUM_SECTIONS);
    }

    #[test]
    fn test_transmission_line_with_sections() {
        let tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            20,
            0,
        );

        assert_eq!(tl.num_sections, 20);
    }

    #[test]
    fn test_lc_values() {
        let tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0, // Z0 = 50 ohms
            1e-9, // TD = 1ns
            10,   // 10 sections
            0,
        );

        // L_section = Z0 * TD / N = 50 * 1e-9 / 10 = 5e-9 H = 5nH
        let l_expected = 5e-9;
        assert!((tl.inductance_per_section() - l_expected).abs() < 1e-15);

        // C_section = TD / (Z0 * N) = 1e-9 / (50 * 10) = 2e-12 F = 2pF
        let c_expected = 2e-12;
        assert!((tl.capacitance_per_section() - c_expected).abs() < 1e-18);
    }

    #[test]
    fn test_min_sections() {
        let tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            1, // Request 1 section, should be clamped to MIN_SECTIONS
            0,
        );

        assert_eq!(tl.num_sections, MIN_SECTIONS);
    }

    #[test]
    fn test_internal_nodes() {
        let tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            5, // 5 sections needs 4 internal nodes
            0,
        );

        assert_eq!(tl.num_internal_nodes(), 4);
        assert_eq!(tl.num_inductors(), 5);
    }

    #[test]
    fn test_num_current_vars() {
        let tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            8,
            0,
        );

        // Should have one current variable per section (for each inductor)
        assert_eq!(Element::num_current_vars(&tl), 8);
    }

    #[test]
    fn test_nodes_without_internal() {
        let tl = TransmissionLine::new(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            0,
        );

        // Without internal nodes set, should only return port nodes
        let nodes = tl.nodes();
        assert_eq!(nodes.len(), 4);
        assert_eq!(nodes[0], NodeId::new(1));
        assert_eq!(nodes[1], NodeId::GROUND);
        assert_eq!(nodes[2], NodeId::new(2));
        assert_eq!(nodes[3], NodeId::GROUND);
    }

    #[test]
    fn test_nodes_with_internal() {
        let mut tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            3, // 3 sections = 2 internal nodes
            0,
        );

        tl.set_internal_nodes(vec![NodeId::new(10), NodeId::new(11)]);

        let nodes = tl.nodes();
        assert_eq!(nodes.len(), 6); // 4 port nodes + 2 internal
        assert_eq!(nodes[4], NodeId::new(10));
        assert_eq!(nodes[5], NodeId::new(11));
    }

    #[test]
    fn test_dc_stamp() {
        // Test that the DC stamp creates proper voltage source stamps
        // for the inductors (short circuits at DC)
        let mut tl = TransmissionLine::with_sections(
            "T1",
            NodeId::new(1),
            NodeId::GROUND,
            NodeId::new(2),
            NodeId::GROUND,
            50.0,
            1e-9,
            2, // 2 sections = 1 internal node
            0,
        );

        tl.set_internal_nodes(vec![NodeId::new(3)]);

        // Create an MNA system with 3 nodes and 2 current variables
        let mut mna = MnaSystem::new(3, 2);
        Stamp::stamp(&tl, &mut mna);

        // The matrix should have voltage source stamps for 2 inductors
        let matrix = mna.to_dense_matrix();

        // First inductor: node 1 to node 3 (internal)
        // Branch 0: stamps at rows/cols for node 0 (node 1), node 2 (node 3), branch 3 (num_nodes + 0)
        assert_eq!(matrix[(0, 3)], 1.0); // node 1 row, branch 0 col
        assert_eq!(matrix[(3, 0)], 1.0); // branch 0 row, node 1 col
        assert_eq!(matrix[(2, 3)], -1.0); // node 3 row, branch 0 col
        assert_eq!(matrix[(3, 2)], -1.0); // branch 0 row, node 3 col

        // Second inductor: node 3 to node 2
        // Branch 1: stamps at rows/cols for node 2 (node 3), node 1 (node 2), branch 4 (num_nodes + 1)
        assert_eq!(matrix[(2, 4)], 1.0); // node 3 row, branch 1 col
        assert_eq!(matrix[(4, 2)], 1.0); // branch 1 row, node 3 col
        assert_eq!(matrix[(1, 4)], -1.0); // node 2 row, branch 1 col
        assert_eq!(matrix[(4, 1)], -1.0); // branch 1 row, node 2 col
    }
}
