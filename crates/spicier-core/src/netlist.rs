//! Netlist: A complete circuit description ready for simulation.

use crate::mna::MnaSystem;
use crate::node::NodeId;

/// A boxed device that can stamp into an MNA matrix.
pub type BoxedStamper = Box<dyn Stamper>;

/// Information about a device for AC analysis stamping.
#[derive(Debug, Clone)]
pub enum AcDeviceInfo {
    /// Resistor: stamp real conductance G.
    Resistor {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        conductance: f64,
    },
    /// Capacitor: stamp jωC admittance.
    Capacitor {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        capacitance: f64,
    },
    /// Inductor: stamp jωL impedance with branch current.
    Inductor {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        inductance: f64,
        branch_idx: usize,
    },
    /// Voltage source: stamp with AC stimulus.
    VoltageSource {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        branch_idx: usize,
        ac_mag: f64,
    },
    /// Current source: stamp AC current.
    CurrentSource {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        ac_mag: f64,
    },
    /// Unknown or nonlinear device (skip in AC).
    None,
}

/// Trait for devices that can stamp into an MNA matrix.
/// This is a local trait to avoid circular dependencies.
pub trait Stamper: std::fmt::Debug + Send + Sync {
    /// Stamp this device into the MNA system.
    fn stamp(&self, mna: &mut MnaSystem);

    /// Number of current variables this device requires.
    fn num_current_vars(&self) -> usize {
        0
    }

    /// Device name (e.g., "R1", "V1").
    fn device_name(&self) -> &str {
        ""
    }

    /// Branch current variable index, if this device has one.
    fn branch_index(&self) -> Option<usize> {
        None
    }

    /// Provide AC analysis information for this device.
    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::None
    }
}

/// A complete netlist ready for simulation.
#[derive(Debug, Default)]
pub struct Netlist {
    /// Circuit title.
    title: Option<String>,
    /// Highest node number used.
    max_node: u32,
    /// All devices in the netlist.
    devices: Vec<BoxedStamper>,
    /// Total number of current variables (voltage sources + inductors).
    num_current_vars: usize,
}

impl Netlist {
    /// Create a new empty netlist.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a netlist with a title.
    pub fn with_title(title: impl Into<String>) -> Self {
        Self {
            title: Some(title.into()),
            ..Default::default()
        }
    }

    /// Get the netlist title.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Register a node, updating max_node if necessary.
    pub fn register_node(&mut self, node: NodeId) {
        if !node.is_ground() && node.as_u32() > self.max_node {
            self.max_node = node.as_u32();
        }
    }

    /// Add a device to the netlist.
    pub fn add_device(&mut self, device: impl Stamper + 'static) {
        self.num_current_vars += device.num_current_vars();
        self.devices.push(Box::new(device));
    }

    /// Get the number of nodes (excluding ground).
    pub fn num_nodes(&self) -> usize {
        self.max_node as usize
    }

    /// Get the number of current variables.
    pub fn num_current_vars(&self) -> usize {
        self.num_current_vars
    }

    /// Get the next available current variable index.
    pub fn next_current_index(&self) -> usize {
        self.num_current_vars
    }

    /// Assemble the MNA system from all devices.
    pub fn assemble_mna(&self) -> MnaSystem {
        let mut mna = MnaSystem::new(self.num_nodes(), self.num_current_vars);

        for device in &self.devices {
            device.stamp(&mut mna);
        }

        mna
    }

    /// Stamp all devices into a pre-existing MNA system.
    pub fn stamp_into(&self, mna: &mut MnaSystem) {
        for device in &self.devices {
            device.stamp(mna);
        }
    }

    /// Find the branch current variable index for a named voltage source.
    ///
    /// Returns `None` if no device with that name has a branch variable.
    pub fn find_vsource_branch_index(&self, name: &str) -> Option<usize> {
        let name_upper = name.to_uppercase();
        for device in &self.devices {
            if device.device_name().to_uppercase() == name_upper {
                return device.branch_index();
            }
        }
        None
    }

    /// Get an iterator over devices.
    pub fn devices(&self) -> &[BoxedStamper] {
        &self.devices
    }

    /// Get the number of devices.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test stamper for unit tests
    #[derive(Debug)]
    struct TestResistor {
        node_pos: NodeId,
        node_neg: NodeId,
        conductance: f64,
    }

    impl Stamper for TestResistor {
        fn stamp(&self, mna: &mut MnaSystem) {
            let i = if self.node_pos.is_ground() {
                None
            } else {
                Some((self.node_pos.as_u32() - 1) as usize)
            };
            let j = if self.node_neg.is_ground() {
                None
            } else {
                Some((self.node_neg.as_u32() - 1) as usize)
            };
            mna.stamp_conductance(i, j, self.conductance);
        }
    }

    #[test]
    fn test_empty_netlist() {
        let netlist = Netlist::new();
        assert_eq!(netlist.num_nodes(), 0);
        assert_eq!(netlist.num_devices(), 0);
    }

    #[test]
    fn test_register_nodes() {
        let mut netlist = Netlist::new();
        netlist.register_node(NodeId::new(3));
        netlist.register_node(NodeId::new(1));
        netlist.register_node(NodeId::GROUND);

        assert_eq!(netlist.num_nodes(), 3);
    }

    #[test]
    fn test_assemble_mna() {
        let mut netlist = Netlist::new();
        netlist.register_node(NodeId::new(1));

        let r = TestResistor {
            node_pos: NodeId::new(1),
            node_neg: NodeId::GROUND,
            conductance: 0.001,
        };
        netlist.add_device(r);

        let mna = netlist.assemble_mna();
        assert_eq!(mna.size(), 1);
        assert!((mna.matrix()[(0, 0)] - 0.001).abs() < 1e-10);
    }
}
