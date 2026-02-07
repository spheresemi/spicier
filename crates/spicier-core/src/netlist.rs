//! Netlist: A complete circuit description ready for simulation.

use nalgebra::DVector;

use crate::mna::MnaSystem;
use crate::node::NodeId;

/// A boxed device that can stamp into an MNA matrix.
pub type BoxedStamper = Box<dyn Stamper>;

/// Information about a device for AC analysis stamping.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// VCVS: voltage-controlled voltage source.
    Vcvs {
        out_pos: Option<usize>,
        out_neg: Option<usize>,
        ctrl_pos: Option<usize>,
        ctrl_neg: Option<usize>,
        branch_idx: usize,
        gain: f64,
    },
    /// VCCS: voltage-controlled current source.
    Vccs {
        out_pos: Option<usize>,
        out_neg: Option<usize>,
        ctrl_pos: Option<usize>,
        ctrl_neg: Option<usize>,
        gm: f64,
    },
    /// CCCS: current-controlled current source.
    Cccs {
        out_pos: Option<usize>,
        out_neg: Option<usize>,
        vsource_branch_idx: usize,
        gain: f64,
    },
    /// CCVS: current-controlled voltage source.
    Ccvs {
        out_pos: Option<usize>,
        out_neg: Option<usize>,
        vsource_branch_idx: usize,
        branch_idx: usize,
        gain: f64,
    },
    /// Diode: linearized as small-signal conductance gd at operating point.
    Diode {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        /// Small-signal conductance gd = dId/dVd at DC operating point.
        gd: f64,
    },
    /// MOSFET: linearized as gds + gm*Vgs at operating point.
    Mosfet {
        drain: Option<usize>,
        gate: Option<usize>,
        source: Option<usize>,
        /// Output conductance gds = dIds/dVds at DC operating point.
        gds: f64,
        /// Transconductance gm = dIds/dVgs at DC operating point.
        gm: f64,
    },
    /// BSIM1 MOSFET (Level 4): linearized as gds + gm*Vgs + gmbs*Vbs at operating point.
    /// DC model only - no intrinsic capacitances.
    Bsim1Mosfet {
        drain: Option<usize>,
        gate: Option<usize>,
        source: Option<usize>,
        bulk: Option<usize>,
        /// Output conductance gds = dIds/dVds at DC operating point.
        gds: f64,
        /// Transconductance gm = dIds/dVgs at DC operating point.
        gm: f64,
        /// Body transconductance gmbs = dIds/dVbs at DC operating point.
        gmbs: f64,
    },
    /// BSIM3 MOSFET: linearized as gds + gm*Vgs + gmbs*Vbs at operating point.
    Bsim3Mosfet {
        drain: Option<usize>,
        gate: Option<usize>,
        source: Option<usize>,
        bulk: Option<usize>,
        /// Output conductance gds = dIds/dVds at DC operating point.
        gds: f64,
        /// Transconductance gm = dIds/dVgs at DC operating point.
        gm: f64,
        /// Body transconductance gmbs = dIds/dVbs at DC operating point.
        gmbs: f64,
        /// Gate-source capacitance (F) - intrinsic + overlap.
        cgs: f64,
        /// Gate-drain capacitance (F) - intrinsic + overlap.
        cgd: f64,
        /// Gate-bulk capacitance (F) - intrinsic + overlap.
        cgb: f64,
        /// Bulk-source junction capacitance (F).
        cbs: f64,
        /// Bulk-drain junction capacitance (F).
        cbd: f64,
    },
    /// JFET: linearized as gds + gm*Vgs at operating point.
    Jfet {
        drain: Option<usize>,
        gate: Option<usize>,
        source: Option<usize>,
        /// Output conductance gds = dIds/dVds at DC operating point.
        gds: f64,
        /// Transconductance gm = dIds/dVgs at DC operating point.
        gm: f64,
    },
    /// BJT: linearized using hybrid-π model at operating point.
    Bjt {
        collector: Option<usize>,
        base: Option<usize>,
        emitter: Option<usize>,
        /// Transconductance gm = dIc/dVbe at DC operating point.
        gm: f64,
        /// Input conductance gpi = gm/beta at DC operating point.
        gpi: f64,
        /// Output conductance go = Ic/Vaf at DC operating point.
        go: f64,
    },
    /// Mutual inductance: coupling between two inductors.
    MutualInductance {
        /// Branch index of the first inductor.
        l1_branch_idx: usize,
        /// Branch index of the second inductor.
        l2_branch_idx: usize,
        /// Mutual inductance M = k * sqrt(L1 * L2).
        mutual_inductance: f64,
    },
    /// Transmission line: lumped LC model.
    TransmissionLine {
        /// Port 1 positive node index.
        port1_pos: Option<usize>,
        /// Port 1 negative node index.
        port1_neg: Option<usize>,
        /// Port 2 positive node index.
        port2_pos: Option<usize>,
        /// Port 2 negative node index.
        port2_neg: Option<usize>,
        /// Characteristic impedance (Ohms).
        z0: f64,
        /// Propagation delay (seconds).
        td: f64,
        /// Number of LC sections.
        num_sections: usize,
        /// Internal node indices for the LC ladder.
        internal_nodes: Vec<Option<usize>>,
        /// Base index for inductor branch currents.
        current_base_index: usize,
    },
    /// Unknown device or no AC contribution.
    None,
}

/// Information about a device for transient analysis.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TransientDeviceInfo {
    /// Capacitor with node indices and capacitance.
    Capacitor {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        capacitance: f64,
    },
    /// Inductor with node indices, inductance, and branch current index.
    Inductor {
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        inductance: f64,
        /// Branch current index in DC solution (for extracting initial current).
        branch_index: usize,
    },
    /// Mutual inductance coupling between two inductors.
    MutualInductance {
        /// Branch index of the first inductor.
        l1_branch_idx: usize,
        /// Branch index of the second inductor.
        l2_branch_idx: usize,
        /// Mutual inductance M = k * sqrt(L1 * L2).
        mutual_inductance: f64,
    },
    /// Transmission line: lumped LC model.
    TransmissionLine {
        /// Port 1 positive node index.
        port1_pos: Option<usize>,
        /// Port 1 negative node index.
        port1_neg: Option<usize>,
        /// Port 2 positive node index.
        port2_pos: Option<usize>,
        /// Port 2 negative node index.
        port2_neg: Option<usize>,
        /// Characteristic impedance (Ohms).
        z0: f64,
        /// Propagation delay (seconds).
        td: f64,
        /// Number of LC sections.
        num_sections: usize,
        /// Internal node indices for the LC ladder.
        internal_nodes: Vec<Option<usize>>,
        /// Base index for inductor branch currents.
        current_base_index: usize,
    },
    /// Not a reactive device.
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
    ///
    /// For linear devices, this returns fixed parameters.
    /// For nonlinear devices, use `ac_info_at()` with the DC solution.
    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::None
    }

    /// Provide AC analysis information at a given DC operating point.
    ///
    /// For nonlinear devices (diodes, MOSFETs), this extracts the
    /// small-signal parameters (gd, gm, gds) from the DC solution.
    /// Linear devices can use the default implementation which just
    /// calls `ac_info()`.
    fn ac_info_at(&self, _solution: &DVector<f64>) -> AcDeviceInfo {
        self.ac_info()
    }

    /// Whether this device is nonlinear (requires Newton-Raphson).
    fn is_nonlinear(&self) -> bool {
        false
    }

    /// Stamp the device linearized at the current solution vector.
    ///
    /// Default implementation falls back to the linear stamp.
    fn stamp_nonlinear(&self, mna: &mut MnaSystem, _solution: &DVector<f64>) {
        self.stamp(mna);
    }

    /// Provide transient analysis information for this device.
    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::None
    }

    /// Stamp this device at a specific time (for transient analysis).
    ///
    /// For time-varying sources (PULSE, SIN, PWL), the source value is
    /// evaluated at the given time. For DC sources and other devices,
    /// this is equivalent to `stamp()`.
    fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
        self.stamp(mna);
    }

    /// Whether this device is an independent source (V or I).
    ///
    /// Used by source stepping convergence aid to identify which devices
    /// should be scaled during the stepping process.
    fn is_source(&self) -> bool {
        false
    }

    /// Stamp the device linearized at the current solution with source scaling.
    ///
    /// For independent sources (V, I), the source value is multiplied by
    /// `source_factor` (0.0 to 1.0). For other devices, this is equivalent
    /// to `stamp_nonlinear`.
    ///
    /// Default implementation ignores the source factor.
    fn stamp_nonlinear_scaled(
        &self,
        mna: &mut MnaSystem,
        solution: &DVector<f64>,
        _source_factor: f64,
    ) {
        self.stamp_nonlinear(mna, solution);
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

    /// Check if the netlist contains any nonlinear devices.
    pub fn has_nonlinear_devices(&self) -> bool {
        self.devices.iter().any(|d| d.is_nonlinear())
    }

    /// Stamp all devices into the MNA system linearized at the given solution.
    ///
    /// Nonlinear devices use their linearized stamp; linear devices use their
    /// normal stamp.
    pub fn stamp_nonlinear_into(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        for device in &self.devices {
            device.stamp_nonlinear(mna, solution);
        }
    }

    /// Stamp all devices with source scaling for source stepping.
    ///
    /// Independent sources (V, I) have their values multiplied by `source_factor`.
    /// This is used by the source stepping convergence aid.
    pub fn stamp_nonlinear_into_scaled(
        &self,
        mna: &mut MnaSystem,
        solution: &DVector<f64>,
        source_factor: f64,
    ) {
        for device in &self.devices {
            device.stamp_nonlinear_scaled(mna, solution, source_factor);
        }
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
        let matrix = mna.to_dense_matrix();
        assert!((matrix[(0, 0)] - 0.001).abs() < 1e-10);
    }
}
