//! Stamper implementations for connecting parsed netlists to solver traits.

use nalgebra::DVector;
use num_complex::Complex;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_solver::{
    AcStamper, CapacitorState, ComplexMna, DcSweepStamper, InductorState, NonlinearStamper,
    TransientStamper,
};

/// DC sweep stamper that re-assembles the netlist with a modified source value.
///
/// Since the Netlist uses trait objects, we re-assemble the entire MNA and then
/// patch the voltage source RHS entry for the swept source.
pub struct NetlistSweepStamper<'a> {
    pub netlist: &'a spicier_core::Netlist,
    pub source_name: String,
}

impl DcSweepStamper for NetlistSweepStamper<'_> {
    fn stamp_with_sweep(&self, mna: &mut MnaSystem, _source_name: &str, value: f64) {
        // First, stamp all devices normally
        self.netlist.stamp_into(mna);

        // Then override the swept source's value in the RHS.
        // For a voltage source, the RHS entry at (num_nodes + branch_idx) contains the voltage.
        // Look up the branch index by source name.
        if let Some(idx) = self.netlist.find_vsource_branch_index(&self.source_name) {
            let bi = self.netlist.num_nodes() + idx;
            mna.rhs_mut()[bi] = value;
        }
    }

    fn num_nodes(&self) -> usize {
        self.netlist.num_nodes()
    }

    fn num_vsources(&self) -> usize {
        self.netlist.num_current_vars()
    }
}

/// Stamper for nested DC sweeps - stamps with two swept source values.
pub struct NestedSweepStamper<'a> {
    pub netlist: &'a spicier_core::Netlist,
    pub source_name1: String,
    pub source_name2: String,
}

impl NestedSweepStamper<'_> {
    pub fn stamp_with_two_sweeps(&self, mna: &mut MnaSystem, value1: f64, value2: f64) {
        // First stamp all devices normally
        self.netlist.stamp_into(mna);

        // Then override both swept sources' values in the RHS
        if let Some(idx1) = self.netlist.find_vsource_branch_index(&self.source_name1) {
            let bi1 = self.netlist.num_nodes() + idx1;
            mna.rhs_mut()[bi1] = value1;
        }

        if let Some(idx2) = self.netlist.find_vsource_branch_index(&self.source_name2) {
            let bi2 = self.netlist.num_nodes() + idx2;
            mna.rhs_mut()[bi2] = value2;
        }
    }
}

/// Nonlinear stamper for Newton-Raphson DC analysis.
///
/// At each NR iteration, stamps all devices linearized at the current solution.
pub struct NetlistNonlinearStamper<'a> {
    pub netlist: &'a spicier_core::Netlist,
}

impl NonlinearStamper for NetlistNonlinearStamper<'_> {
    fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        self.netlist.stamp_nonlinear_into(mna, solution);
    }
}

/// AC analysis stamper for a parsed netlist.
///
/// Stamps resistors as real conductance, capacitors as jωC admittance,
/// inductors with jωL impedance, and the first voltage source as AC stimulus.
/// When a DC solution is provided, nonlinear devices are linearized at their
/// operating point.
pub struct NetlistAcStamper<'a> {
    pub netlist: &'a spicier_core::Netlist,
    /// DC solution for linearizing nonlinear devices.
    pub dc_solution: Option<&'a DVector<f64>>,
}

impl AcStamper for NetlistAcStamper<'_> {
    fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
        for device in self.netlist.devices() {
            // Use ac_info_at() if DC solution is available, otherwise ac_info()
            let ac_info = match self.dc_solution {
                Some(sol) => device.ac_info_at(sol),
                None => device.ac_info(),
            };
            match ac_info {
                AcDeviceInfo::Resistor {
                    node_pos,
                    node_neg,
                    conductance,
                } => {
                    mna.stamp_conductance(node_pos, node_neg, conductance);
                }
                AcDeviceInfo::Capacitor {
                    node_pos,
                    node_neg,
                    capacitance,
                } => {
                    let yc = Complex::new(0.0, omega * capacitance);
                    mna.stamp_admittance(node_pos, node_neg, yc);
                }
                AcDeviceInfo::Inductor {
                    node_pos,
                    node_neg,
                    inductance,
                    branch_idx,
                } => {
                    mna.stamp_inductor(node_pos, node_neg, branch_idx, omega, inductance);
                }
                AcDeviceInfo::VoltageSource {
                    node_pos,
                    node_neg,
                    branch_idx,
                    ac_mag,
                } => {
                    mna.stamp_voltage_source(
                        node_pos,
                        node_neg,
                        branch_idx,
                        Complex::new(ac_mag, 0.0),
                    );
                }
                AcDeviceInfo::CurrentSource {
                    node_pos,
                    node_neg,
                    ac_mag,
                } => {
                    if ac_mag.abs() > 0.0 {
                        mna.stamp_current_source(node_pos, node_neg, Complex::new(ac_mag, 0.0));
                    }
                }
                AcDeviceInfo::Vcvs {
                    out_pos,
                    out_neg,
                    ctrl_pos,
                    ctrl_neg,
                    branch_idx,
                    gain,
                } => {
                    let br = mna.num_nodes() + branch_idx;
                    // Branch current couples to output nodes
                    if let Some(i) = out_pos {
                        mna.add_element(i, br, Complex::new(1.0, 0.0));
                    }
                    if let Some(i) = out_neg {
                        mna.add_element(i, br, Complex::new(-1.0, 0.0));
                    }
                    // Branch equation
                    if let Some(i) = out_pos {
                        mna.add_element(br, i, Complex::new(1.0, 0.0));
                    }
                    if let Some(i) = out_neg {
                        mna.add_element(br, i, Complex::new(-1.0, 0.0));
                    }
                    if let Some(i) = ctrl_pos {
                        mna.add_element(br, i, Complex::new(-gain, 0.0));
                    }
                    if let Some(i) = ctrl_neg {
                        mna.add_element(br, i, Complex::new(gain, 0.0));
                    }
                }
                AcDeviceInfo::Vccs {
                    out_pos,
                    out_neg,
                    ctrl_pos,
                    ctrl_neg,
                    gm,
                } => {
                    if let Some(i) = out_pos {
                        if let Some(j) = ctrl_pos {
                            mna.add_element(i, j, Complex::new(gm, 0.0));
                        }
                        if let Some(j) = ctrl_neg {
                            mna.add_element(i, j, Complex::new(-gm, 0.0));
                        }
                    }
                    if let Some(i) = out_neg {
                        if let Some(j) = ctrl_pos {
                            mna.add_element(i, j, Complex::new(-gm, 0.0));
                        }
                        if let Some(j) = ctrl_neg {
                            mna.add_element(i, j, Complex::new(gm, 0.0));
                        }
                    }
                }
                AcDeviceInfo::Cccs {
                    out_pos,
                    out_neg,
                    vsource_branch_idx,
                    gain,
                } => {
                    let br = mna.num_nodes() + vsource_branch_idx;
                    if let Some(i) = out_pos {
                        mna.add_element(i, br, Complex::new(gain, 0.0));
                    }
                    if let Some(i) = out_neg {
                        mna.add_element(i, br, Complex::new(-gain, 0.0));
                    }
                }
                AcDeviceInfo::Ccvs {
                    out_pos,
                    out_neg,
                    vsource_branch_idx,
                    branch_idx,
                    gain,
                } => {
                    let br = mna.num_nodes() + branch_idx;
                    let ctrl_br = mna.num_nodes() + vsource_branch_idx;
                    if let Some(i) = out_pos {
                        mna.add_element(i, br, Complex::new(1.0, 0.0));
                    }
                    if let Some(i) = out_neg {
                        mna.add_element(i, br, Complex::new(-1.0, 0.0));
                    }
                    if let Some(i) = out_pos {
                        mna.add_element(br, i, Complex::new(1.0, 0.0));
                    }
                    if let Some(i) = out_neg {
                        mna.add_element(br, i, Complex::new(-1.0, 0.0));
                    }
                    mna.add_element(br, ctrl_br, Complex::new(-gain, 0.0));
                }
                AcDeviceInfo::Diode {
                    node_pos,
                    node_neg,
                    gd,
                } => {
                    // Diode is a simple conductance at the operating point
                    mna.stamp_conductance(node_pos, node_neg, gd);
                }
                AcDeviceInfo::Mosfet {
                    drain,
                    gate,
                    source,
                    gds,
                    gm,
                } => {
                    // MOSFET small-signal model:
                    // 1. gds conductance between drain and source
                    mna.stamp_conductance(drain, source, gds);

                    // 2. gm transconductance: current gm*Vgs from drain to source
                    //    controlled by gate-source voltage
                    if let Some(d) = drain {
                        if let Some(g) = gate {
                            mna.add_element(d, g, Complex::new(gm, 0.0));
                        }
                        if let Some(s) = source {
                            mna.add_element(d, s, Complex::new(-gm, 0.0));
                        }
                    }
                    if let Some(s) = source {
                        if let Some(g) = gate {
                            mna.add_element(s, g, Complex::new(-gm, 0.0));
                        }
                        if let Some(s2) = source {
                            mna.add_element(s, s2, Complex::new(gm, 0.0));
                        }
                    }
                }
                AcDeviceInfo::None | _ => {}
            }
        }
    }

    fn num_nodes(&self) -> usize {
        self.netlist.num_nodes()
    }

    fn num_vsources(&self) -> usize {
        self.netlist.num_current_vars()
    }
}

/// Transient stamper that stamps all non-reactive devices from a netlist.
pub struct NetlistTransientStamper<'a> {
    pub netlist: &'a spicier_core::Netlist,
}

impl TransientStamper for NetlistTransientStamper<'_> {
    fn stamp_at_time(&self, mna: &mut MnaSystem, time: f64) {
        // Stamp all devices that are NOT capacitors or inductors.
        // Capacitors and inductors are handled by companion models.
        // For time-varying sources (PULSE, SIN), evaluate at the given time.
        for device in self.netlist.devices() {
            match device.transient_info() {
                TransientDeviceInfo::Capacitor { .. } | TransientDeviceInfo::Inductor { .. } => {
                    // Skip reactive devices; their companion models are stamped separately
                }
                TransientDeviceInfo::None | _ => {
                    device.stamp_at_time(mna, time);
                }
            }
        }
    }

    fn num_nodes(&self) -> usize {
        self.netlist.num_nodes()
    }

    fn num_vsources(&self) -> usize {
        // Count only voltage source current vars, not inductor branch currents.
        // In transient mode, inductors are replaced by companion models (conductance + current source)
        // and don't need branch current variables.
        let mut vs_count = 0;
        for device in self.netlist.devices() {
            match device.transient_info() {
                TransientDeviceInfo::Inductor { .. } => {
                    // Inductor companion model doesn't need branch current var
                }
                _ => {
                    vs_count += device.num_current_vars();
                }
            }
        }
        vs_count
    }
}

/// Build capacitor and inductor state vectors from the netlist for transient analysis.
pub fn build_transient_state(
    netlist: &spicier_core::Netlist,
) -> (Vec<CapacitorState>, Vec<InductorState>) {
    let mut caps = Vec::new();
    let mut inds = Vec::new();

    for device in netlist.devices() {
        match device.transient_info() {
            TransientDeviceInfo::Capacitor {
                node_pos,
                node_neg,
                capacitance,
            } => {
                caps.push(CapacitorState::new(capacitance, node_pos, node_neg));
            }
            TransientDeviceInfo::Inductor {
                node_pos,
                node_neg,
                inductance,
                branch_index,
            } => {
                inds.push(InductorState::new(inductance, node_pos, node_neg, branch_index));
            }
            TransientDeviceInfo::None | _ => {}
        }
    }

    (caps, inds)
}
