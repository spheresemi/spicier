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
                AcDeviceInfo::MutualInductance {
                    l1_branch_idx,
                    l2_branch_idx,
                    mutual_inductance,
                } => {
                    // Mutual inductance coupling between two inductors.
                    // The coupled inductor equations are:
                    //   V1 = jωL1 * I1 + jωM * I2
                    //   V2 = jωM * I1 + jωL2 * I2
                    //
                    // The individual inductors already stamp jωL on the diagonal.
                    // Here we add the off-diagonal coupling terms jωM.
                    let jwm = Complex::new(0.0, omega * mutual_inductance);
                    let br1 = mna.num_nodes() + l1_branch_idx;
                    let br2 = mna.num_nodes() + l2_branch_idx;

                    // Add jωM coupling: L1 branch depends on L2 current and vice versa
                    mna.add_element(br1, br2, jwm);
                    mna.add_element(br2, br1, jwm);
                }
                AcDeviceInfo::Jfet {
                    drain,
                    gate,
                    source,
                    gds,
                    gm,
                } => {
                    // JFET small-signal model (same structure as MOSFET):
                    // 1. gds conductance between drain and source
                    mna.stamp_conductance(drain, source, gds);

                    // 2. gm transconductance: current gm*Vgs from drain to source
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
                        mna.add_element(s, s, Complex::new(gm, 0.0));
                    }
                }
                AcDeviceInfo::Bjt {
                    collector,
                    base,
                    emitter,
                    gm,
                    gpi,
                    go,
                } => {
                    // BJT hybrid-π small-signal model:
                    // 1. gpi conductance between base and emitter (input resistance)
                    mna.stamp_conductance(base, emitter, gpi);

                    // 2. go conductance between collector and emitter (output resistance)
                    mna.stamp_conductance(collector, emitter, go);

                    // 3. gm transconductance: current gm*Vbe from collector to emitter
                    if let Some(c) = collector {
                        if let Some(b) = base {
                            mna.add_element(c, b, Complex::new(gm, 0.0));
                        }
                        if let Some(e) = emitter {
                            mna.add_element(c, e, Complex::new(-gm, 0.0));
                        }
                    }
                    if let Some(e) = emitter {
                        if let Some(b) = base {
                            mna.add_element(e, b, Complex::new(-gm, 0.0));
                        }
                        mna.add_element(e, e, Complex::new(gm, 0.0));
                    }
                }
                AcDeviceInfo::TransmissionLine {
                    port1_pos,
                    port1_neg: _,
                    port2_pos,
                    port2_neg: _,
                    z0,
                    td,
                    num_sections,
                    internal_nodes,
                    current_base_index,
                } => {
                    // Transmission line lumped LC model:
                    // L per section = Z0 * TD / N
                    // C per section = TD / (Z0 * N)
                    //
                    // The LC ladder: Port1+ --L1-- int[0] --L2-- int[1] ... --LN-- Port2+
                    //                              |            |              |
                    //                              C1           C2            CN
                    //                              |            |              |
                    //                            Port1-      Port1-         Port2-
                    //
                    // For simplicity, we assume port1_neg == port2_neg (common ground).
                    // The capacitors are connected to ground at each internal node and ports.

                    let l_section = z0 * td / num_sections as f64;
                    let c_section = td / (z0 * num_sections as f64);

                    // Build node chain: port1_pos, internal_nodes[0..N-2], port2_pos
                    let mut node_chain: Vec<Option<usize>> = Vec::with_capacity(num_sections + 1);
                    node_chain.push(port1_pos);
                    for int_node in &internal_nodes {
                        node_chain.push(*int_node);
                    }
                    node_chain.push(port2_pos);

                    // Stamp each LC section
                    for i in 0..num_sections {
                        let left_node = node_chain[i];
                        let right_node = node_chain[i + 1];
                        let branch_idx = current_base_index + i;

                        // Stamp inductor with jωL impedance
                        mna.stamp_inductor(left_node, right_node, branch_idx, omega, l_section);

                        // Stamp shunt capacitor at right_node with jωC admittance
                        let yc = Complex::new(0.0, omega * c_section);
                        mna.stamp_admittance(right_node, None, yc);
                    }
                }
                AcDeviceInfo::Bsim3Mosfet {
                    drain,
                    gate,
                    source,
                    bulk,
                    gds,
                    gm,
                    gmbs,
                    cgs,
                    cgd,
                    cgb,
                    cbs,
                    cbd,
                } => {
                    // BSIM3 small-signal model with capacitances:
                    // 1. gds conductance between drain and source
                    mna.stamp_conductance(drain, source, gds);

                    // 2. gm transconductance: current gm*Vgs from drain to source
                    mna.stamp_vccs(drain, source, gate, source, gm);

                    // 3. gmbs transconductance: current gmbs*Vbs from drain to source
                    mna.stamp_vccs(drain, source, bulk, source, gmbs);

                    // 4. Capacitances as jωC admittances
                    // Cgs: gate to source
                    if cgs > 0.0 {
                        let yc = Complex::new(0.0, omega * cgs);
                        mna.stamp_admittance(gate, source, yc);
                    }
                    // Cgd: gate to drain
                    if cgd > 0.0 {
                        let yc = Complex::new(0.0, omega * cgd);
                        mna.stamp_admittance(gate, drain, yc);
                    }
                    // Cgb: gate to bulk
                    if cgb > 0.0 {
                        let yc = Complex::new(0.0, omega * cgb);
                        mna.stamp_admittance(gate, bulk, yc);
                    }
                    // Cbs: bulk to source (junction)
                    if cbs > 0.0 {
                        let yc = Complex::new(0.0, omega * cbs);
                        mna.stamp_admittance(bulk, source, yc);
                    }
                    // Cbd: bulk to drain (junction)
                    if cbd > 0.0 {
                        let yc = Complex::new(0.0, omega * cbd);
                        mna.stamp_admittance(bulk, drain, yc);
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
        // Stamp all devices that are NOT capacitors, inductors, or transmission lines.
        // Capacitors, inductors, and transmission lines (which expand to LC sections)
        // are handled by companion models.
        // For time-varying sources (PULSE, SIN), evaluate at the given time.
        for device in self.netlist.devices() {
            match device.transient_info() {
                TransientDeviceInfo::Capacitor { .. }
                | TransientDeviceInfo::Inductor { .. }
                | TransientDeviceInfo::TransmissionLine { .. } => {
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
        // Count only voltage source current vars, not inductor or transmission line branch currents.
        // In transient mode, inductors are replaced by companion models (conductance + current source)
        // and don't need branch current variables. Transmission lines expand to LC sections
        // whose inductors are also handled by companion models.
        let mut vs_count = 0;
        for device in self.netlist.devices() {
            match device.transient_info() {
                TransientDeviceInfo::Inductor { .. }
                | TransientDeviceInfo::TransmissionLine { .. } => {
                    // Inductor companion model doesn't need branch current var
                    // Transmission line inductors are also handled as companion models
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
                inds.push(InductorState::new(
                    inductance,
                    node_pos,
                    node_neg,
                    branch_index,
                ));
            }
            TransientDeviceInfo::TransmissionLine {
                port1_pos,
                port1_neg: _,
                port2_pos,
                port2_neg: _,
                z0,
                td,
                num_sections,
                internal_nodes,
                current_base_index,
            } => {
                // Expand transmission line into LC sections for transient analysis.
                // L per section = Z0 * TD / N
                // C per section = TD / (Z0 * N)

                let l_section = z0 * td / num_sections as f64;
                let c_section = td / (z0 * num_sections as f64);

                // Build node chain: port1_pos, internal_nodes[0..N-2], port2_pos
                let mut node_chain: Vec<Option<usize>> = Vec::with_capacity(num_sections + 1);
                node_chain.push(port1_pos);
                for int_node in &internal_nodes {
                    node_chain.push(*int_node);
                }
                node_chain.push(port2_pos);

                // Create LC elements for each section
                for i in 0..num_sections {
                    let left_node = node_chain[i];
                    let right_node = node_chain[i + 1];
                    let branch_index = current_base_index + i;

                    // Inductor in series
                    inds.push(InductorState::new(
                        l_section,
                        left_node,
                        right_node,
                        branch_index,
                    ));

                    // Shunt capacitor at right_node (to ground)
                    caps.push(CapacitorState::new(c_section, right_node, None));
                }
            }
            TransientDeviceInfo::None | _ => {}
        }
    }

    (caps, inds)
}
