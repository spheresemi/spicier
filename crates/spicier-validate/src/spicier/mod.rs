//! Spicier simulation runner wrapper.
//!
//! This module wraps the spicier_parser and spicier_solver APIs
//! to run simulations and extract results.

use std::collections::HashMap;

use nalgebra::DVector;
use num_complex::Complex;
use spicier_core::NodeId;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_parser::{AcSweepType, AnalysisCommand, ParseResult, parse_full};
use spicier_solver::AcStamper;
use spicier_solver::ac::{
    AcParams, AcResult, AcSweepType as SolverAcSweepType, ComplexMna, solve_ac,
};
use spicier_solver::dc::{DcSolution, solve_dc};
use spicier_solver::newton::{ConvergenceCriteria, NonlinearStamper, solve_newton_raphson};
use spicier_solver::transient::{
    CapacitorState, InductorState, IntegrationMethod, TransientParams, TransientResult,
    TransientStamper, solve_transient,
};

use crate::error::{Error, Result};

/// Result of a spicier DC operating point simulation.
#[derive(Debug, Clone)]
pub struct SpicierDcOp {
    /// Solution from solver.
    pub solution: DcSolution,
    /// Node name to NodeId mapping.
    pub node_map: HashMap<String, NodeId>,
    /// Voltage source names to branch current indices.
    pub vsource_indices: HashMap<String, usize>,
}

impl SpicierDcOp {
    /// Get voltage at a node by name.
    pub fn voltage(&self, name: &str) -> Option<f64> {
        // Handle V(node) syntax
        let node_name = if name.to_lowercase().starts_with("v(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let node_id = self.node_map.get(node_name)?;
        Some(self.solution.voltage(*node_id))
    }

    /// Get current through a voltage source by name.
    pub fn current(&self, name: &str) -> Option<f64> {
        // Handle I(Vx) syntax
        let source_name = if name.to_lowercase().starts_with("i(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let source_upper = source_name.to_uppercase();
        let idx = self.vsource_indices.get(&source_upper)?;
        Some(self.solution.current(*idx))
    }
}

/// Result of a spicier AC simulation.
#[derive(Debug, Clone)]
pub struct SpicierAc {
    /// Result from solver.
    pub result: AcResult,
    /// Node name to NodeId mapping.
    pub node_map: HashMap<String, NodeId>,
}

impl SpicierAc {
    /// Get complex voltage at a node across all frequencies.
    pub fn voltage(&self, name: &str) -> Option<Vec<(f64, Complex<f64>)>> {
        let node_name = if name.to_lowercase().starts_with("v(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let node_id = self.node_map.get(node_name)?;
        if node_id.is_ground() {
            // Ground is always 0V
            return Some(
                self.result
                    .frequencies()
                    .into_iter()
                    .map(|f| (f, Complex::new(0.0, 0.0)))
                    .collect(),
            );
        }

        let idx = node_id.as_u32() as usize - 1;
        Some(self.result.voltage_at(idx))
    }

    /// Get magnitude in dB at a node across all frequencies.
    pub fn magnitude_db(&self, name: &str) -> Option<Vec<(f64, f64)>> {
        let node_name = if name.to_lowercase().starts_with("v(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let node_id = self.node_map.get(node_name)?;
        if node_id.is_ground() {
            return Some(
                self.result
                    .frequencies()
                    .into_iter()
                    .map(|f| (f, f64::NEG_INFINITY))
                    .collect(),
            );
        }

        let idx = node_id.as_u32() as usize - 1;
        Some(self.result.magnitude_db(idx))
    }

    /// Get phase in degrees at a node across all frequencies.
    pub fn phase_deg(&self, name: &str) -> Option<Vec<(f64, f64)>> {
        let node_name = if name.to_lowercase().starts_with("v(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let node_id = self.node_map.get(node_name)?;
        if node_id.is_ground() {
            return Some(
                self.result
                    .frequencies()
                    .into_iter()
                    .map(|f| (f, 0.0))
                    .collect(),
            );
        }

        let idx = node_id.as_u32() as usize - 1;
        Some(self.result.phase_deg(idx))
    }
}

/// Result of a spicier transient simulation.
#[derive(Debug, Clone)]
pub struct SpicierTransient {
    /// Result from solver.
    pub result: TransientResult,
    /// Node name to NodeId mapping.
    pub node_map: HashMap<String, NodeId>,
}

impl SpicierTransient {
    /// Get voltage waveform at a node.
    pub fn voltage(&self, name: &str) -> Option<Vec<(f64, f64)>> {
        let node_name = if name.to_lowercase().starts_with("v(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let node_id = self.node_map.get(node_name)?;
        if node_id.is_ground() {
            return Some(self.result.times().into_iter().map(|t| (t, 0.0)).collect());
        }

        let idx = node_id.as_u32() as usize - 1;
        Some(self.result.voltage_waveform(idx))
    }

    /// Get voltage at a specific time using interpolation.
    pub fn voltage_at(&self, name: &str, time: f64) -> Option<f64> {
        let node_name = if name.to_lowercase().starts_with("v(") && name.ends_with(')') {
            &name[2..name.len() - 1]
        } else {
            name
        };

        let node_id = self.node_map.get(node_name)?;
        if node_id.is_ground() {
            return Some(0.0);
        }

        let idx = node_id.as_u32() as usize - 1;
        self.result.voltage_at(idx, time)
    }
}

/// Unified spicier result type.
#[derive(Debug, Clone)]
pub enum SpicierResult {
    /// DC operating point.
    DcOp(SpicierDcOp),
    /// AC analysis.
    Ac(SpicierAc),
    /// Transient analysis.
    Transient(SpicierTransient),
}

/// Run a netlist through spicier and return the results.
pub fn run_spicier(netlist: &str) -> Result<SpicierResult> {
    let parse_result = parse_full(netlist)?;

    // Find the analysis command
    let analysis = parse_result
        .analyses
        .first()
        .ok_or(Error::NoAnalysisCommand)?;

    match analysis {
        AnalysisCommand::Op => run_dc_op(&parse_result),
        AnalysisCommand::Dc { .. } => {
            // For now, treat DC sweep as DC op at initial point
            run_dc_op(&parse_result)
        }
        AnalysisCommand::Ac {
            sweep_type,
            num_points,
            fstart,
            fstop,
        } => run_ac(&parse_result, *sweep_type, *num_points, *fstart, *fstop),
        AnalysisCommand::Tran {
            tstep,
            tstop,
            tstart,
            ..
        } => run_transient(&parse_result, *tstep, *tstop, *tstart),
        _ => Err(Error::UnsupportedAnalysis(format!("{:?}", analysis))),
    }
}

/// Stamper for Newton-Raphson iteration in DC analysis.
struct DcNonlinearStamper<'a> {
    netlist: &'a spicier_core::Netlist,
}

impl NonlinearStamper for DcNonlinearStamper<'_> {
    fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        // Clear MNA (handled by newton.rs)
        // Stamp linear devices first
        for device in self.netlist.devices() {
            if !device.is_nonlinear() {
                device.stamp(mna);
            }
        }
        // Stamp nonlinear devices at current operating point
        for device in self.netlist.devices() {
            if device.is_nonlinear() {
                device.stamp_nonlinear(mna, solution);
            }
        }
    }
}

/// Run DC operating point analysis.
fn run_dc_op(parse_result: &ParseResult) -> Result<SpicierResult> {
    let netlist = &parse_result.netlist;

    let solution = if netlist.has_nonlinear_devices() {
        // Use Newton-Raphson for nonlinear circuits
        let stamper = DcNonlinearStamper { netlist };
        let criteria = ConvergenceCriteria::default();
        let nr_result = solve_newton_raphson(
            netlist.num_nodes(),
            netlist.num_current_vars(),
            &stamper,
            &criteria,
            None,
        )?;

        if !nr_result.converged {
            eprintln!(
                "Warning: Newton-Raphson did not converge after {} iterations",
                nr_result.iterations
            );
        }

        let num_nodes = netlist.num_nodes();
        let num_vsources = netlist.num_current_vars();
        DcSolution {
            node_voltages: DVector::from_iterator(
                num_nodes,
                nr_result.solution.iter().take(num_nodes).copied(),
            ),
            branch_currents: DVector::from_iterator(
                num_vsources,
                nr_result.solution.iter().skip(num_nodes).copied(),
            ),
            num_nodes,
        }
    } else {
        // Use direct linear solve for linear circuits
        let mut mna = MnaSystem::new(netlist.num_nodes(), netlist.num_current_vars());
        netlist.stamp_into(&mut mna);
        solve_dc(&mna)?
    };

    // Build voltage source index map
    let mut vsource_indices = HashMap::new();
    let mut idx = 0usize;
    for device in netlist.devices() {
        let name = device.device_name();
        let upper = name.to_uppercase();
        if upper.starts_with('V') {
            vsource_indices.insert(upper, idx);
            idx += 1;
        } else if upper.starts_with('L') {
            // Inductors also have branch currents
            idx += 1;
        } else if upper.starts_with('E') || upper.starts_with('H') {
            // VCVS and CCVS have branch currents
            idx += 1;
        }
    }

    Ok(SpicierResult::DcOp(SpicierDcOp {
        solution,
        node_map: parse_result.node_map.clone(),
        vsource_indices,
    }))
}

/// Stamper for AC analysis that wraps a ParseResult.
struct AcCircuitStamper<'a> {
    netlist: &'a spicier_core::Netlist,
}

impl AcStamper for AcCircuitStamper<'_> {
    fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
        for device in self.netlist.devices() {
            let ac_info = device.ac_info();
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
                    mna.stamp_conductance(node_pos, node_neg, gd);
                }
                AcDeviceInfo::Mosfet {
                    drain,
                    gate,
                    source,
                    gds,
                    gm,
                } => {
                    mna.stamp_conductance(drain, source, gds);
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

/// Run AC analysis.
fn run_ac(
    parse_result: &ParseResult,
    sweep_type: AcSweepType,
    num_points: usize,
    fstart: f64,
    fstop: f64,
) -> Result<SpicierResult> {
    let stamper = AcCircuitStamper {
        netlist: &parse_result.netlist,
    };

    let solver_sweep_type = match sweep_type {
        AcSweepType::Dec => SolverAcSweepType::Decade,
        AcSweepType::Oct => SolverAcSweepType::Octave,
        AcSweepType::Lin | _ => SolverAcSweepType::Linear,
    };

    let params = AcParams {
        fstart,
        fstop,
        num_points,
        sweep_type: solver_sweep_type,
    };

    let result = solve_ac(&stamper, &params)?;

    Ok(SpicierResult::Ac(SpicierAc {
        result,
        node_map: parse_result.node_map.clone(),
    }))
}

/// Stamper for transient analysis.
struct TransientCircuitStamper<'a> {
    netlist: &'a spicier_core::Netlist,
}

impl TransientStamper for TransientCircuitStamper<'_> {
    fn stamp_at_time(&self, mna: &mut MnaSystem, time: f64) {
        // Stamp all devices that are NOT capacitors or inductors.
        // Capacitors and inductors are handled by companion models.
        for device in self.netlist.devices() {
            match device.transient_info() {
                TransientDeviceInfo::Capacitor { .. } | TransientDeviceInfo::Inductor { .. } => {
                    // Skip reactive devices
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
        // Count only voltage source current vars, not inductor branch currents
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
fn build_transient_state(
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

/// Run transient analysis.
fn run_transient(
    parse_result: &ParseResult,
    tstep: f64,
    tstop: f64,
    _tstart: f64,
) -> Result<SpicierResult> {
    let netlist = &parse_result.netlist;

    // Build capacitor and inductor states
    let (mut caps, mut inds) = build_transient_state(netlist);

    // Get DC operating point as initial condition
    let mut mna = MnaSystem::new(netlist.num_nodes(), netlist.num_current_vars());
    netlist.stamp_into(&mut mna);
    let dc_solution = solve_dc(&mna)?;
    let dc_vec = DVector::from_iterator(
        dc_solution.node_voltages.len() + dc_solution.branch_currents.len(),
        dc_solution
            .node_voltages
            .iter()
            .chain(dc_solution.branch_currents.iter())
            .copied(),
    );

    let stamper = TransientCircuitStamper { netlist };
    let params = TransientParams {
        tstop,
        tstep,
        method: IntegrationMethod::Trapezoidal,
    };

    let result = solve_transient(&stamper, &mut caps, &mut inds, &params, &dc_vec)?;

    Ok(SpicierResult::Transient(SpicierTransient {
        result,
        node_map: parse_result.node_map.clone(),
    }))
}
