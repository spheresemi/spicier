//! Spicier command-line interface.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use nalgebra::DVector;
use num_complex::Complex;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::NodeId;
use spicier_parser::{AcSweepType, AnalysisCommand, parse_full};
use spicier_solver::{
    AcParams, AcStamper, AcSweepType as SolverAcSweepType, CapacitorState, ComplexMna,
    ComputeBackend, ConvergenceCriteria, DcSolution, DcSweepParams, DcSweepStamper,
    InductorState, IntegrationMethod, NonlinearStamper, TransientParams, TransientStamper,
    solve_ac, solve_dc, solve_dc_sweep, solve_newton_raphson, solve_transient,
};

#[derive(Parser)]
#[command(name = "spicier")]
#[command(about = "A high-performance SPICE circuit simulator", long_about = None)]
#[command(version)]
struct Cli {
    /// Input netlist file
    #[arg(value_name = "FILE")]
    input: Option<PathBuf>,

    /// Run DC operating point analysis (overrides netlist commands)
    #[arg(short = 'o', long = "op")]
    dc_op: bool,

    /// Compute backend: auto, cpu, cuda, or metal
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(ref input) = cli.input {
        // Select compute backend
        let backend = detect_backend(&cli.backend);

        if cli.verbose {
            println!("Backend: {}", backend);
        }

        run_simulation(input, &cli, &backend)?;
    } else {
        println!("Spicier - High Performance SPICE Simulator");
        println!();
        println!("Usage: spicier <netlist.sp> [options]");
        println!();
        println!("Options:");
        println!("  -o, --op           Run DC operating point analysis");
        println!("  --backend <NAME>   Compute backend: auto, cpu, cuda, metal");
        println!("  -v, --verbose      Verbose output");
        println!("  -h, --help         Show help");
        println!("  -V, --version      Show version");
    }

    Ok(())
}

/// Detect and select the compute backend based on CLI argument.
fn detect_backend(name: &str) -> ComputeBackend {
    match name.to_lowercase().as_str() {
        "cpu" => ComputeBackend::Cpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                if spicier_backend_cuda::context::CudaContext::is_available() {
                    ComputeBackend::Cuda { device_id: 0 }
                } else {
                    eprintln!("Warning: CUDA requested but not available, falling back to CPU");
                    ComputeBackend::Cpu
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("Warning: CUDA support not compiled in, falling back to CPU");
                ComputeBackend::Cpu
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                if spicier_backend_metal::context::WgpuContext::is_available() {
                    ComputeBackend::Metal {
                        adapter_name: String::new(),
                    }
                } else {
                    eprintln!("Warning: Metal/WebGPU requested but not available, falling back to CPU");
                    ComputeBackend::Cpu
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                eprintln!("Warning: Metal support not compiled in, falling back to CPU");
                ComputeBackend::Cpu
            }
        }
        _ => {
            // Try Metal first (macOS), then CUDA, then CPU
            #[cfg(feature = "metal")]
            {
                if spicier_backend_metal::context::WgpuContext::is_available() {
                    return ComputeBackend::Metal {
                        adapter_name: String::new(),
                    };
                }
            }
            #[cfg(feature = "cuda")]
            {
                if spicier_backend_cuda::context::CudaContext::is_available() {
                    return ComputeBackend::Cuda { device_id: 0 };
                }
            }
            ComputeBackend::Cpu
        }
    }
}

fn run_simulation(input: &PathBuf, cli: &Cli, _backend: &ComputeBackend) -> Result<()> {
    // Read netlist file
    let content = fs::read_to_string(input)
        .with_context(|| format!("Failed to read netlist: {}", input.display()))?;

    // Parse netlist with analysis commands
    let result = parse_full(&content).map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
    let netlist = result.netlist;
    let analyses = result.analyses;

    if cli.verbose {
        println!("Circuit: {}", netlist.title().unwrap_or("(untitled)"));
        println!("Nodes: {}", netlist.num_nodes());
        println!("Devices: {}", netlist.num_devices());
        println!("Current variables: {}", netlist.num_current_vars());
        println!(
            "Analysis commands: {}",
            if analyses.is_empty() {
                "none (defaulting to .OP)".to_string()
            } else {
                analyses
                    .iter()
                    .map(|a| match a {
                        AnalysisCommand::Op => ".OP".to_string(),
                        AnalysisCommand::Dc { source_name, .. } => {
                            format!(".DC {}", source_name)
                        }
                        AnalysisCommand::Ac { .. } => ".AC".to_string(),
                        AnalysisCommand::Tran { .. } => ".TRAN".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        println!();
    }

    // If --op flag or no analysis commands, run DC operating point
    if cli.dc_op || analyses.is_empty() {
        run_dc_op(&netlist)?;
    }

    // Run each analysis command
    for analysis in &analyses {
        match analysis {
            AnalysisCommand::Op => run_dc_op(&netlist)?,
            AnalysisCommand::Dc {
                source_name,
                start,
                stop,
                step,
            } => run_dc_sweep(&netlist, source_name, *start, *stop, *step)?,
            AnalysisCommand::Ac {
                sweep_type,
                num_points,
                fstart,
                fstop,
            } => run_ac_analysis(&netlist, *sweep_type, *num_points, *fstart, *fstop)?,
            AnalysisCommand::Tran {
                tstep,
                tstop,
                tstart,
            } => run_transient(&netlist, *tstep, *tstop, *tstart)?
        }
    }

    Ok(())
}

fn run_dc_op(netlist: &spicier_core::Netlist) -> Result<()> {
    println!("DC Operating Point Analysis");
    println!("===========================");
    println!();

    let solution = if netlist.has_nonlinear_devices() {
        let stamper = NetlistNonlinearStamper { netlist };
        let criteria = ConvergenceCriteria::default();
        let nr_result = solve_newton_raphson(
            netlist.num_nodes(),
            netlist.num_current_vars(),
            &stamper,
            &criteria,
            None,
        )
        .map_err(|e| anyhow::anyhow!("Newton-Raphson error: {}", e))?;

        if !nr_result.converged {
            eprintln!(
                "Warning: Newton-Raphson did not converge after {} iterations",
                nr_result.iterations
            );
        } else {
            println!(
                "Converged in {} Newton-Raphson iterations.",
                nr_result.iterations
            );
            println!();
        }

        // Convert NrResult to DcSolution
        let num_nodes = netlist.num_nodes();
        DcSolution {
            node_voltages: DVector::from_iterator(
                num_nodes,
                nr_result.solution.iter().take(num_nodes).copied(),
            ),
            branch_currents: DVector::from_iterator(
                netlist.num_current_vars(),
                nr_result.solution.iter().skip(num_nodes).copied(),
            ),
            num_nodes,
        }
    } else {
        let mna = netlist.assemble_mna();
        solve_dc(&mna).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?
    };

    print_dc_solution(netlist, &solution);

    println!("Analysis complete.");
    println!();
    Ok(())
}

fn run_dc_sweep(
    netlist: &spicier_core::Netlist,
    source_name: &str,
    start: f64,
    stop: f64,
    step: f64,
) -> Result<()> {
    println!("DC Sweep Analysis (.DC {} {} {} {})", source_name, start, stop, step);
    println!("==========================================");
    println!();

    let stamper = NetlistSweepStamper {
        netlist,
        source_name: source_name.to_string(),
    };

    let params = DcSweepParams {
        source_name: source_name.to_string(),
        start,
        stop,
        step,
    };

    let result =
        solve_dc_sweep(&stamper, &params).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?;

    // Print header
    print!("{:>12}", source_name);
    for i in 1..=netlist.num_nodes() {
        print!("{:>12}", format!("V({})", i));
    }
    println!();

    // Print separator
    let width = 12 * (1 + netlist.num_nodes());
    println!("{}", "-".repeat(width));

    // Print sweep data
    for (sv, sol) in result.sweep_values.iter().zip(result.solutions.iter()) {
        print!("{:>12.4}", sv);
        for i in 1..=netlist.num_nodes() {
            let v = sol.voltage(NodeId::new(i as u32));
            print!("{:>12.6}", v);
        }
        println!();
    }

    println!();
    println!("Sweep complete ({} points).", result.sweep_values.len());
    println!();
    Ok(())
}

fn run_ac_analysis(
    netlist: &spicier_core::Netlist,
    sweep_type: AcSweepType,
    num_points: usize,
    fstart: f64,
    fstop: f64,
) -> Result<()> {
    let type_name = match sweep_type {
        AcSweepType::Dec => "DEC",
        AcSweepType::Oct => "OCT",
        AcSweepType::Lin => "LIN",
    };

    println!(
        "AC Analysis (.AC {} {} {} {})",
        type_name, num_points, fstart, fstop
    );
    println!("==========================================");
    println!();

    let solver_sweep_type = match sweep_type {
        AcSweepType::Dec => SolverAcSweepType::Decade,
        AcSweepType::Oct => SolverAcSweepType::Octave,
        AcSweepType::Lin => SolverAcSweepType::Linear,
    };

    let stamper = NetlistAcStamper { netlist };

    let params = AcParams {
        fstart,
        fstop,
        num_points,
        sweep_type: solver_sweep_type,
    };

    let result =
        solve_ac(&stamper, &params).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?;

    // Print header
    print!("{:>14}", "Freq(Hz)");
    for i in 1..=netlist.num_nodes() {
        print!("{:>14}{:>14}", format!("VM({})", i), format!("VP({})", i));
    }
    println!();

    let width = 14 + 28 * netlist.num_nodes();
    println!("{}", "-".repeat(width));

    // Print AC data
    for point in &result.points {
        print!("{:>14.4e}", point.frequency);
        for i in 0..netlist.num_nodes() {
            let v = point.solution[i];
            let mag_db = 20.0 * v.norm().log10();
            let phase_deg = v.arg() * 180.0 / PI;
            print!("{:>14.4}{:>14.4}", mag_db, phase_deg);
        }
        println!();
    }

    println!();
    println!("AC analysis complete ({} points).", result.points.len());
    println!();
    Ok(())
}

fn print_dc_solution(netlist: &spicier_core::Netlist, solution: &DcSolution) {
    println!("Node Voltages:");
    for i in 1..=netlist.num_nodes() {
        let node = NodeId::new(i as u32);
        let voltage = solution.voltage(node);
        println!("  V({}) = {:.6} V", i, voltage);
    }

    if netlist.num_current_vars() > 0 {
        println!();
        println!("Branch Currents:");
        for i in 0..netlist.num_current_vars() {
            let current = solution.current(i);
            println!("  I(branch{}) = {:.6} A", i, current);
        }
    }
    println!();
}

/// DC sweep stamper that re-assembles the netlist with a modified source value.
///
/// Since the Netlist uses trait objects, we re-assemble the entire MNA and then
/// patch the voltage source RHS entry for the swept source.
struct NetlistSweepStamper<'a> {
    netlist: &'a spicier_core::Netlist,
    source_name: String,
}

impl DcSweepStamper for NetlistSweepStamper<'_> {
    fn stamp_with_sweep(&self, mna: &mut MnaSystem, _source_name: &str, value: f64) {
        // First, stamp all devices normally
        self.netlist.stamp_into(mna);

        // Then override the swept source's value in the RHS.
        // For a voltage source, the RHS entry at (num_nodes + branch_idx) contains the voltage.
        // We need to find the branch index for the named source.
        // For now, we patch the first voltage source's RHS entry.
        // TODO: look up source by name when Netlist supports device name queries
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

/// Nonlinear stamper for Newton-Raphson DC analysis.
///
/// At each NR iteration, stamps all devices linearized at the current solution.
struct NetlistNonlinearStamper<'a> {
    netlist: &'a spicier_core::Netlist,
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
struct NetlistAcStamper<'a> {
    netlist: &'a spicier_core::Netlist,
}

impl AcStamper for NetlistAcStamper<'_> {
    fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
        for device in self.netlist.devices() {
            match device.ac_info() {
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
                AcDeviceInfo::None => {}
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
struct NetlistTransientStamper<'a> {
    netlist: &'a spicier_core::Netlist,
}

impl TransientStamper for NetlistTransientStamper<'_> {
    fn stamp_static(&self, mna: &mut MnaSystem) {
        // Stamp all devices that are NOT capacitors or inductors.
        // Capacitors and inductors are handled by companion models.
        for device in self.netlist.devices() {
            match device.transient_info() {
                TransientDeviceInfo::Capacitor { .. } | TransientDeviceInfo::Inductor { .. } => {
                    // Skip reactive devices; their companion models are stamped separately
                }
                TransientDeviceInfo::None => {
                    device.stamp(mna);
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
            } => {
                inds.push(InductorState::new(inductance, node_pos, node_neg));
            }
            TransientDeviceInfo::None => {}
        }
    }

    (caps, inds)
}

fn run_transient(
    netlist: &spicier_core::Netlist,
    tstep: f64,
    tstop: f64,
    tstart: f64,
) -> Result<()> {
    println!(
        "Transient Analysis (.TRAN {} {} {})",
        tstep, tstop, tstart
    );
    println!("==========================================");
    println!();

    // 1. Solve DC operating point for initial conditions
    let dc_solution = if netlist.has_nonlinear_devices() {
        let stamper = NetlistNonlinearStamper { netlist };
        let criteria = ConvergenceCriteria::default();
        let nr_result = solve_newton_raphson(
            netlist.num_nodes(),
            netlist.num_current_vars(),
            &stamper,
            &criteria,
            None,
        )
        .map_err(|e| anyhow::anyhow!("DC operating point error: {}", e))?;
        nr_result.solution
    } else {
        let mna = netlist.assemble_mna();
        let dc = solve_dc(&mna).map_err(|e| anyhow::anyhow!("DC operating point error: {}", e))?;
        // Reconstruct full solution vector
        let mut full = DVector::zeros(netlist.num_nodes() + netlist.num_current_vars());
        for i in 0..dc.num_nodes {
            full[i] = dc.node_voltages[i];
        }
        for i in 0..dc.branch_currents.len() {
            full[dc.num_nodes + i] = dc.branch_currents[i];
        }
        full
    };

    // 2. Build reactive element state vectors
    let (mut caps, mut inds) = build_transient_state(netlist);

    // 3. Build transient stamper (stamps non-reactive devices)
    let stamper = NetlistTransientStamper { netlist };

    // 4. Run transient simulation with Trapezoidal method
    let params = TransientParams {
        tstop,
        tstep,
        method: IntegrationMethod::Trapezoidal,
    };

    // Adjust DC solution size if inductor companion models change MNA dimensions
    let tran_size = stamper.num_nodes() + stamper.num_vsources();
    let dc_for_tran = if dc_solution.len() != tran_size {
        let mut adjusted = DVector::zeros(tran_size);
        for i in 0..tran_size.min(dc_solution.len()) {
            adjusted[i] = dc_solution[i];
        }
        adjusted
    } else {
        dc_solution
    };

    let result = solve_transient(&stamper, &mut caps, &mut inds, &params, &dc_for_tran)
        .map_err(|e| anyhow::anyhow!("Transient error: {}", e))?;

    // 5. Print tabular output
    let num_nodes = netlist.num_nodes();

    // Header
    print!("{:>14}", "Time");
    for i in 1..=num_nodes {
        print!("{:>14}", format!("V({})", i));
    }
    println!();

    let width = 14 * (1 + num_nodes);
    println!("{}", "-".repeat(width));

    // Data (skip points before tstart)
    for point in &result.points {
        if point.time < tstart - tstep * 0.5 {
            continue;
        }
        print!("{:>14.6e}", point.time);
        for i in 0..num_nodes {
            let v = if i < point.solution.len() {
                point.solution[i]
            } else {
                0.0
            };
            print!("{:>14.6}", v);
        }
        println!();
    }

    println!();
    println!(
        "Transient analysis complete ({} points).",
        result.points.len()
    );
    println!();
    Ok(())
}
