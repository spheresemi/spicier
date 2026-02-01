//! Spicier command-line interface.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use num_complex::Complex;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::AcDeviceInfo;
use spicier_core::NodeId;
use spicier_parser::{AcSweepType, AnalysisCommand, parse_full};
use spicier_solver::{
    AcParams, AcStamper, AcSweepType as SolverAcSweepType, ComplexMna, DcSweepParams,
    DcSweepStamper, DcSolution, solve_ac, solve_dc, solve_dc_sweep,
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

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(ref input) = cli.input {
        run_simulation(input, &cli)?;
    } else {
        println!("Spicier - High Performance SPICE Simulator");
        println!();
        println!("Usage: spicier <netlist.sp> [options]");
        println!();
        println!("Options:");
        println!("  -o, --op       Run DC operating point analysis");
        println!("  -v, --verbose  Verbose output");
        println!("  -h, --help     Show help");
        println!("  -V, --version  Show version");
    }

    Ok(())
}

fn run_simulation(input: &PathBuf, cli: &Cli) -> Result<()> {
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
                tstep: _,
                tstop: _,
                tstart: _,
            } => {
                println!("Transient analysis (.TRAN) via CLI is not yet supported.");
                println!("Use the solver API directly for transient simulations.");
                println!();
            }
        }
    }

    Ok(())
}

fn run_dc_op(netlist: &spicier_core::Netlist) -> Result<()> {
    println!("DC Operating Point Analysis");
    println!("===========================");
    println!();

    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?;

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
