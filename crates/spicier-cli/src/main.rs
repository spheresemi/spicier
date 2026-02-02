//! Spicier command-line interface.

mod analysis;
mod backend;
mod output;
mod stampers;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use spicier_parser::{
    AnalysisCommand, DcSweepType, MeasureAnalysis, Measurement, PrintAnalysisType, parse_full,
};

use analysis::{
    run_ac_analysis, run_dc_op, run_dc_param_sweep, run_dc_sweep, run_noise_analysis, run_transient,
};
use backend::detect_backend;

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

        run_simulation(input, &cli)?;
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

fn run_simulation(input: &PathBuf, cli: &Cli) -> Result<()> {
    // Read netlist file
    let content = fs::read_to_string(input)
        .with_context(|| format!("Failed to read netlist: {}", input.display()))?;

    // Parse netlist with analysis commands
    let result = parse_full(&content).map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
    let netlist = result.netlist;
    let analyses = result.analyses;
    let initial_conditions = result.initial_conditions;
    let node_map = result.node_map;
    let print_commands = result.print_commands;
    let measurements = result.measurements;

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
                        AnalysisCommand::Dc { sweeps } => {
                            let names: Vec<_> =
                                sweeps.iter().map(|s| s.source_name.as_str()).collect();
                            format!(".DC {}", names.join(", "))
                        }
                        AnalysisCommand::Ac { .. } => ".AC".to_string(),
                        AnalysisCommand::Tran { .. } => ".TRAN".to_string(),
                        AnalysisCommand::Noise { .. } => ".NOISE".to_string(),
                        _ => "(unknown analysis)".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        if !print_commands.is_empty() {
            println!("Print commands: {}", print_commands.len());
        }
        println!();
    }

    // Helper to find print commands for an analysis type
    let get_print_vars = |analysis_type: PrintAnalysisType| -> Vec<&_> {
        print_commands
            .iter()
            .filter(|p| p.analysis_type == analysis_type)
            .flat_map(|p| &p.variables)
            .collect()
    };

    // Helper to get measurements for an analysis type
    let get_measurements = |analysis: MeasureAnalysis| -> Vec<&Measurement> {
        measurements
            .iter()
            .filter(|m| std::mem::discriminant(&m.analysis) == std::mem::discriminant(&analysis))
            .collect()
    };

    // If --op flag or no analysis commands, run DC operating point
    if cli.dc_op || analyses.is_empty() {
        let print_vars = get_print_vars(PrintAnalysisType::Dc);
        let dc_measurements = get_measurements(MeasureAnalysis::Dc);
        run_dc_op(&netlist, &print_vars, &node_map, &dc_measurements)?;
    }

    // Run each analysis command
    for analysis in &analyses {
        match analysis {
            AnalysisCommand::Op => {
                let print_vars = get_print_vars(PrintAnalysisType::Dc);
                let dc_measurements = get_measurements(MeasureAnalysis::Dc);
                run_dc_op(&netlist, &print_vars, &node_map, &dc_measurements)?;
            }
            AnalysisCommand::Dc { sweeps } => {
                let print_vars = get_print_vars(PrintAnalysisType::Dc);
                let dc_measurements = get_measurements(MeasureAnalysis::Dc);

                // Check if this is a parameter sweep
                let has_param_sweep = sweeps.iter().any(|s| s.sweep_type == DcSweepType::Param);

                if has_param_sweep {
                    run_dc_param_sweep(
                        &content,
                        sweeps,
                        &print_vars,
                        &dc_measurements,
                    )?;
                } else {
                    run_dc_sweep(&netlist, sweeps, &print_vars, &node_map, &dc_measurements)?;
                }
            }
            AnalysisCommand::Ac {
                sweep_type,
                num_points,
                fstart,
                fstop,
            } => {
                let print_vars = get_print_vars(PrintAnalysisType::Ac);
                let ac_measurements = get_measurements(MeasureAnalysis::Ac);
                run_ac_analysis(
                    &netlist,
                    *sweep_type,
                    *num_points,
                    *fstart,
                    *fstop,
                    &print_vars,
                    &node_map,
                    &ac_measurements,
                )?;
            }
            AnalysisCommand::Tran {
                tstep,
                tstop,
                tstart,
                uic,
            } => {
                let print_vars = get_print_vars(PrintAnalysisType::Tran);
                let tran_measurements = get_measurements(MeasureAnalysis::Tran);
                run_transient(
                    &netlist,
                    *tstep,
                    *tstop,
                    *tstart,
                    *uic,
                    &initial_conditions,
                    &node_map,
                    &print_vars,
                    &tran_measurements,
                )?;
            }
            AnalysisCommand::Noise {
                output_node,
                output_ref_node,
                input_source,
                sweep_type,
                num_points,
                fstart,
                fstop,
            } => {
                run_noise_analysis(
                    &netlist,
                    output_node,
                    output_ref_node.as_deref(),
                    input_source,
                    *sweep_type,
                    *num_points,
                    *fstart,
                    *fstop,
                    &node_map,
                )?;
            }
            _ => {
                eprintln!("Warning: unsupported analysis type, skipping");
            }
        }
    }

    Ok(())
}
