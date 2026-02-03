//! AC small-signal analysis.

use std::f64::consts::PI;

use anyhow::Result;
use spicier_core::NodeId;
use spicier_parser::{AcSweepType, Measurement, OutputVariable};
use spicier_solver::{
    AcParams, AcSweepType as SolverAcSweepType, ConvergenceCriteria, MeasureEvaluator, solve_ac,
    solve_newton_raphson,
};
use std::collections::HashMap;

use crate::output::get_ac_print_nodes;
use crate::stampers::{NetlistAcStamper, NetlistNonlinearStamper};

/// Run AC small-signal analysis.
#[allow(clippy::too_many_arguments)]
pub fn run_ac_analysis(
    netlist: &spicier_core::Netlist,
    sweep_type: AcSweepType,
    num_points: usize,
    fstart: f64,
    fstop: f64,
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    measurements: &[&Measurement],
) -> Result<()> {
    let type_name = match sweep_type {
        AcSweepType::Dec => "DEC",
        AcSweepType::Oct => "OCT",
        AcSweepType::Lin | _ => "LIN",
    };

    println!(
        "AC Analysis (.AC {} {} {} {})",
        type_name, num_points, fstart, fstop
    );
    println!("==========================================");
    println!();

    // Get nodes to print from .PRINT AC variables
    let nodes_to_print = get_ac_print_nodes(print_vars, node_map, netlist.num_nodes());

    // For nonlinear circuits, first compute DC operating point
    let dc_solution = if netlist.has_nonlinear_devices() {
        println!("Computing DC operating point for linearization...");

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
                "Warning: DC operating point did not converge after {} iterations",
                nr_result.iterations
            );
        } else {
            println!(
                "DC operating point converged in {} iterations.",
                nr_result.iterations
            );
        }
        println!();

        Some(nr_result.solution)
    } else {
        None
    };

    let solver_sweep_type = match sweep_type {
        AcSweepType::Dec => SolverAcSweepType::Decade,
        AcSweepType::Oct => SolverAcSweepType::Octave,
        AcSweepType::Lin | _ => SolverAcSweepType::Linear,
    };

    let stamper = NetlistAcStamper {
        netlist,
        dc_solution: dc_solution.as_ref(),
    };

    let params = AcParams {
        fstart,
        fstop,
        num_points,
        sweep_type: solver_sweep_type,
    };

    let result = solve_ac(&stamper, &params).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?;

    // Print header
    print!("{:>14}", "Freq(Hz)");
    for (name, _) in &nodes_to_print {
        print!(
            "{:>14}{:>14}",
            format!("VM({})", name),
            format!("VP({})", name)
        );
    }
    println!();

    let width = 14 + 28 * nodes_to_print.len();
    println!("{}", "-".repeat(width));

    // Print AC data
    for point in &result.points {
        print!("{:>14.4e}", point.frequency);
        for (_, node_id) in &nodes_to_print {
            let idx = (node_id.as_u32() - 1) as usize;
            let v = point.solution[idx];
            let mag_db = 20.0 * v.norm().log10();
            let phase_deg = v.arg() * 180.0 / PI;
            print!("{:>14.4}{:>14.4}", mag_db, phase_deg);
        }
        println!();
    }

    println!();
    println!("AC analysis complete ({} points).", result.points.len());

    // Evaluate and print measurements
    if !measurements.is_empty() {
        println!();
        println!("Measurements:");
        println!("{}", "-".repeat(50));

        // Build node name to MNA index map for measurement evaluation
        let mna_node_map: HashMap<String, usize> = node_map
            .iter()
            .filter_map(|(name, node_id)| {
                if node_id.is_ground() {
                    None
                } else {
                    Some((name.clone(), node_id.as_u32() as usize - 1))
                }
            })
            .collect();

        for meas in measurements {
            let meas_result = MeasureEvaluator::eval_ac(meas, &result, &mna_node_map);
            if let Some(value) = meas_result.value {
                println!("{} = {:12.6e}", meas_result.name, value);
            } else if let Some(err) = meas_result.error {
                println!("{} = FAILED ({})", meas_result.name, err);
            }
        }
        println!();
    }

    println!();
    Ok(())
}
