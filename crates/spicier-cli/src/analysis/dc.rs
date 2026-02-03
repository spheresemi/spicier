//! DC operating point and sweep analysis.

use anyhow::Result;
use nalgebra::DVector;
use spicier_core::NodeId;
use spicier_core::mna::MnaSystem;
use spicier_parser::{DcSweepSpec, DcSweepType, Measurement, OutputVariable, parse_full};
use spicier_solver::{
    ConvergenceCriteria, DcSolution, DcSweepParams, MeasureEvaluator, solve_dc, solve_dc_sweep,
    solve_newton_raphson,
};
use std::collections::HashMap;

use crate::output::{get_dc_print_nodes, print_dc_solution};
use crate::stampers::{NestedSweepStamper, NetlistNonlinearStamper, NetlistSweepStamper};

/// Run DC operating point analysis.
pub fn run_dc_op(
    netlist: &spicier_core::Netlist,
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    measurements: &[&Measurement],
) -> Result<()> {
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

    print_dc_solution(netlist, &solution, print_vars, node_map);

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
            let meas_result = MeasureEvaluator::eval_dc(meas, &solution, &mna_node_map);
            if let Some(value) = meas_result.value {
                println!("{} = {:12.6e}", meas_result.name, value);
            } else if let Some(err) = meas_result.error {
                println!("{} = FAILED ({})", meas_result.name, err);
            }
        }
        println!();
    }

    println!("Analysis complete.");
    println!();
    Ok(())
}

/// Run DC sweep analysis (single or nested).
pub fn run_dc_sweep(
    netlist: &spicier_core::Netlist,
    sweeps: &[DcSweepSpec],
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    measurements: &[&Measurement],
) -> Result<()> {
    if sweeps.is_empty() {
        return Err(anyhow::anyhow!("No sweep specifications provided"));
    }

    if sweeps.len() == 1 {
        // Single sweep
        run_single_dc_sweep(netlist, &sweeps[0], print_vars, node_map, measurements)
    } else {
        // Nested sweep (2 variables)
        run_nested_dc_sweep(
            netlist,
            &sweeps[0],
            &sweeps[1],
            print_vars,
            node_map,
            measurements,
        )
    }
}

/// Run single-variable DC sweep.
fn run_single_dc_sweep(
    netlist: &spicier_core::Netlist,
    sweep: &DcSweepSpec,
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    measurements: &[&Measurement],
) -> Result<()> {
    println!(
        "DC Sweep Analysis (.DC {} {} {} {})",
        sweep.source_name, sweep.start, sweep.stop, sweep.step
    );
    println!("==========================================");
    println!();

    let stamper = NetlistSweepStamper {
        netlist,
        source_name: sweep.source_name.clone(),
    };

    let params = DcSweepParams {
        source_name: sweep.source_name.clone(),
        start: sweep.start,
        stop: sweep.stop,
        step: sweep.step,
    };

    let result =
        solve_dc_sweep(&stamper, &params).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?;

    // Determine which nodes to print
    let nodes_to_print = get_dc_print_nodes(print_vars, node_map, netlist.num_nodes());

    // Print header
    print!("{:>12}", sweep.source_name);
    for (name, _) in &nodes_to_print {
        print!("{:>12}", format!("V({})", name));
    }
    println!();

    // Print separator
    let width = 12 * (1 + nodes_to_print.len());
    println!("{}", "-".repeat(width));

    // Print sweep data
    for (sv, sol) in result.sweep_values.iter().zip(result.solutions.iter()) {
        print!("{:>12.4}", sv);
        for (_, node_id) in &nodes_to_print {
            let v = sol.voltage(*node_id);
            print!("{:>12.6}", v);
        }
        println!();
    }

    println!();
    println!("Sweep complete ({} points).", result.sweep_values.len());

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
            let meas_result = MeasureEvaluator::eval_dc_sweep(meas, &result, &mna_node_map);
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

/// Run nested two-variable DC sweep.
fn run_nested_dc_sweep(
    netlist: &spicier_core::Netlist,
    outer_sweep: &DcSweepSpec,
    inner_sweep: &DcSweepSpec,
    print_vars: &[&OutputVariable],
    node_map: &HashMap<String, NodeId>,
    _measurements: &[&Measurement],
) -> Result<()> {
    println!(
        "Nested DC Sweep Analysis (.DC {} {} {} {} {} {} {} {})",
        outer_sweep.source_name,
        outer_sweep.start,
        outer_sweep.stop,
        outer_sweep.step,
        inner_sweep.source_name,
        inner_sweep.start,
        inner_sweep.stop,
        inner_sweep.step
    );
    println!("==========================================");
    println!();

    // Generate sweep values for both sweeps
    let outer_values = generate_sweep_values(outer_sweep);
    let inner_values = generate_sweep_values(inner_sweep);

    let stamper = NestedSweepStamper {
        netlist,
        source_name1: outer_sweep.source_name.clone(),
        source_name2: inner_sweep.source_name.clone(),
    };

    // Determine which nodes to print
    let nodes_to_print = get_dc_print_nodes(print_vars, node_map, netlist.num_nodes());

    // Print header
    print!(
        "{:>12}{:>12}",
        outer_sweep.source_name, inner_sweep.source_name
    );
    for (name, _) in &nodes_to_print {
        print!("{:>12}", format!("V({})", name));
    }
    println!();

    // Print separator
    let width = 12 * (2 + nodes_to_print.len());
    println!("{}", "-".repeat(width));

    let mut total_points = 0;

    // Nested sweep: outer loop is slow, inner loop is fast
    for &outer_val in &outer_values {
        for &inner_val in &inner_values {
            // Stamp and solve for this combination
            let mut mna = MnaSystem::new(netlist.num_nodes(), netlist.num_current_vars());
            stamper.stamp_with_two_sweeps(&mut mna, outer_val, inner_val);

            let sol = solve_dc(&mna).map_err(|e| anyhow::anyhow!("Solver error: {}", e))?;

            // Print results
            print!("{:>12.4}{:>12.4}", outer_val, inner_val);
            for (_, node_id) in &nodes_to_print {
                let v = sol.voltage(*node_id);
                print!("{:>12.6}", v);
            }
            println!();

            total_points += 1;
        }
    }

    println!();
    println!(
        "Nested sweep complete ({} outer x {} inner = {} points).",
        outer_values.len(),
        inner_values.len(),
        total_points
    );
    println!();
    Ok(())
}

/// Generate sweep values for a DC sweep specification.
fn generate_sweep_values(sweep: &DcSweepSpec) -> Vec<f64> {
    let mut values = Vec::new();
    let direction = if sweep.step > 0.0 { 1.0 } else { -1.0 };
    let mut value = sweep.start;
    loop {
        values.push(value);
        value += sweep.step;
        if direction * value > direction * sweep.stop * (1.0 + 1e-10) {
            break;
        }
    }
    values
}

/// Run DC parameter sweep analysis.
///
/// For parameter sweeps, we need to re-parse the netlist for each parameter value
/// because device values are computed from parameters during parsing.
pub fn run_dc_param_sweep(
    netlist_content: &str,
    sweeps: &[DcSweepSpec],
    print_vars: &[&OutputVariable],
    _measurements: &[&Measurement],
) -> Result<()> {
    if sweeps.is_empty() {
        return Err(anyhow::anyhow!("No sweep specifications provided"));
    }

    // For now, we only support single parameter sweeps
    // Nested parameter+source sweeps are more complex
    if sweeps.len() > 1 {
        // Check if we have mixed param/source sweeps
        let param_sweeps: Vec<_> = sweeps
            .iter()
            .filter(|s| s.sweep_type == DcSweepType::Param)
            .collect();
        let source_sweeps: Vec<_> = sweeps
            .iter()
            .filter(|s| s.sweep_type == DcSweepType::Source)
            .collect();

        if !param_sweeps.is_empty() && !source_sweeps.is_empty() {
            return Err(anyhow::anyhow!(
                "Mixed parameter and source sweeps are not yet supported"
            ));
        }

        if param_sweeps.len() > 1 {
            return Err(anyhow::anyhow!(
                "Multiple parameter sweeps are not yet supported"
            ));
        }
    }

    let param_sweep = sweeps
        .iter()
        .find(|s| s.sweep_type == DcSweepType::Param)
        .ok_or_else(|| anyhow::anyhow!("No parameter sweep found"))?;

    println!(
        "DC Parameter Sweep Analysis (.DC PARAM {} {} {} {})",
        param_sweep.source_name, param_sweep.start, param_sweep.stop, param_sweep.step
    );
    println!("==========================================");
    println!();

    let sweep_values = generate_sweep_values(param_sweep);
    let param_name = param_sweep.source_name.to_uppercase();

    // Parse once to get node map and print variable info
    let initial_result =
        parse_full(netlist_content).map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
    let node_map = initial_result.node_map.clone();
    let num_nodes = initial_result.netlist.num_nodes();

    // Determine which nodes to print
    let nodes_to_print = get_dc_print_nodes(print_vars, &node_map, num_nodes);

    // Print header
    print!("{:>12}", param_sweep.source_name);
    for (name, _) in &nodes_to_print {
        print!("{:>12}", format!("V({})", name));
    }
    println!();

    // Print separator
    let width = 12 * (1 + nodes_to_print.len());
    println!("{}", "-".repeat(width));

    // For each sweep value, modify the netlist content and re-parse
    for param_value in &sweep_values {
        // Create modified netlist with updated parameter value
        let modified_content = inject_or_update_param(netlist_content, &param_name, *param_value);

        // Parse the modified netlist
        let result = parse_full(&modified_content)
            .map_err(|e| anyhow::anyhow!("Parse error at param={}: {}", param_value, e))?;

        let netlist = result.netlist;

        // Solve DC operating point
        let solution = if netlist.has_nonlinear_devices() {
            let stamper = NetlistNonlinearStamper { netlist: &netlist };
            let criteria = ConvergenceCriteria::default();
            let nr_result = solve_newton_raphson(
                netlist.num_nodes(),
                netlist.num_current_vars(),
                &stamper,
                &criteria,
                None,
            )
            .map_err(|e| anyhow::anyhow!("Newton-Raphson error: {}", e))?;

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

        // Print results for this sweep point
        print!("{:>12.4}", param_value);
        for (_, node_id) in &nodes_to_print {
            let v = solution.voltage(*node_id);
            print!("{:>12.6}", v);
        }
        println!();
    }

    println!();
    println!("Parameter sweep complete ({} points).", sweep_values.len());
    println!();

    Ok(())
}

/// Inject or update a .PARAM line in the netlist content.
///
/// If the parameter already exists, update its value.
/// Otherwise, add a new .PARAM line at the beginning (after the title).
fn inject_or_update_param(content: &str, param_name: &str, value: f64) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut result_lines = Vec::with_capacity(lines.len() + 1);

    let param_pattern_upper = param_name.to_uppercase();
    let mut found = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim().to_uppercase();

        // Check if this line defines the parameter we want to modify
        if trimmed.starts_with(".PARAM") {
            // Check if this line contains our parameter
            // Look for patterns like "PARAM_NAME=value" or "PARAM_NAME = value"
            if let Some(rest) = trimmed.strip_prefix(".PARAM") {
                let rest = rest.trim();
                // Split by whitespace to handle multiple params on one line
                for part in rest.split_whitespace() {
                    if let Some(eq_pos) = part.find('=') {
                        let name = part[..eq_pos].trim().to_uppercase();
                        if name == param_pattern_upper {
                            // Found the parameter, replace this line with updated value
                            // For simplicity, create a new .PARAM line
                            result_lines.push(format!(".PARAM {}={}", param_name, value));
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    continue; // Skip original line
                }
            }
        }

        // Keep original line
        result_lines.push((*line).to_string());

        // If this is the title line (first non-comment line) and we haven't found the param yet,
        // we'll inject it later
        if i == 0 && !found && !trimmed.starts_with('.') && !trimmed.starts_with('*') {
            // Will inject after title if not found
        }
    }

    // If parameter wasn't found in existing .PARAM lines, add it after the title
    if !found {
        // Find a good insertion point (after title, before first element/command)
        let insert_pos = if !result_lines.is_empty() && !result_lines[0].trim().starts_with('.') {
            1 // After title
        } else {
            0
        };
        result_lines.insert(insert_pos, format!(".PARAM {}={}", param_name, value));
    }

    result_lines.join("\n")
}
