//! Noise analysis runner.

use anyhow::Result;
use num_complex::Complex;
use std::collections::HashMap;

use nalgebra::DVector;
use spicier_core::netlist::AcDeviceInfo;
use spicier_core::NodeId;
use spicier_parser::AcSweepType;
use spicier_solver::{
    ComplexMna, DcSolution, NoiseConfig, NoiseSource, NoiseStamper, NoiseSweepType,
    compute_noise, ConvergenceCriteria, solve_newton_raphson,
};

use crate::stampers::NetlistNonlinearStamper;

/// Netlist stamper for noise analysis.
pub struct NetlistNoiseStamper<'a> {
    pub netlist: &'a spicier_core::Netlist,
    /// DC operating point solution (needed for bias-dependent noise).
    pub dc_solution: &'a DcSolution,
}

impl<'a> NoiseStamper for NetlistNoiseStamper<'a> {
    fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
        // Stamp the small-signal AC circuit
        for device in self.netlist.devices().iter() {
            let ac_info = device.ac_info_at(&self.dc_solution.node_voltages);
            stamp_ac_device(mna, &ac_info, omega);
        }
    }

    fn noise_sources(&self) -> Vec<NoiseSource> {
        let mut sources = Vec::new();

        // Iterate through circuit devices and create noise sources
        for device in self.netlist.devices().iter() {
            let ac_info = device.ac_info_at(&self.dc_solution.node_voltages);
            let name = device.device_name();

            match ac_info {
                AcDeviceInfo::Resistor { node_pos, node_neg, conductance } => {
                    // Thermal noise from resistor
                    if conductance > 0.0 {
                        let resistance = 1.0 / conductance;
                        sources.push(NoiseSource::thermal(
                            name,
                            node_pos,
                            node_neg,
                            resistance,
                        ));
                    }
                }
                AcDeviceInfo::Diode { node_pos, node_neg, gd } => {
                    // Shot noise from diode
                    // The diode current can be estimated from gd ≈ Id/Vt
                    // where Vt ≈ 26mV at room temperature
                    let vt: f64 = 0.026;
                    let id = gd * vt;
                    if id.abs() > 1e-15 {
                        sources.push(NoiseSource::shot(
                            format!("{}_shot", name),
                            node_pos,
                            node_neg,
                            id,
                        ));
                    }
                }
                AcDeviceInfo::Mosfet { drain, source, gds, .. } => {
                    // MOSFET thermal noise from channel (simplified)
                    // For simplicity, model as equivalent resistance Req = 1/gds
                    if gds > 0.0 {
                        sources.push(NoiseSource::thermal(
                            format!("{}_chan", name),
                            drain,
                            source,
                            1.0 / gds,
                        ));
                    }
                }
                AcDeviceInfo::Jfet { drain, source, gds, .. } => {
                    // JFET thermal noise from channel
                    if gds > 0.0 {
                        sources.push(NoiseSource::thermal(
                            format!("{}_chan", name),
                            drain,
                            source,
                            1.0 / gds,
                        ));
                    }
                }
                AcDeviceInfo::Bjt { collector, base, emitter, gm, gpi, .. } => {
                    // BJT shot noise from base and collector currents
                    // Ic ≈ gm * Vt, Ib ≈ gpi * Vt (rough estimate)
                    let vt: f64 = 0.026;
                    let ic = gm * vt;
                    let ib = gpi * vt;

                    if ic.abs() > 1e-15 {
                        sources.push(NoiseSource::shot(
                            format!("{}_ic", name),
                            collector,
                            emitter,
                            ic,
                        ));
                    }
                    if ib.abs() > 1e-15 {
                        sources.push(NoiseSource::shot(
                            format!("{}_ib", name),
                            base,
                            emitter,
                            ib,
                        ));
                    }
                }
                _ => {}
            }
        }

        sources
    }

    fn num_nodes(&self) -> usize {
        self.netlist.num_nodes()
    }

    fn num_vsources(&self) -> usize {
        self.netlist.num_current_vars()
    }

    fn input_gain(
        &self,
        omega: f64,
        _input_source_idx: usize,
        output_node: usize,
        output_ref_node: Option<usize>,
    ) -> spicier_solver::Result<Complex<f64>> {
        // Compute AC gain from input source to output
        let mut mna = ComplexMna::new(self.num_nodes(), self.num_vsources());
        self.stamp_ac(&mut mna, omega);

        // Solve AC system
        let solution = spicier_solver::linear::solve_sparse_complex(
            mna.size(),
            &mna.triplets,
            mna.rhs(),
        )?;

        let v_out = if let Some(ref_node) = output_ref_node {
            solution[output_node] - solution[ref_node]
        } else {
            solution[output_node]
        };

        // Assuming 1V input source, gain = Vout/1 = Vout
        Ok(v_out)
    }
}

/// Stamp a device into the complex AC MNA system.
fn stamp_ac_device(mna: &mut ComplexMna, info: &AcDeviceInfo, omega: f64) {
    match info {
        AcDeviceInfo::Resistor { node_pos, node_neg, conductance } => {
            mna.stamp_conductance(*node_pos, *node_neg, *conductance);
        }
        AcDeviceInfo::Capacitor { node_pos, node_neg, capacitance } => {
            let yc = Complex::new(0.0, omega * capacitance);
            mna.stamp_admittance(*node_pos, *node_neg, yc);
        }
        AcDeviceInfo::Inductor { node_pos, node_neg, inductance, branch_idx } => {
            mna.stamp_inductor(*node_pos, *node_neg, *branch_idx, omega, *inductance);
        }
        AcDeviceInfo::VoltageSource { node_pos, node_neg, branch_idx, ac_mag } => {
            mna.stamp_voltage_source(*node_pos, *node_neg, *branch_idx, Complex::new(*ac_mag, 0.0));
        }
        AcDeviceInfo::CurrentSource { node_pos, node_neg, ac_mag } => {
            mna.stamp_current_source(*node_pos, *node_neg, Complex::new(*ac_mag, 0.0));
        }
        AcDeviceInfo::Diode { node_pos, node_neg, gd } => {
            mna.stamp_conductance(*node_pos, *node_neg, *gd);
        }
        AcDeviceInfo::Mosfet { drain, gate, source, gds, gm } => {
            // Stamp output conductance gds between drain and source
            mna.stamp_conductance(*drain, *source, *gds);
            // Stamp transconductance gm as VCCS from gate-source to drain-source
            mna.stamp_vccs(*drain, *source, *gate, *source, *gm);
        }
        AcDeviceInfo::Jfet { drain, gate, source, gds, gm } => {
            mna.stamp_conductance(*drain, *source, *gds);
            mna.stamp_vccs(*drain, *source, *gate, *source, *gm);
        }
        AcDeviceInfo::Bjt { collector, base, emitter, gm, go, gpi } => {
            // Input conductance gpi between base and emitter
            mna.stamp_conductance(*base, *emitter, *gpi);
            // Output conductance go between collector and emitter
            mna.stamp_conductance(*collector, *emitter, *go);
            // Transconductance gm as VCCS from base-emitter to collector-emitter
            mna.stamp_vccs(*collector, *emitter, *base, *emitter, *gm);
        }
        // Skip controlled sources and mutual inductance for now (they don't contribute noise)
        AcDeviceInfo::Vcvs { .. } |
        AcDeviceInfo::Vccs { .. } |
        AcDeviceInfo::Cccs { .. } |
        AcDeviceInfo::Ccvs { .. } |
        AcDeviceInfo::MutualInductance { .. } |
        AcDeviceInfo::None => {}
        // Handle any future variants (non-exhaustive enum)
        _ => {}
    }
}

/// Run noise analysis.
pub fn run_noise_analysis(
    netlist: &spicier_core::Netlist,
    output_node: &str,
    output_ref_node: Option<&str>,
    input_source: &str,
    sweep_type: AcSweepType,
    num_points: usize,
    fstart: f64,
    fstop: f64,
    node_map: &HashMap<String, NodeId>,
) -> Result<()> {
    println!(
        "Noise Analysis (.NOISE V({}{}) {} {} {} {} {})",
        output_node,
        output_ref_node.map(|r| format!(",{}", r)).unwrap_or_default(),
        input_source,
        match sweep_type {
            AcSweepType::Dec => "DEC",
            AcSweepType::Oct => "OCT",
            AcSweepType::Lin => "LIN",
            _ => "DEC", // Default for future variants
        },
        num_points,
        fstart,
        fstop
    );
    println!("==========================================");
    println!();

    // First compute DC operating point
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
        spicier_solver::solve_dc(&mna)
            .map_err(|e| anyhow::anyhow!("DC solver error: {}", e))?
    };

    // Look up node indices (try original case first, then uppercase for compatibility)
    let output_node_id = node_map
        .get(output_node)
        .or_else(|| node_map.get(&output_node.to_uppercase()))
        .ok_or_else(|| anyhow::anyhow!("Output node '{}' not found", output_node))?;

    let output_idx = if output_node_id.is_ground() {
        return Err(anyhow::anyhow!("Output node cannot be ground"));
    } else {
        output_node_id.as_u32() as usize - 1
    };

    let output_ref_idx = if let Some(ref_node) = output_ref_node {
        let ref_id = node_map
            .get(ref_node)
            .or_else(|| node_map.get(&ref_node.to_uppercase()))
            .ok_or_else(|| anyhow::anyhow!("Reference node '{}' not found", ref_node))?;
        if ref_id.is_ground() {
            None
        } else {
            Some(ref_id.as_u32() as usize - 1)
        }
    } else {
        None
    };

    // Find input source index
    let input_source_upper = input_source.to_uppercase();
    let input_source_idx = netlist
        .devices()
        .iter()
        .enumerate()
        .find(|(_, d)| {
            let name = d.device_name().to_uppercase();
            name == input_source_upper && name.starts_with('V')
        })
        .map(|(i, _)| i)
        .ok_or_else(|| anyhow::anyhow!("Input source '{}' not found", input_source))?;

    // Build noise configuration
    let noise_sweep_type = match sweep_type {
        AcSweepType::Dec => NoiseSweepType::Decade,
        AcSweepType::Oct => NoiseSweepType::Octave,
        AcSweepType::Lin => NoiseSweepType::Linear,
        _ => NoiseSweepType::Decade, // Default for future variants
    };

    let config = NoiseConfig {
        output_node: output_idx,
        output_ref_node: output_ref_idx,
        input_source_idx: Some(input_source_idx),
        fstart,
        fstop,
        num_points,
        sweep_type: noise_sweep_type,
        temperature: 300.0, // 27°C
    };

    // Create stamper and run analysis
    let stamper = NetlistNoiseStamper {
        netlist,
        dc_solution: &dc_solution,
    };

    let result = compute_noise(&stamper, &config)
        .map_err(|e| anyhow::anyhow!("Noise analysis error: {}", e))?;

    // Print results
    println!("Frequency (Hz)    Output Noise (V/√Hz)    Input Noise (V/√Hz)    Equiv Rn (Ω)");
    println!("{}", "-".repeat(78));

    for i in 0..result.frequencies.len() {
        let freq = result.frequencies[i];
        let out_noise = result.output_noise[i];
        let in_noise = if !result.input_noise.is_empty() {
            result.input_noise[i]
        } else {
            0.0
        };
        let rn = result.equiv_input_noise_resistance[i];

        println!(
            "{:>14.4e}    {:>18.4e}    {:>18.4e}    {:>12.2}",
            freq, out_noise, in_noise, rn
        );
    }

    println!();

    // Print noise contributions at a representative frequency (geometric mean)
    let mid_freq = (fstart * fstop).sqrt();
    println!("Noise Contributions at {:.2e} Hz:", mid_freq);
    println!("{}", "-".repeat(50));

    // Find the index closest to mid frequency
    let mid_idx = result.frequencies
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - mid_freq).abs()).partial_cmp(&((*b - mid_freq).abs())).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Sort contributions by percentage at mid frequency
    let mut sorted_contribs: Vec<_> = result.contributions.iter().collect();
    sorted_contribs.sort_by(|a, b| {
        b.contribution_percent[mid_idx]
            .partial_cmp(&a.contribution_percent[mid_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for contrib in sorted_contribs.iter().take(10) {
        let percent = contrib.contribution_percent[mid_idx];
        if percent > 0.01 {
            println!(
                "  {:<20} {:>6.2}%",
                contrib.source_name, percent
            );
        }
    }

    println!();

    // Compute integrated noise over the frequency range
    let integrated = result.integrated_noise(fstart, fstop);
    println!("Integrated Output Noise ({:.0} Hz - {:.0} Hz): {:.4e} V RMS", fstart, fstop, integrated);
    println!();
    println!("Noise analysis complete.");
    println!();

    Ok(())
}
