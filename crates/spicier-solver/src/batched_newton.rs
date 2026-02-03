//! Batched Newton-Raphson solver using SIMD-friendly device evaluation.
//!
//! This module provides a Newton-Raphson solver that uses batched device
//! evaluation for improved performance on circuits with many nonlinear devices.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_devices::{BatchMosfetType, DiodeBatch, MosfetBatch};
use spicier_simd::SimdCapability;

use crate::error::Result;
use crate::linear::{CachedSparseLu, SPARSE_THRESHOLD, solve_dense};
use crate::newton::{ConvergenceCriteria, NrResult};
use crate::parallel::{ParallelTripletAccumulator, parallel_ranges, stamp_conductance_triplets};

/// Linear device stamp callback for batched Newton-Raphson.
///
/// Called once per iteration to stamp all linear devices.
pub trait LinearStamper: Send + Sync {
    /// Stamp all linear devices into the MNA system.
    fn stamp_linear(&self, mna: &mut MnaSystem);
}

/// Batched nonlinear device collection for Newton-Raphson.
///
/// Contains device batches extracted from a circuit for SIMD-accelerated evaluation.
#[derive(Debug, Clone)]
pub struct BatchedNonlinearDevices {
    /// Batched diodes.
    pub diodes: DiodeBatch,
    /// Batched MOSFETs.
    pub mosfets: MosfetBatch,
    /// SIMD capability for evaluation dispatch.
    pub capability: SimdCapability,
    /// Pre-allocated evaluation buffers.
    buffers: EvaluationBuffers,
}

/// Pre-allocated buffers for device evaluation.
#[derive(Debug, Clone, Default)]
struct EvaluationBuffers {
    // Diode buffers
    diode_id: Vec<f64>,
    diode_gd: Vec<f64>,
    diode_ieq: Vec<f64>,
    // MOSFET buffers
    mosfet_ids: Vec<f64>,
    mosfet_gds: Vec<f64>,
    mosfet_gm: Vec<f64>,
    mosfet_ieq: Vec<f64>,
}

impl BatchedNonlinearDevices {
    /// Create a new batched device collection.
    pub fn new() -> Self {
        Self {
            diodes: DiodeBatch::new(),
            mosfets: MosfetBatch::new(),
            capability: SimdCapability::detect(),
            buffers: EvaluationBuffers::default(),
        }
    }

    /// Add a diode to the batch.
    ///
    /// # Arguments
    /// * `is` - Saturation current
    /// * `n` - Emission coefficient
    /// * `node_pos` - Anode node index (None for ground)
    /// * `node_neg` - Cathode node index (None for ground)
    pub fn add_diode(&mut self, is: f64, n: f64, node_pos: Option<usize>, node_neg: Option<usize>) {
        self.diodes.push(is, n, node_pos, node_neg);
    }

    /// Add a MOSFET to the batch.
    ///
    /// # Arguments
    /// * `is_nmos` - true for NMOS, false for PMOS
    /// * `vth` - Threshold voltage magnitude
    /// * `beta` - Transconductance parameter (kp * W / L)
    /// * `lambda` - Channel-length modulation
    /// * `node_drain/gate/source` - Node indices (None for ground)
    #[allow(clippy::too_many_arguments)]
    pub fn add_mosfet(
        &mut self,
        is_nmos: bool,
        vth: f64,
        beta: f64,
        lambda: f64,
        node_drain: Option<usize>,
        node_gate: Option<usize>,
        node_source: Option<usize>,
    ) {
        let mos_type = if is_nmos {
            BatchMosfetType::Nmos
        } else {
            BatchMosfetType::Pmos
        };
        self.mosfets.push(
            mos_type,
            vth,
            beta,
            lambda,
            node_drain,
            node_gate,
            node_source,
        );
    }

    /// Finalize batches (pad for SIMD) and allocate evaluation buffers.
    pub fn finalize(&mut self) {
        self.diodes.finalize();
        self.mosfets.finalize();

        // Allocate buffers
        if self.diodes.count > 0 {
            self.buffers.diode_id = vec![0.0; self.diodes.count];
            self.buffers.diode_gd = vec![0.0; self.diodes.count];
            self.buffers.diode_ieq = vec![0.0; self.diodes.count];
        }
        if self.mosfets.count > 0 {
            self.buffers.mosfet_ids = vec![0.0; self.mosfets.count];
            self.buffers.mosfet_gds = vec![0.0; self.mosfets.count];
            self.buffers.mosfet_gm = vec![0.0; self.mosfets.count];
            self.buffers.mosfet_ieq = vec![0.0; self.mosfets.count];
        }
    }

    /// Check if there are any nonlinear devices.
    pub fn has_devices(&self) -> bool {
        self.diodes.count > 0 || self.mosfets.count > 0
    }

    /// Get total count of nonlinear devices.
    pub fn device_count(&self) -> usize {
        self.diodes.count + self.mosfets.count
    }

    /// Evaluate all devices and stamp into the MNA system.
    ///
    /// This is the core batched evaluation that replaces individual device stamping.
    pub fn evaluate_and_stamp(&mut self, solution: &[f64], mna: &mut MnaSystem) {
        // Evaluate diodes
        if self.diodes.count > 0 {
            self.diodes.evaluate_linearized_batch(
                solution,
                &mut self.buffers.diode_id,
                &mut self.buffers.diode_gd,
                &mut self.buffers.diode_ieq,
                self.capability,
            );

            // Stamp diodes
            for i in 0..self.diodes.count {
                let node_pos = if self.diodes.node_pos[i] == usize::MAX {
                    None
                } else {
                    Some(self.diodes.node_pos[i])
                };
                let node_neg = if self.diodes.node_neg[i] == usize::MAX {
                    None
                } else {
                    Some(self.diodes.node_neg[i])
                };

                // Stamp conductance
                mna.stamp_conductance(node_pos, node_neg, self.buffers.diode_gd[i]);
                // Stamp equivalent current source
                mna.stamp_current_source(node_pos, node_neg, self.buffers.diode_ieq[i]);
            }
        }

        // Evaluate MOSFETs
        if self.mosfets.count > 0 {
            self.mosfets.evaluate_linearized_batch(
                solution,
                &mut self.buffers.mosfet_ids,
                &mut self.buffers.mosfet_gds,
                &mut self.buffers.mosfet_gm,
                &mut self.buffers.mosfet_ieq,
                self.capability,
            );

            // Stamp MOSFETs
            for i in 0..self.mosfets.count {
                let d = if self.mosfets.node_drain[i] == usize::MAX {
                    None
                } else {
                    Some(self.mosfets.node_drain[i])
                };
                let g = if self.mosfets.node_gate[i] == usize::MAX {
                    None
                } else {
                    Some(self.mosfets.node_gate[i])
                };
                let s = if self.mosfets.node_source[i] == usize::MAX {
                    None
                } else {
                    Some(self.mosfets.node_source[i])
                };

                let gds = self.buffers.mosfet_gds[i];
                let gm = self.buffers.mosfet_gm[i];
                let ieq = self.buffers.mosfet_ieq[i];

                // Stamp gds (drain-source conductance)
                mna.stamp_conductance(d, s, gds);

                // Stamp gm (transconductance) as VCCS
                if let Some(di) = d {
                    if let Some(gi) = g {
                        mna.add_element(di, gi, gm);
                    }
                    if let Some(si) = s {
                        mna.add_element(di, si, -gm);
                    }
                }
                if let Some(si) = s {
                    if let Some(gi) = g {
                        mna.add_element(si, gi, -gm);
                    }
                    if let Some(si2) = s {
                        mna.add_element(si, si2, gm);
                    }
                }

                // Stamp equivalent current source
                mna.stamp_current_source(d, s, -ieq);
            }
        }
    }
}

impl Default for BatchedNonlinearDevices {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchedNonlinearDevices {
    /// Evaluate all devices and stamp into triplet buffer (for parallel assembly).
    ///
    /// This variant outputs to a triplet buffer instead of MnaSystem,
    /// enabling parallel assembly when combined with other device types.
    ///
    /// # Arguments
    /// * `solution` - Current solution vector
    /// * `triplets` - Output triplet buffer to append to
    /// * `rhs` - Output RHS buffer to modify
    pub fn evaluate_and_stamp_triplets(
        &mut self,
        solution: &[f64],
        triplets: &mut Vec<(usize, usize, f64)>,
        rhs: &mut [f64],
    ) {
        // Evaluate diodes
        if self.diodes.count > 0 {
            self.diodes.evaluate_linearized_batch(
                solution,
                &mut self.buffers.diode_id,
                &mut self.buffers.diode_gd,
                &mut self.buffers.diode_ieq,
                self.capability,
            );

            for i in 0..self.diodes.count {
                let node_pos = if self.diodes.node_pos[i] == usize::MAX {
                    None
                } else {
                    Some(self.diodes.node_pos[i])
                };
                let node_neg = if self.diodes.node_neg[i] == usize::MAX {
                    None
                } else {
                    Some(self.diodes.node_neg[i])
                };

                // Stamp conductance to triplets
                stamp_conductance_triplets(triplets, node_pos, node_neg, self.buffers.diode_gd[i]);

                // Stamp equivalent current source to RHS
                let ieq = self.buffers.diode_ieq[i];
                if let Some(p) = node_pos {
                    rhs[p] -= ieq;
                }
                if let Some(n) = node_neg {
                    rhs[n] += ieq;
                }
            }
        }

        // Evaluate MOSFETs
        if self.mosfets.count > 0 {
            self.mosfets.evaluate_linearized_batch(
                solution,
                &mut self.buffers.mosfet_ids,
                &mut self.buffers.mosfet_gds,
                &mut self.buffers.mosfet_gm,
                &mut self.buffers.mosfet_ieq,
                self.capability,
            );

            for i in 0..self.mosfets.count {
                let d = if self.mosfets.node_drain[i] == usize::MAX {
                    None
                } else {
                    Some(self.mosfets.node_drain[i])
                };
                let g = if self.mosfets.node_gate[i] == usize::MAX {
                    None
                } else {
                    Some(self.mosfets.node_gate[i])
                };
                let s = if self.mosfets.node_source[i] == usize::MAX {
                    None
                } else {
                    Some(self.mosfets.node_source[i])
                };

                let gds = self.buffers.mosfet_gds[i];
                let gm = self.buffers.mosfet_gm[i];
                let ieq = self.buffers.mosfet_ieq[i];

                // Stamp gds
                stamp_conductance_triplets(triplets, d, s, gds);

                // Stamp gm as VCCS
                if let Some(di) = d {
                    if let Some(gi) = g {
                        triplets.push((di, gi, gm));
                    }
                    if let Some(si) = s {
                        triplets.push((di, si, -gm));
                    }
                }
                if let Some(si) = s {
                    if let Some(gi) = g {
                        triplets.push((si, gi, -gm));
                    }
                    if let Some(si2) = s {
                        triplets.push((si, si2, gm));
                    }
                }

                // Stamp equivalent current source
                if let Some(di) = d {
                    rhs[di] += ieq;
                }
                if let Some(si) = s {
                    rhs[si] -= ieq;
                }
            }
        }
    }

    /// Parallel batch evaluation for circuits with many devices.
    ///
    /// Splits devices across threads for evaluation, then merges results.
    /// Most beneficial when device_count > 1000.
    pub fn evaluate_parallel(
        &mut self,
        solution: &[f64],
        accumulator: &ParallelTripletAccumulator,
        rhs: &mut [f64],
    ) {
        let num_threads = accumulator.num_threads();

        // For small device counts, just use sequential evaluation
        if self.device_count() < 100 || num_threads <= 1 {
            let mut buf = accumulator.get_buffer(0);
            self.evaluate_and_stamp_triplets(solution, &mut buf, rhs);
            return;
        }

        // Split diode evaluation across threads
        if self.diodes.count > 0 {
            // First, evaluate all diodes (this is the SIMD part)
            self.diodes.evaluate_linearized_batch(
                solution,
                &mut self.buffers.diode_id,
                &mut self.buffers.diode_gd,
                &mut self.buffers.diode_ieq,
                self.capability,
            );

            // Then stamp in parallel using ranges
            let ranges = parallel_ranges(self.diodes.count, num_threads);
            for (thread_id, (start, end)) in ranges.into_iter().enumerate() {
                let mut buf = accumulator.get_buffer(thread_id);
                for i in start..end {
                    let node_pos = if self.diodes.node_pos[i] == usize::MAX {
                        None
                    } else {
                        Some(self.diodes.node_pos[i])
                    };
                    let node_neg = if self.diodes.node_neg[i] == usize::MAX {
                        None
                    } else {
                        Some(self.diodes.node_neg[i])
                    };

                    stamp_conductance_triplets(
                        &mut buf,
                        node_pos,
                        node_neg,
                        self.buffers.diode_gd[i],
                    );
                }
            }

            // RHS stamping is sequential (usually much smaller)
            for i in 0..self.diodes.count {
                let ieq = self.buffers.diode_ieq[i];
                if self.diodes.node_pos[i] != usize::MAX {
                    rhs[self.diodes.node_pos[i]] -= ieq;
                }
                if self.diodes.node_neg[i] != usize::MAX {
                    rhs[self.diodes.node_neg[i]] += ieq;
                }
            }
        }

        // Similar for MOSFETs
        if self.mosfets.count > 0 {
            self.mosfets.evaluate_linearized_batch(
                solution,
                &mut self.buffers.mosfet_ids,
                &mut self.buffers.mosfet_gds,
                &mut self.buffers.mosfet_gm,
                &mut self.buffers.mosfet_ieq,
                self.capability,
            );

            let ranges = parallel_ranges(self.mosfets.count, num_threads);
            for (thread_id, (start, end)) in ranges.into_iter().enumerate() {
                let mut buf = accumulator.get_buffer(thread_id);
                for i in start..end {
                    let d = if self.mosfets.node_drain[i] == usize::MAX {
                        None
                    } else {
                        Some(self.mosfets.node_drain[i])
                    };
                    let g = if self.mosfets.node_gate[i] == usize::MAX {
                        None
                    } else {
                        Some(self.mosfets.node_gate[i])
                    };
                    let s = if self.mosfets.node_source[i] == usize::MAX {
                        None
                    } else {
                        Some(self.mosfets.node_source[i])
                    };

                    stamp_conductance_triplets(&mut buf, d, s, self.buffers.mosfet_gds[i]);

                    let gm = self.buffers.mosfet_gm[i];
                    if let Some(di) = d {
                        if let Some(gi) = g {
                            buf.push((di, gi, gm));
                        }
                        if let Some(si) = s {
                            buf.push((di, si, -gm));
                        }
                    }
                    if let Some(si) = s {
                        if let Some(gi) = g {
                            buf.push((si, gi, -gm));
                        }
                        if let Some(si2) = s {
                            buf.push((si, si2, gm));
                        }
                    }
                }
            }

            // RHS stamping
            for i in 0..self.mosfets.count {
                let ieq = self.buffers.mosfet_ieq[i];
                if self.mosfets.node_drain[i] != usize::MAX {
                    rhs[self.mosfets.node_drain[i]] += ieq;
                }
                if self.mosfets.node_source[i] != usize::MAX {
                    rhs[self.mosfets.node_source[i]] -= ieq;
                }
            }
        }
    }
}

/// Solve a nonlinear system using batched Newton-Raphson iteration.
///
/// This is a more efficient variant of `solve_newton_raphson` that uses
/// SIMD-accelerated batch evaluation of nonlinear devices.
///
/// # Arguments
/// * `num_nodes` - Number of nodes (excluding ground)
/// * `num_vsources` - Number of voltage sources
/// * `linear_stamper` - Callback to stamp linear devices
/// * `nonlinear_devices` - Batched nonlinear devices
/// * `criteria` - Convergence criteria
/// * `initial_guess` - Optional initial solution guess
pub fn solve_batched_newton_raphson(
    num_nodes: usize,
    num_vsources: usize,
    linear_stamper: &dyn LinearStamper,
    nonlinear_devices: &mut BatchedNonlinearDevices,
    criteria: &ConvergenceCriteria,
    initial_guess: Option<&DVector<f64>>,
) -> Result<NrResult> {
    let size = num_nodes + num_vsources;

    // Initialize solution
    let mut solution = match initial_guess {
        Some(guess) => guess.clone(),
        None => DVector::zeros(size),
    };

    let mut mna = MnaSystem::new(num_nodes, num_vsources);

    // Cached sparse solver
    let mut cached_solver: Option<CachedSparseLu> = None;

    for iteration in 0..criteria.max_iterations {
        // Clear MNA
        mna.clear();

        // Stamp linear devices
        linear_stamper.stamp_linear(&mut mna);

        // Evaluate and stamp nonlinear devices (batched)
        nonlinear_devices.evaluate_and_stamp(solution.as_slice(), &mut mna);

        // Solve the linearized system
        let new_solution = if size >= SPARSE_THRESHOLD {
            let solver = match &cached_solver {
                Some(s) => s,
                None => {
                    cached_solver = Some(CachedSparseLu::new(size, &mna.triplets)?);
                    cached_solver.as_ref().unwrap()
                }
            };
            solver.solve(&mna.triplets, mna.rhs())?
        } else {
            solve_dense(&mna.to_dense_matrix(), mna.rhs())?
        };

        // Check convergence
        let converged = check_convergence(&solution, &new_solution, num_nodes, criteria);

        solution = new_solution;

        if converged {
            return Ok(NrResult {
                solution,
                iterations: iteration + 1,
                converged: true,
            });
        }
    }

    Ok(NrResult {
        solution,
        iterations: criteria.max_iterations,
        converged: false,
    })
}

/// Check if the solution has converged.
fn check_convergence(
    old: &DVector<f64>,
    new: &DVector<f64>,
    num_nodes: usize,
    criteria: &ConvergenceCriteria,
) -> bool {
    // Check voltage convergence
    for i in 0..num_nodes {
        let delta = (new[i] - old[i]).abs();
        let tol = criteria.v_reltol * new[i].abs().max(old[i].abs()) + criteria.v_abstol;
        if delta > tol {
            return false;
        }
    }

    // Check current convergence
    for i in num_nodes..old.len() {
        let delta = (new[i] - old[i]).abs();
        let tol = criteria.v_reltol * new[i].abs().max(old[i].abs()) + criteria.i_abstol;
        if delta > tol {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear stamper that stamps a voltage source and resistor.
    struct DiodeCircuitLinearStamper {
        v_source: f64,
        resistance: f64,
    }

    impl LinearStamper for DiodeCircuitLinearStamper {
        fn stamp_linear(&self, mna: &mut MnaSystem) {
            // Voltage source at node 0
            mna.stamp_voltage_source(Some(0), None, 0, self.v_source);
            // Resistor from node 0 to node 1
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(0), Some(1), g);
        }
    }

    #[test]
    fn test_batched_newton_diode() {
        let linear = DiodeCircuitLinearStamper {
            v_source: 5.0,
            resistance: 1000.0,
        };

        let mut devices = BatchedNonlinearDevices::new();
        // Diode from node 1 to ground
        devices.add_diode(1e-14, 1.0, Some(1), None);
        devices.finalize();

        let criteria = ConvergenceCriteria::default();
        let result = solve_batched_newton_raphson(2, 1, &linear, &mut devices, &criteria, None)
            .expect("Should converge");

        assert!(result.converged, "Should converge");
        assert!(result.iterations < 50, "Should converge quickly");

        // V(0) should be 5V
        assert!(
            (result.solution[0] - 5.0).abs() < 1e-6,
            "V(0) = {} (expected 5.0)",
            result.solution[0]
        );

        // V(1) should be ~0.6-0.7V (diode drop)
        let vd = result.solution[1];
        assert!(vd > 0.5 && vd < 0.8, "V(diode) = {} (expected 0.5-0.8)", vd);
    }

    #[test]
    fn test_batched_vs_single_diode() {
        // Test that batched evaluation gives same results as single device
        let linear = DiodeCircuitLinearStamper {
            v_source: 5.0,
            resistance: 1000.0,
        };

        let mut devices = BatchedNonlinearDevices::new();
        devices.add_diode(1e-14, 1.0, Some(1), None);
        devices.finalize();

        let criteria = ConvergenceCriteria::default();
        let batched_result =
            solve_batched_newton_raphson(2, 1, &linear, &mut devices, &criteria, None)
                .expect("Should converge");

        // Use regular NR with simple stamper for comparison
        use crate::newton::{NonlinearStamper, solve_newton_raphson};

        struct SimpleStamper {
            v_source: f64,
            resistance: f64,
            is: f64,
            nvt: f64,
        }

        impl NonlinearStamper for SimpleStamper {
            fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
                mna.stamp_voltage_source(Some(0), None, 0, self.v_source);
                let g = 1.0 / self.resistance;
                mna.stamp_conductance(Some(0), Some(1), g);

                // Diode at node 1
                let vd = solution[1];
                let vd_limited = if vd > 0.8 { 0.8 + (vd - 0.8) * 0.1 } else { vd };
                let exp_term = (vd_limited / self.nvt).exp();
                let id = self.is * (exp_term - 1.0);
                let gd = (self.is * exp_term / self.nvt).max(1e-12);
                let ieq = id - gd * vd_limited;

                mna.stamp_conductance(Some(1), None, gd);
                mna.stamp_current_source(Some(1), None, ieq);
            }
        }

        let simple = SimpleStamper {
            v_source: 5.0,
            resistance: 1000.0,
            is: 1e-14,
            nvt: 0.02585,
        };

        let simple_result =
            solve_newton_raphson(2, 1, &simple, &criteria, None).expect("Should converge");

        // Results should be very close (may differ slightly due to voltage limiting)
        assert!(
            (batched_result.solution[0] - simple_result.solution[0]).abs() < 1e-4,
            "V(0) differs: {} vs {}",
            batched_result.solution[0],
            simple_result.solution[0]
        );
        assert!(
            (batched_result.solution[1] - simple_result.solution[1]).abs() < 0.05,
            "V(1) differs: {} vs {}",
            batched_result.solution[1],
            simple_result.solution[1]
        );
    }

    #[test]
    fn test_batched_multiple_diodes() {
        // Circuit with multiple diodes in parallel
        struct MultiDiodeLinear;

        impl LinearStamper for MultiDiodeLinear {
            fn stamp_linear(&self, mna: &mut MnaSystem) {
                // 5V source at node 0
                mna.stamp_voltage_source(Some(0), None, 0, 5.0);
                // 100 ohm resistor from node 0 to node 1
                mna.stamp_conductance(Some(0), Some(1), 0.01);
            }
        }

        let mut devices = BatchedNonlinearDevices::new();
        // 10 diodes in parallel from node 1 to ground
        for _ in 0..10 {
            devices.add_diode(1e-14, 1.0, Some(1), None);
        }
        devices.finalize();

        let criteria = ConvergenceCriteria::default();
        let result =
            solve_batched_newton_raphson(2, 1, &MultiDiodeLinear, &mut devices, &criteria, None)
                .expect("Should converge");

        assert!(result.converged);
        // With 10 parallel diodes, the voltage should still be around diode drop
        let vd = result.solution[1];
        assert!(vd > 0.4 && vd < 0.9, "V(diodes) = {}", vd);
    }

    #[test]
    fn test_batched_mosfet() {
        // Simple MOSFET circuit: Vdd -> R -> drain, gate tied to Vdd
        struct MosfetLinear;

        impl LinearStamper for MosfetLinear {
            fn stamp_linear(&self, mna: &mut MnaSystem) {
                // 5V Vdd at node 0
                mna.stamp_voltage_source(Some(0), None, 0, 5.0);
                // 2V Vgate at node 1
                mna.stamp_voltage_source(Some(1), None, 1, 2.0);
                // 1k resistor from Vdd to drain (node 2)
                mna.stamp_conductance(Some(0), Some(2), 0.001);
            }
        }

        let mut devices = BatchedNonlinearDevices::new();
        // NMOS: drain=2, gate=1, source=ground, Vth=0.7, beta=2e-4
        devices.add_mosfet(true, 0.7, 2e-4, 0.0, Some(2), Some(1), None);
        devices.finalize();

        let criteria = ConvergenceCriteria::default();
        let result =
            solve_batched_newton_raphson(3, 2, &MosfetLinear, &mut devices, &criteria, None)
                .expect("Should converge");

        assert!(result.converged);
        // Vgs = 2V, Vth = 0.7V, so MOSFET should be on
        // Drain voltage should be between 0 and 5V
        let vd = result.solution[2];
        assert!((0.0..=5.0).contains(&vd), "V(drain) = {} out of range", vd);
    }
}
