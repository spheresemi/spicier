//! Newton-Raphson nonlinear solver.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;

use crate::error::Result;
use crate::linear::{CachedSparseLu, SPARSE_THRESHOLD, solve_dense};

/// Convergence criteria for Newton-Raphson iteration.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Absolute voltage tolerance (V).
    pub v_abstol: f64,
    /// Relative voltage tolerance.
    pub v_reltol: f64,
    /// Absolute current tolerance (A).
    pub i_abstol: f64,
    /// Maximum iterations before failure.
    pub max_iterations: usize,
    /// Gmin value for initial convergence aid.
    pub gmin: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            v_abstol: 1e-6,
            v_reltol: 1e-3,
            i_abstol: 1e-12,
            max_iterations: 50,
            gmin: 1e-12,
        }
    }
}

/// Callback for stamping nonlinear devices at each iteration.
///
/// Given the current solution vector, this function should:
/// 1. Clear the MNA matrix
/// 2. Stamp all linear devices
/// 3. Stamp all nonlinear devices linearized at the current solution
pub trait NonlinearStamper {
    /// Re-stamp the MNA system for the current solution.
    fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>);
}

/// Result of Newton-Raphson iteration.
#[derive(Debug, Clone)]
pub struct NrResult {
    /// Solution vector.
    pub solution: DVector<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Solve a nonlinear system using Newton-Raphson iteration.
///
/// # Arguments
/// * `num_nodes` - Number of nodes (excluding ground)
/// * `num_vsources` - Number of voltage sources
/// * `stamper` - Callback to stamp the system at each iteration point
/// * `criteria` - Convergence criteria
/// * `initial_guess` - Optional initial solution guess
///
/// For large systems (>= SPARSE_THRESHOLD), uses cached symbolic factorization
/// to speed up repeated solves. The sparsity pattern is determined on the first
/// iteration and reused for all subsequent iterations.
pub fn solve_newton_raphson(
    num_nodes: usize,
    num_vsources: usize,
    stamper: &dyn NonlinearStamper,
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

    // Cached sparse solver (created on first iteration if needed)
    let mut cached_solver: Option<CachedSparseLu> = None;

    for iteration in 0..criteria.max_iterations {
        // Clear and re-stamp at current operating point
        mna.clear();
        stamper.stamp_at(&mut mna, &solution);

        // Solve the linearized system
        let new_solution = if size >= SPARSE_THRESHOLD {
            // Use cached sparse solver for large systems
            let solver = match &cached_solver {
                Some(s) => s,
                None => {
                    // First iteration: create cached solver with symbolic factorization
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

    // Failed to converge - return last solution
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

    // Check current convergence (for voltage source currents)
    for i in num_nodes..old.len() {
        let delta = (new[i] - old[i]).abs();
        let tol = criteria.v_reltol * new[i].abs().max(old[i].abs()) + criteria.i_abstol;
        if delta > tol {
            return false;
        }
    }

    true
}

/// Callback for stamping nonlinear devices with source scaling.
///
/// Extends [`NonlinearStamper`] with the ability to scale independent sources
/// for the source stepping convergence aid.
pub trait ScaledNonlinearStamper: NonlinearStamper {
    /// Re-stamp the MNA system with sources scaled by `source_factor`.
    ///
    /// `source_factor` ranges from 0.0 to 1.0, where 1.0 means full source values.
    fn stamp_at_scaled(&self, mna: &mut MnaSystem, solution: &DVector<f64>, source_factor: f64);
}

/// Parameters for source stepping convergence aid.
#[derive(Debug, Clone)]
pub struct SourceSteppingParams {
    /// Initial source factor (default 0.1).
    pub initial_factor: f64,
    /// Factor increment per step (default 0.1).
    pub factor_step: f64,
    /// Maximum attempts per source level (default 5).
    pub max_attempts_per_level: usize,
    /// Minimum factor step before giving up (default 0.01).
    pub min_factor_step: f64,
}

impl Default for SourceSteppingParams {
    fn default() -> Self {
        Self {
            initial_factor: 0.1,
            factor_step: 0.1,
            max_attempts_per_level: 5,
            min_factor_step: 0.01,
        }
    }
}

/// Result of source stepping.
#[derive(Debug, Clone)]
pub struct SourceSteppingResult {
    /// Final solution.
    pub solution: DVector<f64>,
    /// Total Newton-Raphson iterations across all source levels.
    pub total_iterations: usize,
    /// Number of source stepping levels used.
    pub num_levels: usize,
    /// Whether the full solution (source_factor = 1.0) was achieved.
    pub converged: bool,
}

/// Solve a nonlinear system using source stepping.
///
/// Source stepping is a convergence aid that starts with sources scaled to a
/// small fraction (e.g., 0.1) and gradually increases to full value (1.0).
/// Each source level uses the previous solution as an initial guess.
///
/// This helps difficult circuits converge by finding an easier operating point
/// first and tracking the solution as sources are ramped up.
///
/// # Arguments
/// * `num_nodes` - Number of nodes (excluding ground)
/// * `num_vsources` - Number of voltage sources
/// * `stamper` - Callback to stamp the system with source scaling
/// * `criteria` - Convergence criteria for each Newton-Raphson solve
/// * `params` - Source stepping parameters
pub fn solve_with_source_stepping(
    num_nodes: usize,
    num_vsources: usize,
    stamper: &dyn ScaledNonlinearStamper,
    criteria: &ConvergenceCriteria,
    params: &SourceSteppingParams,
) -> Result<SourceSteppingResult> {
    let size = num_nodes + num_vsources;
    let mut solution = DVector::zeros(size);
    let mut total_iterations = 0;
    let mut num_levels = 0;
    let mut source_factor = params.initial_factor;
    let mut factor_step = params.factor_step;

    // Create a wrapper stamper for each source level
    struct LevelStamper<'a> {
        inner: &'a dyn ScaledNonlinearStamper,
        factor: f64,
    }

    impl NonlinearStamper for LevelStamper<'_> {
        fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
            self.inner.stamp_at_scaled(mna, solution, self.factor);
        }
    }

    while source_factor <= 1.0 + 1e-9 {
        // Cap at exactly 1.0
        let current_factor = source_factor.min(1.0);

        let level_stamper = LevelStamper {
            inner: stamper,
            factor: current_factor,
        };

        // Try to converge at this source level
        let mut attempts = 0;
        let mut level_converged = false;

        while attempts < params.max_attempts_per_level && !level_converged {
            let result = solve_newton_raphson(
                num_nodes,
                num_vsources,
                &level_stamper,
                criteria,
                Some(&solution),
            )?;

            total_iterations += result.iterations;

            if result.converged {
                solution = result.solution;
                level_converged = true;
            } else {
                attempts += 1;
            }
        }

        if !level_converged {
            // Try reducing step size
            if factor_step > params.min_factor_step {
                factor_step /= 2.0;
                source_factor -= factor_step; // Back up
                continue;
            } else {
                // Give up
                return Ok(SourceSteppingResult {
                    solution,
                    total_iterations,
                    num_levels,
                    converged: false,
                });
            }
        }

        num_levels += 1;

        // Move to next source level
        if (current_factor - 1.0).abs() < 1e-9 {
            break;
        }

        source_factor += factor_step;
        // Gradually increase step size if we're doing well
        if factor_step < params.factor_step && level_converged {
            factor_step = (factor_step * 1.5).min(params.factor_step);
        }
    }

    Ok(SourceSteppingResult {
        solution,
        total_iterations,
        num_levels,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple nonlinear test: resistor + diode circuit
    /// Uses a simplified exponential model.
    struct DiodeCircuitStamper {
        v_source: f64,
        resistance: f64,
        is: f64,
        nvt: f64,
    }

    impl NonlinearStamper for DiodeCircuitStamper {
        fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
            // Circuit: V1 -- R1 -- node1 -- D1 -- GND
            // Node 0 = node connected to V1 through R1 and to D1 anode
            // V1 is voltage source at node 1

            // Stamp voltage source (V1 at node 1, current var index 0)
            mna.stamp_voltage_source(Some(0), None, 0, self.v_source);

            // Stamp resistor from node 1 to node 0 (our diode node is 1, shifted)
            // Actually, let's simplify: 2 nodes
            // node 0 = V1 output (fixed at v_source)
            // node 1 = R1-D1 junction
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(0), Some(1), g);

            // Stamp diode at node 1 to ground
            let vd = solution[1]; // voltage across diode
            let vd_limited = if vd > 0.8 {
                0.8 + (vd - 0.8) * 0.1 // limit step
            } else {
                vd
            };
            let exp_term = (vd_limited / self.nvt).exp();
            let id = self.is * (exp_term - 1.0);
            let gd = (self.is * exp_term / self.nvt).max(1e-12);
            let ieq = id - gd * vd_limited;

            // Stamp diode conductance (node 1 to ground)
            mna.stamp_conductance(Some(1), None, gd);
            // Stamp equivalent current source
            mna.stamp_current_source(Some(1), None, ieq);
        }
    }

    #[test]
    fn test_newton_raphson_diode_circuit() {
        let stamper = DiodeCircuitStamper {
            v_source: 5.0,
            resistance: 1000.0,
            is: 1e-14,
            nvt: 0.02585,
        };

        let criteria = ConvergenceCriteria::default();

        let result =
            solve_newton_raphson(2, 1, &stamper, &criteria, None).expect("NR should succeed");

        assert!(result.converged, "Should converge");
        assert!(
            result.iterations < 50,
            "Should converge in < 50 iterations, took {}",
            result.iterations
        );

        // V(node 0) should be 5V (voltage source)
        assert!(
            (result.solution[0] - 5.0).abs() < 1e-6,
            "V(0) = {} (expected 5.0)",
            result.solution[0]
        );

        // V(node 1) should be roughly 0.6-0.7V (diode forward voltage)
        let vd = result.solution[1];
        assert!(vd > 0.5 && vd < 0.8, "V(diode) = {} (expected 0.5-0.8)", vd);

        println!(
            "Diode circuit converged in {} iterations",
            result.iterations
        );
        println!("  V(source) = {:.4} V", result.solution[0]);
        println!("  V(diode)  = {:.4} V", vd);
        println!("  I(diode)  = {:.4} mA", (5.0 - vd) / 1000.0 * 1000.0);
    }

    #[test]
    fn test_convergence_check() {
        let old = DVector::from_vec(vec![1.0, 2.0, 0.001]);
        let new = DVector::from_vec(vec![1.0000001, 2.0000001, 0.001]);

        let criteria = ConvergenceCriteria::default();
        assert!(check_convergence(&old, &new, 2, &criteria));

        // Large change should not converge
        let new_far = DVector::from_vec(vec![1.1, 2.0, 0.001]);
        assert!(!check_convergence(&old, &new_far, 2, &criteria));
    }

    /// Scaled version of DiodeCircuitStamper for source stepping tests.
    struct ScaledDiodeCircuitStamper {
        v_source: f64,
        resistance: f64,
        is: f64,
        nvt: f64,
    }

    impl NonlinearStamper for ScaledDiodeCircuitStamper {
        fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
            // Full source value
            self.stamp_at_scaled(mna, solution, 1.0);
        }
    }

    impl ScaledNonlinearStamper for ScaledDiodeCircuitStamper {
        fn stamp_at_scaled(
            &self,
            mna: &mut MnaSystem,
            solution: &DVector<f64>,
            source_factor: f64,
        ) {
            // Stamp voltage source with scaled value
            mna.stamp_voltage_source(Some(0), None, 0, self.v_source * source_factor);

            // Stamp resistor (unchanged by source stepping)
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(0), Some(1), g);

            // Stamp diode at node 1 to ground
            let vd = solution[1];
            let vd_limited = if vd > 0.8 {
                0.8 + (vd - 0.8) * 0.1
            } else {
                vd
            };
            let exp_term = (vd_limited / self.nvt).exp();
            let id = self.is * (exp_term - 1.0);
            let gd = (self.is * exp_term / self.nvt).max(1e-12);
            let ieq = id - gd * vd_limited;

            mna.stamp_conductance(Some(1), None, gd);
            mna.stamp_current_source(Some(1), None, ieq);
        }
    }

    #[test]
    fn test_source_stepping_diode_circuit() {
        let stamper = ScaledDiodeCircuitStamper {
            v_source: 5.0,
            resistance: 1000.0,
            is: 1e-14,
            nvt: 0.02585,
        };

        let criteria = ConvergenceCriteria::default();
        let params = SourceSteppingParams::default();

        let result = solve_with_source_stepping(2, 1, &stamper, &criteria, &params)
            .expect("Source stepping should succeed");

        assert!(result.converged, "Should converge");
        assert!(
            result.num_levels > 1,
            "Should use multiple source levels, used {}",
            result.num_levels
        );

        // Final solution should match regular NR result
        let vd = result.solution[1];
        assert!(vd > 0.5 && vd < 0.8, "V(diode) = {} (expected 0.5-0.8)", vd);

        println!(
            "Source stepping converged with {} levels, {} total iterations",
            result.num_levels, result.total_iterations
        );
    }
}
