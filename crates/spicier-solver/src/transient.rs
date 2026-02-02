//! Transient analysis engine.

use std::collections::HashMap;

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;

use crate::dispatch::DispatchConfig;
use crate::error::Result;
use crate::gmres::GmresConfig;
use crate::linear::{CachedSparseLu, SPARSE_THRESHOLD, solve_dense};
use crate::operator::RealOperator;
use crate::preconditioner::{JacobiPreconditioner, RealPreconditioner};
use crate::sparse_operator::SparseRealOperator;

/// TR-BDF2 gamma parameter: γ = 2 - √2 ≈ 0.5858 for the fraction of step using Trapezoidal.
///
/// The method takes a Trapezoidal step of size γ*h, then a BDF2 step of size (1-γ)*h.
/// This value maximizes order of accuracy while maintaining L-stability.
pub const TRBDF2_GAMMA: f64 = 2.0 - std::f64::consts::SQRT_2;

/// Initial conditions for transient analysis.
///
/// Stores node name -> voltage mappings from .IC commands.
#[derive(Debug, Clone, Default)]
pub struct InitialConditions {
    /// Node voltages keyed by node name.
    pub voltages: HashMap<String, f64>,
}

impl InitialConditions {
    /// Create an empty initial conditions set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node voltage initial condition.
    pub fn set_voltage(&mut self, node: &str, voltage: f64) {
        self.voltages.insert(node.to_string(), voltage);
    }

    /// Apply initial conditions to a solution vector.
    ///
    /// The `node_map` maps node names to their MNA matrix indices.
    pub fn apply(&self, solution: &mut DVector<f64>, node_map: &HashMap<String, usize>) {
        for (node, &voltage) in &self.voltages {
            if let Some(&idx) = node_map.get(node) {
                solution[idx] = voltage;
            }
        }
    }

    /// Check if any initial conditions are set.
    pub fn is_empty(&self) -> bool {
        self.voltages.is_empty()
    }
}

/// Integration method for transient analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    /// Backward Euler (first order, A-stable).
    BackwardEuler,
    /// Trapezoidal (second order, A-stable).
    Trapezoidal,
    /// TR-BDF2 (second order, L-stable, good for stiff circuits).
    ///
    /// A composite method that uses Trapezoidal for γ*h (γ ≈ 0.2929),
    /// then BDF2 for the remaining (1-γ)*h. Provides L-stability
    /// without the numerical ringing issues of pure Trapezoidal.
    TrBdf2,
}

/// Transient analysis parameters.
#[derive(Debug, Clone)]
pub struct TransientParams {
    /// Stop time (s).
    pub tstop: f64,
    /// Maximum timestep (s).
    pub tstep: f64,
    /// Integration method.
    pub method: IntegrationMethod,
}

/// Parameters for adaptive timestep control.
#[derive(Debug, Clone)]
pub struct AdaptiveTransientParams {
    /// Stop time (s).
    pub tstop: f64,
    /// Initial timestep (s).
    pub h_init: f64,
    /// Minimum timestep (s).
    pub h_min: f64,
    /// Maximum timestep (s).
    pub h_max: f64,
    /// Relative tolerance for LTE.
    pub reltol: f64,
    /// Absolute tolerance for LTE.
    pub abstol: f64,
    /// Integration method.
    pub method: IntegrationMethod,
}

impl Default for AdaptiveTransientParams {
    fn default() -> Self {
        Self {
            tstop: 1e-3,
            h_init: 1e-9,
            h_min: 1e-15,
            h_max: 1e-6,
            reltol: 1e-3,
            abstol: 1e-6,
            method: IntegrationMethod::Trapezoidal,
        }
    }
}

impl AdaptiveTransientParams {
    /// Create parameters for a specific stop time with defaults.
    pub fn for_tstop(tstop: f64) -> Self {
        Self {
            tstop,
            h_max: tstop / 100.0, // At least 100 points
            ..Default::default()
        }
    }
}

/// State of a capacitor for companion model.
#[derive(Debug, Clone)]
pub struct CapacitorState {
    /// Capacitance (F).
    pub capacitance: f64,
    /// Voltage at previous timestep.
    pub v_prev: f64,
    /// Current at previous timestep (for trapezoidal).
    pub i_prev: f64,
    /// Voltage at two timesteps ago (for TR-BDF2).
    pub v_prev_prev: f64,
    /// Positive node MNA index (None for ground).
    pub node_pos: Option<usize>,
    /// Negative node MNA index (None for ground).
    pub node_neg: Option<usize>,
}

impl CapacitorState {
    /// Create a new capacitor state.
    pub fn new(capacitance: f64, node_pos: Option<usize>, node_neg: Option<usize>) -> Self {
        Self {
            capacitance,
            v_prev: 0.0,
            i_prev: 0.0,
            v_prev_prev: 0.0,
            node_pos,
            node_neg,
        }
    }

    /// Stamp the companion model for Backward Euler.
    ///
    /// C is replaced by: G_eq = C/h in parallel with I_eq = C/h * V_prev
    pub fn stamp_be(&self, mna: &mut MnaSystem, h: f64) {
        let geq = self.capacitance / h;
        let ieq = geq * self.v_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source: ieq flows from neg to pos (charging)
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Stamp the companion model for Trapezoidal rule.
    ///
    /// C is replaced by: G_eq = 2C/h in parallel with I_eq = 2C/h * V_prev + I_prev
    pub fn stamp_trap(&self, mna: &mut MnaSystem, h: f64) {
        let geq = 2.0 * self.capacitance / h;
        let ieq = geq * self.v_prev + self.i_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Update state after solving a timestep.
    pub fn update(&mut self, v_new: f64, h: f64, method: IntegrationMethod) {
        match method {
            IntegrationMethod::BackwardEuler => {
                self.i_prev = self.capacitance / h * (v_new - self.v_prev);
            }
            IntegrationMethod::Trapezoidal => {
                self.i_prev = 2.0 * self.capacitance / h * (v_new - self.v_prev) - self.i_prev;
            }
            IntegrationMethod::TrBdf2 => {
                // TR-BDF2 update after full step completion
                // Current is computed from the BDF2 formula
                let gamma = TRBDF2_GAMMA;
                let alpha = (1.0 - gamma) / (gamma * (2.0 - gamma));
                self.i_prev = self.capacitance / h
                    * ((1.0 + alpha) * v_new - (1.0 + 2.0 * alpha) * self.v_prev
                        + alpha * self.v_prev_prev);
            }
        }
        self.v_prev_prev = self.v_prev;
        self.v_prev = v_new;
    }

    /// Update state after TR-BDF2 intermediate (Trapezoidal) step.
    ///
    /// Called after the first stage of TR-BDF2 with the intermediate voltage.
    pub fn update_trbdf2_intermediate(&mut self, v_gamma: f64, h: f64) {
        let gamma = TRBDF2_GAMMA;
        let h_gamma = gamma * h;
        // Store current v_prev as v_prev_prev for BDF2 stage
        self.v_prev_prev = self.v_prev;
        // Update i_prev using trapezoidal current
        self.i_prev = 2.0 * self.capacitance / h_gamma * (v_gamma - self.v_prev_prev) - self.i_prev;
        // Update v_prev to intermediate value
        self.v_prev = v_gamma;
    }

    /// Stamp companion model for TR-BDF2 BDF2 stage.
    ///
    /// Uses v_prev (at γ*h) and v_prev_prev (at 0) for BDF2 formula.
    pub fn stamp_trbdf2_bdf2(&self, mna: &mut MnaSystem, h: f64) {
        let gamma = TRBDF2_GAMMA;
        // BDF2 coefficients for non-uniform step: h1 = γ*h, h2 = (1-γ)*h
        // The step we're taking is h2 = (1-γ)*h
        let h2 = (1.0 - gamma) * h;
        let h1 = gamma * h;
        let rho = h2 / h1; // ratio of step sizes

        // BDF2 for non-uniform steps:
        // y_{n+1} = a1 * y_n + a2 * y_{n-1} + b0 * h2 * y'_{n+1}
        // where:
        //   a1 = (1+ρ)² / (1+2ρ)
        //   a2 = -ρ² / (1+2ρ)
        //   b0 = (1+ρ) / (1+2ρ)
        let denom = 1.0 + 2.0 * rho;
        let a1 = (1.0 + rho).powi(2) / denom;
        let a2 = -rho * rho / denom;
        let b0 = (1.0 + rho) / denom;

        // For capacitor: i = C * dv/dt
        // Geq = C / (b0 * h2)
        let geq = self.capacitance / (b0 * h2);
        // Ieq represents the history terms: current = Geq * (a1*v_n + a2*v_{n-1})
        let ieq = geq * (a1 * self.v_prev + a2 * self.v_prev_prev);

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Estimate Local Truncation Error for the capacitor voltage.
    ///
    /// Uses the difference between Trapezoidal and Backward Euler predictions.
    /// For a capacitor: LTE ≈ h²/12 * C * d²v/dt²
    ///
    /// This method computes the LTE estimate using the "Milne device":
    /// LTE ≈ |v_trap - v_be| / 3
    pub fn estimate_lte(&self, v_new: f64, h: f64) -> f64 {
        // Current computed by Trapezoidal
        let i_trap = 2.0 * self.capacitance / h * (v_new - self.v_prev) - self.i_prev;

        // Current computed by Backward Euler
        let i_be = self.capacitance / h * (v_new - self.v_prev);

        // The difference gives an error estimate
        // For Trapezoidal with Milne device: LTE ≈ |i_trap - i_be| / 3
        // This estimates the error in the capacitor current
        (i_trap - i_be).abs() / 3.0
    }

    /// Get voltage across capacitor from solution vector.
    pub fn voltage_from_solution(&self, solution: &DVector<f64>) -> f64 {
        let vp = self.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = self.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        vp - vn
    }
}

/// State of an inductor for companion model.
#[derive(Debug, Clone)]
pub struct InductorState {
    /// Inductance (H).
    pub inductance: f64,
    /// Current at previous timestep.
    pub i_prev: f64,
    /// Voltage at previous timestep (for trapezoidal).
    pub v_prev: f64,
    /// Current at two timesteps ago (for TR-BDF2).
    pub i_prev_prev: f64,
    /// Positive node MNA index (None for ground).
    pub node_pos: Option<usize>,
    /// Negative node MNA index (None for ground).
    pub node_neg: Option<usize>,
}

impl InductorState {
    /// Create a new inductor state.
    pub fn new(inductance: f64, node_pos: Option<usize>, node_neg: Option<usize>) -> Self {
        Self {
            inductance,
            i_prev: 0.0,
            v_prev: 0.0,
            i_prev_prev: 0.0,
            node_pos,
            node_neg,
        }
    }

    /// Stamp the companion model for Backward Euler.
    ///
    /// L is replaced by: G_eq = h/L in parallel with I_eq = I_prev
    /// The inductor current flows from node_pos to node_neg.
    pub fn stamp_be(&self, mna: &mut MnaSystem, h: f64) {
        let geq = h / self.inductance;
        let ieq = self.i_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source ieq flows from node_pos to node_neg (same direction as i_prev)
        mna.stamp_current_source(self.node_pos, self.node_neg, ieq);
    }

    /// Stamp the companion model for Trapezoidal rule.
    ///
    /// L is replaced by: G_eq = h/(2L) in parallel with I_eq = I_prev + h/(2L) * V_prev
    /// The inductor current flows from node_pos to node_neg.
    pub fn stamp_trap(&self, mna: &mut MnaSystem, h: f64) {
        let geq = h / (2.0 * self.inductance);
        let ieq = self.i_prev + geq * self.v_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source ieq flows from node_pos to node_neg (same direction as i_prev)
        mna.stamp_current_source(self.node_pos, self.node_neg, ieq);
    }

    /// Update state after solving a timestep.
    pub fn update(&mut self, v_new: f64, h: f64, method: IntegrationMethod) {
        match method {
            IntegrationMethod::BackwardEuler => {
                self.i_prev += h / self.inductance * v_new;
            }
            IntegrationMethod::Trapezoidal => {
                self.i_prev += h / (2.0 * self.inductance) * (v_new + self.v_prev);
            }
            IntegrationMethod::TrBdf2 => {
                // TR-BDF2 update after full step completion
                let gamma = TRBDF2_GAMMA;
                let alpha = (1.0 - gamma) / (gamma * (2.0 - gamma));
                let di = h / self.inductance
                    * ((1.0 + alpha) * v_new - (1.0 + 2.0 * alpha) * self.v_prev
                        + alpha * self.v_prev);
                self.i_prev += di;
            }
        }
        self.i_prev_prev = self.i_prev;
        self.v_prev = v_new;
    }

    /// Update state after TR-BDF2 intermediate (Trapezoidal) step.
    pub fn update_trbdf2_intermediate(&mut self, v_gamma: f64, h: f64) {
        let gamma = TRBDF2_GAMMA;
        let h_gamma = gamma * h;
        // Save current i_prev for BDF2 stage
        self.i_prev_prev = self.i_prev;
        // Trapezoidal update for intermediate step
        self.i_prev += h_gamma / (2.0 * self.inductance) * (v_gamma + self.v_prev);
        // v_prev updated to intermediate value (don't update yet, done in main loop)
    }

    /// Stamp companion model for TR-BDF2 BDF2 stage.
    pub fn stamp_trbdf2_bdf2(&self, mna: &mut MnaSystem, h: f64) {
        let gamma = TRBDF2_GAMMA;
        let h2 = (1.0 - gamma) * h;
        let h1 = gamma * h;
        let rho = h2 / h1;

        // BDF2 coefficients for non-uniform steps
        // i_{n+1} = a1 * i_n + a2 * i_{n-1} + b0 * h2 / L * v_{n+1}
        let denom = 1.0 + 2.0 * rho;
        let a1 = (1.0 + rho).powi(2) / denom;
        let a2 = -rho * rho / denom;
        let b0 = (1.0 + rho) / denom;

        // For inductor: L * di/dt = v
        // Geq = b0 * h2 / L (conductance seen by the circuit)
        let geq = b0 * h2 / self.inductance;
        // Ieq represents the history terms
        let ieq = a1 * self.i_prev + a2 * self.i_prev_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        // Current source ieq flows from node_pos to node_neg (same direction as i_prev)
        mna.stamp_current_source(self.node_pos, self.node_neg, ieq);
    }

    /// Estimate Local Truncation Error for the inductor current.
    ///
    /// Uses the difference between Trapezoidal and Backward Euler predictions.
    /// For an inductor: LTE ≈ h²/12 * L * d²i/dt²
    pub fn estimate_lte(&self, v_new: f64, h: f64) -> f64 {
        // Current increment by Trapezoidal
        let di_trap = h / (2.0 * self.inductance) * (v_new + self.v_prev);

        // Current increment by Backward Euler
        let di_be = h / self.inductance * v_new;

        // The difference gives an error estimate
        (di_trap - di_be).abs() / 3.0
    }

    /// Get voltage across inductor from solution vector.
    pub fn voltage_from_solution(&self, solution: &DVector<f64>) -> f64 {
        let vp = self.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = self.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        vp - vn
    }
}

/// A single timepoint in a transient simulation result.
#[derive(Debug, Clone)]
pub struct TimePoint {
    /// Time value (s).
    pub time: f64,
    /// Solution vector at this time.
    pub solution: DVector<f64>,
}

/// Result of a transient simulation.
#[derive(Debug, Clone)]
pub struct TransientResult {
    /// All computed timepoints.
    pub points: Vec<TimePoint>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
}

impl TransientResult {
    /// Get the voltage at a node across all timepoints.
    pub fn voltage_waveform(&self, node_idx: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|tp| (tp.time, tp.solution[node_idx]))
            .collect()
    }

    /// Get all time values.
    pub fn times(&self) -> Vec<f64> {
        self.points.iter().map(|tp| tp.time).collect()
    }

    /// Interpolate the solution at a specific time.
    ///
    /// Uses linear interpolation between the two nearest timepoints.
    /// Returns None if time is outside the simulation range.
    pub fn interpolate_at(&self, time: f64) -> Option<DVector<f64>> {
        if self.points.is_empty() {
            return None;
        }

        // Handle boundary cases
        if time <= self.points[0].time {
            return Some(self.points[0].solution.clone());
        }
        if time >= self.points.last()?.time {
            return Some(self.points.last()?.solution.clone());
        }

        // Find the interval containing time
        for i in 0..self.points.len() - 1 {
            let t0 = self.points[i].time;
            let t1 = self.points[i + 1].time;

            if time >= t0 && time <= t1 {
                // Linear interpolation
                let alpha = (time - t0) / (t1 - t0);
                let v0 = &self.points[i].solution;
                let v1 = &self.points[i + 1].solution;
                return Some(v0 * (1.0 - alpha) + v1 * alpha);
            }
        }

        None
    }

    /// Sample the waveform at evenly-spaced times.
    ///
    /// Returns a new TransientResult with timepoints at regular intervals.
    ///
    /// # Arguments
    /// * `tstep` - Time step between samples
    /// * `tstart` - Start time (default 0.0)
    /// * `tstop` - Stop time (uses simulation end time if None)
    pub fn sample_at_times(
        &self,
        tstep: f64,
        tstart: Option<f64>,
        tstop: Option<f64>,
    ) -> TransientResult {
        let tstart = tstart.unwrap_or(0.0);
        let tstop = tstop.unwrap_or_else(|| self.points.last().map(|p| p.time).unwrap_or(0.0));

        let mut sampled_points = Vec::new();
        let mut t = tstart;

        while t <= tstop + tstep * 0.001 {
            if let Some(solution) = self.interpolate_at(t) {
                sampled_points.push(TimePoint { time: t, solution });
            }
            t += tstep;
        }

        TransientResult {
            points: sampled_points,
            num_nodes: self.num_nodes,
        }
    }

    /// Get the voltage at a node at a specific time (interpolated).
    pub fn voltage_at(&self, node_idx: usize, time: f64) -> Option<f64> {
        self.interpolate_at(time).map(|sol| sol[node_idx])
    }
}

/// Callback for stamping the circuit at each transient timestep.
pub trait TransientStamper {
    /// Stamp all non-reactive (resistive + source) elements at the given time.
    ///
    /// For time-varying sources (PULSE, SIN, PWL), the source value should be
    /// evaluated at the specified time. For DC sources, time is ignored.
    fn stamp_at_time(&self, mna: &mut MnaSystem, time: f64);

    /// Get the number of nodes.
    fn num_nodes(&self) -> usize;

    /// Get the number of voltage source current variables.
    fn num_vsources(&self) -> usize;
}

/// Run a transient simulation.
///
/// # Arguments
/// * `stamper` - Stamps resistive elements and sources
/// * `caps` - Capacitor companion model states
/// * `inds` - Inductor companion model states
/// * `params` - Transient parameters
/// * `dc_solution` - Initial DC operating point
pub fn solve_transient(
    stamper: &dyn TransientStamper,
    caps: &mut [CapacitorState],
    inds: &mut [InductorState],
    params: &TransientParams,
    dc_solution: &DVector<f64>,
) -> Result<TransientResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let mut solution = dc_solution.clone();
    let h = params.tstep;

    // Initialize reactive element states from DC solution
    for cap in caps.iter_mut() {
        let vp = cap.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = cap.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        cap.v_prev = vp - vn;
        cap.i_prev = 0.0; // No current through caps at DC
    }

    for ind in inds.iter_mut() {
        let vp = ind.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = ind.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        ind.v_prev = vp - vn;
        // ind.i_prev is set from DC solution if available
    }

    let mut result = TransientResult {
        points: Vec::new(),
        num_nodes,
    };

    // Store initial point
    result.points.push(TimePoint {
        time: 0.0,
        solution: solution.clone(),
    });

    let num_steps = (params.tstop / h).ceil() as usize;
    let mna_size = num_nodes + num_vsources;

    // Cached sparse solver (created on first timestep if needed)
    let mut cached_solver: Option<CachedSparseLu> = None;

    for step in 1..=num_steps {
        let t = (step as f64) * h;

        // Build MNA system for this timestep
        let mut mna = MnaSystem::new(num_nodes, num_vsources);

        // Stamp static elements (resistors, sources)
        stamper.stamp_at_time(&mut mna, t);

        // Stamp companion models for reactive elements and solve
        match params.method {
            IntegrationMethod::BackwardEuler => {
                for cap in caps.iter() {
                    cap.stamp_be(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_be(&mut mna, h);
                }

                // Solve
                solution = if mna_size >= SPARSE_THRESHOLD {
                    let solver = match &cached_solver {
                        Some(s) => s,
                        None => {
                            cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached_solver.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())?
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())?
                };

                // Update state
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::Trapezoidal => {
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h);
                }

                // Solve
                solution = if mna_size >= SPARSE_THRESHOLD {
                    let solver = match &cached_solver {
                        Some(s) => s,
                        None => {
                            cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached_solver.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())?
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())?
                };

                // Update state
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::TrBdf2 => {
                // TR-BDF2: Two-stage method
                // Stage 1: Trapezoidal step for γ*h
                let h_gamma = TRBDF2_GAMMA * h;
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h_gamma);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h_gamma);
                }

                // Solve stage 1
                let solution_gamma = if mna_size >= SPARSE_THRESHOLD {
                    let solver = match &cached_solver {
                        Some(s) => s,
                        None => {
                            cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached_solver.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())?
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())?
                };

                // Update state to intermediate point
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution_gamma);
                    cap.update_trbdf2_intermediate(v, h);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution_gamma);
                    ind.update_trbdf2_intermediate(v, h);
                    ind.v_prev = v; // Update v_prev for BDF2 stage
                }

                // Stage 2: BDF2 step for (1-γ)*h
                let mut mna2 = MnaSystem::new(num_nodes, num_vsources);
                stamper.stamp_at_time(&mut mna2, t);
                for cap in caps.iter() {
                    cap.stamp_trbdf2_bdf2(&mut mna2, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trbdf2_bdf2(&mut mna2, h);
                }

                // Solve stage 2
                solution = if mna_size >= SPARSE_THRESHOLD {
                    cached_solver
                        .as_ref()
                        .unwrap()
                        .solve(&mna2.triplets, mna2.rhs())?
                } else {
                    solve_dense(&mna2.to_dense_matrix(), mna2.rhs())?
                };

                // Final state update
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
        }

        result.points.push(TimePoint {
            time: t,
            solution: solution.clone(),
        });
    }

    Ok(result)
}

/// Run transient simulation with configurable dispatch.
///
/// This variant allows specifying the compute backend and solver strategy.
/// For large systems, can use GMRES instead of direct LU.
///
/// # Arguments
/// * `stamper` - Stamps resistive elements and sources
/// * `caps` - Capacitor companion model states
/// * `inds` - Inductor companion model states
/// * `params` - Transient parameters
/// * `dc_solution` - Initial DC operating point
/// * `config` - Dispatch configuration
pub fn solve_transient_dispatched(
    stamper: &dyn TransientStamper,
    caps: &mut [CapacitorState],
    inds: &mut [InductorState],
    params: &TransientParams,
    dc_solution: &DVector<f64>,
    config: &DispatchConfig,
) -> Result<TransientResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let mut solution = dc_solution.clone();
    let h = params.tstep;
    let mna_size = num_nodes + num_vsources;

    // Decide solver strategy based on size
    let use_gmres = config.use_gmres(mna_size);

    // Initialize reactive element states from DC solution
    for cap in caps.iter_mut() {
        let vp = cap.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = cap.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        cap.v_prev = vp - vn;
        cap.i_prev = 0.0;
    }

    for ind in inds.iter_mut() {
        let vp = ind.node_pos.map(|i| solution[i]).unwrap_or(0.0);
        let vn = ind.node_neg.map(|i| solution[i]).unwrap_or(0.0);
        ind.v_prev = vp - vn;
    }

    let mut result = TransientResult {
        points: Vec::new(),
        num_nodes,
    };

    result.points.push(TimePoint {
        time: 0.0,
        solution: solution.clone(),
    });

    let num_steps = (params.tstop / h).ceil() as usize;

    // Cached sparse solver for direct LU
    let mut cached_solver: Option<CachedSparseLu> = None;

    for step in 1..=num_steps {
        let t = (step as f64) * h;

        let mut mna = MnaSystem::new(num_nodes, num_vsources);
        stamper.stamp_at_time(&mut mna, t);

        // Helper closure for solving
        let solve_mna = |mna: &MnaSystem,
                         cached: &mut Option<CachedSparseLu>|
         -> Result<DVector<f64>> {
            if use_gmres {
                solve_transient_gmres(mna, &config.gmres_config)
            } else if mna_size >= SPARSE_THRESHOLD {
                let solver = match cached.as_ref() {
                    Some(s) => s,
                    None => {
                        *cached = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                        cached.as_ref().unwrap()
                    }
                };
                solver.solve(&mna.triplets, mna.rhs())
            } else {
                solve_dense(&mna.to_dense_matrix(), mna.rhs())
            }
        };

        match params.method {
            IntegrationMethod::BackwardEuler => {
                for cap in caps.iter() {
                    cap.stamp_be(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_be(&mut mna, h);
                }
                solution = solve_mna(&mna, &mut cached_solver)?;
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::Trapezoidal => {
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h);
                }
                solution = solve_mna(&mna, &mut cached_solver)?;
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::TrBdf2 => {
                // Stage 1: Trapezoidal for γ*h
                let h_gamma = TRBDF2_GAMMA * h;
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h_gamma);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h_gamma);
                }
                let solution_gamma = solve_mna(&mna, &mut cached_solver)?;

                // Update to intermediate state
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution_gamma);
                    cap.update_trbdf2_intermediate(v, h);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution_gamma);
                    ind.update_trbdf2_intermediate(v, h);
                    ind.v_prev = v;
                }

                // Stage 2: BDF2 for (1-γ)*h
                let mut mna2 = MnaSystem::new(num_nodes, num_vsources);
                stamper.stamp_at_time(&mut mna2, t);
                for cap in caps.iter() {
                    cap.stamp_trbdf2_bdf2(&mut mna2, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trbdf2_bdf2(&mut mna2, h);
                }
                solution = solve_mna(&mna2, &mut cached_solver)?;

                // Final state update
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
        }

        result.points.push(TimePoint {
            time: t,
            solution: solution.clone(),
        });
    }

    Ok(result)
}

/// Solve a transient timestep using GMRES.
fn solve_transient_gmres(mna: &MnaSystem, config: &GmresConfig) -> Result<DVector<f64>> {
    let size = mna.size();

    let operator = SparseRealOperator::from_triplets(size, &mna.triplets)
        .ok_or_else(|| crate::error::Error::SolverError("Failed to build sparse operator".into()))?;

    let preconditioner = JacobiPreconditioner::from_triplets(size, &mna.triplets);
    let rhs: Vec<f64> = mna.rhs().iter().copied().collect();

    let gmres_result = crate::gmres::solve_gmres_real_preconditioned(
        &operator as &dyn RealOperator,
        &preconditioner as &dyn RealPreconditioner,
        &rhs,
        config,
    );

    if !gmres_result.converged {
        log::warn!(
            "Transient GMRES did not converge after {} iterations (residual: {:.2e})",
            gmres_result.iterations,
            gmres_result.residual
        );
    }

    Ok(DVector::from_vec(gmres_result.x))
}

/// Result of adaptive transient simulation with statistics.
#[derive(Debug, Clone)]
pub struct AdaptiveTransientResult {
    /// All computed timepoints.
    pub points: Vec<TimePoint>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
    /// Total number of timesteps taken.
    pub total_steps: usize,
    /// Number of rejected timesteps.
    pub rejected_steps: usize,
    /// Minimum timestep used.
    pub min_step_used: f64,
    /// Maximum timestep used.
    pub max_step_used: f64,
}

impl AdaptiveTransientResult {
    /// Get the voltage at a node across all timepoints.
    pub fn voltage_waveform(&self, node_idx: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|tp| (tp.time, tp.solution[node_idx]))
            .collect()
    }

    /// Get all time values.
    pub fn times(&self) -> Vec<f64> {
        self.points.iter().map(|tp| tp.time).collect()
    }

    /// Interpolate the solution at a specific time.
    ///
    /// Uses linear interpolation between the two nearest timepoints.
    /// Returns None if time is outside the simulation range.
    pub fn interpolate_at(&self, time: f64) -> Option<DVector<f64>> {
        if self.points.is_empty() {
            return None;
        }

        // Handle boundary cases
        if time <= self.points[0].time {
            return Some(self.points[0].solution.clone());
        }
        if time >= self.points.last()?.time {
            return Some(self.points.last()?.solution.clone());
        }

        // Find the interval containing time
        for i in 0..self.points.len() - 1 {
            let t0 = self.points[i].time;
            let t1 = self.points[i + 1].time;

            if time >= t0 && time <= t1 {
                // Linear interpolation
                let alpha = (time - t0) / (t1 - t0);
                let v0 = &self.points[i].solution;
                let v1 = &self.points[i + 1].solution;
                return Some(v0 * (1.0 - alpha) + v1 * alpha);
            }
        }

        None
    }

    /// Sample the waveform at evenly-spaced times.
    ///
    /// Returns a new TransientResult with timepoints at regular intervals.
    /// This is useful for producing output at uniform time steps from an
    /// adaptive simulation that used variable step sizes.
    ///
    /// # Arguments
    /// * `tstep` - Time step between samples
    /// * `tstart` - Start time (default 0.0)
    /// * `tstop` - Stop time (uses simulation end time if None)
    pub fn sample_at_times(
        &self,
        tstep: f64,
        tstart: Option<f64>,
        tstop: Option<f64>,
    ) -> TransientResult {
        let tstart = tstart.unwrap_or(0.0);
        let tstop = tstop.unwrap_or_else(|| self.points.last().map(|p| p.time).unwrap_or(0.0));

        let mut sampled_points = Vec::new();
        let mut t = tstart;

        while t <= tstop + tstep * 0.001 {
            if let Some(solution) = self.interpolate_at(t) {
                sampled_points.push(TimePoint { time: t, solution });
            }
            t += tstep;
        }

        TransientResult {
            points: sampled_points,
            num_nodes: self.num_nodes,
        }
    }

    /// Get the voltage at a node at a specific time (interpolated).
    pub fn voltage_at(&self, node_idx: usize, time: f64) -> Option<f64> {
        self.interpolate_at(time).map(|sol| sol[node_idx])
    }
}

/// Run adaptive transient simulation with automatic timestep control.
///
/// Uses Local Truncation Error (LTE) estimation to automatically adjust
/// the timestep. Larger steps are taken when the solution is smooth,
/// smaller steps when it changes rapidly.
///
/// # Arguments
/// * `stamper` - Stamps resistive elements and sources
/// * `caps` - Capacitor companion model states
/// * `inds` - Inductor companion model states
/// * `params` - Adaptive transient parameters
/// * `dc_solution` - Initial DC operating point
pub fn solve_transient_adaptive(
    stamper: &dyn TransientStamper,
    caps: &mut [CapacitorState],
    inds: &mut [InductorState],
    params: &AdaptiveTransientParams,
    dc_solution: &DVector<f64>,
) -> Result<AdaptiveTransientResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let mna_size = num_nodes + num_vsources;

    let mut solution = dc_solution.clone();
    let mut t = 0.0;
    let mut h = params.h_init;

    // Initialize reactive element states from DC solution
    for cap in caps.iter_mut() {
        cap.v_prev = cap.voltage_from_solution(&solution);
        cap.i_prev = 0.0;
    }
    for ind in inds.iter_mut() {
        ind.v_prev = ind.voltage_from_solution(&solution);
    }

    let mut result = AdaptiveTransientResult {
        points: Vec::new(),
        num_nodes,
        total_steps: 0,
        rejected_steps: 0,
        min_step_used: f64::INFINITY,
        max_step_used: 0.0,
    };

    // Store initial point
    result.points.push(TimePoint {
        time: 0.0,
        solution: solution.clone(),
    });

    // Cached sparse solver
    let mut cached_solver: Option<CachedSparseLu> = None;

    // Save states for potential rollback
    let mut saved_cap_states: Vec<(f64, f64)> = caps.iter().map(|c| (c.v_prev, c.i_prev)).collect();
    let mut saved_ind_states: Vec<(f64, f64)> = inds.iter().map(|i| (i.i_prev, i.v_prev)).collect();

    while t < params.tstop {
        // Clamp timestep
        h = h.clamp(params.h_min, params.h_max);

        // Don't overshoot tstop
        if t + h > params.tstop {
            h = params.tstop - t;
        }

        // Build MNA system for this timestep
        let mut mna = MnaSystem::new(num_nodes, num_vsources);
        stamper.stamp_at_time(&mut mna, t);

        // Stamp companion models (using Trapezoidal for better accuracy)
        for cap in caps.iter() {
            cap.stamp_trap(&mut mna, h);
        }
        for ind in inds.iter() {
            ind.stamp_trap(&mut mna, h);
        }

        // Solve
        let new_solution = if mna_size >= SPARSE_THRESHOLD {
            let solver = match &cached_solver {
                Some(s) => s,
                None => {
                    cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                    cached_solver.as_ref().unwrap()
                }
            };
            solver.solve(&mna.triplets, mna.rhs())?
        } else {
            solve_dense(&mna.to_dense_matrix(), mna.rhs())?
        };

        // Estimate LTE for all reactive elements
        let mut max_lte = 0.0_f64;
        let mut max_ref = 0.0_f64; // Reference value for relative error

        for cap in caps.iter() {
            let v_new = cap.voltage_from_solution(&new_solution);
            let lte = cap.estimate_lte(v_new, h);
            max_lte = max_lte.max(lte);
            max_ref = max_ref.max(v_new.abs());
        }

        for ind in inds.iter() {
            let v_new = ind.voltage_from_solution(&new_solution);
            let lte = ind.estimate_lte(v_new, h);
            max_lte = max_lte.max(lte);
            max_ref = max_ref.max(ind.i_prev.abs());
        }

        // Compute tolerance: max(abstol, reltol * max_ref)
        let tol = params.abstol.max(params.reltol * max_ref);

        result.total_steps += 1;

        if max_lte > tol && h > params.h_min {
            // Reject step: LTE too large
            result.rejected_steps += 1;

            // Restore previous states
            for (cap, (v, i)) in caps.iter_mut().zip(saved_cap_states.iter()) {
                cap.v_prev = *v;
                cap.i_prev = *i;
            }
            for (ind, (i, v)) in inds.iter_mut().zip(saved_ind_states.iter()) {
                ind.i_prev = *i;
                ind.v_prev = *v;
            }

            // Reduce timestep (safety factor of 0.8)
            let factor = (tol / max_lte).sqrt().min(0.5);
            h *= factor.max(0.1); // Don't reduce by more than 10x
        } else {
            // Accept step
            t += h;
            solution = new_solution;

            // Update reactive element states
            for cap in caps.iter_mut() {
                let v_new = cap.voltage_from_solution(&solution);
                cap.update(v_new, h, IntegrationMethod::Trapezoidal);
            }
            for ind in inds.iter_mut() {
                let v_new = ind.voltage_from_solution(&solution);
                ind.update(v_new, h, IntegrationMethod::Trapezoidal);
            }

            // Save states for potential rollback
            saved_cap_states = caps.iter().map(|c| (c.v_prev, c.i_prev)).collect();
            saved_ind_states = inds.iter().map(|i| (i.i_prev, i.v_prev)).collect();

            // Track step size statistics
            result.min_step_used = result.min_step_used.min(h);
            result.max_step_used = result.max_step_used.max(h);

            // Store result
            result.points.push(TimePoint {
                time: t,
                solution: solution.clone(),
            });

            // Increase timestep for next step if LTE is small
            if max_lte < tol * 0.5 && h < params.h_max {
                let factor = (tol / max_lte.max(1e-20)).sqrt().min(2.0);
                h *= factor.min(1.5); // Don't increase by more than 1.5x
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DispatchConfig;

    #[test]
    fn test_transient_dispatched() {
        // Simple RC circuit with dispatched solver
        struct SimpleRcStamper;
        impl TransientStamper for SimpleRcStamper {
            fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
                mna.stamp_voltage_source(Some(0), None, 0, 5.0);
                mna.stamp_conductance(Some(0), Some(1), 1.0 / 1000.0);
            }
            fn num_nodes(&self) -> usize { 2 }
            fn num_vsources(&self) -> usize { 1 }
        }

        let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];
        let params = TransientParams {
            tstop: 1e-3,
            tstep: 100e-6,
            method: IntegrationMethod::BackwardEuler,
        };
        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);
        let config = DispatchConfig::default();

        let result = solve_transient_dispatched(
            &SimpleRcStamper, &mut caps, &mut [], &params, &dc, &config
        ).unwrap();

        // Should have 11 points (0 to 1ms in 100us steps)
        assert_eq!(result.points.len(), 11);
        // Capacitor should be charging
        assert!(result.points.last().unwrap().solution[1] > 0.0);
    }

    /// Simple RC circuit stamper: V1 -- R -- node0 -- C -- GND
    struct RcCircuitStamper {
        voltage: f64,
        resistance: f64,
    }

    impl TransientStamper for RcCircuitStamper {
        fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
            // Voltage source at node 0, current var index 0
            mna.stamp_voltage_source(Some(0), None, 0, self.voltage);
            // Resistor from node 0 to node 1
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(0), Some(1), g);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_rc_charging_be() {
        // RC circuit: V1=5V, R=1k, C=1uF
        // Time constant: tau = RC = 1k * 1uF = 1ms
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let capacitance = 1e-6;
        let mut caps = vec![CapacitorState::new(capacitance, Some(1), None)];

        let params = TransientParams {
            tstop: 5e-3,  // 5 time constants
            tstep: 10e-6, // 10us steps
            method: IntegrationMethod::BackwardEuler,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]); // V(0)=5, V(1)=0, I(V1)=-5mA

        let result = solve_transient(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        // After 5 tau, capacitor should be nearly charged to 5V
        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.05,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // At t = tau (1ms), voltage should be ~3.16V (= 5 * (1 - e^-1))
        let tau_step = (1e-3 / params.tstep).round() as usize;
        let v_at_tau = result.points[tau_step].solution[1];
        let expected_v_tau = 5.0 * (1.0 - (-1.0_f64).exp());
        assert!(
            (v_at_tau - expected_v_tau).abs() < 0.2,
            "V(cap) at tau = {} (expected ≈ {})",
            v_at_tau,
            expected_v_tau
        );
    }

    #[test]
    fn test_rc_charging_trapezoidal() {
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];

        let params = TransientParams {
            tstop: 5e-3,
            tstep: 10e-6,
            method: IntegrationMethod::Trapezoidal,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);

        let result = solve_transient(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.05,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // Trapezoidal should be more accurate at tau
        let tau_step = (1e-3 / params.tstep).round() as usize;
        let v_at_tau = result.points[tau_step].solution[1];
        let expected_v_tau = 5.0 * (1.0 - (-1.0_f64).exp());
        assert!(
            (v_at_tau - expected_v_tau).abs() < 0.1,
            "V(cap) at tau = {} (expected ≈ {}) [trapezoidal]",
            v_at_tau,
            expected_v_tau
        );
    }

    #[test]
    fn test_capacitor_companion_be() {
        let cap = CapacitorState {
            capacitance: 1e-6,
            v_prev: 2.5,
            i_prev: 0.0,
            v_prev_prev: 0.0,
            node_pos: Some(0),
            node_neg: None,
        };

        let mut mna = MnaSystem::new(1, 0);
        let h = 1e-6;
        cap.stamp_be(&mut mna, h);
        let matrix = mna.to_dense_matrix();

        // Geq = C/h = 1e-6/1e-6 = 1.0
        assert!(
            (matrix[(0, 0)] - 1.0).abs() < 1e-10,
            "Geq = {} (expected 1.0)",
            matrix[(0, 0)]
        );

        // Ieq = Geq * V_prev = 1.0 * 2.5 = 2.5
        assert!(
            (mna.rhs()[0] - 2.5).abs() < 1e-10,
            "Ieq = {} (expected 2.5)",
            mna.rhs()[0]
        );
    }

    #[test]
    fn test_adaptive_rc_charging() {
        // RC circuit: V1=5V, R=1k, C=1uF, tau=1ms
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let capacitance = 1e-6;
        let mut caps = vec![CapacitorState::new(capacitance, Some(1), None)];

        let params = AdaptiveTransientParams {
            tstop: 5e-3, // 5 time constants
            h_init: 1e-7,
            h_min: 1e-9,
            h_max: 1e-4,
            reltol: 1e-3,
            abstol: 1e-6,
            method: IntegrationMethod::Trapezoidal,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);

        let result = solve_transient_adaptive(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        // After 5 tau, capacitor should be nearly charged to 5V
        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.05,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // Adaptive should use fewer steps than fixed timestep
        // With fixed 10us steps, we'd need 500 steps
        // Adaptive should use fewer
        assert!(
            result.total_steps < 200,
            "Adaptive used {} steps (expected < 200 for efficiency)",
            result.total_steps
        );

        // Timestep should increase as capacitor approaches steady state
        assert!(
            result.max_step_used > params.h_init * 10.0,
            "Max step {} should grow from initial {}",
            result.max_step_used,
            params.h_init
        );

        println!(
            "Adaptive transient: {} total steps, {} rejected, h: [{:.2e}, {:.2e}]",
            result.total_steps,
            result.rejected_steps,
            result.min_step_used,
            result.max_step_used
        );
    }

    #[test]
    fn test_lte_estimation() {
        // Test that LTE estimate is reasonable for a smooth (constant rate) change.
        // For a constant dV/dt, the capacitor current is constant: i = C * dV/dt.
        // Trapezoidal and Backward Euler should agree, giving near-zero LTE.
        let h = 1e-6;
        let dv_dt = 1e5; // 0.1V per microsecond
        let v_prev = 0.0;
        let v_new = v_prev + dv_dt * h; // = 0.1V

        // Current is constant at C * dV/dt for linear voltage ramp
        let capacitance = 1e-6;
        let i_const = capacitance * dv_dt; // = 0.1 A

        let cap = CapacitorState {
            capacitance,
            v_prev,
            i_prev: i_const, // Current at previous step (same as current step for linear ramp)
            v_prev_prev: 0.0,
            node_pos: Some(0),
            node_neg: None,
        };

        let lte = cap.estimate_lte(v_new, h);

        // LTE should be non-negative
        assert!(lte >= 0.0, "LTE should be non-negative: {}", lte);

        // For a perfectly linear ramp with consistent i_prev, LTE should be very small
        assert!(
            lte < 1e-6,
            "LTE {} seems too large for constant-rate change",
            lte
        );
    }

    #[test]
    fn test_interpolate_at() {
        // Create a simple result with known values
        let points = vec![
            TimePoint {
                time: 0.0,
                solution: DVector::from_vec(vec![0.0, 0.0]),
            },
            TimePoint {
                time: 1.0,
                solution: DVector::from_vec(vec![1.0, 2.0]),
            },
            TimePoint {
                time: 2.0,
                solution: DVector::from_vec(vec![2.0, 4.0]),
            },
        ];

        let result = TransientResult {
            points,
            num_nodes: 2,
        };

        // Test interpolation at midpoint
        let interp = result.interpolate_at(0.5).unwrap();
        assert!((interp[0] - 0.5).abs() < 1e-10);
        assert!((interp[1] - 1.0).abs() < 1e-10);

        // Test interpolation at 1.5
        let interp = result.interpolate_at(1.5).unwrap();
        assert!((interp[0] - 1.5).abs() < 1e-10);
        assert!((interp[1] - 3.0).abs() < 1e-10);

        // Test at exact points
        let interp = result.interpolate_at(1.0).unwrap();
        assert!((interp[0] - 1.0).abs() < 1e-10);
        assert!((interp[1] - 2.0).abs() < 1e-10);

        // Test voltage_at helper
        assert!((result.voltage_at(0, 0.5).unwrap() - 0.5).abs() < 1e-10);
        assert!((result.voltage_at(1, 1.5).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_at_times() {
        // Create a result with 3 points at t=0, 0.3, 1.0
        let points = vec![
            TimePoint {
                time: 0.0,
                solution: DVector::from_vec(vec![0.0]),
            },
            TimePoint {
                time: 0.3,
                solution: DVector::from_vec(vec![0.3]),
            },
            TimePoint {
                time: 1.0,
                solution: DVector::from_vec(vec![1.0]),
            },
        ];

        let result = TransientResult {
            points,
            num_nodes: 1,
        };

        // Sample at tstep=0.25
        let sampled = result.sample_at_times(0.25, None, None);

        // Should have 5 points: 0.0, 0.25, 0.5, 0.75, 1.0
        assert_eq!(sampled.points.len(), 5);

        // Check times
        assert!((sampled.points[0].time - 0.0).abs() < 1e-10);
        assert!((sampled.points[1].time - 0.25).abs() < 1e-10);
        assert!((sampled.points[2].time - 0.5).abs() < 1e-10);
        assert!((sampled.points[3].time - 0.75).abs() < 1e-10);
        assert!((sampled.points[4].time - 1.0).abs() < 1e-10);

        // Check interpolated values (linear from 0 to 1)
        assert!((sampled.points[0].solution[0] - 0.0).abs() < 1e-10);
        assert!((sampled.points[4].solution[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rc_charging_trbdf2() {
        // Same RC circuit as other tests but using TR-BDF2
        let stamper = RcCircuitStamper {
            voltage: 5.0,
            resistance: 1000.0,
        };

        let mut caps = vec![CapacitorState::new(1e-6, Some(1), None)];

        let params = TransientParams {
            tstop: 5e-3,
            tstep: 10e-6,
            method: IntegrationMethod::TrBdf2,
        };

        let dc = DVector::from_vec(vec![5.0, 0.0, -0.005]);

        let result = solve_transient(&stamper, &mut caps, &mut [], &params, &dc).unwrap();

        // TR-BDF2 should produce reasonable results (similar to Trapezoidal)
        // At 5τ, voltage should be ~99.3% of final (very close to 5.0)
        let final_voltage = result.points.last().unwrap().solution[1];
        assert!(
            (final_voltage - 5.0).abs() < 0.15,
            "Final V(cap) = {} (expected ≈ 5.0)",
            final_voltage
        );

        // Check voltage at tau (time constant = RC = 1ms)
        // TR-BDF2 with 10µs steps should be within 20% at tau
        let tau_step = (1e-3 / params.tstep).round() as usize;
        let v_at_tau = result.points[tau_step].solution[1];
        let expected_v_tau = 5.0 * (1.0 - (-1.0_f64).exp()); // ~3.16V
        assert!(
            (v_at_tau - expected_v_tau).abs() < 0.6,
            "V(cap) at tau = {} (expected ≈ {}) [TR-BDF2]",
            v_at_tau,
            expected_v_tau
        );
    }

    /// Simple LC circuit stamper for oscillation test.
    /// Circuit: Initial voltage on capacitor, connected to inductor.
    /// Node 0: capacitor top / inductor top
    /// Ground: capacitor bottom / inductor bottom
    /// The inductor uses companion model (conductance + current source), not branch current.
    struct LcOscillatorStamper;

    impl TransientStamper for LcOscillatorStamper {
        fn stamp_at_time(&self, _mna: &mut MnaSystem, _time: f64) {
            // No static elements - capacitor and inductor are handled by companion models
            // The LC circuit has only reactive elements
        }

        fn num_nodes(&self) -> usize {
            1 // Just node 0 (top of L and C)
        }

        fn num_vsources(&self) -> usize {
            0 // No voltage sources - inductor uses companion model
        }
    }

    #[test]
    fn test_lc_oscillation() {
        // LC circuit: L = 1mH, C = 1µF
        // Resonant frequency: f = 1/(2π√(LC)) = 1/(2π√(1e-3 * 1e-6)) = 5033 Hz
        // Period: T = 1/f ≈ 0.199 ms ≈ 200 µs
        let inductance = 1e-3; // 1 mH
        let capacitance = 1e-6; // 1 µF

        let lc_product: f64 = inductance * capacitance;
        let expected_freq = 1.0 / (2.0 * std::f64::consts::PI * lc_product.sqrt());
        let expected_period: f64 = 1.0 / expected_freq;

        // Initial conditions: capacitor charged to 5V, zero inductor current
        // dc_solution: [V(0)] - just node voltage, no branch currents
        let dc = DVector::from_vec(vec![5.0]);

        // Create state for reactive elements
        let mut caps = vec![CapacitorState::new(capacitance, Some(0), None)];
        let mut inds = vec![InductorState::new(inductance, Some(0), None)];

        // Simulate for 5 periods using Trapezoidal (good for oscillators)
        let params = TransientParams {
            tstop: 5.0 * expected_period,
            tstep: expected_period / 50.0, // 50 points per period
            method: IntegrationMethod::Trapezoidal,
        };

        let result =
            solve_transient(&LcOscillatorStamper, &mut caps, &mut inds, &params, &dc).unwrap();

        // Find zero crossings to measure the period
        let voltages: Vec<f64> = result.points.iter().map(|p| p.solution[0]).collect();
        let times: Vec<f64> = result.points.iter().map(|p| p.time).collect();

        // Find first zero crossing from positive to negative (after initial positive)
        let mut zero_crossings = Vec::new();
        for i in 1..voltages.len() {
            if voltages[i - 1] > 0.0 && voltages[i] <= 0.0 {
                // Linear interpolation for more accurate crossing time
                let t_cross =
                    times[i - 1] + (0.0 - voltages[i - 1]) * (times[i] - times[i - 1])
                        / (voltages[i] - voltages[i - 1]);
                zero_crossings.push(t_cross);
            }
        }

        // Need at least 2 zero crossings to measure a full period
        assert!(
            zero_crossings.len() >= 2,
            "Not enough zero crossings found: {}",
            zero_crossings.len()
        );

        // Measure period from consecutive positive-to-negative zero crossings
        // These are separated by exactly one full period
        let measured_period = zero_crossings[1] - zero_crossings[0];
        let measured_freq = 1.0 / measured_period;

        // Check frequency is within 5% of expected
        let freq_error = (measured_freq - expected_freq).abs() / expected_freq;
        assert!(
            freq_error < 0.05,
            "LC oscillation frequency {} Hz differs from expected {} Hz by {:.1}%",
            measured_freq,
            expected_freq,
            freq_error * 100.0
        );

        // Check that amplitude is preserved (energy conservation)
        // For ideal LC, max voltage should stay close to initial voltage
        let max_voltage = voltages.iter().cloned().fold(0.0_f64, f64::max);
        let min_voltage = voltages.iter().cloned().fold(0.0_f64, f64::min);
        let amplitude = (max_voltage - min_voltage) / 2.0;

        assert!(
            (amplitude - 5.0).abs() < 0.5,
            "LC amplitude {} differs from expected 5.0 (amplitude decay)",
            amplitude
        );

        println!(
            "LC oscillator: expected f={:.1} Hz, measured f={:.1} Hz, error={:.2}%",
            expected_freq,
            measured_freq,
            freq_error * 100.0
        );
        println!(
            "LC oscillator: expected T={:.2} µs, measured T={:.2} µs",
            expected_period * 1e6,
            measured_period * 1e6
        );
    }
}
