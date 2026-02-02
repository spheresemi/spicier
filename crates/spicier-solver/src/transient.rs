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
        }
        self.v_prev = v_new;
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
            node_pos,
            node_neg,
        }
    }

    /// Stamp the companion model for Backward Euler.
    ///
    /// L is replaced by: G_eq = h/L in parallel with I_eq = I_prev
    pub fn stamp_be(&self, mna: &mut MnaSystem, h: f64) {
        let geq = h / self.inductance;
        let ieq = self.i_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
    }

    /// Stamp the companion model for Trapezoidal rule.
    ///
    /// L is replaced by: G_eq = h/(2L) in parallel with I_eq = I_prev + h/(2L) * V_prev
    pub fn stamp_trap(&self, mna: &mut MnaSystem, h: f64) {
        let geq = h / (2.0 * self.inductance);
        let ieq = self.i_prev + geq * self.v_prev;

        mna.stamp_conductance(self.node_pos, self.node_neg, geq);
        mna.stamp_current_source(self.node_neg, self.node_pos, ieq);
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
        }
        self.v_prev = v_new;
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
    /// Stamp all non-reactive (resistive + source) elements.
    fn stamp_static(&self, mna: &mut MnaSystem);

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
        stamper.stamp_static(&mut mna);

        // Stamp companion models for reactive elements
        match params.method {
            IntegrationMethod::BackwardEuler => {
                for cap in caps.iter() {
                    cap.stamp_be(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_be(&mut mna, h);
                }
            }
            IntegrationMethod::Trapezoidal => {
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h);
                }
            }
        }

        // Solve using cached symbolic factorization for large systems
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

        // Update reactive element states
        for cap in caps.iter_mut() {
            let vp = cap.node_pos.map(|i| solution[i]).unwrap_or(0.0);
            let vn = cap.node_neg.map(|i| solution[i]).unwrap_or(0.0);
            cap.update(vp - vn, h, params.method);
        }
        for ind in inds.iter_mut() {
            let vp = ind.node_pos.map(|i| solution[i]).unwrap_or(0.0);
            let vn = ind.node_neg.map(|i| solution[i]).unwrap_or(0.0);
            ind.update(vp - vn, h, params.method);
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
        stamper.stamp_static(&mut mna);

        match params.method {
            IntegrationMethod::BackwardEuler => {
                for cap in caps.iter() {
                    cap.stamp_be(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_be(&mut mna, h);
                }
            }
            IntegrationMethod::Trapezoidal => {
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h);
                }
            }
        }

        solution = if use_gmres {
            solve_transient_gmres(&mna, &config.gmres_config)?
        } else if mna_size >= SPARSE_THRESHOLD {
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

        for cap in caps.iter_mut() {
            let vp = cap.node_pos.map(|i| solution[i]).unwrap_or(0.0);
            let vn = cap.node_neg.map(|i| solution[i]).unwrap_or(0.0);
            cap.update(vp - vn, h, params.method);
        }
        for ind in inds.iter_mut() {
            let vp = ind.node_pos.map(|i| solution[i]).unwrap_or(0.0);
            let vn = ind.node_neg.map(|i| solution[i]).unwrap_or(0.0);
            ind.update(vp - vn, h, params.method);
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
        stamper.stamp_static(&mut mna);

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
            fn stamp_static(&self, mna: &mut MnaSystem) {
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
        fn stamp_static(&self, mna: &mut MnaSystem) {
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
}
