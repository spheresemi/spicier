//! Transient analysis engine.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;

use crate::error::Result;
use crate::linear::{CachedSparseLu, SPARSE_THRESHOLD, solve_dense};

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
            solve_dense(mna.matrix(), mna.rhs())?
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

#[cfg(test)]
mod tests {
    use super::*;

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

        // Geq = C/h = 1e-6/1e-6 = 1.0
        assert!(
            (mna.matrix()[(0, 0)] - 1.0).abs() < 1e-10,
            "Geq = {} (expected 1.0)",
            mna.matrix()[(0, 0)]
        );

        // Ieq = Geq * V_prev = 1.0 * 2.5 = 2.5
        assert!(
            (mna.rhs()[0] - 2.5).abs() < 1e-10,
            "Ieq = {} (expected 2.5)",
            mna.rhs()[0]
        );
    }
}
