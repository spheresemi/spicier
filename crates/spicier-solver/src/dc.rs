//! DC operating point and DC sweep analysis.

use nalgebra::DVector;
use spicier_core::NodeId;
use spicier_core::mna::MnaSystem;

use crate::error::Result;
use crate::linear::solve_dense;

/// DC sweep parameters.
#[derive(Debug, Clone)]
pub struct DcSweepParams {
    /// Name of the source to sweep.
    pub source_name: String,
    /// Start value.
    pub start: f64,
    /// Stop value.
    pub stop: f64,
    /// Step size.
    pub step: f64,
}

/// Callback for stamping a circuit with a swept source value.
pub trait DcSweepStamper {
    /// Stamp the circuit into the MNA system with the given swept source value.
    fn stamp_with_sweep(&self, mna: &mut MnaSystem, source_name: &str, value: f64);

    /// Number of nodes (excluding ground).
    fn num_nodes(&self) -> usize;

    /// Number of branch current variables.
    fn num_vsources(&self) -> usize;
}

/// Result of a DC sweep analysis.
#[derive(Debug, Clone)]
pub struct DcSweepResult {
    /// Name of the swept source.
    pub source_name: String,
    /// Sweep values at each point.
    pub sweep_values: Vec<f64>,
    /// DC solution at each sweep point.
    pub solutions: Vec<DcSolution>,
}

impl DcSweepResult {
    /// Get the voltage at a node across all sweep points.
    pub fn voltage_waveform(&self, node: NodeId) -> Vec<(f64, f64)> {
        self.sweep_values
            .iter()
            .zip(self.solutions.iter())
            .map(|(&sv, sol)| (sv, sol.voltage(node)))
            .collect()
    }

    /// Get a branch current across all sweep points.
    pub fn current_waveform(&self, index: usize) -> Vec<(f64, f64)> {
        self.sweep_values
            .iter()
            .zip(self.solutions.iter())
            .map(|(&sv, sol)| (sv, sol.current(index)))
            .collect()
    }
}

/// Result of a DC operating point analysis.
#[derive(Debug, Clone)]
pub struct DcSolution {
    /// Node voltages (indexed by node number - 1, ground is implicit 0V).
    pub node_voltages: DVector<f64>,
    /// Branch currents through voltage sources and inductors.
    pub branch_currents: DVector<f64>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
}

impl DcSolution {
    /// Get the voltage at a node.
    pub fn voltage(&self, node: NodeId) -> f64 {
        if node.is_ground() {
            0.0
        } else {
            let idx = (node.as_u32() - 1) as usize;
            if idx < self.num_nodes {
                self.node_voltages[idx]
            } else {
                0.0
            }
        }
    }

    /// Get the voltage difference between two nodes.
    pub fn voltage_diff(&self, node_pos: NodeId, node_neg: NodeId) -> f64 {
        self.voltage(node_pos) - self.voltage(node_neg)
    }

    /// Get a branch current by index.
    pub fn current(&self, index: usize) -> f64 {
        if index < self.branch_currents.len() {
            self.branch_currents[index]
        } else {
            0.0
        }
    }
}

/// Run a DC sweep analysis.
///
/// Sweeps a source from `start` to `stop` in increments of `step`,
/// solving the DC operating point at each value.
pub fn solve_dc_sweep(
    stamper: &dyn DcSweepStamper,
    params: &DcSweepParams,
) -> Result<DcSweepResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();

    // Generate sweep values
    let mut sweep_values = Vec::new();
    let direction = if params.step > 0.0 { 1.0 } else { -1.0 };
    let mut value = params.start;
    loop {
        sweep_values.push(value);
        value += params.step;
        if direction * value > direction * params.stop * (1.0 + 1e-10) {
            break;
        }
    }

    let mut solutions = Vec::with_capacity(sweep_values.len());

    for &sv in &sweep_values {
        let mut mna = MnaSystem::new(num_nodes, num_vsources);
        stamper.stamp_with_sweep(&mut mna, &params.source_name, sv);
        let sol = solve_dc(&mna)?;
        solutions.push(sol);
    }

    Ok(DcSweepResult {
        source_name: params.source_name.clone(),
        sweep_values,
        solutions,
    })
}

/// Solve the DC operating point for a pre-assembled MNA system.
pub fn solve_dc(mna: &MnaSystem) -> Result<DcSolution> {
    let solution = solve_dense(mna.matrix(), mna.rhs())?;

    let num_nodes = mna.num_nodes;
    let num_vsources = mna.num_vsources;

    // Split solution into node voltages and branch currents
    let node_voltages = DVector::from_iterator(num_nodes, solution.iter().take(num_nodes).copied());

    let branch_currents =
        DVector::from_iterator(num_vsources, solution.iter().skip(num_nodes).copied());

    Ok(DcSolution {
        node_voltages,
        branch_currents,
        num_nodes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voltage_divider() {
        // Simple voltage divider: V1 = 10V, R1 = R2 = 1k
        // Expected: V(node1) = 10V, V(node2) = 5V
        //
        //  V1(+) --- node1 --- R1 --- node2 --- R2 --- GND
        //   |                                          |
        //  GND ----------------------------------------+

        let mut mna = MnaSystem::new(2, 1); // 2 nodes, 1 voltage source

        // V1: 10V between node1 (+) and ground (-)
        // Current variable index 0
        mna.stamp_voltage_source(Some(0), None, 0, 10.0);

        // R1: 1k between node1 and node2
        let g1 = 1.0 / 1000.0;
        mna.stamp_conductance(Some(0), Some(1), g1);

        // R2: 1k between node2 and ground
        let g2 = 1.0 / 1000.0;
        mna.stamp_conductance(Some(1), None, g2);

        let solution = solve_dc(&mna).unwrap();

        // V(node1) should be 10V
        assert!((solution.voltage(NodeId::new(1)) - 10.0).abs() < 1e-10);

        // V(node2) should be 5V (voltage divider)
        assert!((solution.voltage(NodeId::new(2)) - 5.0).abs() < 1e-10);

        // Current through V1 should be -5mA (into the source)
        // I = (10-5)/1000 = 5mA flowing through circuit
        assert!((solution.current(0) + 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_current_divider() {
        // Current divider: I1 = 10mA, R1 = R2 = 1k in parallel
        // Expected: V(node1) = 5V, 5mA through each resistor
        //
        //  I1 --> node1 --+-- R1 --+-- GND
        //                 |        |
        //                 +-- R2 --+

        let mut mna = MnaSystem::new(1, 0); // 1 node, no voltage sources

        // I1: 10mA into node1
        mna.stamp_current_source(None, Some(0), 0.010);

        // R1: 1k between node1 and ground
        let g1 = 1.0 / 1000.0;
        mna.stamp_conductance(Some(0), None, g1);

        // R2: 1k between node1 and ground
        let g2 = 1.0 / 1000.0;
        mna.stamp_conductance(Some(0), None, g2);

        let solution = solve_dc(&mna).unwrap();

        // V(node1) = I * R_parallel = 0.01 * 500 = 5V
        assert!((solution.voltage(NodeId::new(1)) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dc_sweep_voltage_divider() {
        // Sweep V1 from 0 to 10V in 1V steps
        // Voltage divider: R1 = R2 = 1k → V(node2) = V1/2
        struct DividerStamper;

        impl DcSweepStamper for DividerStamper {
            fn stamp_with_sweep(&self, mna: &mut MnaSystem, _source_name: &str, value: f64) {
                // V1 at node 0
                mna.stamp_voltage_source(Some(0), None, 0, value);
                // R1 from node 0 to node 1
                mna.stamp_conductance(Some(0), Some(1), 1.0 / 1000.0);
                // R2 from node 1 to ground
                mna.stamp_conductance(Some(1), None, 1.0 / 1000.0);
            }

            fn num_nodes(&self) -> usize {
                2
            }
            fn num_vsources(&self) -> usize {
                1
            }
        }

        let params = DcSweepParams {
            source_name: "V1".to_string(),
            start: 0.0,
            stop: 10.0,
            step: 1.0,
        };

        let result = solve_dc_sweep(&DividerStamper, &params).unwrap();

        assert_eq!(result.sweep_values.len(), 11); // 0, 1, ..., 10
        assert_eq!(result.solutions.len(), 11);

        // At each point, V(node2) = V1/2
        for (i, &sv) in result.sweep_values.iter().enumerate() {
            let v2 = result.solutions[i].voltage(NodeId::new(2));
            assert!(
                (v2 - sv / 2.0).abs() < 1e-10,
                "At V1={}, V(2)={} (expected {})",
                sv,
                v2,
                sv / 2.0
            );
        }

        // Test voltage_waveform accessor
        let waveform = result.voltage_waveform(NodeId::new(2));
        assert_eq!(waveform.len(), 11);
        assert!((waveform[5].0 - 5.0).abs() < 1e-10); // sweep value
        assert!((waveform[5].1 - 2.5).abs() < 1e-10); // V(2)=5/2=2.5
    }

    #[test]
    fn test_ground_voltage() {
        let solution = DcSolution {
            node_voltages: DVector::from_vec(vec![5.0]),
            branch_currents: DVector::from_vec(vec![]),
            num_nodes: 1,
        };

        assert_eq!(solution.voltage(NodeId::GROUND), 0.0);
    }
}
