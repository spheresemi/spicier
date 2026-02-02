//! Behavioral sources (B elements) for arbitrary mathematical expressions.
//!
//! B elements allow defining voltage or current sources using mathematical
//! expressions that can depend on node voltages, branch currents, and time.
//!
//! # Examples
//!
//! ```spice
//! * Voltage-controlled voltage source (2x gain)
//! B1 out 0 V=V(in)*2
//!
//! * Nonlinear resistor (acts like 1k resistor)
//! B2 1 2 I=V(1,2)/1k
//!
//! * Time-varying source
//! B3 out 0 V=sin(2*pi*1k*time)
//!
//! * Current-controlled source
//! B4 out 0 I=I(V1)*0.5
//! ```

use crate::expression::{Expr, EvalContext};
use crate::stamp::Stamp;
use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, Stamper, TransientDeviceInfo};
use spicier_core::{Element, NodeId};

/// Convert a NodeId to an MNA matrix index (None for ground).
fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

/// Behavioral voltage source: B name n+ n- V=expression
#[derive(Debug, Clone)]
pub struct BehavioralVoltageSource {
    name: String,
    node_pos: NodeId,
    node_neg: NodeId,
    branch_index: usize,
    expression: Expr,
}

impl BehavioralVoltageSource {
    /// Create a new behavioral voltage source.
    pub fn new(
        name: &str,
        node_pos: NodeId,
        node_neg: NodeId,
        branch_index: usize,
        expression: Expr,
    ) -> Self {
        Self {
            name: name.to_string(),
            node_pos,
            node_neg,
            branch_index,
            expression,
        }
    }

    /// Get the expression.
    pub fn expression(&self) -> &Expr {
        &self.expression
    }

    /// Check if this source is time-dependent.
    pub fn is_time_dependent(&self) -> bool {
        self.expression.is_time_dependent()
    }

    /// Stamp the voltage source with a specific voltage value.
    fn stamp_voltage(&self, mna: &mut MnaSystem, voltage: f64) {
        let i = node_to_index(self.node_pos);
        let j = node_to_index(self.node_neg);
        mna.stamp_voltage_source(i, j, self.branch_index, voltage);
    }
}

impl Stamp for BehavioralVoltageSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        // For DC analysis, evaluate expr at zero voltages (initial guess)
        let ctx = EvalContext::new();
        let voltage = self.expression.eval(&ctx);
        self.stamp_voltage(mna, voltage);
    }
}

impl Element for BehavioralVoltageSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }

    fn num_current_vars(&self) -> usize {
        1
    }
}

impl Stamper for BehavioralVoltageSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn num_current_vars(&self) -> usize {
        1
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn branch_index(&self) -> Option<usize> {
        Some(self.branch_index)
    }

    fn ac_info(&self) -> AcDeviceInfo {
        // For AC analysis, behavioral sources are 0 AC unless explicitly specified
        // Time-varying or nonlinear expressions don't have a simple AC representation
        if self.expression.is_nonlinear() || self.expression.is_time_dependent() {
            AcDeviceInfo::None
        } else {
            AcDeviceInfo::VoltageSource {
                node_pos: node_to_index(self.node_pos),
                node_neg: node_to_index(self.node_neg),
                branch_idx: self.branch_index,
                ac_mag: 0.0, // AC source is 0 unless explicitly specified
            }
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::None
    }

    fn is_nonlinear(&self) -> bool {
        self.expression.is_nonlinear()
    }

    fn stamp_nonlinear(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        // Build evaluation context from current solution
        let mut ctx = EvalContext::new();

        // Populate voltages from solution
        for node in self.expression.voltage_nodes() {
            if let Ok(idx) = node.parse::<usize>() {
                if idx > 0 && idx <= solution.len() {
                    ctx.set_voltage(&node, solution[idx - 1]);
                }
            }
        }

        let voltage = self.expression.eval(&ctx);
        self.stamp_voltage(mna, voltage);
    }

    fn stamp_at_time(&self, mna: &mut MnaSystem, time: f64) {
        let mut ctx = EvalContext::new();
        ctx.set_time(time);

        let voltage = self.expression.eval(&ctx);
        self.stamp_voltage(mna, voltage);
    }
}

/// Behavioral current source: B name n+ n- I=expression
#[derive(Debug, Clone)]
pub struct BehavioralCurrentSource {
    name: String,
    node_pos: NodeId,
    node_neg: NodeId,
    expression: Expr,
}

impl BehavioralCurrentSource {
    /// Create a new behavioral current source.
    pub fn new(name: &str, node_pos: NodeId, node_neg: NodeId, expression: Expr) -> Self {
        Self {
            name: name.to_string(),
            node_pos,
            node_neg,
            expression,
        }
    }

    /// Get the expression.
    pub fn expression(&self) -> &Expr {
        &self.expression
    }

    /// Check if this source is time-dependent.
    pub fn is_time_dependent(&self) -> bool {
        self.expression.is_time_dependent()
    }
}

impl Stamp for BehavioralCurrentSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Current source stamp: current flows from n+ to n-
        let ctx = EvalContext::new();
        let current = self.expression.eval(&ctx);
        mna.stamp_current_source(node_to_index(self.node_pos), node_to_index(self.node_neg), current);
    }
}

impl Element for BehavioralCurrentSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_pos, self.node_neg]
    }

    fn num_current_vars(&self) -> usize {
        0
    }
}

impl Stamper for BehavioralCurrentSource {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        if self.expression.is_nonlinear() || self.expression.is_time_dependent() {
            AcDeviceInfo::None
        } else {
            AcDeviceInfo::CurrentSource {
                node_pos: node_to_index(self.node_pos),
                node_neg: node_to_index(self.node_neg),
                ac_mag: 0.0,
            }
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::None
    }

    fn is_nonlinear(&self) -> bool {
        self.expression.is_nonlinear()
    }

    fn stamp_nonlinear(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        // Build evaluation context from current solution
        let mut ctx = EvalContext::new();

        // Populate voltages from solution
        for node in self.expression.voltage_nodes() {
            if let Ok(idx) = node.parse::<usize>() {
                if idx > 0 && idx <= solution.len() {
                    ctx.set_voltage(&node, solution[idx - 1]);
                }
            }
        }

        let current = self.expression.eval(&ctx);

        // Stamp the current source
        mna.stamp_current_source(node_to_index(self.node_pos), node_to_index(self.node_neg), current);

        // Stamp the linearized conductances (Jacobian contributions)
        // For I = f(V), we need dI/dV contributions to the Jacobian
        for node in self.expression.voltage_nodes() {
            if let Ok(idx) = node.parse::<usize>() {
                if idx > 0 && idx <= solution.len() {
                    let deriv = self.expression.derivative_voltage(&node, &ctx);

                    if deriv.abs() > 1e-15 {
                        let node_id = NodeId::new(idx as u32);
                        let i = node_to_index(self.node_pos);
                        let j = node_to_index(self.node_neg);
                        let k = node_to_index(node_id);

                        // dI/dVnode contributes to the Jacobian
                        // Current flows from node_pos to node_neg
                        // So +deriv at node_pos row, -deriv at node_neg row
                        if let (Some(row), Some(col)) = (i, k) {
                            mna.add_element(row, col, deriv);
                        }
                        if let (Some(row), Some(col)) = (j, k) {
                            mna.add_element(row, col, -deriv);
                        }

                        // Adjust RHS for linearization: I = I(V0) + dI/dV * (V - V0)
                        // RHS contribution: -dI/dV * V0
                        let v_node = solution[idx - 1];
                        let correction = deriv * v_node;
                        if let Some(row) = i {
                            mna.add_rhs(row, correction);
                        }
                        if let Some(row) = j {
                            mna.add_rhs(row, -correction);
                        }
                    }
                }
            }
        }
    }

    fn stamp_at_time(&self, mna: &mut MnaSystem, time: f64) {
        let mut ctx = EvalContext::new();
        ctx.set_time(time);

        let current = self.expression.eval(&ctx);
        mna.stamp_current_source(node_to_index(self.node_pos), node_to_index(self.node_neg), current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::parse_expression;

    #[test]
    fn test_behavioral_voltage_source_constant() {
        let expr = parse_expression("5").unwrap();
        let bv = BehavioralVoltageSource::new("B1", NodeId::new(1), NodeId::GROUND, 0, expr);

        let mut mna = MnaSystem::new(1, 1);
        Stamp::stamp(&bv, &mut mna);

        // Check that the voltage source is stamped correctly
        let matrix = mna.to_dense_matrix();
        assert_eq!(matrix[(0, 1)], 1.0); // node 1 to branch
        assert_eq!(matrix[(1, 0)], 1.0); // branch to node 1

        let rhs = mna.rhs();
        assert_eq!(rhs[1], 5.0);
    }

    #[test]
    fn test_behavioral_current_source_constant() {
        let expr = parse_expression("1m").unwrap();
        let bc = BehavioralCurrentSource::new("B1", NodeId::new(1), NodeId::GROUND, expr);

        let mut mna = MnaSystem::new(1, 0);
        Stamp::stamp(&bc, &mut mna);

        let rhs = mna.rhs();
        // Current flows from node_pos to node_neg, so -I at node_pos
        assert!((rhs[0] - (-0.001)).abs() < 1e-10);
    }

    #[test]
    fn test_behavioral_voltage_source_expression_linear() {
        // B1 1 0 V=V(2)*2 - this is linear (scaling)
        let expr = parse_expression("V(2) * 2").unwrap();
        let bv = BehavioralVoltageSource::new("B1", NodeId::new(1), NodeId::GROUND, 0, expr);

        assert!(!bv.is_nonlinear()); // Linear scaling
        assert!(!bv.is_time_dependent());
    }

    #[test]
    fn test_behavioral_voltage_source_expression_nonlinear() {
        // B1 1 0 V=V(2)*V(2) - this is nonlinear (quadratic)
        let expr = parse_expression("V(2) * V(2)").unwrap();
        let bv = BehavioralVoltageSource::new("B1", NodeId::new(1), NodeId::GROUND, 0, expr);

        assert!(bv.is_nonlinear()); // Quadratic is nonlinear
        assert!(!bv.is_time_dependent());
    }

    #[test]
    fn test_behavioral_current_source_time_varying() {
        let expr = parse_expression("sin(2 * pi * 1k * time)").unwrap();
        let bc = BehavioralCurrentSource::new("B1", NodeId::new(1), NodeId::GROUND, expr);

        assert!(bc.is_time_dependent());
    }

    #[test]
    fn test_behavioral_current_source_linear() {
        // I = V(1) / 1k - acts like a 1k resistor
        let expr = parse_expression("V(1) / 1k").unwrap();
        let bc = BehavioralCurrentSource::new("B1", NodeId::new(1), NodeId::GROUND, expr);

        // V(1)/1k has V(1) in numerator with constant denominator - linear
        assert!(!bc.is_nonlinear());
    }

    #[test]
    fn test_behavioral_voltage_at_time() {
        let expr = parse_expression("sin(2 * pi * 1k * time)").unwrap();
        let bv = BehavioralVoltageSource::new("B1", NodeId::new(1), NodeId::GROUND, 0, expr);

        let mut mna = MnaSystem::new(1, 1);
        // At t=0.25ms, sin(2π * 1000 * 0.00025) = sin(π/2) = 1.0
        bv.stamp_at_time(&mut mna, 0.00025);

        let rhs = mna.rhs();
        assert!((rhs[1] - 1.0).abs() < 1e-10);
    }
}
