//! Expression evaluation and analysis.

use std::collections::HashMap;

use super::ast::{BinaryOp, Expr, UnaryOp};
use super::functions::{derivative_function, derivative_function_current, eval_function};

/// Context for expression evaluation.
#[derive(Debug, Clone, Default)]
pub struct EvalContext {
    /// Node voltages by name.
    pub voltages: HashMap<String, f64>,
    /// Branch currents by source name.
    pub currents: HashMap<String, f64>,
    /// Current simulation time.
    pub time: f64,
}

impl EvalContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a node voltage.
    pub fn set_voltage(&mut self, node: &str, voltage: f64) {
        self.voltages.insert(node.to_uppercase(), voltage);
    }

    /// Set a branch current.
    pub fn set_current(&mut self, source: &str, current: f64) {
        self.currents.insert(source.to_uppercase(), current);
    }

    /// Set the simulation time.
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }
}

impl Expr {
    /// Evaluate the expression in the given context.
    pub fn eval(&self, ctx: &EvalContext) -> f64 {
        match self {
            Expr::Constant(v) => *v,
            Expr::Voltage { node_pos, node_neg } => {
                let v_pos = ctx
                    .voltages
                    .get(&node_pos.to_uppercase())
                    .copied()
                    .unwrap_or(0.0);
                let v_neg = node_neg
                    .as_ref()
                    .map(|n| ctx.voltages.get(&n.to_uppercase()).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                v_pos - v_neg
            }
            Expr::Current { source_name } => ctx
                .currents
                .get(&source_name.to_uppercase())
                .copied()
                .unwrap_or(0.0),
            Expr::Time => ctx.time,
            Expr::BinaryOp { op, left, right } => {
                let l = left.eval(ctx);
                let r = right.eval(ctx);
                match op {
                    BinaryOp::Add => l + r,
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => {
                        if r.abs() < 1e-30 {
                            if l >= 0.0 { 1e30 } else { -1e30 }
                        } else {
                            l / r
                        }
                    }
                    BinaryOp::Pow => l.powf(r),
                }
            }
            Expr::UnaryOp { op, operand } => {
                let v = operand.eval(ctx);
                match op {
                    UnaryOp::Neg => -v,
                }
            }
            Expr::Function { name, args } => {
                let arg_values: Vec<f64> = args.iter().map(|a| a.eval(ctx)).collect();
                eval_function(name, &arg_values)
            }
        }
    }

    /// Compute the partial derivative with respect to a node voltage.
    ///
    /// Returns the symbolic derivative, evaluated at the current context.
    pub fn derivative_voltage(&self, node: &str, ctx: &EvalContext) -> f64 {
        let node_upper = node.to_uppercase();
        match self {
            Expr::Constant(_) => 0.0,
            Expr::Voltage { node_pos, node_neg } => {
                let mut deriv = 0.0;
                if node_pos.to_uppercase() == node_upper {
                    deriv += 1.0;
                }
                if let Some(neg) = node_neg {
                    if neg.to_uppercase() == node_upper {
                        deriv -= 1.0;
                    }
                }
                deriv
            }
            Expr::Current { .. } => 0.0,
            Expr::Time => 0.0,
            Expr::BinaryOp { op, left, right } => {
                let l = left.eval(ctx);
                let r = right.eval(ctx);
                let dl = left.derivative_voltage(node, ctx);
                let dr = right.derivative_voltage(node, ctx);
                match op {
                    BinaryOp::Add => dl + dr,
                    BinaryOp::Sub => dl - dr,
                    BinaryOp::Mul => dl * r + l * dr,
                    BinaryOp::Div => {
                        if r.abs() < 1e-30 {
                            0.0
                        } else {
                            (dl * r - l * dr) / (r * r)
                        }
                    }
                    BinaryOp::Pow => {
                        // d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f)
                        let f = l;
                        let g = r;
                        let df = dl;
                        let dg = dr;
                        if f.abs() < 1e-30 {
                            0.0
                        } else {
                            f.powf(g) * (dg * f.ln() + g * df / f)
                        }
                    }
                }
            }
            Expr::UnaryOp { op, operand } => {
                let d = operand.derivative_voltage(node, ctx);
                match op {
                    UnaryOp::Neg => -d,
                }
            }
            Expr::Function { name, args } => derivative_function(name, args, node, ctx),
        }
    }

    /// Compute the partial derivative with respect to a branch current.
    pub fn derivative_current(&self, source: &str, ctx: &EvalContext) -> f64 {
        let source_upper = source.to_uppercase();
        match self {
            Expr::Constant(_) => 0.0,
            Expr::Voltage { .. } => 0.0,
            Expr::Current { source_name } => {
                if source_name.to_uppercase() == source_upper {
                    1.0
                } else {
                    0.0
                }
            }
            Expr::Time => 0.0,
            Expr::BinaryOp { op, left, right } => {
                let l = left.eval(ctx);
                let r = right.eval(ctx);
                let dl = left.derivative_current(source, ctx);
                let dr = right.derivative_current(source, ctx);
                match op {
                    BinaryOp::Add => dl + dr,
                    BinaryOp::Sub => dl - dr,
                    BinaryOp::Mul => dl * r + l * dr,
                    BinaryOp::Div => {
                        if r.abs() < 1e-30 {
                            0.0
                        } else {
                            (dl * r - l * dr) / (r * r)
                        }
                    }
                    BinaryOp::Pow => {
                        let f = l;
                        let g = r;
                        let df = dl;
                        let dg = dr;
                        if f.abs() < 1e-30 {
                            0.0
                        } else {
                            f.powf(g) * (dg * f.ln() + g * df / f)
                        }
                    }
                }
            }
            Expr::UnaryOp { op, operand } => {
                let d = operand.derivative_current(source, ctx);
                match op {
                    UnaryOp::Neg => -d,
                }
            }
            Expr::Function { name, args } => derivative_function_current(name, args, source, ctx),
        }
    }

    /// Check if this expression depends on time.
    pub fn is_time_dependent(&self) -> bool {
        match self {
            Expr::Constant(_) => false,
            Expr::Voltage { .. } => false,
            Expr::Current { .. } => false,
            Expr::Time => true,
            Expr::BinaryOp { left, right, .. } => {
                left.is_time_dependent() || right.is_time_dependent()
            }
            Expr::UnaryOp { operand, .. } => operand.is_time_dependent(),
            Expr::Function { args, .. } => args.iter().any(|a| a.is_time_dependent()),
        }
    }

    /// Check if this expression is nonlinear (depends on voltages/currents in a nonlinear way).
    pub fn is_nonlinear(&self) -> bool {
        match self {
            Expr::Constant(_) | Expr::Time => false,
            Expr::Voltage { .. } | Expr::Current { .. } => false,
            Expr::BinaryOp { op, left, right } => {
                let left_has_var = left.has_voltage_or_current();
                let right_has_var = right.has_voltage_or_current();
                match op {
                    BinaryOp::Add | BinaryOp::Sub => left.is_nonlinear() || right.is_nonlinear(),
                    BinaryOp::Mul => {
                        // Nonlinear if both sides have variables, or if either is nonlinear
                        (left_has_var && right_has_var)
                            || left.is_nonlinear()
                            || right.is_nonlinear()
                    }
                    BinaryOp::Div => {
                        // Nonlinear if denominator has variables
                        right_has_var || left.is_nonlinear() || right.is_nonlinear()
                    }
                    BinaryOp::Pow => {
                        // Nonlinear if base has variables
                        left_has_var || left.is_nonlinear() || right.is_nonlinear()
                    }
                }
            }
            Expr::UnaryOp { operand, .. } => operand.is_nonlinear(),
            Expr::Function { args, .. } => {
                // Functions of voltages/currents are nonlinear
                args.iter()
                    .any(|a| a.has_voltage_or_current() || a.is_nonlinear())
            }
        }
    }

    /// Check if this expression contains voltage or current references.
    pub fn has_voltage_or_current(&self) -> bool {
        match self {
            Expr::Constant(_) | Expr::Time => false,
            Expr::Voltage { .. } | Expr::Current { .. } => true,
            Expr::BinaryOp { left, right, .. } => {
                left.has_voltage_or_current() || right.has_voltage_or_current()
            }
            Expr::UnaryOp { operand, .. } => operand.has_voltage_or_current(),
            Expr::Function { args, .. } => args.iter().any(|a| a.has_voltage_or_current()),
        }
    }

    /// Get all voltage node references in this expression.
    pub fn voltage_nodes(&self) -> Vec<String> {
        let mut nodes = Vec::new();
        self.collect_voltage_nodes(&mut nodes);
        nodes.sort();
        nodes.dedup();
        nodes
    }

    fn collect_voltage_nodes(&self, nodes: &mut Vec<String>) {
        match self {
            Expr::Voltage { node_pos, node_neg } => {
                nodes.push(node_pos.to_uppercase());
                if let Some(neg) = node_neg {
                    nodes.push(neg.to_uppercase());
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                left.collect_voltage_nodes(nodes);
                right.collect_voltage_nodes(nodes);
            }
            Expr::UnaryOp { operand, .. } => {
                operand.collect_voltage_nodes(nodes);
            }
            Expr::Function { args, .. } => {
                for arg in args {
                    arg.collect_voltage_nodes(nodes);
                }
            }
            _ => {}
        }
    }

    /// Get all current source references in this expression.
    pub fn current_sources(&self) -> Vec<String> {
        let mut sources = Vec::new();
        self.collect_current_sources(&mut sources);
        sources.sort();
        sources.dedup();
        sources
    }

    fn collect_current_sources(&self, sources: &mut Vec<String>) {
        match self {
            Expr::Current { source_name } => {
                sources.push(source_name.to_uppercase());
            }
            Expr::BinaryOp { left, right, .. } => {
                left.collect_current_sources(sources);
                right.collect_current_sources(sources);
            }
            Expr::UnaryOp { operand, .. } => {
                operand.collect_current_sources(sources);
            }
            Expr::Function { args, .. } => {
                for arg in args {
                    arg.collect_current_sources(sources);
                }
            }
            _ => {}
        }
    }
}
