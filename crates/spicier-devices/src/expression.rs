//! Mathematical expression parsing and evaluation for behavioral sources.
//!
//! Supports expressions like:
//! - `V(1) * 2 + 1` - node voltage with arithmetic
//! - `I(V1) * 0.5` - branch current reference
//! - `sin(2 * pi * 1k * time)` - time-varying with functions
//! - `V(in, out) / 1k` - differential voltage

use std::collections::HashMap;
use std::f64::consts::{E, PI};

/// Expression AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric constant.
    Constant(f64),
    /// Node voltage: V(node) or V(node1, node2) for differential.
    Voltage {
        node_pos: String,
        node_neg: Option<String>,
    },
    /// Branch current: I(Vsource_name).
    Current { source_name: String },
    /// Time variable.
    Time,
    /// Binary operation.
    BinaryOp {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation.
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
    /// Function call.
    Function { name: String, args: Vec<Expr> },
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

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
                let v_pos = ctx.voltages.get(&node_pos.to_uppercase()).copied().unwrap_or(0.0);
                let v_neg = node_neg
                    .as_ref()
                    .map(|n| ctx.voltages.get(&n.to_uppercase()).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                v_pos - v_neg
            }
            Expr::Current { source_name } => {
                ctx.currents.get(&source_name.to_uppercase()).copied().unwrap_or(0.0)
            }
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
            Expr::Function { name, args } => {
                derivative_function(name, args, node, ctx)
            }
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
            Expr::Function { name, args } => {
                derivative_function_current(name, args, source, ctx)
            }
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
                    BinaryOp::Add | BinaryOp::Sub => {
                        left.is_nonlinear() || right.is_nonlinear()
                    }
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
                args.iter().any(|a| a.has_voltage_or_current() || a.is_nonlinear())
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

/// Evaluate a built-in function.
fn eval_function(name: &str, args: &[f64]) -> f64 {
    let name_upper = name.to_uppercase();
    match name_upper.as_str() {
        // Trigonometric
        "SIN" => args.first().copied().unwrap_or(0.0).sin(),
        "COS" => args.first().copied().unwrap_or(0.0).cos(),
        "TAN" => args.first().copied().unwrap_or(0.0).tan(),
        "ASIN" => args.first().copied().unwrap_or(0.0).asin(),
        "ACOS" => args.first().copied().unwrap_or(0.0).acos(),
        "ATAN" => args.first().copied().unwrap_or(0.0).atan(),
        "ATAN2" => {
            let y = args.first().copied().unwrap_or(0.0);
            let x = args.get(1).copied().unwrap_or(1.0);
            y.atan2(x)
        }
        "SINH" => args.first().copied().unwrap_or(0.0).sinh(),
        "COSH" => args.first().copied().unwrap_or(0.0).cosh(),
        "TANH" => args.first().copied().unwrap_or(0.0).tanh(),

        // Exponential/logarithmic
        "EXP" => args.first().copied().unwrap_or(0.0).exp(),
        "LOG" | "LN" => {
            let x = args.first().copied().unwrap_or(1.0);
            if x > 0.0 { x.ln() } else { -1e30 }
        }
        "LOG10" => {
            let x = args.first().copied().unwrap_or(1.0);
            if x > 0.0 { x.log10() } else { -1e30 }
        }
        "SQRT" => {
            let x = args.first().copied().unwrap_or(0.0);
            if x >= 0.0 { x.sqrt() } else { 0.0 }
        }
        "POW" => {
            let base = args.first().copied().unwrap_or(0.0);
            let exp = args.get(1).copied().unwrap_or(1.0);
            base.powf(exp)
        }

        // Absolute value and sign
        "ABS" => args.first().copied().unwrap_or(0.0).abs(),
        "SGN" | "SIGN" => {
            let x = args.first().copied().unwrap_or(0.0);
            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
        }

        // Min/max
        "MIN" => args.iter().copied().fold(f64::INFINITY, f64::min),
        "MAX" => args.iter().copied().fold(f64::NEG_INFINITY, f64::max),

        // Limiting functions
        "LIMIT" => {
            // limit(x, lo, hi) - clamp x to [lo, hi]
            let x = args.first().copied().unwrap_or(0.0);
            let lo = args.get(1).copied().unwrap_or(f64::NEG_INFINITY);
            let hi = args.get(2).copied().unwrap_or(f64::INFINITY);
            x.clamp(lo, hi)
        }

        // Step function
        "U" | "STEP" => {
            // u(x) = 0 for x < 0, 1 for x >= 0
            let x = args.first().copied().unwrap_or(0.0);
            if x >= 0.0 { 1.0 } else { 0.0 }
        }

        // Smooth step (sigmoid approximation)
        "URAMP" => {
            // uramp(x) = 0 for x < 0, x for x >= 0
            let x = args.first().copied().unwrap_or(0.0);
            if x >= 0.0 { x } else { 0.0 }
        }

        // Floor/ceil
        "FLOOR" => args.first().copied().unwrap_or(0.0).floor(),
        "CEIL" => args.first().copied().unwrap_or(0.0).ceil(),
        "ROUND" => args.first().copied().unwrap_or(0.0).round(),

        // If-then-else
        "IF" => {
            // if(cond, then, else)
            let cond = args.first().copied().unwrap_or(0.0);
            let then_val = args.get(1).copied().unwrap_or(0.0);
            let else_val = args.get(2).copied().unwrap_or(0.0);
            if cond != 0.0 { then_val } else { else_val }
        }

        _ => 0.0, // Unknown function
    }
}

/// Compute derivative of a function with respect to a voltage node.
fn derivative_function(name: &str, args: &[Expr], node: &str, ctx: &EvalContext) -> f64 {
    let name_upper = name.to_uppercase();

    // For single-argument functions: d/dx f(g(x)) = f'(g(x)) * g'(x)
    if args.len() == 1 {
        let g = args[0].eval(ctx);
        let dg = args[0].derivative_voltage(node, ctx);

        if dg.abs() < 1e-30 {
            return 0.0;
        }

        let df = match name_upper.as_str() {
            "SIN" => g.cos(),
            "COS" => -g.sin(),
            "TAN" => 1.0 / g.cos().powi(2),
            "ASIN" => 1.0 / (1.0 - g * g).sqrt(),
            "ACOS" => -1.0 / (1.0 - g * g).sqrt(),
            "ATAN" => 1.0 / (1.0 + g * g),
            "SINH" => g.cosh(),
            "COSH" => g.sinh(),
            "TANH" => 1.0 / g.cosh().powi(2),
            "EXP" => g.exp(),
            "LOG" | "LN" => 1.0 / g,
            "LOG10" => 1.0 / (g * 10.0_f64.ln()),
            "SQRT" => 0.5 / g.sqrt(),
            "ABS" => if g >= 0.0 { 1.0 } else { -1.0 },
            "U" | "STEP" => 0.0, // Discontinuous
            "URAMP" => if g >= 0.0 { 1.0 } else { 0.0 },
            "FLOOR" | "CEIL" | "ROUND" => 0.0, // Discontinuous
            _ => 0.0,
        };

        return df * dg;
    }

    // Multi-argument functions
    match name_upper.as_str() {
        "POW" => {
            if args.len() >= 2 {
                let base = args[0].eval(ctx);
                let exp = args[1].eval(ctx);
                let dbase = args[0].derivative_voltage(node, ctx);
                let dexp = args[1].derivative_voltage(node, ctx);

                if base.abs() < 1e-30 {
                    return 0.0;
                }

                // d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f)
                base.powf(exp) * (dexp * base.ln() + exp * dbase / base)
            } else {
                0.0
            }
        }
        "ATAN2" => {
            if args.len() >= 2 {
                let y = args[0].eval(ctx);
                let x = args[1].eval(ctx);
                let dy = args[0].derivative_voltage(node, ctx);
                let dx = args[1].derivative_voltage(node, ctx);
                let r2 = x * x + y * y;
                if r2.abs() < 1e-30 {
                    return 0.0;
                }
                (x * dy - y * dx) / r2
            } else {
                0.0
            }
        }
        "MIN" | "MAX" => {
            // Derivative of the selected argument
            if args.is_empty() {
                return 0.0;
            }
            let values: Vec<f64> = args.iter().map(|a| a.eval(ctx)).collect();
            let (idx, _) = if name_upper == "MIN" {
                values.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap()
            } else {
                values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap()
            };
            args[idx].derivative_voltage(node, ctx)
        }
        "IF" => {
            if args.len() >= 3 {
                let cond = args[0].eval(ctx);
                if cond != 0.0 {
                    args[1].derivative_voltage(node, ctx)
                } else {
                    args[2].derivative_voltage(node, ctx)
                }
            } else {
                0.0
            }
        }
        "LIMIT" => {
            if args.len() >= 3 {
                let x = args[0].eval(ctx);
                let lo = args[1].eval(ctx);
                let hi = args[2].eval(ctx);
                if x < lo || x > hi {
                    0.0 // At limits, derivative is 0
                } else {
                    args[0].derivative_voltage(node, ctx)
                }
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Compute derivative of a function with respect to a branch current.
fn derivative_function_current(name: &str, args: &[Expr], source: &str, ctx: &EvalContext) -> f64 {
    let name_upper = name.to_uppercase();

    // Same logic as voltage derivative, but using current derivatives
    if args.len() == 1 {
        let g = args[0].eval(ctx);
        let dg = args[0].derivative_current(source, ctx);

        if dg.abs() < 1e-30 {
            return 0.0;
        }

        let df = match name_upper.as_str() {
            "SIN" => g.cos(),
            "COS" => -g.sin(),
            "TAN" => 1.0 / g.cos().powi(2),
            "EXP" => g.exp(),
            "LOG" | "LN" => 1.0 / g,
            "SQRT" => 0.5 / g.sqrt(),
            "ABS" => if g >= 0.0 { 1.0 } else { -1.0 },
            _ => 0.0,
        };

        return df * dg;
    }

    0.0
}

/// Parse a mathematical expression from a string.
pub fn parse_expression(input: &str) -> Result<Expr, String> {
    let mut parser = ExprParser::new(input);
    parser.parse()
}

/// Expression parser using recursive descent.
struct ExprParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse(&mut self) -> Result<Expr, String> {
        self.skip_whitespace();
        let expr = self.parse_additive()?;
        self.skip_whitespace();
        if self.pos < self.input.len() {
            Err(format!("Unexpected character at position {}", self.pos))
        } else {
            Ok(expr)
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input.as_bytes()[self.pos] as char;
            if c.is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.as_bytes().get(self.pos).map(|&b| b as char)
    }

    fn advance(&mut self) {
        if self.pos < self.input.len() {
            self.pos += 1;
        }
    }

    fn parse_additive(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_multiplicative()?;

        loop {
            self.skip_whitespace();
            match self.peek() {
                Some('+') => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    left = Expr::BinaryOp {
                        op: BinaryOp::Add,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                Some('-') => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    left = Expr::BinaryOp {
                        op: BinaryOp::Sub,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_power()?;

        loop {
            self.skip_whitespace();
            match self.peek() {
                Some('*') => {
                    self.advance();
                    let right = self.parse_power()?;
                    left = Expr::BinaryOp {
                        op: BinaryOp::Mul,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                Some('/') => {
                    self.advance();
                    let right = self.parse_power()?;
                    left = Expr::BinaryOp {
                        op: BinaryOp::Div,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Expr, String> {
        let base = self.parse_unary()?;

        self.skip_whitespace();
        if self.peek() == Some('^') || self.check_str("**") {
            if self.peek() == Some('*') {
                self.advance(); // consume first *
            }
            self.advance(); // consume ^ or second *
            let exp = self.parse_power()?; // Right associative
            Ok(Expr::BinaryOp {
                op: BinaryOp::Pow,
                left: Box::new(base),
                right: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    fn check_str(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        self.skip_whitespace();
        match self.peek() {
            Some('-') => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                })
            }
            Some('+') => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        self.skip_whitespace();

        match self.peek() {
            Some('(') => {
                self.advance();
                let expr = self.parse_additive()?;
                self.skip_whitespace();
                if self.peek() != Some(')') {
                    return Err("Expected ')'".to_string());
                }
                self.advance();
                Ok(expr)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => self.parse_number(),
            Some(c) if c.is_ascii_alphabetic() || c == '_' => self.parse_identifier(),
            Some(c) => Err(format!("Unexpected character: '{}'", c)),
            None => Err("Unexpected end of expression".to_string()),
        }
    }

    fn parse_number(&mut self) -> Result<Expr, String> {
        let start = self.pos;
        let mut has_dot = false;
        let mut has_exp = false;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else if c == '.' && !has_dot && !has_exp {
                has_dot = true;
                self.advance();
            } else if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                self.advance();
                if self.peek() == Some('+') || self.peek() == Some('-') {
                    self.advance();
                }
            } else {
                break;
            }
        }

        let num_str = &self.input[start..self.pos];

        // Check for SPICE suffix
        let suffix_start = self.pos;
        if let Some(c) = self.peek() {
            if c.is_ascii_alphabetic() {
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        let suffix = &self.input[suffix_start..self.pos];
        let multiplier = parse_spice_suffix(suffix);

        let value: f64 = num_str.parse().map_err(|_| format!("Invalid number: {}", num_str))?;
        Ok(Expr::Constant(value * multiplier))
    }

    fn parse_identifier(&mut self) -> Result<Expr, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let ident = &self.input[start..self.pos];
        let ident_upper = ident.to_uppercase();

        // Check for constants
        match ident_upper.as_str() {
            "PI" => return Ok(Expr::Constant(PI)),
            "E" => return Ok(Expr::Constant(E)),
            "TIME" | "T" => return Ok(Expr::Time),
            _ => {}
        }

        self.skip_whitespace();

        // Check for function call or special forms
        if self.peek() == Some('(') {
            self.advance();

            // Special handling for V() and I()
            if ident_upper == "V" {
                return self.parse_voltage_reference();
            }
            if ident_upper == "I" {
                return self.parse_current_reference();
            }

            // Regular function call
            let args = self.parse_function_args()?;
            return Ok(Expr::Function {
                name: ident.to_string(),
                args,
            });
        }

        // Bare identifier - treat as V(ident) for convenience
        Ok(Expr::Voltage {
            node_pos: ident.to_string(),
            node_neg: None,
        })
    }

    fn parse_voltage_reference(&mut self) -> Result<Expr, String> {
        self.skip_whitespace();

        // Parse first node
        let node_pos = self.parse_node_name()?;

        self.skip_whitespace();

        let node_neg = if self.peek() == Some(',') {
            self.advance();
            self.skip_whitespace();
            Some(self.parse_node_name()?)
        } else {
            None
        };

        self.skip_whitespace();
        if self.peek() != Some(')') {
            return Err("Expected ')' in V()".to_string());
        }
        self.advance();

        Ok(Expr::Voltage { node_pos, node_neg })
    }

    fn parse_current_reference(&mut self) -> Result<Expr, String> {
        self.skip_whitespace();
        let source_name = self.parse_node_name()?;

        self.skip_whitespace();
        if self.peek() != Some(')') {
            return Err("Expected ')' in I()".to_string());
        }
        self.advance();

        Ok(Expr::Current { source_name })
    }

    fn parse_node_name(&mut self) -> Result<String, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        if self.pos == start {
            return Err("Expected node name".to_string());
        }

        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_function_args(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();

        self.skip_whitespace();
        if self.peek() == Some(')') {
            self.advance();
            return Ok(args);
        }

        loop {
            let arg = self.parse_additive()?;
            args.push(arg);

            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                Some(')') => {
                    self.advance();
                    break;
                }
                _ => return Err("Expected ',' or ')' in function arguments".to_string()),
            }
        }

        Ok(args)
    }
}

/// Parse SPICE suffix to multiplier.
fn parse_spice_suffix(suffix: &str) -> f64 {
    let suffix_upper = suffix.to_uppercase();
    match suffix_upper.as_str() {
        "" => 1.0,
        "T" => 1e12,
        "G" => 1e9,
        "MEG" | "X" => 1e6,
        "K" => 1e3,
        "M" => 1e-3,
        "U" => 1e-6,
        "N" => 1e-9,
        "P" => 1e-12,
        "F" => 1e-15,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_constant() {
        let expr = parse_expression("42").unwrap();
        assert_eq!(expr, Expr::Constant(42.0));
    }

    #[test]
    fn test_parse_constant_with_suffix() {
        let expr = parse_expression("1k").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), 1000.0);
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_expression("1 + 2").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), 3.0);
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse_expression("3 * 4").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), 12.0);
    }

    #[test]
    fn test_parse_precedence() {
        let expr = parse_expression("2 + 3 * 4").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), 14.0);
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = parse_expression("(2 + 3) * 4").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), 20.0);
    }

    #[test]
    fn test_parse_unary_minus() {
        let expr = parse_expression("-5").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), -5.0);
    }

    #[test]
    fn test_parse_power() {
        let expr = parse_expression("2^3").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), 8.0);
    }

    #[test]
    fn test_parse_voltage_reference() {
        let expr = parse_expression("V(1)").unwrap();
        let mut ctx = EvalContext::new();
        ctx.set_voltage("1", 5.0);
        assert_eq!(expr.eval(&ctx), 5.0);
    }

    #[test]
    fn test_parse_differential_voltage() {
        let expr = parse_expression("V(in, out)").unwrap();
        let mut ctx = EvalContext::new();
        ctx.set_voltage("in", 10.0);
        ctx.set_voltage("out", 3.0);
        assert_eq!(expr.eval(&ctx), 7.0);
    }

    #[test]
    fn test_parse_current_reference() {
        let expr = parse_expression("I(V1)").unwrap();
        let mut ctx = EvalContext::new();
        ctx.set_current("V1", 0.001);
        assert_eq!(expr.eval(&ctx), 0.001);
    }

    #[test]
    fn test_parse_time() {
        let expr = parse_expression("time").unwrap();
        let mut ctx = EvalContext::new();
        ctx.set_time(0.5);
        assert_eq!(expr.eval(&ctx), 0.5);
    }

    #[test]
    fn test_parse_function_sin() {
        let expr = parse_expression("sin(0)").unwrap();
        let ctx = EvalContext::new();
        assert!((expr.eval(&ctx) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_function_exp() {
        let expr = parse_expression("exp(1)").unwrap();
        let ctx = EvalContext::new();
        assert!((expr.eval(&ctx) - E).abs() < 1e-10);
    }

    #[test]
    fn test_parse_pi_constant() {
        let expr = parse_expression("pi").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(&ctx), PI);
    }

    #[test]
    fn test_complex_expression() {
        // V(1) * 2 + sin(pi / 2)
        let expr = parse_expression("V(1) * 2 + sin(pi / 2)").unwrap();
        let mut ctx = EvalContext::new();
        ctx.set_voltage("1", 5.0);
        let result = expr.eval(&ctx);
        assert!((result - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_constant() {
        let expr = parse_expression("42").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.derivative_voltage("1", &ctx), 0.0);
    }

    #[test]
    fn test_derivative_voltage() {
        let expr = parse_expression("V(1)").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.derivative_voltage("1", &ctx), 1.0);
        assert_eq!(expr.derivative_voltage("2", &ctx), 0.0);
    }

    #[test]
    fn test_derivative_linear() {
        // 2 * V(1) + 3
        let expr = parse_expression("2 * V(1) + 3").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.derivative_voltage("1", &ctx), 2.0);
    }

    #[test]
    fn test_derivative_quadratic() {
        // V(1) * V(1)
        let expr = parse_expression("V(1) * V(1)").unwrap();
        let mut ctx = EvalContext::new();
        ctx.set_voltage("1", 3.0);
        // d/dV1 (V1^2) = 2*V1 = 6
        assert_eq!(expr.derivative_voltage("1", &ctx), 6.0);
    }

    #[test]
    fn test_is_nonlinear() {
        assert!(!parse_expression("5").unwrap().is_nonlinear());
        assert!(!parse_expression("V(1)").unwrap().is_nonlinear());
        assert!(!parse_expression("2 * V(1) + 3").unwrap().is_nonlinear());
        assert!(parse_expression("V(1) * V(2)").unwrap().is_nonlinear());
        assert!(parse_expression("V(1) * V(1)").unwrap().is_nonlinear());
        assert!(parse_expression("exp(V(1))").unwrap().is_nonlinear());
    }

    #[test]
    fn test_is_time_dependent() {
        assert!(!parse_expression("V(1)").unwrap().is_time_dependent());
        assert!(parse_expression("time").unwrap().is_time_dependent());
        assert!(parse_expression("sin(2 * pi * 1k * time)").unwrap().is_time_dependent());
    }

    #[test]
    fn test_voltage_nodes() {
        let expr = parse_expression("V(1) + V(2, 3)").unwrap();
        let nodes = expr.voltage_nodes();
        assert_eq!(nodes, vec!["1", "2", "3"]);
    }

    #[test]
    fn test_current_sources() {
        let expr = parse_expression("I(V1) + I(V2)").unwrap();
        let sources = expr.current_sources();
        assert_eq!(sources, vec!["V1", "V2"]);
    }
}
