//! Expression AST types.

/// Expression AST node.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnaryOp {
    Neg,
}
