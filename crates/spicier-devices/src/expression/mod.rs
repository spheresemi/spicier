//! Mathematical expression parsing and evaluation for behavioral sources.
//!
//! Supports expressions like:
//! - `V(1) * 2 + 1` - node voltage with arithmetic
//! - `I(V1) * 0.5` - branch current reference
//! - `sin(2 * pi * 1k * time)` - time-varying with functions
//! - `V(in, out) / 1k` - differential voltage

mod ast;
mod eval;
mod functions;
mod parser;

pub use ast::{BinaryOp, Expr, UnaryOp};
pub use eval::EvalContext;
pub use parser::parse_expression;

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, PI};

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
        assert!(
            parse_expression("sin(2 * pi * 1k * time)")
                .unwrap()
                .is_time_dependent()
        );
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
