//! Built-in function evaluation and derivatives.

use super::{EvalContext, Expr};

/// Evaluate a built-in function.
pub fn eval_function(name: &str, args: &[f64]) -> f64 {
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
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
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
pub fn derivative_function(name: &str, args: &[Expr], node: &str, ctx: &EvalContext) -> f64 {
    let name_upper = name.to_uppercase();

    // For single-argument functions: d/dx f(g(x)) = f'(g(x)) * g'(x)
    if args.len() == 1 {
        let g = args[0].eval(ctx);
        let dg = args[0].derivative_voltage(node, ctx);

        if dg.abs() < 1e-30 {
            return 0.0;
        }

        let df = single_arg_derivative(&name_upper, g);
        return df * dg;
    }

    // Multi-argument functions
    multi_arg_derivative_voltage(&name_upper, args, node, ctx)
}

/// Compute derivative of a function with respect to a branch current.
pub fn derivative_function_current(name: &str, args: &[Expr], source: &str, ctx: &EvalContext) -> f64 {
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
            "ABS" => {
                if g >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => 0.0,
        };

        return df * dg;
    }

    0.0
}

/// Derivative of single-argument functions.
fn single_arg_derivative(name: &str, g: f64) -> f64 {
    match name {
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
        "ABS" => {
            if g >= 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        "U" | "STEP" => 0.0, // Discontinuous
        "URAMP" => {
            if g >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        "FLOOR" | "CEIL" | "ROUND" => 0.0, // Discontinuous
        _ => 0.0,
    }
}

/// Derivative of multi-argument functions with respect to voltage.
fn multi_arg_derivative_voltage(name: &str, args: &[Expr], node: &str, ctx: &EvalContext) -> f64 {
    match name {
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
            let (idx, _) = if name == "MIN" {
                values
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
            } else {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
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
