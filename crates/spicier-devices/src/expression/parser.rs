//! Expression parsing using recursive descent.

use std::collections::HashSet;
use std::f64::consts::{E, PI};

use super::ast::{BinaryOp, Expr, UnaryOp};

/// Parse a mathematical expression from a string.
pub fn parse_expression(input: &str) -> Result<Expr, String> {
    let mut parser = ExprParser::new(input, None);
    parser.parse()
}

/// Parse a mathematical expression with parameter context.
///
/// Identifiers that match a parameter name will be parsed as `Expr::Parameter`
/// instead of being treated as voltage references.
pub fn parse_expression_with_params(
    input: &str,
    param_names: &HashSet<String>,
) -> Result<Expr, String> {
    let mut parser = ExprParser::new(input, Some(param_names));
    parser.parse()
}

/// Expression parser using recursive descent.
struct ExprParser<'a> {
    input: &'a str,
    pos: usize,
    /// Known parameter names (uppercase).
    param_names: Option<&'a HashSet<String>>,
}

impl<'a> ExprParser<'a> {
    fn new(input: &'a str, param_names: Option<&'a HashSet<String>>) -> Self {
        Self {
            input,
            pos: 0,
            param_names,
        }
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

        let value: f64 = num_str
            .parse()
            .map_err(|_| format!("Invalid number: {}", num_str))?;
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

        // Check if identifier is a known parameter
        if let Some(params) = &self.param_names {
            if params.contains(&ident_upper) {
                return Ok(Expr::Parameter { name: ident_upper });
            }
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
