//! Subcircuit parsing and expansion (.SUBCKT/.ENDS, X instances).

use std::collections::HashMap;

use spicier_core::units::parse_value;
use spicier_devices::diode::{Diode, DiodeParams};
use spicier_devices::mosfet::{Mosfet, MosfetParams, MosfetType};
use spicier_devices::passive::{Capacitor, Inductor, Resistor};
use spicier_devices::sources::{CurrentSource, VoltageSource};

use crate::error::{Error, Result};
use crate::lexer::{Lexer, SpannedToken, Token};

use super::types::RawElementLine;
use super::{ModelDefinition, ParamContext, Parser};

impl<'a> Parser<'a> {
    /// Parse Xname node1 node2 ... subckt_name [PARAMS: param=value ...]
    ///
    /// If we're inside a subcircuit definition, store as raw line.
    /// Otherwise, expand the subcircuit inline.
    pub(super) fn parse_subcircuit_instance(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume instance name

        // If inside a subcircuit definition, capture the raw line for later expansion
        if self.current_subckt.is_some() {
            let raw_line = self.capture_x_instance_line(name);
            let subckt = self.current_subckt.as_mut().unwrap();
            subckt.instances.push(RawElementLine { line: raw_line });
            self.skip_to_eol();
            return Ok(());
        }

        // At top level - parse and expand immediately
        // Collect all node names until we hit EOL or PARAMS:
        // The last token before PARAMS: (or EOL) is the subcircuit name, others are connections
        let mut tokens: Vec<String> = Vec::new();
        let mut instance_params: HashMap<String, f64> = HashMap::new();

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let n_upper = n.to_uppercase();
                    if n_upper == "PARAMS" || n_upper.starts_with("PARAMS:") {
                        self.advance();
                        // Parse instance parameter overrides
                        instance_params = self.parse_instance_params(line)?;
                        break;
                    } else {
                        tokens.push(n.clone());
                        self.advance();
                    }
                }
                Token::Value(n) => {
                    tokens.push(n.clone());
                    self.advance();
                }
                Token::CurlyExpr(expr) => {
                    // Curly expression in instance line - store as-is for later evaluation
                    tokens.push(format!("{{{}}}", expr));
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        if tokens.is_empty() {
            return Err(Error::ParseError {
                line,
                message: format!(
                    "Subcircuit instance {} requires nodes and subcircuit name",
                    name
                ),
            });
        }

        // Last token is the subcircuit name
        let subckt_name = tokens.pop().unwrap().to_uppercase();
        let connection_nodes = tokens;

        // At top level - first register connection nodes to ensure they get proper IDs
        // before any internal subcircuit nodes are created
        for node_name in &connection_nodes {
            self.get_or_create_node(node_name);
        }
        // Then expand the subcircuit with params
        self.expand_subcircuit_with_params(
            name,
            &connection_nodes,
            &subckt_name,
            &instance_params,
            line,
        )?;

        self.skip_to_eol();
        Ok(())
    }

    /// Capture an X instance line as raw text for subcircuit storage.
    /// Preserves curly expressions without evaluation.
    fn capture_x_instance_line(&mut self, name: &str) -> String {
        let mut parts = vec![name.to_string()];

        // Collect remaining tokens until EOL
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) | Token::Value(n) => {
                    parts.push(n.clone());
                    self.advance();
                }
                Token::CurlyExpr(expr) => {
                    // Preserve curly expressions with braces
                    parts.push(format!("{{{}}}", expr));
                    self.advance();
                }
                Token::Equals => {
                    parts.push("=".to_string());
                    self.advance();
                }
                Token::LParen => {
                    parts.push("(".to_string());
                    self.advance();
                }
                Token::RParen => {
                    parts.push(")".to_string());
                    self.advance();
                }
                Token::Comma => {
                    parts.push(",".to_string());
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        parts.join(" ")
    }

    /// Parse instance parameter overrides: `param=value param2=value2 ...`
    fn parse_instance_params(&mut self, line: usize) -> Result<HashMap<String, f64>> {
        let mut params = HashMap::new();

        // Build context from existing parameters for expression evaluation
        let ctx = ParamContext::from_global(self.parameters.clone());

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let pname = n.to_uppercase();
                    self.advance();

                    // Expect '='
                    if !matches!(self.peek(), Token::Equals) {
                        return Err(Error::ParseError {
                            line,
                            message: format!(
                                "Expected '=' after parameter name '{}' in instance PARAMS:",
                                pname
                            ),
                        });
                    }
                    self.advance(); // consume '='

                    // Get value (numeric, parameter name, or curly expression)
                    let value = match self.peek() {
                        Token::Value(v) | Token::Name(v) => {
                            let v = v.clone();
                            self.advance();
                            // Try numeric first, then parameter lookup
                            if let Some(val) = parse_value(&v) {
                                val
                            } else if let Some(val) = ctx.get(&v) {
                                val
                            } else {
                                return Err(Error::ParseError {
                                    line,
                                    message: format!(
                                        "Unknown value '{}' for parameter '{}'",
                                        v, pname
                                    ),
                                });
                            }
                        }
                        Token::CurlyExpr(expr) => {
                            let expr = expr.clone();
                            self.advance();
                            self.eval_curly_expr_with_context(&expr, &ctx, line)?
                        }
                        _ => {
                            return Err(Error::ParseError {
                                line,
                                message: format!("Expected value for parameter '{}'", pname),
                            });
                        }
                    };

                    params.insert(pname, value);
                }
                _ => {
                    // Skip unexpected tokens
                    self.advance();
                }
            }
        }

        Ok(params)
    }

    /// Expand a subcircuit instance into the netlist.
    fn expand_subcircuit(
        &mut self,
        instance_name: &str,
        connections: &[String],
        subckt_name: &str,
        line: usize,
    ) -> Result<()> {
        // Look up subcircuit definition
        let subckt = match self.subcircuits.get(subckt_name) {
            Some(s) => s.clone(),
            None => {
                return Err(Error::ParseError {
                    line,
                    message: format!("Unknown subcircuit: {}", subckt_name),
                });
            }
        };

        // Verify port count matches
        if connections.len() != subckt.ports.len() {
            return Err(Error::ParseError {
                line,
                message: format!(
                    "Subcircuit {} expects {} ports but {} provided",
                    subckt_name,
                    subckt.ports.len(),
                    connections.len()
                ),
            });
        }

        // Build node mapping: port_name (uppercase) -> connection_node
        let mut node_map: HashMap<String, String> = HashMap::new();
        for (port, conn) in subckt.ports.iter().zip(connections.iter()) {
            // Store with uppercase key for case-insensitive lookup
            node_map.insert(port.to_uppercase(), conn.clone());
        }

        // Expand element lines with node substitution
        for elem in &subckt.elements {
            let expanded = self.expand_element_line(instance_name, &elem.line, &node_map);
            self.parse_expanded_element(&expanded, line)?;
        }

        // Expand nested subcircuit instances
        for inst in &subckt.instances {
            let expanded = self.expand_element_line(instance_name, &inst.line, &node_map);
            self.parse_expanded_element(&expanded, line)?;
        }

        Ok(())
    }

    /// Expand a subcircuit instance with parameter support.
    ///
    /// Builds a ParamContext with proper precedence:
    /// - global: top-level .PARAM values
    /// - subcircuit_defaults: subcircuit's PARAMS: defaults
    /// - instance_overrides: X instance's PARAMS: values
    fn expand_subcircuit_with_params(
        &mut self,
        instance_name: &str,
        connections: &[String],
        subckt_name: &str,
        instance_params: &HashMap<String, f64>,
        line: usize,
    ) -> Result<()> {
        // Look up subcircuit definition
        let subckt = match self.subcircuits.get(subckt_name) {
            Some(s) => s.clone(),
            None => {
                return Err(Error::ParseError {
                    line,
                    message: format!("Unknown subcircuit: {}", subckt_name),
                });
            }
        };

        // Verify port count matches
        if connections.len() != subckt.ports.len() {
            return Err(Error::ParseError {
                line,
                message: format!(
                    "Subcircuit {} expects {} ports but {} provided",
                    subckt_name,
                    subckt.ports.len(),
                    connections.len()
                ),
            });
        }

        // Build node mapping: port_name (uppercase) -> connection_node
        let mut node_map: HashMap<String, String> = HashMap::new();
        for (port, conn) in subckt.ports.iter().zip(connections.iter()) {
            node_map.insert(port.to_uppercase(), conn.clone());
        }

        // Build ParamContext with proper precedence
        let ctx = ParamContext::with_all(
            self.parameters.clone(), // global params
            subckt.params.clone(),   // subcircuit defaults
            instance_params.clone(), // instance overrides
        );

        // Expand element lines with node substitution and parameter evaluation
        for elem in &subckt.elements {
            let expanded = self.expand_element_line_with_params(
                instance_name,
                &elem.line,
                &node_map,
                &ctx,
                line,
            )?;
            self.parse_expanded_element_with_context(&expanded, &ctx, line)?;
        }

        // Expand nested subcircuit instances
        for inst in &subckt.instances {
            self.expand_nested_instance(instance_name, &inst.line, &node_map, &ctx, line)?;
        }

        Ok(())
    }

    /// Expand a single element line with node substitution and parameter evaluation.
    fn expand_element_line_with_params(
        &self,
        instance_prefix: &str,
        line: &str,
        node_map: &HashMap<String, String>,
        ctx: &ParamContext,
        source_line: usize,
    ) -> Result<String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(line.to_string());
        }

        let mut expanded = Vec::new();

        // First part is element name - preserve element type prefix, add instance hierarchy
        let elem_name = parts[0];
        let first_char = elem_name.chars().next().unwrap_or('R');
        let rest = if elem_name.len() > 1 {
            &elem_name[1..]
        } else {
            ""
        };
        expanded.push(format!("{}{}_{}", first_char, instance_prefix, rest));

        // Remaining parts: substitute nodes, evaluate curly expressions
        for part in &parts[1..] {
            // Check for curly brace expression
            if part.starts_with('{') && part.ends_with('}') {
                let expr = &part[1..part.len() - 1];
                let value = self.eval_curly_expr_with_context(expr, ctx, source_line)?;
                expanded.push(format!("{}", value));
            } else if let Some(mapped) = node_map.get(&part.to_uppercase()) {
                // Port node - use the external connection
                expanded.push(mapped.clone());
            } else if part.parse::<f64>().is_ok() || parse_value(part).is_some() {
                // Numeric value - keep as-is
                expanded.push(part.to_string());
            } else if part.contains('=') {
                // Parameter assignment like W=1u - keep as-is
                expanded.push(part.to_string());
            } else if part.to_uppercase() == "0" || part.to_uppercase() == "GND" {
                // Ground - keep as-is
                expanded.push(part.to_string());
            } else if let Some(val) = ctx.get(part) {
                // Parameter reference - substitute value
                expanded.push(format!("{}", val));
            } else if part
                .chars()
                .next()
                .map(|c| c.is_alphabetic())
                .unwrap_or(false)
            {
                let upper = part.to_uppercase();
                if self.models.contains_key(&upper) || self.subcircuits.contains_key(&upper) {
                    // Model or subcircuit reference - keep as-is
                    expanded.push(part.to_string());
                } else {
                    // Internal node - prefix with instance name
                    expanded.push(format!("{}_{}", instance_prefix, part));
                }
            } else {
                // Internal node with numeric start - prefix
                expanded.push(format!("{}_{}", instance_prefix, part));
            }
        }

        Ok(expanded.join(" "))
    }

    /// Expand a nested subcircuit instance line with parameter propagation.
    fn expand_nested_instance(
        &mut self,
        parent_prefix: &str,
        line: &str,
        node_map: &HashMap<String, String>,
        parent_ctx: &ParamContext,
        source_line: usize,
    ) -> Result<()> {
        // Parse the nested X instance line
        // Format: Xname node1 node2 ... subckt_name [PARAMS: param=value ...]
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let elem_name = parts[0];
        let first_char = elem_name.chars().next().unwrap_or('X').to_ascii_uppercase();
        if first_char != 'X' {
            // Not a subcircuit instance, shouldn't happen but handle gracefully
            return Ok(());
        }

        // Build hierarchical instance name
        let rest = if elem_name.len() > 1 {
            &elem_name[1..]
        } else {
            ""
        };
        let nested_instance_name = format!("{}_{}", parent_prefix, rest);

        // Find PARAMS: position if present
        let mut params_idx = None;
        for (i, part) in parts.iter().enumerate() {
            let upper = part.to_uppercase();
            if upper == "PARAMS" || upper == "PARAMS:" || upper.starts_with("PARAMS:") {
                params_idx = Some(i);
                break;
            }
        }

        // Split into node+subckt parts and param parts
        let node_parts: Vec<&str> = if let Some(idx) = params_idx {
            parts[1..idx].to_vec()
        } else {
            parts[1..].to_vec()
        };

        if node_parts.is_empty() {
            return Ok(());
        }

        // Last is subcircuit name, rest are connection nodes
        let nested_subckt_name = node_parts.last().unwrap().to_uppercase();
        let connection_names: Vec<&str> = node_parts[..node_parts.len() - 1].to_vec();

        // Map connection nodes through parent's node_map
        let mut nested_connections: Vec<String> = Vec::new();
        for conn in &connection_names {
            if let Some(mapped) = node_map.get(&conn.to_uppercase()) {
                nested_connections.push(mapped.clone());
            } else if conn.to_uppercase() == "0" || conn.to_uppercase() == "GND" {
                nested_connections.push(conn.to_string());
            } else {
                // Internal node - prefix with parent instance name
                nested_connections.push(format!("{}_{}", parent_prefix, conn));
            }
        }

        // Parse nested instance params and evaluate any expressions
        let mut nested_instance_params: HashMap<String, f64> = HashMap::new();
        if let Some(idx) = params_idx {
            let mut i = idx + 1;
            while i < parts.len() {
                let part = parts[i];
                if let Some(eq_pos) = part.find('=') {
                    let pname = part[..eq_pos].to_uppercase();
                    let pval = &part[eq_pos + 1..];
                    // Evaluate value expression in parent context
                    if pval.starts_with('{') && pval.ends_with('}') {
                        let expr = &pval[1..pval.len() - 1];
                        let value =
                            self.eval_curly_expr_with_context(expr, parent_ctx, source_line)?;
                        nested_instance_params.insert(pname, value);
                    } else if let Some(val) = parse_value(pval) {
                        nested_instance_params.insert(pname, val);
                    } else if let Some(val) = parent_ctx.get(pval) {
                        nested_instance_params.insert(pname, val);
                    }
                } else {
                    // Check for name=value on next iteration
                    let pname = part.to_uppercase();
                    if i + 1 < parts.len() && parts[i + 1] == "=" && i + 2 < parts.len() {
                        let pval = parts[i + 2];
                        if pval.starts_with('{') && pval.ends_with('}') {
                            let expr = &pval[1..pval.len() - 1];
                            let value =
                                self.eval_curly_expr_with_context(expr, parent_ctx, source_line)?;
                            nested_instance_params.insert(pname, value);
                        } else if let Some(val) = parse_value(pval) {
                            nested_instance_params.insert(pname, val);
                        } else if let Some(val) = parent_ctx.get(pval) {
                            nested_instance_params.insert(pname, val);
                        }
                        i += 2;
                    }
                }
                i += 1;
            }
        }

        // Register connection nodes
        for node_name in &nested_connections {
            self.get_or_create_node(node_name);
        }

        // Look up nested subcircuit definition
        let nested_subckt = match self.subcircuits.get(&nested_subckt_name) {
            Some(s) => s.clone(),
            None => {
                return Err(Error::ParseError {
                    line: source_line,
                    message: format!("Unknown subcircuit: {}", nested_subckt_name),
                });
            }
        };

        // Create child context: parent's merged becomes child's global
        let child_ctx =
            parent_ctx.child_context(nested_subckt.params.clone(), nested_instance_params);

        // Build node mapping for nested instance
        let mut nested_node_map: HashMap<String, String> = HashMap::new();
        for (port, conn) in nested_subckt.ports.iter().zip(nested_connections.iter()) {
            nested_node_map.insert(port.to_uppercase(), conn.clone());
        }

        // Expand nested subcircuit elements
        for elem in &nested_subckt.elements {
            let expanded = self.expand_element_line_with_params(
                &nested_instance_name,
                &elem.line,
                &nested_node_map,
                &child_ctx,
                source_line,
            )?;
            self.parse_expanded_element_with_context(&expanded, &child_ctx, source_line)?;
        }

        // Recursively expand any deeper nested instances
        for inst in &nested_subckt.instances {
            self.expand_nested_instance(
                &nested_instance_name,
                &inst.line,
                &nested_node_map,
                &child_ctx,
                source_line,
            )?;
        }

        Ok(())
    }

    /// Parse an expanded element line with parameter context.
    fn parse_expanded_element_with_context(
        &mut self,
        line: &str,
        ctx: &ParamContext,
        source_line: usize,
    ) -> Result<()> {
        // For now, values should already be resolved in the expanded line.
        // If we encounter a parameter reference, we'll resolve it here.
        // This reuses most of the existing parse_expanded_element logic.
        self.parse_expanded_element_internal(line, Some(ctx), source_line)
    }

    /// Expand a single element line with node substitution.
    fn expand_element_line(
        &self,
        instance_prefix: &str,
        line: &str,
        node_map: &HashMap<String, String>,
    ) -> String {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return line.to_string();
        }

        let mut expanded = Vec::new();

        // First part is element name - preserve element type prefix, add instance hierarchy
        // e.g., R1 in instance X1 becomes R_X1_1 (preserving 'R' as first char)
        let elem_name = parts[0];
        let first_char = elem_name.chars().next().unwrap_or('R');
        let rest = if elem_name.len() > 1 {
            &elem_name[1..]
        } else {
            ""
        };
        expanded.push(format!("{}{}_{}", first_char, instance_prefix, rest));

        // Remaining parts: substitute nodes if in port map, otherwise prefix internal nodes
        for part in &parts[1..] {
            if let Some(mapped) = node_map.get(&part.to_uppercase()) {
                // Port node - use the external connection
                expanded.push(mapped.clone());
            } else if part.parse::<f64>().is_ok()
                || part.contains('=')
                || parse_value(part).is_some()
            {
                // Value (including SPICE suffixes like 1k, 1u) or parameter - keep as-is
                expanded.push(part.to_string());
            } else if part.to_uppercase() == "0" || part.to_uppercase() == "GND" {
                // Ground - keep as-is
                expanded.push(part.to_string());
            } else if part
                .chars()
                .next()
                .map(|c| c.is_alphabetic())
                .unwrap_or(false)
            {
                // Possibly a model name or internal node
                // Check if it looks like a model reference (for D, M elements)
                let upper = part.to_uppercase();
                if self.models.contains_key(&upper) || self.subcircuits.contains_key(&upper) {
                    // Model or subcircuit reference - keep as-is
                    expanded.push(part.to_string());
                } else {
                    // Internal node - prefix with instance name (use _ as separator)
                    expanded.push(format!("{}_{}", instance_prefix, part));
                }
            } else {
                // Internal node with numeric start - prefix (use _ as separator)
                expanded.push(format!("{}_{}", instance_prefix, part));
            }
        }

        expanded.join(" ")
    }

    /// Parse an expanded element line.
    ///
    /// This creates a mini-parser to handle the expanded line and adds the
    /// resulting device to this parser's netlist.
    fn parse_expanded_element(&mut self, line: &str, source_line: usize) -> Result<()> {
        self.parse_expanded_element_internal(line, None, source_line)
    }

    /// Internal implementation that can optionally use a ParamContext.
    fn parse_expanded_element_internal(
        &mut self,
        line: &str,
        _ctx: Option<&ParamContext>,
        source_line: usize,
    ) -> Result<()> {
        // Tokenize the expanded line
        let lexer = Lexer::new(line);
        let tokens = lexer.tokenize()?;

        if tokens.is_empty() {
            return Ok(());
        }

        // Get the element name from first token
        let name = match &tokens[0].token {
            Token::Name(n) => n.clone(),
            _ => return Ok(()),
        };

        let first_char = name.chars().next().unwrap_or(' ').to_ascii_uppercase();

        // Mini-parser: manually extract nodes and values from tokens
        // This is a simplified approach that handles common element types
        // Note: With params support, values should already be resolved in the
        // expanded line by expand_element_line_with_params.
        match first_char {
            'R' => {
                // Resistor: R name node1 node2 value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let r = Resistor::new(&name, node_pos, node_neg, value);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(r);
                    }
                }
            }
            'C' => {
                // Capacitor: C name node1 node2 value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let c = Capacitor::new(&name, node_pos, node_neg, value);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(c);
                    }
                }
            }
            'L' => {
                // Inductor: L name node1 node2 value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let idx = self.next_current_index;
                        self.next_current_index += 1;
                        let l = Inductor::new(&name, node_pos, node_neg, value, idx);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(l);
                    }
                }
            }
            'V' => {
                // Voltage source: V name node+ node- value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let idx = self.next_current_index;
                        self.next_current_index += 1;
                        let v = VoltageSource::new(&name, node_pos, node_neg, value, idx);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(v);
                    }
                }
            }
            'I' => {
                // Current source: I name node+ node- value
                if tokens.len() >= 4 {
                    let node_pos = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let node_neg = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    if let Some(value) = parse_value(&Self::token_to_string(&tokens[3])) {
                        let i = CurrentSource::new(&name, node_pos, node_neg, value);
                        self.netlist.register_node(node_pos);
                        self.netlist.register_node(node_neg);
                        self.netlist.add_device(i);
                    }
                }
            }
            'D' => {
                // Diode: D name anode cathode [model]
                if tokens.len() >= 3 {
                    let anode = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let cathode = self.get_or_create_node(&Self::token_to_string(&tokens[2]));

                    let params = if tokens.len() >= 4 {
                        let model_name = Self::token_to_string(&tokens[3]).to_uppercase();
                        if let Some(ModelDefinition::Diode(p)) = self.models.get(&model_name) {
                            p.clone()
                        } else {
                            DiodeParams::default()
                        }
                    } else {
                        DiodeParams::default()
                    };

                    let d = Diode::with_params(&name, anode, cathode, params);
                    self.netlist.register_node(anode);
                    self.netlist.register_node(cathode);
                    self.netlist.add_device(d);
                }
            }
            'M' => {
                // MOSFET: M name drain gate source bulk [model] [W=val L=val]
                if tokens.len() >= 5 {
                    let drain = self.get_or_create_node(&Self::token_to_string(&tokens[1]));
                    let gate = self.get_or_create_node(&Self::token_to_string(&tokens[2]));
                    let source = self.get_or_create_node(&Self::token_to_string(&tokens[3]));
                    let _bulk = self.get_or_create_node(&Self::token_to_string(&tokens[4]));

                    // Parse model and W/L
                    let mut params = MosfetParams::nmos_default();
                    let mut mos_type = MosfetType::Nmos;

                    for token in tokens.iter().skip(5) {
                        let s = Self::token_to_string(token);
                        let upper = s.to_uppercase();
                        if let Some(ModelDefinition::Nmos(p)) = self.models.get(&upper) {
                            params = p.clone();
                            mos_type = MosfetType::Nmos;
                        } else if let Some(ModelDefinition::Pmos(p)) = self.models.get(&upper) {
                            params = p.clone();
                            mos_type = MosfetType::Pmos;
                        } else if upper.starts_with("W=") {
                            if let Some(v) = parse_value(&s[2..]) {
                                params.w = v;
                            }
                        } else if upper.starts_with("L=") {
                            if let Some(v) = parse_value(&s[2..]) {
                                params.l = v;
                            }
                        }
                    }

                    let m = Mosfet::with_params(&name, drain, gate, source, mos_type, params);
                    self.netlist.register_node(drain);
                    self.netlist.register_node(gate);
                    self.netlist.register_node(source);
                    self.netlist.add_device(m);
                }
            }
            'X' => {
                // Nested subcircuit instance - need to recursively expand
                // Collect only Name and Value tokens (skip Eol, Eof, etc.)
                let mut node_names: Vec<String> = Vec::new();
                for token in tokens.iter().skip(1) {
                    match &token.token {
                        Token::Name(s) | Token::Value(s) => {
                            node_names.push(s.clone());
                        }
                        Token::Eol | Token::Eof => break,
                        _ => {}
                    }
                }
                if !node_names.is_empty() {
                    let subckt_name = node_names.pop().unwrap().to_uppercase();
                    self.expand_subcircuit(&name, &node_names, &subckt_name, source_line)?;
                }
            }
            _ => {
                // Unknown element type in subcircuit - skip
            }
        }

        Ok(())
    }

    /// Helper to convert a SpannedToken to its string value.
    fn token_to_string(token: &SpannedToken) -> String {
        match &token.token {
            Token::Name(s) | Token::Value(s) => s.clone(),
            _ => String::new(),
        }
    }
}
