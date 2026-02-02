//! SPICE netlist parser.

use std::collections::HashMap;

use spicier_core::{Netlist, NodeId, units::parse_value};
use spicier_devices::diode::{Diode, DiodeParams};
use spicier_devices::mosfet::{Mosfet, MosfetParams, MosfetType};
use spicier_devices::passive::{Capacitor, Inductor, Resistor};
use spicier_devices::sources::{CurrentSource, VoltageSource};
use spicier_devices::Waveform;

use crate::error::{Error, Result};
use crate::lexer::{Lexer, SpannedToken, Token};

/// AC sweep type parsed from netlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcSweepType {
    /// Linear frequency spacing.
    Lin,
    /// Logarithmic spacing per decade.
    Dec,
    /// Logarithmic spacing per octave.
    Oct,
}

/// A single DC sweep specification.
#[derive(Debug, Clone)]
pub struct DcSweepSpec {
    /// Name of the source to sweep.
    pub source_name: String,
    /// Start value.
    pub start: f64,
    /// Stop value.
    pub stop: f64,
    /// Step size.
    pub step: f64,
}

/// An analysis command parsed from the netlist.
#[derive(Debug, Clone)]
pub enum AnalysisCommand {
    /// DC operating point (.OP).
    Op,
    /// DC sweep (.DC source start stop step [source2 start2 stop2 step2]).
    ///
    /// Supports nested sweeps: the first sweep is the outer (slow) sweep,
    /// the second (if present) is the inner (fast) sweep.
    Dc {
        /// One or two sweep specifications.
        sweeps: Vec<DcSweepSpec>,
    },
    /// AC sweep (.AC type npoints fstart fstop).
    Ac {
        sweep_type: AcSweepType,
        num_points: usize,
        fstart: f64,
        fstop: f64,
    },
    /// Transient analysis (.TRAN tstep tstop [tstart] [tmax] [UIC]).
    Tran {
        tstep: f64,
        tstop: f64,
        tstart: f64,
        /// Use Initial Conditions - skip DC operating point, use .IC values directly.
        uic: bool,
    },
}

/// Initial condition for a node voltage.
#[derive(Debug, Clone)]
pub struct InitialCondition {
    /// Node name (e.g., "1", "out").
    pub node: String,
    /// Initial voltage value.
    pub voltage: f64,
}

/// Type of analysis for .PRINT command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintAnalysisType {
    /// DC operating point or DC sweep.
    Dc,
    /// AC analysis.
    Ac,
    /// Transient analysis.
    Tran,
}

/// An output variable specification from .PRINT command.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputVariable {
    /// Node voltage: V(node) or V(node1, node2) for differential.
    Voltage {
        node: String,
        node2: Option<String>,
    },
    /// Device current: I(device).
    Current { device: String },
    /// Real part of voltage (AC): VR(node).
    VoltageReal { node: String },
    /// Imaginary part of voltage (AC): VI(node).
    VoltageImag { node: String },
    /// Magnitude of voltage (AC): VM(node).
    VoltageMag { node: String },
    /// Phase of voltage (AC): VP(node).
    VoltagePhase { node: String },
    /// Magnitude in dB (AC): VDB(node).
    VoltageDb { node: String },
}

/// A .PRINT command specifying output variables for an analysis type.
#[derive(Debug, Clone)]
pub struct PrintCommand {
    /// Type of analysis this print applies to.
    pub analysis_type: PrintAnalysisType,
    /// Variables to output.
    pub variables: Vec<OutputVariable>,
}

/// A raw element line stored in a subcircuit definition.
///
/// We store element lines as strings to be re-parsed during expansion,
/// allowing proper node name substitution.
#[derive(Debug, Clone)]
pub struct RawElementLine {
    /// The full element line (e.g., "R1 1 2 1k").
    pub line: String,
}

/// A subcircuit definition from .SUBCKT/.ENDS block.
#[derive(Debug, Clone)]
pub struct SubcircuitDefinition {
    /// Subcircuit name (e.g., "NAND", "OPAMP").
    pub name: String,
    /// Port node names in order (external interface).
    pub ports: Vec<String>,
    /// Element lines inside the subcircuit (stored as raw text).
    pub elements: Vec<RawElementLine>,
    /// Nested subcircuit instantiations (X lines).
    pub instances: Vec<RawElementLine>,
}

impl SubcircuitDefinition {
    fn new(name: String, ports: Vec<String>) -> Self {
        Self {
            name,
            ports,
            elements: Vec::new(),
            instances: Vec::new(),
        }
    }
}

/// Result of parsing a SPICE netlist.
///
/// Contains both the circuit (Netlist) and analysis commands.
#[derive(Debug)]
pub struct ParseResult {
    /// The circuit netlist.
    pub netlist: Netlist,
    /// Analysis commands found in the netlist.
    pub analyses: Vec<AnalysisCommand>,
    /// Initial conditions from .IC commands.
    pub initial_conditions: Vec<InitialCondition>,
    /// Node name to NodeId mapping.
    pub node_map: HashMap<String, NodeId>,
    /// Print commands specifying output variables.
    pub print_commands: Vec<PrintCommand>,
    /// Subcircuit definitions from .SUBCKT blocks.
    pub subcircuits: HashMap<String, SubcircuitDefinition>,
}

/// Parse a SPICE netlist string into a Netlist.
///
/// This is a convenience function that discards analysis commands.
/// Use [`parse_full`] to also get analysis commands.
pub fn parse(input: &str) -> Result<Netlist> {
    let result = parse_full(input)?;
    Ok(result.netlist)
}

/// Parse a SPICE netlist string, returning both circuit and analysis commands.
pub fn parse_full(input: &str) -> Result<ParseResult> {
    let lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let parser = Parser::new(&tokens);
    parser.parse_all()
}

/// A model definition from .MODEL command.
#[derive(Debug, Clone)]
enum ModelDefinition {
    Diode(DiodeParams),
    Nmos(MosfetParams),
    Pmos(MosfetParams),
}

/// Parser state.
struct Parser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
    netlist: Netlist,
    analyses: Vec<AnalysisCommand>,
    initial_conditions: Vec<InitialCondition>,
    print_commands: Vec<PrintCommand>,
    node_map: HashMap<String, NodeId>,
    next_current_index: usize,
    models: HashMap<String, ModelDefinition>,
    /// Subcircuit definitions.
    subcircuits: HashMap<String, SubcircuitDefinition>,
    /// Current subcircuit being parsed (None if at top level).
    current_subckt: Option<SubcircuitDefinition>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [SpannedToken]) -> Self {
        let mut node_map = HashMap::new();
        // Pre-register ground node aliases
        node_map.insert("0".to_string(), NodeId::GROUND);
        node_map.insert("gnd".to_string(), NodeId::GROUND);
        node_map.insert("GND".to_string(), NodeId::GROUND);

        Self {
            tokens,
            pos: 0,
            netlist: Netlist::new(),
            analyses: Vec::new(),
            initial_conditions: Vec::new(),
            print_commands: Vec::new(),
            node_map,
            next_current_index: 0,
            models: HashMap::new(),
            subcircuits: HashMap::new(),
            current_subckt: None,
        }
    }

    fn parse_all(mut self) -> Result<ParseResult> {
        // First non-comment line is the title
        self.skip_eol();
        if let Some(title) = self.parse_title() {
            self.netlist = Netlist::with_title(title);
        }

        while !self.is_at_end() {
            self.skip_eol();
            if self.is_at_end() {
                break;
            }

            match self.peek() {
                Token::Command(cmd) => {
                    self.parse_command(&cmd.clone())?;
                }
                Token::Name(name) => {
                    self.parse_element(&name.clone())?;
                }
                Token::Eof => break,
                _ => {
                    self.advance();
                }
            }
        }

        Ok(ParseResult {
            netlist: self.netlist,
            analyses: self.analyses,
            initial_conditions: self.initial_conditions,
            node_map: self.node_map,
            print_commands: self.print_commands,
            subcircuits: self.subcircuits,
        })
    }

    fn parse_title(&mut self) -> Option<String> {
        // Collect all tokens until EOL as the title
        let mut title_parts = Vec::new();

        while !self.is_at_end() {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    title_parts.push(n.clone());
                    self.advance();
                }
                Token::Value(v) => {
                    title_parts.push(v.clone());
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        if title_parts.is_empty() {
            None
        } else {
            Some(title_parts.join(" "))
        }
    }

    fn parse_command(&mut self, cmd: &str) -> Result<()> {
        let line = self.current_line();
        self.advance(); // consume command

        match cmd {
            "END" => {
                // End of netlist
            }
            "OP" => {
                self.analyses.push(AnalysisCommand::Op);
                self.skip_to_eol();
            }
            "DC" => {
                self.parse_dc_command(line)?;
            }
            "AC" => {
                self.parse_ac_command(line)?;
            }
            "TRAN" => {
                self.parse_tran_command(line)?;
            }
            "MODEL" => {
                self.parse_model_command(line)?;
            }
            "IC" => {
                self.parse_ic_command(line)?;
            }
            "PRINT" => {
                self.parse_print_command(line)?;
            }
            "SUBCKT" => {
                self.parse_subckt_command(line)?;
            }
            "ENDS" => {
                self.parse_ends_command(line)?;
            }
            _ => {
                // Unknown command - skip to EOL
                self.skip_to_eol();
            }
        }

        Ok(())
    }

    /// Parse .DC source start stop step [source2 start2 stop2 step2]
    fn parse_dc_command(&mut self, line: usize) -> Result<()> {
        let mut sweeps = Vec::new();

        // Parse first (required) sweep specification
        let source_name = match self.peek() {
            Token::Name(n) | Token::Value(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "expected source name for .DC".to_string(),
                });
            }
        };

        let start = self.expect_value(line)?;
        let stop = self.expect_value(line)?;
        let step = self.expect_value(line)?;

        sweeps.push(DcSweepSpec {
            source_name,
            start,
            stop,
            step,
        });

        // Check for optional second sweep specification
        if let Token::Name(n) | Token::Value(n) = self.peek() {
            let source_name2 = n.clone();
            self.advance();

            // If we got a second source name, we need all four values
            let start2 = self.expect_value(line)?;
            let stop2 = self.expect_value(line)?;
            let step2 = self.expect_value(line)?;

            sweeps.push(DcSweepSpec {
                source_name: source_name2,
                start: start2,
                stop: stop2,
                step: step2,
            });
        }

        self.analyses.push(AnalysisCommand::Dc { sweeps });

        self.skip_to_eol();
        Ok(())
    }

    /// Parse .AC type npoints fstart fstop
    fn parse_ac_command(&mut self, line: usize) -> Result<()> {
        let sweep_type = match self.peek() {
            Token::Name(n) => {
                let st = match n.to_uppercase().as_str() {
                    "DEC" => AcSweepType::Dec,
                    "OCT" => AcSweepType::Oct,
                    "LIN" => AcSweepType::Lin,
                    other => {
                        return Err(Error::ParseError {
                            line,
                            message: format!(
                                "unknown AC sweep type '{}' (expected DEC, OCT, or LIN)",
                                other
                            ),
                        });
                    }
                };
                self.advance();
                st
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "expected sweep type (DEC, OCT, LIN) for .AC".to_string(),
                });
            }
        };

        let num_points = self.expect_value(line)? as usize;
        let fstart = self.expect_value(line)?;
        let fstop = self.expect_value(line)?;

        self.analyses.push(AnalysisCommand::Ac {
            sweep_type,
            num_points,
            fstart,
            fstop,
        });

        self.skip_to_eol();
        Ok(())
    }

    /// Parse .TRAN tstep tstop [tstart]
    fn parse_tran_command(&mut self, _line: usize) -> Result<()> {
        let tstep = self.try_value().unwrap_or(1e-9);
        let tstop = self.try_value().unwrap_or(1e-6);

        // Optional tstart and tmax (we skip tmax)
        let tstart = self.try_value().unwrap_or(0.0);
        let _tmax = self.try_value(); // tmax is ignored for now

        // Check for UIC keyword
        let mut uic = false;
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    if n.to_uppercase() == "UIC" {
                        uic = true;
                        self.advance();
                    } else {
                        self.advance();
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        self.analyses.push(AnalysisCommand::Tran {
            tstep,
            tstop,
            tstart,
            uic,
        });

        self.skip_to_eol();
        Ok(())
    }

    /// Parse .IC V(node1)=value V(node2)=value ...
    fn parse_ic_command(&mut self, line: usize) -> Result<()> {
        // Parse multiple V(node)=value pairs
        // The lexer tokenizes V(1) as: V, (, 1, )
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(name) => {
                    let name = name.clone();
                    self.advance();

                    let upper = name.to_uppercase();
                    if upper == "V" {
                        // Expect ( node ) = value
                        if !matches!(self.peek(), Token::LParen) {
                            return Err(Error::ParseError {
                                line,
                                message: "Expected '(' after V in .IC".to_string(),
                            });
                        }
                        self.advance(); // consume (

                        // Get node name
                        let node = match self.peek() {
                            Token::Name(n) | Token::Value(n) => {
                                let n = n.clone();
                                self.advance();
                                n
                            }
                            _ => {
                                return Err(Error::ParseError {
                                    line,
                                    message: "Expected node name after V( in .IC".to_string(),
                                });
                            }
                        };

                        // Expect )
                        if !matches!(self.peek(), Token::RParen) {
                            return Err(Error::ParseError {
                                line,
                                message: "Expected ')' after node name in .IC".to_string(),
                            });
                        }
                        self.advance(); // consume )

                        // Expect =
                        if !matches!(self.peek(), Token::Equals) {
                            return Err(Error::ParseError {
                                line,
                                message: "Expected '=' after V(node) in .IC".to_string(),
                            });
                        }
                        self.advance(); // consume =

                        // Get the value
                        let voltage = self.expect_value(line)?;

                        self.initial_conditions.push(InitialCondition { node, voltage });
                    } else {
                        return Err(Error::ParseError {
                            line,
                            message: format!("Expected V(node)=value in .IC, got: {}", name),
                        });
                    }
                }
                _ => {
                    // Skip unexpected tokens
                    self.advance();
                }
            }
        }

        self.skip_to_eol();
        Ok(())
    }

    /// Parse .PRINT analysis_type var1 var2 ...
    /// Examples:
    ///   .PRINT DC V(1) V(2) I(R1)
    ///   .PRINT AC VM(out) VP(out) VDB(out)
    ///   .PRINT TRAN V(1) V(2)
    fn parse_print_command(&mut self, line: usize) -> Result<()> {
        // Parse analysis type (DC, AC, TRAN)
        let analysis_type = match self.peek() {
            Token::Name(n) => {
                let at = match n.to_uppercase().as_str() {
                    "DC" => PrintAnalysisType::Dc,
                    "AC" => PrintAnalysisType::Ac,
                    "TRAN" => PrintAnalysisType::Tran,
                    other => {
                        return Err(Error::ParseError {
                            line,
                            message: format!("Unknown .PRINT analysis type: {}", other),
                        });
                    }
                };
                self.advance();
                at
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "Expected analysis type (DC, AC, TRAN) for .PRINT".to_string(),
                });
            }
        };

        let mut variables = Vec::new();

        // Parse output variables until EOL
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(name) => {
                    let name = name.clone();
                    self.advance();

                    // Parse output variable: V(node), I(device), VM(node), VP(node), etc.
                    if let Some(var) = self.parse_output_variable(&name, line)? {
                        variables.push(var);
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        if !variables.is_empty() {
            self.print_commands.push(PrintCommand {
                analysis_type,
                variables,
            });
        }

        self.skip_to_eol();
        Ok(())
    }

    /// Parse a single output variable like V(node), I(device), VM(node), etc.
    fn parse_output_variable(&mut self, name: &str, line: usize) -> Result<Option<OutputVariable>> {
        let upper = name.to_uppercase();

        // Check if this is a function-style variable (needs parentheses)
        if !matches!(self.peek(), Token::LParen) {
            // Not a function - might be a simple reference or unknown
            return Ok(None);
        }
        self.advance(); // consume (

        // Get the first argument (node or device name)
        let arg1 = match self.peek() {
            Token::Name(n) | Token::Value(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: format!("Expected node/device name after {}(", name),
                });
            }
        };

        // Check for second argument (differential voltage)
        let arg2 = if matches!(self.peek(), Token::Comma) {
            self.advance(); // consume ,
            match self.peek() {
                Token::Name(n) | Token::Value(n) => {
                    let n = n.clone();
                    self.advance();
                    Some(n)
                }
                _ => None,
            }
        } else {
            None
        };

        // Expect closing paren
        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: format!("Expected ')' after {} argument", name),
            });
        }
        self.advance(); // consume )

        // Map function name to OutputVariable
        let var = match upper.as_str() {
            "V" => OutputVariable::Voltage {
                node: arg1,
                node2: arg2,
            },
            "I" => OutputVariable::Current { device: arg1 },
            "VR" => OutputVariable::VoltageReal { node: arg1 },
            "VI" => OutputVariable::VoltageImag { node: arg1 },
            "VM" => OutputVariable::VoltageMag { node: arg1 },
            "VP" => OutputVariable::VoltagePhase { node: arg1 },
            "VDB" => OutputVariable::VoltageDb { node: arg1 },
            _ => {
                // Unknown function, skip it
                return Ok(None);
            }
        };

        Ok(Some(var))
    }

    /// Parse .SUBCKT name port1 port2 ...
    fn parse_subckt_command(&mut self, line: usize) -> Result<()> {
        // Get subcircuit name
        let name = match self.peek() {
            Token::Name(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "Expected subcircuit name after .SUBCKT".to_string(),
                });
            }
        };

        // Get port names until EOL
        let mut ports = Vec::new();
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) | Token::Value(n) => {
                    ports.push(n.clone());
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        // Start a new subcircuit definition
        self.current_subckt = Some(SubcircuitDefinition::new(name.to_uppercase(), ports));
        self.skip_to_eol();
        Ok(())
    }

    /// Parse .ENDS [name]
    fn parse_ends_command(&mut self, line: usize) -> Result<()> {
        // Optionally get subcircuit name (for validation)
        let _end_name = match self.peek() {
            Token::Name(n) => {
                let n = n.clone();
                self.advance();
                Some(n)
            }
            _ => None,
        };

        // Finish current subcircuit and store it
        if let Some(subckt) = self.current_subckt.take() {
            self.subcircuits.insert(subckt.name.clone(), subckt);
        } else {
            return Err(Error::ParseError {
                line,
                message: ".ENDS without matching .SUBCKT".to_string(),
            });
        }

        self.skip_to_eol();
        Ok(())
    }

    /// Parse Xname node1 node2 ... subckt_name
    ///
    /// If we're inside a subcircuit definition, store as raw line.
    /// Otherwise, expand the subcircuit inline.
    fn parse_subcircuit_instance(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume instance name

        // Collect all node names until we hit EOL
        // The last name is the subcircuit name, others are connections
        let mut tokens: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) | Token::Value(n) => {
                    tokens.push(n.clone());
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
                message: format!("Subcircuit instance {} requires nodes and subcircuit name", name),
            });
        }

        // Last token is the subcircuit name
        let subckt_name = tokens.pop().unwrap().to_uppercase();
        let connection_nodes = tokens;

        // Build the raw line for storage
        let raw_line = format!("{} {} {}", name, connection_nodes.join(" "), subckt_name);

        if self.current_subckt.is_some() {
            // Inside a subcircuit definition - store raw line
            let subckt = self.current_subckt.as_mut().unwrap();
            subckt.instances.push(RawElementLine { line: raw_line });
        } else {
            // At top level - first register connection nodes to ensure they get proper IDs
            // before any internal subcircuit nodes are created
            for node_name in &connection_nodes {
                self.get_or_create_node(node_name);
            }
            // Then expand the subcircuit
            self.expand_subcircuit(name, &connection_nodes, &subckt_name, line)?;
        }

        self.skip_to_eol();
        Ok(())
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
        let rest = if elem_name.len() > 1 { &elem_name[1..] } else { "" };
        expanded.push(format!("{}{}_{}", first_char, instance_prefix, rest));

        // Remaining parts: substitute nodes if in port map, otherwise prefix internal nodes
        for part in &parts[1..] {
            if let Some(mapped) = node_map.get(&part.to_uppercase()) {
                // Port node - use the external connection
                expanded.push(mapped.clone());
            } else if part.parse::<f64>().is_ok() || part.contains('=') || parse_value(part).is_some() {
                // Value (including SPICE suffixes like 1k, 1u) or parameter - keep as-is
                expanded.push(part.to_string());
            } else if part.to_uppercase() == "0" || part.to_uppercase() == "GND" {
                // Ground - keep as-is
                expanded.push(part.to_string());
            } else if part.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
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

                    for i in 5..tokens.len() {
                        let s = Self::token_to_string(&tokens[i]);
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
                for i in 1..tokens.len() {
                    match &tokens[i].token {
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

    fn parse_element(&mut self, name: &str) -> Result<()> {
        let line = self.current_line();
        let first_char = name.chars().next().unwrap_or(' ').to_ascii_uppercase();

        // If we're inside a subcircuit, capture the raw line instead of parsing
        if self.current_subckt.is_some() && first_char != 'X' {
            let raw_line = self.capture_element_line(name);
            let subckt = self.current_subckt.as_mut().unwrap();
            subckt.elements.push(RawElementLine { line: raw_line });
            self.skip_to_eol();
            return Ok(());
        }

        match first_char {
            'R' => self.parse_resistor(name, line),
            'C' => self.parse_capacitor(name, line),
            'L' => self.parse_inductor(name, line),
            'V' => self.parse_voltage_source(name, line),
            'I' => self.parse_current_source(name, line),
            'D' => self.parse_diode(name, line),
            'M' => self.parse_mosfet(name, line),
            'E' => self.parse_vcvs(name, line),
            'G' => self.parse_vccs(name, line),
            'F' => self.parse_cccs(name, line),
            'H' => self.parse_ccvs(name, line),
            'X' => self.parse_subcircuit_instance(name, line),
            _ => {
                // Unknown element - skip line
                self.skip_to_eol();
                Ok(())
            }
        }
    }

    /// Capture an element line as raw text for subcircuit storage.
    fn capture_element_line(&mut self, name: &str) -> String {
        let mut parts = vec![name.to_string()];

        // Advance past the element name
        self.advance();

        // Collect remaining tokens until EOL
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) | Token::Value(n) => {
                    parts.push(n.clone());
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

    fn parse_resistor(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?;
        let node_neg = self.expect_node(line)?;
        let value = self.expect_value(line)?;

        let resistor = Resistor::new(name, node_pos, node_neg, value);
        self.netlist.add_device(resistor);

        self.skip_to_eol();
        Ok(())
    }

    fn parse_capacitor(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?;
        let node_neg = self.expect_node(line)?;
        let value = self.expect_value(line)?;

        let capacitor = Capacitor::new(name, node_pos, node_neg, value);
        self.netlist.add_device(capacitor);

        self.skip_to_eol();
        Ok(())
    }

    fn parse_inductor(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?;
        let node_neg = self.expect_node(line)?;
        let value = self.expect_value(line)?;

        let current_index = self.next_current_index;
        self.next_current_index += 1;

        let inductor = Inductor::new(name, node_pos, node_neg, value, current_index);
        self.netlist.add_device(inductor);

        self.skip_to_eol();
        Ok(())
    }

    fn parse_voltage_source(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?;
        let node_neg = self.expect_node(line)?;

        let current_index = self.next_current_index;
        self.next_current_index += 1;

        // Parse source specification: [DC value] [AC mag [phase]] [PULSE|SIN|PWL]
        let mut dc_value = 0.0;
        let mut waveform: Option<Waveform> = None;

        // Keep parsing until we hit end of line or no more valid tokens
        loop {
            match self.peek() {
                Token::Name(n) => {
                    let upper = n.to_uppercase();
                    match upper.as_str() {
                        "DC" => {
                            self.advance();
                            dc_value = self.expect_value(line)?;
                        }
                        "AC" => {
                            // Skip AC specification for now (used in AC analysis)
                            self.advance();
                            let _ = self.expect_value(line)?; // mag
                            // Optional phase
                            if let Token::Value(_) = self.peek() {
                                let _ = self.expect_value(line)?;
                            }
                        }
                        "PULSE" => {
                            self.advance();
                            waveform = Some(self.parse_pulse_waveform(line)?);
                        }
                        "SIN" => {
                            self.advance();
                            waveform = Some(self.parse_sin_waveform(line)?);
                        }
                        "PWL" => {
                            self.advance();
                            waveform = Some(self.parse_pwl_waveform(line)?);
                        }
                        _ => break, // Unknown keyword, stop parsing
                    }
                }
                Token::Value(_) => {
                    // Plain value is treated as DC value
                    dc_value = self.expect_value(line)?;
                }
                _ => break, // End of source spec
            }
        }

        let vsource = match waveform {
            Some(w) => VoltageSource::with_waveform(name, node_pos, node_neg, w, current_index),
            None => VoltageSource::new(name, node_pos, node_neg, dc_value, current_index),
        };
        self.netlist.add_device(vsource);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse PULSE(v1 v2 td tr tf pw per)
    fn parse_pulse_waveform(&mut self, line: usize) -> Result<Waveform> {
        // Expect opening paren
        if !matches!(self.peek(), Token::LParen) {
            return Err(Error::ParseError {
                line,
                message: "expected '(' after PULSE".to_string(),
            });
        }
        self.advance();

        let v1 = self.expect_value(line)?;
        let v2 = self.expect_value(line)?;

        // Optional parameters with defaults
        let td = self.try_expect_value().unwrap_or(0.0);
        let tr = self.try_expect_value().unwrap_or(0.0);
        let tf = self.try_expect_value().unwrap_or(0.0);
        let pw = self.try_expect_value().unwrap_or(0.0);
        let per = self.try_expect_value().unwrap_or(0.0);

        // Expect closing paren
        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: "expected ')' after PULSE parameters".to_string(),
            });
        }
        self.advance();

        Ok(Waveform::pulse(v1, v2, td, tr, tf, pw, per))
    }

    /// Parse SIN(vo va freq [td [theta [phase]]])
    fn parse_sin_waveform(&mut self, line: usize) -> Result<Waveform> {
        if !matches!(self.peek(), Token::LParen) {
            return Err(Error::ParseError {
                line,
                message: "expected '(' after SIN".to_string(),
            });
        }
        self.advance();

        let vo = self.expect_value(line)?;
        let va = self.expect_value(line)?;
        let freq = self.expect_value(line)?;

        // Optional parameters with defaults
        let td = self.try_expect_value().unwrap_or(0.0);
        let theta = self.try_expect_value().unwrap_or(0.0);
        let phase = self.try_expect_value().unwrap_or(0.0);

        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: "expected ')' after SIN parameters".to_string(),
            });
        }
        self.advance();

        Ok(Waveform::sin_full(vo, va, freq, td, theta, phase))
    }

    /// Parse PWL(t1 v1 t2 v2 ...)
    fn parse_pwl_waveform(&mut self, line: usize) -> Result<Waveform> {
        if !matches!(self.peek(), Token::LParen) {
            return Err(Error::ParseError {
                line,
                message: "expected '(' after PWL".to_string(),
            });
        }
        self.advance();

        let mut points = Vec::new();
        while let Some(t) = self.try_expect_value() {
            let v = self.expect_value(line)?;
            points.push((t, v));
        }

        if points.is_empty() {
            return Err(Error::ParseError {
                line,
                message: "PWL requires at least one time-value pair".to_string(),
            });
        }

        if !matches!(self.peek(), Token::RParen) {
            return Err(Error::ParseError {
                line,
                message: "expected ')' after PWL parameters".to_string(),
            });
        }
        self.advance();

        Ok(Waveform::pwl(points))
    }

    /// Try to parse a value, returning None if not available.
    fn try_expect_value(&mut self) -> Option<f64> {
        match self.peek() {
            Token::Value(s) => {
                let s = s.clone();
                if let Some(v) = parse_value(&s) {
                    self.advance();
                    Some(v)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn parse_current_source(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?;
        let node_neg = self.expect_node(line)?;

        // Value can be DC value or just a number
        let value = self.expect_value_or_dc(line)?;

        let isource = CurrentSource::new(name, node_pos, node_neg, value);
        self.netlist.add_device(isource);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse .MODEL name type (param=value ...)
    fn parse_model_command(&mut self, line: usize) -> Result<()> {
        let model_name = self.expect_name(line)?;
        let model_type = self.expect_name(line)?;

        // Parse optional parameter list in parentheses or bare params
        let mut params: Vec<(String, f64)> = Vec::new();

        // Check for opening paren
        let has_paren = matches!(self.peek(), Token::LParen);
        if has_paren {
            self.advance(); // consume '('
        }

        // Read param=value pairs until ')' or EOL
        loop {
            match self.peek() {
                Token::RParen if has_paren => {
                    self.advance();
                    break;
                }
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let pname = n.clone().to_uppercase();
                    self.advance();
                    // Expect '='
                    if matches!(self.peek(), Token::Equals) {
                        self.advance();
                        let val = self.expect_value(line)?;
                        params.push((pname, val));
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        let model_type_upper = model_type.to_uppercase();
        let model = match model_type_upper.as_str() {
            "D" => {
                let mut dp = DiodeParams::default();
                for (k, v) in &params {
                    match k.as_str() {
                        "IS" => dp.is = *v,
                        "N" => dp.n = *v,
                        "RS" => dp.rs = *v,
                        "CJO" | "CJ0" => dp.cj0 = *v,
                        "VJ" => dp.vj = *v,
                        "BV" => dp.bv = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Diode(dp)
            }
            "NMOS" => {
                let mut mp = MosfetParams::nmos_default();
                for (k, v) in &params {
                    match k.as_str() {
                        "VTO" => mp.vto = *v,
                        "KP" => mp.kp = *v,
                        "LAMBDA" => mp.lambda = *v,
                        "COX" => mp.cox = *v,
                        "W" => mp.w = *v,
                        "L" => mp.l = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Nmos(mp)
            }
            "PMOS" => {
                let mut mp = MosfetParams::pmos_default();
                for (k, v) in &params {
                    match k.as_str() {
                        "VTO" => mp.vto = *v,
                        "KP" => mp.kp = *v,
                        "LAMBDA" => mp.lambda = *v,
                        "COX" => mp.cox = *v,
                        "W" => mp.w = *v,
                        "L" => mp.l = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Pmos(mp)
            }
            _ => {
                self.skip_to_eol();
                return Ok(());
            }
        };

        self.models.insert(model_name.to_uppercase(), model);
        self.skip_to_eol();
        Ok(())
    }

    /// Parse D1 anode cathode [modelname]
    fn parse_diode(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?; // anode
        let node_neg = self.expect_node(line)?; // cathode

        // Optional model name
        let params = if let Token::Name(n) = self.peek() {
            let model_name = n.clone().to_uppercase();
            self.advance();
            if let Some(ModelDefinition::Diode(dp)) = self.models.get(&model_name) {
                dp.clone()
            } else {
                DiodeParams::default()
            }
        } else {
            DiodeParams::default()
        };

        let diode = Diode::with_params(name, node_pos, node_neg, params);
        self.netlist.add_device(diode);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse M1 drain gate source bulk [modelname] [W=val L=val]
    fn parse_mosfet(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_drain = self.expect_node(line)?;
        let node_gate = self.expect_node(line)?;
        let node_source = self.expect_node(line)?;
        let _node_bulk = self.expect_node(line)?; // bulk node (not used in Level 1)

        // Optional model name and W/L parameters
        let mut mos_type = MosfetType::Nmos;
        let mut params = MosfetParams::nmos_default();

        // Try to read model name
        if let Token::Name(n) = self.peek() {
            let model_name = n.clone().to_uppercase();
            // Check if it's a param assignment like W=...
            if !model_name.contains('=') && model_name != "W" && model_name != "L" {
                self.advance();
                match self.models.get(&model_name) {
                    Some(ModelDefinition::Nmos(mp)) => {
                        mos_type = MosfetType::Nmos;
                        params = mp.clone();
                    }
                    Some(ModelDefinition::Pmos(mp)) => {
                        mos_type = MosfetType::Pmos;
                        params = mp.clone();
                    }
                    _ => {}
                }
            }
        }

        // Parse optional W=val L=val parameters
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let pname = n.clone().to_uppercase();
                    self.advance();
                    if matches!(self.peek(), Token::Equals) {
                        self.advance();
                        if let Ok(val) = self.expect_value(line) {
                            match pname.as_str() {
                                "W" => params.w = val,
                                "L" => params.l = val,
                                _ => {}
                            }
                        }
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        let mosfet = Mosfet::with_params(name, node_drain, node_gate, node_source, mos_type, params);
        self.netlist.add_device(mosfet);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse E1 out+ out- ctrl+ ctrl- gain (VCVS)
    fn parse_vcvs(&mut self, name: &str, line: usize) -> Result<()> {
        use spicier_devices::controlled::Vcvs;
        self.advance(); // consume name

        let out_pos = self.expect_node(line)?;
        let out_neg = self.expect_node(line)?;
        let ctrl_pos = self.expect_node(line)?;
        let ctrl_neg = self.expect_node(line)?;
        let gain = self.expect_value(line)?;

        let current_index = self.next_current_index;
        self.next_current_index += 1;

        let vcvs = Vcvs::new(name, out_pos, out_neg, ctrl_pos, ctrl_neg, gain, current_index);
        self.netlist.add_device(vcvs);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse G1 out+ out- ctrl+ ctrl- gm (VCCS)
    fn parse_vccs(&mut self, name: &str, line: usize) -> Result<()> {
        use spicier_devices::controlled::Vccs;
        self.advance(); // consume name

        let out_pos = self.expect_node(line)?;
        let out_neg = self.expect_node(line)?;
        let ctrl_pos = self.expect_node(line)?;
        let ctrl_neg = self.expect_node(line)?;
        let gm = self.expect_value(line)?;

        let vccs = Vccs::new(name, out_pos, out_neg, ctrl_pos, ctrl_neg, gm);
        self.netlist.add_device(vccs);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse F1 out+ out- Vsource gain (CCCS)
    fn parse_cccs(&mut self, name: &str, line: usize) -> Result<()> {
        use spicier_devices::controlled::Cccs;
        self.advance(); // consume name

        let out_pos = self.expect_node(line)?;
        let out_neg = self.expect_node(line)?;
        let vsource_name = self.expect_name(line)?;
        let gain = self.expect_value(line)?;

        // Defer branch index resolution: store the name for now, resolve after parsing
        // For simplicity, look up the vsource branch index from the netlist
        let branch_idx = self
            .netlist
            .find_vsource_branch_index(&vsource_name)
            .ok_or_else(|| Error::ParseError {
                line,
                message: format!(
                    "CCCS '{}' references unknown voltage source '{}'",
                    name, vsource_name
                ),
            })?;

        let cccs = Cccs::new(name, out_pos, out_neg, branch_idx, gain);
        self.netlist.add_device(cccs);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse H1 out+ out- Vsource gain (CCVS)
    fn parse_ccvs(&mut self, name: &str, line: usize) -> Result<()> {
        use spicier_devices::controlled::Ccvs;
        self.advance(); // consume name

        let out_pos = self.expect_node(line)?;
        let out_neg = self.expect_node(line)?;
        let vsource_name = self.expect_name(line)?;
        let gain = self.expect_value(line)?;

        let vsource_branch_idx = self
            .netlist
            .find_vsource_branch_index(&vsource_name)
            .ok_or_else(|| Error::ParseError {
                line,
                message: format!(
                    "CCVS '{}' references unknown voltage source '{}'",
                    name, vsource_name
                ),
            })?;

        let current_index = self.next_current_index;
        self.next_current_index += 1;

        let ccvs = Ccvs::new(
            name,
            out_pos,
            out_neg,
            vsource_branch_idx,
            gain,
            current_index,
        );
        self.netlist.add_device(ccvs);

        self.skip_to_eol();
        Ok(())
    }

    fn expect_name(&mut self, line: usize) -> Result<String> {
        match self.peek() {
            Token::Name(n) => {
                let n = n.clone();
                self.advance();
                Ok(n)
            }
            Token::Value(v) => {
                let v = v.clone();
                self.advance();
                Ok(v)
            }
            _ => Err(Error::ParseError {
                line,
                message: "expected name".to_string(),
            }),
        }
    }

    fn expect_node(&mut self, line: usize) -> Result<NodeId> {
        match self.peek() {
            Token::Name(name) | Token::Value(name) => {
                let name = name.clone();
                self.advance();
                Ok(self.get_or_create_node(&name))
            }
            _ => Err(Error::ParseError {
                line,
                message: "expected node name".to_string(),
            }),
        }
    }

    fn expect_value(&mut self, line: usize) -> Result<f64> {
        match self.peek() {
            Token::Value(v) | Token::Name(v) => {
                let v = v.clone();
                self.advance();
                parse_value(&v).ok_or(Error::InvalidValue(v))
            }
            _ => Err(Error::ParseError {
                line,
                message: "expected value".to_string(),
            }),
        }
    }

    fn try_value(&mut self) -> Option<f64> {
        match self.peek() {
            Token::Value(v) | Token::Name(v) => {
                let v = v.clone();
                if let Some(val) = parse_value(&v) {
                    self.advance();
                    Some(val)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn expect_value_or_dc(&mut self, line: usize) -> Result<f64> {
        // Handle "DC 5" or just "5"
        if let Token::Name(n) = self.peek()
            && n.to_uppercase() == "DC"
        {
            self.advance(); // skip DC keyword
        }
        self.expect_value(line)
    }

    fn get_or_create_node(&mut self, name: &str) -> NodeId {
        if let Some(&id) = self.node_map.get(name) {
            return id;
        }

        // Try to parse as integer for traditional SPICE node numbers
        if let Ok(num) = name.parse::<u32>() {
            if num == 0 {
                return NodeId::GROUND;
            }
            let id = NodeId::new(num);
            // Check if this ID is already used by a different node name
            // (can happen if named nodes were assigned sequential IDs)
            let conflict = self.node_map.iter().any(|(k, &v)| v == id && k != name);
            if !conflict {
                self.node_map.insert(name.to_string(), id);
                self.netlist.register_node(id);
                return id;
            }
            // Fall through to assign a new unique ID if there's a conflict
        }

        // Named node (or numeric node with ID conflict) - assign next available ID
        let max_id = self
            .node_map
            .values()
            .filter(|id| !id.is_ground())
            .map(|id| id.as_u32())
            .max()
            .unwrap_or(0);
        let id = NodeId::new(max_id + 1);
        self.node_map.insert(name.to_string(), id);
        self.netlist.register_node(id);
        id
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map(|t| &t.token)
            .unwrap_or(&Token::Eof)
    }

    fn current_line(&self) -> usize {
        self.tokens.get(self.pos).map(|t| t.line).unwrap_or(0)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn skip_eol(&mut self) {
        while matches!(self.peek(), Token::Eol) {
            self.advance();
        }
    }

    fn skip_to_eol(&mut self) {
        while !matches!(self.peek(), Token::Eol | Token::Eof) {
            self.advance();
        }
        if matches!(self.peek(), Token::Eol) {
            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_circuit() {
        let input = r#"Simple RC Circuit
R1 1 2 1k
C1 2 0 1u
V1 1 0 5
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.title(), Some("Simple RC Circuit"));
        assert_eq!(netlist.num_devices(), 3);
        assert_eq!(netlist.num_nodes(), 2);
    }

    #[test]
    fn test_parse_voltage_divider() {
        let input = r#"Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3);
        assert_eq!(netlist.num_nodes(), 2);
        assert_eq!(netlist.num_current_vars(), 1); // One voltage source
    }

    #[test]
    fn test_parse_with_comments() {
        let input = r#"Test Circuit
* This is a comment
R1 1 0 1k ; inline comment
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 1);
    }

    #[test]
    fn test_parse_ground_aliases() {
        let input = r#"Ground Test
R1 1 0 1k
R2 2 gnd 1k
R3 3 GND 1k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3);
        // All should connect to ground (node 0)
    }

    #[test]
    fn test_parse_inductor() {
        let input = r#"Inductor Test
V1 1 0 10
L1 1 2 1m
R1 2 0 100
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3);
        assert_eq!(netlist.num_current_vars(), 2); // V1 + L1
    }

    #[test]
    fn test_parse_scientific_notation() {
        let input = r#"Scientific Test
C1 1 0 1e-12
R1 1 0 1e6
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2);
    }

    #[test]
    fn test_parse_op_command() {
        let input = r#"Op Test
V1 1 0 10
R1 1 0 1k
.op
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.netlist.num_devices(), 2);
        assert_eq!(result.analyses.len(), 1);
        assert!(matches!(result.analyses[0], AnalysisCommand::Op));
    }

    #[test]
    fn test_parse_dc_command() {
        let input = r#"DC Sweep Test
V1 1 0 10
R1 1 2 1k
R2 2 0 1k
.dc V1 0 10 0.5
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.analyses.len(), 1);
        match &result.analyses[0] {
            AnalysisCommand::Dc { sweeps } => {
                assert_eq!(sweeps.len(), 1);
                assert_eq!(sweeps[0].source_name, "V1");
                assert!((sweeps[0].start - 0.0).abs() < 1e-10);
                assert!((sweeps[0].stop - 10.0).abs() < 1e-10);
                assert!((sweeps[0].step - 0.5).abs() < 1e-10);
            }
            _ => panic!("Expected DC analysis command"),
        }
    }

    #[test]
    fn test_parse_dc_command_nested_sweep() {
        let input = r#"Nested DC Sweep Test
V1 1 0 10
V2 2 0 5
R1 1 2 1k
R2 2 0 1k
.dc V1 0 10 1 V2 0 5 0.5
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.analyses.len(), 1);
        match &result.analyses[0] {
            AnalysisCommand::Dc { sweeps } => {
                assert_eq!(sweeps.len(), 2);
                // Primary (outer) sweep
                assert_eq!(sweeps[0].source_name, "V1");
                assert!((sweeps[0].start - 0.0).abs() < 1e-10);
                assert!((sweeps[0].stop - 10.0).abs() < 1e-10);
                assert!((sweeps[0].step - 1.0).abs() < 1e-10);
                // Secondary (inner) sweep
                assert_eq!(sweeps[1].source_name, "V2");
                assert!((sweeps[1].start - 0.0).abs() < 1e-10);
                assert!((sweeps[1].stop - 5.0).abs() < 1e-10);
                assert!((sweeps[1].step - 0.5).abs() < 1e-10);
            }
            _ => panic!("Expected DC analysis command"),
        }
    }

    #[test]
    fn test_parse_ac_command() {
        let input = r#"AC Test
V1 1 0 10
R1 1 2 1k
C1 2 0 1u
.ac dec 10 1 1e6
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.analyses.len(), 1);
        match &result.analyses[0] {
            AnalysisCommand::Ac {
                sweep_type,
                num_points,
                fstart,
                fstop,
            } => {
                assert_eq!(*sweep_type, AcSweepType::Dec);
                assert_eq!(*num_points, 10);
                assert!((fstart - 1.0).abs() < 1e-10);
                assert!((fstop - 1e6).abs() < 1e-4);
            }
            _ => panic!("Expected AC analysis command"),
        }
    }

    #[test]
    fn test_parse_tran_command() {
        let input = r#"Tran Test
V1 1 0 5
R1 1 2 1k
C1 2 0 1u
.tran 1u 5m
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.analyses.len(), 1);
        match &result.analyses[0] {
            AnalysisCommand::Tran {
                tstep,
                tstop,
                tstart,
                uic,
            } => {
                assert!((tstep - 1e-6).abs() < 1e-12);
                assert!((tstop - 5e-3).abs() < 1e-9);
                assert!((tstart - 0.0).abs() < 1e-12);
                assert!(!uic, "UIC should be false by default");
            }
            _ => panic!("Expected TRAN analysis command"),
        }
    }

    #[test]
    fn test_parse_ic_command() {
        let input = r#"IC Test
V1 1 0 5
R1 1 2 1k
C1 2 0 1u
.IC V(1)=2.5 V(2)=1.0
.tran 1u 5m
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.initial_conditions.len(), 2);
        assert_eq!(result.initial_conditions[0].node, "1");
        assert!((result.initial_conditions[0].voltage - 2.5).abs() < 1e-12);
        assert_eq!(result.initial_conditions[1].node, "2");
        assert!((result.initial_conditions[1].voltage - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_ic_with_named_nodes() {
        let input = r#"IC Named Nodes
VIN in 0 5
R1 in out 1k
C1 out 0 1u
.IC V(in)=3.3 V(out)=1.65
.tran 1u 5m
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.initial_conditions.len(), 2);
        assert_eq!(result.initial_conditions[0].node, "in");
        assert!((result.initial_conditions[0].voltage - 3.3).abs() < 1e-12);
        assert_eq!(result.initial_conditions[1].node, "out");
        assert!((result.initial_conditions[1].voltage - 1.65).abs() < 1e-12);
    }

    #[test]
    fn test_parse_multiple_analyses() {
        let input = r#"Multi Analysis
V1 1 0 10
R1 1 2 1k
R2 2 0 1k
.op
.dc V1 0 10 1
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.analyses.len(), 2);
        assert!(matches!(result.analyses[0], AnalysisCommand::Op));
        assert!(matches!(result.analyses[1], AnalysisCommand::Dc { .. }));
    }

    #[test]
    fn test_parse_diode() {
        let input = r#"Diode Test
V1 1 0 5
R1 1 2 1k
D1 2 0
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3);
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_diode_with_model() {
        let input = r#"Diode Model Test
.MODEL DMOD D (IS=1e-12 N=2)
V1 1 0 5
D1 1 0 DMOD
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2); // V1, D1
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_mosfet() {
        let input = r#"MOSFET Test
V1 1 0 5
VG 2 0 3
M1 1 2 0 0
R1 1 0 10k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 4);
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_mosfet_with_model() {
        let input = r#"MOSFET Model Test
.MODEL NMOD NMOS (VTO=0.5 KP=1e-4)
V1 1 0 5
M1 1 2 0 0 NMOD W=20u L=1u
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2); // V1, M1
    }

    #[test]
    fn test_parse_vcvs() {
        let input = r#"VCVS Test
V1 1 0 10
R1 1 2 1k
R2 3 0 1k
E1 3 0 1 2 2.0
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 4);
        assert_eq!(netlist.num_current_vars(), 2); // V1 + E1
    }

    #[test]
    fn test_parse_vccs() {
        let input = r#"VCCS Test
V1 1 0 10
R1 2 0 1k
G1 2 0 1 0 0.001
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3);
        assert_eq!(netlist.num_current_vars(), 1); // V1 only (G has no current var)
    }

    #[test]
    fn test_parse_cccs() {
        let input = r#"CCCS Test
V1 1 0 10
R1 1 0 1k
R2 2 0 1k
F1 2 0 V1 3.0
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 4);
        assert_eq!(netlist.num_current_vars(), 1); // V1 only
    }

    #[test]
    fn test_parse_ccvs() {
        let input = r#"CCVS Test
V1 1 0 10
R1 1 0 1k
R2 2 0 1k
H1 2 0 V1 100.0
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 4);
        assert_eq!(netlist.num_current_vars(), 2); // V1 + H1
    }

    #[test]
    fn test_parse_print_command() {
        let input = r#"Print Test
V1 1 0 10
R1 1 2 1k
R2 2 0 1k
.PRINT DC V(1) V(2)
.OP
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.print_commands.len(), 1);
        assert_eq!(result.print_commands[0].analysis_type, PrintAnalysisType::Dc);
        assert_eq!(result.print_commands[0].variables.len(), 2);

        // Check first variable is V(1)
        assert!(matches!(
            &result.print_commands[0].variables[0],
            OutputVariable::Voltage { node, node2: None } if node == "1"
        ));

        // Check second variable is V(2)
        assert!(matches!(
            &result.print_commands[0].variables[1],
            OutputVariable::Voltage { node, node2: None } if node == "2"
        ));
    }

    #[test]
    fn test_parse_print_ac() {
        let input = r#"AC Print Test
V1 1 0 1
R1 1 2 1k
C1 2 0 1u
.PRINT AC VM(2) VP(2) VDB(2)
.AC DEC 10 1 1e6
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.print_commands.len(), 1);
        assert_eq!(result.print_commands[0].analysis_type, PrintAnalysisType::Ac);
        assert_eq!(result.print_commands[0].variables.len(), 3);

        assert!(matches!(
            &result.print_commands[0].variables[0],
            OutputVariable::VoltageMag { node } if node == "2"
        ));
        assert!(matches!(
            &result.print_commands[0].variables[1],
            OutputVariable::VoltagePhase { node } if node == "2"
        ));
        assert!(matches!(
            &result.print_commands[0].variables[2],
            OutputVariable::VoltageDb { node } if node == "2"
        ));
    }

    #[test]
    fn test_parse_subcircuit_definition() {
        let input = r#"Subcircuit Test
.SUBCKT VDIV in out
R1 in out 1k
R2 out 0 1k
.ENDS VDIV
V1 1 0 10
X1 1 2 VDIV
.end
"#;

        let result = parse_full(input).unwrap();

        // Check subcircuit was parsed
        assert!(result.subcircuits.contains_key("VDIV"), "VDIV should be defined");
        let subckt = &result.subcircuits["VDIV"];
        assert_eq!(subckt.name, "VDIV");
        assert_eq!(subckt.ports, vec!["in", "out"]);
        assert_eq!(subckt.elements.len(), 2); // R1 and R2

        // Check netlist has expanded devices: V1, X1.R1, X1.R2
        assert_eq!(result.netlist.num_devices(), 3);
    }
}
