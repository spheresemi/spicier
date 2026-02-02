//! SPICE netlist parser.

use std::collections::HashMap;

use spicier_core::{Netlist, NodeId, units::parse_value};
use spicier_devices::diode::{Diode, DiodeParams};
use spicier_devices::mosfet::{Mosfet, MosfetParams, MosfetType};
use spicier_devices::passive::{Capacitor, Inductor, Resistor};
use spicier_devices::sources::{CurrentSource, VoltageSource};

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

/// An analysis command parsed from the netlist.
#[derive(Debug, Clone)]
pub enum AnalysisCommand {
    /// DC operating point (.OP).
    Op,
    /// DC sweep (.DC source start stop step).
    Dc {
        source_name: String,
        start: f64,
        stop: f64,
        step: f64,
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
    node_map: HashMap<String, NodeId>,
    next_current_index: usize,
    models: HashMap<String, ModelDefinition>,
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
            node_map,
            next_current_index: 0,
            models: HashMap::new(),
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
            _ => {
                // Unknown command - skip to EOL
                self.skip_to_eol();
            }
        }

        Ok(())
    }

    /// Parse .DC source start stop step
    fn parse_dc_command(&mut self, line: usize) -> Result<()> {
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

        self.analyses.push(AnalysisCommand::Dc {
            source_name,
            start,
            stop,
            step,
        });

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

    fn parse_element(&mut self, name: &str) -> Result<()> {
        let line = self.current_line();
        let first_char = name.chars().next().unwrap_or(' ').to_ascii_uppercase();

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
            _ => {
                // Unknown element - skip line
                self.skip_to_eol();
                Ok(())
            }
        }
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

        // Value can be DC value or just a number
        let value = self.expect_value_or_dc(line)?;

        let current_index = self.next_current_index;
        self.next_current_index += 1;

        let vsource = VoltageSource::new(name, node_pos, node_neg, value, current_index);
        self.netlist.add_device(vsource);

        self.skip_to_eol();
        Ok(())
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
            self.node_map.insert(name.to_string(), id);
            self.netlist.register_node(id);
            return id;
        }

        // Named node - assign next available ID
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
            AnalysisCommand::Dc {
                source_name,
                start,
                stop,
                step,
            } => {
                assert_eq!(source_name, "V1");
                assert!((start - 0.0).abs() < 1e-10);
                assert!((stop - 10.0).abs() < 1e-10);
                assert!((step - 0.5).abs() < 1e-10);
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
}
