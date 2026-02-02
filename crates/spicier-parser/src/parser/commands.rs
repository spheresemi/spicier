//! Command parsing (.DC, .AC, .TRAN, .IC, .PRINT, .MODEL, .PARAM, .SUBCKT, .ENDS).

use spicier_core::units::parse_value;
use spicier_devices::bjt::BjtParams;
use spicier_devices::diode::DiodeParams;
use spicier_devices::jfet::JfetParams;
use spicier_devices::mosfet::MosfetParams;

use crate::error::{Error, Result};
use crate::lexer::Token;

use super::types::{
    AcSweepType, AnalysisCommand, DcSweepSpec, InitialCondition, OutputVariable, PrintAnalysisType,
    PrintCommand, SubcircuitDefinition,
};
use super::{ModelDefinition, Parser};

impl<'a> Parser<'a> {
    pub(super) fn parse_command(&mut self, cmd: &str) -> Result<()> {
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
            "PARAM" => {
                // Already parsed in Pass 1, skip
                self.skip_to_eol();
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

                        self.initial_conditions
                            .push(InitialCondition { node, voltage });
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

    /// Parse .MODEL name type (param=value ...)
    pub(super) fn parse_model_command(&mut self, line: usize) -> Result<()> {
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
            "NJF" => {
                let mut jp = JfetParams::njf_default();
                for (k, v) in &params {
                    match k.as_str() {
                        "VTO" => jp.vto = *v,
                        "BETA" => jp.beta = *v,
                        "LAMBDA" => jp.lambda = *v,
                        "IS" => jp.is = *v,
                        "CGS" => jp.cgs = *v,
                        "CGD" => jp.cgd = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Njf(jp)
            }
            "PJF" => {
                let mut jp = JfetParams::pjf_default();
                for (k, v) in &params {
                    match k.as_str() {
                        "VTO" => jp.vto = *v,
                        "BETA" => jp.beta = *v,
                        "LAMBDA" => jp.lambda = *v,
                        "IS" => jp.is = *v,
                        "CGS" => jp.cgs = *v,
                        "CGD" => jp.cgd = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Pjf(jp)
            }
            "NPN" => {
                let mut bp = BjtParams::npn_default();
                for (k, v) in &params {
                    match k.as_str() {
                        "IS" => bp.is = *v,
                        "BF" => bp.bf = *v,
                        "BR" => bp.br = *v,
                        "NF" => bp.nf = *v,
                        "NR" => bp.nr = *v,
                        "VAF" => bp.vaf = *v,
                        "VAR" => bp.var = *v,
                        "RB" => bp.rb = *v,
                        "RE" => bp.re = *v,
                        "RC" => bp.rc = *v,
                        "CJE" => bp.cje = *v,
                        "CJC" => bp.cjc = *v,
                        "TF" => bp.tf = *v,
                        "TR" => bp.tr = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Npn(bp)
            }
            "PNP" => {
                let mut bp = BjtParams::pnp_default();
                for (k, v) in &params {
                    match k.as_str() {
                        "IS" => bp.is = *v,
                        "BF" => bp.bf = *v,
                        "BR" => bp.br = *v,
                        "NF" => bp.nf = *v,
                        "NR" => bp.nr = *v,
                        "VAF" => bp.vaf = *v,
                        "VAR" => bp.var = *v,
                        "RB" => bp.rb = *v,
                        "RE" => bp.re = *v,
                        "RC" => bp.rc = *v,
                        "CJE" => bp.cje = *v,
                        "CJC" => bp.cjc = *v,
                        "TF" => bp.tf = *v,
                        "TR" => bp.tr = *v,
                        _ => {}
                    }
                }
                ModelDefinition::Pnp(bp)
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

    /// Parse .SUBCKT name port1 port2 ...
    pub(super) fn parse_subckt_command(&mut self, line: usize) -> Result<()> {
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
    pub(super) fn parse_ends_command(&mut self, line: usize) -> Result<()> {
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

    /// Parse .PARAM name=value [name=value ...]
    /// Values must be numeric literals (with optional SI suffix).
    pub(super) fn parse_param_command(&mut self, line: usize) -> Result<()> {
        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let pname = n.clone().to_uppercase();
                    self.advance();

                    // Expect '='
                    if !matches!(self.peek(), Token::Equals) {
                        return Err(Error::ParseError {
                            line,
                            message: format!(".PARAM: expected '=' after '{}'", pname),
                        });
                    }
                    self.advance(); // consume '='

                    // Get value - use parse_value directly for numeric literals
                    let value = match self.peek() {
                        Token::Value(v) | Token::Name(v) => {
                            let v = v.clone();
                            self.advance();
                            parse_value(&v).ok_or_else(|| Error::ParseError {
                                line,
                                message: format!(".PARAM: invalid value '{}' for '{}'", v, pname),
                            })?
                        }
                        _ => {
                            return Err(Error::ParseError {
                                line,
                                message: format!(".PARAM: expected value for '{}'", pname),
                            });
                        }
                    };
                    self.parameters.insert(pname, value);
                }
                _ => {
                    self.advance();
                }
            }
        }
        self.skip_to_eol();
        Ok(())
    }
}
