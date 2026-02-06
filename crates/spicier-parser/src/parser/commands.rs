//! Command parsing (.DC, .AC, .TRAN, .IC, .PRINT, .MODEL, .PARAM, .SUBCKT, .ENDS).

use std::collections::{HashMap, HashSet};

use spicier_core::units::parse_value;
use spicier_devices::bjt::BjtParams;
use spicier_devices::diode::DiodeParams;
use spicier_devices::expression::{EvalContext, parse_expression_with_params};
use spicier_devices::jfet::JfetParams;
use spicier_devices::mosfet::{Bsim3Params, Bsim4Params, MosfetParams, MosfetType};
use spicier_devices::passive::CapacitorParams;

use super::ParamContext;
use crate::error::{Error, Result};
use crate::lexer::Token;

use super::types::{
    AcSweepType, AnalysisCommand, DcSweepSpec, DcSweepType, InitialCondition, MeasureAnalysis,
    MeasureType, Measurement, OutputVariable, PrintAnalysisType, PrintCommand, StatFunc,
    SubcircuitDefinition, TriggerType,
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
            "MEAS" | "MEASURE" => {
                self.parse_meas_command(line)?;
            }
            "NOISE" => {
                self.parse_noise_command(line)?;
            }
            _ => {
                // Unknown command - skip to EOL
                self.skip_to_eol();
            }
        }

        Ok(())
    }

    /// Parse .DC source start stop step [source2 start2 stop2 step2]
    /// Also supports .DC PARAM name start stop step for parameter sweeps
    fn parse_dc_command(&mut self, line: usize) -> Result<()> {
        let mut sweeps = Vec::new();

        // Parse first (required) sweep specification
        let (source_name, sweep_type) = match self.peek() {
            Token::Name(n) | Token::Value(n) => {
                let n_upper = n.to_uppercase();
                if n_upper == "PARAM" {
                    // .DC PARAM name start stop step
                    self.advance(); // consume "PARAM"
                    let param_name = match self.peek() {
                        Token::Name(pn) | Token::Value(pn) => {
                            let pn = pn.clone();
                            self.advance();
                            pn
                        }
                        _ => {
                            return Err(Error::ParseError {
                                line,
                                message: "expected parameter name after .DC PARAM".to_string(),
                            });
                        }
                    };
                    (param_name, DcSweepType::Param)
                } else {
                    let n = n.clone();
                    self.advance();
                    (n, DcSweepType::Source)
                }
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "expected source name or PARAM for .DC".to_string(),
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
            sweep_type,
        });

        // Check for optional second sweep specification
        if let Token::Name(n) | Token::Value(n) = self.peek() {
            let n_upper = n.to_uppercase();
            let (source_name2, sweep_type2) = if n_upper == "PARAM" {
                self.advance(); // consume "PARAM"
                let param_name = match self.peek() {
                    Token::Name(pn) | Token::Value(pn) => {
                        let pn = pn.clone();
                        self.advance();
                        pn
                    }
                    _ => {
                        return Err(Error::ParseError {
                            line,
                            message: "expected parameter name after .DC PARAM".to_string(),
                        });
                    }
                };
                (param_name, DcSweepType::Param)
            } else {
                let n = n.clone();
                self.advance();
                (n, DcSweepType::Source)
            };

            // If we got a second source name, we need all four values
            let start2 = self.expect_value(line)?;
            let stop2 = self.expect_value(line)?;
            let step2 = self.expect_value(line)?;

            sweeps.push(DcSweepSpec {
                source_name: source_name2,
                start: start2,
                stop: stop2,
                step: step2,
                sweep_type: sweep_type2,
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

    /// Parse .NOISE V(output[,ref]) Vinput type npoints fstart fstop
    ///
    /// Examples:
    /// - `.NOISE V(out) V1 DEC 10 1 1MEG` - Single-ended output
    /// - `.NOISE V(out,ref) V1 DEC 10 1 1MEG` - Differential output
    fn parse_noise_command(&mut self, line: usize) -> Result<()> {
        // Parse output specification: V(node) or V(node,ref)
        let output_name = match self.peek() {
            Token::Name(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "expected output specification (e.g., V(out)) for .NOISE".to_string(),
                });
            }
        };

        // Parse V(node) or V(node,ref)
        let (output_node, output_ref_node) = if output_name.to_uppercase() == "V" {
            // Expect ( node )
            if !matches!(self.peek(), Token::LParen) {
                return Err(Error::ParseError {
                    line,
                    message: "expected '(' after V in .NOISE".to_string(),
                });
            }
            self.advance(); // consume (

            let node = match self.peek() {
                Token::Name(n) | Token::Value(n) => {
                    let n = n.clone();
                    self.advance();
                    n
                }
                _ => {
                    return Err(Error::ParseError {
                        line,
                        message: "expected output node name in .NOISE".to_string(),
                    });
                }
            };

            // Check for optional reference node
            let ref_node = if matches!(self.peek(), Token::Comma) {
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

            // Expect )
            if !matches!(self.peek(), Token::RParen) {
                return Err(Error::ParseError {
                    line,
                    message: "expected ')' in .NOISE output specification".to_string(),
                });
            }
            self.advance(); // consume )

            (node, ref_node)
        } else {
            return Err(Error::ParseError {
                line,
                message: format!("expected V(node) for .NOISE output, got '{}'", output_name),
            });
        };

        // Parse input source name
        let input_source = match self.peek() {
            Token::Name(n) | Token::Value(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: "expected input source name for .NOISE".to_string(),
                });
            }
        };

        // Parse sweep type (DEC, OCT, LIN)
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
                                "unknown sweep type '{}' for .NOISE (expected DEC, OCT, or LIN)",
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
                    message: "expected sweep type (DEC, OCT, LIN) for .NOISE".to_string(),
                });
            }
        };

        let num_points = self.expect_value(line)? as usize;
        let fstart = self.expect_value(line)?;
        let fstop = self.expect_value(line)?;

        self.analyses.push(AnalysisCommand::Noise {
            output_node,
            output_ref_node,
            input_source,
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
            "C" | "CAP" => {
                let cp = parse_capacitor_params(&params);
                ModelDefinition::Capacitor(cp)
            }
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
                // Check for LEVEL parameter to determine model type
                let level = params
                    .iter()
                    .find(|(k, _)| k == "LEVEL")
                    .map(|(_, v)| *v as i32)
                    .unwrap_or(1);
                if level == 54 || level == 14 {
                    // BSIM4 model
                    let bp = parse_bsim4_params(&params, MosfetType::Nmos);
                    ModelDefinition::Nmos54(bp)
                } else if level == 49 || level == 8 {
                    // BSIM3 model
                    let bp = parse_bsim3_params(&params, MosfetType::Nmos);
                    ModelDefinition::Nmos49(bp)
                } else {
                    // Level 1 (Shichman-Hodges) model
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
            }
            "PMOS" => {
                // Check for LEVEL parameter to determine model type
                let level = params
                    .iter()
                    .find(|(k, _)| k == "LEVEL")
                    .map(|(_, v)| *v as i32)
                    .unwrap_or(1);
                if level == 54 || level == 14 {
                    // BSIM4 model
                    let bp = parse_bsim4_params(&params, MosfetType::Pmos);
                    ModelDefinition::Pmos54(bp)
                } else if level == 49 || level == 8 {
                    // BSIM3 model
                    let bp = parse_bsim3_params(&params, MosfetType::Pmos);
                    ModelDefinition::Pmos49(bp)
                } else {
                    // Level 1 (Shichman-Hodges) model
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

    /// Parse .SUBCKT name port1 port2 ... [PARAMS: param=default ...]
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

        // Get port names until EOL or PARAMS:
        let mut ports = Vec::new();
        let mut params = HashMap::new();

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let n_upper = n.to_uppercase();
                    if n_upper == "PARAMS" {
                        self.advance();
                        // Expect colon after PARAMS (it may be part of the name or separate)
                        // Check if there's a colon by looking for Name ending with ':'
                        // Or skip if next token looks like param assignment
                        // PARAMS: is often written as a single token "PARAMS:" or "PARAMS" ":"
                        // For simplicity, just parse the param assignments
                        params = self.parse_param_defaults(line)?;
                        break;
                    } else if n_upper.starts_with("PARAMS:") {
                        // Handle case where "PARAMS:" is a single token
                        self.advance();
                        params = self.parse_param_defaults(line)?;
                        break;
                    } else {
                        ports.push(n.clone());
                        self.advance();
                    }
                }
                Token::Value(n) => {
                    ports.push(n.clone());
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        // Start a new subcircuit definition with params
        self.current_subckt = Some(SubcircuitDefinition::new_with_params(
            name.to_uppercase(),
            ports,
            params,
        ));
        self.skip_to_eol();
        Ok(())
    }

    /// Parse parameter defaults: `param=value param2=value2 ...`
    /// Returns a HashMap of uppercase parameter names to values.
    fn parse_param_defaults(&mut self, line: usize) -> Result<HashMap<String, f64>> {
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
                                "Expected '=' after parameter name '{}' in PARAMS:",
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

    /// Parse .PARAM name=expression [name=expression ...]
    /// Expressions can include arithmetic operations and references to other parameters.
    /// Parameters are resolved immediately during Pass 1, using previously defined parameters.
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

                    // Collect expression tokens until we hit a new parameter assignment or EOL
                    let expr_str = self.collect_param_expression();

                    if expr_str.is_empty() {
                        return Err(Error::ParseError {
                            line,
                            message: format!(".PARAM: expected value for '{}'", pname),
                        });
                    }

                    // Build set of known parameter names for the expression parser
                    let param_names: HashSet<String> = self.parameters.keys().cloned().collect();

                    // Parse expression with parameter context
                    let expr =
                        parse_expression_with_params(&expr_str, &param_names).map_err(|e| {
                            Error::ParseError {
                                line,
                                message: format!(
                                    ".PARAM: invalid expression for '{}': {}",
                                    pname, e
                                ),
                            }
                        })?;

                    // Evaluate immediately (parameters are compile-time constants)
                    let ctx = EvalContext::params_only(&self.parameters);
                    let value = expr.eval(&ctx);

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

    /// Collect tokens for a parameter expression, returning as a string.
    ///
    /// Collects tokens until:
    /// - EOL/EOF
    /// - A name followed by '=' (start of next parameter)
    fn collect_param_expression(&mut self) -> String {
        let mut tokens = Vec::new();
        let mut paren_depth = 0;

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::LParen => {
                    paren_depth += 1;
                    tokens.push("(".to_string());
                    self.advance();
                }
                Token::RParen => {
                    paren_depth -= 1;
                    tokens.push(")".to_string());
                    self.advance();
                }
                Token::Comma => {
                    tokens.push(",".to_string());
                    self.advance();
                }
                Token::Star => {
                    tokens.push("*".to_string());
                    self.advance();
                }
                Token::Slash => {
                    tokens.push("/".to_string());
                    self.advance();
                }
                Token::Caret => {
                    tokens.push("^".to_string());
                    self.advance();
                }
                Token::Name(n) if paren_depth == 0 => {
                    // Check if this looks like a new param assignment (name=...)
                    let name = n.clone();
                    // Peek ahead to see if there's an equals sign
                    if self.is_next_equals() {
                        // This is a new parameter, stop collecting
                        break;
                    }
                    tokens.push(name);
                    self.advance();
                }
                Token::Name(n) => {
                    tokens.push(n.clone());
                    self.advance();
                }
                Token::Value(v) => {
                    tokens.push(v.clone());
                    self.advance();
                }
                Token::Equals if paren_depth == 0 => {
                    // Unexpected equals outside parens - stop
                    break;
                }
                _ => {
                    self.advance();
                }
            }
        }

        tokens.join(" ")
    }

    /// Check if the next non-whitespace token is '='.
    fn is_next_equals(&self) -> bool {
        // Look at position + 1 for the next token
        if self.pos + 1 < self.tokens.len() {
            matches!(self.tokens[self.pos + 1].token, Token::Equals)
        } else {
            false
        }
    }

    /// Parse .MEAS/.MEASURE command.
    ///
    /// Syntax examples:
    /// - `.MEAS TRAN delay TRIG V(in) VAL=0.5 RISE=1 TARG V(out) VAL=0.5 RISE=1`
    /// - `.MEAS TRAN vmax MAX V(out)`
    /// - `.MEAS TRAN vavg AVG V(out) FROM=0 TO=10u`
    /// - `.MEAS DC gain FIND V(out) AT=2.5`
    fn parse_meas_command(&mut self, line: usize) -> Result<()> {
        // Parse analysis type: TRAN, DC, AC
        let analysis = match self.peek() {
            Token::Name(n) => {
                let a = match n.to_uppercase().as_str() {
                    "TRAN" => MeasureAnalysis::Tran,
                    "DC" => MeasureAnalysis::Dc,
                    "AC" => MeasureAnalysis::Ac,
                    other => {
                        return Err(Error::ParseError {
                            line,
                            message: format!(
                                ".MEAS: unknown analysis type '{}' (expected TRAN, DC, AC)",
                                other
                            ),
                        });
                    }
                };
                self.advance();
                a
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: ".MEAS: expected analysis type (TRAN, DC, AC)".to_string(),
                });
            }
        };

        // Parse measurement name
        let name = self.expect_name(line)?;

        // Parse measurement type based on first keyword
        let measure_type = self.parse_measure_type(line)?;

        self.measurements.push(Measurement {
            name,
            analysis,
            measure_type,
        });

        self.skip_to_eol();
        Ok(())
    }

    /// Parse the measure type and parameters.
    fn parse_measure_type(&mut self, line: usize) -> Result<MeasureType> {
        let keyword = match self.peek() {
            Token::Name(n) => {
                let k = n.to_uppercase();
                self.advance();
                k
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: ".MEAS: expected measurement keyword".to_string(),
                });
            }
        };

        match keyword.as_str() {
            "TRIG" => self.parse_trig_targ(line),
            "FIND" => self.parse_find(line),
            "AVG" => self.parse_statistic(StatFunc::Avg, line),
            "RMS" => self.parse_statistic(StatFunc::Rms, line),
            "MIN" => self.parse_statistic(StatFunc::Min, line),
            "MAX" => self.parse_statistic(StatFunc::Max, line),
            "PP" => self.parse_statistic(StatFunc::Pp, line),
            "INTEG" => self.parse_statistic(StatFunc::Integ, line),
            other => Err(Error::ParseError {
                line,
                message: format!(
                    ".MEAS: unknown measurement type '{}' (expected TRIG, FIND, AVG, RMS, MIN, MAX, PP, INTEG)",
                    other
                ),
            }),
        }
    }

    /// Parse TRIG...TARG measurement.
    fn parse_trig_targ(&mut self, line: usize) -> Result<MeasureType> {
        let trig_expr = self.collect_meas_expression()?;
        let (trig_val, trig_type) = self.parse_val_and_trigger(line)?;

        match self.peek() {
            Token::Name(n) if n.to_uppercase() == "TARG" => {
                self.advance();
            }
            _ => {
                return Err(Error::ParseError {
                    line,
                    message: ".MEAS: expected TARG after TRIG specification".to_string(),
                });
            }
        }

        let targ_expr = self.collect_meas_expression()?;
        let (targ_val, targ_type) = self.parse_val_and_trigger(line)?;

        Ok(MeasureType::TrigTarg {
            trig_expr,
            trig_val,
            trig_type,
            targ_expr,
            targ_val,
            targ_type,
        })
    }

    /// Parse FIND measurement (FIND...WHEN or FIND...AT).
    fn parse_find(&mut self, line: usize) -> Result<MeasureType> {
        let find_expr = self.collect_meas_expression()?;

        match self.peek() {
            Token::Name(n) => {
                let kw = n.to_uppercase();
                match kw.as_str() {
                    "WHEN" => {
                        self.advance();
                        let when_expr = self.collect_meas_expression()?;
                        let (when_val, when_type) = self.parse_val_and_trigger(line)?;
                        Ok(MeasureType::FindWhen {
                            find_expr,
                            when_expr,
                            when_val,
                            when_type,
                        })
                    }
                    "AT" => {
                        self.advance();
                        if matches!(self.peek(), Token::Equals) {
                            self.advance();
                        }
                        let at_value = self.expect_value(line)?;
                        Ok(MeasureType::FindAt {
                            find_expr,
                            at_value,
                        })
                    }
                    _ => Err(Error::ParseError {
                        line,
                        message: ".MEAS FIND: expected WHEN or AT".to_string(),
                    }),
                }
            }
            _ => Err(Error::ParseError {
                line,
                message: ".MEAS FIND: expected WHEN or AT".to_string(),
            }),
        }
    }

    /// Parse statistic measurement (AVG, RMS, MIN, MAX, PP, INTEG).
    fn parse_statistic(&mut self, func: StatFunc, line: usize) -> Result<MeasureType> {
        let expr = self.collect_meas_expression()?;

        let mut from = None;
        let mut to = None;

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let kw = n.to_uppercase();
                    match kw.as_str() {
                        "FROM" => {
                            self.advance();
                            if matches!(self.peek(), Token::Equals) {
                                self.advance();
                            }
                            from = Some(self.expect_value(line)?);
                        }
                        "TO" => {
                            self.advance();
                            if matches!(self.peek(), Token::Equals) {
                                self.advance();
                            }
                            to = Some(self.expect_value(line)?);
                        }
                        _ => {
                            self.advance();
                        }
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        Ok(MeasureType::Statistic {
            func,
            expr,
            from,
            to,
        })
    }

    /// Collect expression tokens for .MEAS until hitting a keyword.
    fn collect_meas_expression(&mut self) -> Result<String> {
        let mut tokens = Vec::new();
        let mut paren_depth = 0;

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::LParen => {
                    paren_depth += 1;
                    tokens.push("(".to_string());
                    self.advance();
                }
                Token::RParen => {
                    if paren_depth == 0 {
                        break;
                    }
                    paren_depth -= 1;
                    tokens.push(")".to_string());
                    self.advance();
                }
                Token::Name(n) if paren_depth == 0 => {
                    let upper = n.to_uppercase();
                    if matches!(
                        upper.as_str(),
                        "VAL" | "RISE" | "FALL" | "CROSS" | "TARG" | "WHEN" | "AT" | "FROM" | "TO"
                    ) {
                        break;
                    }
                    tokens.push(n.clone());
                    self.advance();
                }
                Token::Name(n) => {
                    tokens.push(n.clone());
                    self.advance();
                }
                Token::Value(v) => {
                    tokens.push(v.clone());
                    self.advance();
                }
                Token::Comma => {
                    tokens.push(",".to_string());
                    self.advance();
                }
                Token::Star => {
                    tokens.push("*".to_string());
                    self.advance();
                }
                Token::Slash => {
                    tokens.push("/".to_string());
                    self.advance();
                }
                Token::Caret => {
                    tokens.push("^".to_string());
                    self.advance();
                }
                Token::Equals if paren_depth == 0 => break,
                _ => {
                    self.advance();
                }
            }
        }

        Ok(tokens.join(""))
    }

    /// Parse VAL=value and optional RISE=n/FALL=n/CROSS=n.
    fn parse_val_and_trigger(&mut self, line: usize) -> Result<(f64, TriggerType)> {
        let mut val = 0.0;
        let mut trigger = TriggerType::default();

        loop {
            match self.peek() {
                Token::Eol | Token::Eof => break,
                Token::Name(n) => {
                    let kw = n.to_uppercase();
                    match kw.as_str() {
                        "VAL" => {
                            self.advance();
                            if matches!(self.peek(), Token::Equals) {
                                self.advance();
                            }
                            val = self.expect_value(line)?;
                        }
                        "RISE" => {
                            self.advance();
                            if matches!(self.peek(), Token::Equals) {
                                self.advance();
                            }
                            let n = self.expect_value(line)? as usize;
                            trigger = TriggerType::Rise(n.max(1));
                        }
                        "FALL" => {
                            self.advance();
                            if matches!(self.peek(), Token::Equals) {
                                self.advance();
                            }
                            let n = self.expect_value(line)? as usize;
                            trigger = TriggerType::Fall(n.max(1));
                        }
                        "CROSS" => {
                            self.advance();
                            if matches!(self.peek(), Token::Equals) {
                                self.advance();
                            }
                            let n = self.expect_value(line)? as usize;
                            trigger = TriggerType::Cross(n.max(1));
                        }
                        "TARG" | "WHEN" | "AT" | "FROM" | "TO" => break,
                        _ => {
                            self.advance();
                        }
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        Ok((val, trigger))
    }
}

/// Parse BSIM3v3 model parameters from parameter list.
fn parse_bsim3_params(params: &[(String, f64)], mos_type: MosfetType) -> Bsim3Params {
    let mut bp = match mos_type {
        MosfetType::Nmos => Bsim3Params::nmos_default(),
        MosfetType::Pmos => Bsim3Params::pmos_default(),
        _ => Bsim3Params::nmos_default(), // Fallback for future variants
    };

    for (k, v) in params {
        match k.as_str() {
            // Geometry parameters
            "TOX" => bp.tox = *v,
            "LINT" | "DL" => bp.lint = *v,
            "WINT" | "DW" => bp.wint = *v,

            // Threshold voltage parameters
            "VTH0" | "VTHO" => bp.vth0 = *v,
            "K1" => bp.k1 = *v,
            "K2" => bp.k2 = *v,
            "DVT0" => bp.dvt0 = *v,
            "DVT1" => bp.dvt1 = *v,
            "DVT2" => bp.dvt2 = *v,
            "NLX" => bp.nlx = *v,
            "VOFF" => bp.voff = *v,
            "NFACTOR" => bp.nfactor = *v,

            // Mobility parameters
            "U0" | "UO" => bp.u0 = *v,
            "UA" => bp.ua = *v,
            "UB" => bp.ub = *v,
            "UC" => bp.uc = *v,
            "VSAT" => bp.vsat = *v,

            // Output conductance parameters
            "PCLM" => bp.pclm = *v,
            "PDIBLC1" => bp.pdiblc1 = *v,
            "PDIBLC2" => bp.pdiblc2 = *v,
            "DROUT" => bp.drout = *v,
            "DELTA" => bp.delta = *v,

            // DIBL parameters
            "ETA0" => bp.eta0 = *v,
            "ETAB" => bp.etab = *v,
            "DSUB" => bp.dsub = *v,

            // Width effect parameters (Phase 2)
            "K3" => bp.k3 = *v,
            "K3B" => bp.k3b = *v,
            "W0" => bp.w0 = *v,
            "DVT0W" => bp.dvt0w = *v,
            "DVT1W" => bp.dvt1w = *v,
            "DVT2W" => bp.dvt2w = *v,

            // Enhanced DIBL parameters (Phase 2)
            "PDIBLCB" => bp.pdiblcb = *v,
            "FPROUT" => bp.fprout = *v,
            "PVAG" => bp.pvag = *v,

            // Substrate current parameters (Phase 2)
            "ALPHA0" => bp.alpha0 = *v,
            "BETA0" => bp.beta0 = *v,

            // Parasitic resistance
            "RDSW" => bp.rdsw = *v,
            "RD" => bp.rd = *v,
            "RS" => bp.rs = *v,
            "PRWB" => bp.prwb = *v,
            "PRWG" => bp.prwg = *v,

            // Process parameters
            "NCH" => bp.nch = *v,
            "NGATE" => bp.ngate = *v,
            "NSUB" => bp.nsub = *v,
            "XT" => bp.xt = *v,

            // Capacitance parameters (Phase 3)
            "CGSO" => bp.cgso = *v,
            "CGDO" => bp.cgdo = *v,
            "CGBO" => bp.cgbo = *v,
            "CJ" => bp.cj = *v,
            "CJSW" => bp.cjsw = *v,
            "CJSWG" => bp.cjswg = *v,
            "MJ" => bp.mj = *v,
            "MJSW" => bp.mjsw = *v,
            "MJSWG" => bp.mjswg = *v,
            "PB" => bp.pb = *v,
            "PBSW" => bp.pbsw = *v,
            "PBSWG" => bp.pbswg = *v,

            // Temperature parameters (Phase 4)
            "TNOM" => bp.tnom = *v + 273.15, // Convert °C to K if needed
            "KT1" => bp.kt1 = *v,
            "KT1L" => bp.kt1l = *v,
            "KT2" => bp.kt2 = *v,
            "UTE" => bp.ute = *v,
            "UA1" => bp.ua1 = *v,
            "UB1" => bp.ub1 = *v,
            "UC1" => bp.uc1 = *v,
            "AT" => bp.at = *v,
            "PRT" => bp.prt = *v,

            // Instance parameters (model defaults)
            "W" => bp.w = *v,
            "L" => bp.l = *v,
            "NF" => bp.nf = *v,
            "AS" => bp.as_ = *v,
            "AD" => bp.ad = *v,
            "PS" => bp.ps = *v,
            "PD" => bp.pd = *v,

            // Skip LEVEL and unrecognized parameters
            _ => {}
        }
    }

    bp
}

/// Parse capacitor model parameters from parameter list.
fn parse_capacitor_params(params: &[(String, f64)]) -> CapacitorParams {
    let mut cp = CapacitorParams::default();

    for (k, v) in params {
        match k.as_str() {
            "CJ" | "CJAREA" | "C_PER_AREA" => cp.c_per_area = *v,
            "C" | "CAP" => cp.c_base = *v,
            "VC1" => cp.vc1 = *v,
            "VC2" => cp.vc2 = *v,
            "TC1" => cp.tc1 = *v,
            "TC2" => cp.tc2 = *v,
            "RS" => cp.rs = *v,
            "RP" => cp.rp = *v,
            "W" => cp.w = *v,
            "L" => cp.l = *v,
            "TNOM" => cp.tnom = *v + 273.15, // Convert °C to K
            _ => {}
        }
    }

    cp
}

/// Parse BSIM4 model parameters from parameter list.
fn parse_bsim4_params(params: &[(String, f64)], mos_type: MosfetType) -> Bsim4Params {
    let mut bp = match mos_type {
        MosfetType::Nmos => Bsim4Params::nmos_default(),
        MosfetType::Pmos => Bsim4Params::pmos_default(),
        _ => Bsim4Params::nmos_default(),
    };

    for (k, v) in params {
        match k.as_str() {
            // Geometry parameters
            "TOXE" => bp.toxe = *v,
            "TOXP" => bp.toxp = *v,
            "TOXM" => bp.toxm = *v,
            // Also accept TOX as alias for TOXE (backward compat with BSIM3 netlists)
            "TOX" => {
                bp.toxe = *v;
                bp.toxp = *v;
                bp.toxm = *v;
            }
            "LINT" | "DL" => bp.lint = *v,
            "WINT" | "DW" => bp.wint = *v,
            "LMIN" => bp.lmin = *v,
            "WMIN" => bp.wmin = *v,

            // Threshold voltage parameters
            "VTH0" | "VTHO" => bp.vth0 = *v,
            "K1" => bp.k1 = *v,
            "K2" => bp.k2 = *v,
            "DVT0" => bp.dvt0 = *v,
            "DVT1" => bp.dvt1 = *v,
            "DVT2" => bp.dvt2 = *v,
            "NLX" => bp.nlx = *v,
            "VOFF" => bp.voff = *v,
            "NFACTOR" => bp.nfactor = *v,
            "VFB" => bp.vfb = *v,

            // Width effect parameters
            "K3" => bp.k3 = *v,
            "K3B" => bp.k3b = *v,
            "W0" => bp.w0 = *v,
            "DVT0W" => bp.dvt0w = *v,
            "DVT1W" => bp.dvt1w = *v,
            "DVT2W" => bp.dvt2w = *v,

            // Mobility parameters
            "U0" | "UO" => bp.u0 = *v,
            "UA" => bp.ua = *v,
            "UB" => bp.ub = *v,
            "UC" => bp.uc = *v,
            "VSAT" => bp.vsat = *v,
            "MOBMOD" => bp.mobmod = *v as i32,
            "EU" => bp.eu = *v,

            // Output conductance parameters
            "PCLM" => bp.pclm = *v,
            "PDIBLC1" => bp.pdiblc1 = *v,
            "PDIBLC2" => bp.pdiblc2 = *v,
            "DROUT" => bp.drout = *v,
            "DELTA" => bp.delta = *v,
            "PDIBLCB" => bp.pdiblcb = *v,
            "FPROUT" => bp.fprout = *v,
            "PVAG" => bp.pvag = *v,

            // DIBL parameters
            "ETA0" => bp.eta0 = *v,
            "ETAB" => bp.etab = *v,
            "DSUB" => bp.dsub = *v,

            // Substrate current parameters
            "ALPHA0" => bp.alpha0 = *v,
            "BETA0" => bp.beta0 = *v,

            // Parasitic resistance
            "RDSW" => bp.rdsw = *v,
            "RD" => bp.rd = *v,
            "RS" => bp.rs = *v,
            "PRWB" => bp.prwb = *v,
            "PRWG" => bp.prwg = *v,
            "RSH" => bp.rsh = *v,

            // Process parameters
            "NCH" => bp.nch = *v,
            "NGATE" => bp.ngate = *v,
            "NSUB" => bp.nsub = *v,
            "XT" => bp.xt = *v,
            "NDEP" => bp.ndep = *v,

            // Quantum mechanical effect parameters
            "QME1" => bp.qme1 = *v,
            "QME2" => bp.qme2 = *v,
            "QME3" => bp.qme3 = *v,
            "POLYMOD" => bp.polymod = *v as i32,

            // Stress effect parameters
            "SAREF" => bp.saref = *v,
            "SBREF" => bp.sbref = *v,
            "KU0" => bp.ku0 = *v,
            "KVTH0" => bp.kvth0 = *v,
            "STK2" => bp.stk2 = *v,
            "STHETA" => bp.stheta = *v,

            // Gate tunneling current
            "AGIDL" => bp.agidl = *v,
            "BGIDL" => bp.bgidl = *v,
            "CGIDL" => bp.cgidl = *v,
            "EGIDL" => bp.egidl = *v,

            // Capacitance parameters
            "CGSO" => bp.cgso = *v,
            "CGDO" => bp.cgdo = *v,
            "CGBO" => bp.cgbo = *v,
            "CJ" => bp.cj = *v,
            "CJSW" => bp.cjsw = *v,
            "CJSWG" => bp.cjswg = *v,
            "MJ" => bp.mj = *v,
            "MJSW" => bp.mjsw = *v,
            "MJSWG" => bp.mjswg = *v,
            "PB" => bp.pb = *v,
            "PBSW" => bp.pbsw = *v,
            "PBSWG" => bp.pbswg = *v,
            "CAPMOD" => bp.capmod = *v as i32,

            // Temperature parameters
            "TNOM" => bp.tnom = *v + 273.15,
            "KT1" => bp.kt1 = *v,
            "KT1L" => bp.kt1l = *v,
            "KT2" => bp.kt2 = *v,
            "UTE" => bp.ute = *v,
            "UA1" => bp.ua1 = *v,
            "UB1" => bp.ub1 = *v,
            "UC1" => bp.uc1 = *v,
            "AT" => bp.at = *v,
            "PRT" => bp.prt = *v,

            // Instance parameters (model defaults)
            "W" => bp.w = *v,
            "L" => bp.l = *v,
            "NF" => bp.nf = *v,
            "AS" => bp.as_ = *v,
            "AD" => bp.ad = *v,
            "PS" => bp.ps = *v,
            "PD" => bp.pd = *v,
            "NRD" => bp.nrd = *v,
            "NRS" => bp.nrs = *v,
            "MULT" => bp.mult = *v,

            // Skip LEVEL and unrecognized parameters
            _ => {}
        }
    }

    bp
}
