//! SPICE netlist parser.

use spicier_core::{Netlist, NodeId, units::parse_value};
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
    /// Transient analysis (.TRAN tstep tstop [tstart]).
    Tran {
        tstep: f64,
        tstop: f64,
        tstart: f64,
    },
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

/// Parser state.
struct Parser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
    netlist: Netlist,
    analyses: Vec<AnalysisCommand>,
    node_map: std::collections::HashMap<String, NodeId>,
    next_current_index: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [SpannedToken]) -> Self {
        let mut node_map = std::collections::HashMap::new();
        // Pre-register ground node aliases
        node_map.insert("0".to_string(), NodeId::GROUND);
        node_map.insert("gnd".to_string(), NodeId::GROUND);
        node_map.insert("GND".to_string(), NodeId::GROUND);

        Self {
            tokens,
            pos: 0,
            netlist: Netlist::new(),
            analyses: Vec::new(),
            node_map,
            next_current_index: 0,
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
    fn parse_tran_command(&mut self, line: usize) -> Result<()> {
        let tstep = self.expect_value(line)?;
        let tstop = self.expect_value(line)?;

        // Optional tstart
        let tstart = self.try_value().unwrap_or(0.0);

        self.analyses.push(AnalysisCommand::Tran {
            tstep,
            tstop,
            tstart,
        });

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
            } => {
                assert!((tstep - 1e-6).abs() < 1e-12);
                assert!((tstop - 5e-3).abs() < 1e-9);
                assert!((tstart - 0.0).abs() < 1e-12);
            }
            _ => panic!("Expected TRAN analysis command"),
        }
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
}
