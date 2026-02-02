//! SPICE netlist parser.

use std::collections::HashMap;

use spicier_core::{Netlist, NodeId, units::parse_value};
use spicier_devices::bjt::BjtParams;
use spicier_devices::diode::DiodeParams;
use spicier_devices::jfet::JfetParams;
use spicier_devices::mosfet::MosfetParams;

use crate::error::{Error, Result};
use crate::lexer::{Lexer, SpannedToken, Token};

mod commands;
mod elements;
mod subcircuit;
pub mod types;
mod waveforms;

pub use types::{
    AcSweepType, AnalysisCommand, DcSweepSpec, InitialCondition, OutputVariable, ParseResult,
    PrintAnalysisType, PrintCommand, RawElementLine, SubcircuitDefinition,
};

use types::SubcircuitDefinition as SubcircuitDef;

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
pub(crate) enum ModelDefinition {
    Diode(DiodeParams),
    Nmos(MosfetParams),
    Pmos(MosfetParams),
    Njf(JfetParams),
    Pjf(JfetParams),
    Npn(BjtParams),
    Pnp(BjtParams),
}

/// Parser state.
pub(crate) struct Parser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
    pub(crate) netlist: Netlist,
    pub(crate) analyses: Vec<AnalysisCommand>,
    pub(crate) initial_conditions: Vec<InitialCondition>,
    pub(crate) print_commands: Vec<PrintCommand>,
    pub(crate) node_map: HashMap<String, NodeId>,
    pub(crate) next_current_index: usize,
    pub(crate) models: HashMap<String, ModelDefinition>,
    /// Subcircuit definitions.
    pub(crate) subcircuits: HashMap<String, SubcircuitDef>,
    /// Current subcircuit being parsed (None if at top level).
    pub(crate) current_subckt: Option<SubcircuitDef>,
    /// Parameters from .PARAM commands (stored as uppercase keys).
    pub(crate) parameters: HashMap<String, f64>,
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
            parameters: HashMap::new(),
        }
    }

    fn parse_all(mut self) -> Result<ParseResult> {
        // First non-comment line is the title
        self.skip_eol();
        if let Some(title) = self.parse_title() {
            self.netlist = Netlist::with_title(title);
        }

        // Two-pass parsing to handle forward model references and parameters:
        // Pass 1: Scan for all .MODEL and .PARAM commands first
        let saved_pos = self.pos;
        while !self.is_at_end() {
            self.skip_eol();
            if self.is_at_end() {
                break;
            }

            if let Token::Command(cmd) = self.peek() {
                // Note: Token::Command stores command without the leading dot
                if cmd == "MODEL" {
                    self.advance(); // consume .MODEL
                    let line = self.current_line();
                    self.parse_model_command(line)?;
                } else if cmd == "PARAM" {
                    self.advance(); // consume .PARAM
                    let line = self.current_line();
                    self.parse_param_command(line)?;
                } else {
                    self.skip_to_eol();
                }
            } else {
                self.skip_to_eol();
            }
        }

        // Reset position and parse everything
        self.pos = saved_pos;

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
            parameters: self.parameters,
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

    // Utility methods

    pub(crate) fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map(|t| &t.token)
            .unwrap_or(&Token::Eof)
    }

    pub(crate) fn current_line(&self) -> usize {
        self.tokens.get(self.pos).map(|t| t.line).unwrap_or(0)
    }

    pub(crate) fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    pub(crate) fn skip_eol(&mut self) {
        while matches!(self.peek(), Token::Eol) {
            self.advance();
        }
    }

    pub(crate) fn skip_to_eol(&mut self) {
        while !matches!(self.peek(), Token::Eol | Token::Eof) {
            self.advance();
        }
        if matches!(self.peek(), Token::Eol) {
            self.advance();
        }
    }

    // Expect/try helpers

    pub(crate) fn expect_name(&mut self, line: usize) -> Result<String> {
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

    pub(crate) fn expect_node(&mut self, line: usize) -> Result<NodeId> {
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

    /// Resolve a value string: first try numeric parsing, then parameter lookup.
    pub(crate) fn resolve_value(&self, s: &str) -> Option<f64> {
        // First try direct numeric parsing
        if let Some(val) = parse_value(s) {
            return Some(val);
        }
        // Then try parameter lookup (case-insensitive)
        self.parameters.get(&s.to_uppercase()).copied()
    }

    pub(crate) fn expect_value(&mut self, line: usize) -> Result<f64> {
        match self.peek() {
            Token::Value(v) | Token::Name(v) => {
                let v = v.clone();
                self.advance();
                self.resolve_value(&v).ok_or(Error::InvalidValue(v))
            }
            _ => Err(Error::ParseError {
                line,
                message: "expected value".to_string(),
            }),
        }
    }

    pub(crate) fn try_value(&mut self) -> Option<f64> {
        match self.peek() {
            Token::Value(v) | Token::Name(v) => {
                let v = v.clone();
                if let Some(val) = self.resolve_value(&v) {
                    self.advance();
                    Some(val)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to parse a value, returning None if not available.
    pub(crate) fn try_expect_value(&mut self) -> Option<f64> {
        match self.peek() {
            Token::Value(s) => {
                let s = s.clone();
                if let Some(v) = self.resolve_value(&s) {
                    self.advance();
                    Some(v)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn get_or_create_node(&mut self, name: &str) -> NodeId {
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
    fn test_parse_behavioral_voltage() {
        let input = r#"Behavioral Voltage Test
V1 1 0 10
R1 2 0 1k
B1 2 0 V=V(1)*0.5
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3); // V1, R1, B1
        assert_eq!(netlist.num_current_vars(), 2); // V1 + B1 (voltage source)
    }

    #[test]
    fn test_parse_behavioral_current() {
        let input = r#"Behavioral Current Test
V1 1 0 10
R1 1 2 1k
B1 2 0 I=V(2)/1k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3); // V1, R1, B1
        assert_eq!(netlist.num_current_vars(), 1); // V1 only (B1 is current source)
    }

    #[test]
    fn test_parse_behavioral_time_varying() {
        let input = r#"Behavioral Time-Varying Test
B1 1 0 V=sin(2*pi*1k*time)
R1 1 0 1k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2); // B1, R1
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
        assert_eq!(
            result.print_commands[0].analysis_type,
            PrintAnalysisType::Dc
        );
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
        assert_eq!(
            result.print_commands[0].analysis_type,
            PrintAnalysisType::Ac
        );
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
        assert!(
            result.subcircuits.contains_key("VDIV"),
            "VDIV should be defined"
        );
        let subckt = &result.subcircuits["VDIV"];
        assert_eq!(subckt.name, "VDIV");
        assert_eq!(subckt.ports, vec!["in", "out"]);
        assert_eq!(subckt.elements.len(), 2); // R1 and R2

        // Check netlist has expanded devices: V1, X1.R1, X1.R2
        assert_eq!(result.netlist.num_devices(), 3);
    }

    #[test]
    fn test_parse_param_simple() {
        let input = r#"Param Test
.PARAM R_val=1k
R1 1 0 R_val
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.parameters.len(), 1);
        assert!((result.parameters["R_VAL"] - 1000.0).abs() < 1e-10);
        assert_eq!(result.netlist.num_devices(), 1);
    }

    #[test]
    fn test_parse_param_multiple_on_line() {
        let input = r#"Multiple Params
.PARAM R_val=1k C_val=10u V_val=5
R1 1 2 R_val
C1 2 0 C_val
V1 1 0 V_val
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.parameters.len(), 3);
        assert!((result.parameters["R_VAL"] - 1000.0).abs() < 1e-10);
        assert!((result.parameters["C_VAL"] - 10e-6).abs() < 1e-12);
        assert!((result.parameters["V_VAL"] - 5.0).abs() < 1e-10);
        assert_eq!(result.netlist.num_devices(), 3);
    }

    #[test]
    fn test_parse_param_with_si_suffix() {
        let input = r#"SI Suffix Test
.PARAM freq=1MEG cap=100p ind=10n
R1 1 0 1k
.end
"#;

        let result = parse_full(input).unwrap();
        assert!((result.parameters["FREQ"] - 1e6).abs() < 1e-4);
        assert!((result.parameters["CAP"] - 100e-12).abs() < 1e-18);
        assert!((result.parameters["IND"] - 10e-9).abs() < 1e-15);
    }

    #[test]
    fn test_parse_param_case_insensitive() {
        let input = r#"Case Test
.PARAM MyValue=100
R1 1 0 myvalue
R2 2 0 MYVALUE
R3 3 0 MyValue
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.parameters.len(), 1);
        assert!((result.parameters["MYVALUE"] - 100.0).abs() < 1e-10);
        assert_eq!(result.netlist.num_devices(), 3);
    }

    #[test]
    fn test_parse_param_in_various_elements() {
        let input = r#"Param Elements Test
.PARAM R=1k C=1u L=1m V=5 I=10m
R1 1 2 R
C1 2 0 C
L1 1 3 L
V1 1 0 V
I1 0 2 I
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.netlist.num_devices(), 5);
    }

    #[test]
    fn test_parse_param_multiple_lines() {
        let input = r#"Multi Line Param Test
.PARAM R1_val=1k
.PARAM R2_val=2k
R1 1 2 R1_val
R2 2 0 R2_val
V1 1 0 10
.end
"#;

        let result = parse_full(input).unwrap();
        assert_eq!(result.parameters.len(), 2);
        assert!((result.parameters["R1_VAL"] - 1000.0).abs() < 1e-10);
        assert!((result.parameters["R2_VAL"] - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_param_spaces_around_equals() {
        let input = r#"Spaces Test
.PARAM R_val = 1k
R1 1 0 R_val
.end
"#;

        let result = parse_full(input).unwrap();
        assert!((result.parameters["R_VAL"] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_jfet() {
        let input = r#"JFET Test
V1 1 0 10
VG 2 0 -1
J1 1 2 0
R1 1 0 10k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 4);
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_jfet_with_model() {
        let input = r#"JFET Model Test
.MODEL JMOD NJF (VTO=-2.5 BETA=2e-4 LAMBDA=0.01)
V1 1 0 10
J1 1 2 0 JMOD
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2); // V1, J1
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_pjf_model() {
        let input = r#"PJF Model Test
.MODEL PMOD PJF (VTO=2.0 BETA=1e-4)
V1 1 0 -10
J1 0 2 1 PMOD
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2);
    }

    #[test]
    fn test_parse_bjt() {
        let input = r#"BJT Test
V1 1 0 5
VB 2 0 0.7
Q1 1 2 0
R1 1 0 1k
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 4);
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_bjt_with_model() {
        let input = r#"BJT Model Test
.MODEL QMOD NPN (IS=1e-15 BF=200 VAF=100)
V1 1 0 5
Q1 1 2 0 QMOD
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2); // V1, Q1
        assert!(netlist.has_nonlinear_devices());
    }

    #[test]
    fn test_parse_pnp_model() {
        let input = r#"PNP Model Test
.MODEL PMOD PNP (IS=1e-15 BF=150)
V1 1 0 5
Q1 0 2 1 PMOD
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 2);
    }

    #[test]
    fn test_parse_mutual_inductance() {
        let input = r#"Mutual Inductance Test
V1 1 0 10
L1 1 2 1m
L2 3 0 1m
R1 2 0 1k
R2 3 0 1k
K1 L1 L2 0.9
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 6); // V1, L1, L2, R1, R2, K1
    }

    #[test]
    fn test_parse_bjt_full_params() {
        let input = r#"BJT Full Params
.MODEL Q2N2222 NPN (IS=1e-14 BF=100 BR=1 NF=1 NR=1 VAF=100 RB=10 RE=0.1 RC=1 CJE=25p CJC=8p TF=0.4n TR=10n)
VCC 1 0 5
VB 2 0 0.7
Q1 1 2 0 Q2N2222
.end
"#;

        let netlist = parse(input).unwrap();
        assert_eq!(netlist.num_devices(), 3); // VCC, VB, Q1
    }
}
