//! Element parsing (R, C, L, V, I, D, M, J, Q, K, E, G, F, H, B, T).

use spicier_devices::behavioral::{BehavioralCurrentSource, BehavioralVoltageSource};
use spicier_devices::bjt::{Bjt, BjtParams, BjtType};
use spicier_devices::controlled::{Cccs, Ccvs, Vccs, Vcvs};
use spicier_devices::diode::{Diode, DiodeParams};
use spicier_devices::expression::parse_expression;
use spicier_devices::jfet::{Jfet, JfetParams, JfetType};
use spicier_devices::mosfet::{Bsim3Mosfet, Bsim3Params, Mosfet, MosfetParams, MosfetType};
use spicier_devices::mutual::MutualInductance;
use spicier_devices::passive::{Capacitor, Inductor, Resistor};
use spicier_devices::sources::{CurrentSource, VoltageSource};
use spicier_devices::tline::TransmissionLine;

use crate::error::{Error, Result};
use crate::lexer::Token;

use super::types::RawElementLine;
use super::{ModelDefinition, Parser};

impl<'a> Parser<'a> {
    pub(super) fn parse_element(&mut self, name: &str) -> Result<()> {
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
            'J' => self.parse_jfet(name, line),
            'Q' => self.parse_bjt(name, line),
            'K' => self.parse_mutual_inductance(name, line),
            'E' => self.parse_vcvs(name, line),
            'G' => self.parse_vccs(name, line),
            'F' => self.parse_cccs(name, line),
            'H' => self.parse_ccvs(name, line),
            'B' => self.parse_behavioral(name, line),
            'T' => self.parse_transmission_line(name, line),
            'X' => self.parse_subcircuit_instance(name, line),
            _ => {
                // Unknown element - skip line
                self.skip_to_eol();
                Ok(())
            }
        }
    }

    /// Capture an element line as raw text for subcircuit storage.
    pub(super) fn capture_element_line(&mut self, name: &str) -> String {
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
        let mut waveform: Option<spicier_devices::Waveform> = None;

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
        let node_bulk = self.expect_node(line)?;

        /// Model type indicator for MOSFET parsing.
        #[derive(Clone)]
        #[allow(clippy::large_enum_variant)]
        enum MosfetModel {
            Level1(MosfetType, MosfetParams),
            Bsim3(Bsim3Params),
        }

        // Default to Level 1 NMOS
        let mut model = MosfetModel::Level1(MosfetType::Nmos, MosfetParams::nmos_default());

        // Try to read model name
        if let Token::Name(n) = self.peek() {
            let model_name = n.clone().to_uppercase();
            // Check if it's a param assignment like W=...
            if !model_name.contains('=') && model_name != "W" && model_name != "L" {
                self.advance();
                match self.models.get(&model_name) {
                    Some(ModelDefinition::Nmos(mp)) => {
                        model = MosfetModel::Level1(MosfetType::Nmos, mp.clone());
                    }
                    Some(ModelDefinition::Pmos(mp)) => {
                        model = MosfetModel::Level1(MosfetType::Pmos, mp.clone());
                    }
                    Some(ModelDefinition::Nmos49(bp)) => {
                        model = MosfetModel::Bsim3(bp.clone());
                    }
                    Some(ModelDefinition::Pmos49(bp)) => {
                        model = MosfetModel::Bsim3(bp.clone());
                    }
                    _ => {}
                }
            }
        }

        // Parse optional W=val L=val NF=val parameters
        let mut w_override: Option<f64> = None;
        let mut l_override: Option<f64> = None;
        let mut nf_override: Option<f64> = None;

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
                                "W" => w_override = Some(val),
                                "L" => l_override = Some(val),
                                "NF" | "M" => nf_override = Some(val),
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

        // Create the appropriate device
        match model {
            MosfetModel::Level1(mos_type, mut params) => {
                // Apply instance parameter overrides
                if let Some(w) = w_override {
                    params.w = w;
                }
                if let Some(l) = l_override {
                    params.l = l;
                }
                let mosfet =
                    Mosfet::with_params(name, node_drain, node_gate, node_source, mos_type, params);
                self.netlist.add_device(mosfet);
            }
            MosfetModel::Bsim3(mut params) => {
                // Apply instance parameter overrides
                if let Some(w) = w_override {
                    params.w = w;
                }
                if let Some(l) = l_override {
                    params.l = l;
                }
                if let Some(nf) = nf_override {
                    params.nf = nf;
                }
                let mosfet = Bsim3Mosfet::with_params(
                    name,
                    node_drain,
                    node_gate,
                    node_source,
                    node_bulk,
                    params,
                );
                self.netlist.add_device(mosfet);
            }
        }

        self.skip_to_eol();
        Ok(())
    }

    /// Parse J1 drain gate source [modelname]
    fn parse_jfet(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_drain = self.expect_node(line)?;
        let node_gate = self.expect_node(line)?;
        let node_source = self.expect_node(line)?;

        // Optional model name
        let mut jfet_type = JfetType::Njf;
        let mut params = JfetParams::njf_default();

        // Try to read model name
        if let Token::Name(n) = self.peek() {
            let model_name = n.clone().to_uppercase();
            if !model_name.contains('=') {
                self.advance();
                match self.models.get(&model_name) {
                    Some(ModelDefinition::Njf(jp)) => {
                        jfet_type = JfetType::Njf;
                        params = jp.clone();
                    }
                    Some(ModelDefinition::Pjf(jp)) => {
                        jfet_type = JfetType::Pjf;
                        params = jp.clone();
                    }
                    _ => {}
                }
            }
        }

        let jfet = Jfet::with_params(name, node_drain, node_gate, node_source, jfet_type, params);
        self.netlist.add_device(jfet);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse Q1 collector base emitter [modelname]
    fn parse_bjt(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_collector = self.expect_node(line)?;
        let node_base = self.expect_node(line)?;
        let node_emitter = self.expect_node(line)?;

        // Optional model name
        let mut bjt_type = BjtType::Npn;
        let mut params = BjtParams::npn_default();

        // Try to read model name
        if let Token::Name(n) = self.peek() {
            let model_name = n.clone().to_uppercase();
            if !model_name.contains('=') {
                self.advance();
                match self.models.get(&model_name) {
                    Some(ModelDefinition::Npn(bp)) => {
                        bjt_type = BjtType::Npn;
                        params = bp.clone();
                    }
                    Some(ModelDefinition::Pnp(bp)) => {
                        bjt_type = BjtType::Pnp;
                        params = bp.clone();
                    }
                    _ => {}
                }
            }
        }

        let bjt = Bjt::with_params(
            name,
            node_collector,
            node_base,
            node_emitter,
            bjt_type,
            params,
        );
        self.netlist.add_device(bjt);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse K1 L1 L2 [L3 ...] coupling_coefficient
    ///
    /// Supports multi-winding transformers with 2 or more inductors.
    /// For N inductors, generates N*(N-1)/2 pairwise mutual inductance elements.
    /// Example: K1 L1 L2 L3 0.9 creates K1_L1_L2, K1_L1_L3, K1_L2_L3
    fn parse_mutual_inductance(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        // Collect inductor names until we hit a number (the coupling coefficient)
        let mut inductor_names = Vec::new();
        loop {
            match self.peek() {
                Token::Name(n) => {
                    // Check if it starts with L (inductor) - inductor names should start with L
                    let upper = n.to_uppercase();
                    if upper.starts_with('L') || inductor_names.is_empty() {
                        inductor_names.push(n.to_string());
                        self.advance();
                    } else {
                        // Not an inductor name, might be a model name or something else
                        // Try to parse as value
                        break;
                    }
                }
                Token::Value(_) => {
                    // This is the coupling coefficient
                    break;
                }
                _ => break,
            }
        }

        // Need at least 2 inductors
        if inductor_names.len() < 2 {
            return Err(Error::ParseError {
                message: format!(
                    "K element requires at least 2 inductors, got {}",
                    inductor_names.len()
                ),
                line,
            });
        }

        let coupling = self.expect_value(line)?;

        // For 2 inductors, create a single mutual inductance (original behavior)
        // For N > 2 inductors, create N*(N-1)/2 pairwise couplings
        if inductor_names.len() == 2 {
            let mut mutual =
                MutualInductance::new(name, &inductor_names[0], &inductor_names[1], coupling);

            // Try to resolve the inductor references immediately if they exist
            if let (Some(l1_idx), Some(l2_idx)) = (
                self.netlist.find_vsource_branch_index(&inductor_names[0]),
                self.netlist.find_vsource_branch_index(&inductor_names[1]),
            ) {
                mutual.resolve(l1_idx, l2_idx, 0.0, 0.0);
            }

            self.netlist.add_device(mutual);
        } else {
            // Multi-winding: create pairwise couplings
            for i in 0..inductor_names.len() {
                for j in (i + 1)..inductor_names.len() {
                    let pair_name = format!("{}_{}_{}", name, inductor_names[i], inductor_names[j]);
                    let mut mutual = MutualInductance::new(
                        &pair_name,
                        &inductor_names[i],
                        &inductor_names[j],
                        coupling,
                    );

                    // Try to resolve immediately
                    if let (Some(li_idx), Some(lj_idx)) = (
                        self.netlist.find_vsource_branch_index(&inductor_names[i]),
                        self.netlist.find_vsource_branch_index(&inductor_names[j]),
                    ) {
                        mutual.resolve(li_idx, lj_idx, 0.0, 0.0);
                    }

                    self.netlist.add_device(mutual);
                }
            }
        }

        self.skip_to_eol();
        Ok(())
    }

    /// Parse E1 out+ out- ctrl+ ctrl- gain (VCVS)
    fn parse_vcvs(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let out_pos = self.expect_node(line)?;
        let out_neg = self.expect_node(line)?;
        let ctrl_pos = self.expect_node(line)?;
        let ctrl_neg = self.expect_node(line)?;
        let gain = self.expect_value(line)?;

        let current_index = self.next_current_index;
        self.next_current_index += 1;

        let vcvs = Vcvs::new(
            name,
            out_pos,
            out_neg,
            ctrl_pos,
            ctrl_neg,
            gain,
            current_index,
        );
        self.netlist.add_device(vcvs);

        self.skip_to_eol();
        Ok(())
    }

    /// Parse G1 out+ out- ctrl+ ctrl- gm (VCCS)
    fn parse_vccs(&mut self, name: &str, line: usize) -> Result<()> {
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

    /// Parse B1 n+ n- V=expr or B1 n+ n- I=expr (behavioral source)
    fn parse_behavioral(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let node_pos = self.expect_node(line)?;
        let node_neg = self.expect_node(line)?;

        // First, check for V= or I= indicator
        let mut is_voltage = false;
        let mut is_current = false;

        // Check if next token is V or I followed by =
        if let Token::Name(n) = self.peek() {
            let n_upper = n.to_uppercase();
            if n_upper == "V" || n_upper == "I" {
                let is_v = n_upper == "V";
                self.advance();
                if matches!(self.peek(), Token::Equals) {
                    self.advance(); // consume =
                    if is_v {
                        is_voltage = true;
                    } else {
                        is_current = true;
                    }
                } else {
                    // Not followed by =, this shouldn't happen in valid B element
                    return Err(Error::ParseError {
                        line,
                        message: format!(
                            "Behavioral source '{}' must specify V= or I= expression",
                            name
                        ),
                    });
                }
            }
        }

        if !is_voltage && !is_current {
            return Err(Error::ParseError {
                line,
                message: format!(
                    "Behavioral source '{}' must specify V= or I= expression",
                    name
                ),
            });
        }

        // Collect the rest of the line as the expression
        let mut expr_str = String::new();

        while !matches!(self.peek(), Token::Eol | Token::Eof) {
            match self.peek() {
                Token::Name(n) => {
                    expr_str.push_str(n);
                    expr_str.push(' ');
                    self.advance();
                }
                Token::Value(v) => {
                    expr_str.push_str(v);
                    expr_str.push(' ');
                    self.advance();
                }
                Token::Equals => {
                    expr_str.push('=');
                    self.advance();
                }
                Token::LParen => {
                    expr_str.push('(');
                    self.advance();
                }
                Token::RParen => {
                    expr_str.push(')');
                    self.advance();
                }
                Token::Comma => {
                    expr_str.push(',');
                    self.advance();
                }
                Token::Star => {
                    expr_str.push('*');
                    self.advance();
                }
                Token::Slash => {
                    expr_str.push('/');
                    self.advance();
                }
                Token::Caret => {
                    expr_str.push('^');
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        let expr_str = expr_str.trim();

        // Parse the expression
        let expr = parse_expression(expr_str).map_err(|e| Error::ParseError {
            line,
            message: format!("Invalid expression '{}': {}", expr_str, e),
        })?;

        if is_voltage {
            let branch_index = self.next_current_index;
            self.next_current_index += 1;
            let source = BehavioralVoltageSource::new(name, node_pos, node_neg, branch_index, expr);
            self.netlist.add_device(source);
        } else if is_current {
            let source = BehavioralCurrentSource::new(name, node_pos, node_neg, expr);
            self.netlist.add_device(source);
        } else {
            return Err(Error::ParseError {
                line,
                message: format!(
                    "Behavioral source '{}' must specify V= or I= expression",
                    name
                ),
            });
        }

        self.skip_to_eol();
        Ok(())
    }

    pub(super) fn expect_value_or_dc(&mut self, line: usize) -> Result<f64> {
        // Handle "DC 5" or just "5"
        if let Token::Name(n) = self.peek()
            && n.to_uppercase() == "DC"
        {
            self.advance(); // skip DC keyword
        }
        self.expect_value(line)
    }

    /// Parse T1 port1+ port1- port2+ port2- Z0=val TD=val [NL=val]
    fn parse_transmission_line(&mut self, name: &str, line: usize) -> Result<()> {
        self.advance(); // consume name

        let port1_pos = self.expect_node(line)?;
        let port1_neg = self.expect_node(line)?;
        let port2_pos = self.expect_node(line)?;
        let port2_neg = self.expect_node(line)?;

        let mut z0: Option<f64> = None;
        let mut td: Option<f64> = None;
        let mut nl: Option<usize> = None;

        // Parse Z0=val TD=val NL=val parameters
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
                                "Z0" => z0 = Some(val),
                                "TD" => td = Some(val),
                                "NL" => nl = Some(val as usize),
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

        let z0 = z0.ok_or_else(|| Error::ParseError {
            line,
            message: format!(
                "Transmission line '{}' requires Z0 parameter (characteristic impedance)",
                name
            ),
        })?;

        let td = td.ok_or_else(|| Error::ParseError {
            line,
            message: format!(
                "Transmission line '{}' requires TD parameter (propagation delay)",
                name
            ),
        })?;

        // Get current index base for the inductors
        let current_base_index = self.next_current_index;

        // Create the transmission line
        let mut tline = if let Some(num_sections) = nl {
            TransmissionLine::with_sections(
                name,
                port1_pos,
                port1_neg,
                port2_pos,
                port2_neg,
                z0,
                td,
                num_sections,
                current_base_index,
            )
        } else {
            TransmissionLine::new(
                name,
                port1_pos,
                port1_neg,
                port2_pos,
                port2_neg,
                z0,
                td,
                current_base_index,
            )
        };

        // Update current index for the inductors
        self.next_current_index += tline.num_inductors();

        // Create internal nodes for the LC ladder
        let num_internal = tline.num_internal_nodes();
        if num_internal > 0 {
            let mut internal_nodes = Vec::with_capacity(num_internal);
            for _ in 0..num_internal {
                // Generate unique internal node names
                let internal_node = self.get_or_create_internal_node(&format!(
                    "{}_int_{}",
                    name,
                    internal_nodes.len()
                ));
                internal_nodes.push(internal_node);
            }
            tline.set_internal_nodes(internal_nodes);
        }

        self.netlist.add_device(tline);

        self.skip_to_eol();
        Ok(())
    }
}
