//! Public types for the SPICE parser.

use std::collections::HashMap;

use spicier_core::{Netlist, NodeId};

/// AC sweep type parsed from netlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AcSweepType {
    /// Linear frequency spacing.
    Lin,
    /// Logarithmic spacing per decade.
    Dec,
    /// Logarithmic spacing per octave.
    Oct,
}

/// Type of DC sweep variable.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DcSweepType {
    /// Sweep a voltage or current source value.
    #[default]
    Source,
    /// Sweep a parameter value.
    Param,
}

/// A single DC sweep specification.
#[derive(Debug, Clone)]
pub struct DcSweepSpec {
    /// Name of the source or parameter to sweep.
    pub source_name: String,
    /// Start value.
    pub start: f64,
    /// Stop value.
    pub stop: f64,
    /// Step size.
    pub step: f64,
    /// Type of sweep (source or parameter).
    pub sweep_type: DcSweepType,
}

/// An analysis command parsed from the netlist.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Transient analysis (.TRAN tstep tstop \[tstart\] \[tmax\] \[UIC\]).
    Tran {
        tstep: f64,
        tstop: f64,
        tstart: f64,
        /// Use Initial Conditions - skip DC operating point, use .IC values directly.
        uic: bool,
    },
    /// Noise analysis (.NOISE V(output) Vinput sweep_type npoints fstart fstop).
    Noise {
        /// Output node name (e.g., "out" or "2").
        output_node: String,
        /// Optional reference node for differential output.
        output_ref_node: Option<String>,
        /// Input source name (e.g., "V1").
        input_source: String,
        /// Frequency sweep type.
        sweep_type: AcSweepType,
        /// Number of points (per decade/octave for log, total for linear).
        num_points: usize,
        /// Start frequency in Hz.
        fstart: f64,
        /// Stop frequency in Hz.
        fstop: f64,
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum OutputVariable {
    /// Node voltage: V(node) or V(node1, node2) for differential.
    Voltage { node: String, node2: Option<String> },
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
    /// Parameter defaults from PARAMS: section (e.g., `PARAMS: W=1u L=100n`).
    pub params: HashMap<String, f64>,
}

impl SubcircuitDefinition {
    /// Create a new subcircuit definition with optional parameter defaults.
    pub(super) fn new_with_params(
        name: String,
        ports: Vec<String>,
        params: HashMap<String, f64>,
    ) -> Self {
        Self {
            name,
            ports,
            elements: Vec::new(),
            instances: Vec::new(),
            params,
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
    /// Parameters from .PARAM commands (name -> value).
    pub parameters: HashMap<String, f64>,
    /// Measurement statements from .MEAS commands.
    pub measurements: Vec<Measurement>,
}

// ============================================================================
// .MEASURE types
// ============================================================================

/// Analysis type for .MEAS command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeasureAnalysis {
    /// Transient analysis.
    Tran,
    /// DC sweep analysis.
    Dc,
    /// AC analysis.
    Ac,
}

/// Trigger type for TRIG/TARG measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerType {
    /// Rising edge crossing (RISE=n, nth occurrence).
    Rise(usize),
    /// Falling edge crossing (FALL=n, nth occurrence).
    Fall(usize),
    /// Any crossing (CROSS=n, nth occurrence).
    Cross(usize),
}

impl Default for TriggerType {
    fn default() -> Self {
        TriggerType::Rise(1)
    }
}

/// Statistical function for .MEAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatFunc {
    /// Average value.
    Avg,
    /// Root mean square.
    Rms,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Peak-to-peak (max - min).
    Pp,
    /// Integral (trapezoidal).
    Integ,
}

/// Type of measurement from .MEAS command.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MeasureType {
    /// TRIG/TARG: Measure time between two trigger conditions.
    TrigTarg {
        /// Expression to trigger on.
        trig_expr: String,
        /// Trigger threshold value.
        trig_val: f64,
        /// Type of trigger (rise/fall/cross).
        trig_type: TriggerType,
        /// Expression to target.
        targ_expr: String,
        /// Target threshold value.
        targ_val: f64,
        /// Type of target (rise/fall/cross).
        targ_type: TriggerType,
    },
    /// FIND/WHEN: Find value of one expression when another crosses threshold.
    FindWhen {
        /// Expression to find the value of.
        find_expr: String,
        /// Expression to watch for threshold crossing.
        when_expr: String,
        /// Threshold value for when_expr.
        when_val: f64,
        /// Type of crossing.
        when_type: TriggerType,
    },
    /// FIND/AT: Find value of expression at a specific point.
    FindAt {
        /// Expression to find the value of.
        find_expr: String,
        /// The value at which to find (time/dc sweep value/frequency).
        at_value: f64,
    },
    /// Statistical measurement over a range.
    Statistic {
        /// Statistical function to apply.
        func: StatFunc,
        /// Expression to measure.
        expr: String,
        /// Start of measurement range (None = simulation start).
        from: Option<f64>,
        /// End of measurement range (None = simulation end).
        to: Option<f64>,
    },
}

/// A .MEAS statement from the netlist.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Name of the measurement result.
    pub name: String,
    /// Analysis type this measurement applies to.
    pub analysis: MeasureAnalysis,
    /// Type and parameters of the measurement.
    pub measure_type: MeasureType,
}
