//! AC small-signal frequency-domain analysis.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::error::Result;
use crate::linear::solve_complex;

/// AC sweep type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcSweepType {
    /// Linear frequency spacing.
    Linear,
    /// Logarithmic spacing, points per decade.
    Decade,
    /// Logarithmic spacing, points per octave.
    Octave,
}

/// AC analysis parameters.
#[derive(Debug, Clone)]
pub struct AcParams {
    /// Start frequency (Hz).
    pub fstart: f64,
    /// Stop frequency (Hz).
    pub fstop: f64,
    /// Number of points (total for Linear, per decade/octave for log sweeps).
    pub num_points: usize,
    /// Sweep type.
    pub sweep_type: AcSweepType,
}

/// Complex MNA system for AC analysis.
///
/// The matrix equation is A*x = b where A, x, b are all complex-valued.
/// Rows/columns 0..num_nodes are node voltages; num_nodes..size are branch currents.
pub struct ComplexMna {
    matrix: DMatrix<Complex<f64>>,
    rhs: DVector<Complex<f64>>,
    num_nodes: usize,
}

impl ComplexMna {
    /// Create a new complex MNA system.
    pub fn new(num_nodes: usize, num_vsources: usize) -> Self {
        let size = num_nodes + num_vsources;
        Self {
            matrix: DMatrix::from_element(size, size, Complex::new(0.0, 0.0)),
            rhs: DVector::from_element(size, Complex::new(0.0, 0.0)),
            num_nodes,
        }
    }

    /// Get a reference to the matrix.
    pub fn matrix(&self) -> &DMatrix<Complex<f64>> {
        &self.matrix
    }

    /// Get a mutable reference to the matrix.
    pub fn matrix_mut(&mut self) -> &mut DMatrix<Complex<f64>> {
        &mut self.matrix
    }

    /// Get a reference to the RHS vector.
    pub fn rhs(&self) -> &DVector<Complex<f64>> {
        &self.rhs
    }

    /// Get a mutable reference to the RHS vector.
    pub fn rhs_mut(&mut self) -> &mut DVector<Complex<f64>> {
        &mut self.rhs
    }

    /// Stamp a complex admittance between two nodes.
    ///
    /// For a two-terminal element with admittance Y between nodes i and j:
    ///   matrix[i,i] += Y,  matrix[j,j] += Y
    ///   matrix[i,j] -= Y,  matrix[j,i] -= Y
    pub fn stamp_admittance(
        &mut self,
        node_i: Option<usize>,
        node_j: Option<usize>,
        y: Complex<f64>,
    ) {
        if let Some(i) = node_i {
            self.matrix[(i, i)] += y;
        }
        if let Some(j) = node_j {
            self.matrix[(j, j)] += y;
        }
        if let (Some(i), Some(j)) = (node_i, node_j) {
            self.matrix[(i, j)] -= y;
            self.matrix[(j, i)] -= y;
        }
    }

    /// Stamp a real conductance between two nodes.
    pub fn stamp_conductance(
        &mut self,
        node_i: Option<usize>,
        node_j: Option<usize>,
        g: f64,
    ) {
        self.stamp_admittance(node_i, node_j, Complex::new(g, 0.0));
    }

    /// Stamp a complex current source flowing from neg to pos.
    pub fn stamp_current_source(
        &mut self,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        current: Complex<f64>,
    ) {
        if let Some(p) = node_pos {
            self.rhs[p] += current;
        }
        if let Some(n) = node_neg {
            self.rhs[n] -= current;
        }
    }

    /// Stamp a voltage source: V(pos) - V(neg) = voltage.
    ///
    /// `branch_idx` is the index of the current variable (0-based, offset from num_nodes).
    pub fn stamp_voltage_source(
        &mut self,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        branch_idx: usize,
        voltage: Complex<f64>,
    ) {
        let bi = self.num_nodes + branch_idx;
        let one = Complex::new(1.0, 0.0);

        if let Some(p) = node_pos {
            self.matrix[(p, bi)] += one;
            self.matrix[(bi, p)] += one;
        }
        if let Some(n) = node_neg {
            self.matrix[(n, bi)] -= one;
            self.matrix[(bi, n)] -= one;
        }
        self.rhs[bi] += voltage;
    }

    /// Stamp an inductor in AC (impedance jωL).
    ///
    /// The inductor branch equation: V(pos) - V(neg) = jωL * I_branch.
    /// `branch_idx` is the current variable index.
    pub fn stamp_inductor(
        &mut self,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        branch_idx: usize,
        omega: f64,
        inductance: f64,
    ) {
        let bi = self.num_nodes + branch_idx;
        let one = Complex::new(1.0, 0.0);

        // KCL contributions (same structure as voltage source)
        if let Some(p) = node_pos {
            self.matrix[(p, bi)] += one;
            self.matrix[(bi, p)] += one;
        }
        if let Some(n) = node_neg {
            self.matrix[(n, bi)] -= one;
            self.matrix[(bi, n)] -= one;
        }

        // Impedance term: -jωL on the branch diagonal
        self.matrix[(bi, bi)] -= Complex::new(0.0, omega * inductance);
    }

    /// Stamp a VCCS (voltage-controlled current source) for small-signal gm.
    ///
    /// Current gm * V(ctrl_pos, ctrl_neg) flows from out_neg to out_pos.
    pub fn stamp_vccs(
        &mut self,
        out_pos: Option<usize>,
        out_neg: Option<usize>,
        ctrl_pos: Option<usize>,
        ctrl_neg: Option<usize>,
        gm: f64,
    ) {
        let gm_c = Complex::new(gm, 0.0);
        if let Some(op) = out_pos {
            if let Some(cp) = ctrl_pos {
                self.matrix[(op, cp)] += gm_c;
            }
            if let Some(cn) = ctrl_neg {
                self.matrix[(op, cn)] -= gm_c;
            }
        }
        if let Some(on) = out_neg {
            if let Some(cp) = ctrl_pos {
                self.matrix[(on, cp)] -= gm_c;
            }
            if let Some(cn) = ctrl_neg {
                self.matrix[(on, cn)] += gm_c;
            }
        }
    }
}

/// Callback for stamping the circuit at each AC frequency point.
pub trait AcStamper {
    /// Stamp all elements into the complex MNA system.
    ///
    /// Called once per frequency point. `omega` = 2πf.
    fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64);

    /// Number of nodes (excluding ground).
    fn num_nodes(&self) -> usize;

    /// Number of voltage source / inductor branch current variables.
    fn num_vsources(&self) -> usize;
}

/// Generate frequency points for an AC sweep.
pub fn generate_frequencies(params: &AcParams) -> Vec<f64> {
    match params.sweep_type {
        AcSweepType::Linear => {
            if params.num_points <= 1 {
                return vec![params.fstart];
            }
            let step = (params.fstop - params.fstart) / (params.num_points as f64 - 1.0);
            (0..params.num_points)
                .map(|i| params.fstart + step * i as f64)
                .collect()
        }
        AcSweepType::Decade => {
            let decades = (params.fstop / params.fstart).log10();
            let total_points = (params.num_points as f64 * decades).ceil() as usize + 1;
            (0..total_points)
                .map(|i| params.fstart * 10.0_f64.powf(i as f64 / params.num_points as f64))
                .filter(|&f| f <= params.fstop * 1.001)
                .collect()
        }
        AcSweepType::Octave => {
            let octaves = (params.fstop / params.fstart).log2();
            let total_points = (params.num_points as f64 * octaves).ceil() as usize + 1;
            (0..total_points)
                .map(|i| params.fstart * 2.0_f64.powf(i as f64 / params.num_points as f64))
                .filter(|&f| f <= params.fstop * 1.001)
                .collect()
        }
    }
}

/// A single frequency point in AC analysis.
#[derive(Debug, Clone)]
pub struct AcPoint {
    /// Frequency (Hz).
    pub frequency: f64,
    /// Complex solution vector (node voltages + branch currents).
    pub solution: DVector<Complex<f64>>,
}

/// Result of AC analysis.
#[derive(Debug, Clone)]
pub struct AcResult {
    /// All computed frequency points.
    pub points: Vec<AcPoint>,
    /// Number of nodes (excluding ground).
    pub num_nodes: usize,
}

impl AcResult {
    /// Get complex voltage at a node across all frequencies.
    ///
    /// `node_idx` is 0-based (node 1 → index 0).
    pub fn voltage_at(&self, node_idx: usize) -> Vec<(f64, Complex<f64>)> {
        self.points
            .iter()
            .map(|p| (p.frequency, p.solution[node_idx]))
            .collect()
    }

    /// Get voltage magnitude in dB at a node across all frequencies.
    pub fn magnitude_db(&self, node_idx: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|p| {
                let mag = p.solution[node_idx].norm();
                (p.frequency, 20.0 * mag.log10())
            })
            .collect()
    }

    /// Get voltage phase in degrees at a node across all frequencies.
    pub fn phase_deg(&self, node_idx: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|p| {
                let phase = p.solution[node_idx].arg() * 180.0 / PI;
                (p.frequency, phase)
            })
            .collect()
    }

    /// Get all frequency values.
    pub fn frequencies(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.frequency).collect()
    }
}

/// Run an AC small-signal analysis.
///
/// The stamper is called at each frequency point to build the complex MNA system.
/// Typically the stamper stamps:
/// - Resistors as real conductance
/// - Capacitors as jωC admittance
/// - Inductors with jωL impedance (using branch current variables)
/// - Voltage sources with AC stimulus value
/// - Linearized nonlinear devices (gm, gds from DC operating point)
pub fn solve_ac(stamper: &dyn AcStamper, params: &AcParams) -> Result<AcResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let frequencies = generate_frequencies(params);

    let mut result = AcResult {
        points: Vec::with_capacity(frequencies.len()),
        num_nodes,
    };

    for &freq in &frequencies {
        let omega = 2.0 * PI * freq;
        let mut mna = ComplexMna::new(num_nodes, num_vsources);

        stamper.stamp_ac(&mut mna, omega);

        let solution = solve_complex(mna.matrix(), mna.rhs())?;

        result.points.push(AcPoint {
            frequency: freq,
            solution,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_linear_frequencies() {
        let params = AcParams {
            fstart: 1.0,
            fstop: 100.0,
            num_points: 100,
            sweep_type: AcSweepType::Linear,
        };

        let freqs = generate_frequencies(&params);

        assert_eq!(freqs.len(), 100);
        assert!((freqs[0] - 1.0).abs() < 1e-10);
        assert!((freqs[99] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_decade_frequencies() {
        let params = AcParams {
            fstart: 1.0,
            fstop: 1000.0,
            num_points: 10,
            sweep_type: AcSweepType::Decade,
        };

        let freqs = generate_frequencies(&params);

        // 3 decades, 10 pts/decade → 31 points
        assert_eq!(freqs.len(), 31);
        assert!((freqs[0] - 1.0).abs() < 1e-10);
        // At index 10: 1.0 * 10^(10/10) = 10.0
        assert!(
            (freqs[10] - 10.0).abs() < 1e-6,
            "freq[10] = {} (expected 10.0)",
            freqs[10]
        );
        // At index 20: 1.0 * 10^(20/10) = 100.0
        assert!(
            (freqs[20] - 100.0).abs() < 1e-4,
            "freq[20] = {} (expected 100.0)",
            freqs[20]
        );
    }

    #[test]
    fn test_generate_octave_frequencies() {
        let params = AcParams {
            fstart: 100.0,
            fstop: 1600.0,
            num_points: 5,
            sweep_type: AcSweepType::Octave,
        };

        let freqs = generate_frequencies(&params);

        // 4 octaves (100→200→400→800→1600), 5 pts/octave → 21 points
        assert_eq!(freqs.len(), 21);
        assert!((freqs[0] - 100.0).abs() < 1e-10);
        // At index 5: 100 * 2^(5/5) = 200
        assert!(
            (freqs[5] - 200.0).abs() < 1e-6,
            "freq[5] = {} (expected 200.0)",
            freqs[5]
        );
    }

    /// RC low-pass filter AC stamper.
    ///
    /// Circuit: V1 (AC=1V) -- R -- node1 -- C -- GND
    /// Transfer function: H(f) = 1/(1 + j*2*pi*f*R*C)
    struct RcLowPassStamper {
        resistance: f64,
        capacitance: f64,
    }

    impl AcStamper for RcLowPassStamper {
        fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
            // Voltage source V1=1V at node 0 (AC stimulus)
            mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));

            // Resistor R from node 0 to node 1
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(0), Some(1), g);

            // Capacitor C from node 1 to ground: Y = jωC
            let yc = Complex::new(0.0, omega * self.capacitance);
            mna.stamp_admittance(Some(1), None, yc);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_rc_lowpass_3db_point() {
        // RC low-pass: R=1kΩ, C=1µF
        // f_3dB = 1/(2*pi*R*C) = 1/(2*pi*1000*1e-6) ≈ 159.15 Hz
        let r = 1000.0;
        let c = 1e-6;
        let f3db = 1.0 / (2.0 * PI * r * c);

        let stamper = RcLowPassStamper {
            resistance: r,
            capacitance: c,
        };

        let params = AcParams {
            fstart: 1.0,
            fstop: 1e6,
            num_points: 50,
            sweep_type: AcSweepType::Decade,
        };

        let result = solve_ac(&stamper, &params).unwrap();

        // Find the point closest to the -3dB frequency
        let mag_db = result.magnitude_db(1); // node 1 is output

        // At very low frequency: should be ≈ 0 dB
        let (f_low, db_low) = mag_db[0];
        assert!(
            db_low.abs() < 0.1,
            "At f={:.1} Hz, magnitude = {:.2} dB (expected ≈ 0 dB)",
            f_low,
            db_low
        );

        // Find point nearest to f_3dB and verify it's ≈ -3 dB
        let mut closest_idx = 0;
        let mut closest_dist = f64::INFINITY;
        for (i, &(f, _)) in mag_db.iter().enumerate() {
            let dist = ((f / f3db).ln()).abs();
            if dist < closest_dist {
                closest_dist = dist;
                closest_idx = i;
            }
        }
        let (f_at_3db, db_at_3db) = mag_db[closest_idx];
        assert!(
            (db_at_3db - (-3.01)).abs() < 1.0,
            "At f={:.1} Hz (≈ f_3dB={:.1}), magnitude = {:.2} dB (expected ≈ -3 dB)",
            f_at_3db,
            f3db,
            db_at_3db
        );

        // At high frequency: should show -20 dB/decade rolloff
        // Compare magnitude at two points a decade apart in the rolloff region
        let high_freq_points: Vec<_> = mag_db
            .iter()
            .filter(|&&(f, _)| f > f3db * 10.0)
            .collect();
        if high_freq_points.len() >= 2 {
            let &(f1, db1) = high_freq_points[0];
            // Find a point approximately one decade higher
            if let Some(&&(f2, db2)) = high_freq_points
                .iter()
                .find(|&&&(f, _)| (f / f1).log10() > 0.8)
            {
                let decades = (f2 / f1).log10();
                let db_per_decade = (db2 - db1) / decades;
                assert!(
                    (db_per_decade - (-20.0)).abs() < 3.0,
                    "Rolloff = {:.1} dB/decade (expected ≈ -20 dB/decade)",
                    db_per_decade
                );
            }
        }
    }

    #[test]
    fn test_rc_lowpass_phase() {
        let r = 1000.0;
        let c = 1e-6;
        let f3db = 1.0 / (2.0 * PI * r * c);

        let stamper = RcLowPassStamper {
            resistance: r,
            capacitance: c,
        };

        let params = AcParams {
            fstart: 1.0,
            fstop: 1e6,
            num_points: 50,
            sweep_type: AcSweepType::Decade,
        };

        let result = solve_ac(&stamper, &params).unwrap();
        let phase = result.phase_deg(1);

        // At low frequency: phase ≈ 0°
        let (_, phase_low) = phase[0];
        assert!(
            phase_low.abs() < 2.0,
            "Low-freq phase = {:.2}° (expected ≈ 0°)",
            phase_low
        );

        // At f_3dB: phase ≈ -45°
        let mut closest_idx = 0;
        let mut closest_dist = f64::INFINITY;
        for (i, &(f, _)) in phase.iter().enumerate() {
            let dist = ((f / f3db).ln()).abs();
            if dist < closest_dist {
                closest_dist = dist;
                closest_idx = i;
            }
        }
        let (_, phase_3db) = phase[closest_idx];
        assert!(
            (phase_3db - (-45.0)).abs() < 5.0,
            "Phase at f_3dB = {:.2}° (expected ≈ -45°)",
            phase_3db
        );

        // At high frequency: phase → -90°
        let (_, phase_high) = phase.last().unwrap();
        assert!(
            (phase_high - (-90.0)).abs() < 2.0,
            "High-freq phase = {:.2}° (expected ≈ -90°)",
            phase_high
        );
    }

    /// RL high-pass filter AC stamper.
    ///
    /// Circuit: V1 (AC=1V) -- node0 -- R -- GND
    ///                         node0 -- L -- node1 (output)
    /// Wait, a simpler high-pass: V1 -- C -- node1 -- R -- GND
    /// But we want to test inductors. Let's do:
    ///
    /// V1 (AC=1V) -- node0 -- L (branch current) -- node1 -- R -- GND
    /// Output at node1.
    /// H(f) = R / (R + jωL) = 1 / (1 + jωL/R)
    /// This is a low-pass with f_3dB = R/(2πL).
    struct RlLowPassStamper {
        resistance: f64,
        inductance: f64,
    }

    impl AcStamper for RlLowPassStamper {
        fn stamp_ac(&self, mna: &mut ComplexMna, omega: f64) {
            // Voltage source V1=1V at node 0
            mna.stamp_voltage_source(Some(0), None, 0, Complex::new(1.0, 0.0));

            // Inductor from node 0 to node 1 (branch_idx=1, since V1 uses 0)
            mna.stamp_inductor(Some(0), Some(1), 1, omega, self.inductance);

            // Resistor R from node 1 to ground
            let g = 1.0 / self.resistance;
            mna.stamp_conductance(Some(1), None, g);
        }

        fn num_nodes(&self) -> usize {
            2
        }

        fn num_vsources(&self) -> usize {
            2 // V1 + inductor branch
        }
    }

    #[test]
    fn test_rl_lowpass() {
        // RL low-pass: R=1kΩ, L=0.1H
        // f_3dB = R/(2πL) = 1000/(2π*0.1) ≈ 1591.5 Hz
        let r = 1000.0;
        let l = 0.1;
        let f3db = r / (2.0 * PI * l);

        let stamper = RlLowPassStamper {
            resistance: r,
            inductance: l,
        };

        let params = AcParams {
            fstart: 10.0,
            fstop: 1e6,
            num_points: 50,
            sweep_type: AcSweepType::Decade,
        };

        let result = solve_ac(&stamper, &params).unwrap();
        let mag_db = result.magnitude_db(1);

        // At low frequency: should be ≈ 0 dB
        let (f_low, db_low) = mag_db[0];
        assert!(
            db_low.abs() < 0.5,
            "At f={:.1} Hz, magnitude = {:.2} dB (expected ≈ 0 dB)",
            f_low,
            db_low
        );

        // Find point nearest to f_3dB
        let mut closest_idx = 0;
        let mut closest_dist = f64::INFINITY;
        for (i, &(f, _)) in mag_db.iter().enumerate() {
            let dist = ((f / f3db).ln()).abs();
            if dist < closest_dist {
                closest_dist = dist;
                closest_idx = i;
            }
        }
        let (f_near, db_near) = mag_db[closest_idx];
        assert!(
            (db_near - (-3.01)).abs() < 1.5,
            "At f={:.1} Hz (≈ f_3dB={:.1}), magnitude = {:.2} dB (expected ≈ -3 dB)",
            f_near,
            f3db,
            db_near
        );
    }

    #[test]
    fn test_complex_mna_admittance_stamp() {
        let mut mna = ComplexMna::new(2, 0);

        // Stamp admittance Y between nodes 0 and 1
        let y = Complex::new(1.0, 2.0);
        mna.stamp_admittance(Some(0), Some(1), y);

        assert_eq!(mna.matrix()[(0, 0)], y);
        assert_eq!(mna.matrix()[(1, 1)], y);
        assert_eq!(mna.matrix()[(0, 1)], -y);
        assert_eq!(mna.matrix()[(1, 0)], -y);
    }

    #[test]
    fn test_complex_mna_voltage_source_stamp() {
        let mut mna = ComplexMna::new(2, 1);

        let v = Complex::new(1.0, 0.5);
        mna.stamp_voltage_source(Some(0), Some(1), 0, v);

        let one = Complex::new(1.0, 0.0);

        // Check KCL stamps
        assert_eq!(mna.matrix()[(0, 2)], one);
        assert_eq!(mna.matrix()[(1, 2)], -one);
        // Check branch equation stamps
        assert_eq!(mna.matrix()[(2, 0)], one);
        assert_eq!(mna.matrix()[(2, 1)], -one);
        // Check RHS
        assert_eq!(mna.rhs()[2], v);
    }
}
