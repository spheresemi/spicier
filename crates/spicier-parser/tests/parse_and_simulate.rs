//! End-to-end tests: parse netlist, simulate, verify results.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::TransientDeviceInfo;
use spicier_core::NodeId;
use spicier_parser::{AnalysisCommand, parse, parse_full};
use spicier_solver::{
    CapacitorState, ConvergenceCriteria, DcSweepParams, DcSweepStamper, IntegrationMethod,
    NonlinearStamper, TransientParams, TransientStamper, solve_dc, solve_dc_sweep,
    solve_newton_raphson, solve_transient,
};

/// Parse and simulate a voltage divider.
#[test]
fn test_parse_simulate_voltage_divider() {
    let netlist_str = r#"
Voltage Divider Test
* 10V source with two 1k resistors
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.end
"#;

    // Parse
    let netlist = parse(netlist_str).expect("parse should succeed");
    assert_eq!(netlist.title(), Some("Voltage Divider Test"));
    assert_eq!(netlist.num_devices(), 3);

    // Simulate
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // Verify
    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));

    assert!((v1 - 10.0).abs() < 1e-9, "V(1) = {} (expected 10.0)", v1);
    assert!((v2 - 5.0).abs() < 1e-9, "V(2) = {} (expected 5.0)", v2);
}

/// Parse and simulate a current source with parallel resistors.
#[test]
fn test_parse_simulate_current_source() {
    let netlist_str = r#"
Current Source Test
I1 0 1 10m
R1 1 0 1k
R2 1 0 1k
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // V = I * R_parallel = 10mA * 500 = 5V
    let v1 = solution.voltage(NodeId::new(1));
    assert!((v1 - 5.0).abs() < 1e-9, "V(1) = {} (expected 5.0)", v1);
}

/// Parse and simulate a circuit with an inductor (DC short).
#[test]
fn test_parse_simulate_inductor() {
    let netlist_str = r#"
Inductor Test
* Inductor is short circuit at DC
V1 1 0 10
L1 1 2 1m
R1 2 0 100
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    assert_eq!(netlist.num_current_vars(), 2); // V1 + L1

    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // Inductor is short at DC, so V(2) = V(1) = 10V
    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));

    assert!((v1 - 10.0).abs() < 1e-9, "V(1) = {} (expected 10.0)", v1);
    assert!(
        (v2 - 10.0).abs() < 1e-9,
        "V(2) = {} (expected 10.0, inductor is DC short)",
        v2
    );

    // Current through inductor = V(2)/R1 = 10/100 = 0.1A
    let i_l1 = solution.current(1); // L1's current index
    assert!((i_l1 - 0.1).abs() < 1e-9, "I(L1) = {} (expected 0.1)", i_l1);
}

/// Parse a more complex multi-node circuit.
#[test]
fn test_parse_simulate_complex_network() {
    let netlist_str = r#"
Complex Network
V1 in 0 12
R1 in mid1 2k
R2 mid1 0 1k
R3 mid1 mid2 3k
R4 mid2 0 1k
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // Get node IDs (they're assigned dynamically for named nodes)
    // in=1, mid1=2, mid2=3 (based on order of appearance)
    let v_in = solution.voltage(NodeId::new(1));
    let v_mid1 = solution.voltage(NodeId::new(2));
    let v_mid2 = solution.voltage(NodeId::new(3));

    // V(in) should be 12V
    assert!(
        (v_in - 12.0).abs() < 1e-9,
        "V(in) = {} (expected 12.0)",
        v_in
    );

    // Analytical solution:
    // R2 || (R3 + R4) = 1k || 4k = 800 ohms
    // V(mid1) = 12 * 800 / 2800 = 3.4286V
    let r_parallel = 1000.0 * 4000.0 / 5000.0;
    let expected_mid1 = 12.0 * r_parallel / (2000.0 + r_parallel);
    assert!(
        (v_mid1 - expected_mid1).abs() < 1e-6,
        "V(mid1) = {} (expected {})",
        v_mid1,
        expected_mid1
    );

    // V(mid2) = V(mid1) * R4 / (R3 + R4) = V(mid1) * 1k / 4k
    let expected_mid2 = expected_mid1 * 1000.0 / 4000.0;
    assert!(
        (v_mid2 - expected_mid2).abs() < 1e-6,
        "V(mid2) = {} (expected {})",
        v_mid2,
        expected_mid2
    );

    println!("Complex network results:");
    println!("  V(in)   = {:.4} V", v_in);
    println!("  V(mid1) = {:.4} V", v_mid1);
    println!("  V(mid2) = {:.4} V", v_mid2);
}

/// Parse a netlist with .DC command and run DC sweep.
#[test]
fn test_parse_and_dc_sweep() {
    let netlist_str = r#"
DC Sweep Test
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.dc V1 0 5 1
.end
"#;

    let result = parse_full(netlist_str).expect("parse should succeed");
    let netlist = result.netlist;

    // Verify analysis command was parsed
    assert_eq!(result.analyses.len(), 1);
    match &result.analyses[0] {
        AnalysisCommand::Dc { sweeps } => {
            assert_eq!(sweeps.len(), 1);
            assert_eq!(sweeps[0].source_name, "V1");
            assert!((sweeps[0].start - 0.0).abs() < 1e-10);
            assert!((sweeps[0].stop - 5.0).abs() < 1e-10);
            assert!((sweeps[0].step - 1.0).abs() < 1e-10);
        }
        _ => panic!("Expected DC analysis command"),
    }

    // Run DC sweep using the parsed netlist
    struct SweepStamper<'a> {
        netlist: &'a spicier_core::Netlist,
    }

    impl DcSweepStamper for SweepStamper<'_> {
        fn stamp_with_sweep(&self, mna: &mut MnaSystem, _source_name: &str, value: f64) {
            self.netlist.stamp_into(mna);
            if let Some(idx) = self.netlist.find_vsource_branch_index("V1") {
                let bi = self.netlist.num_nodes() + idx;
                mna.rhs_mut()[bi] = value;
            }
        }
        fn num_nodes(&self) -> usize {
            self.netlist.num_nodes()
        }
        fn num_vsources(&self) -> usize {
            self.netlist.num_current_vars()
        }
    }

    let stamper = SweepStamper { netlist: &netlist };
    let params = DcSweepParams {
        source_name: "V1".to_string(),
        start: 0.0,
        stop: 5.0,
        step: 1.0,
    };

    let sweep_result = solve_dc_sweep(&stamper, &params).expect("DC sweep should succeed");

    assert_eq!(sweep_result.sweep_values.len(), 6); // 0, 1, 2, 3, 4, 5

    // V(2) = V1/2 at each point
    for (i, &sv) in sweep_result.sweep_values.iter().enumerate() {
        let v2 = sweep_result.solutions[i].voltage(NodeId::new(2));
        assert!(
            (v2 - sv / 2.0).abs() < 1e-9,
            "At V1={}, V(2)={} (expected {})",
            sv,
            v2,
            sv / 2.0
        );
    }
}

// ────────────────────── Nonlinear DC tests ──────────────────────

/// Test: Diode + R + V circuit converges with Newton-Raphson.
#[test]
fn test_diode_dc_operating_point() {
    let netlist_str = r#"
Diode DC Test
V1 1 0 DC 5
R1 1 2 1k
D1 2 0
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    assert!(netlist.has_nonlinear_devices());

    struct NlStamper<'a> {
        netlist: &'a spicier_core::Netlist,
    }
    impl NonlinearStamper for NlStamper<'_> {
        fn stamp_at(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
            self.netlist.stamp_nonlinear_into(mna, solution);
        }
    }

    let stamper = NlStamper { netlist: &netlist };
    let criteria = ConvergenceCriteria::default();
    let result = solve_newton_raphson(
        netlist.num_nodes(),
        netlist.num_current_vars(),
        &stamper,
        &criteria,
        None,
    )
    .expect("NR should succeed");

    assert!(result.converged, "Should converge");

    // V(1) = 5V (voltage source)
    assert!(
        (result.solution[0] - 5.0).abs() < 1e-6,
        "V(1) = {} (expected 5.0)",
        result.solution[0]
    );

    // V(2) should be roughly 0.6-0.8V (diode forward voltage)
    let vd = result.solution[1];
    assert!(
        vd > 0.5 && vd < 0.85,
        "V(diode) = {} (expected 0.5-0.85)",
        vd
    );
}

/// Test: Parsing D element with .MODEL
#[test]
fn test_parse_diode_with_model() {
    let netlist_str = r#"
Diode Model Test
.MODEL DMOD D (IS=1e-12 N=2)
V1 1 0 5
R1 1 2 1k
D1 2 0 DMOD
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    assert_eq!(netlist.num_devices(), 3); // V1, R1, D1
    assert!(netlist.has_nonlinear_devices());
}

/// Test: Parsing M element with .MODEL and W/L
#[test]
fn test_parse_mosfet_with_model() {
    let netlist_str = r#"
MOSFET Test
.MODEL NMOD NMOS (VTO=0.5 KP=1e-4 LAMBDA=0.02)
V1 1 0 5
VG 2 0 3
M1 1 2 0 0 NMOD W=20u L=1u
R1 1 0 10k
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    assert_eq!(netlist.num_devices(), 4); // V1, VG, M1, R1
    assert!(netlist.has_nonlinear_devices());
}

/// Test: Parsing controlled sources E/G
#[test]
fn test_parse_controlled_sources() {
    let netlist_str = r#"
Controlled Source Test
V1 1 0 10
R1 1 2 1k
R2 3 0 1k
E1 3 0 1 2 2.0
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    assert_eq!(netlist.num_devices(), 4); // V1, R1, R2, E1
    assert_eq!(netlist.num_current_vars(), 2); // V1 + E1

    // Simulate: E1 is a VCVS with gain 2, V(ctrl) = V(1)-V(2)
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));
    let v3 = solution.voltage(NodeId::new(3));

    // V(1) = 10V
    assert!(
        (v1 - 10.0).abs() < 1e-6,
        "V(1) = {} (expected 10.0)",
        v1
    );

    // V(3) = 2 * (V(1) - V(2))
    let expected_v3 = 2.0 * (v1 - v2);
    assert!(
        (v3 - expected_v3).abs() < 1e-6,
        "V(3) = {} (expected {})",
        v3,
        expected_v3
    );
}

/// Test: VCCS (G element) gain circuit
#[test]
fn test_vccs_circuit() {
    let netlist_str = r#"
VCCS Test
V1 1 0 2
R1 1 0 1k
R2 2 0 1k
G1 2 0 1 0 0.001
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // V(1) = 2V
    let v1 = solution.voltage(NodeId::new(1));
    assert!((v1 - 2.0).abs() < 1e-6);

    // G1 injects gm * V(1) = 0.001 * 2 = 2mA into node 2
    // V(2) = 2mA * 1k = 2V
    let v2 = solution.voltage(NodeId::new(2));
    assert!(
        (v2 - 2.0).abs() < 1e-6,
        "V(2) = {} (expected 2.0)",
        v2
    );
}

// ────────────────────── Transient tests ──────────────────────

/// Test: RC transient charging from the parser.
#[test]
fn test_transient_rc_from_netlist() {
    let netlist_str = r#"
RC Transient
V1 1 0 5
R1 1 2 1k
C1 2 0 1u
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");

    // Build transient state
    let mut caps = Vec::new();
    let mut inds = Vec::new();
    for device in netlist.devices() {
        match device.transient_info() {
            TransientDeviceInfo::Capacitor {
                node_pos,
                node_neg,
                capacitance,
            } => {
                caps.push(CapacitorState::new(capacitance, node_pos, node_neg));
            }
            TransientDeviceInfo::Inductor { .. } => {}
            TransientDeviceInfo::None => {}
        }
    }
    assert_eq!(caps.len(), 1);

    // Build transient stamper (stamps non-reactive devices)
    struct TranStamper<'a> {
        netlist: &'a spicier_core::Netlist,
    }
    impl TransientStamper for TranStamper<'_> {
        fn stamp_at_time(&self, mna: &mut MnaSystem, _time: f64) {
            for device in self.netlist.devices() {
                match device.transient_info() {
                    TransientDeviceInfo::Capacitor { .. } => {}
                    _ => device.stamp(mna),
                }
            }
        }
        fn num_nodes(&self) -> usize {
            self.netlist.num_nodes()
        }
        fn num_vsources(&self) -> usize {
            self.netlist.num_current_vars()
        }
    }

    let stamper = TranStamper { netlist: &netlist };

    // DC initial condition: V(1)=5, V(2)=5 (capacitor is open, so V(2)=V(1)),
    // I(V1) = 0 (no current at DC through R-C)
    let dc_solution = DVector::from_vec(vec![5.0, 5.0, 0.0]);

    let params = TransientParams {
        tstop: 5e-3,
        tstep: 10e-6,
        method: IntegrationMethod::Trapezoidal,
    };

    let result = solve_transient(&stamper, &mut caps, &mut inds, &params, &dc_solution)
        .expect("transient should succeed");

    // At steady state, V(2) = V(1) = 5V (already there from DC)
    let final_v2 = result.points.last().unwrap().solution[1];
    assert!(
        (final_v2 - 5.0).abs() < 0.1,
        "Final V(2) = {} (expected ≈ 5.0)",
        final_v2
    );
}

/// Test that capacitors are treated as open circuits at DC.
#[test]
fn test_parse_simulate_capacitor_dc() {
    let netlist_str = r#"
Capacitor DC Test
V1 1 0 10
R1 1 2 1k
C1 2 0 1u
R2 2 0 10k
.end
"#;

    let netlist = parse(netlist_str).expect("parse should succeed");
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // At DC, capacitor is open, so current only flows through R1-R2 divider
    // V(2) = 10 * 10k / (1k + 10k) = 10 * 10/11 = 9.09V
    let v1 = solution.voltage(NodeId::new(1));
    let v2 = solution.voltage(NodeId::new(2));

    assert!((v1 - 10.0).abs() < 1e-9);

    let expected_v2 = 10.0 * 10000.0 / 11000.0;
    assert!(
        (v2 - expected_v2).abs() < 1e-6,
        "V(2) = {} (expected {})",
        v2,
        expected_v2
    );
}

// ────────────────────── Subcircuit tests ──────────────────────

/// Test: Simple subcircuit with voltage divider
#[test]
fn test_subcircuit_voltage_divider() {
    let netlist_str = r#"
Subcircuit Test
* Define a voltage divider subcircuit
.SUBCKT VDIV in out
R1 in out 1k
R2 out 0 1k
.ENDS VDIV

* Main circuit
V1 1 0 DC 10
X1 1 2 VDIV
.end
"#;

    let result = parse_full(netlist_str).expect("parse should succeed");

    // Should have parsed the subcircuit definition
    assert!(
        result.subcircuits.contains_key("VDIV"),
        "VDIV subcircuit should be defined"
    );

    let subckt = &result.subcircuits["VDIV"];
    assert_eq!(subckt.ports.len(), 2);
    assert_eq!(subckt.elements.len(), 2); // R1 and R2

    // The netlist should have the expanded devices
    // V1, X1.R1, X1.R2
    assert_eq!(
        result.netlist.num_devices(),
        3,
        "Should have 3 devices (V1, X1.R1, X1.R2)"
    );

    // Simulate
    let mna = result.netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // V(1) = 10V (source node)
    let v1 = solution.voltage(NodeId::new(1));
    assert!((v1 - 10.0).abs() < 1e-6, "V(1) = {} (expected 10.0)", v1);

    // V(2) = V(out) = 5V (voltage divider output)
    let v2 = solution.voltage(NodeId::new(2));
    assert!((v2 - 5.0).abs() < 1e-6, "V(2) = {} (expected 5.0)", v2);
}

/// Test: Nested subcircuits
#[test]
fn test_nested_subcircuits() {
    let netlist_str = r#"
Nested Subcircuit Test
* Inner subcircuit: single resistor
.SUBCKT RES a b
R1 a b 1k
.ENDS RES

* Outer subcircuit: two resistors in series using inner subcircuit
.SUBCKT TWORES in out
X1 in mid RES
X2 mid out RES
.ENDS TWORES

* Main circuit: source + two resistor pairs to ground
V1 1 0 DC 10
X1 1 2 TWORES
R3 2 0 2k
.end
"#;

    let result = parse_full(netlist_str).expect("parse should succeed");

    // Should have both subcircuit definitions
    assert!(result.subcircuits.contains_key("RES"));
    assert!(result.subcircuits.contains_key("TWORES"));

    // The netlist should have expanded all subcircuit instances
    // V1, X1.X1.R1, X1.X2.R1, R3
    assert!(
        result.netlist.num_devices() >= 4,
        "Should have at least 4 devices, got {}",
        result.netlist.num_devices()
    );

    // Simulate
    let mna = result.netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solve should succeed");

    // Total resistance: 2k (from TWORES) in series with 2k (R3 parallel path consideration)
    // V(1) = 10V
    let v1 = solution.voltage(NodeId::new(1));
    assert!((v1 - 10.0).abs() < 1e-6, "V(1) = {} (expected 10.0)", v1);

    // V(2) = 10 * 2k / (2k + 2k) = 5V (voltage divider: TWORES 2k on top, R3 2k on bottom)
    let v2 = solution.voltage(NodeId::new(2));
    assert!((v2 - 5.0).abs() < 1e-6, "V(2) = {} (expected 5.0)", v2);
}
