//! End-to-end tests: parse netlist, simulate, verify results.

use spicier_core::mna::MnaSystem;
use spicier_core::NodeId;
use spicier_parser::{AnalysisCommand, parse, parse_full};
use spicier_solver::{DcSweepParams, DcSweepStamper, solve_dc, solve_dc_sweep};

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
        AnalysisCommand::Dc {
            source_name,
            start,
            stop,
            step,
        } => {
            assert_eq!(source_name, "V1");
            assert!((start - 0.0).abs() < 1e-10);
            assert!((stop - 5.0).abs() < 1e-10);
            assert!((step - 1.0).abs() < 1e-10);
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
