//! Integration tests for DC analysis.

use spicier_core::{Netlist, NodeId};
use spicier_devices::passive::Resistor;
use spicier_devices::sources::{CurrentSource, VoltageSource};
use spicier_solver::solve_dc;

/// Test a simple voltage divider circuit:
///
/// ```text
///        V1 = 10V
///          +
///          |
///        node1
///          |
///         R1 = 1k
///          |
///        node2
///          |
///         R2 = 1k
///          |
///         GND
/// ```
///
/// Expected: V(node1) = 10V, V(node2) = 5V
#[test]
fn test_voltage_divider_netlist() {
    let mut netlist = Netlist::with_title("Voltage Divider");

    let n1 = NodeId::new(1);
    let n2 = NodeId::new(2);
    let gnd = NodeId::GROUND;

    netlist.register_node(n1);
    netlist.register_node(n2);

    // V1: 10V source from node1 to ground
    let v1 = VoltageSource::new("V1", n1, gnd, 10.0, netlist.next_current_index());
    netlist.add_device(v1);

    // R1: 1k between node1 and node2
    let r1 = Resistor::new("R1", n1, n2, 1000.0);
    netlist.add_device(r1);

    // R2: 1k between node2 and ground
    let r2 = Resistor::new("R2", n2, gnd, 1000.0);
    netlist.add_device(r2);

    // Assemble and solve
    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solution should succeed");

    // Check voltages
    assert!(
        (solution.voltage(n1) - 10.0).abs() < 1e-10,
        "V(node1) = {} (expected 10.0)",
        solution.voltage(n1)
    );
    assert!(
        (solution.voltage(n2) - 5.0).abs() < 1e-10,
        "V(node2) = {} (expected 5.0)",
        solution.voltage(n2)
    );
    assert_eq!(solution.voltage(gnd), 0.0);

    // Check current through V1 (should be -5mA, negative = into source)
    let i_v1 = solution.current(0);
    assert!(
        (i_v1 + 0.005).abs() < 1e-10,
        "I(V1) = {} (expected -0.005)",
        i_v1
    );
}

/// Test a current divider circuit:
///
/// ```text
///     I1 = 10mA
///        |
///        v
///      node1 ---+--- R1 = 1k ---+--- GND
///               |               |
///               +--- R2 = 1k ---+
/// ```
///
/// Expected: V(node1) = 5V (parallel resistance = 500 ohms)
#[test]
fn test_current_divider_netlist() {
    let mut netlist = Netlist::with_title("Current Divider");

    let n1 = NodeId::new(1);
    let gnd = NodeId::GROUND;

    netlist.register_node(n1);

    // I1: 10mA into node1
    let i1 = CurrentSource::new("I1", gnd, n1, 0.010);
    netlist.add_device(i1);

    // R1: 1k between node1 and ground
    let r1 = Resistor::new("R1", n1, gnd, 1000.0);
    netlist.add_device(r1);

    // R2: 1k between node1 and ground (parallel with R1)
    let r2 = Resistor::new("R2", n1, gnd, 1000.0);
    netlist.add_device(r2);

    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solution should succeed");

    // V = I * R_parallel = 0.01 * 500 = 5V
    assert!(
        (solution.voltage(n1) - 5.0).abs() < 1e-10,
        "V(node1) = {} (expected 5.0)",
        solution.voltage(n1)
    );
}

/// Test a more complex resistor network:
///
/// ```text
///        V1 = 12V
///          +
///          |
///        node1
///          |
///         R1 = 2k
///          |
///        node2 ------- R3 = 3k ------- node3
///          |                             |
///         R2 = 1k                       R4 = 1k
///          |                             |
///         GND                           GND
/// ```
#[test]
fn test_resistor_network() {
    let mut netlist = Netlist::with_title("Resistor Network");

    let n1 = NodeId::new(1);
    let n2 = NodeId::new(2);
    let n3 = NodeId::new(3);
    let gnd = NodeId::GROUND;

    netlist.register_node(n1);
    netlist.register_node(n2);
    netlist.register_node(n3);

    // V1: 12V
    let v1 = VoltageSource::new("V1", n1, gnd, 12.0, netlist.next_current_index());
    netlist.add_device(v1);

    // R1: 2k (node1 to node2)
    netlist.add_device(Resistor::new("R1", n1, n2, 2000.0));

    // R2: 1k (node2 to ground)
    netlist.add_device(Resistor::new("R2", n2, gnd, 1000.0));

    // R3: 3k (node2 to node3)
    netlist.add_device(Resistor::new("R3", n2, n3, 3000.0));

    // R4: 1k (node3 to ground)
    netlist.add_device(Resistor::new("R4", n3, gnd, 1000.0));

    let mna = netlist.assemble_mna();
    let solution = solve_dc(&mna).expect("DC solution should succeed");

    // Verify V(node1) = 12V (directly connected to voltage source)
    assert!(
        (solution.voltage(n1) - 12.0).abs() < 1e-10,
        "V(node1) = {} (expected 12.0)",
        solution.voltage(n1)
    );

    // Calculate expected voltages analytically:
    // Looking from node2: R2 in parallel with (R3 + R4) = 1k || 4k = 800 ohms
    // Voltage divider: V(node2) = 12 * 800 / (2000 + 800) = 12 * 800/2800 = 3.4286V
    let r_parallel = 1000.0 * 4000.0 / (1000.0 + 4000.0); // 800 ohms
    let v_node2_expected = 12.0 * r_parallel / (2000.0 + r_parallel);

    assert!(
        (solution.voltage(n2) - v_node2_expected).abs() < 1e-9,
        "V(node2) = {} (expected {})",
        solution.voltage(n2),
        v_node2_expected
    );

    // V(node3) = V(node2) * R4 / (R3 + R4) = V(node2) * 1k / 4k
    let v_node3_expected = v_node2_expected * 1000.0 / 4000.0;

    assert!(
        (solution.voltage(n3) - v_node3_expected).abs() < 1e-9,
        "V(node3) = {} (expected {})",
        solution.voltage(n3),
        v_node3_expected
    );

    println!("Resistor network solution:");
    println!("  V(node1) = {:.4}V", solution.voltage(n1));
    println!("  V(node2) = {:.4}V", solution.voltage(n2));
    println!("  V(node3) = {:.4}V", solution.voltage(n3));
}

/// Test that the netlist correctly counts current variables.
#[test]
fn test_current_variable_counting() {
    let mut netlist = Netlist::new();

    netlist.register_node(NodeId::new(1));
    netlist.register_node(NodeId::new(2));

    // Add a voltage source (requires 1 current variable)
    let v1 = VoltageSource::new("V1", NodeId::new(1), NodeId::GROUND, 5.0, 0);
    netlist.add_device(v1);

    assert_eq!(netlist.num_current_vars(), 1);

    // Add a resistor (no current variable)
    netlist.add_device(Resistor::new("R1", NodeId::new(1), NodeId::new(2), 1000.0));

    assert_eq!(netlist.num_current_vars(), 1);

    // MNA system should have correct dimensions
    let mna = netlist.assemble_mna();
    assert_eq!(mna.num_nodes, 2);
    assert_eq!(mna.num_vsources, 1);
    assert_eq!(mna.size(), 3);
}
