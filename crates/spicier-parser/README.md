# spicier-parser

SPICE netlist parser for the Spicier circuit simulator.

## Supported Elements

| Element | Syntax | Description |
|---------|--------|-------------|
| R | `R1 n1 n2 value` | Resistor |
| C | `C1 n1 n2 value` | Capacitor |
| L | `L1 n1 n2 value` | Inductor |
| V | `V1 n+ n- [DC val] [AC mag]` | Voltage source |
| I | `I1 n+ n- [DC val]` | Current source |
| D | `D1 anode cathode [model]` | Diode |
| M | `M1 d g s b model [W=w L=l]` | MOSFET |
| J | `J1 d g s [model]` | JFET |
| Q | `Q1 c b e [model]` | BJT |
| K | `K1 L1 L2 coupling` | Mutual inductance |
| T | `T1 p1+ p1- p2+ p2- Z0=val TD=val` | Transmission line |
| E | `E1 n+ n- nc+ nc- gain` | VCVS |
| G | `G1 n+ n- nc+ nc- gm` | VCCS |
| F | `F1 n+ n- Vsource gain` | CCCS |
| H | `H1 n+ n- Vsource rm` | CCVS |
| B | `B1 n+ n- V=expr` or `I=expr` | Behavioral |
| X | `X1 n1 n2 ... subckt` | Subcircuit instance |

## Commands

- `.OP` - DC operating point
- `.DC source start stop step` - DC sweep (also supports `.DC PARAM name start stop step`)
- `.AC type npts fstart fstop` - AC analysis
- `.TRAN tstep tstop [tstart] [UIC]` - Transient
- `.NOISE V(out) Vin type npts fstart fstop` - Noise analysis
- `.PRINT type var1 var2 ...` - Output selection
- `.MEAS type name ...` - Post-simulation measurements
- `.IC V(node)=value` - Initial conditions
- `.PARAM name=value` - Parameter definitions
- `.MODEL name type (params)` - Device models
- `.SUBCKT name ports [PARAMS: ...]` / `.ENDS` - Subcircuit definition

## Usage

```rust
use spicier_parser::parse_full;

let netlist = r#"
Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.OP
.END
"#;

let result = parse_full(netlist)?;
println!("Nodes: {}", result.netlist.num_nodes());
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
