# spicier-devices

Device models and MNA stamps for the Spicier circuit simulator.

## Supported Devices

- **Passive elements** - Resistor, Capacitor, Inductor
- **Independent sources** - Voltage source, Current source
- **Controlled sources** - VCVS (E), VCCS (G), CCCS (F), CCVS (H)
- **Semiconductors** - Diode (Shockley), MOSFET Level 1 (NMOS/PMOS), JFET, BJT
- **Coupled elements** - Mutual inductance (K), Transmission line (T)
- **Behavioral** - B elements with expression-based V/I

## Waveforms

Time-varying source waveforms for transient analysis:
- **PULSE** - Periodic pulse with rise/fall times
- **SIN** - Sinusoidal with optional damping
- **PWL** - Piecewise linear arbitrary waveform

## Usage

```rust
use spicier_core::NodeId;
use spicier_devices::{Resistor, Capacitor, VoltageSource, Diode, TransmissionLine};

let n1 = NodeId::new(1);
let n2 = NodeId::new(2);
let gnd = NodeId::GROUND;

let r1 = Resistor::new("R1", n1, n2, 1000.0);           // 1k resistor
let c1 = Capacitor::new("C1", n2, gnd, 1e-6);          // 1uF capacitor
let v1 = VoltageSource::new("V1", n1, gnd, 5.0, 0);    // 5V source
let d1 = Diode::new("D1", n2, gnd);                     // Diode with defaults
let t1 = TransmissionLine::new("T1", n1, gnd, n2, gnd, 50.0, 1e-9, 0); // 50 ohm, 1ns
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
