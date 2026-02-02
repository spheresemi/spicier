# Spicier

A high-performance SPICE circuit simulator written in Rust.

[![Crates.io](https://img.shields.io/crates/v/spicier.svg)](https://crates.io/crates/spicier)
[![Documentation](https://docs.rs/spicier/badge.svg)](https://docs.rs/spicier)
[![License](https://img.shields.io/crates/l/spicier.svg)](LICENSE)

## Features

- **Complete SPICE analysis suite**
  - DC operating point and DC sweep
  - AC small-signal frequency response
  - Transient time-domain simulation

- **Comprehensive device support**
  - Passive elements: R, L, C
  - Independent sources: V, I with DC and time-varying waveforms (PULSE, SIN, PWL)
  - Controlled sources: VCVS (E), VCCS (G), CCCS (F), CCVS (H)
  - Semiconductors: Diode, MOSFET Level 1
  - Behavioral sources: B elements with expressions

- **High performance**
  - Sparse matrix solvers with symbolic factorization caching
  - SIMD-accelerated numerical kernels (AVX2/AVX-512)
  - Optional GPU acceleration (CUDA, Metal/WebGPU)
  - GMRES iterative solver for large circuits

- **Standard SPICE compatibility**
  - Familiar netlist syntax
  - Subcircuit hierarchy (.SUBCKT/.ENDS)
  - Model definitions (.MODEL)
  - Analysis commands (.OP, .DC, .AC, .TRAN)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
spicier = "0.1"
```

For GPU acceleration:

```toml
[dependencies]
spicier = { version = "0.1", features = ["cuda"] }  # NVIDIA GPU
# or
spicier = { version = "0.1", features = ["metal"] } # macOS/WebGPU
```

## Quick Start

```rust
use spicier::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse a voltage divider circuit
    let netlist = r#"
Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.OP
.END
"#;

    let result = spicier::parse_full(netlist)?;
    println!("Parsed {} nodes", result.netlist.num_nodes());

    Ok(())
}
```

## Example: RC Low-Pass Filter AC Analysis

```rust
use spicier::prelude::*;

let netlist = r#"
RC Low-Pass Filter
V1 1 0 DC 0 AC 1
R1 1 2 1k
C1 2 0 1u
.AC DEC 10 1 100k
.END
"#;

let result = spicier::parse_full(netlist)?;
// Run AC analysis and examine frequency response
```

## Crate Structure

This crate re-exports functionality from several sub-crates:

| Crate | Description |
|-------|-------------|
| `spicier-core` | Circuit graph, MNA matrices |
| `spicier-solver` | DC, AC, transient solvers |
| `spicier-devices` | Device models and stamps |
| `spicier-parser` | Netlist lexer and parser |
| `spicier-simd` | SIMD-accelerated kernels |
| `spicier-backend-cpu` | CPU dense operators |
| `spicier-backend-cuda` | CUDA GPU operators (optional) |
| `spicier-backend-metal` | Metal/WebGPU operators (optional) |

## Documentation

- [API Documentation](https://docs.rs/spicier)
- [Examples](https://github.com/rwalters/spicier/tree/main/examples)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
