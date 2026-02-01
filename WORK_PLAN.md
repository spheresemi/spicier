# Work Plan

## Overview

Phased roadmap for building Spicier, a high-performance SPICE circuit simulator in Rust. Each phase builds upon the previous, with clear acceptance criteria.

---

## Phase 0: Project Bootstrap

**Goal:** Establish project structure and development infrastructure.

### Tasks

- [x] Initialize Cargo workspace
- [x] Define crate structure:
  - `spicier-core` - Circuit graph, MNA matrices, solution types
  - `spicier-solver` - Linear/nonlinear solvers, Newton-Raphson
  - `spicier-devices` - Device models and stamps
  - `spicier-parser` - SPICE netlist parsing
  - `spicier-cli` - Command-line interface
- [x] Set up GitHub Actions CI
  - Build on Linux, macOS, Windows
  - Run tests and clippy
- [x] Configure testing infrastructure
- [x] Set up benchmarking with criterion

**Dependencies:** None

**Acceptance Criteria:** `cargo build` and `cargo test` pass on all crates. ✅

---

## Phase 1: Core Data Structures

**Goal:** Define fundamental circuit representation types.

### Tasks

- [x] Circuit graph representation
  - Node struct with unique IDs
  - Branch struct connecting nodes
  - Element trait for circuit components
- [x] Node numbering system
  - Ground node (node 0) handling
  - Automatic node renumbering
- [x] MNA matrix structure
  - Dense matrix (nalgebra DMatrix) - sparse to come later
  - Matrix builder interface with stamping methods
- [x] Solution vector types
  - Node voltages
  - Branch currents

**Dependencies:** Phase 0

**Acceptance Criteria:** Can represent a simple RLC circuit in memory and access node/branch data. ✅

---

## Phase 2: Netlist Parsing

**Goal:** Parse SPICE netlists into circuit representation.

### Tasks

- [x] SPICE netlist lexer
  - Tokenize element lines, commands, comments
  - Handle continuations (+)
- [x] Parser for basic elements
  - R (resistor)
  - C (capacitor)
  - L (inductor)
  - V (voltage source)
  - I (current source)
- [x] Circuit builder from parsed AST
- [x] Error reporting
  - Line numbers
  - Descriptive messages

**Dependencies:** Phase 1

**Acceptance Criteria:** Parse a `.sp` file and build a valid circuit graph. ✅

---

## Phase 3: Linear Passive Devices

**Goal:** Implement device models and MNA stamps for passive elements.

### Tasks

- [x] Resistor
  - Model: R = V/I
  - MNA stamp: conductance G = 1/R
- [x] Capacitor
  - Model: I = C * dV/dt
  - DC stamp: open circuit
- [x] Inductor
  - Model: V = L * dI/dt
  - DC stamp: short circuit (current variable)
- [x] Independent voltage source
  - MNA stamp with current variable
- [x] Independent current source
  - Direct RHS contribution

**Dependencies:** Phase 1

**Acceptance Criteria:** Stamp matrices correctly assembled for RC, RL, and RLC circuits. Unit tests verify stamp values. ✅

---

## Phase 4: DC Analysis

**Goal:** Solve DC operating point and DC sweeps.

### Tasks

- [x] MNA matrix assembly
  - Iterate circuit elements
  - Stamp into matrix
  - Netlist type with automatic assembly
- [x] Linear solver integration
  - Dense solver first (for correctness)
  - [ ] Sparse solver (e.g., faer, sprs, or nalgebra-sparse)
- [x] DC operating point (.OP)
  - Assemble and solve
  - Extract node voltages
  - DcSolution type with voltage/current accessors
- [x] DC sweep (.DC)
  - DcSweepStamper trait and solve_dc_sweep()
  - DcSweepResult with voltage/current waveform accessors
  - CLI output with tabular sweep results
  - [ ] Multiple sweep variables
- [ ] Output infrastructure
  - .PRINT support
  - Node voltage queries

**Dependencies:** Phase 2, Phase 3

**Acceptance Criteria:** Correctly solve voltage divider, current divider, and Thevenin equivalent circuits. Results match hand calculations. ✅ (integration tests pass)

---

## Phase 5: Nonlinear Devices & Newton-Raphson

**Goal:** Handle nonlinear devices with iterative solution.

### Tasks

- [x] Diode model
  - Shockley equation: I = Is * (exp(V/Vt) - 1)
  - Linearized conductance and current
  - Configurable model parameters (Is, n, Rs, Cj0, Vj, Bv)
- [x] Newton-Raphson iteration loop
  - Evaluate devices at current solution
  - Assemble Jacobian
  - Solve for update
  - Check convergence
- [x] Jacobian assembly
  - NonlinearStamper trait for device-by-device contribution
- [x] Convergence criteria
  - Voltage tolerance (absolute + relative)
  - Current tolerance
  - Iteration limits
- [x] Convergence aids
  - Voltage limiting (critical voltage)
  - Gmin shunt for initial guess
  - [ ] Source stepping
  - [ ] Gmin stepping
- [x] MOSFET Level 1 model
  - Cutoff, linear, saturation regions
  - Threshold voltage, transconductance
  - NMOS/PMOS support
  - Output conductance (gds), transconductance (gm)
  - Channel-length modulation (lambda)
  - Nonlinear stamp method for Newton-Raphson

**Dependencies:** Phase 4

**Acceptance Criteria:** Diode I-V curve matches theory. MOSFET regions verified analytically. ✅

---

## Phase 6: Transient Analysis

**Goal:** Time-domain simulation of dynamic circuits.

### Tasks

- [x] Integration methods
  - Backward Euler (BE)
  - Trapezoidal (TR)
  - [ ] Method switching (TR-BDF2 optional)
- [x] Companion models
  - Capacitor: Geq + Ieq (BE and Trap)
  - Inductor: Geq + Ieq (BE and Trap)
- [ ] Timestep control
  - Local Truncation Error (LTE) estimation
  - Adaptive step sizing
- [ ] Initial conditions (.IC)
  - Node voltage specification
  - UIC option
- [x] Transient simulation (.TRAN)
  - TransientStamper trait for per-step circuit assembly
  - Fixed timestep simulation loop
  - TransientResult with waveform extraction
  - [ ] Output at specified times (interpolation)

**Dependencies:** Phase 5

**Acceptance Criteria:** RC charge/discharge matches exponential. ✅ LC circuit oscillates at correct frequency.

---

## Phase 7: AC Analysis

**Goal:** Small-signal frequency-domain analysis.

### Tasks

- [x] Complex number arithmetic
  - ComplexMna system (complex DMatrix + DVector)
  - Complex linear solver (LU decomposition)
- [ ] Linearization at DC operating point
  - Compute small-signal parameters
- [x] Small-signal device models
  - Resistor: real conductance stamp
  - Capacitor: jωC admittance stamp
  - Inductor: jωL impedance with branch current
  - VCCS stamp for gm (transconductance)
  - [ ] Nonlinear devices: automatic linearization from DC point
- [x] AC sweep (.AC)
  - Linear, decade, octave sweep types
  - Frequency vector generation
  - AcResult with magnitude_db/phase_deg accessors
  - AcStamper trait for per-frequency stamping

**Dependencies:** Phase 5 (needs linearized operating point)

**Acceptance Criteria:** RC low-pass filter shows correct -3dB point and -20dB/decade rolloff. ✅

---

## Phase 8: Performance - SIMD & Parallelism

**Goal:** Optimize performance with vectorization and parallelism.

### Tasks

- [ ] SIMD-friendly data layouts
  - Structure of Arrays (SoA) for devices
  - Aligned memory allocation
- [ ] Vectorized device evaluation
  - Batch evaluate same-type devices
  - Use portable SIMD or intrinsics
- [ ] Parallel matrix assembly
  - Per-device-type parallelism
  - Thread-safe stamping or reduction
- [ ] Batched parameter sweeps
  - Monte Carlo
  - Corner analysis
  - Parallel sweep execution

**Dependencies:** Phase 6, Phase 7 (core functionality complete)

**Acceptance Criteria:** Measurable speedup on multi-core systems. Benchmark suite shows improvement.

---

## Phase 9: GPU Backend (Experimental)

**Goal:** Explore GPU acceleration for large circuits.

### Tasks

- [ ] GPU sparse matrix operations
  - cuSPARSE or equivalent
  - Custom kernels if needed
- [ ] Device evaluation kernels
  - Parallel device model evaluation
  - Batched nonlinear solves
- [ ] Memory transfer optimization
  - Minimize CPU-GPU transfers
  - Pinned memory
- [ ] Backend abstraction
  - CUDA for NVIDIA
  - Metal for Apple
  - wgpu for portability

**Dependencies:** Phase 8

**Acceptance Criteria:** GPU backend functional. Benchmark shows speedup for circuits with >10k devices.

---

## Future Considerations

- Subcircuit/hierarchical netlists
- Behavioral sources (B elements)
- Controlled sources (E, F, G, H)
- Noise analysis
- Additional MOSFET models (BSIM)
- S-parameter analysis
