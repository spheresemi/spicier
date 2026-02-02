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
- [x] Parser for nonlinear elements
  - D (diode) with optional model reference
  - M (MOSFET) with W/L parameters and optional model reference
- [x] Parser for controlled sources
  - E (VCVS), G (VCCS), F (CCCS), H (CCVS)
- [x] .MODEL command parsing
  - Diode parameters (IS, N, RS, CJO, VJ, BV)
  - MOSFET parameters (VTO, KP, LAMBDA, COX, W, L) for NMOS/PMOS
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
  - [x] Sparse solver (faer 0.24 — sparse LU, real + complex, auto-selected at size >= 50)
- [x] DC operating point (.OP)
  - Assemble and solve
  - Extract node voltages
  - DcSolution type with voltage/current accessors
- [x] DC sweep (.DC)
  - DcSweepStamper trait and solve_dc_sweep()
  - DcSweepResult with voltage/current waveform accessors
  - CLI output with tabular sweep results
  - [x] Multiple sweep variables (nested .DC sweeps)
- [x] Output infrastructure
  - .PRINT support (DC, AC, TRAN analysis types)
  - V(node), I(device), VM, VP, VDB output variables

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
  - [x] Source stepping (`solve_with_source_stepping`)
  - [x] Gmin stepping (`solve_with_gmin_stepping`)
- [x] MOSFET Level 1 model
  - Cutoff, linear, saturation regions
  - Threshold voltage, transconductance
  - NMOS/PMOS support
  - Output conductance (gds), transconductance (gm)
  - Channel-length modulation (lambda)
  - Nonlinear stamp method for Newton-Raphson
- [x] Nonlinear DC in CLI
  - Stamper trait extended with is_nonlinear(), stamp_nonlinear()
  - NetlistNonlinearStamper wires Netlist to Newton-Raphson solver
  - Automatic NR dispatch when nonlinear devices present
  - Diode circuit converges: V(diode) ≈ 0.74V in 10 iterations

**Dependencies:** Phase 4

**Acceptance Criteria:** Diode I-V curve matches theory. MOSFET regions verified analytically. ✅

---

## Phase 6: Transient Analysis

**Goal:** Time-domain simulation of dynamic circuits.

### Tasks

- [x] Integration methods
  - Backward Euler (BE)
  - Trapezoidal (TR)
  - [x] Method switching (TR-BDF2 L-stable composite method)
- [x] Companion models
  - Capacitor: Geq + Ieq (BE and Trap)
  - Inductor: Geq + Ieq (BE and Trap)
- [x] Timestep control
  - Local Truncation Error (LTE) estimation
  - Adaptive step sizing (`solve_transient_adaptive`)
- [x] Initial conditions (.IC)
  - Node voltage specification (parser + solver integration)
  - [x] UIC option (skip DC operating point, use .IC directly)
- [x] Transient simulation (.TRAN)
  - TransientStamper trait for per-step circuit assembly
  - Fixed timestep simulation loop
  - TransientResult with waveform extraction
  - [x] Output at specified times (interpolation via `sample_at_times()`)
- [x] Transient CLI integration
  - Stamper trait extended with transient_info() returning TransientDeviceInfo
  - NetlistTransientStamper for per-step circuit assembly from Netlist
  - DC operating point as initial condition, then time-stepping
  - Tabular time-domain output (Time, V(1), V(2), ...)
- [x] Time-varying sources
  - PULSE waveform: v1, v2, td, tr, tf, pw, per
  - SIN waveform: vo, va, freq, td, theta, phase (damped sinusoid)
  - PWL waveform: arbitrary time-value pairs with linear interpolation
  - Stamper::stamp_at_time() for time-dependent device stamping
  - Parser support for PULSE(...), SIN(...), PWL(...) syntax

**Dependencies:** Phase 5

**Acceptance Criteria:** RC charge/discharge matches exponential. ✅ LC circuit oscillates at correct frequency.

---

## Phase 7: AC Analysis

**Goal:** Small-signal frequency-domain analysis.

### Tasks

- [x] Complex number arithmetic
  - ComplexMna system (complex DMatrix + DVector)
  - Complex linear solver (LU decomposition)
- [x] Linearization at DC operating point
  - Compute small-signal parameters (via `ac_info_at()` method)
- [x] Small-signal device models
  - Resistor: real conductance stamp
  - Capacitor: jωC admittance stamp
  - Inductor: jωL impedance with branch current
  - VCCS stamp for gm (transconductance)
  - [x] Controlled source AC stamps (VCVS, VCCS, CCCS, CCVS)
  - [x] Nonlinear devices: automatic linearization from DC point
    - `ac_info_at(solution)` method computes small-signal params from DC
    - Diode: gd conductance at operating point
    - MOSFET: gds (output conductance) + gm (transconductance) VCCS
    - AC analysis automatically runs DC OP first when nonlinear devices present
- [x] AC sweep (.AC)
  - Linear, decade, octave sweep types
  - Frequency vector generation
  - AcResult with magnitude_db/phase_deg accessors
  - AcStamper trait for per-frequency stamping

**Dependencies:** Phase 5 (needs linearized operating point)

**Acceptance Criteria:** RC low-pass filter shows correct -3dB point and -20dB/decade rolloff. ✅

---

## Phase 7b: Controlled Sources ✅

**Goal:** Implement voltage- and current-controlled dependent sources.

### Tasks

- [x] VCVS (E element) — V(out) = gain * V(ctrl), branch current variable
- [x] VCCS (G element) — I(out) = gm * V(ctrl), direct matrix stamps
- [x] CCCS (F element) — I(out) = gain * I(Vsource), references Vsource branch
- [x] CCVS (H element) — V(out) = gain * I(Vsource), branch current variable
- [x] All implement Stamp, Element, Stamper traits with correct MNA stamps
- [x] AcDeviceInfo variants for AC analysis
- [x] Parser support for E/G/F/H element lines
- [x] Unit tests for all four stamp patterns

**Dependencies:** Phase 3

**Acceptance Criteria:** Controlled source stamps verified against analytical MNA entries. ✅

---

## Phase 7c: Sparse Solver Integration ✅ (Foundation)

**Goal:** Add sparse LU solver alongside dense, with automatic selection based on system size.

### Completed Tasks

- [x] faer 0.24 dependency (pure Rust, no external C/Fortran libraries)
- [x] Triplet accumulation in MnaSystem (`add_element()`, `add_rhs()`)
- [x] Triplet accumulation in ComplexMna (`add_element()`, `add_rhs()`)
- [x] `solve_sparse()` and `solve_sparse_complex()` in linear.rs
- [x] All device stamping migrated to `add_element()` (controlled sources, MOSFET, passives)
- [x] Auto-selection in all analysis paths (DC, NR, transient, AC) at SPARSE_THRESHOLD=50
- [x] 5 new tests, updated benchmarks (dense vs sparse at 10/50/100/500)

### Remaining Optimizations

- [x] Symbolic factorization caching
  - For NR iterations and transient timesteps, the sparsity pattern is fixed
  - `CachedSparseLu` and `CachedSparseLuComplex` cache `SymbolicLu` and only redo numeric factorization
  - Integrated into Newton-Raphson, transient, and AC analysis paths
  - Major speedup for repeated solves with the same structure
- [x] Sparse-only MnaSystem
  - Removed `matrix: DMatrix<f64>` field from MnaSystem
  - `add_element()` now only pushes to triplets (duplicates summed during construction)
  - New `to_dense_matrix()` method builds dense matrix on demand for tests and small circuits
  - Reduces memory from O(n²) to O(nnz) for large circuits
- [x] Sparse-only ComplexMna
  - Removed `matrix: DMatrix<Complex<f64>>` field from ComplexMna
  - Same sparse-only pattern as real MnaSystem
  - New `to_dense_matrix()` method for on-demand dense construction

### GMRES Integration (leverages Phase 8 infrastructure)

- [x] `SparseRealOperator` wrapper
  - Wraps faer `SparseColMat<f64>` and implements `RealOperator`
  - CSC matvec for y = A * x
  - Enables GMRES as alternative to direct LU for real systems (DC, transient)
- [x] `SparseComplexOperator` wrapper
  - Wraps faer `SparseColMat<c64>` and implements `ComplexOperator`
  - CSC matvec for y = A * x
  - Enables GMRES as alternative for complex systems (AC)
- [x] Real-valued GMRES (`solve_gmres_real`)
  - Ported complex GMRES to work with `RealOperator`
  - Uses `real_dot_product` SIMD kernels from `spicier-simd`
  - `RealGmresResult` return type with solution, iterations, residual, converged flag
  - Real Givens rotations for stability
- [x] Solver selection heuristic
  - `SolverStrategy` enum: `Auto`, `DirectLU`, `IterativeGmres`
  - `SolverConfig` with configurable threshold (default 10k nodes)
  - `solve_auto()` function auto-selects based on system size
  - Direct LU for small/medium systems, GMRES for >10k nodes
  - GMRES fallback to LU on non-convergence
  - `from_name()` for CLI integration
- [x] Preconditioned GMRES
  - Jacobi (diagonal) preconditioner for faster convergence
  - `RealPreconditioner` and `ComplexPreconditioner` traits
  - `solve_gmres_real_preconditioned()` and `solve_gmres_preconditioned()`
  - Right preconditioning: solves A*M^(-1)*y = b, then x = M^(-1)*y

**Dependencies:** Phase 7 (AC analysis uses ComplexMna), Phase 8 (operator traits, SIMD)

**Acceptance Criteria (remaining):** NR and transient solves reuse symbolic factorization ✅. MnaSystem memory scales with nnz, not n² ✅. GMRES available as alternative solver for large systems.

---

## Phase 8: Performance - SIMD & Parallelism

**Goal:** Optimize performance with vectorization and parallelism.

**Status:** Infrastructure ported from `mom` project. SIMD detection and kernels in place.

### Completed Tasks

- [x] SIMD runtime detection (`spicier-simd` crate)
  - `SimdCapability::detect()` — AVX-512, AVX2, or scalar fallback
  - Runtime dispatch based on CPU features
- [x] SIMD dot products
  - Real f64 dot product with AVX-512/AVX2/scalar paths
  - Complex C64 dot product with SIMD acceleration
  - Conjugate dot product for Hermitian operations
- [x] SIMD matrix-vector multiplication
  - Real and complex matvec kernels
  - Used by CPU dense operators
- [x] Operator traits (`spicier-solver`)
  - `RealOperator` trait for f64 matvec
  - `ComplexOperator` trait for C64 matvec
  - Both are `Send + Sync` for parallel use
- [x] CPU dense operators (`spicier-backend-cpu` crate)
  - `CpuRealDenseOperator` — f64 dense matvec using SIMD
  - `CpuComplexDenseOperator` — C64 dense matvec using SIMD
  - Implement `RealOperator`/`ComplexOperator` traits

### Remaining Tasks

- [x] SIMD-friendly data layouts
  - `DiodeBatch` and `MosfetBatch` with SoA layout
  - Padding to SIMD lane count (4 for AVX2)
  - AVX2 batch evaluation for diodes
- [x] Vectorized device evaluation
  - `BatchedNonlinearDevices` container for batched diodes/MOSFETs
  - `solve_batched_newton_raphson()` uses batched evaluation
  - Pre-allocated buffers, separate eval/stamp loops
- [x] Parallel matrix assembly
  - `ParallelTripletAccumulator` for thread-local buffer management
  - `evaluate_parallel()` splits device stamping across threads
  - Thread-safe triplet accumulation with final merge
- [x] Batched parameter sweeps
  - `MonteCarloGenerator` for random sampling
  - `CornerGenerator` for 2^n worst-case combinations
  - `LinearSweepGenerator` for systematic sweeps
  - `solve_batched_sweep()` runs multiple simulations

**Dependencies:** Phase 6, Phase 7 (core functionality complete)

**Acceptance Criteria:** Measurable speedup on multi-core systems. Benchmark suite shows improvement.

---

## Phase 9: Compute Backend Abstraction & GPU Acceleration

**Goal:** Abstract compute dispatch behind a backend trait with automatic hardware detection, enabling GPU acceleration for sweeps, device evaluation, and large-circuit solves while maintaining a robust CPU fallback.

**Status:** Infrastructure ported from `mom` project. Backend crates created with dense operators; sparse GPU operators and solver integration remain.

### 9a: ComputeBackend Enum & Auto-Detection ✅

- [x] `ComputeBackend` enum (`spicier-solver/src/backend.rs`)
  - Variants: `Cpu`, `Cuda { device_id }`, `Metal { adapter_name }`
  - `from_name()` for CLI parsing
  - `Display` impl for user-friendly output
- [x] CLI `--backend` flag (`spicier-cli`)
  - Options: `auto`, `cpu`, `cuda`, `metal`
  - Auto-detection: Metal first (macOS), then CUDA, then CPU
  - Graceful fallback with warnings when requested backend unavailable
  - Verbose mode prints selected backend
- [x] CUDA context (`spicier-backend-cuda` crate)
  - `CudaContext` with cudarc 0.16 dynamic loading
  - `is_available()` probe without hard dependency on CUDA runtime
  - cuBLAS handle for BLAS operations
- [x] Metal/WebGPU context (`spicier-backend-metal` crate)
  - `WgpuContext` with wgpu 23 (Metal/Vulkan/DX12 backends)
  - `is_available()` probe via adapter request
  - f64 shader support detection
- [x] Dense GPU operators (both crates)
  - `CudaRealDenseOperator` / `CudaComplexDenseOperator` — cuBLAS dgemv/zgemv
  - `WgpuRealDenseOperator` / `WgpuComplexDenseOperator` — WGSL compute shaders
  - CPU fallback threshold (small matrices stay on CPU)
  - Implement `RealOperator`/`ComplexOperator` traits
- [x] GMRES iterative solver (`spicier-solver/src/gmres.rs`)
  - Works with any `ComplexOperator` or `RealOperator`
  - Configurable restart, tolerance, max iterations
  - Givens rotations for stable least-squares

### Remaining Tasks

- [x] Wire GPU operators into analysis paths
  - `solve_ac_dispatched()` with GMRES + Jacobi preconditioner for large AC systems
  - `solve_dc_dispatched()` with GMRES + Jacobi preconditioner for large DC systems
  - `solve_transient_dispatched()` with GMRES for large transient timesteps
  - GPU operators available for GMRES preconditioning via `ComplexOperator`/`RealOperator` traits
- [x] Size-based dispatch heuristic
  - `DispatchConfig` with `cpu_threshold` (default 1k) and `gmres_threshold` (default 10k)
  - `use_gpu(size)` and `use_gmres(size)` decision methods
  - `SolverDispatchStrategy` enum: Auto, DirectLU, IterativeGmres
  - Thresholds tunable via builder pattern
- [ ] Shared memory management
  - Pinned/page-locked host memory for async transfers
  - Double-buffering for overlapping compute and transfer
  - Upload matrix structure once, stream value diffs per sweep point

### 9b: Batched Sweep Solver (Primary GPU Win)

Sweep analyses (DC sweep, AC sweep, Monte Carlo, corners) produce many matrices sharing the same sparsity pattern that differ in only a few entries. Batch all sweep points into a single GPU dispatch.

- [ ] Batched matrix preparation
  - Shared symbolic structure, per-sweep-point numeric values
  - Single symbolic factorization, batched numeric factorization + solve
  - Applies to DC sweep, AC sweep, Monte Carlo, and corner analysis
- [ ] GPU sparse batched solve
  - cuSPARSE batched sparse solver (CUDA)
  - Metal Performance Shaders / Accelerate batched solve (macOS)
  - Batched dense solver fallback for small circuits (cuBLAS/Accelerate batched LU)
- [ ] Monte Carlo / statistical analysis on GPU
  - GPU-side random number generation and parameter sampling
  - Histogram and statistics computation without CPU round-trip
  - Yield analysis with thousands of samples

### 9c: Batched Device Evaluation

A circuit with 100k MOSFETs evaluates I-V + derivatives at every NR iteration. Each device is independent — ideal for GPU data-parallel compute.

- [ ] Device evaluation kernels
  - Per-device-type GPU kernels (diode, MOSFET, etc.)
  - Uniform computation within a type, minimal branching
  - Evaluate I, dI/dV, and companion model values in one pass
- [ ] Cross-sweep batched NR
  - NR iterations per sweep point in parallel
  - Converged points retire early, active points continue
  - Shared Jacobian structure across sweep points

### 9d: Large-Circuit Sparse Solve

For circuits with 50k+ nodes, sparse LU factorization itself is the bottleneck.

- [ ] GPU-accelerated sparse direct solve
  - cuSPARSE / cuSOLVER for large sparse LU
  - Supernodal GPU factorization for dense subblocks
  - Reuse symbolic factorization from Phase 7c across NR iterations
- [ ] GPU-accelerated matrix assembly
  - Parallel stamping with atomic adds
  - All devices stamp concurrently, single kernel launch
  - Worth it for large circuits; CPU fallback for small ones

### 9e: Post-Processing & Analysis on GPU

- [ ] Transient waveform post-processing
  - FFT / spectral analysis on output waveforms (cuFFT / Accelerate vDSP)
  - THD computation, spectral density
  - Avoids large waveform transfer back to CPU for analysis
- [ ] Sensitivity analysis / adjoint method
  - dOutput/dParam for every parameter — each perturbation is independent
  - GPU-parallel forward or adjoint sensitivity solves
  - Useful for optimization and yield analysis
- [ ] Pole-zero analysis
  - Eigenvalue problems on the linearized system
  - GPU-accelerated eigensolvers (cuSOLVER)

**Dependencies:** Phase 7c (sparse-only MNA), Phase 8

**Acceptance Criteria:**
- Auto-detection correctly selects CUDA on Linux/Windows with NVIDIA GPU, Metal on macOS, CPU elsewhere
- `--backend=cpu` always works as fallback
- 1000-point DC sweep of a nonlinear circuit shows measurable speedup on GPU vs CPU
- Batched device evaluation of 100k MOSFETs faster on GPU than CPU
- All results match CPU reference to within solver tolerance

---

## Phase 10: Validation Test Suite

**Goal:** Import and adapt test circuits from ngspice and spice21 to validate solver accuracy against established SPICE implementations.

### 10a: ngspice Test Import

ngspice provides 113 regression tests with expected outputs and 462 example circuits.

**High-priority imports:**
- [ ] Basic circuit validation
  - `rc.cir`, `lowpass.cir` — RC filter validation
  - Resistor divider, current divider circuits
  - Simple amplifier circuits from `tests/general/`
- [ ] Parser validation
  - `tests/regression/parser/` — expression parsing edge cases
  - `tests/regression/misc/` — bug fixes and parser validation (13 tests)
- [ ] Transient validation
  - LC oscillator frequency verification
  - RC charge/discharge curves
  - PULSE/SIN/PWL source waveforms
- [ ] AC validation
  - Low-pass filter -3dB points and rolloff
  - High-pass filter response
  - Pole-zero analysis circuits

**Test infrastructure:**
- [ ] Create `tests/ngspice_compat/` directory
- [ ] Implement toleranced comparison (relative + absolute tolerance)
- [ ] Add `.out` baseline file support or JSON golden data
- [ ] CI integration for regression testing

### 10b: spice21 Test Adaptation

spice21 has 87 tests with golden data for ring oscillators and device characterization.

- [ ] Ring oscillator validation
  - 3-stage CMOS ring oscillator (transient frequency)
  - NMOS-R and PMOS-R variants
- [ ] Device characterization
  - Diode-connected MOSFET I-V curves
  - CMOS inverter DC transfer curve
- [ ] Hierarchical circuit validation
  - Module instantiation and port mapping

### 10c: Cross-Simulator Comparison

- [ ] Run identical circuits through ngspice and spicier
- [ ] Compare DC operating points (voltage/current tolerance)
- [ ] Compare transient waveforms (RMS error, peak detection)
- [ ] Compare AC magnitude/phase (dB tolerance, phase tolerance)
- [ ] Document any intentional deviations from ngspice behavior

**Dependencies:** Core analysis types complete (Phases 4-7)

**Acceptance Criteria:**
- 50+ ngspice test circuits pass with <1% error vs ngspice results
- Ring oscillator frequency within 5% of spice21 golden data
- All existing 297 tests continue to pass

---

## Phase 11: crates.io Release Preparation

**Goal:** Prepare spicier crates for publication on crates.io.

### 11a: API Stability Review

- [x] Review public API surface for each crate
  - `spicier-core`: Circuit, Node, MnaSystem, Netlist
  - `spicier-solver`: DC, AC, transient solvers, GMRES, operators
  - `spicier-devices`: Device models, stamps, waveforms
  - `spicier-parser`: Parser, AST types, parse functions
  - `spicier-simd`: SimdCapability, dot products, matvec
- [x] Mark internal APIs as `pub(crate)` where appropriate
- [x] Ensure `#[non_exhaustive]` on enums that may grow
- [x] Stabilize error types and Result patterns

### 11b: Documentation

- [x] Rustdoc for all public types and functions
- [x] Crate-level documentation with examples
- [x] README.md for each crate
- [x] Example programs in `examples/` directories
- [x] CHANGELOG.md with version history

### 11c: Metadata & Licensing

- [x] Finalize license (MIT OR Apache-2.0)
- [x] Add LICENSE-MIT and LICENSE-APACHE files to repository root
- [x] Complete Cargo.toml metadata for each crate:
  - `license`, `description`, `repository`, `keywords`, `categories`
  - `readme`, `documentation` links
- [x] Verify crate names available on crates.io

### 11d: Quality Checks

- [x] `cargo clippy` clean (all targets)
- [x] `cargo fmt` consistent
- [x] `cargo doc` builds without warnings
- [ ] `cargo publish --dry-run` succeeds for each crate
- [ ] Minimum Rust version (MSRV) documented and tested

### 11e: Release Strategy

- [x] Define version 0.1.0 scope
- [x] Determine crate publication order (dependencies first)
- [ ] Create GitHub release with changelog
- [ ] Announce on relevant forums (Reddit r/rust, etc.)

#### Publication Order

Crates must be published to crates.io in dependency order. Each level must be fully published before proceeding to the next:

```
Level 1 (no internal deps):
  cargo publish -p spicier-core
  cargo publish -p spicier-simd

Level 2 (depends on Level 1):
  cargo publish -p spicier-devices    # → core, simd

Level 3 (depends on Level 2):
  cargo publish -p spicier-solver     # → core, devices, simd

Level 4 (depends on Level 3):
  cargo publish -p spicier-parser     # → core, devices, solver
  cargo publish -p spicier-backend-cpu    # → simd, solver
  cargo publish -p spicier-backend-cuda   # → solver
  cargo publish -p spicier-backend-metal  # → solver

Level 5 (depends on Level 4):
  cargo publish -p spicier-batched-sweep  # → solver, backend-cuda (opt), backend-metal (opt)
  cargo publish -p spicier-cli            # → core, solver, devices, parser
  cargo publish -p spicier-validate       # → core, parser, solver, devices

Level 6 (umbrella crate - last):
  cargo publish -p spicier                # → all of the above
```

**Note:** After publishing each crate, wait for crates.io index to update before publishing dependents (~1-2 minutes).

**Dependencies:** Phase 10 (validation gives confidence for release)

**Acceptance Criteria:**
- All crates pass `cargo publish --dry-run`
- Documentation coverage >80% for public API
- License and metadata complete

---

## Phase 12: Extended Features

**Goal:** Add commonly-requested SPICE features for broader compatibility.

### 12a: .PARAM / Parameter Expressions

Parameter expressions enable parameterized circuits and design exploration.

- [ ] `.PARAM name = expression` parsing
- [ ] Parameter substitution in element values
- [ ] Parameter expressions with math operations
- [ ] Nested parameter references
- [ ] Global vs local parameter scoping
- [ ] Integration with DC sweep (sweep parameter values)

### 12b: .MEASURE Statements

Post-processing measurements extract key metrics from simulation results.

- [ ] `.MEAS TRAN` — transient measurements
  - `TRIG`/`TARG` for timing measurements
  - `FIND`/`WHEN` for value extraction
  - `AVG`, `RMS`, `MIN`, `MAX`, `PP` (peak-to-peak)
- [ ] `.MEAS DC` — DC sweep measurements
- [ ] `.MEAS AC` — AC analysis measurements
- [ ] Measurement result output and export

### 12c: Noise Analysis

Small-signal noise analysis for analog circuit design.

- [ ] Thermal noise (resistors): 4kTR
- [ ] Shot noise (diodes, BJTs): 2qI
- [ ] Flicker noise (MOSFETs): Kf/f
- [ ] Noise figure calculation
- [ ] Input/output referred noise
- [ ] Noise spectral density plots

### 12d: Additional Device Models

- [ ] BJT (Gummel-Poon model)
- [ ] JFET
- [ ] BSIM3v3 MOSFET
- [ ] BSIM4 MOSFET (partial — complex model)
- [ ] Transmission lines (lossless, lossy)

**Dependencies:** Core functionality complete

---

## Future Considerations

### Completed
- [x] Subcircuit/hierarchical netlists (.SUBCKT/.ENDS) — implemented with nested expansion
- [x] Behavioral sources (B elements) — expression AST with voltage/current/time refs, auto-diff for NR

### Planned (Phase 12)
- [ ] .PARAM / parameter expressions
- [ ] .MEASURE statements
- [ ] Noise analysis
- [ ] Additional MOSFET models (BSIM3/BSIM4)

### Long-term / Research
- S-parameter analysis
- Mixed-signal / event-driven simulation (XSPICE-style)
- Physics-based device models (CIDER-style)
- Optimization / tuning (.OPTIM)
- Verilog-A model import
- Distributed simulation for very large circuits
