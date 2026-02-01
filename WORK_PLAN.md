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
- [x] Transient CLI integration
  - Stamper trait extended with transient_info() returning TransientDeviceInfo
  - NetlistTransientStamper for per-step circuit assembly from Netlist
  - DC operating point as initial condition, then time-stepping
  - Tabular time-domain output (Time, V(1), V(2), ...)

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
  - [x] Controlled source AC stamps (VCVS, VCCS, CCCS, CCVS)
  - [ ] Nonlinear devices: automatic linearization from DC point
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

- [ ] Symbolic factorization caching
  - For NR iterations and transient timesteps, the sparsity pattern is fixed
  - Cache `SymbolicLu` and only redo numeric factorization
  - Major speedup for repeated solves with the same structure
- [ ] Sparse-only MnaSystem
  - Remove dense matrix field from MnaSystem
  - Build dense representation on demand only when needed (tests, small circuits)
  - Reduces memory from O(n²) to O(nnz)
- [ ] Sparse-only ComplexMna
  - Apply same sparse-only treatment to AC analysis complex MNA system

### GMRES Integration (leverages Phase 8 infrastructure)

- [ ] `SparseRealOperator` wrapper
  - Wraps faer `SparseColMat<f64>` and implements `RealOperator`
  - Enables GMRES as alternative to direct LU for real systems (DC, transient)
- [ ] `SparseComplexOperator` wrapper
  - Wraps faer `SparseColMat<c64>` and implements `ComplexOperator`
  - Enables GMRES as alternative for complex systems (AC)
- [ ] Real-valued GMRES (`solve_gmres_real`)
  - Port complex GMRES to work with `RealOperator`
  - Uses `real_dot_product` SIMD kernels from `spicier-simd`
- [ ] Solver selection heuristic
  - Auto-choose direct LU vs iterative GMRES based on system size
  - Direct LU preferred for small/medium systems (fast, exact)
  - GMRES preferred for very large systems (>10k nodes) where LU memory/time dominates
  - Configurable threshold and override via CLI/API
- [ ] Preconditioned GMRES
  - ILU(0) or Jacobi preconditioner for faster convergence
  - Preconditioner reuse across NR iterations (same sparsity pattern)

**Dependencies:** Phase 7 (AC analysis uses ComplexMna), Phase 8 (operator traits, SIMD)

**Acceptance Criteria (remaining):** NR and transient solves reuse symbolic factorization. MnaSystem memory scales with nnz, not n². GMRES available as alternative solver for large systems.

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

- [ ] SIMD-friendly data layouts
  - Structure of Arrays (SoA) for devices
  - Aligned memory allocation
- [ ] Vectorized device evaluation
  - Batch evaluate same-type devices
  - Use SIMD kernels for diode/MOSFET evaluation
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

- [ ] Wire GPU operators into analysis paths
  - Use GPU dense operators for AC analysis matrix solves
  - Use GPU operators for GMRES preconditioning
- [ ] Size-based dispatch heuristic
  - Small circuits (<1k nodes) stay on CPU even when GPU is available
  - Threshold tunable via config; auto-calibrate with a one-time microbenchmark
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

## Future Considerations

- Subcircuit/hierarchical netlists (.SUBCKT)
- Behavioral sources (B elements)
- Noise analysis
- Additional MOSFET models (BSIM3/BSIM4)
- S-parameter analysis
- .MEASURE statements
- .PARAM / parameter expressions
