# Work Log

## 2026-02-01

- Created project tracking documents (WORK_PLAN.md and WORK_LOG.md)
- Defined phased roadmap with 10 phases (0-9) covering bootstrap through GPU backend
- Established work log format for tracking daily progress
- Added ngspice as git submodule for reference implementation study
- Added spice21 as git submodule for reference

### Phase 0: Project Bootstrap - COMPLETE

- Initialized Cargo workspace with 5 crates:
  - `spicier-core` - Circuit graph, nodes, MNA matrix, units
  - `spicier-solver` - Linear solver (LU decomposition)
  - `spicier-devices` - R, L, C, V, I device models with MNA stamps
  - `spicier-parser` - Netlist parsing (skeleton)
  - `spicier-cli` - Command-line interface with clap
- Set up GitHub Actions CI (.github/workflows/ci.yml)
- Configured criterion benchmarks for all crates
- All 25 tests passing, clippy clean

### Phase 1: Core Data Structures - COMPLETE

- Implemented `NodeId` with ground node handling
- Implemented `Node` struct with optional naming
- Implemented `Circuit` graph with node management
- Implemented `MnaSystem` with stamping methods:
  - `stamp_conductance()` - for resistive elements
  - `stamp_current_source()` - for current sources
  - `stamp_voltage_source()` - for voltage sources/inductors
- Added SI unit parsing (T, G, MEG, K, M, U, N, P, F suffixes)

### Phase 3: Linear Passive Devices - COMPLETE

- Implemented `Resistor` with conductance stamp
- Implemented `Capacitor` (DC: open circuit)
- Implemented `Inductor` (DC: short circuit via voltage source)
- Implemented `VoltageSource` with current variable
- Implemented `CurrentSource` with RHS contribution
- All devices implement `Stamp` trait

### Phase 4: DC Analysis - MOSTLY COMPLETE

- Implemented dense linear solver using nalgebra LU decomposition
- Created `Netlist` type for assembling circuits programmatically
- Created `Stamper` trait in core for device stamping
- Implemented `solve_dc()` returning `DcSolution` with node voltages and branch currents
- Added 4 integration tests:
  - Voltage divider (10V source, two 1k resistors)
  - Current divider (10mA source, parallel resistors)
  - Complex resistor network (4 resistors, analytical verification)
  - Current variable counting
- All integration tests pass with analytically verified results
- Remaining: DC sweep (.DC), sparse solver, output infrastructure

### Phase 5: Nonlinear Devices & Newton-Raphson - MOSTLY COMPLETE

- Implemented diode model (Shockley equation)
  - Forward/reverse bias evaluation
  - Linearized conductance + equivalent current source
  - Voltage limiting to prevent exp() overflow
  - Configurable parameters: Is, n, Rs, Cj0, Vj, Bv
  - Thermal voltage calculation
- Implemented Newton-Raphson iteration loop
  - NonlinearStamper trait for per-iteration restamping
  - Configurable convergence criteria (Vabstol, Vreltol, Iabstol, max iterations)
  - Convergence checking for voltages and currents
- Implemented MOSFET Level 1 model
  - Cutoff, linear, saturation regions with correct I-V equations
  - NMOS and PMOS support (PMOS via voltage/current negation)
  - Output conductance (gds), transconductance (gm)
  - Channel-length modulation (lambda parameter)
  - Configurable model parameters (Vto, Kp, W, L, Cox)
  - Nonlinear stamp method for Newton-Raphson integration
  - 5 unit tests verifying regions and analytical I-V values
- Integration test: V=5V + R=1k + diode converges to Vd ≈ 0.6-0.7V
- Remaining: source stepping, Gmin stepping convergence aids

### Phase 6: Transient Analysis - IN PROGRESS

- Implemented integration methods:
  - Backward Euler (1st order, A-stable)
  - Trapezoidal (2nd order, A-stable)
- Implemented companion models:
  - Capacitor: Geq + Ieq for both BE and Trapezoidal
  - Inductor: Geq + Ieq for both BE and Trapezoidal
- Implemented transient simulation loop (`solve_transient`)
  - TransientStamper trait for per-step circuit assembly
  - Fixed timestep simulation
  - TransientResult with voltage waveform extraction
- Tests: RC charging matches exponential curve (BE and Trapezoidal)
- Remaining: adaptive timestep (LTE), initial conditions (.IC), method switching

### Phase 7: AC Analysis - MOSTLY COMPLETE

- Implemented complex MNA system (`ComplexMna`)
  - Complex admittance, conductance, current source, voltage source stamps
  - Inductor AC stamp with jωL impedance and branch current
  - VCCS stamp for small-signal transconductance (gm)
- Implemented complex linear solver using nalgebra LU decomposition
- Implemented AC sweep engine (`solve_ac`)
  - AcStamper trait for per-frequency stamping
  - Three sweep types: Linear, Decade, Octave
  - Frequency vector generation for all sweep types
- Implemented AcResult with analysis accessors:
  - Complex voltage at node
  - Magnitude in dB
  - Phase in degrees
- Tests:
  - RC low-pass filter: -3dB point at f=1/(2πRC) ≈ 159 Hz ✓
  - RC low-pass rolloff: -20 dB/decade ✓
  - RC low-pass phase: 0° at DC, -45° at f_3dB, -90° at high freq ✓
  - RL low-pass filter: -3dB point at f=R/(2πL) ✓
  - Frequency generation for linear, decade, octave sweeps ✓
  - ComplexMna stamp correctness (admittance, voltage source) ✓
- Remaining: automatic linearization of nonlinear devices from DC point

### Phase 4 Update: DC Sweep

- Implemented DC sweep solver (`solve_dc_sweep`)
  - DcSweepStamper trait for per-point circuit stamping
  - DcSweepResult with voltage/current waveform accessors
  - Sweep value generation with configurable start/stop/step
- Integration test: voltage divider sweep verifies V(2) = V1/2 at all points

### Analysis Command Parsing

- Extended parser to extract analysis commands from netlists:
  - `.OP` - DC operating point
  - `.DC source start stop step` - DC sweep parameters
  - `.AC type npoints fstart fstop` - AC sweep (DEC/OCT/LIN)
  - `.TRAN tstep tstop [tstart]` - Transient parameters
- Added `ParseResult` struct with `netlist` + `analyses` fields
- `parse_full()` function returns both; `parse()` remains backward-compatible
- AcSweepType and AnalysisCommand enums exported from parser

### Stamper Trait Extensions

- Added `device_name()` for named device lookup
- Added `branch_index()` for voltage source branch identification
- Added `ac_info()` returning AcDeviceInfo enum for AC stamping
- Added `stamp_into()` and `find_vsource_branch_index()` to Netlist
- All passive devices and sources implement extended Stamper methods

### CLI Multi-Analysis Support

- CLI now dispatches based on parsed analysis commands
- Supported analyses via CLI:
  - .OP: DC operating point (tabular node voltages + branch currents)
  - .DC: DC sweep (tabular output with source value + node voltages)
  - .AC: AC analysis (tabular output with frequency, magnitude dB, phase degrees)
  - .TRAN: recognized but deferred to solver API
- Defaults to .OP when no analysis commands present
- `--op` flag still works for explicit DC operating point
- New example netlists: dc_sweep.sp, ac_filter.sp

### Phase 2: Netlist Parsing - COMPLETE

- Implemented SPICE lexer with token types:
  - Name, Value, Command, Node, Equals, LParen, RParen, Eol, Eof
  - Comments (* and ;), line continuations (+)
- Implemented parser for elements: R, C, L, V, I
- Support for SI suffixes (k, M, u, n, p, etc.) and scientific notation
- Ground node aliases: 0, gnd, GND
- Named nodes (e.g., "vdd", "in", "out")
- 17 lexer/parser unit tests, 6 end-to-end parse+simulate tests
- All tests pass with analytically verified results

### CLI Integration

- Updated CLI to parse netlist files and run multiple analysis types
- Verbose mode shows circuit statistics and detected analysis commands
- Example netlists:
  - voltage_divider.sp - V(2)=5V verified
  - current_source.sp - V(1)=5V verified
  - rc_circuit.sp - DC capacitor open circuit
  - inductor_dc.sp - DC inductor short circuit verified
  - dc_sweep.sp - DC sweep of voltage divider
  - ac_filter.sp - RC low-pass filter AC analysis

### Documentation

- Updated README with competitive landscape section
- Added positioning vs Spice21, krets, and other Rust SPICE projects
- Added public roadmap with 5 phases (Foundations through Advanced Models)
- Added design philosophy statement
- 83 total tests passing, clippy clean

### Nonlinear DC, Transient CLI, D/M Parsing, Controlled Sources

Extended Stamper trait and wired remaining solver capabilities into CLI end-to-end.

**Stamper trait extensions:**
- Added `is_nonlinear()`, `stamp_nonlinear(mna, solution)`, `transient_info()` to Stamper trait
- Added `TransientDeviceInfo` enum (Capacitor, Inductor, None)
- Added `has_nonlinear_devices()` and `stamp_nonlinear_into()` to Netlist

**Nonlinear device integration:**
- Diode and MOSFET `stamp_nonlinear()` renamed to `stamp_linearized_at()` (avoids trait collision)
- Implemented Stamper::stamp_nonlinear() on both — extracts operating point from solution vector
- Fixed diode voltage limiting consistency bug: `limit_voltage()` now applied before both `evaluate()` and `ieq` computation
- NetlistNonlinearStamper bridges Netlist to Newton-Raphson solver
- CLI auto-dispatches to NR when nonlinear devices present
- Diode circuit (V=5V + R=1k + D1): converges in 10 iterations, V(diode)=0.74V

**Transient CLI integration:**
- Capacitor and Inductor implement `transient_info()` returning device parameters
- NetlistTransientStamper: stamps all non-reactive devices per timestep
- `build_transient_state()` extracts CapacitorState/InductorState from netlist
- `run_transient()`: DC OP → time-stepping with Trapezoidal method → tabular output

**Parser: D/M elements + .MODEL:**
- `ModelDefinition` enum (Diode, Nmos, Pmos) with `HashMap<String, ModelDefinition>` storage
- `.MODEL name type (param=value ...)` parsing for D, NMOS, PMOS types
- `D1 anode cathode [modelname]` parsing
- `M1 drain gate source bulk [modelname] [W=val L=val]` parsing

**Controlled sources (E/G/F/H):**
- New file: `crates/spicier-devices/src/controlled.rs`
- VCVS (E): branch current variable + gain coupling
- VCCS (G): direct gm matrix stamps (sign convention: positive current enters out_pos)
- CCCS (F): gain * I(Vsource) into output nodes
- CCVS (H): branch current variable + transresistance coupling
- All implement Stamp, Element, Stamper traits with ac_info()
- Parser support for E/G/F/H element lines
- AC stamper handles all controlled source variants with complex stamps
- Fixed VCCS sign convention (SPICE G element: current enters out_pos terminal)

**Tests:** 101 total passing (was 83), clippy clean
**New examples:** diode_circuit.sp, rc_transient.sp

### Sparse Solver Integration via faer

Integrated faer 0.24 (pure Rust) for sparse LU factorization alongside the existing nalgebra dense solver. All analysis paths now auto-select sparse or dense based on system size.

**Architecture changes:**
- MnaSystem gains `triplets: Vec<(usize, usize, f64)>` accumulator
- New `add_element(row, col, value)` method writes to both dense matrix and triplets
- New `add_rhs(row, value)` method for RHS entries
- ComplexMna gains equivalent `triplets` and `add_element()` for AC analysis
- `stamp_conductance()`, `stamp_current_source()`, `stamp_voltage_source()` refactored to use `add_element()`/`add_rhs()` internally
- `clear()` now also clears triplets

**Device stamping migration:**
- All 4 controlled sources (VCVS, VCCS, CCCS, CCVS) migrated from `mna.matrix_mut()[(i,j)] += v` to `mna.add_element(i, j, v)` (~20 call sites)
- MOSFET `stamp_linearized_at()` gm stamping migrated (~4 call sites)
- AC controlled source stamping in CLI migrated (~18 call sites)

**New solver functions (`linear.rs`):**
- `solve_sparse(size, triplets, rhs)` — builds `SparseColMat` from triplets, sparse LU via `sp_lu()`
- `solve_sparse_complex(size, triplets, rhs)` — same for complex systems using `c64`
- `SPARSE_THRESHOLD = 50` — systems with 50+ variables use sparse path

**Analysis function updates:**
- `solve_dc()` — auto-selects sparse when `mna.size() >= 50`
- `solve_newton_raphson()` — auto-selects sparse per NR iteration
- `solve_transient()` — auto-selects sparse per timestep
- `solve_ac()` — auto-selects sparse complex per frequency point

**Tests:** 106 total passing (was 101), clippy clean
- `test_solve_sparse_simple` — 2x2 real system
- `test_solve_sparse_complex_simple` — 2x2 complex system, verified via Ax=b
- `test_solve_sparse_matches_dense` — 20x20 diagonally dominant system, sparse == dense within 1e-10
- `test_solve_sparse_dimension_mismatch` — error handling
- `test_solve_sparse_with_duplicate_triplets` — verifies faer sums duplicates

**Benchmarks:** Updated with `solve_sparse` group at sizes 10/50/100/500 for crossover analysis

**Files modified:** Cargo.toml (workspace + solver), mna.rs, linear.rs, dc.rs, newton.rs, transient.rs, ac.rs, controlled.rs, mosfet.rs, main.rs, solver.rs (bench)

### SIMD, GMRES, and GPU Backend Infrastructure

Ported reusable infrastructure from the `mom` (Method of Moments EM solver) project to accelerate Phases 8 and 9. This brings in battle-tested SIMD kernels, operator traits, and GPU backend abstractions.

**New crates:**

| Crate | Purpose | Source |
|-------|---------|--------|
| `spicier-simd` | SIMD detection + dot products | `mom-core/src/simd.rs`, `mom-backend-cpu/src/simd.rs` |
| `spicier-backend-cpu` | CPU dense operators using SIMD | `mom-backend-cpu/src/dense_operator.rs` |
| `spicier-backend-cuda` | CUDA context + cuBLAS operators | `mom-backend-cuda/` |
| `spicier-backend-metal` | WebGPU/Metal context + compute shaders | `mom-backend-metal/` |

**spicier-simd (new crate):**
- `SimdCapability` enum with `detect()` — runtime AVX-512/AVX2/scalar selection
- `real_dot()`, `real_matvec()` — f64 SIMD kernels
- `complex_dot()`, `complex_matvec()` — C64 SIMD kernels
- `conjugate_dot()` — Hermitian dot product for GMRES
- 19 unit tests

**spicier-solver additions:**
- `RealOperator` trait — `dim()`, `apply(&[f64], &mut [f64])`
- `ComplexOperator` trait — `dim()`, `apply(&[C64], &mut [C64])`
- `ComputeBackend` enum — `Cpu`, `Cuda { device_id }`, `Metal { adapter_name }`
- `from_name()` for CLI parsing, `Display` impl
- GMRES iterative solver (`gmres.rs`)
  - Generic over `ComplexOperator`/`RealOperator`
  - Configurable restart (default 30), tolerance (1e-10), max iterations (1000)
  - Givens rotations for stable least-squares minimization
  - 9 unit tests

**spicier-backend-cpu (new crate):**
- `CpuRealDenseOperator` — f64 dense matvec using `real_matvec()`
- `CpuComplexDenseOperator` — C64 dense matvec using `complex_matvec()`
- Both implement `RealOperator`/`ComplexOperator` traits
- 12 unit tests

**spicier-backend-cuda (new crate):**
- `CudaContext` — cudarc 0.16 with dynamic CUDA loading
- `is_available()` — probes for CUDA without hard runtime dependency
- `CudaRealDenseOperator` — cuBLAS `dgemv` with CPU fallback
- `CudaComplexDenseOperator` — cuBLAS `zgemv` with CPU fallback
- CPU fallback threshold (default 64) for small matrices
- 4 unit tests (graceful skip on non-CUDA systems)

**spicier-backend-metal (new crate):**
- `WgpuContext` — wgpu 23 with Metal/Vulkan/DX12 backend selection
- `is_available()` — probes for GPU adapter
- `supports_f64()` — checks for SHADER_F64 feature
- `WgpuRealDenseOperator` — WGSL compute shader (`real_matvec.wgsl`)
- `WgpuComplexDenseOperator` — WGSL compute shader (`complex_matvec.wgsl`)
- f32 GPU computation with f64 CPU fallback for precision
- CPU fallback threshold (default 64) for small matrices
- 6 unit tests

**CLI integration:**
- New `--backend` flag: `auto` (default), `cpu`, `cuda`, `metal`
- Auto-detection priority: Metal (macOS) → CUDA → CPU
- Graceful fallback with warnings when requested backend unavailable
- Verbose mode (`-v`) prints selected backend
- GPU backends behind feature flags (`cuda`, `metal`) to keep default binary lightweight

**Tests:** 169 total passing (was 106), clippy clean

**Files added:**
- `crates/spicier-simd/` (4 modules + lib.rs)
- `crates/spicier-backend-cpu/` (dense_operator.rs + lib.rs)
- `crates/spicier-backend-cuda/` (context.rs, error.rs, dense_operator.rs + lib.rs)
- `crates/spicier-backend-metal/` (context.rs, error.rs, dense_operator.rs, real_matvec.wgsl, complex_matvec.wgsl + lib.rs)
- `crates/spicier-solver/src/operator.rs`
- `crates/spicier-solver/src/gmres.rs`
- `crates/spicier-solver/src/backend.rs`

**Files modified:**
- `Cargo.toml` (workspace members + dependencies)
- `crates/spicier-solver/Cargo.toml` (spicier-simd dependency)
- `crates/spicier-solver/src/lib.rs` (new module exports)
- `crates/spicier-cli/Cargo.toml` (optional GPU backend deps)
- `crates/spicier-cli/src/main.rs` (--backend flag + detect_backend())

### Symbolic Factorization Caching

Added cached sparse LU solvers that separate symbolic and numeric factorization. The symbolic factorization (elimination tree, fill-in pattern) is computed once and reused for repeated solves with the same sparsity pattern.

**New types (`linear.rs`):**
- `CachedSparseLu` — real-valued cached solver
  - `new(size, triplets)` — creates solver with symbolic factorization
  - `solve(triplets, rhs)` — solves using cached symbolic, only numeric factorization
- `CachedSparseLuComplex` — complex-valued cached solver (for AC analysis)
  - Same API as real version

**Integration:**
- Newton-Raphson (`newton.rs`) — creates cached solver on first iteration, reuses for subsequent NR iterations
- Transient (`transient.rs`) — creates cached solver on first timestep, reuses for all timesteps
- AC analysis (`ac.rs`) — creates cached solver on first frequency point, reuses for all frequencies

**Tests:** 175 total passing (was 169), 6 new tests for cached solvers

**Files modified:**
- `crates/spicier-solver/src/linear.rs` (CachedSparseLu, CachedSparseLuComplex + tests)
- `crates/spicier-solver/src/error.rs` (SolverError variant)
- `crates/spicier-solver/src/lib.rs` (exports)
- `crates/spicier-solver/src/newton.rs` (use cached solver)
- `crates/spicier-solver/src/transient.rs` (use cached solver)
- `crates/spicier-solver/src/ac.rs` (use cached solver)
