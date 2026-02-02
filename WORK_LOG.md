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

### Sparse-Only MNA Systems

Converted both `MnaSystem` and `ComplexMna` to sparse-only storage, eliminating redundant O(n²) dense matrix memory. Values are now stored as triplets; dense matrix is built on demand for tests and small circuits.

**MnaSystem changes (`spicier-core/src/mna.rs`):**
- Removed `matrix: DMatrix<f64>` field
- Kept `triplets: Vec<(usize, usize, f64)>` as sole storage
- `add_element()` now only pushes to triplets (duplicates allowed, summed during construction)
- New `to_dense_matrix()` method builds `DMatrix<f64>` on demand
- Removed `matrix()` and `matrix_mut()` accessors

**ComplexMna changes (`spicier-solver/src/ac.rs`):**
- Removed `matrix: DMatrix<Complex<f64>>` field
- Added `num_vsources` field to calculate size without matrix
- `add_element()` now only pushes to triplets
- New `to_dense_matrix()` method builds `DMatrix<Complex<f64>>` on demand
- Removed `matrix()` and `matrix_mut()` accessors

**Solver updates:**
- `solve_dc()` — uses `&mna.to_dense_matrix()` for small circuits
- `solve_newton_raphson()` — uses `&mna.to_dense_matrix()` for small circuits
- `solve_transient()` — uses `&mna.to_dense_matrix()` for small circuits
- `solve_ac()` — uses `&mna.to_dense_matrix()` for small circuits

**Test updates:**
- All tests (~20 locations) updated from `sys.matrix[...]` to `let matrix = sys.to_dense_matrix(); matrix[...]`
- Files updated: mna.rs, netlist.rs, passive.rs, sources.rs, controlled.rs, transient.rs, ac.rs

**Tests:** 174 total passing, clippy clean

### Sparse Operator Wrappers

Added operator wrappers for faer sparse matrices to enable iterative solvers like GMRES.

**New types (`sparse_operator.rs`):**
- `SparseRealOperator` — wraps `SparseColMat<usize, f64>`, implements `RealOperator`
- `SparseComplexOperator` — wraps `SparseColMat<usize, c64>`, implements `ComplexOperator`

**Features:**
- CSC matrix-vector multiplication (y = A * x)
- Construction from triplets or existing faer sparse matrices
- Compatible with trait object dispatch for use with GMRES

**Tests:** 180 total passing (6 new tests)

### Real-Valued GMRES

Added real-valued GMRES solver as a more efficient alternative to complex GMRES for DC and transient analysis.

**New functions (`gmres.rs`):**
- `solve_gmres_real(&dyn RealOperator, &[f64], &GmresConfig) -> RealGmresResult`
  - SIMD-accelerated dot products via `spicier_simd::real_dot_product`
  - Modified Gram-Schmidt orthogonalization
  - Givens rotations for stable least-squares minimization
  - Restarted GMRES with configurable restart parameter

**New types:**
- `RealGmresResult` — solution, iterations, residual, converged flag

**Tests:** 189 total passing (9 new GMRES tests)

### Solver Selection Heuristic

Added automatic solver selection between direct LU and iterative GMRES based on system size.

**New module (`solver_select.rs`):**
- `SolverStrategy` enum — `Auto`, `DirectLU`, `IterativeGmres`
- `SolverConfig` — strategy, GMRES threshold (default 10k), GMRES config
- `solve_auto()` — automatically selects solver based on system size
- `SolveResult` — solution + metadata (solver used, iterations, residual)

**Features:**
- Auto mode: uses LU for systems < 10k nodes, GMRES for larger systems
- GMRES fallback to LU if iteration doesn't converge
- CLI-friendly `from_name()` for parsing strategy strings
- Configurable threshold via `SolverConfig::with_threshold()`

**Tests:** 197 total passing (8 new solver selection tests)

### Preconditioned GMRES

Added preconditioner infrastructure for GMRES iterative solver to improve convergence on ill-conditioned systems.

**New module (`preconditioner.rs`):**
- `RealPreconditioner` trait — `apply(&self, x: &[f64], y: &mut [f64])`, `dim()`
- `ComplexPreconditioner` trait — same interface for complex systems
- `JacobiPreconditioner` — diagonal preconditioner for real systems
  - `from_triplets(size, triplets)` — extracts and inverts diagonal entries
  - `from_diagonal(diag)` — direct construction from diagonal vector
  - Handles zero/near-zero diagonal entries (treats as 1.0)
- `ComplexJacobiPreconditioner` — same for complex systems
- `IdentityPreconditioner` — no-op preconditioner for both real and complex

**GMRES extensions (`gmres.rs`):**
- `solve_gmres_real_preconditioned()` — right-preconditioned real GMRES
  - Solves A*M^(-1)*y = b, then x = M^(-1)*y
  - Uses preconditioner for improved spectral properties
- `solve_gmres_preconditioned()` — right-preconditioned complex GMRES
- Both functions take `&dyn RealPreconditioner` / `&dyn ComplexPreconditioner`

**Tests:** 211 total passing (14 new preconditioner + preconditioned GMRES tests)

### Phase 7c Complete

All Phase 7c tasks completed:
- ✅ Sparse-only MnaSystem and ComplexMna
- ✅ Sparse operator wrappers (SparseRealOperator, SparseComplexOperator)
- ✅ Real-valued GMRES (solve_gmres_real)
- ✅ Solver selection heuristic (solve_auto, SolverStrategy)
- ✅ Preconditioned GMRES (Jacobi preconditioner, right preconditioning)

### Phase 8: SIMD-Friendly Data Layouts

Added batched device structures with Structure-of-Arrays (SoA) layout for SIMD-efficient device evaluation.

**New module (`spicier-devices/src/batch.rs`):**
- `DiodeBatch` — batched diode parameters in SoA layout
  - `is`, `n`, `nvt`, `node_pos`, `node_neg` as separate vectors
  - `push()` adds diodes, `finalize()` pads to SIMD lane count
  - `evaluate_batch()` evaluates all diodes with SIMD dispatch
  - `evaluate_batch_scalar()` for scalar fallback
  - AVX2 implementation using intrinsics (`target_feature(enable = "avx2")`)
  - `evaluate_linearized_batch()` computes id, gd, ieq for NR stamping
- `MosfetBatch` — batched MOSFET parameters in SoA layout
  - `mos_type`, `vth`, `beta`, `lambda`, `node_drain/gate/source`
  - Scalar evaluation (branching prevents effective vectorization)
  - `evaluate_batch()` returns ids, gds, gm for all devices
  - `evaluate_linearized_batch()` computes ieq for NR stamping
- `BatchMosfetType` — enum for NMOS/PMOS
- `round_up_to_simd()` — utility for SIMD lane padding

**Key design decisions:**
- Ground nodes encoded as `usize::MAX` for branchless voltage lookup
- Padding with neutral devices (high Vth MOSFET, low Is diode)
- Diode batch uses AVX2 SIMD for arithmetic, scalar exp() per element
- MOSFET batch stays scalar due to region branching

**Tests:** 220 total passing (9 new batch tests)

### Vectorized Device Evaluation Integration

Wired batched device evaluation into Newton-Raphson solver for SIMD-accelerated nonlinear circuit simulation.

**New module (`spicier-solver/src/batched_newton.rs`):**
- `BatchedNonlinearDevices` — container for batched diodes and MOSFETs
  - `add_diode()`, `add_mosfet()` for populating batches
  - `finalize()` pads batches and allocates evaluation buffers
  - `evaluate_and_stamp()` batch-evaluates all devices and stamps MNA
- `LinearStamper` trait — callback for stamping linear devices
- `solve_batched_newton_raphson()` — batched variant of NR solver
  - Uses `LinearStamper` for linear devices (per-iteration)
  - Uses `BatchedNonlinearDevices::evaluate_and_stamp()` for nonlinear
  - Same convergence checking and sparse solver caching as regular NR
- Pre-allocated evaluation buffers to avoid per-iteration allocation

**Performance benefits:**
- Diode evaluation uses AVX2 SIMD (4 diodes per iteration)
- Memory layout optimized for cache locality (SoA vs AoS)
- Evaluation buffers reused across NR iterations
- Stamping loop separate from evaluation (better instruction pipelining)

**Tests:** 224 total passing (4 new batched_newton tests)
- `test_batched_newton_diode` — single diode circuit
- `test_batched_vs_single_diode` — comparison with regular NR
- `test_batched_multiple_diodes` — 10 parallel diodes
- `test_batched_mosfet` — NMOS in saturation

### Parallel Matrix Assembly

Added infrastructure for parallel matrix assembly using thread-local triplet accumulation.

**New module (`spicier-solver/src/parallel.rs`):**
- `ParallelTripletAccumulator` — thread-local triplet buffer management
  - `new(num_threads)` creates accumulator with thread buffers
  - `with_available_parallelism()` uses all CPU cores
  - `get_buffer(thread_id)` returns mutable access to thread's buffer
  - `clear()` resets all buffers for reuse
  - `merge()` combines all buffers into single triplet list
- `stamp_conductance_triplets()` — stamps conductance to triplet buffer
- `stamp_current_source_rhs()` — stamps current source to RHS
- `parallel_ranges()` — splits work across threads evenly

**BatchedNonlinearDevices extensions:**
- `evaluate_and_stamp_triplets()` — outputs to triplet buffer (for parallel merge)
- `evaluate_parallel()` — parallel evaluation with thread-local accumulation
  - Splits device stamping across threads
  - Uses SIMD for evaluation, parallel for stamping
  - Falls back to sequential for small device counts (<100)

**Tests:** 228 total passing (5 new parallel tests)

### Batched Parameter Sweeps

Added infrastructure for Monte Carlo, corner analysis, and parameter sweep simulations.

**New module (`spicier-solver/src/sweep.rs`):**
- `ParameterVariation` — defines parameter with nominal, min/max, sigma
- `SweepPoint` — stores parameter values at a single sweep point
- `SweepStatistics` — mean, std_dev, min, max from samples

**Point generators:**
- `MonteCarloGenerator` — random sampling with Box-Muller normal distribution
  - Reproducible via seed, samples clamped to min/max bounds
- `CornerGenerator` — generates 2^n combinations of min/max values
- `LinearSweepGenerator` — linear interpolation between bounds

**Sweep execution:**
- `SweepStamperFactory` trait — creates stampers for varied parameters
- `SweepStamper` trait — stamps linear devices with specific parameters
- `solve_batched_sweep()` — runs multiple simulations across sweep points
- `BatchedSweepResult` — solutions + statistics accessors

**Tests:** 233 total passing (5 new sweep tests)
- `test_monte_carlo_generator` — reproducibility, bounds checking
- `test_corner_generator` — 2^n corners with all combinations
- `test_linear_sweep_generator` — linear interpolation
- `test_sweep_statistics` — mean/std_dev calculation
- `test_batched_sweep_simple` — voltage divider sweep

### Phase 8 Complete

All Phase 8 tasks completed:
- ✅ SIMD-friendly data layouts (DiodeBatch, MosfetBatch)
- ✅ Vectorized device evaluation (solve_batched_newton_raphson)
- ✅ Parallel matrix assembly (ParallelTripletAccumulator)
- ✅ Batched parameter sweeps (MonteCarloGenerator, CornerGenerator)

### Phase 9: Dispatch Integration for Analysis Paths

Integrated solver dispatch configuration into all analysis paths, enabling automatic selection between direct LU and iterative GMRES solvers based on system size.

**New module (`spicier-solver/src/dispatch.rs`):**
- `DispatchConfig` — unified configuration for solver dispatch
  - `backend: ComputeBackend` — CPU, CUDA, or Metal
  - `strategy: SolverDispatchStrategy` — Auto, DirectLU, or IterativeGmres
  - `cpu_threshold: usize` — systems below this size always use CPU (default 1000)
  - `gmres_threshold: usize` — systems at or above this size use GMRES (default 10000)
  - `gmres_config: GmresConfig` — GMRES configuration for iterative solving
  - Builder pattern: `with_strategy()`, `with_cpu_threshold()`, `with_gmres_threshold()`
  - Decision methods: `use_gpu(size)`, `use_gmres(size)`, `describe(size)`
- `SolverDispatchStrategy` — enum for solver selection strategy
  - `from_name()` for CLI parsing

**Dispatched analysis functions:**

*AC Analysis (`spicier-solver/src/ac.rs`):*
- `solve_ac_dispatched(stamper, params, config)` — AC sweep with dispatch config
- `solve_ac_gmres(mna, gmres_config)` — internal GMRES solver for AC
  - Builds `SparseComplexOperator` from triplets
  - Uses `ComplexJacobiPreconditioner` for faster convergence
  - Logs warning if GMRES doesn't converge

*DC Analysis (`spicier-solver/src/dc.rs`):*
- `solve_dc_dispatched(mna, config)` — DC operating point with dispatch config
- `solve_dc_gmres(mna, gmres_config)` — internal GMRES solver for DC
  - Builds `SparseRealOperator` from triplets
  - Uses `JacobiPreconditioner` for faster convergence
- `solve_dc_sweep_dispatched(stamper, params, config)` — DC sweep with dispatch

*Transient Analysis (`spicier-solver/src/transient.rs`):*
- `solve_transient_dispatched(stamper, caps, inds, params, dc, config)` — transient with dispatch
- `solve_transient_gmres(mna, gmres_config)` — internal GMRES solver for timesteps

**Key design:**
- All dispatched functions fall back to cached sparse LU for medium systems
- GMRES uses Jacobi (diagonal) preconditioner for fast setup
- Non-convergence logged but doesn't error (returns best guess)
- GPU operators available via `RealOperator`/`ComplexOperator` traits

**Tests:** 242 total passing (3 new dispatched tests)
- `test_voltage_divider_dispatched` — DC dispatch with default config
- `test_solve_ac_dispatched` — AC dispatch with RC lowpass
- `test_transient_dispatched` — Transient dispatch with RC charging

**Phase 9 progress:**
- ✅ Wire GPU operators into analysis paths (via dispatched functions)
- ✅ Size-based dispatch heuristic (DispatchConfig with thresholds)
- ⏳ Shared memory management
- ⏳ 9b: Batched sweep solver on GPU
- ⏳ 9c: Batched device evaluation on GPU
- ⏳ 9d: Large-circuit sparse solve
- ⏳ 9e: Post-processing on GPU

### Phase 7: AC Linearization of Nonlinear Devices

Added automatic linearization of nonlinear devices (diodes, MOSFETs) for AC small-signal analysis. AC analysis now computes the DC operating point first, then extracts small-signal parameters for frequency-domain simulation.

**New AcDeviceInfo variants (`spicier-core/src/netlist.rs`):**
- `AcDeviceInfo::Diode { node_pos, node_neg, gd }` — small-signal conductance
- `AcDeviceInfo::Mosfet { drain, gate, source, gds, gm }` — output conductance + transconductance

**New Stamper trait method:**
- `ac_info_at(&self, solution: &DVector<f64>) -> AcDeviceInfo`
  - Computes AC parameters from DC operating point
  - Default implementation falls back to `ac_info()` for linear devices
  - Diode: extracts gd = dId/dVd at Vd from DC solution
  - MOSFET: extracts gds = dIds/dVds and gm = dIds/dVgs at (Vgs, Vds)

**CLI integration (`spicier-cli/src/main.rs`):**
- `NetlistAcStamper` now takes optional `dc_solution` reference
- When netlist has nonlinear devices, AC analysis first runs DC OP
- DC solution passed to `ac_info_at()` for each device
- Diode stamped as conductance gd between anode/cathode
- MOSFET stamped as gds + gm VCCS (small-signal model)

**New example circuits:**
- `examples/diode_ac.sp` — Diode small-signal analysis
- `examples/mosfet_amp_ac.sp` — MOSFET common-source amplifier
- `examples/mosfet_amp_op.sp` — MOSFET DC operating point

**Tests:** 244 total passing (2 new tests)
- `test_ac_info_at_forward_bias` — Diode linearization at 0.7V
- `test_ac_info_at_saturation` — MOSFET linearization in saturation

**Verified behavior:**
- Diode at 0.74V DC shows ~6 ohm small-signal resistance
- MOSFET amplifier shows inverting gain (180° phase shift) as expected
- Both automatically linearized from DC operating point

### Phase 6: Adaptive Timestep Control

Added LTE-based adaptive timestep control for transient analysis using the Milne device method (comparing Trapezoidal vs Backward Euler error estimates).

**New types (`spicier-solver/src/transient.rs`):**
- `AdaptiveTransientParams` — adaptive timestep configuration
  - `tstop: f64` — end time
  - `h_init: f64` — initial timestep
  - `h_min: f64` — minimum allowed timestep
  - `h_max: f64` — maximum allowed timestep
  - `reltol: f64` — relative tolerance (default 0.001)
  - `abstol: f64` — absolute tolerance (default 1e-6)
- `AdaptiveTransientResult` — result with step statistics
  - `timepoints: Vec<TimePoint>` — solution at all accepted timesteps
  - `accepted_steps: usize` — number of accepted timesteps
  - `rejected_steps: usize` — number of rejected (and retried) timesteps
  - Same waveform accessors as `TransientResult`

**LTE estimation methods:**
- `CapacitorState::estimate_lte(v_new, h)` — capacitor current error from Milne device
  - Compares Trapezoidal (2nd order) vs Backward Euler (1st order)
  - Error estimate: |i_trap - i_be| / 3
- `InductorState::estimate_lte(v_new, h)` — inductor current increment error
  - Same Milne device principle for inductor current

**New solver function:**
- `solve_transient_adaptive()` — LTE-based adaptive timestep transient solver
  - Starts from DC operating point
  - Estimates LTE from all capacitors and inductors
  - Computes timestep scale factor: 0.9 * (tol / lte)^(1/order)
  - Rejects step and halves h if LTE > tolerance
  - Increases h up to h_max when LTE << tolerance
  - Enforces h_min to prevent timestep collapse
  - Returns error if h_min reached (circuit too stiff)

**Tests:** 244 total passing
- `test_adaptive_rc_charging` — RC circuit with adaptive timestep
- `test_lte_estimation` — LTE accuracy for constant-rate voltage change

**Remaining Phase 6 items:**
- Output at specified times (interpolation)
- UIC (Use Initial Conditions) option for transient
- Method switching (gear orders)

### Phase 6: Initial Conditions (.IC)

Added support for `.IC` commands to specify initial node voltages for transient analysis.

**Parser changes (`spicier-parser/src/parser.rs`):**
- `InitialCondition` struct: `{ node: String, voltage: f64 }`
- `ParseResult` now includes `initial_conditions: Vec<InitialCondition>` and `node_map: HashMap<String, NodeId>`
- `parse_ic_command()` parses `.IC V(node1)=value V(node2)=value ...`
  - Handles tokenized input: V, (, node, ), =, value
  - Supports both numbered and named nodes

**Solver changes (`spicier-solver/src/transient.rs`):**
- `InitialConditions` struct with `voltages: HashMap<String, f64>`
  - `set_voltage(&mut self, node: &str, voltage: f64)`
  - `apply(&self, solution: &mut DVector<f64>, node_map: &HashMap<String, usize>)`

**CLI integration (`spicier-cli/src/main.rs`):**
- `run_transient()` now accepts `initial_conditions` and `node_map`
- Applies `.IC` values to DC solution before transient simulation
- Prints applied initial conditions when verbose

**New example:**
- `examples/rc_ic.sp` — RC circuit with `.IC V(2)=2.5`

**Tests:** 248 total passing (2 new parser tests)

### Phase 5: Source Stepping Convergence Aid

Added source stepping as a convergence aid for difficult nonlinear circuits. Source stepping gradually ramps independent sources from a small fraction to full value, helping circuits find an operating point when standard Newton-Raphson fails.

**Stamper trait extensions (`spicier-core/src/netlist.rs`):**
- `is_source(&self) -> bool` — identifies independent sources (V, I)
- `stamp_nonlinear_scaled(&self, mna, solution, source_factor)` — stamps with scaled source values
- `Netlist::stamp_nonlinear_into_scaled()` — stamps all devices with source scaling

**Source device updates (`spicier-devices/src/sources.rs`):**
- `VoltageSource::is_source()` returns true
- `VoltageSource::stamp_nonlinear_scaled()` scales voltage by source_factor
- `CurrentSource::is_source()` returns true
- `CurrentSource::stamp_nonlinear_scaled()` scales current by source_factor

**New solver types (`spicier-solver/src/newton.rs`):**
- `ScaledNonlinearStamper` trait — extends NonlinearStamper with `stamp_at_scaled()`
- `SourceSteppingParams` — initial_factor (0.1), factor_step (0.1), max_attempts (5)
- `SourceSteppingResult` — solution, total_iterations, num_levels, converged
- `solve_with_source_stepping()` — progressively increases source_factor from initial to 1.0

**Algorithm:**
1. Start with sources at initial_factor (e.g., 0.1)
2. Solve at each source level using previous solution as initial guess
3. If NR fails, reduce step size and retry
4. Gradually increase to full source value (1.0)
5. Report total iterations and number of levels used

**Tests:** 250 total passing (1 new source stepping test)

### Phase 5: Gmin Stepping Convergence Aid

Added Gmin stepping as a convergence aid that adds a small conductance from each node to ground. Starting with a large Gmin (e.g., 1e-3), the circuit is solved and Gmin is gradually reduced to its final value (e.g., 1e-12).

**MNA extension (`spicier-core/src/mna.rs`):**
- `MnaSystem::stamp_gmin(gmin: f64)` — adds conductance from each node to ground

**New solver types (`spicier-solver/src/newton.rs`):**
- `GminSteppingParams` — initial_gmin (1e-3), final_gmin (1e-12), reduction_factor (10)
- `GminSteppingResult` — solution, total_iterations, num_levels, converged
- `solve_with_gmin_stepping()` — progressively reduces Gmin from initial to final

**Algorithm:**
1. Start with large Gmin (e.g., 1e-3) from each node to ground
2. Solve circuit with this Gmin shunt
3. Reduce Gmin by reduction_factor (e.g., /10)
4. Repeat until final Gmin reached
5. Each level uses previous solution as initial guess

**Tests:** 251 total passing (1 new Gmin stepping test)

### Phase 6: Output at Specified Times

Added interpolation and resampling methods to transient analysis results, enabling output at regular time intervals from simulations with variable (adaptive) timesteps.

**TransientResult extensions (`spicier-solver/src/transient.rs`):**
- `interpolate_at(time)` — linear interpolation between nearest timepoints
- `sample_at_times(tstep, tstart, tstop)` — resample waveform at regular intervals
- `voltage_at(node_idx, time)` — convenience for single-node voltage at specific time

**AdaptiveTransientResult extensions:**
- Same methods as TransientResult
- `sample_at_times()` returns regular TransientResult for uniform output

**Use case:**
- Adaptive solver computes at variable timesteps (e.g., 1ns to 100ns based on LTE)
- User wants output at uniform 10ns intervals
- Call `result.sample_at_times(10e-9, None, None)` to get resampled result

**Tests:** 253 total passing (2 new interpolation/sampling tests)

### Phase 6: UIC (Use Initial Conditions) Option

Added UIC option for `.TRAN` command that skips DC operating point calculation and uses `.IC` values directly as initial conditions.

**Parser changes (`spicier-parser/src/parser.rs`):**
- `AnalysisCommand::Tran` now includes `uic: bool` field
- `parse_tran_command()` checks for "UIC" keyword after time parameters

**CLI integration (`spicier-cli/src/main.rs`):**
- `run_transient()` accepts `uic: bool` parameter
- When `uic=true`: skips DC operating point, starts from zero vector
- `.IC` values then applied to override specific nodes
- Prints "UIC: Skipping DC operating point calculation." in output

**New example:**
- `examples/rc_uic.sp` — RC discharge with UIC (capacitor pre-charged to 5V)
- `examples/rc_ic_no_uic.sp` — RC with .IC but without UIC (shows DC OP override)

**Use case:**
- Capacitor pre-charged to specific voltage without DC OP
- Oscillators and other circuits where DC OP doesn't reflect initial state
- Faster simulation startup by skipping DC OP when initial conditions known

### Phase 6: TR-BDF2 Method Switching

Added TR-BDF2 (Trapezoidal Rule - 2nd order Backward Differentiation Formula) integration method for transient analysis. TR-BDF2 is a composite method that provides L-stability (better than Trapezoidal for stiff circuits) while maintaining 2nd order accuracy.

**New integration method (`spicier-solver/src/transient.rs`):**
- `IntegrationMethod::TrBdf2` — L-stable, 2nd order composite method
- `TRBDF2_GAMMA = 2 - √2 ≈ 0.5858` — optimal gamma for TR-BDF2

**TR-BDF2 algorithm:**
1. Stage 1: Trapezoidal step from t_n to t_n + γ*h (intermediate point)
2. Stage 2: BDF2 step from t_n + γ*h to t_n + h (using non-uniform step formula)

**Capacitor companion model extensions:**
- `v_prev_prev: f64` — stores voltage two timesteps ago for BDF2 history
- `update_trbdf2_intermediate()` — updates state after Stage 1
- `stamp_trbdf2_bdf2()` — stamps BDF2 companion model with non-uniform step coefficients

**Inductor companion model extensions:**
- `i_prev_prev: f64` — stores current two timesteps ago for BDF2 history
- `update_trbdf2_intermediate()` — updates state after Stage 1
- `stamp_trbdf2_bdf2()` — stamps BDF2 companion model

**BDF2 coefficients for non-uniform steps (h1, h2 with ρ = h2/h1):**
- a1 = (1+ρ)² / (1+2ρ)
- a2 = -ρ² / (1+2ρ)
- b0 = (1+ρ) / (1+2ρ)

**Benefits of TR-BDF2:**
- L-stability: Dampens spurious oscillations in stiff circuits
- No numerical ringing unlike pure Trapezoidal
- Better handling of discontinuities (switching, pulsed sources)
- Same 2nd order accuracy as Trapezoidal

**Tests:** 1 new test (test_rc_charging_trbdf2)

### Phase 4: .PRINT Output Infrastructure

Added .PRINT command support for specifying which variables to output in analysis results. This is a standard SPICE feature that allows users to control output instead of printing all nodes.

**Parser additions (`spicier-parser/src/parser.rs`):**
- `PrintAnalysisType` enum: Dc, Ac, Tran
- `OutputVariable` enum: Voltage, Current, VoltageReal, VoltageImag, VoltageMag, VoltagePhase, VoltageDb
- `PrintCommand` struct with analysis_type and variables list
- `parse_print_command()` parses `.PRINT type var1 var2 ...`
- `parse_output_variable()` parses V(node), I(device), VM(node), VP(node), VDB(node), etc.
- Added `Comma` token to lexer for V(node1, node2) differential voltage syntax
- `ParseResult.print_commands` field added

**CLI integration (`spicier-cli/src/main.rs`):**
- `get_dc_print_nodes()` helper determines which nodes to print
- `print_dc_solution()` respects print variables
- `run_dc_op()`, `run_dc_sweep()`, `run_ac_analysis()`, `run_transient()` accept print_vars
- Without .PRINT: outputs all nodes + branch currents
- With .PRINT: outputs only specified variables

**Supported output variables:**
- `V(node)` - Node voltage
- `V(node1, node2)` - Differential voltage (parsed, not yet used)
- `I(device)` - Device current
- `VM(node)` - AC voltage magnitude
- `VP(node)` - AC voltage phase
- `VDB(node)` - AC voltage in dB
- `VR(node)` - AC voltage real part
- `VI(node)` - AC voltage imaginary part

**Example:**
```spice
.PRINT DC V(1) V(2)
.PRINT AC VM(out) VP(out)
.PRINT TRAN V(1) V(2)
```

**Tests:** 2 new parser tests (test_parse_print_command, test_parse_print_ac)
**New examples:** print_test.sp, no_print_test.sp

### Phase 4: Multiple Sweep Variables for .DC

Added support for nested DC sweeps, allowing two sources to be swept in a nested loop. This is commonly used for transistor I-V characteristic curves (e.g., sweeping Vds for multiple Vgs values).

**Parser changes (`spicier-parser/src/parser.rs`):**
- New `DcSweepSpec` struct containing source_name, start, stop, step
- Changed `AnalysisCommand::Dc` to hold `sweeps: Vec<DcSweepSpec>` instead of individual fields
- Updated `parse_dc_command()` to parse optional second sweep specification
- Exported `DcSweepSpec` from parser lib.rs

**CLI changes (`spicier-cli/src/main.rs`):**
- `run_dc_sweep()` now accepts `&[DcSweepSpec]` and dispatches to single or nested sweep
- New `run_single_dc_sweep()` for single source sweep (original behavior)
- New `run_nested_dc_sweep()` for two-source nested sweep
- New `NestedSweepStamper` that patches both swept sources in the RHS
- New `generate_sweep_values()` helper function

**Syntax:**
```spice
.DC V1 0 10 1                     ; Single sweep: V1 from 0 to 10V, step 1V
.DC V1 0 10 2 V2 0 5 1            ; Nested sweep: outer V1 0-10V, inner V2 0-5V
```

In nested sweeps, the first source is the outer (slow) sweep and the second is the inner (fast) sweep.

**Tests:** 1 new parser test (test_parse_dc_command_nested_sweep)
**New examples:** nested_dc_sweep.sp

### Transient .PRINT Output Filtering

Implemented `.PRINT TRAN` output filtering so transient analysis respects the specified output variables instead of printing all nodes.

- Removed `let _ = print_vars` placeholder in `run_transient()`
- Uses `get_dc_print_nodes()` to determine which nodes to print (consistent with DC analysis)
- Only specified nodes appear in transient output when `.PRINT TRAN` is present

**New examples:**
- `print_test_tran.sp` - RC circuit with `.PRINT TRAN V(2)` showing filtered output
- `rc_charging_uic.sp` - RC charging with UIC demonstrating exponential rise

### AC .PRINT Output Filtering

Implemented `.PRINT AC` output filtering so AC analysis respects the specified output variables.

- Added `get_ac_print_nodes()` function that handles AC-specific output types:
  - `V(node)` - Voltage (prints VM and VP)
  - `VM(node)` - Voltage magnitude in dB
  - `VP(node)` - Voltage phase in degrees
  - `VDB(node)` - Voltage dB (same as VM)
  - `VR(node)` - Voltage real part
  - `VI(node)` - Voltage imaginary part
- Deduplicates nodes when same node appears in multiple print vars
- Only specified nodes appear in AC output when `.PRINT AC` is present

**New examples:** `print_test_ac.sp` - RC lowpass filter with `.PRINT AC VM(2) VP(2)`

### README Update

Updated README.md to reflect current project status:
- Test count updated from 106 to 255
- Crate count updated from 5 to 9 (added simd, backend-cpu, backend-cuda, backend-metal)
- Moved completed features from "in progress" to "completed" (adaptive timestep, AC linearization)
- Added new completed features: TR-BDF2, UIC, .PRINT, SIMD, batched evaluation, GMRES, backend abstraction

### Time-Varying Sources (PULSE, SIN, PWL)

Added time-varying waveform support for transient analysis, enabling realistic stimulus patterns for sources.

**New module (`spicier-devices/src/waveforms.rs`):**
- `Waveform` enum with variants: `Dc`, `Pulse`, `Sin`, `Pwl`
- `Waveform::value_at(time)` — evaluates waveform at specific time
- `Waveform::dc_value()` — returns DC value for operating point calculation

**PULSE waveform:**
- Parameters: v1, v2, td (delay), tr (rise), tf (fall), pw (width), per (period)
- Supports periodic pulses with proper edge timing
- Example: `PULSE(0 5 1m 0.1m 0.1m 2m 5m)` — 5V pulse, 1ms delay, 2ms width, 5ms period

**SIN waveform:**
- Parameters: vo (offset), va (amplitude), freq, td (delay), theta (damping), phase
- Supports damped sinusoids: vo + va * sin(2πf(t-td) + phase) * exp(-theta*(t-td))
- Example: `SIN(0 1 1k 0 0 0)` — 1kHz, 1V amplitude sine wave

**PWL (Piecewise Linear) waveform:**
- Arbitrary time-value pairs with linear interpolation
- Holds first value before first time, last value after last time
- Example: `PWL(0 0 1m 5 2m 5 3m 0)` — ramp up at 1ms, hold, ramp down at 3ms

**Source device extensions (`spicier-devices/src/sources.rs`):**
- `VoltageSource::with_waveform()` — creates source with time-varying waveform
- `VoltageSource::value_at(time)` — evaluates waveform at time
- `CurrentSource::with_waveform()` and `CurrentSource::value_at(time)` — same for current sources
- `stamp_at_time()` method stamps source value at specific time

**Stamper trait extension (`spicier-core/src/netlist.rs`):**
- `stamp_at_time(&self, mna, time)` — stamps device at specific simulation time
- Default implementation calls regular `stamp()` for non-time-varying devices
- Time-varying sources override to evaluate waveform at given time

**TransientStamper update (`spicier-solver/src/transient.rs`):**
- Renamed `stamp_static()` to `stamp_at_time(time)` for clarity
- All call sites updated to pass current simulation time

**Parser support (`spicier-parser/src/parser.rs`):**
- `parse_pulse_waveform()` — parses PULSE(...) specification
- `parse_sin_waveform()` — parses SIN(...) specification
- `parse_pwl_waveform()` — parses PWL(...) specification
- `try_expect_value()` helper for optional parameters with defaults
- Voltage and current source parsing now handles DC + waveform syntax

**CLI integration (`spicier-cli/src/main.rs`):**
- `NetlistTransientStamper::stamp_at_time()` passes time to each device
- Time-varying sources automatically use correct value at each timestep

**New example:**
- `examples/rc_pulse.sp` — RC circuit with PULSE source showing proper charging/discharging response

**Verified behavior:**
- V(1) shows PULSE waveform: 0V → 5V transition with proper rise/fall times
- V(2) shows RC response: exponential charging during pulse, exponential discharge after

### Bug Fix: Inductor Companion Model Sign

Fixed sign error in inductor companion model current source stamping. The inductor current source was being stamped in the wrong direction, causing instability in circuits with inductors.

**The bug:**
- `stamp_current_source(node_neg, node_pos, ieq)` was stamping current FROM node_neg TO node_pos
- But inductor current i_prev flows from node_pos to node_neg
- This caused unstable simulation behavior (exponential growth instead of oscillation)

**The fix:**
- Changed to `stamp_current_source(node_pos, node_neg, ieq)` in:
  - `InductorState::stamp_be()` (Backward Euler)
  - `InductorState::stamp_trap()` (Trapezoidal)
  - `InductorState::stamp_trbdf2_bdf2()` (TR-BDF2)

**Verification:**
- Added `test_lc_oscillation` test that simulates an LC circuit
- L = 1mH, C = 1µF, expected frequency = 5033 Hz
- Measured frequency = 5026 Hz (0.13% error)
- Amplitude preserved over 5 oscillation periods

**Tests:** 256 total passing

### Subcircuit Support (.SUBCKT/.ENDS)

Implemented hierarchical subcircuit definitions and instance expansion for modular netlist organization.

**New data structures (`spicier-parser/src/parser.rs`):**
- `RawElementLine` — stores unexpanded element lines within subcircuits
- `SubcircuitDefinition` — name, ports, elements, and nested instances
- `ParseResult.subcircuits` — HashMap of all subcircuit definitions

**Parsing support:**
- `.SUBCKT name port1 port2 ...` — starts subcircuit definition
- `.ENDS [name]` — ends subcircuit definition
- `Xname node1 node2 ... subckt_name` — subcircuit instance
- Elements inside subcircuits stored as raw lines for later expansion
- Nested subcircuit instances tracked separately from regular elements

**Subcircuit expansion:**
- `expand_subcircuit()` — flattens subcircuit instance into parent netlist
- `expand_element_line()` — performs node substitution and name prefixing
- Port nodes map to external connections (e.g., `in` → node `1`)
- Internal nodes get unique prefixes (e.g., `mid` → `X1_mid`)
- Element names preserve type prefix (e.g., `R1` → `RX1_1`)
- Nested subcircuit instances recursively expanded

**Node ID conflict handling:**
- Fixed node ID collision between named internal nodes and numeric external nodes
- Connection nodes registered before expansion to claim their IDs first
- Numeric node names check for existing ID conflicts before assignment
- Internal nodes assigned IDs that don't conflict with explicit numeric nodes

**Element types supported in subcircuits:**
- R, C, L (passive elements)
- V, I (sources)
- D (diodes)
- M (MOSFETs)
- X (nested subcircuit instances)

**Tests:**
- `test_parse_subcircuit_definition` — basic parsing of .SUBCKT/.ENDS blocks
- `test_subcircuit_voltage_divider` — simple voltage divider as subcircuit
- `test_nested_subcircuits` — two-level nesting with inner RES subcircuit

**Verified behavior:**
- Subcircuit voltage divider: V(out) = 5V from 10V source with equal resistors ✓
- Nested subcircuits: TWORES (2×RES in series) + R3 gives proper divider ✓
- Internal nodes correctly isolated between instances ✓

**Tests:** 264 total passing

### Behavioral Sources (B Elements)

Implemented behavioral sources (B elements) that allow arbitrary mathematical expressions for voltage and current sources.

**New module (`spicier-devices/src/expression.rs`):**
- `Expr` enum — AST for mathematical expressions
  - `Constant(f64)` — numeric constant with SI suffix support
  - `Voltage { node_pos, node_neg }` — voltage reference V(node) or V(n1, n2)
  - `Current { source_name }` — current reference I(Vsource)
  - `Time` — simulation time variable
  - `BinaryOp { op, left, right }` — +, -, *, /, ^
  - `UnaryOp { op, operand }` — unary minus
  - `Function { name, args }` — math functions
- `EvalContext` — evaluation context with voltage/current/time values
- `parse_expression()` — recursive descent expression parser
- Expression methods:
  - `eval(&ctx)` — evaluates expression with given context
  - `is_nonlinear()` — detects if expression requires Newton-Raphson
  - `is_time_dependent()` — detects if expression uses `time` variable
  - `voltage_nodes()` — returns set of voltage nodes referenced
  - `derivative_voltage(node, &ctx)` — automatic differentiation for Jacobian

**Supported math functions:**
- Trigonometric: sin, cos, tan, asin, acos, atan
- Exponential/log: exp, log, log10, sqrt
- Utility: abs, min, max, if
- Constants: pi

**Behavioral source devices (`spicier-devices/src/behavioral.rs`):**
- `BehavioralVoltageSource` — B element with V=expression
  - Implements Stamp, Element, Stamper traits
  - `stamp_nonlinear()` evaluates expression at current solution
  - `stamp_at_time()` evaluates expression at specific time
  - Adds branch current variable like regular voltage source
- `BehavioralCurrentSource` — B element with I=expression
  - Implements Stamp, Element, Stamper traits
  - `stamp_nonlinear()` includes Jacobian contributions for Newton-Raphson
  - `stamp_at_time()` evaluates expression at specific time
  - No branch current variable (like regular current source)

**Lexer extensions (`spicier-parser/src/lexer.rs`):**
- Added `Star`, `Slash`, `Caret` tokens for expression operators (*, /, ^)
- Non-comment `*` now lexes as Star token

**Parser support (`spicier-parser/src/parser.rs`):**
- `parse_behavioral()` — parses B element lines
- Detects V= or I= at start of expression (not mid-expression V(node) references)
- Collects expression tokens and delegates to expression parser

**Syntax examples:**
```spice
* Voltage-controlled voltage source (2x gain)
B1 out 0 V=V(in)*2

* Nonlinear resistor (acts like 1k resistor)
B2 1 2 I=V(1,2)/1k

* Time-varying source
B3 out 0 V=sin(2*pi*1k*time)

* Quadratic function (nonlinear)
B4 out 0 V=V(in)*V(in)
```

**Nonlinearity detection:**
- `V(node)*constant` — linear (scaling), no NR needed
- `V(node)*V(node)` — nonlinear (quadratic), NR required
- `sin(V(node))` — nonlinear (transcendental), NR required
- `V(node)/constant` — linear (division by constant), no NR needed
- `time*constant` — linear in time, evaluated per timestep

**Tests:**
- Expression parsing: constants, operations, functions, precedence
- Derivative computation for Newton-Raphson
- Behavioral source stamping (voltage and current)
- Time-dependent expression evaluation
- Parser tests for B element syntax

**Tests:** 297 total passing (33 new behavioral/expression tests)
