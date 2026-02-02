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

## Phase 8b: Apple Accelerate Framework Integration

**Goal:** Leverage Apple's Accelerate framework for optimal performance on macOS/Apple Silicon, using AMX coprocessor for BLAS/LAPACK operations.

**Status:** ✅ Complete. All SIMD operations (real/complex dot products, matvec) and dense LU use Accelerate. Threshold auto-tuned.

### 8b-1: Accelerate SIMD Backend (spicier-simd) ✅

Apple Silicon falls back to scalar code - missing AMX coprocessor entirely. Add Accelerate dispatch.

- [x] `cblas_ddot` for real dot products
  - Runtime detection: Accelerate available → use cblas_ddot, else scalar
  - `SimdCapability::Accelerate` variant added
- [x] `cblas_dgemv` for dense matvec
- [x] `cblas_zdotu_sub` for complex dot products (unconjugated)
- [x] `cblas_zdotc_sub` for complex conjugate dot products
- [x] `cblas_zgemv` for complex matvec
- [x] Feature flag: `accelerate` (default on macOS)
- [ ] Benchmark comparison: scalar vs Accelerate (GMRES performance)

### 8b-2: Accelerate Linear Solver (spicier-solver) ✅

Replace nalgebra dense LU with Accelerate's optimized LAPACK.

- [x] Direct FFI to `dgesv_` for dense solve (< SPARSE_THRESHOLD)
  - 1.86x speedup at n=100, **8.5x speedup at n=500**
- [x] `zgesv_` for complex dense solve (AC analysis)
- [x] Feature flag: `accelerate` in spicier-solver (default)
- [x] `dgetrf_` + `dgetrs_` for factorization reuse in Newton-Raphson
  - `CachedDenseLu` struct stores LU factors and pivot indices
  - Factor once with `new()`, solve many times with `solve()`
  - 8 unit tests verify correctness and reuse pattern
- [x] `zgetrf_` + `zgetrs_` for complex factorization reuse
  - `CachedDenseLuComplex` for AC analysis and complex systems
  - Same factorize-once, solve-many pattern

### 8b-3: Sparse-to-Dense Threshold Tuning ✅

- [x] Profile sparse vs dense+Accelerate crossover point
  - Benchmarked: Dense+Accelerate beats sparse faer up to n≈100
  - `SPARSE_THRESHOLD` now 100 with Accelerate, 50 without
- [x] Adaptive threshold based on backend availability
  - Conditional compilation: `#[cfg(all(target_os = "macos", feature = "accelerate"))]`

**Benchmark Results (banded matrix):**
| Size | Dense (Accelerate) | Sparse (faer) | Winner |
|------|-------------------|---------------|--------|
| n=10 | 401 ns | 3.3 µs | Dense 8x faster |
| n=50 | 7.8 µs | 14.8 µs | Dense 1.9x faster |
| n=100 | 29 µs | 29 µs | Tie |
| n=500 | 884 µs | 147 µs | Sparse 6x faster |

### Measured Performance Gains (Dense LU Solve)

| Matrix Size | nalgebra | Accelerate | Speedup |
|-------------|----------|------------|---------|
| n=10 | 327 ns | 400 ns | 0.82x (overhead) |
| n=50 | 8.5 µs | 7.8 µs | 1.09x |
| n=100 | 54 µs | 29 µs | **1.86x** |
| n=500 | 7.8 ms | 919 µs | **8.5x** |

Crossover point: ~n=50. Accelerate wins big for medium/large circuits.

### Batched Sweep Performance (spicier-batched-sweep)

| Backend | Time (1000×100) | vs nalgebra |
|---------|----------------|-------------|
| nalgebra | 74.8 ms | baseline |
| Accelerate | 52.2 ms | **1.43x faster** |

**Dependencies:** Phase 8 (SIMD infrastructure)

**Acceptance Criteria:**
- [x] SIMD dot products use Accelerate on macOS, scalar elsewhere
- [x] Dense LU uses Accelerate when available
- [x] Benchmark shows 2-8x speedup on Apple Silicon vs nalgebra
- [x] All tests pass on both macOS (Accelerate) and Linux (fallback)

---

## Phase 8c: Parallel CPU Sweeps (Rayon) ✅

**Goal:** Leverage multi-core parallelism for batched sweeps using rayon. Each sweep point is independent, making this embarrassingly parallel.

**Rationale:** On Apple Silicon, Accelerate is very fast but single-threaded per call. Parallel iteration over sweep points using rayon can provide 8-16x speedup on M3 Ultra (8 P-cores) without GPU complexity.

### Tasks

- [x] Add rayon dependency to spicier-batched-sweep
  - Optional `parallel` feature flag
  - `rayon = "1.10"` in workspace dependencies
- [x] Implement parallel sweep iteration
  - `solve_batched_sweep_parallel()` using `par_iter()` over sweep points
  - Each thread creates its own solver instance and processes independently
  - `ParallelSweepConfig` for tuning (min parallel threshold, chunk size)
- [x] Chunk size tuning
  - Optional `chunk_size` parameter for batched chunk processing
  - Default uses rayon's work-stealing for optimal distribution
- [x] Thread-safe result collection
  - Pre-allocated result buffer
  - Each thread returns results, collected into final vector

**Implementation:** `crates/spicier-batched-sweep/src/parallel.rs`

**Benchmark Results (M3 Ultra):**

| Config | Sequential | Parallel | Speedup |
|--------|-----------|----------|---------|
| 100×50 | 1.25 ms | 0.41 ms | **3.1x** |
| 500×50 | 6.30 ms | 0.80 ms | **7.9x** |
| 1000×50 | 12.5 ms | 1.69 ms | **7.4x** |
| 1000×100 | 47.8 ms | 12.3 ms | **3.9x** |
| 2000×50 | 25.4 ms | 1.99 ms | **12.8x** |
| 2000×100 | 95.6 ms | 22.0 ms | **4.3x** |

**Key findings:**
- 7-13x speedup for smaller matrices (50×50) where parallelism dominates
- 4x speedup for larger matrices (100×100) where memory bandwidth limits scaling
- Near-linear scaling with sweep point count

**Dependencies:** Phase 8b (Accelerate integration)

**Acceptance Criteria:**
- [x] 1000×100 sweep <15ms on M3 Ultra (achieved 12.3ms)
- [x] Near-linear scaling up to physical core count
- [x] No correctness regressions (test verifies sequential == parallel)

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

### 9b: Batched Sweep Solver (GPU)

**Current Status:** Metal GPU implementation has ~10ms overhead that makes it slower than Accelerate for tested matrix sizes. Benchmarks on M3 Ultra:

| Config | Accelerate | Metal GPU | Gap |
|--------|------------|-----------|-----|
| 1000×10 | 728 µs | 11 ms | Metal 15x slower |
| 1000×50 | 13.7 ms | 24 ms | Metal 1.8x slower |
| 1000×100 | 50 ms | 77 ms | Metal 1.5x slower |

**Root Cause:** Per-call buffer allocation, synchronous completion, f64↔f32 conversion overhead.

**Goal:** Reduce GPU overhead to find the crossover point where GPU beats CPU. Expected wins:
- Very large batches (>10k sweep points) where overhead is amortized
- Larger matrices (>256×256) where GPU parallelism dominates
- NVIDIA CUDA should have lower overhead than Metal/wgpu

**✅ Completed - Backend Infrastructure:**

- [x] `spicier-batched-sweep` crate with unified API
  - `BatchedLuSolver` trait for backend-agnostic batched solves
  - `BackendSelector` with auto-detection (CUDA → MPS → Metal → Accelerate → Faer → CPU)
  - `GpuBatchConfig` with size thresholds for GPU dispatch
- [x] CUDA backend (`spicier-backend-cuda`)
  - cuSOLVER batched getrf/getrs for medium matrices
  - cuBLAS batched operations
- [x] Metal backends
  - [x] wgpu/Metal compute shaders (`spicier-backend-metal`) - custom WGSL kernels
  - [x] MPS backend (`spicier-backend-mps`) - Apple's optimized kernels
- [x] High-performance CPU fallback
  - Faer backend with SIMD-optimized LU
  - Accelerate backend for macOS (LAPACK)

---

#### 9b-1: Metal GPU Overhead Analysis ✅

Investigated Metal GPU overhead and found it is not competitive with parallel CPU for batched LU.

**Implemented optimizations:**
- [x] Buffer pool / reuse
  - `CachedBuffers` struct caches allocated buffers
  - 2x headroom for future growth without reallocation
  - Bind groups cached when dimensions unchanged
- [x] Benchmark at larger scales
  - Tested up to 10k batch sizes and 100×100 matrices
  - Added timing instrumentation to measure breakdown

**Benchmark Results (M3 Ultra):**

| Config | GPU Total | GPU Per-Matrix | Parallel CPU Per-Matrix | GPU Slowdown |
|--------|-----------|----------------|------------------------|--------------|
| 50×50, 1k | 12.9ms | 12.9µs | ~1µs | **13x slower** |
| 50×50, 5k | 42.4ms | 8.5µs | ~1µs | **8x slower** |
| 50×50, 10k | 72.3ms | 7.2µs | ~1µs | **7x slower** |
| 100×100, 1k | 50.8ms | 50.8µs | ~11µs | **5x slower** |
| 100×100, 2k | 66.5ms | 33.3µs | ~11µs | **3x slower** |

**Root causes (not fixable without major architecture changes):**

1. **f64→f32 conversion overhead** - Required because Metal lacks native f64 support. For 10k × 50×50: 25M f64 values converted on CPU before upload.

2. **Data transfer latency** - Even with unified memory on Apple Silicon, explicit buffer writes have latency. Each batch uploads ~10MB+ of matrix data.

3. **Suboptimal GPU algorithm** - LU factorization with pivot selection is inherently sequential. Each pivot step depends on the previous. GPU workgroups can't parallelize within a single matrix well.

4. **Parallel CPU is extremely fast** - Accelerate + rayon achieves near-linear scaling across P-cores. 8 threads × highly optimized LAPACK = hard to beat.

**Conclusion:** For batched linear solves on Apple Silicon, parallel CPU (Accelerate + rayon) significantly outperforms GPU. GPU acceleration would only be beneficial for:

- Operations that are truly data-parallel (device evaluation, statistics)
- Platforms without fast CPU LAPACK (embedded, WASM)
- Very large matrices (>1000×1000) where GPU parallelism dominates - but these exceed wgpu buffer limits

**Recommendation:** Deprioritize GPU batched LU. Focus on:
- Parallel CPU sweeps (Phase 8c ✅)
- GPU for device evaluation (Phase 9c)
- GPU for statistics/reduction (Phase 9b-7 ✅)

---

#### 9b-2: MPS Batched Operations ✅

MPS's `MPSMatrixDecompositionLU` doesn't support true matrix batching (the `matrices` parameter in `MPSMatrixDescriptor` is for storage, not parallel execution). However, parallel command buffer submission provides GPU overlap.

- [x] Investigate MPS batch dimension behavior
  - `MPSMatrixDescriptor` batched descriptors tested - kernel still operates on single matrices
  - MPS LU kernels initialized with rows/columns only, no batch count
- [x] Parallel command buffer submission
  - Commit all command buffers BEFORE waiting on any
  - GPU scheduler can overlap execution where possible
  - All 5 MPS backend tests pass
- [ ] Compare with wgpu custom shaders after 9b-1 optimization
  - wgpu shaders can implement true batched LU with parallel thread dispatch

**Acceptance:** MPS processes N matrices in parallel, not sequentially.

---

#### 9b-3: Pipelined Assembly + Solve

Hide CPU matrix assembly latency behind GPU computation.

- [ ] Double-buffered batch processing
  - Buffer A: GPU solving batch K
  - Buffer B: CPU assembling batch K+1
  - Swap on completion
- [ ] Async GPU dispatch
  - Non-blocking solve submission
  - Completion callback triggers next batch

**Acceptance:** Pipeline hides >50% of assembly latency.

---

#### 9b-4: Memory Layout Optimization ✅

Optimize data layout for GPU memory access patterns.

- [x] Contiguous batch storage
  - All matrices packed into single buffer (batch × n × n)
  - All RHS vectors packed into single buffer (batch × n)
- [x] Alignment padding
  - `BatchLayout` aligns matrix rows to warp size (32) for coalesced access
  - `pack_matrices_f32()` handles f64→f32 conversion + col-major→row-major transpose + padding
  - WGSL shader updated with `row_stride` and `matrix_stride` uniforms

**Implementation:**
- `spicier-batched-sweep/src/batch_layout.rs` - Layout utilities with alignment calculations
- `spicier-backend-metal/src/batch_layout.rs` - Backend-specific layout utilities
- `spicier-backend-metal/src/batched_lu.rs` - Updated to use aligned layout
- `spicier-backend-metal/src/batched_lu.wgsl` - Stride-based matrix access

---

#### 9b-5: Shared Sparsity Structure ✅

Sweep matrices share the same sparsity pattern, only values differ. Benefits both CPU and GPU paths.

- [x] Symbolic structure caching for sparse sweeps
  - `FaerSparseCachedBatchedSolver` with `RwLock<CachedSymbolic>` for thread-safe caching
  - Compute sparsity pattern once from first sweep point
  - Reuse `SymbolicLu` for subsequent batches
  - `reset_cache()` method when topology changes
- [x] `FaerTripletBatchedSolver` for direct triplet input
  - Accepts `Vec<(row, col, value)>` directly from stamper
  - Avoids dense→sparse conversion overhead
  - Integrates with `SweepStamper::stamp_triplets()` method
- [x] Value-only updates
  - Only update changed matrix values per sweep point
  - Symbolic factorization reused across batches

**Implementation:** `crates/spicier-batched-sweep/src/faer_sparse_solver.rs`

**Benchmark Results (M3 Ultra, tridiagonal matrices):**

| Config | Dense (faer) | Sparse (cold) | Sparse (warm) | Speedup |
|--------|-------------|---------------|---------------|---------|
| 100 batches, 20×20 | 175µs | 283µs | 249µs | 0.70x (sparse slower) |
| 100 batches, 50×50 | 995µs | 601µs | 536µs | **1.86x faster** |
| 500 batches, 50×50 | 5.08ms | 3.01ms | 2.76ms | **1.84x faster** |
| 1000 batches, 50×50 | 10.27ms | 6.09ms | 5.69ms | **1.80x faster** |

**Key findings:**
- For small matrices (20×20), sparse overhead exceeds benefits
- For larger sparse matrices (≥50×50), sparse solver is **1.8x faster**
- Warm cache (symbolic reuse) provides ~10% additional speedup

**Acceptance:** ✅ Sweep of 1000 points reuses symbolic factorization. Tests verify caching works across multiple `solve_batch()` calls.

---

#### 9b-6: GPU-Side Random Number Generation ✅

Implemented hash-based RNG for GPU-parallel Monte Carlo.

- [x] Hash-based stateless RNG (`rng.rs`)
  - SplitMix64 hash function for mixing
  - `uniform(seed, sweep_idx, param_idx)` returns [0, 1)
  - `gaussian(seed, sweep_idx, param_idx)` via Box-Muller transform
  - f64 and f32 versions
- [x] GPU shader code ready for integration
  - `WGSL_RNG_CODE` for Metal/WebGPU compute shaders
  - `CUDA_RNG_CODE` for NVIDIA kernels
- [x] Seeded reproducibility
  - Same seed → deterministic sequence
  - Each (sweep_idx, param_idx) gets unique value
- [x] Batch generation helper
  - `generate_gaussian_parameters(seed, num_sweeps, means, sigmas)`

**Acceptance:** ✅ 10 tests pass, GPU shader code ready.

---

#### 9b-7: GPU-Side Statistics ✅

Compute sweep statistics without CPU round-trip.

- [x] Reduction kernels (sum, min, max, mean, variance)
  - `StatisticsAccumulator` with mergeable partial results for parallel reduction
  - `SweepStatistics` with count, min, max, mean, variance, std_dev
  - `compute_statistics()` / `compute_all_statistics()` for sweep results
- [x] Histogram computation
  - `Histogram` with configurable bins, auto min/max detection
  - `Histogram::from_sweep()` for direct sweep analysis
  - Mode detection, bin percentages
- [x] Yield analysis
  - `YieldSpec` for single-node pass/fail checking
  - `YieldAnalysis` for multi-specification yield
  - `SweepSummary` combining statistics + yield

**GPU shader code:**
- `WGSL_REDUCTION_CODE` - Tree-based parallel reduction for Metal/WebGPU
- `CUDA_REDUCTION_CODE` - Warp shuffle + block reduction for NVIDIA

**Implementation:** `spicier-batched-sweep/src/statistics.rs`

---

#### 9b-8: Early Termination for Converged Points ✅

Don't waste compute on already-converged sweep points.

- [x] Per-point convergence tracking
  - `ConvergenceStatus` enum: Active, Converged, Failed, Singular
  - `ConvergenceTracker` manages per-point status and iteration counts
  - Updates after each NR iteration
- [x] Compaction or masking
  - Both strategies implemented:
    - `compact_active()` / `expand_active()` for reducing batch size
    - `active_mask_u32()` for GPU masking (simpler, some wasted compute)
- [x] Iteration count limits
  - Per-point iteration counter
  - Mark as failed if limit exceeded

**Implementation:** `crates/spicier-batched-sweep/src/convergence.rs`

---

#### 9b-9: Benchmarking & Documentation

Final validation before release.

- [ ] Comprehensive benchmarks
  - Sweep sizes: 100, 1k, 10k, 100k points
  - Matrix sizes: 10, 50, 100, 500 nodes
  - All backends: CUDA, MPS, Metal, Accelerate, Faer, CPU
- [ ] Performance documentation
  - Speedup charts vs CPU baseline
  - Guidance on when GPU is beneficial
  - Backend selection recommendations
- [ ] README updates
  - GPU acceleration section
  - Backend feature flags
  - Example usage with batched sweeps
- [ ] API documentation
  - `spicier-batched-sweep` rustdoc
  - `BackendSelector` usage examples

**Acceptance:** README and docs show clear GPU performance story.

---

**Float-Float Precision (Contingency):**

If f32 precision causes convergence failures in batched solves:

- [ ] FF-accumulating dot products for residual checks
- [ ] Mixed-precision refinement: f32 solve + FF residual correction
- [ ] Per-sweep-point precision escalation (f32 → FF → f64 fallback)

This is a fallback, not the primary approach. Most circuits solve fine in f32.

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

**Phase 9 Acceptance Criteria:**
- [x] Auto-detection correctly selects CUDA on Linux/Windows with NVIDIA GPU, Metal on macOS, CPU elsewhere
- [x] `--backend=cpu` always works as fallback
- [ ] 10k-point Monte Carlo sweep completes with GPU acceleration
- [ ] Benchmark suite covers all backends with documented speedups
- [ ] All results match CPU reference to within solver tolerance
- [ ] README documents GPU performance characteristics

---

## Phase 10: Validation Test Suite

**Goal:** Import and adapt test circuits from ngspice and spice21 to validate solver accuracy against established SPICE implementations.

**Status:** Core validation infrastructure complete. 30+ validation tests in ngspice_validation.rs, 11 cross-simulator tests in spicier-validate.

### 10a: ngspice Test Import

ngspice provides 113 regression tests with expected outputs and 462 example circuits.

**High-priority imports:**
- [x] Basic circuit validation
  - Voltage divider, current divider, series/parallel resistors
  - Wheatstone bridge (balanced and unbalanced)
  - Superposition principle verification
- [x] Transient validation
  - RC charging (analytical vs simulated)
  - LC oscillation frequency verification
  - RL time constant verification
- [x] AC validation
  - RC low-pass filter -3dB and rolloff
  - RL high-pass filter -3dB
  - RLC series resonance
- [ ] Parser validation (partial)
  - Expression parsing edge cases
- [ ] Additional ngspice regression tests

**Test infrastructure:**
- [x] JSON golden data format (`tests/golden_data/`)
- [x] Toleranced comparison (relative + absolute tolerance)
- [x] `spicier-validate` crate for cross-simulator comparison
- [ ] CI integration for regression testing

### 10b: spice21 Test Adaptation

spice21 has 87 tests with golden data for ring oscillators and device characterization.

- [ ] Ring oscillator validation
- [ ] Device characterization
- [ ] Hierarchical circuit validation

### 10c: Cross-Simulator Comparison

- [x] Run identical circuits through ngspice and spicier
- [x] Compare DC operating points (voltage/current tolerance)
- [x] Compare transient waveforms (RMS error, peak detection)
- [x] Compare AC magnitude/phase (dB tolerance, phase tolerance)
- [x] Document any intentional deviations from ngspice behavior

**Known Issues (documented in tests):**
- ~~Diode model: ~9% voltage difference in some circuits~~ ✅ RESOLVED - matches ngspice within 0.00%
- ~~MOSFET model: ~20% drain voltage difference (W/L parsing issue suspected)~~ ✅ RESOLVED - actual error ~0.6%, W/L parsing correct
- ~~Inductor transient: Matrix dimension mismatch for branch currents~~ ✅ RESOLVED - initial current now extracted from DC solution

**MOSFET refinements for future (not blocking):**
- LD (lateral diffusion) - channel length shortening
- Temperature compensation
- COX-based KP scaling

**Dependencies:** Core analysis types complete (Phases 4-7)

**Acceptance Criteria:**
- [x] 30+ validation tests pass with analytical/golden data
- [x] 11 cross-simulator tests pass (linear circuits match ngspice)
- [x] All existing tests continue to pass
- [x] Nonlinear device model discrepancies resolved (diode ~0%, MOSFET ~0.6%)

---

## Phase 11: crates.io Release Preparation

**Goal:** Prepare spicier crates for publication on crates.io.

**Blocked by:** Phase 9b (GPU batched sweep optimizations must be complete first)

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
- [ ] `cargo publish --dry-run` succeeds for each crate (requires publishing in order)
- [x] Minimum Rust version (MSRV) documented and tested (1.85 for edition 2024)

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

#### K Element — Mutual Inductance ✅

**Parser support:**
- [x] `K<name> L1 L2 <coupling_coefficient>` syntax
- [ ] Multi-winding support: `K<name> L1 L2 L3 ... <k>` (future)
- [x] Coupling coefficient validation (0 < k ≤ 1)
- [x] Reference to existing inductor elements by name

**Device model:**
- [x] `MutualInductance` struct with coupling coefficient k
- [x] Mutual inductance calculation: M = k√(L1·L2)
- [x] Resolution to actual inductor elements via `resolve()`

**DC analysis:**
- [x] Short-circuit model (same as individual inductors)

**AC analysis (small-signal):**
- [x] `AcDeviceInfo::MutualInductance` with inductance values and M
- [ ] Coupled impedance stamping (future)

**Transient analysis:**
- [ ] Companion model for coupled inductors (future)

#### Q Element — BJT ✅

**Parser support:**
- [x] `Q<name> <collector> <base> <emitter> <model>` (NPN/PNP)
- [x] `.MODEL <name> NPN/PNP (parameters...)` parsing
- [x] Model parameters: IS, BF, BR, NF, NR, VAF, RB, RE, RC, CJE, CJC, TF, TR

**Device model (Ebers-Moll):**
- [x] `Bjt` struct with model parameters and operating region
- [x] Forward/reverse currents with exponential model
- [x] Early effect (output conductance): go = Ic/VA + reverse current derivative
- [x] Region detection: cutoff, forward active, reverse active, saturation

**DC analysis (Newton-Raphson):**
- [x] Nonlinear BJT stamper with voltage limiting
- [x] Consistent limiting in stamp_linearized_at for convergence
- [x] Fixed saturation convergence with reverse current derivative in go

**AC analysis (small-signal):**
- [x] Hybrid-π model: gm, gpi, go
- [x] `AcDeviceInfo::Bjt` with small-signal parameters

**Validation tests:**
- [x] Common-emitter amplifier
- [x] Emitter follower
- [x] NPN cutoff, forward active, saturation
- [x] PNP common-emitter
- [x] Early effect test

#### J Element — JFET ✅

**Parser support:**
- [x] `J<name> <drain> <gate> <source> <model>` syntax
- [x] `.MODEL <name> NJF/PJF (parameters...)` parsing
- [x] Shichman-Hodges parameters: VTO, BETA, LAMBDA

**Device model (Shichman-Hodges):**
- [x] `Jfet` struct with model parameters
- [x] Pinch-off voltage VTO
- [x] Drain current equations for cutoff, linear, saturation
- [x] Symmetric JFET handling for drain/source interchange

**DC analysis:**
- [x] Nonlinear JFET stamper with region detection
- [x] Voltage limiting for convergence

**AC analysis (small-signal):**
- [x] gm, gds computation
- [x] `AcDeviceInfo::Jfet` with small-signal parameters

**Validation tests:**
- [x] JFET common-source amplifier
- [x] NJF and PJF polarity tests
- [x] Region detection tests

#### Advanced MOSFET Models

- [ ] BSIM3v3 MOSFET — industry-standard short-channel model
- [ ] BSIM4 MOSFET (partial) — complex model with quantum effects
- [ ] Parameter extraction from foundry model files

#### Transmission Lines

- [ ] Lossless transmission line (T element): delay-based model
- [ ] Lossy transmission line: RLGC model with skin effect
- [ ] Coupled transmission lines: crosstalk modeling

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
- [ ] K element (mutual inductance / coupled inductors / transformers)
- [ ] Q element (BJT — Gummel-Poon model)
- [ ] J element (JFET — Shichman-Hodges model)
- [ ] Additional MOSFET models (BSIM3/BSIM4)
- [ ] Transmission lines (lossless, lossy)

### Long-term / Research
- S-parameter analysis
- Mixed-signal / event-driven simulation (XSPICE-style)
- Physics-based device models (CIDER-style)
- Optimization / tuning (.OPTIM)
- Verilog-A model import
- Distributed simulation for very large circuits
