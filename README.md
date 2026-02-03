# Spicier

**Spicier** is a modern re-implementation of the SPICE circuit simulator, written in **Rust** and designed from the ground up for **SIMD**, **multicore**, and **GPU-accelerated** simulation.

It aims to preserve the spirit and capabilities of classic SPICE—analog, mixed-signal, and behavioral simulation—while rethinking the architecture for contemporary hardware and large-scale computation.

> *Same equations. Much hotter execution.*

---

## Why Spicier?

Classic SPICE simulators (including ngspice) are the result of decades of incremental evolution, layering new capabilities onto architectures originally designed for single-core machines and memory-constrained environments.

Spicier asks a different question:

> *What would SPICE look like if it were designed today?*

Core goals:

* **Performance-first**: Explicit vectorization, parallelism, and GPU backends
* **Modern architecture**: Clear separation of graph construction, device models, and solvers
* **Correctness & safety**: Rust's ownership and type system eliminate entire classes of bugs
* **Extensibility**: Device models, solvers, and backends are pluggable
* **Research-friendly**: Designed to support experimentation with new numerical methods

---

## Design Principles

### 1. Explicit Parallelism

Spicier treats parallelism as a first-class concern:

* SIMD-friendly data layouts
* Batched solves and vectorized device evaluation
* GPU kernels for matrix assembly and linear solves (where appropriate)

### 2. Solver-Centric Architecture

Rather than baking assumptions into the simulator core, Spicier cleanly separates:

* Circuit graph construction
* Device model evaluation
* Nonlinear and linear solver backends

This allows experimentation with:

* Alternative Newton methods
* Matrix-free techniques
* Domain decomposition
* Hardware-specific solvers

### 3. Mixed-Level from the Start

Inspired by SPICE3, CIDER, and XSPICE, Spicier is designed to support:

* Compact device models
* Physics-based / numerical device models
* Behavioral and event-driven components
* Hybrid analog–digital simulation

---

## Scope (Initial)

Planned early focus:

* DC, AC, and transient analysis
* Modified nodal analysis (MNA)
* Core passive devices (R, C, L)
* Nonlinear devices (diodes, MOSFETs—initially compact models)
* Newton–Raphson–based nonlinear solves
* CPU SIMD acceleration
* Experimental GPU backend

Compatibility with existing SPICE netlists is a **goal**, not a constraint.

---

## What Spicier Is *Not*

* A drop-in replacement for ngspice (yet)
* A GUI schematic editor
* A vendor-specific EDA tool
* Afraid to break legacy assumptions when performance or clarity demand it

---

## Status

**Active development — full analysis suite functional with GPU acceleration**

### Core Simulation

* **Circuit Representation**: MNA matrix stamping with automatic ground node handling
* **Device Models**: R, L, C, V, I (passive + sources), Diode (Shockley), MOSFET Level 1 (NMOS/PMOS), JFET, BJT
* **Controlled Sources**: VCVS (E), VCCS (G), CCCS (F), CCVS (H)
* **Coupled Elements**: K (mutual inductance), T (transmission line)
* **Behavioral Sources**: B elements with expression parsing (V(node), I(device), time, math functions)
* **Subcircuits**: Hierarchical .SUBCKT/.ENDS with nested expansion

### Analysis Types

* **DC Operating Point** (.OP) — Newton-Raphson with voltage limiting, source stepping, Gmin stepping
* **Transient** (.TRAN) — Backward Euler, Trapezoidal, TR-BDF2 with adaptive timestep
* **AC Small-Signal** (.AC) — Linear, Decade, Octave sweeps with automatic linearization
* **DC Sweep** (.DC) — Single and nested sweep variables for I-V curves
* **Initial Conditions** (.IC) with UIC option

### Linear Solvers

* **Dense LU** — Real and complex, nalgebra-based
* **Sparse LU** — faer-based, auto-selected for 50+ variables
* **GMRES** — Iterative with Jacobi preconditioner for large systems
* **Accelerate** — Apple's optimized LAPACK on macOS

### GPU & Parallel Acceleration

* **Batched Sweep Solving** — Monte Carlo, corner analysis, parameter sweeps
* **Multiple Backends**:
  - CUDA (NVIDIA GPUs via cuSOLVER)
  - Metal (Apple GPUs via wgpu compute shaders)
  - MPS (Apple Metal Performance Shaders)
  - Accelerate (Apple optimized LAPACK)
  - Faer (high-performance SIMD CPU)
  - Parallel CPU (rayon + Accelerate/nalgebra)
* **GPU-Side RNG** — Hash-based stateless random number generation
* **GPU-Side Statistics** — Parallel reduction for sweep analysis
* **Early Termination** — Convergence tracking for batched Newton-Raphson
* **Memory Optimization** — Warp-aligned layouts, symbolic factorization caching

### Infrastructure

* **Cargo workspace**: 13 crates (core, solver, devices, parser, cli, simd, validate, backend-cpu, backend-cuda, backend-metal, backend-mps, batched-sweep)
* **SPICE Parser**: R, C, L, V, I, D, M, J, Q, K, T, E, G, F, H, B elements; .MODEL; .PARAM; SI suffixes
* **Validation Suite**: 40+ tests against analytical solutions and ngspice
* **~400 tests passing**, clippy clean
* **GitHub Actions CI** (Linux, macOS, Windows)
* **Criterion benchmarks** for all performance-critical paths

### Performance Notes

On Apple Silicon (M3 Ultra), parallel CPU sweeps using Accelerate + rayon outperform GPU for batched LU solving by 3-13x due to:
- f64→f32 conversion overhead (Metal lacks native f64)
- Excellent CPU LAPACK optimization on Apple Silicon
- LU factorization's inherently sequential pivot selection

GPU acceleration remains beneficial for:
- NVIDIA GPUs with native f64 support
- Device evaluation (embarrassingly parallel)
- Statistics and reduction operations
- Platforms without fast CPU LAPACK

---

## Why Rust?

Rust enables:

* Zero-cost abstractions
* Safe parallelism
* Explicit control over memory layout
* Long-term maintainability for a complex numerical codebase

In short: Rust makes it possible to build a **fast SPICE** without inheriting decades of technical debt.

---

## Inspiration

Spicier draws inspiration from:

* Berkeley SPICE (SPICE3f5)
* CIDER (mixed-level device simulation)
* XSPICE (behavioral and event-driven modeling)
* Modern HPC and GPU programming models

This project stands on the shoulders of giants—then adds vector lanes.

---

## Competitive Landscape

A small but growing ecosystem of Rust-based circuit simulation projects already exists. Spicier builds on lessons from these efforts while deliberately targeting a different design point.

### Spice21

Spice21 is the most mature and principled "modern SPICE" project in Rust. It takes a **library-first** approach, emphasizes clean APIs and schemas, and intentionally avoids strict compatibility with legacy SPICE netlists. Spice21 is well-suited for:

* Embedded simulation workflows
* Programmatic circuit generation
* Research and tooling that benefits from structured, schema-driven I/O

**How Spicier differs:**

* Spicier is **performance-first**, explicitly targeting SIMD, multicore, and GPU acceleration from day one.
* Spicier prioritizes **large-scale numerical throughput** and solver performance over API elegance alone.
* Spicier aims to remain closer to traditional SPICE workflows (netlists, analyses, solver semantics), even when rearchitected internally.
* Spicier is designed around **hardware-aware execution models**, including vectorized device evaluation and accelerator-friendly matrix assembly.

In short:

> *Spice21 modernizes SPICE as a software library.*
> *Spicier modernizes SPICE as a numerical engine.*

These goals are complementary, not competitive.

### Other Rust-Based SPICE Efforts

* **krets** – A practical Rust SPICE-like simulator with a TOML-based workflow and broad analysis ambitions. Demonstrates viable end-to-end simulation in Rust, but does not focus on accelerator-driven performance.
* **tiny-spice-rs / ftspice / similar projects** – Educational or experimental implementations useful for understanding MNA and netlist parsing, but not intended for large or high-performance simulations.
* **ngspice bindings and frontends** – Several projects wrap or embed ngspice via FFI. These provide Rust ergonomics but inherit ngspice's legacy architecture and performance limits.

Spicier is a **ground-up reimplementation**, not a wrapper.

### Why Another Simulator?

The motivation for Spicier is not dissatisfaction with correctness or features in existing tools, but with **architectural constraints**:

* Legacy SPICE engines predate SIMD, GPUs, and modern memory hierarchies
* Parallelism is often implicit or bolted on, not fundamental
* Device evaluation, matrix assembly, and solving are tightly coupled
* Experimentation with new numerical methods is difficult

Spicier exists to explore what becomes possible when these constraints are removed.

---

## Roadmap

### Completed Phases

#### Phase 0-6 — Core Simulation ✅

* Circuit graph and MNA representation
* Device models (R, L, C, V, I, Diode, MOSFET, controlled sources, B elements)
* Newton-Raphson nonlinear solver with convergence aids
* DC, AC, Transient, DC Sweep analysis
* SPICE netlist parser with subcircuit support
* Dense and sparse linear solvers

#### Phase 7-8 — CPU Performance ✅

* SIMD-accelerated operations (AVX-512/AVX2 with runtime detection)
* Batched device evaluation (SoA layout)
* Sparse solver auto-selection (faer LU for 50+ nodes)
* Apple Accelerate integration for macOS
* Parallel CPU sweeps with rayon

#### Phase 9a-9b — GPU Acceleration ✅

* Multi-backend batched sweep solving (CUDA, Metal, MPS, Accelerate, Faer)
* GPU-side random number generation (hash-based, stateless)
* GPU-side statistics (parallel reduction)
* Memory layout optimization (warp-aligned padding)
* Shared sparsity structure caching
* Convergence tracking for early termination
* Pipelined assembly/solve execution

### In Progress

#### Phase 9c — GPU-Native Parallel Sweeps

* GPU device evaluation kernels (MOSFET, Diode, BJT)
* GPU matrix assembly with parallel stamping
* Batched GMRES iterative solver
* Full Newton-Raphson loop on GPU

#### Phase 10 — Cross-Simulator Validation ✅

* JSON golden data infrastructure
* Automated ngspice comparison
* 40+ validation tests

#### Phase 11 — Release Preparation

* API stabilization
* Documentation
* Performance benchmarks
* crates.io publication

### Future

#### Phase 12 — Extended Features ✅

* Noise analysis (.NOISE)
* Additional device models (BJT, JFET)
* Mutual inductance (K element)
* Transmission lines (T element — lumped LC model)
* .PARAM / .MEASURE support

---

## Design Philosophy

> Spicier treats SPICE not as a legacy program to be preserved, but as a numerical workload to be accelerated.

---

## License

TBD (likely permissive: BSD/MIT/Apache-style)

---

## Contributing

Contributions, discussions, and ideas are welcome—especially around:

* Numerical methods
* Performance optimization
* Device modeling
* GPU architectures

If you've ever looked at SPICE source code and thought *"we can do better now"*, this project is for you.

---

**Spicier** — because the equations were never the slow part.
