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

**Active development — full analysis suite functional**

Completed:
* Cargo workspace with 9 crates (core, solver, devices, parser, cli, simd, backend-cpu, backend-cuda, backend-metal)
* Circuit graph representation with MNA matrix stamping
* Device models: R, L, C, V, I (passive + sources) with MNA stamps
* Nonlinear devices: Diode (Shockley equation), MOSFET Level 1 (NMOS/PMOS)
* Controlled sources: VCVS (E), VCCS (G), CCCS (F), CCVS (H) with MNA stamps
* Behavioral sources (B elements) with arbitrary expression parsing (V(node), I(device), time, math functions)
* Dense linear solver (LU decomposition) — real and complex
* Sparse linear solver (faer LU) — real and complex, auto-selected for systems with 50+ variables
* Newton-Raphson nonlinear solver with convergence criteria, voltage limiting, source stepping, Gmin stepping
* Nonlinear DC operating point via Newton-Raphson (auto-dispatched when diodes/MOSFETs present)
* SPICE netlist parser (R, C, L, V, I, D, M, E, G, F, H, B elements; .SUBCKT/.ENDS; .MODEL; SI suffixes)
* DC operating point analysis (.OP) with CLI integration
* Transient analysis (.TRAN) — Backward Euler, Trapezoidal, and TR-BDF2 integration with adaptive timestep
* AC small-signal analysis (.AC) — Linear, Decade, and Octave sweeps with automatic linearization
* DC sweep analysis (.DC) with single and nested sweep variables for I-V curve tracing
* Initial conditions (.IC) with UIC option to skip DC operating point
* .PRINT command support (V, I, VM, VP, VDB, VR, VI output variables)
* SIMD-accelerated dot products and matvec (AVX-512/AVX2/scalar with runtime detection)
* Batched device evaluation for diodes and MOSFETs (SoA layout, AVX2 kernels)
* GMRES iterative solver with Jacobi preconditioner for large systems
* Compute backend abstraction (CPU, CUDA, Metal) with automatic detection
* 297 tests passing, clippy clean
* GitHub Actions CI (Linux, macOS, Windows), benchmarking infrastructure

In progress:
* GPU-accelerated sparse solve for large circuits
* Batched parameter sweeps on GPU (Monte Carlo, corners)

Expect sharp edges, incomplete features, and rapid iteration.

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

This roadmap emphasizes **differentiation from Spice21** by focusing on performance architecture, solver design, and execution backends.

### Phase 0 — Foundations ✅

* Core circuit graph and MNA representation
* Explicit data layouts designed for vectorization
* Deterministic, testable stamping and assembly
* Clear separation of:
  * Graph construction
  * Device evaluation
  * Solver execution

### Phase 1 — CPU Performance

* SIMD-friendly device model evaluation
* Batched Newton-Raphson iterations
* Parallel matrix assembly
* Pluggable linear solvers (dense + sparse) ✅ (faer sparse LU integrated)
* Performance benchmarking against ngspice

### Phase 2 — GPU Acceleration

* GPU-backed matrix assembly experiments
* Accelerator-friendly nonlinear solve workflows
* Investigation of hybrid CPU/GPU execution
* Focus on throughput for large sweeps and Monte Carlo runs

### Phase 3 — Compatibility Layer

* Pragmatic SPICE netlist ingestion
* Compatibility modes where feasible
* Clear documentation of intentional deviations
* Tooling to compare results against ngspice

### Phase 4 — Advanced Models & Methods

* Physics-based / numerical device models
* Mixed-level simulation hooks
* Solver experimentation (matrix-free, domain decomposition, etc.)
* Research-driven features enabled by the new architecture

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
