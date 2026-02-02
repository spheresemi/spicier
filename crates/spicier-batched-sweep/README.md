# spicier-batched-sweep

Unified GPU-accelerated batched sweep solving for SPICE circuit simulation.

This crate provides a common API for GPU-accelerated batched LU solving across different backends:
- **CUDA** - NVIDIA GPUs via cuBLAS batched LU operations
- **Metal** - Apple Silicon GPUs (M1/M2/M3) - *GPU acceleration pending*
- **CPU** - Fallback using nalgebra's LU decomposition

## Features

- Unified `BatchedLuSolver` trait for backend abstraction
- Automatic backend detection and selection
- Graceful fallback when preferred GPU unavailable
- Efficient parallel solving for Monte Carlo, corner analysis, and parameter sweeps

## Usage

```rust
use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};
use spicier_solver::{
    ConvergenceCriteria, DispatchConfig, MonteCarloGenerator, ParameterVariation,
};

// Automatic backend selection (CUDA → Metal → CPU)
let backend = BackendSelector::auto();

// Or prefer a specific backend
// let backend = BackendSelector::prefer_cuda();
// let backend = BackendSelector::prefer_metal();
// let backend = BackendSelector::cpu_only();

let result = solve_batched_sweep_gpu(
    &backend,
    &factory,
    &generator,
    &variations,
    &ConvergenceCriteria::default(),
    &DispatchConfig::default(),
)?;

println!("Backend used: {}", result.backend_used);
println!("Converged: {}/{}", result.converged_count, result.total_count);
```

## Benchmark Results

Benchmarked on Mac Studio M3 Ultra (96GB RAM):

| Batch × Matrix | CPU Time | Metal Time* | Notes |
|----------------|----------|-------------|-------|
| 100 × 10       | 65 µs    | 70 µs       | Small problems, CPU overhead dominates |
| 100 × 50       | 1.25 ms  | 1.25 ms     | |
| 100 × 100      | 7.4 ms   | 7.9 ms      | |
| 500 × 10       | 327 µs   | 331 µs      | |
| 500 × 50       | 6.3 ms   | 6.3 ms      | |
| 500 × 100      | 37 ms    | 37 ms       | |
| 1000 × 10      | 651 µs   | 664 µs      | |
| 1000 × 50      | 12.6 ms  | 12.7 ms     | |
| 1000 × 100     | 76 ms    | 75 ms       | |

*\*Metal backend currently falls back to CPU. True GPU acceleration via Metal Performance Shaders is planned.*

### CUDA Performance (on NVIDIA GPUs)

The CUDA backend uses cuBLAS batched LU operations for significant speedups on large batches:
- `cublasDgetrfBatched` - Batched LU factorization
- `cublasDgetrsBatched` - Batched triangular solve

Expected speedups of 10-100x on NVIDIA GPUs for large batches (>100) with medium-sized matrices (>32×32).

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cuda`  | Enable NVIDIA CUDA backend |
| `metal` | Enable Apple Metal backend |

## Thresholds

GPU acceleration is automatically used when:
- Batch size ≥ 16 (below this, CPU overhead dominates)
- Matrix size ≥ 32×32 (below this, CPU LU is competitive)

## License

MIT OR Apache-2.0
