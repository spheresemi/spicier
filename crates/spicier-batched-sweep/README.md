# spicier-batched-sweep

Unified batched sweep solving for SPICE circuit simulation with GPU and parallel CPU backends.

This crate provides a common API for high-performance batched LU solving:
- **Parallel CPU** - Rayon-based parallel sweeps with Accelerate/LAPACK (recommended)
- **CUDA** - NVIDIA GPUs via cuBLAS batched LU operations
- **Metal** - Apple Silicon GPUs via wgpu compute shaders
- **Faer** - High-performance single-threaded CPU with SIMD and sparse solvers
- **CPU** - Fallback using nalgebra's LU decomposition

## Features

- Unified `BatchedLuSolver` trait for backend abstraction
- Automatic backend detection and selection
- **Parallel CPU sweeps with rayon** for embarrassingly parallel workloads
- Sparse solver caching for repeated sweeps with same sparsity pattern
- GPU-side statistics computation (min, max, mean, variance)
- Convergence tracking with early termination

## Usage

### Parallel CPU Sweeps (Recommended)

```rust
use spicier_batched_sweep::{
    solve_batched_sweep_parallel, ParallelSweepConfig, BackendSelector,
};

// Configure parallel execution
let config = ParallelSweepConfig::default();  // auto-tune based on workload

// Run sweep in parallel across CPU cores
let result = solve_batched_sweep_parallel(
    &backend,
    &factory,
    &generator,
    &variations,
    &ConvergenceCriteria::default(),
    &DispatchConfig::default(),
    &config,
)?;

println!("Converged: {}/{}", result.converged_count, result.total_count);
```

### GPU Backend Selection

```rust
use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};

// Automatic backend selection (CUDA → MPS → Metal → Accelerate → CPU)
let backend = BackendSelector::auto();

// Or prefer a specific backend
// let backend = BackendSelector::prefer_cuda();
// let backend = BackendSelector::prefer_metal();
// let backend = BackendSelector::cpu_only();

let result = solve_batched_sweep_gpu(&backend, &factory, /* ... */)?;
println!("Backend used: {}", result.backend_used);
```

## Performance Analysis

### Parallel CPU vs GPU (M3 Ultra)

After extensive benchmarking, **parallel CPU significantly outperforms GPU** for batched LU solving on Apple Silicon:

| Config | GPU Total | GPU Per-Matrix | Parallel CPU Per-Matrix | GPU Slowdown |
|--------|-----------|----------------|------------------------|--------------|
| 50×50, 1k | 12.9ms | 12.9µs | ~1µs | **13x slower** |
| 50×50, 5k | 42.4ms | 8.5µs | ~1µs | **8x slower** |
| 50×50, 10k | 72.3ms | 7.2µs | ~1µs | **7x slower** |
| 100×100, 1k | 50.8ms | 50.8µs | ~11µs | **5x slower** |
| 100×100, 2k | 66.5ms | 33.3µs | ~11µs | **3x slower** |

### Why GPU is Slower for Batched LU

1. **f64→f32 conversion** - Metal lacks native f64 support, requiring CPU conversion before upload
2. **Data transfer latency** - Even with unified memory, explicit buffer writes have latency
3. **Sequential algorithm** - LU factorization with pivot selection is inherently sequential; GPU can't parallelize within a single matrix well
4. **Excellent CPU performance** - Apple Accelerate + rayon achieves near-linear scaling across P-cores

### Parallel CPU Speedup (vs Sequential)

| Config | Sequential | Parallel (8 threads) | Speedup |
|--------|------------|---------------------|---------|
| 50×50, 1k | 1.1ms | 167µs | **6.6x** |
| 50×50, 5k | 5.5ms | 744µs | **7.4x** |
| 100×100, 1k | 11ms | 1.6ms | **6.9x** |
| 100×100, 2k | 22ms | 3.5ms | **6.3x** |

### Sparse Solver Benefits

For circuits with sparse matrices (typical of most real circuits), the sparse cached solver provides additional speedup:

| Config | Dense (faer) | Sparse (cached) | Speedup |
|--------|-------------|-----------------|---------|
| 50×50 sparse, 100 batches | 995µs | 536µs | **1.86x** |
| 50×50 sparse, 1k batches | 10.27ms | 5.69ms | **1.80x** |

### Recommendations

1. **Use parallel CPU** (`parallel` feature) for batched sweeps on Apple Silicon
2. **Enable sparse solver** for circuits with sparse connectivity (>50 nodes)
3. **Use GPU** only for truly data-parallel operations:
   - Device evaluation (thousands of independent MOSFETs)
   - Statistics reduction (mean, variance across sweep points)
4. **CUDA on NVIDIA** may still be beneficial with cuBLAS optimized batched routines

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Enable rayon-based parallel CPU sweeps (recommended) |
| `cuda`  | Enable NVIDIA CUDA backend |
| `metal` | Enable Apple Metal backend |
| `faer`  | Enable Faer high-performance CPU backend |
| `accelerate` | Enable Apple Accelerate framework (macOS only) |

## Backend Selection Logic

`BackendSelector::auto()` chooses backends in this priority order:
1. CUDA (if available and `cuda` feature enabled)
2. MPS (Apple Metal Performance Shaders, if available)
3. Metal (wgpu compute shaders, if available)
4. Accelerate (macOS native LAPACK)
5. Faer (cross-platform SIMD)
6. CPU (nalgebra fallback)

However, for most workloads, **parallel CPU with Accelerate** provides the best performance.

## License

MIT OR Apache-2.0
