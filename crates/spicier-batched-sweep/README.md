# spicier-batched-sweep

Unified GPU-accelerated batched sweep solving for CUDA and Metal backends.

## Features

This crate provides efficient parallel solving for:
- Monte Carlo analysis (statistical sampling)
- Corner analysis (worst-case combinations)
- Parameter sweeps (systematic exploration)

The batched solver exploits the fact that sweep/Monte Carlo simulations produce
many matrices sharing the same sparsity pattern, differing only in values.

## GPU Backends

- `cuda` - NVIDIA GPUs via cuBLAS batched LU
- `metal` - Apple GPUs via Metal Performance Shaders

## Usage

```rust,ignore
use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};

let backend = BackendSelector::auto();
let result = solve_batched_sweep_gpu(
    &backend,
    &factory,
    &generator,
    &variations,
    &criteria,
    &config,
)?;

println!("GPU used: {}", result.used_gpu);
println!("Mean V(1): {}", result.inner.statistics(0).mean);
```

## Performance

The batched solver provides speedup when:
- Matrix size >= 64 (smaller matrices stay on CPU)
- Batch size >= 16 (overhead amortization)
- GPU is available and properly configured

Automatic fallback to CPU when GPU is unavailable or for small problems.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
