//! Benchmark comparing CPU vs Faer vs Accelerate vs Metal batched sweep performance.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
#[cfg(any(feature = "metal", feature = "faer", feature = "accelerate"))]
use spicier_batched_sweep::GpuBatchConfig;
use spicier_batched_sweep::{BackendSelector, solve_batched_sweep_gpu};
#[cfg(feature = "faer")]
use spicier_batched_sweep::{BatchedLuSolver, FaerBatchedSolver, FaerSparseCachedBatchedSolver};
#[cfg(feature = "parallel")]
use spicier_batched_sweep::{ParallelSweepConfig, solve_batched_sweep_parallel};
use spicier_solver::{
    ConvergenceCriteria, DispatchConfig, MonteCarloGenerator, ParameterVariation, SweepStamper,
    SweepStamperFactory,
};
use std::sync::Arc;

/// Voltage divider factory for benchmarking.
struct DividerFactory {
    r2_nominal: f64,
    num_nodes: usize,
}

impl DividerFactory {
    fn new(num_nodes: usize) -> Self {
        Self {
            r2_nominal: 1000.0,
            num_nodes,
        }
    }
}

impl SweepStamperFactory for DividerFactory {
    fn create_stamper(&self, parameters: &[f64]) -> Arc<dyn SweepStamper> {
        let r1 = parameters.first().copied().unwrap_or(1000.0);
        Arc::new(DividerStamper {
            r1,
            r2: self.r2_nominal,
            v_source: 10.0,
            num_nodes: self.num_nodes,
        })
    }
}

struct DividerStamper {
    r1: f64,
    r2: f64,
    v_source: f64,
    num_nodes: usize,
}

impl SweepStamper for DividerStamper {
    fn stamp_linear(&self, matrix: &mut nalgebra::DMatrix<f64>, rhs: &mut DVector<f64>) {
        let g1 = 1.0 / self.r1;
        let g2 = 1.0 / self.r2;

        // Build a chain of resistors to create larger matrices
        // Node 0 is voltage source, nodes 1..num_nodes-1 are internal
        for i in 0..self.num_nodes - 1 {
            let g = if i == 0 { g1 } else { g2 };
            matrix[(i, i)] += g;
            matrix[(i + 1, i + 1)] += g;
            matrix[(i, i + 1)] -= g;
            matrix[(i + 1, i)] -= g;
        }

        // Last node to ground
        matrix[(self.num_nodes - 1, self.num_nodes - 1)] += g2;

        // Voltage source at node 0 (current variable at index num_nodes)
        matrix[(0, self.num_nodes)] += 1.0;
        matrix[(self.num_nodes, 0)] += 1.0;
        rhs[self.num_nodes] = self.v_source;
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn num_vsources(&self) -> usize {
        1
    }
}

fn run_cpu_sweep(batch_size: usize, matrix_size: usize) {
    let factory = DividerFactory::new(matrix_size);
    let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
    let variations = vec![
        ParameterVariation::new("R1", 1000.0)
            .with_bounds(500.0, 1500.0)
            .with_sigma(0.1),
    ];
    let criteria = ConvergenceCriteria::default();
    let config = DispatchConfig::default();
    let backend = BackendSelector::cpu_only();

    let _ = solve_batched_sweep_gpu(
        &backend,
        &factory,
        &generator,
        &variations,
        &criteria,
        &config,
    )
    .unwrap();
}

#[cfg(feature = "faer")]
fn run_faer_sweep(batch_size: usize, matrix_size: usize) {
    let factory = DividerFactory::new(matrix_size);
    let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
    let variations = vec![
        ParameterVariation::new("R1", 1000.0)
            .with_bounds(500.0, 1500.0)
            .with_sigma(0.1),
    ];
    let criteria = ConvergenceCriteria::default();
    let config = DispatchConfig::default();
    let backend = BackendSelector::prefer_faer();

    let _ = solve_batched_sweep_gpu(
        &backend,
        &factory,
        &generator,
        &variations,
        &criteria,
        &config,
    )
    .unwrap();
}

#[cfg(feature = "accelerate")]
fn run_accelerate_sweep(batch_size: usize, matrix_size: usize) {
    let factory = DividerFactory::new(matrix_size);
    let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
    let variations = vec![
        ParameterVariation::new("R1", 1000.0)
            .with_bounds(500.0, 1500.0)
            .with_sigma(0.1),
    ];
    let criteria = ConvergenceCriteria::default();
    let config = DispatchConfig::default();
    let backend = BackendSelector::prefer_accelerate();

    let _ = solve_batched_sweep_gpu(
        &backend,
        &factory,
        &generator,
        &variations,
        &criteria,
        &config,
    )
    .unwrap();
}

#[cfg(feature = "metal")]
fn run_metal_sweep(batch_size: usize, matrix_size: usize) {
    let factory = DividerFactory::new(matrix_size);
    let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
    let variations = vec![
        ParameterVariation::new("R1", 1000.0)
            .with_bounds(500.0, 1500.0)
            .with_sigma(0.1),
    ];
    let criteria = ConvergenceCriteria::default();
    let config = DispatchConfig::default();
    let backend = BackendSelector::prefer_metal().with_config(GpuBatchConfig {
        min_batch_size: 1,
        min_matrix_size: 1,
        max_batch_per_launch: 65535,
    });

    let _ = solve_batched_sweep_gpu(
        &backend,
        &factory,
        &generator,
        &variations,
        &criteria,
        &config,
    )
    .unwrap();
}

#[cfg(feature = "mps")]
fn run_mps_sweep(batch_size: usize, matrix_size: usize) {
    let factory = DividerFactory::new(matrix_size);
    let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
    let variations = vec![
        ParameterVariation::new("R1", 1000.0)
            .with_bounds(500.0, 1500.0)
            .with_sigma(0.1),
    ];
    let criteria = ConvergenceCriteria::default();
    let config = DispatchConfig::default();
    let backend = BackendSelector::prefer_mps().with_config(GpuBatchConfig {
        min_batch_size: 1,
        min_matrix_size: 1,
        max_batch_per_launch: 65535,
    });

    let _ = solve_batched_sweep_gpu(
        &backend,
        &factory,
        &generator,
        &variations,
        &criteria,
        &config,
    )
    .unwrap();
}

fn bench_sweep_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_sweep");
    group.sample_size(20);

    // Test different batch sizes and matrix sizes
    let configs = [
        (100, 10),
        (100, 50),
        (100, 100),
        (500, 10),
        (500, 50),
        (500, 100),
        (1000, 10),
        (1000, 50),
        (1000, 100),
    ];

    for (batch_size, matrix_size) in configs {
        let param = format!("{}x{}", batch_size, matrix_size);

        // CPU benchmark (nalgebra)
        group.bench_with_input(
            BenchmarkId::new("CPU-nalgebra", &param),
            &(batch_size, matrix_size),
            |b, &(bs, ms)| b.iter(|| run_cpu_sweep(bs, ms)),
        );

        // Faer benchmark (if available)
        #[cfg(feature = "faer")]
        {
            let faer_backend = BackendSelector::prefer_faer();
            if let Ok(solver) = faer_backend.create_solver() {
                if solver.backend_type() == BackendType::Faer {
                    group.bench_with_input(
                        BenchmarkId::new("CPU-faer", &param),
                        &(batch_size, matrix_size),
                        |b, &(bs, ms)| b.iter(|| run_faer_sweep(bs, ms)),
                    );
                }
            }
        }

        // Accelerate benchmark (if available, macOS only)
        #[cfg(feature = "accelerate")]
        {
            let accel_backend = BackendSelector::prefer_accelerate();
            if let Ok(solver) = accel_backend.create_solver() {
                if solver.backend_type() == BackendType::Accelerate {
                    group.bench_with_input(
                        BenchmarkId::new("CPU-accelerate", &param),
                        &(batch_size, matrix_size),
                        |b, &(bs, ms)| b.iter(|| run_accelerate_sweep(bs, ms)),
                    );
                }
            }
        }

        // Metal benchmark (if available)
        #[cfg(feature = "metal")]
        {
            let metal_backend = BackendSelector::prefer_metal().with_config(GpuBatchConfig {
                min_batch_size: 1,
                min_matrix_size: 1,
                max_batch_per_launch: 65535,
            });

            if let Ok(solver) = metal_backend.create_solver() {
                if solver.backend_type() == BackendType::Metal {
                    group.bench_with_input(
                        BenchmarkId::new("Metal", &param),
                        &(batch_size, matrix_size),
                        |b, &(bs, ms)| b.iter(|| run_metal_sweep(bs, ms)),
                    );
                }
            }
        }

        // MPS benchmark (if available, macOS only with Apple Silicon)
        #[cfg(feature = "mps")]
        {
            let mps_backend = BackendSelector::prefer_mps().with_config(GpuBatchConfig {
                min_batch_size: 1,
                min_matrix_size: 1,
                max_batch_per_launch: 65535,
            });

            if let Ok(solver) = mps_backend.create_solver() {
                if solver.backend_type() == BackendType::Mps {
                    group.bench_with_input(
                        BenchmarkId::new("MPS", &param),
                        &(batch_size, matrix_size),
                        |b, &(bs, ms)| b.iter(|| run_mps_sweep(bs, ms)),
                    );
                }
            }
        }
    }

    group.finish();
}

/// Benchmark comparing dense vs sparse cached faer solvers directly.
///
/// This benchmark isolates the LU solve operation to measure the benefit of
/// symbolic factorization caching in sparse matrices.
#[cfg(feature = "faer")]
fn bench_dense_vs_sparse_cached(c: &mut Criterion) {
    let mut group = c.benchmark_group("faer_dense_vs_sparse_cached");
    group.sample_size(50);

    // Test configurations: (batch_size, matrix_size, sparsity_percent)
    // Sparsity matters - sparse solver benefits when matrices are sparse
    let configs = [
        (100, 20),
        (100, 50),
        (500, 20),
        (500, 50),
        (1000, 20),
        (1000, 50),
    ];

    for (batch_size, n) in configs {
        let param = format!("{}batches_{}x{}", batch_size, n, n);

        // Generate diagonally-dominant sparse matrices
        // Tridiagonal pattern (very sparse) with some fill
        let mut matrices = Vec::with_capacity(batch_size * n * n);
        let mut rhs = Vec::with_capacity(batch_size * n);

        for batch_idx in 0..batch_size {
            // Create a sparse matrix in column-major order
            for col in 0..n {
                for row in 0..n {
                    let val = if row == col {
                        10.0 + (batch_idx as f64) * 0.001 // Diagonal: dominant
                    } else if (row as i32 - col as i32).abs() == 1 {
                        -1.0 // Sub/super diagonal
                    } else {
                        0.0 // Off-diagonal (sparse)
                    };
                    matrices.push(val);
                }
            }
            // RHS vector
            for i in 0..n {
                rhs.push((i + 1) as f64);
            }
        }

        // Dense Faer benchmark
        let dense_solver = FaerBatchedSolver::new(GpuBatchConfig::default());
        group.bench_with_input(
            BenchmarkId::new("faer-dense", &param),
            &(batch_size, n),
            |b, &(bs, sz)| b.iter(|| dense_solver.solve_batch(&matrices, &rhs, sz, bs).unwrap()),
        );

        // Sparse cached Faer benchmark
        let sparse_solver = FaerSparseCachedBatchedSolver::new(GpuBatchConfig::default());
        group.bench_with_input(
            BenchmarkId::new("faer-sparse-cached", &param),
            &(batch_size, n),
            |b, &(bs, sz)| {
                // Reset cache before each iteration to measure cold + warm behavior
                sparse_solver.reset_cache();
                b.iter(|| sparse_solver.solve_batch(&matrices, &rhs, sz, bs).unwrap())
            },
        );

        // Sparse cached with warm cache (only numeric factorization)
        // Pre-warm the cache
        let _ = sparse_solver.solve_batch(&matrices, &rhs, n, batch_size);
        group.bench_with_input(
            BenchmarkId::new("faer-sparse-cached-warm", &param),
            &(batch_size, n),
            |b, &(bs, sz)| {
                // Cache is already warm, measure pure numeric factorization
                b.iter(|| sparse_solver.solve_batch(&matrices, &rhs, sz, bs).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark comparing sequential vs parallel CPU sweep execution.
///
/// This benchmark measures the speedup from rayon parallelization.
#[cfg(feature = "parallel")]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.sample_size(20);

    // Test configurations: (batch_size, matrix_size)
    // Larger batches benefit more from parallelism
    let configs = [
        (100, 50),
        (500, 50),
        (1000, 50),
        (1000, 100),
        (2000, 50),
        (2000, 100),
    ];

    for (batch_size, matrix_size) in configs {
        let param = format!("{}x{}", batch_size, matrix_size);

        let factory = DividerFactory::new(matrix_size);
        let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
        let variations = vec![
            ParameterVariation::new("R1", 1000.0)
                .with_bounds(500.0, 1500.0)
                .with_sigma(0.1),
        ];
        let criteria = ConvergenceCriteria::default();
        let config = DispatchConfig::default();
        let backend = BackendSelector::auto();
        let parallel_config = ParallelSweepConfig::default();

        // Sequential (batched) benchmark
        group.bench_with_input(BenchmarkId::new("sequential", &param), &(), |b, _| {
            b.iter(|| {
                solve_batched_sweep_gpu(
                    &backend,
                    &factory,
                    &generator,
                    &variations,
                    &criteria,
                    &config,
                )
                .unwrap()
            })
        });

        // Parallel benchmark
        group.bench_with_input(BenchmarkId::new("parallel", &param), &(), |b, _| {
            b.iter(|| {
                solve_batched_sweep_parallel(
                    &backend,
                    &factory,
                    &generator,
                    &variations,
                    &criteria,
                    &config,
                    &parallel_config,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark to find GPU crossover point at larger scales.
///
/// Tests larger batch sizes and matrix sizes to find where GPU beats CPU.
#[cfg(feature = "metal")]
fn bench_gpu_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_crossover");
    group.sample_size(10); // Fewer samples for expensive large-scale tests

    // Test at larger scales where GPU might win
    let configs = [
        // (batch_size, matrix_size)
        (5000, 50),
        (5000, 100),
        (10000, 50),
        (10000, 100),
        (20000, 50),
        (20000, 100),
    ];

    for (batch_size, matrix_size) in configs {
        let param = format!("{}x{}", batch_size, matrix_size);

        let factory = DividerFactory::new(matrix_size);
        let generator = MonteCarloGenerator::new(batch_size).with_seed(42);
        let variations = vec![
            ParameterVariation::new("R1", 1000.0)
                .with_bounds(500.0, 1500.0)
                .with_sigma(0.1),
        ];
        let criteria = ConvergenceCriteria::default();
        let config = DispatchConfig::default();

        // Sequential Accelerate (baseline)
        #[cfg(feature = "accelerate")]
        {
            let backend = BackendSelector::prefer_accelerate();
            group.bench_with_input(BenchmarkId::new("accelerate-seq", &param), &(), |b, _| {
                b.iter(|| {
                    solve_batched_sweep_gpu(
                        &backend,
                        &factory,
                        &generator,
                        &variations,
                        &criteria,
                        &config,
                    )
                    .unwrap()
                })
            });
        }

        // Parallel Accelerate
        #[cfg(all(feature = "accelerate", feature = "parallel"))]
        {
            let backend = BackendSelector::prefer_accelerate();
            let parallel_config = ParallelSweepConfig::default();
            group.bench_with_input(BenchmarkId::new("accelerate-par", &param), &(), |b, _| {
                b.iter(|| {
                    solve_batched_sweep_parallel(
                        &backend,
                        &factory,
                        &generator,
                        &variations,
                        &criteria,
                        &config,
                        &parallel_config,
                    )
                    .unwrap()
                })
            });
        }

        // Metal GPU
        {
            let backend = BackendSelector::prefer_metal().with_config(GpuBatchConfig {
                min_batch_size: 1,
                min_matrix_size: 1,
                max_batch_per_launch: 65535,
            });

            if let Ok(solver) = backend.create_solver() {
                if solver.backend_type() == BackendType::Metal {
                    group.bench_with_input(BenchmarkId::new("metal-gpu", &param), &(), |b, _| {
                        b.iter(|| {
                            solve_batched_sweep_gpu(
                                &backend,
                                &factory,
                                &generator,
                                &variations,
                                &criteria,
                                &config,
                            )
                            .unwrap()
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

#[cfg(all(feature = "faer", feature = "parallel", feature = "metal"))]
criterion_group!(
    benches,
    bench_sweep_backends,
    bench_dense_vs_sparse_cached,
    bench_parallel_vs_sequential,
    bench_gpu_crossover
);

#[cfg(all(feature = "faer", feature = "parallel", not(feature = "metal")))]
criterion_group!(
    benches,
    bench_sweep_backends,
    bench_dense_vs_sparse_cached,
    bench_parallel_vs_sequential
);

#[cfg(all(feature = "faer", not(feature = "parallel")))]
criterion_group!(benches, bench_sweep_backends, bench_dense_vs_sparse_cached);

#[cfg(all(not(feature = "faer"), feature = "parallel"))]
criterion_group!(benches, bench_sweep_backends, bench_parallel_vs_sequential);

#[cfg(all(not(feature = "faer"), not(feature = "parallel")))]
criterion_group!(benches, bench_sweep_backends);

criterion_main!(benches);
