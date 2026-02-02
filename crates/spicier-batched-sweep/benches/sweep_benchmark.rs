//! Benchmark comparing CPU vs Metal batched sweep performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::DVector;
use spicier_batched_sweep::{solve_batched_sweep_gpu, BackendSelector};
#[cfg(feature = "metal")]
use spicier_batched_sweep::{BackendType, GpuBatchConfig};
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

    let _ = solve_batched_sweep_gpu(&backend, &factory, &generator, &variations, &criteria, &config)
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

    let _ = solve_batched_sweep_gpu(&backend, &factory, &generator, &variations, &criteria, &config)
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

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("CPU", &param), &(batch_size, matrix_size), |b, &(bs, ms)| {
            b.iter(|| run_cpu_sweep(bs, ms))
        });

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
                    group.bench_with_input(BenchmarkId::new("Metal", &param), &(batch_size, matrix_size), |b, &(bs, ms)| {
                        b.iter(|| run_metal_sweep(bs, ms))
                    });
                }
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_sweep_backends);
criterion_main!(benches);
