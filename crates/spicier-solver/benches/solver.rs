//! Benchmarks for linear solvers.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use spicier_solver::linear::solve_dense;

fn bench_solve_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_dense");

    for size in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bencher, &size| {
                // Create a diagonally dominant matrix (guaranteed non-singular)
                let a = DMatrix::from_fn(size, size, |i, j| {
                    if i == j {
                        (size as f64) + 1.0
                    } else {
                        1.0 / ((i as f64 - j as f64).abs() + 1.0)
                    }
                });
                let rhs = DVector::from_fn(size, |i, _| (i + 1) as f64);

                bencher.iter(|| solve_dense(black_box(&a), black_box(&rhs)).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_solve_dense);
criterion_main!(benches);
