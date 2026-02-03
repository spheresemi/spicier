//! Benchmark comparing Accelerate vs scalar SIMD operations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex64 as C64;
use spicier_simd::{
    SimdCapability, complex_conjugate_dot_product, complex_dot_product, complex_dot_scalar,
    conjugate_dot_scalar, real_dot_product, real_dot_scalar, real_matvec, real_matvec_scalar,
};

fn bench_real_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_dot_product");
    let cap = SimdCapability::detect();

    for size in [100, 500, 1000, 5000, 10000] {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.2).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, _| {
            bencher.iter(|| real_dot_scalar(&a, &b));
        });

        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", cap), size),
            &size,
            |bencher, _| {
                bencher.iter(|| real_dot_product(&a, &b, cap));
            },
        );
    }
    group.finish();
}

fn bench_complex_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_dot_product");
    let cap = SimdCapability::detect();

    for size in [100, 500, 1000, 5000] {
        let a: Vec<C64> = (0..size)
            .map(|i| C64::new(i as f64 * 0.1, i as f64 * 0.2))
            .collect();
        let b: Vec<C64> = (0..size)
            .map(|i| C64::new((size - i) as f64 * 0.3, i as f64 * 0.05))
            .collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, _| {
            bencher.iter(|| complex_dot_scalar(&a, &b));
        });

        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", cap), size),
            &size,
            |bencher, _| {
                bencher.iter(|| complex_dot_product(&a, &b, cap));
            },
        );
    }
    group.finish();
}

fn bench_conjugate_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("conjugate_dot_product");
    let cap = SimdCapability::detect();

    for size in [100, 500, 1000, 5000] {
        let a: Vec<C64> = (0..size)
            .map(|i| C64::new(i as f64 * 0.1, -(i as f64) * 0.2))
            .collect();
        let b: Vec<C64> = (0..size)
            .map(|i| C64::new((size - i) as f64 * 0.3, i as f64 * 0.05))
            .collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, _| {
            bencher.iter(|| conjugate_dot_scalar(&a, &b));
        });

        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", cap), size),
            &size,
            |bencher, _| {
                bencher.iter(|| complex_conjugate_dot_product(&a, &b, cap));
            },
        );
    }
    group.finish();
}

fn bench_real_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_matvec");
    let cap = SimdCapability::detect();

    for n in [50, 100, 200, 500] {
        let matrix: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.01).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let mut y_scalar = vec![0.0; n];
        let mut y_simd = vec![0.0; n];

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bencher, _| {
            bencher.iter(|| {
                real_matvec_scalar(&matrix, n, &x, &mut y_scalar);
            });
        });

        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", cap), n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    real_matvec(&matrix, n, &x, &mut y_simd, cap);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_real_dot,
    bench_complex_dot,
    bench_conjugate_dot,
    bench_real_matvec
);
criterion_main!(benches);
