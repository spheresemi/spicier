//! Benchmarks for MNA matrix operations.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use spicier_core::mna::MnaSystem;

fn bench_stamp_conductance(c: &mut Criterion) {
    c.bench_function("stamp_conductance_10x10", |b| {
        let mut mna = MnaSystem::new(10, 0);
        b.iter(|| {
            mna.clear();
            for i in 0..9 {
                mna.stamp_conductance(Some(i), Some(i + 1), black_box(0.001));
            }
        });
    });
}

fn bench_stamp_voltage_source(c: &mut Criterion) {
    c.bench_function("stamp_voltage_source", |b| {
        let mut mna = MnaSystem::new(10, 5);
        b.iter(|| {
            mna.clear();
            for i in 0..5 {
                mna.stamp_voltage_source(Some(i * 2), Some(i * 2 + 1), i, black_box(5.0));
            }
        });
    });
}

criterion_group!(benches, bench_stamp_conductance, bench_stamp_voltage_source);
criterion_main!(benches);
