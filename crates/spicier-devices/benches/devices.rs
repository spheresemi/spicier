//! Benchmarks for device stamping.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use spicier_core::NodeId;
use spicier_core::mna::MnaSystem;
use spicier_devices::Stamp;
use spicier_devices::passive::Resistor;

fn bench_resistor_stamp(c: &mut Criterion) {
    c.bench_function("stamp_100_resistors", |b| {
        let mut mna = MnaSystem::new(101, 0);
        let resistors: Vec<_> = (0..100)
            .map(|i| {
                Resistor::new(
                    format!("R{}", i),
                    NodeId::new(i + 1),
                    NodeId::new(i + 2),
                    1000.0,
                )
            })
            .collect();

        b.iter(|| {
            mna.clear();
            for r in &resistors {
                r.stamp(black_box(&mut mna));
            }
        });
    });
}

criterion_group!(benches, bench_resistor_stamp);
criterion_main!(benches);
