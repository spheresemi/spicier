//! Benchmarks for netlist parsing.
//!
//! Currently a placeholder - will be populated when parser is implemented.

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("parser_placeholder", |b| {
        b.iter(|| {
            // Placeholder benchmark
            let _ = 1 + 1;
        });
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
