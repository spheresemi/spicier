// Real matrix-vector multiplication shader.
//
// Computes y = A * x where A is an NxN real matrix and x, y are real vectors.

struct Uniforms {
    n: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Matrix A stored in row-major order: A[i,j] at index i*n + j
@group(0) @binding(1) var<storage, read> matrix: array<f32>;

// Input vector x
@group(0) @binding(2) var<storage, read> x: array<f32>;

// Output vector y
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// Each invocation computes one row of the result: y[i] = sum_j A[i,j] * x[j]
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = uniforms.n;

    if (i >= n) {
        return;
    }

    var sum = 0.0;
    let row_offset = i * n;

    for (var j = 0u; j < n; j = j + 1u) {
        sum = sum + matrix[row_offset + j] * x[j];
    }

    y[i] = sum;
}
