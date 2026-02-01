// Complex matrix-vector multiplication shader.
//
// Computes y = A * x where A is an NxN complex matrix and x, y are complex vectors.
// Complex numbers are represented as vec2<f32> (real, imag).

struct Uniforms {
    n: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Matrix A stored in row-major order: A[i,j] at index i*n + j
@group(0) @binding(1) var<storage, read> matrix: array<vec2<f32>>;

// Input vector x
@group(0) @binding(2) var<storage, read> x: array<vec2<f32>>;

// Output vector y
@group(0) @binding(3) var<storage, read_write> y: array<vec2<f32>>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Each invocation computes one row of the result: y[i] = sum_j A[i,j] * x[j]
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = uniforms.n;

    if (i >= n) {
        return;
    }

    var sum = vec2<f32>(0.0, 0.0);
    let row_offset = i * n;

    for (var j = 0u; j < n; j = j + 1u) {
        let a_ij = matrix[row_offset + j];
        let x_j = x[j];
        sum = sum + complex_mul(a_ij, x_j);
    }

    y[i] = sum;
}
