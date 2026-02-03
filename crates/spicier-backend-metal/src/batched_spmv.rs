//! GPU batched sparse matrix-vector multiply (SpMV) for iterative solvers.
//!
//! For parameter sweeps, all matrices share the same sparsity pattern - only
//! values differ. This enables efficient batched SpMV where:
//! - CSR indices (row_ptr, col_idx) are shared across all sweeps
//! - CSR values are different per sweep (num_sweeps × nnz)
//! - One kernel computes y = A*x for all sweep points simultaneously
//!
//! # Performance
//!
//! The key optimization is that index arrays are loaded once and reused
//! for all sweep points. For 1000 sweeps with a 100-node circuit (~1000 nnz):
//! - Index data: 2 × 1000 × 4 bytes = 8 KB (shared)
//! - Value data: 1000 × 1000 × 4 bytes = 4 MB (per sweep)
//! - Ratio: 500:1 value:index data

use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// CSR matrix structure for batched SpMV.
///
/// The sparsity pattern (row_ptr, col_idx) is shared across all sweep points.
/// Only the values differ per sweep.
#[derive(Clone, Debug)]
pub struct BatchedCsrMatrix {
    /// Number of rows (= number of columns for square matrices).
    pub n: usize,
    /// Number of non-zero elements.
    pub nnz: usize,
    /// Row pointers: row_ptr[i] is the start of row i in col_idx/values.
    /// Length: n + 1
    pub row_ptr: Vec<u32>,
    /// Column indices for each non-zero.
    /// Length: nnz
    pub col_idx: Vec<u32>,
}

impl BatchedCsrMatrix {
    /// Create a new CSR matrix structure.
    pub fn new(n: usize, row_ptr: Vec<u32>, col_idx: Vec<u32>) -> Self {
        let nnz = col_idx.len();
        Self {
            n,
            nnz,
            row_ptr,
            col_idx,
        }
    }
}

/// GPU kernel for batched sparse matrix-vector multiply.
///
/// Computes y = A*x for multiple matrices sharing the same sparsity pattern
/// but with different values.
pub struct GpuBatchedSpmv {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBatchedSpmv {
    /// Create a new batched SpMV kernel.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        let shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Batched SpMV Shader"),
                source: wgpu::ShaderSource::Wgsl(BATCHED_SPMV_SHADER.into()),
            });

        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Batched SpMV Bind Group Layout"),
                    entries: &[
                        // Uniforms (n, nnz, num_sweeps)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // row_ptr (shared across sweeps)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // col_idx (shared across sweeps)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // values (per sweep × nnz)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // x (per sweep × n)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // y output (per sweep × n)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            ctx.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Batched SpMV Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Batched SpMV Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("spmv"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            ctx,
            pipeline,
            bind_group_layout,
        })
    }

    /// Compute y = A*x for all sweep points.
    ///
    /// # Arguments
    /// * `csr` - CSR sparsity pattern (shared across sweeps)
    /// * `values` - CSR values (num_sweeps × nnz)
    /// * `x` - Input vectors (num_sweeps × n)
    /// * `num_sweeps` - Number of sweep points
    ///
    /// # Returns
    /// Output vectors y (num_sweeps × n)
    pub fn multiply(
        &self,
        csr: &BatchedCsrMatrix,
        values: &[f32],
        x: &[f32],
        num_sweeps: usize,
    ) -> Result<Vec<f32>> {
        if values.len() != num_sweeps * csr.nnz {
            return Err(WgpuError::InvalidDimension(format!(
                "values length {} != num_sweeps {} × nnz {}",
                values.len(),
                num_sweeps,
                csr.nnz
            )));
        }
        if x.len() != num_sweeps * csr.n {
            return Err(WgpuError::InvalidDimension(format!(
                "x length {} != num_sweeps {} × n {}",
                x.len(),
                num_sweeps,
                csr.n
            )));
        }

        // Uniforms
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Uniforms {
            n: u32,
            nnz: u32,
            num_sweeps: u32,
            _pad: u32,
        }
        let uniforms = Uniforms {
            n: csr.n as u32,
            nnz: csr.nnz as u32,
            num_sweeps: num_sweeps as u32,
            _pad: 0,
        };

        let uniform_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SpMV Uniforms"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let row_ptr_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CSR row_ptr"),
                    contents: bytemuck::cast_slice(&csr.row_ptr),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let col_idx_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CSR col_idx"),
                    contents: bytemuck::cast_slice(&csr.col_idx),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let values_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CSR values"),
                    contents: bytemuck::cast_slice(values),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let x_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpMV x"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (num_sweeps * csr.n * std::mem::size_of::<f32>()) as u64;
        let y_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("SpMV y"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("SpMV y Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batched SpMV Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: row_ptr_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: col_idx_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: values_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: y_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SpMV Encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpMV Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per (row, sweep) pair
            let total_work = (csr.n * num_sweeps) as u32;
            let workgroups = total_work.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&y_buffer, 0, &staging_buffer, 0, output_size);
        self.ctx.queue().submit(std::iter::once(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| WgpuError::Compute("Failed to receive map result".into()))?
            .map_err(|e| WgpuError::Buffer(format!("Buffer map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// WGSL shader for batched sparse matrix-vector multiply.
///
/// Each thread computes one row of y = A*x for one sweep point.
/// The CSR indices (row_ptr, col_idx) are shared across all sweeps.
const BATCHED_SPMV_SHADER: &str = r#"
struct Uniforms {
    n: u32,
    nnz: u32,
    num_sweeps: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = uniforms.n * uniforms.num_sweeps;
    if idx >= total {
        return;
    }

    // Decode (row, sweep) from linear index
    let row = idx / uniforms.num_sweeps;
    let sweep = idx % uniforms.num_sweeps;

    // Get row bounds from shared CSR structure
    let row_start = row_ptr[row];
    let row_end = row_ptr[row + 1u];

    // Compute dot product for this row
    var sum: f32 = 0.0;
    for (var k = row_start; k < row_end; k = k + 1u) {
        let col = col_idx[k];
        // Values are laid out as [sweep0_val0, sweep0_val1, ..., sweep1_val0, ...]
        let val_idx = sweep * uniforms.nnz + k;
        let x_idx = sweep * uniforms.n + col;
        sum = sum + values[val_idx] * x[x_idx];
    }

    // Write result
    let y_idx = sweep * uniforms.n + row;
    y[y_idx] = sum;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_context() -> Result<Arc<WgpuContext>> {
        Ok(Arc::new(WgpuContext::new()?))
    }

    #[test]
    fn test_simple_spmv() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let spmv = GpuBatchedSpmv::new(ctx).unwrap();

        // Simple 2×2 identity matrix:
        // [1, 0]
        // [0, 1]
        let csr = BatchedCsrMatrix::new(
            2,
            vec![0, 1, 2], // row_ptr
            vec![0, 1],    // col_idx (diagonals)
        );

        // Single sweep with identity values
        let values = vec![1.0, 1.0];
        let x = vec![3.0, 4.0];

        let y = spmv.multiply(&csr, &values, &x, 1).unwrap();

        assert_eq!(y.len(), 2);
        assert!((y[0] - 3.0).abs() < 1e-6, "y[0] = {}", y[0]);
        assert!((y[1] - 4.0).abs() < 1e-6, "y[1] = {}", y[1]);
    }

    #[test]
    fn test_dense_matrix_spmv() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let spmv = GpuBatchedSpmv::new(ctx).unwrap();

        // Dense 2×2 matrix:
        // [1, 2]
        // [3, 4]
        let csr = BatchedCsrMatrix::new(
            2,
            vec![0, 2, 4],    // row_ptr
            vec![0, 1, 0, 1], // col_idx
        );

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 2.0];

        // y = [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
        let y = spmv.multiply(&csr, &values, &x, 1).unwrap();

        assert_eq!(y.len(), 2);
        assert!((y[0] - 5.0).abs() < 1e-6, "y[0] = {}", y[0]);
        assert!((y[1] - 11.0).abs() < 1e-6, "y[1] = {}", y[1]);
    }

    #[test]
    fn test_batched_spmv() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let spmv = GpuBatchedSpmv::new(ctx).unwrap();

        // 2×2 diagonal matrix
        let csr = BatchedCsrMatrix::new(2, vec![0, 1, 2], vec![0, 1]);

        // Two sweeps with different diagonal values
        let values = vec![
            2.0, 3.0, // sweep 0: diag = [2, 3]
            4.0, 5.0, // sweep 1: diag = [4, 5]
        ];
        let x = vec![
            1.0, 1.0, // sweep 0: x = [1, 1]
            1.0, 1.0, // sweep 1: x = [1, 1]
        ];

        let y = spmv.multiply(&csr, &values, &x, 2).unwrap();

        assert_eq!(y.len(), 4);
        // Sweep 0: y = [2*1, 3*1] = [2, 3]
        assert!((y[0] - 2.0).abs() < 1e-6, "y[0] = {}", y[0]);
        assert!((y[1] - 3.0).abs() < 1e-6, "y[1] = {}", y[1]);
        // Sweep 1: y = [4*1, 5*1] = [4, 5]
        assert!((y[2] - 4.0).abs() < 1e-6, "y[2] = {}", y[2]);
        assert!((y[3] - 5.0).abs() < 1e-6, "y[3] = {}", y[3]);
    }

    #[test]
    fn test_spmv_performance() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let spmv = GpuBatchedSpmv::new(ctx).unwrap();

        // Simulate a 100-node circuit with ~1000 nnz
        let n = 100;
        let nnz_per_row = 10;
        let nnz = n * nnz_per_row;
        let num_sweeps = 1000;

        // Build CSR structure
        let mut row_ptr = vec![0u32];
        let mut col_idx = Vec::with_capacity(nnz);
        for row in 0..n {
            for j in 0..nnz_per_row {
                col_idx.push(((row + j) % n) as u32);
            }
            row_ptr.push((row + 1) as u32 * nnz_per_row as u32);
        }
        let csr = BatchedCsrMatrix::new(n, row_ptr, col_idx);

        // Generate test data
        let values: Vec<f32> = (0..num_sweeps * nnz).map(|i| (i as f32) * 0.001).collect();
        let x: Vec<f32> = (0..num_sweeps * n).map(|i| (i as f32) * 0.01).collect();

        let start = std::time::Instant::now();
        let y = spmv.multiply(&csr, &values, &x, num_sweeps).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(y.len(), num_sweeps * n);
        let total_ops = num_sweeps * n * nnz_per_row * 2; // 2 ops per nnz (mul + add)
        println!(
            "GPU SpMV: {}×{} matrix, {} sweeps in {:?} ({:.2} GFLOPS)",
            n,
            n,
            num_sweeps,
            elapsed,
            total_ops as f64 / elapsed.as_secs_f64() / 1e9
        );
    }
}
