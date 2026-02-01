//! Dense matrix operators backed by GPU memory.
//!
//! Uses wgpu compute shaders for matrix-vector multiplication.
//! Implements both [`ComplexOperator`] and [`RealOperator`].

use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use bytemuck::{Pod, Zeroable};
use num_complex::Complex64 as C64;
use spicier_solver::operator::{ComplexOperator, RealOperator};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-resident representation of a complex number.
///
/// Uses f32 for broad GPU compatibility (most GPUs don't support f64).
/// Layout matches vec2<f32> in WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuComplex {
    pub re: f32,
    pub im: f32,
}

impl From<C64> for GpuComplex {
    fn from(c: C64) -> Self {
        Self {
            re: c.re as f32,
            im: c.im as f32,
        }
    }
}

impl From<GpuComplex> for C64 {
    fn from(c: GpuComplex) -> Self {
        C64::new(c.re as f64, c.im as f64)
    }
}

/// Uniform buffer layout for shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    n: u32,
    _padding: u32,
}

// ============================================================================
// Complex dense operator (compute shader)
// ============================================================================

/// Dense NxN complex matrix operator with GPU acceleration via compute shaders.
pub struct WgpuComplexDenseOperator {
    n: usize,
    gpu_matrix: wgpu::Buffer,
    cpu_matrix: Vec<C64>,
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    cpu_threshold: usize,
}

impl WgpuComplexDenseOperator {
    const SHADER_SOURCE: &'static str = include_str!("complex_matvec.wgsl");

    /// Build from an existing matrix.
    pub fn from_matrix(ctx: Arc<WgpuContext>, matrix: Vec<C64>, n: usize) -> Result<Self> {
        if matrix.len() != n * n {
            return Err(WgpuError::InvalidDimension(format!(
                "Matrix length {} doesn't match n*n = {}",
                matrix.len(),
                n * n
            )));
        }

        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spicier Complex Matvec Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Complex Matvec Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Complex Matvec Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Complex Matvec Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("matvec"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniforms = Uniforms {
            n: n as u32,
            _padding: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let gpu_data: Vec<GpuComplex> = matrix.iter().map(|&c| c.into()).collect();
        let gpu_matrix = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Complex Matrix"),
            contents: bytemuck::cast_slice(&gpu_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        log::debug!(
            "Uploaded {}x{} complex matrix to GPU ({} bytes)",
            n,
            n,
            n * n * std::mem::size_of::<GpuComplex>()
        );

        Ok(Self {
            n,
            gpu_matrix,
            cpu_matrix: matrix,
            ctx,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            cpu_threshold: 64,
        })
    }

    /// Set the threshold below which CPU fallback is used.
    pub fn with_cpu_threshold(mut self, threshold: usize) -> Self {
        self.cpu_threshold = threshold;
        self
    }

    fn apply_cpu(&self, x: &[C64], y: &mut [C64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let mut sum = C64::new(0.0, 0.0);
            let row_start = i * self.n;
            for (j, xj) in x.iter().enumerate().take(self.n) {
                sum += self.cpu_matrix[row_start + j] * xj;
            }
            *yi = sum;
        }
    }

    fn apply_gpu(&self, x: &[C64], y: &mut [C64]) -> Result<()> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let x_gpu: Vec<GpuComplex> = x.iter().map(|&c| c.into()).collect();

        let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Vector"),
            contents: bytemuck::cast_slice(&x_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Vector"),
            size: (self.n * std::mem::size_of::<GpuComplex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.n * std::mem::size_of::<GpuComplex>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex Matvec Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Complex Matvec Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Complex Matvec Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (self.n as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &y_buffer,
            0,
            &staging_buffer,
            0,
            (self.n * std::mem::size_of::<GpuComplex>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| WgpuError::Buffer(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| WgpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

        {
            let data = buffer_slice.get_mapped_range();
            let y_gpu: &[GpuComplex] = bytemuck::cast_slice(&data);
            for (i, c) in y_gpu.iter().enumerate() {
                y[i] = C64::from(*c);
            }
        }

        staging_buffer.unmap();
        Ok(())
    }
}

impl ComplexOperator for WgpuComplexDenseOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[C64], y: &mut [C64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);

        if self.n <= self.cpu_threshold {
            self.apply_cpu(x, y);
            return;
        }

        if let Err(e) = self.apply_gpu(x, y) {
            log::warn!("GPU complex apply failed, falling back to CPU: {}", e);
            self.apply_cpu(x, y);
        }
    }
}

// ============================================================================
// Real dense operator (compute shader)
// ============================================================================

/// Dense NxN real matrix operator with GPU acceleration via compute shaders.
pub struct WgpuRealDenseOperator {
    n: usize,
    gpu_matrix: wgpu::Buffer,
    cpu_matrix: Vec<f64>,
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    cpu_threshold: usize,
}

impl WgpuRealDenseOperator {
    const SHADER_SOURCE: &'static str = include_str!("real_matvec.wgsl");

    /// Build from an existing matrix.
    pub fn from_matrix(ctx: Arc<WgpuContext>, matrix: Vec<f64>, n: usize) -> Result<Self> {
        if matrix.len() != n * n {
            return Err(WgpuError::InvalidDimension(format!(
                "Matrix length {} doesn't match n*n = {}",
                matrix.len(),
                n * n
            )));
        }

        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spicier Real Matvec Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Real Matvec Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Real Matvec Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Real Matvec Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("matvec"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniforms = Uniforms {
            n: n as u32,
            _padding: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Convert f64 -> f32 for GPU
        let gpu_data: Vec<f32> = matrix.iter().map(|&v| v as f32).collect();
        let gpu_matrix = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Real Matrix"),
            contents: bytemuck::cast_slice(&gpu_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        log::debug!(
            "Uploaded {}x{} real matrix to GPU ({} bytes)",
            n,
            n,
            n * n * std::mem::size_of::<f32>()
        );

        Ok(Self {
            n,
            gpu_matrix,
            cpu_matrix: matrix,
            ctx,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            cpu_threshold: 64,
        })
    }

    /// Set the threshold below which CPU fallback is used.
    pub fn with_cpu_threshold(mut self, threshold: usize) -> Self {
        self.cpu_threshold = threshold;
        self
    }

    fn apply_cpu(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let mut sum = 0.0;
            let row_start = i * self.n;
            for (j, xj) in x.iter().enumerate().take(self.n) {
                sum += self.cpu_matrix[row_start + j] * xj;
            }
            *yi = sum;
        }
    }

    fn apply_gpu(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let x_gpu: Vec<f32> = x.iter().map(|&v| v as f32).collect();

        let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Vector"),
            contents: bytemuck::cast_slice(&x_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let y_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Vector"),
            size: (self.n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Real Matvec Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.gpu_matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: y_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Real Matvec Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Real Matvec Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (self.n as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &y_buffer,
            0,
            &staging_buffer,
            0,
            (self.n * std::mem::size_of::<f32>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| WgpuError::Buffer(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| WgpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

        {
            let data = buffer_slice.get_mapped_range();
            let y_gpu: &[f32] = bytemuck::cast_slice(&data);
            for (i, &v) in y_gpu.iter().enumerate() {
                y[i] = v as f64;
            }
        }

        staging_buffer.unmap();
        Ok(())
    }
}

impl RealOperator for WgpuRealDenseOperator {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);

        if self.n <= self.cpu_threshold {
            self.apply_cpu(x, y);
            return;
        }

        if let Err(e) = self.apply_gpu(x, y) {
            log::warn!("GPU real apply failed, falling back to CPU: {}", e);
            self.apply_cpu(x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_wgpu_context() -> Option<Arc<WgpuContext>> {
        std::panic::catch_unwind(WgpuContext::new)
            .ok()
            .and_then(|result| result.ok())
            .map(Arc::new)
    }

    #[test]
    fn test_complex_dimension_check() {
        let ctx = match try_create_wgpu_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let matrix = vec![C64::new(1.0, 0.0); 9];
        assert!(WgpuComplexDenseOperator::from_matrix(ctx.clone(), matrix.clone(), 3).is_ok());
        assert!(WgpuComplexDenseOperator::from_matrix(ctx, matrix, 4).is_err());
    }

    #[test]
    fn test_real_dimension_check() {
        let ctx = match try_create_wgpu_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let matrix = vec![1.0; 9];
        assert!(WgpuRealDenseOperator::from_matrix(ctx.clone(), matrix.clone(), 3).is_ok());
        assert!(WgpuRealDenseOperator::from_matrix(ctx, matrix, 4).is_err());
    }

    #[test]
    fn test_complex_cpu_fallback() {
        let ctx = match try_create_wgpu_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let matrix = vec![
            C64::new(1.0, 0.0), C64::new(1.0, 1.0), C64::new(1.0, 2.0),
            C64::new(2.0, 0.0), C64::new(2.0, 1.0), C64::new(2.0, 2.0),
            C64::new(3.0, 0.0), C64::new(3.0, 1.0), C64::new(3.0, 2.0),
        ];
        let op = WgpuComplexDenseOperator::from_matrix(ctx, matrix, 3)
            .unwrap()
            .with_cpu_threshold(100);

        let x = vec![C64::new(1.0, 0.0); 3];
        let mut y = vec![C64::new(0.0, 0.0); 3];
        op.apply(&x, &mut y);

        assert!((y[0].re - 3.0).abs() < 1e-10);
        assert!((y[0].im - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_cpu_fallback() {
        let ctx = match try_create_wgpu_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let op = WgpuRealDenseOperator::from_matrix(ctx, matrix, 3)
            .unwrap()
            .with_cpu_threshold(100);

        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        op.apply(&x, &mut y);

        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 15.0).abs() < 1e-10);
        assert!((y[2] - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_gpu_matches_cpu() {
        let ctx = match try_create_wgpu_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let n = 100;
        let matrix: Vec<C64> = (0..n * n)
            .map(|i| C64::new((i + 1) as f64, (i % n) as f64))
            .collect();

        let op_gpu = WgpuComplexDenseOperator::from_matrix(ctx.clone(), matrix.clone(), n)
            .unwrap()
            .with_cpu_threshold(0);
        let op_cpu = WgpuComplexDenseOperator::from_matrix(ctx, matrix, n)
            .unwrap()
            .with_cpu_threshold(1000);

        let x: Vec<C64> = (0..n)
            .map(|i| C64::new(i as f64 * 0.1, (n - i) as f64 * 0.05))
            .collect();

        let mut y_gpu = vec![C64::new(0.0, 0.0); n];
        let mut y_cpu = vec![C64::new(0.0, 0.0); n];

        op_gpu.apply(&x, &mut y_gpu);
        op_cpu.apply(&x, &mut y_cpu);

        for i in 0..n {
            let diff = (y_gpu[i] - y_cpu[i]).norm();
            let magnitude = y_cpu[i].norm().max(1.0);
            let relative_error = diff / magnitude;
            assert!(
                relative_error < 1e-4,
                "Mismatch at {}: gpu={:?}, cpu={:?}, relative={}",
                i, y_gpu[i], y_cpu[i], relative_error
            );
        }
    }

    #[test]
    fn test_real_gpu_matches_cpu() {
        let ctx = match try_create_wgpu_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let n = 100;
        let matrix: Vec<f64> = (0..n * n)
            .map(|i| (i + 1) as f64 * 0.01)
            .collect();

        let op_gpu = WgpuRealDenseOperator::from_matrix(ctx.clone(), matrix.clone(), n)
            .unwrap()
            .with_cpu_threshold(0);
        let op_cpu = WgpuRealDenseOperator::from_matrix(ctx, matrix, n)
            .unwrap()
            .with_cpu_threshold(1000);

        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

        let mut y_gpu = vec![0.0; n];
        let mut y_cpu = vec![0.0; n];

        op_gpu.apply(&x, &mut y_gpu);
        op_cpu.apply(&x, &mut y_cpu);

        for i in 0..n {
            let diff = (y_gpu[i] - y_cpu[i]).abs();
            let magnitude = y_cpu[i].abs().max(1.0);
            let relative_error = diff / magnitude;
            assert!(
                relative_error < 1e-4,
                "Mismatch at {}: gpu={}, cpu={}, relative={}",
                i, y_gpu[i], y_cpu[i], relative_error
            );
        }
    }
}
