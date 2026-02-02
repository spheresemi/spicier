//! GPU-accelerated batched LU factorization and solve using wgpu/Metal.
//!
//! Each matrix in the batch is processed by a separate workgroup, enabling
//! massive parallelism for Monte Carlo, corner analysis, and parameter sweeps.

use crate::batch_layout::{BatchLayout, pack_matrices_f32, pack_rhs_f32, unpack_solutions_f64};
use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, RwLock};
use wgpu::util::DeviceExt;

/// Maximum matrix dimension supported (limited by workgroup shared memory).
pub const MAX_MATRIX_SIZE: usize = 128;

/// Minimum batch size for GPU to be worthwhile.
/// Note: Current implementation has high overhead, so this is set high.
pub const MIN_BATCH_SIZE: usize = 2000;

/// Minimum matrix size for GPU to be worthwhile.
/// Note: Current implementation needs large matrices to amortize overhead.
pub const MIN_MATRIX_SIZE: usize = 100;

/// Uniform buffer layout for shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    n: u32,
    batch_size: u32,
    row_stride: u32,
    matrix_stride: u32,
}

/// Result of a batched LU solve operation.
#[derive(Debug, Clone)]
pub struct BatchedSolveResult {
    /// Solutions for each system (flattened: batch_size * n elements).
    pub solutions: Vec<f64>,
    /// Indices of matrices that were singular.
    pub singular_indices: Vec<usize>,
    /// Matrix dimension.
    pub n: usize,
    /// Number of systems solved.
    pub batch_size: usize,
}

impl BatchedSolveResult {
    /// Get the solution for a specific system.
    pub fn solution(&self, index: usize) -> Option<&[f64]> {
        if index >= self.batch_size {
            return None;
        }
        let start = index * self.n;
        let end = start + self.n;
        Some(&self.solutions[start..end])
    }

    /// Check if a specific system was singular.
    pub fn is_singular(&self, index: usize) -> bool {
        self.singular_indices.contains(&index)
    }

    /// Number of successfully solved systems.
    pub fn num_solved(&self) -> usize {
        self.batch_size - self.singular_indices.len()
    }
}

/// Configuration for GPU batched operations.
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Minimum batch size to use GPU.
    pub min_batch_size: usize,
    /// Minimum matrix dimension to use GPU.
    pub min_matrix_size: usize,
    /// Maximum matrix size (limited by shader).
    pub max_matrix_size: usize,
}

impl Default for GpuBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: MIN_BATCH_SIZE,
            min_matrix_size: MIN_MATRIX_SIZE,
            max_matrix_size: MAX_MATRIX_SIZE,
        }
    }
}

impl GpuBatchConfig {
    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        matrix_size >= self.min_matrix_size
            && matrix_size <= self.max_matrix_size
            && batch_size >= self.min_batch_size
    }
}

/// Minimum buffer capacity to allocate (prevents thrashing for tiny allocations).
const MIN_BUFFER_ELEMENTS: usize = 1024;

/// Cached GPU buffers for reuse across solve_batch calls.
struct CachedBuffers {
    /// Capacity for matrix data (in f32 elements, not bytes).
    matrix_capacity: usize,
    /// Capacity for RHS/solution data (in f32 elements).
    rhs_capacity: usize,
    /// Capacity for info data (in i32 elements = batch_size).
    info_capacity: usize,
    /// Cached matrix dimension (bind group depends on uniform buffer contents).
    cached_n: usize,
    /// Cached batch size (uniform buffer contains this, shader checks against it).
    cached_batch_size: usize,
    /// Cached layout info for bind group invalidation.
    cached_row_stride: usize,
    cached_matrix_stride: usize,

    // GPU buffers (None until first use)
    matrix_buffer: Option<wgpu::Buffer>,
    rhs_buffer: Option<wgpu::Buffer>,
    info_buffer: Option<wgpu::Buffer>,
    solution_staging: Option<wgpu::Buffer>,
    info_staging: Option<wgpu::Buffer>,

    // Bind group (recreated when buffers change or dimensions change)
    bind_group: Option<wgpu::BindGroup>,
    // Uniform buffer cached for the bind group
    uniform_buffer: Option<wgpu::Buffer>,
}

impl CachedBuffers {
    fn new() -> Self {
        Self {
            matrix_capacity: 0,
            rhs_capacity: 0,
            info_capacity: 0,
            cached_n: 0,
            cached_batch_size: 0,
            cached_row_stride: 0,
            cached_matrix_stride: 0,
            matrix_buffer: None,
            rhs_buffer: None,
            info_buffer: None,
            solution_staging: None,
            info_staging: None,
            bind_group: None,
            uniform_buffer: None,
        }
    }

    /// Check if cached buffers can accommodate the given sizes.
    fn can_accommodate(&self, matrix_elems: usize, rhs_elems: usize, info_elems: usize) -> bool {
        self.matrix_capacity >= matrix_elems
            && self.rhs_capacity >= rhs_elems
            && self.info_capacity >= info_elems
    }

    /// Check if bind group can be reused (same dimensions, batch size, and layout).
    fn can_reuse_bind_group(
        &self,
        n: usize,
        batch_size: usize,
        row_stride: usize,
        matrix_stride: usize,
    ) -> bool {
        self.bind_group.is_some()
            && self.cached_n == n
            && self.cached_batch_size == batch_size
            && self.cached_row_stride == row_stride
            && self.cached_matrix_stride == matrix_stride
    }
}

/// GPU-accelerated batched LU solver using wgpu/Metal compute shaders.
pub struct MetalBatchedLuSolver {
    ctx: Arc<WgpuContext>,
    config: GpuBatchConfig,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Cached buffers with interior mutability for reuse across calls.
    cached: RwLock<CachedBuffers>,
}

impl MetalBatchedLuSolver {
    const SHADER_SOURCE: &'static str = include_str!("batched_lu.wgsl");

    /// Create a new batched LU solver.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        Self::with_config(ctx, GpuBatchConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(ctx: Arc<WgpuContext>, config: GpuBatchConfig) -> Result<Self> {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batched LU Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Batched LU Bind Group Layout"),
            entries: &[
                // Uniforms
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
                // Matrices (read-write, modified in place during factorization)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // RHS/Solution vectors (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Info array (singularity flags)
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
            label: Some("Batched LU Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched LU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        log::info!(
            "Created Metal batched LU solver (GPU: {})",
            ctx.adapter_name()
        );

        Ok(Self {
            ctx,
            config,
            pipeline,
            bind_group_layout,
            cached: RwLock::new(CachedBuffers::new()),
        })
    }

    /// Clear the cached buffers.
    ///
    /// Call this to free GPU memory or when changing problem characteristics significantly.
    pub fn clear_cache(&self) {
        let mut cache = self.cached.write().unwrap();
        *cache = CachedBuffers::new();
    }

    /// Check if buffers are cached.
    pub fn has_cached_buffers(&self) -> bool {
        let cache = self.cached.read().unwrap();
        cache.matrix_buffer.is_some()
    }

    /// Get the configuration.
    pub fn config(&self) -> &GpuBatchConfig {
        &self.config
    }

    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        self.config.should_use_gpu(matrix_size, batch_size)
    }

    /// Solve a batch of linear systems Ax = b.
    ///
    /// # Arguments
    /// * `matrices` - Flattened matrices in column-major order (batch_size * n * n)
    /// * `rhs` - Flattened RHS vectors (batch_size * n)
    /// * `n` - Matrix/vector dimension
    /// * `batch_size` - Number of systems to solve
    ///
    /// # Returns
    /// Solutions and information about any singular systems.
    pub fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        let expected_matrix_len = batch_size * n * n;
        let expected_rhs_len = batch_size * n;

        if matrices.len() != expected_matrix_len {
            return Err(WgpuError::InvalidDimension(format!(
                "Expected {} matrix elements, got {}",
                expected_matrix_len,
                matrices.len()
            )));
        }

        if rhs.len() != expected_rhs_len {
            return Err(WgpuError::InvalidDimension(format!(
                "Expected {} RHS elements, got {}",
                expected_rhs_len,
                rhs.len()
            )));
        }

        if n > self.config.max_matrix_size {
            return Err(WgpuError::InvalidDimension(format!(
                "Matrix size {} exceeds maximum {}",
                n, self.config.max_matrix_size
            )));
        }

        if batch_size == 0 {
            return Ok(BatchedSolveResult {
                solutions: vec![],
                singular_indices: vec![],
                n,
                batch_size: 0,
            });
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Create batch layout for aligned memory access
        let layout = BatchLayout::new(n, batch_size);
        let row_stride = layout.padded_row_stride();
        let matrix_stride = layout.padded_matrix_size();

        // Convert f64 -> f32 with col-major -> row-major transpose and alignment padding
        let matrices_f32 = pack_matrices_f32(matrices, n, batch_size, &layout);
        let rhs_f32 = pack_rhs_f32(rhs);

        // Calculate required buffer capacities
        let needed_matrix_elems = matrices_f32.len();
        let needed_rhs_elems = rhs_f32.len();
        let needed_info_elems = batch_size;

        // Acquire write lock to check and potentially update cache
        let mut cache = self.cached.write().unwrap();

        // Check if we need to reallocate buffers
        let need_buffer_realloc =
            !cache.can_accommodate(needed_matrix_elems, needed_rhs_elems, needed_info_elems);

        if need_buffer_realloc {
            // Allocate with 2x headroom for future growth
            let new_matrix_cap = (needed_matrix_elems * 2).max(MIN_BUFFER_ELEMENTS);
            let new_rhs_cap = (needed_rhs_elems * 2).max(MIN_BUFFER_ELEMENTS);
            let new_info_cap = (needed_info_elems * 2).max(MIN_BUFFER_ELEMENTS);

            log::debug!(
                "Reallocating GPU buffers: matrix {} -> {}, rhs {} -> {}, info {} -> {}",
                cache.matrix_capacity,
                new_matrix_cap,
                cache.rhs_capacity,
                new_rhs_cap,
                cache.info_capacity,
                new_info_cap
            );

            // Create matrix buffer (read-write, modified during factorization)
            cache.matrix_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batched LU Matrices (Cached)"),
                size: (new_matrix_cap * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Create RHS buffer (will also hold solutions)
            cache.rhs_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batched LU RHS (Cached)"),
                size: (new_rhs_cap * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Create info buffer
            cache.info_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Batched LU Info (Cached)"),
                size: (new_info_cap * std::mem::size_of::<i32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Create staging buffers for reading results
            cache.solution_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Solution Staging (Cached)"),
                size: (new_rhs_cap * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            cache.info_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Info Staging (Cached)"),
                size: (new_info_cap * std::mem::size_of::<i32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            cache.matrix_capacity = new_matrix_cap;
            cache.rhs_capacity = new_rhs_cap;
            cache.info_capacity = new_info_cap;

            // Invalidate bind group since buffers changed
            cache.bind_group = None;
            cache.uniform_buffer = None;
        }

        // Check if we need to recreate bind group (buffers changed or dimensions changed)
        let need_bind_group_update = need_buffer_realloc
            || !cache.can_reuse_bind_group(n, batch_size, row_stride, matrix_stride);

        if need_bind_group_update {
            // Create uniform buffer with stride information for shader
            let uniforms = Uniforms {
                n: n as u32,
                batch_size: batch_size as u32,
                row_stride: row_stride as u32,
                matrix_stride: matrix_stride as u32,
            };

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched LU Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batched LU Bind Group (Cached)"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cache.matrix_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cache.rhs_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cache.info_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            });

            cache.uniform_buffer = Some(uniform_buffer);
            cache.bind_group = Some(bind_group);
            cache.cached_n = n;
            cache.cached_batch_size = batch_size;
            cache.cached_row_stride = row_stride;
            cache.cached_matrix_stride = matrix_stride;
        }

        // Write data to cached buffers
        queue.write_buffer(
            cache.matrix_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&matrices_f32),
        );
        queue.write_buffer(
            cache.rhs_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&rhs_f32),
        );

        // Clear info buffer (reset singularity flags)
        let info_zeros: Vec<i32> = vec![0; batch_size];
        queue.write_buffer(
            cache.info_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&info_zeros),
        );

        // Get references to cached resources
        let bind_group = cache.bind_group.as_ref().unwrap();
        let rhs_buffer = cache.rhs_buffer.as_ref().unwrap();
        let info_buffer = cache.info_buffer.as_ref().unwrap();
        let solution_staging = cache.solution_staging.as_ref().unwrap();
        let info_staging = cache.info_staging.as_ref().unwrap();

        // Encode and submit compute work
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batched LU Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched LU Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            // One workgroup per matrix in the batch
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            rhs_buffer,
            0,
            solution_staging,
            0,
            (expected_rhs_len * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            info_buffer,
            0,
            info_staging,
            0,
            (batch_size * std::mem::size_of::<i32>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Drop cache lock before blocking on GPU readback
        drop(cache);

        // Re-acquire read lock to access staging buffers for readback
        let cache = self.cached.read().unwrap();
        let solution_staging = cache.solution_staging.as_ref().unwrap();
        let info_staging = cache.info_staging.as_ref().unwrap();

        // Read solutions
        let solutions = {
            let buffer_slice = solution_staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .map_err(|e| WgpuError::Buffer(format!("Failed to receive map result: {}", e)))?
                .map_err(|e| WgpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

            let data = buffer_slice.get_mapped_range();
            let solutions_f32: &[f32] = bytemuck::cast_slice(&data);
            let solutions = unpack_solutions_f64(&solutions_f32[..expected_rhs_len]);
            drop(data);
            solution_staging.unmap();
            solutions
        };

        // Read info
        let singular_indices = {
            let buffer_slice = info_staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .map_err(|e| WgpuError::Buffer(format!("Failed to receive map result: {}", e)))?
                .map_err(|e| WgpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

            let data = buffer_slice.get_mapped_range();
            let info_array: &[i32] = bytemuck::cast_slice(&data);
            let singular: Vec<usize> = info_array[..batch_size]
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > 0 { Some(i) } else { None })
                .collect();
            drop(data);
            info_staging.unmap();
            singular
        };

        if !singular_indices.is_empty() {
            log::warn!(
                "{} of {} matrices were singular",
                singular_indices.len(),
                batch_size
            );
        }

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_context() -> Option<Arc<WgpuContext>> {
        WgpuContext::new().ok().map(Arc::new)
    }

    #[test]
    fn test_batched_lu_identity() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();
        let n = 2;
        let batch_size = 2;

        // Two 2x2 identity matrices in column-major order
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];

        let rhs = vec![
            1.0, 2.0, // b0 = [1, 2]
            3.0, 4.0, // b1 = [3, 4]
        ];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        let sol0 = result.solution(0).unwrap();
        assert!((sol0[0] - 1.0).abs() < 1e-4, "sol0[0] = {} (expected 1.0)", sol0[0]);
        assert!((sol0[1] - 2.0).abs() < 1e-4, "sol0[1] = {} (expected 2.0)", sol0[1]);

        let sol1 = result.solution(1).unwrap();
        assert!((sol1[0] - 3.0).abs() < 1e-4, "sol1[0] = {} (expected 3.0)", sol1[0]);
        assert!((sol1[1] - 4.0).abs() < 1e-4, "sol1[1] = {} (expected 4.0)", sol1[1]);
    }

    #[test]
    fn test_batched_lu_simple() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();
        let n = 2;
        let batch_size = 1;

        // Matrix: [[2, 1], [1, 3]] in column-major: [2, 1, 1, 3]
        // Solving Ax = b where b = [5, 5]
        // Solution should be x = [2, 1]
        let matrices = vec![2.0, 1.0, 1.0, 3.0];
        let rhs = vec![5.0, 5.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.singular_indices.is_empty());

        let sol = result.solution(0).unwrap();
        assert!((sol[0] - 2.0).abs() < 1e-4, "x[0] = {} (expected 2.0)", sol[0]);
        assert!((sol[1] - 1.0).abs() < 1e-4, "x[1] = {} (expected 1.0)", sol[1]);
    }

    #[test]
    fn test_batched_lu_singular() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();
        let n = 2;
        let batch_size = 2;

        // Matrix 0: identity (non-singular)
        // Matrix 1: [[1, 2], [1, 2]] (singular - rows are identical) in column-major: [1, 1, 2, 2]
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity
            1.0, 1.0, 2.0, 2.0, // Singular
        ];
        let rhs = vec![1.0, 2.0, 1.0, 2.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.is_singular(1), "Matrix 1 should be detected as singular");
        assert!(!result.is_singular(0), "Matrix 0 should not be singular");
    }

    #[test]
    fn test_config_thresholds() {
        let config = GpuBatchConfig::default();

        // Default thresholds: min_batch=2000, min_matrix=100
        assert!(!config.should_use_gpu(50, 100)); // Matrix too small
        assert!(!config.should_use_gpu(100, 100)); // Batch too small
        assert!(config.should_use_gpu(100, 2000)); // Both OK
        assert!(!config.should_use_gpu(200, 2000)); // Matrix too large
    }

    #[test]
    fn test_buffer_caching() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();

        // Initially no buffers cached
        assert!(!solver.has_cached_buffers());

        let n = 2;
        let batch_size = 2;
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        // First solve creates buffers
        let result1 = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();
        assert!(solver.has_cached_buffers());
        assert_eq!(result1.batch_size, 2);

        // Second solve reuses buffers (same size)
        let result2 = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();
        assert!(solver.has_cached_buffers());
        assert_eq!(result2.batch_size, 2);

        // Verify both give correct results
        let sol1 = result1.solution(0).unwrap();
        let sol2 = result2.solution(0).unwrap();
        assert!((sol1[0] - sol2[0]).abs() < 1e-10);
        assert!((sol1[1] - sol2[1]).abs() < 1e-10);

        // Clear cache
        solver.clear_cache();
        assert!(!solver.has_cached_buffers());

        // Solve again after clearing
        let result3 = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();
        assert!(solver.has_cached_buffers());
        let sol3 = result3.solution(0).unwrap();
        assert!((sol1[0] - sol3[0]).abs() < 1e-10);
        assert!((sol1[1] - sol3[1]).abs() < 1e-10);
    }

    #[test]
    fn test_buffer_reallocation_on_size_increase() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();

        // Start with small batch
        let n = 2;
        let small_matrices = vec![1.0, 0.0, 0.0, 1.0];
        let small_rhs = vec![1.0, 2.0];

        let result1 = solver.solve_batch(&small_matrices, &small_rhs, n, 1).unwrap();
        assert!(solver.has_cached_buffers());
        let sol1 = result1.solution(0).unwrap();
        assert!((sol1[0] - 1.0).abs() < 1e-4);
        assert!((sol1[1] - 2.0).abs() < 1e-4);

        // Now use larger batch - should reallocate if needed
        let large_matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
            1.0, 0.0, 0.0, 1.0, // Identity 2
            1.0, 0.0, 0.0, 1.0, // Identity 3
        ];
        let large_rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result2 = solver.solve_batch(&large_matrices, &large_rhs, n, 4).unwrap();
        assert_eq!(result2.batch_size, 4);

        // Verify all solutions correct
        for i in 0..4 {
            let sol = result2.solution(i).unwrap();
            let expected_0 = (i * 2 + 1) as f64;
            let expected_1 = (i * 2 + 2) as f64;
            assert!((sol[0] - expected_0).abs() < 1e-4,
                "batch {}: sol[0] = {} (expected {})", i, sol[0], expected_0);
            assert!((sol[1] - expected_1).abs() < 1e-4,
                "batch {}: sol[1] = {} (expected {})", i, sol[1], expected_1);
        }
    }

    #[test]
    fn test_bind_group_update_on_batch_size_change() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();

        // First solve with batch_size = 4
        let n = 2;
        let matrices_4 = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
            1.0, 0.0, 0.0, 1.0, // Identity 2
            1.0, 0.0, 0.0, 1.0, // Identity 3
        ];
        let rhs_4 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result1 = solver.solve_batch(&matrices_4, &rhs_4, n, 4).unwrap();
        assert_eq!(result1.batch_size, 4);

        // Now solve with smaller batch_size = 2 (buffers are large enough, but
        // batch_size in uniform buffer must change)
        let matrices_2 = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];
        let rhs_2 = vec![10.0, 20.0, 30.0, 40.0];

        let result2 = solver.solve_batch(&matrices_2, &rhs_2, n, 2).unwrap();
        assert_eq!(result2.batch_size, 2);

        // Verify solutions are correct for the new batch
        let sol0 = result2.solution(0).unwrap();
        assert!((sol0[0] - 10.0).abs() < 1e-4, "sol0[0] = {} (expected 10.0)", sol0[0]);
        assert!((sol0[1] - 20.0).abs() < 1e-4, "sol0[1] = {} (expected 20.0)", sol0[1]);

        let sol1 = result2.solution(1).unwrap();
        assert!((sol1[0] - 30.0).abs() < 1e-4, "sol1[0] = {} (expected 30.0)", sol1[0]);
        assert!((sol1[1] - 40.0).abs() < 1e-4, "sol1[1] = {} (expected 40.0)", sol1[1]);
    }
}
