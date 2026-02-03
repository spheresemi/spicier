//! GPU matrix assembly for parallel stamping across sweep points.
//!
//! For parameter sweeps, all matrices share the same sparsity pattern - only
//! values differ. This module provides GPU kernels for:
//! 1. Assembling device contributions into sparse matrices
//! 2. Building RHS vectors from device equivalent currents
//!
//! # Architecture
//!
//! ```text
//! CPU (once):
//!   - Compute sparsity pattern (CSR indices)
//!   - Compute stamp locations for each device
//!   - Upload indices and stamp maps to GPU
//!
//! GPU (per NR iteration):
//!   - Device evaluation (9c-1) produces Id, gd, Ieq for each device × sweep
//!   - Matrix assembly kernel stamps conductances into CSR values
//!   - RHS assembly kernel stamps equivalent currents
//! ```

use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Stamp location for a single conductance contribution.
///
/// A conductance G between nodes i and j contributes to 4 matrix locations:
/// - (i,i): +G
/// - (i,j): -G
/// - (j,i): -G
/// - (j,j): +G
///
/// For nodes connected to ground, some entries are omitted.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ConductanceStamp {
    /// CSR value index for (i,i) entry, or u32::MAX if grounded.
    pub idx_ii: u32,
    /// CSR value index for (i,j) entry, or u32::MAX if grounded.
    pub idx_ij: u32,
    /// CSR value index for (j,i) entry, or u32::MAX if grounded.
    pub idx_ji: u32,
    /// CSR value index for (j,j) entry, or u32::MAX if grounded.
    pub idx_jj: u32,
}

impl ConductanceStamp {
    /// Create a stamp with all indices invalid (for ground connections).
    pub fn ground() -> Self {
        Self {
            idx_ii: u32::MAX,
            idx_ij: u32::MAX,
            idx_ji: u32::MAX,
            idx_jj: u32::MAX,
        }
    }

    /// Create a stamp for a conductance between two non-ground nodes.
    pub fn new(idx_ii: u32, idx_ij: u32, idx_ji: u32, idx_jj: u32) -> Self {
        Self {
            idx_ii,
            idx_ij,
            idx_ji,
            idx_jj,
        }
    }
}

/// Stamp location for a current source contribution to RHS.
///
/// A current source I from node i to node j contributes:
/// - RHS[i]: -I (current leaving node i)
/// - RHS[j]: +I (current entering node j)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CurrentStamp {
    /// RHS index for positive node, or u32::MAX if grounded.
    pub idx_pos: u32,
    /// RHS index for negative node, or u32::MAX if grounded.
    pub idx_neg: u32,
    /// Padding for alignment.
    pub _pad: [u32; 2],
}

impl CurrentStamp {
    /// Create a stamp for a current source.
    pub fn new(idx_pos: u32, idx_neg: u32) -> Self {
        Self {
            idx_pos,
            idx_neg,
            _pad: [0; 2],
        }
    }
}

/// GPU kernel for assembling conductances into CSR matrix values.
pub struct GpuMatrixAssembler {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuMatrixAssembler {
    /// Create a new matrix assembler.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        let shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Matrix Assembly Shader"),
                source: wgpu::ShaderSource::Wgsl(MATRIX_ASSEMBLY_SHADER.into()),
            });

        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Matrix Assembly Bind Group Layout"),
                    entries: &[
                        // Uniforms (num_devices, num_sweeps, nnz)
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
                        // Conductance stamps (per device)
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
                        // Conductance values (per device × sweep)
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
                        // CSR values output (per sweep × nnz)
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

        let pipeline_layout =
            ctx.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Matrix Assembly Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Matrix Assembly Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("assemble_matrix"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            ctx,
            pipeline,
            bind_group_layout,
        })
    }

    /// Assemble conductances into CSR matrix values for all sweep points.
    ///
    /// # Arguments
    /// * `stamps` - Conductance stamp locations (one per device)
    /// * `conductances` - Conductance values (num_devices × num_sweeps)
    /// * `num_sweeps` - Number of sweep points
    /// * `nnz` - Number of non-zeros in the CSR matrix
    ///
    /// # Returns
    /// CSR values array (num_sweeps × nnz)
    pub fn assemble(
        &self,
        stamps: &[ConductanceStamp],
        conductances: &[f32],
        num_sweeps: usize,
        nnz: usize,
    ) -> Result<Vec<f32>> {
        let num_devices = stamps.len();
        if conductances.len() != num_devices * num_sweeps {
            return Err(WgpuError::InvalidDimension(format!(
                "conductances length {} != num_devices {} × num_sweeps {}",
                conductances.len(),
                num_devices,
                num_sweeps
            )));
        }

        // Uniforms
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Uniforms {
            num_devices: u32,
            num_sweeps: u32,
            nnz: u32,
            _pad: u32,
        }
        let uniforms = Uniforms {
            num_devices: num_devices as u32,
            num_sweeps: num_sweeps as u32,
            nnz: nnz as u32,
            _pad: 0,
        };

        let uniform_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Assembly Uniforms"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let stamps_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Conductance Stamps"),
                    contents: bytemuck::cast_slice(stamps),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let conductances_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Conductances"),
                    contents: bytemuck::cast_slice(conductances),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_size = (num_sweeps * nnz * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("CSR Values"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("CSR Values Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Matrix Assembly Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: stamps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: conductances_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Matrix Assembly Encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matrix Assembly Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per (device, sweep) pair
            let total_work = (num_devices * num_sweeps) as u32;
            let workgroups = total_work.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
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

/// GPU kernel for assembling currents into RHS vectors.
pub struct GpuRhsAssembler {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuRhsAssembler {
    /// Create a new RHS assembler.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        let shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RHS Assembly Shader"),
                source: wgpu::ShaderSource::Wgsl(RHS_ASSEMBLY_SHADER.into()),
            });

        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RHS Assembly Bind Group Layout"),
                    entries: &[
                        // Uniforms (num_devices, num_sweeps, rhs_size)
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
                        // Current stamps (per device)
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
                        // Current values (per device × sweep)
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
                        // RHS output (per sweep × rhs_size)
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

        let pipeline_layout =
            ctx.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RHS Assembly Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RHS Assembly Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("assemble_rhs"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            ctx,
            pipeline,
            bind_group_layout,
        })
    }

    /// Assemble currents into RHS vectors for all sweep points.
    ///
    /// # Arguments
    /// * `stamps` - Current stamp locations (one per device)
    /// * `currents` - Current values (num_devices × num_sweeps)
    /// * `num_sweeps` - Number of sweep points
    /// * `rhs_size` - Size of the RHS vector (number of nodes)
    ///
    /// # Returns
    /// RHS vectors array (num_sweeps × rhs_size)
    pub fn assemble(
        &self,
        stamps: &[CurrentStamp],
        currents: &[f32],
        num_sweeps: usize,
        rhs_size: usize,
    ) -> Result<Vec<f32>> {
        let num_devices = stamps.len();
        if currents.len() != num_devices * num_sweeps {
            return Err(WgpuError::InvalidDimension(format!(
                "currents length {} != num_devices {} × num_sweeps {}",
                currents.len(),
                num_devices,
                num_sweeps
            )));
        }

        // Uniforms
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Uniforms {
            num_devices: u32,
            num_sweeps: u32,
            rhs_size: u32,
            _pad: u32,
        }
        let uniforms = Uniforms {
            num_devices: num_devices as u32,
            num_sweeps: num_sweeps as u32,
            rhs_size: rhs_size as u32,
            _pad: 0,
        };

        let uniform_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RHS Assembly Uniforms"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let stamps_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Current Stamps"),
                    contents: bytemuck::cast_slice(stamps),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let currents_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Currents"),
                    contents: bytemuck::cast_slice(currents),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_size = (num_sweeps * rhs_size * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("RHS Values"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("RHS Values Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RHS Assembly Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: stamps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: currents_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("RHS Assembly Encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RHS Assembly Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one thread per (device, sweep) pair
            let total_work = (num_devices * num_sweeps) as u32;
            let workgroups = total_work.div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
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

/// WGSL shader for RHS assembly.
///
/// Each thread handles one (device, sweep) pair, stamping the current
/// into the appropriate RHS locations using atomic adds.
const RHS_ASSEMBLY_SHADER: &str = r#"
struct Uniforms {
    num_devices: u32,
    num_sweeps: u32,
    rhs_size: u32,
    _pad: u32,
}

struct CurrentStamp {
    idx_pos: u32,
    idx_neg: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> stamps: array<CurrentStamp>;
@group(0) @binding(2) var<storage, read> currents: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhs_values: array<atomic<u32>>;

@compute @workgroup_size(256)
fn assemble_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = uniforms.num_devices * uniforms.num_sweeps;
    if idx >= total {
        return;
    }

    // Decode (device, sweep) from linear index
    let device_idx = idx / uniforms.num_sweeps;
    let sweep_idx = idx % uniforms.num_sweeps;

    // Get stamp locations for this device
    let stamp = stamps[device_idx];

    // Get current value for this (device, sweep)
    let i = currents[idx];

    // Compute base offset for this sweep's RHS vector
    let sweep_offset = sweep_idx * uniforms.rhs_size;

    // Stamp current into RHS (handling ground nodes)
    let invalid = 0xFFFFFFFFu;

    // RHS[pos] += I (current entering positive node)
    if stamp.idx_pos != invalid {
        let rhs_idx = sweep_offset + stamp.idx_pos;
        var old_bits = atomicLoad(&rhs_values[rhs_idx]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + i;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&rhs_values[rhs_idx], old_bits, new_bits);
            if result.exchanged { break; }
            old_bits = result.old_value;
        }
    }

    // RHS[neg] -= I (current leaving negative node)
    if stamp.idx_neg != invalid {
        let rhs_idx = sweep_offset + stamp.idx_neg;
        var old_bits = atomicLoad(&rhs_values[rhs_idx]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val - i;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&rhs_values[rhs_idx], old_bits, new_bits);
            if result.exchanged { break; }
            old_bits = result.old_value;
        }
    }
}
"#;

/// WGSL shader for matrix assembly.
///
/// Each thread handles one (device, sweep) pair, stamping the conductance
/// into the appropriate CSR value locations using atomic adds.
const MATRIX_ASSEMBLY_SHADER: &str = r#"
struct Uniforms {
    num_devices: u32,
    num_sweeps: u32,
    nnz: u32,
    _pad: u32,
}

struct ConductanceStamp {
    idx_ii: u32,
    idx_ij: u32,
    idx_ji: u32,
    idx_jj: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> stamps: array<ConductanceStamp>;
@group(0) @binding(2) var<storage, read> conductances: array<f32>;
@group(0) @binding(3) var<storage, read_write> csr_values: array<atomic<u32>>;

@compute @workgroup_size(256)
fn assemble_matrix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = uniforms.num_devices * uniforms.num_sweeps;
    if idx >= total {
        return;
    }

    // Decode (device, sweep) from linear index
    let device_idx = idx / uniforms.num_sweeps;
    let sweep_idx = idx % uniforms.num_sweeps;

    // Get stamp locations for this device
    let stamp = stamps[device_idx];

    // Get conductance value for this (device, sweep)
    let g = conductances[idx];

    // Compute base offset for this sweep's CSR values
    let sweep_offset = sweep_idx * uniforms.nnz;

    // Stamp conductance into matrix (handling ground nodes)
    // Use atomic compare-exchange loop for f32 addition
    let invalid = 0xFFFFFFFFu;

    // Stamp (i,i): +G
    if stamp.idx_ii != invalid {
        let csr_idx = sweep_offset + stamp.idx_ii;
        var old_bits = atomicLoad(&csr_values[csr_idx]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + g;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&csr_values[csr_idx], old_bits, new_bits);
            if result.exchanged { break; }
            old_bits = result.old_value;
        }
    }

    // Stamp (i,j): -G
    if stamp.idx_ij != invalid {
        let csr_idx = sweep_offset + stamp.idx_ij;
        var old_bits = atomicLoad(&csr_values[csr_idx]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val - g;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&csr_values[csr_idx], old_bits, new_bits);
            if result.exchanged { break; }
            old_bits = result.old_value;
        }
    }

    // Stamp (j,i): -G
    if stamp.idx_ji != invalid {
        let csr_idx = sweep_offset + stamp.idx_ji;
        var old_bits = atomicLoad(&csr_values[csr_idx]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val - g;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&csr_values[csr_idx], old_bits, new_bits);
            if result.exchanged { break; }
            old_bits = result.old_value;
        }
    }

    // Stamp (j,j): +G
    if stamp.idx_jj != invalid {
        let csr_idx = sweep_offset + stamp.idx_jj;
        var old_bits = atomicLoad(&csr_values[csr_idx]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + g;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&csr_values[csr_idx], old_bits, new_bits);
            if result.exchanged { break; }
            old_bits = result.old_value;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_context() -> Result<Arc<WgpuContext>> {
        Ok(Arc::new(WgpuContext::new()?))
    }

    #[test]
    fn test_simple_resistor_assembly() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuMatrixAssembler::new(ctx).unwrap();

        // Simple 2-node circuit: resistor between node 0 and node 1
        // Matrix is 2×2 with 4 non-zeros:
        // [G, -G]
        // [-G, G]
        let stamps = vec![
            ConductanceStamp::new(0, 1, 2, 3), // indices into CSR values
        ];

        // 2 sweep points with different conductance values
        let conductances = vec![
            1.0, // sweep 0: G = 1.0
            2.0, // sweep 1: G = 2.0
        ];

        let result = assembler.assemble(&stamps, &conductances, 2, 4).unwrap();

        // Expected CSR values for sweep 0: [1, -1, -1, 1]
        // Expected CSR values for sweep 1: [2, -2, -2, 2]
        assert_eq!(result.len(), 8); // 2 sweeps × 4 nnz

        // Sweep 0
        assert!((result[0] - 1.0).abs() < 1e-6, "result[0] = {}", result[0]);
        assert!(
            (result[1] - (-1.0)).abs() < 1e-6,
            "result[1] = {}",
            result[1]
        );
        assert!(
            (result[2] - (-1.0)).abs() < 1e-6,
            "result[2] = {}",
            result[2]
        );
        assert!((result[3] - 1.0).abs() < 1e-6, "result[3] = {}", result[3]);

        // Sweep 1
        assert!((result[4] - 2.0).abs() < 1e-6, "result[4] = {}", result[4]);
        assert!(
            (result[5] - (-2.0)).abs() < 1e-6,
            "result[5] = {}",
            result[5]
        );
        assert!(
            (result[6] - (-2.0)).abs() < 1e-6,
            "result[6] = {}",
            result[6]
        );
        assert!((result[7] - 2.0).abs() < 1e-6, "result[7] = {}", result[7]);
    }

    #[test]
    fn test_ground_connection() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuMatrixAssembler::new(ctx).unwrap();

        // Resistor from node 0 to ground
        // Only stamps into (0,0) position
        let stamps = vec![ConductanceStamp {
            idx_ii: 0,
            idx_ij: u32::MAX, // ground
            idx_ji: u32::MAX, // ground
            idx_jj: u32::MAX, // ground
        }];

        let conductances = vec![1.0];
        let result = assembler.assemble(&stamps, &conductances, 1, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_devices_same_node() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuMatrixAssembler::new(ctx).unwrap();

        // Two resistors to ground from node 0 - should sum
        let stamps = vec![
            ConductanceStamp {
                idx_ii: 0,
                idx_ij: u32::MAX,
                idx_ji: u32::MAX,
                idx_jj: u32::MAX,
            },
            ConductanceStamp {
                idx_ii: 0, // Same location - needs atomic add
                idx_ij: u32::MAX,
                idx_ji: u32::MAX,
                idx_jj: u32::MAX,
            },
        ];

        let conductances = vec![
            1.0, // device 0, sweep 0
            2.0, // device 1, sweep 0
        ];
        let result = assembler.assemble(&stamps, &conductances, 1, 1).unwrap();

        assert_eq!(result.len(), 1);
        // Should be sum of both conductances
        assert!(
            (result[0] - 3.0).abs() < 1e-6,
            "result[0] = {} (expected 3.0)",
            result[0]
        );
    }

    #[test]
    fn test_assembly_performance() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuMatrixAssembler::new(ctx).unwrap();

        // Simulate a circuit with 1000 devices, 1000 sweeps, 5000 nnz
        let num_devices = 1000;
        let num_sweeps = 1000;
        let nnz = 5000;

        // Create random-ish stamps
        let stamps: Vec<ConductanceStamp> = (0..num_devices)
            .map(|i| {
                let base = (i * 4) % nnz;
                ConductanceStamp::new(
                    base as u32,
                    ((base + 1) % nnz) as u32,
                    ((base + 2) % nnz) as u32,
                    ((base + 3) % nnz) as u32,
                )
            })
            .collect();

        let conductances: Vec<f32> = (0..num_devices * num_sweeps)
            .map(|i| (i as f32) * 0.001)
            .collect();

        let start = std::time::Instant::now();
        let result = assembler
            .assemble(&stamps, &conductances, num_sweeps, nnz)
            .unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result.len(), num_sweeps * nnz);
        println!(
            "GPU matrix assembly: {} devices × {} sweeps = {}M stamps in {:?} ({:.2}M stamps/sec)",
            num_devices,
            num_sweeps,
            num_devices * num_sweeps / 1_000_000,
            elapsed,
            (num_devices * num_sweeps) as f64 / elapsed.as_secs_f64() / 1e6
        );
    }

    // RHS Assembly Tests

    #[test]
    fn test_simple_current_source() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuRhsAssembler::new(ctx).unwrap();

        // Current source from node 0 to node 1 (I = 1A)
        // RHS[0] += 1, RHS[1] -= 1
        let stamps = vec![CurrentStamp::new(0, 1)];

        // 2 sweep points with different current values
        let currents = vec![
            1.0, // sweep 0: I = 1.0
            2.0, // sweep 1: I = 2.0
        ];

        let result = assembler.assemble(&stamps, &currents, 2, 2).unwrap();

        // Expected RHS for sweep 0: [1, -1]
        // Expected RHS for sweep 1: [2, -2]
        assert_eq!(result.len(), 4); // 2 sweeps × 2 nodes

        // Sweep 0
        assert!((result[0] - 1.0).abs() < 1e-6, "result[0] = {}", result[0]);
        assert!(
            (result[1] - (-1.0)).abs() < 1e-6,
            "result[1] = {}",
            result[1]
        );

        // Sweep 1
        assert!((result[2] - 2.0).abs() < 1e-6, "result[2] = {}", result[2]);
        assert!(
            (result[3] - (-2.0)).abs() < 1e-6,
            "result[3] = {}",
            result[3]
        );
    }

    #[test]
    fn test_current_to_ground() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuRhsAssembler::new(ctx).unwrap();

        // Current source from node 0 to ground
        // Only RHS[0] is affected
        let stamps = vec![
            CurrentStamp::new(0, u32::MAX), // negative node is ground
        ];

        let currents = vec![1.0];
        let result = assembler.assemble(&stamps, &currents, 1, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_currents_same_node() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuRhsAssembler::new(ctx).unwrap();

        // Two current sources into node 0 - should sum
        let stamps = vec![
            CurrentStamp::new(0, u32::MAX),
            CurrentStamp::new(0, u32::MAX), // Same location - needs atomic add
        ];

        let currents = vec![
            1.0, // device 0, sweep 0
            2.0, // device 1, sweep 0
        ];
        let result = assembler.assemble(&stamps, &currents, 1, 1).unwrap();

        assert_eq!(result.len(), 1);
        // Should be sum of both currents
        assert!(
            (result[0] - 3.0).abs() < 1e-6,
            "result[0] = {} (expected 3.0)",
            result[0]
        );
    }

    #[test]
    fn test_rhs_assembly_performance() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let assembler = GpuRhsAssembler::new(ctx).unwrap();

        // Simulate a circuit with 1000 devices, 1000 sweeps, 100 nodes
        let num_devices = 1000;
        let num_sweeps = 1000;
        let rhs_size = 100;

        // Create stamps with varying node connections
        let stamps: Vec<CurrentStamp> = (0..num_devices)
            .map(|i| CurrentStamp::new((i % rhs_size) as u32, ((i + 1) % rhs_size) as u32))
            .collect();

        let currents: Vec<f32> = (0..num_devices * num_sweeps)
            .map(|i| (i as f32) * 0.001)
            .collect();

        let start = std::time::Instant::now();
        let result = assembler
            .assemble(&stamps, &currents, num_sweeps, rhs_size)
            .unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result.len(), num_sweeps * rhs_size);
        println!(
            "GPU RHS assembly: {} devices × {} sweeps = {}M stamps in {:?} ({:.2}M stamps/sec)",
            num_devices,
            num_sweeps,
            num_devices * num_sweeps / 1_000_000,
            elapsed,
            (num_devices * num_sweeps) as f64 / elapsed.as_secs_f64() / 1e6
        );
    }
}
