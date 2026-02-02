//! GPU device evaluation kernels for massively parallel sweeps.
//!
//! This module provides GPU kernels for evaluating device models (MOSFET, diode, BJT)
//! across thousands of sweep points simultaneously. The key insight is that device
//! evaluation is embarrassingly parallel - each device × sweep point is independent.
//!
//! # Data Layout
//!
//! We use Structure-of-Arrays (SoA) layout for coalesced GPU memory access:
//! - All Vgs values together, all Vds values together, etc.
//! - Sweep-major ordering: device[0] for all sweeps, then device[1], etc.
//!
//! # Performance Target
//!
//! 10M device evaluations in <1ms on modern GPUs.

use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// MOSFET model parameters for GPU evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMosfetParams {
    /// Threshold voltage (V).
    pub vto: f32,
    /// Transconductance parameter (A/V²).
    pub kp: f32,
    /// Channel-length modulation (1/V).
    pub lambda: f32,
    /// Body effect coefficient (V^0.5).
    pub gamma: f32,
    /// Surface potential (V).
    pub phi: f32,
    /// +1 for NMOS, -1 for PMOS.
    pub polarity: f32,
    /// Padding for alignment.
    pub _pad: [f32; 2],
}

impl Default for GpuMosfetParams {
    fn default() -> Self {
        Self {
            vto: 0.7,
            kp: 110e-6,
            lambda: 0.04,
            gamma: 0.4,
            phi: 0.65,
            polarity: 1.0, // NMOS
            _pad: [0.0; 2],
        }
    }
}

/// Results from MOSFET evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MosfetEvalResult {
    /// Drain current (A).
    pub id: f32,
    /// Transconductance dId/dVgs (S).
    pub gm: f32,
    /// Output conductance dId/dVds (S).
    pub gds: f32,
    /// Body transconductance dId/dVbs (S).
    pub gmb: f32,
}

/// GPU kernel for batched MOSFET evaluation.
pub struct GpuMosfetEvaluator {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuMosfetEvaluator {
    /// Create a new MOSFET evaluator.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        let shader = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOSFET Evaluation Shader"),
            source: wgpu::ShaderSource::Wgsl(MOSFET_SHADER.into()),
        });

        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MOSFET Eval Bind Group Layout"),
                    entries: &[
                        // Params buffer (uniform)
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
                        // Vgs input buffer
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
                        // Vds input buffer
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
                        // Vbs input buffer
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
                        // Output buffer (Id, gm, gds, gmb)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                    label: Some("MOSFET Eval Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MOSFET Eval Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("eval_mosfet"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            ctx,
            pipeline,
            bind_group_layout,
        })
    }

    /// Evaluate MOSFETs for all devices × all sweep points.
    ///
    /// # Arguments
    /// * `params` - MOSFET model parameters (shared across all devices of this type)
    /// * `vgs` - Gate-source voltages, length = num_devices × num_sweeps
    /// * `vds` - Drain-source voltages, length = num_devices × num_sweeps
    /// * `vbs` - Bulk-source voltages, length = num_devices × num_sweeps
    ///
    /// # Returns
    /// Results array with Id, gm, gds, gmb for each device × sweep.
    pub fn evaluate(
        &self,
        params: &GpuMosfetParams,
        vgs: &[f32],
        vds: &[f32],
        vbs: &[f32],
    ) -> Result<Vec<MosfetEvalResult>> {
        let count = vgs.len();
        if vds.len() != count || vbs.len() != count {
            return Err(WgpuError::InvalidDimension(
                "vgs, vds, vbs must have same length".into(),
            ));
        }
        if count == 0 {
            return Ok(vec![]);
        }

        // Create uniform buffer for params
        let params_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MOSFET Params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create input buffers
        let vgs_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vgs Input"),
                contents: bytemuck::cast_slice(vgs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vds_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vds Input"),
                contents: bytemuck::cast_slice(vds),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vbs_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vbs Input"),
                contents: bytemuck::cast_slice(vbs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer
        let output_size = (count * std::mem::size_of::<MosfetEvalResult>()) as u64;
        let output_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOSFET Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOSFET Output Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MOSFET Eval Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vgs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vds_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vbs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch compute
        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MOSFET Eval Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MOSFET Eval Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 256 threads per workgroup
            let workgroups = (count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy to staging
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.ctx.queue().submit(std::iter::once(encoder.finish()));

        // Read back results
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
        let results: Vec<MosfetEvalResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Evaluate with timing information for benchmarking.
    pub fn evaluate_timed(
        &self,
        params: &GpuMosfetParams,
        vgs: &[f32],
        vds: &[f32],
        vbs: &[f32],
    ) -> Result<(Vec<MosfetEvalResult>, std::time::Duration)> {
        let start = std::time::Instant::now();
        let results = self.evaluate(params, vgs, vds, vbs)?;
        let elapsed = start.elapsed();
        Ok((results, elapsed))
    }
}

/// WGSL shader for MOSFET evaluation.
///
/// Level 1 MOSFET model with body effect. Computes Id, gm, gds, gmb.
const MOSFET_SHADER: &str = r#"
struct MosfetParams {
    vto: f32,
    kp: f32,
    lambda: f32,
    gamma: f32,
    phi: f32,
    polarity: f32,  // +1 for NMOS, -1 for PMOS
    _pad: vec2<f32>,
}

struct MosfetResult {
    id: f32,
    gm: f32,
    gds: f32,
    gmb: f32,
}

@group(0) @binding(0) var<uniform> params: MosfetParams;
@group(0) @binding(1) var<storage, read> vgs_in: array<f32>;
@group(0) @binding(2) var<storage, read> vds_in: array<f32>;
@group(0) @binding(3) var<storage, read> vbs_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> results: array<MosfetResult>;

// Smooth max function to avoid branching
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

// Smooth min function
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    return -smooth_max(-a, -b, k);
}

@compute @workgroup_size(256)
fn eval_mosfet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&vgs_in) {
        return;
    }

    // Read inputs and apply polarity for PMOS
    let p = params.polarity;
    let vgs = p * vgs_in[idx];
    let vds_raw = p * vds_in[idx];
    let vbs = p * vbs_in[idx];

    // Ensure vds >= 0 (swap drain/source if needed for symmetric model)
    let vds = abs(vds_raw);
    let swap_sign = select(1.0, -1.0, vds_raw < 0.0);

    // Body effect: threshold voltage shift
    let sqrt_phi = sqrt(params.phi);
    let sqrt_phi_vbs = sqrt(max(params.phi - vbs, 0.001));
    let vth = params.vto + params.gamma * (sqrt_phi_vbs - sqrt_phi);

    // Effective gate overdrive
    let vgs_eff = vgs - vth;

    // Smoothing parameter for region transitions
    let k = 0.01;

    // Cutoff check
    let in_cutoff = vgs_eff < 0.0;

    // Saturation voltage
    let vdsat = max(vgs_eff, 0.0);

    // Linear/saturation boundary (smooth transition)
    let vds_eff = smooth_min(vds, vdsat, k);

    // Drain current (quadratic model with channel-length modulation)
    // Linear: Id = Kp * (Vgs_eff * Vds - 0.5 * Vds^2) * (1 + lambda * Vds)
    // Saturation: Id = 0.5 * Kp * Vgs_eff^2 * (1 + lambda * Vds)
    let clm = 1.0 + params.lambda * vds;
    var id: f32;
    var gm: f32;
    var gds: f32;

    if in_cutoff {
        // Cutoff region - small leakage current for numerical stability
        id = 1e-12;
        gm = 1e-12;
        gds = 1e-12;
    } else if vds < vdsat {
        // Linear region
        id = params.kp * (vgs_eff * vds_eff - 0.5 * vds_eff * vds_eff) * clm;
        gm = params.kp * vds_eff * clm;
        gds = params.kp * (vgs_eff - vds_eff) * clm + params.kp * (vgs_eff * vds_eff - 0.5 * vds_eff * vds_eff) * params.lambda;
    } else {
        // Saturation region
        id = 0.5 * params.kp * vgs_eff * vgs_eff * clm;
        gm = params.kp * vgs_eff * clm;
        gds = 0.5 * params.kp * vgs_eff * vgs_eff * params.lambda;
    }

    // Body transconductance: gmb = gm * (gamma / (2 * sqrt(phi - vbs)))
    let gmb = gm * params.gamma / (2.0 * sqrt_phi_vbs);

    // Apply polarity and swap sign back
    let final_id = p * swap_sign * max(id, 0.0);
    let final_gm = max(gm, 1e-12);
    let final_gds = max(gds, 1e-12);
    let final_gmb = max(gmb, 1e-12);

    results[idx] = MosfetResult(final_id, final_gm, final_gds, final_gmb);
}
"#;

// ============================================================================
// Diode Evaluation
// ============================================================================

/// Diode model parameters for GPU evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuDiodeParams {
    /// Saturation current (A).
    pub is: f32,
    /// Emission coefficient.
    pub n: f32,
    /// Thermal voltage (V) - typically ~0.026V at room temp.
    pub vt: f32,
    /// Padding for alignment.
    pub _pad: f32,
}

impl Default for GpuDiodeParams {
    fn default() -> Self {
        Self {
            is: 1e-14,
            n: 1.0,
            vt: 0.02585, // 300K
            _pad: 0.0,
        }
    }
}

/// Results from diode evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DiodeEvalResult {
    /// Diode current (A).
    pub id: f32,
    /// Diode conductance dId/dVd (S).
    pub gd: f32,
}

/// GPU kernel for batched diode evaluation.
pub struct GpuDiodeEvaluator {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuDiodeEvaluator {
    /// Create a new diode evaluator.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        let shader = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Diode Evaluation Shader"),
            source: wgpu::ShaderSource::Wgsl(DIODE_SHADER.into()),
        });

        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Diode Eval Bind Group Layout"),
                    entries: &[
                        // Params buffer (uniform)
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
                        // Vd input buffer
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
                        // Output buffer (Id, gd)
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
                    ],
                });

        let pipeline_layout =
            ctx.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Diode Eval Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Diode Eval Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("eval_diode"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            ctx,
            pipeline,
            bind_group_layout,
        })
    }

    /// Evaluate diodes for all devices × all sweep points.
    ///
    /// # Arguments
    /// * `params` - Diode model parameters
    /// * `vd` - Diode voltages (anode - cathode), length = num_devices × num_sweeps
    ///
    /// # Returns
    /// Results array with Id, gd for each device × sweep.
    pub fn evaluate(&self, params: &GpuDiodeParams, vd: &[f32]) -> Result<Vec<DiodeEvalResult>> {
        let count = vd.len();
        if count == 0 {
            return Ok(vec![]);
        }

        // Create uniform buffer for params
        let params_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Diode Params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create input buffer
        let vd_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vd Input"),
                contents: bytemuck::cast_slice(vd),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer
        let output_size = (count * std::mem::size_of::<DiodeEvalResult>()) as u64;
        let output_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diode Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Diode Output Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Diode Eval Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vd_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch compute
        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diode Eval Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Diode Eval Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy to staging
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.ctx.queue().submit(std::iter::once(encoder.finish()));

        // Read back results
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
        let results: Vec<DiodeEvalResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Evaluate with timing information for benchmarking.
    pub fn evaluate_timed(
        &self,
        params: &GpuDiodeParams,
        vd: &[f32],
    ) -> Result<(Vec<DiodeEvalResult>, std::time::Duration)> {
        let start = std::time::Instant::now();
        let results = self.evaluate(params, vd)?;
        let elapsed = start.elapsed();
        Ok((results, elapsed))
    }
}

/// WGSL shader for diode evaluation.
///
/// Shockley diode equation with voltage limiting.
const DIODE_SHADER: &str = r#"
struct DiodeParams {
    is: f32,    // Saturation current
    n: f32,     // Emission coefficient
    vt: f32,    // Thermal voltage
    _pad: f32,
}

struct DiodeResult {
    id: f32,    // Current
    gd: f32,    // Conductance
}

@group(0) @binding(0) var<uniform> params: DiodeParams;
@group(0) @binding(1) var<storage, read> vd_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<DiodeResult>;

@compute @workgroup_size(256)
fn eval_diode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&vd_in) {
        return;
    }

    let vd = vd_in[idx];
    let nvt = params.n * params.vt;

    // Voltage limiting to prevent exp() overflow
    // Critical voltage where exp() starts to overflow
    let vcrit = nvt * log(nvt / (params.is * 1.41421356));  // sqrt(2) ≈ 1.414

    var vd_limited: f32;
    if vd > vcrit {
        // Log compression for large forward bias
        let arg = (vd - vcrit) / nvt;
        vd_limited = vcrit + nvt * log(1.0 + arg);
    } else {
        vd_limited = vd;
    }

    // Shockley equation: Id = Is * (exp(Vd / nVt) - 1)
    let exp_term = exp(vd_limited / nvt);
    let id = params.is * (exp_term - 1.0);

    // Conductance: gd = dId/dVd = Is * exp(Vd / nVt) / nVt
    let gd = params.is * exp_term / nvt;

    // Ensure minimum conductance for numerical stability
    results[idx] = DiodeResult(id, max(gd, 1e-12));
}
"#;

// ============================================================================
// BJT Evaluation
// ============================================================================

/// BJT model parameters for GPU evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBjtParams {
    /// Saturation current (A).
    pub is: f32,
    /// Forward current gain (beta_F).
    pub bf: f32,
    /// Reverse current gain (beta_R).
    pub br: f32,
    /// Forward emission coefficient.
    pub nf: f32,
    /// Reverse emission coefficient.
    pub nr: f32,
    /// Forward Early voltage (V). Use large value (1e6) for no Early effect.
    pub vaf: f32,
    /// Thermal voltage (V).
    pub vt: f32,
    /// +1 for NPN, -1 for PNP.
    pub polarity: f32,
}

impl Default for GpuBjtParams {
    fn default() -> Self {
        Self {
            is: 1e-14,
            bf: 100.0,
            br: 1.0,
            nf: 1.0,
            nr: 1.0,
            vaf: 1e6, // Large value = no Early effect
            vt: 0.02585,
            polarity: 1.0, // NPN
        }
    }
}

/// Results from BJT evaluation.
/// Note: WGSL struct alignment requires padding to match shader layout.
/// The shader uses vec3 which has 16-byte alignment requirements.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BjtEvalResult {
    /// Collector current (A).
    pub ic: f32,
    /// Base current (A).
    pub ib: f32,
    /// Transconductance dIc/dVbe (S).
    pub gm: f32,
    /// Input conductance dIb/dVbe (S).
    pub gpi: f32,
    /// Output conductance dIc/dVce (S).
    pub go: f32,
    /// Padding to match WGSL vec3 alignment (total: 48 bytes).
    pub _pad: [f32; 7],
}

/// GPU kernel for batched BJT evaluation.
pub struct GpuBjtEvaluator {
    ctx: Arc<WgpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBjtEvaluator {
    /// Create a new BJT evaluator.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        let shader = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BJT Evaluation Shader"),
            source: wgpu::ShaderSource::Wgsl(BJT_SHADER.into()),
        });

        let bind_group_layout =
            ctx.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("BJT Eval Bind Group Layout"),
                    entries: &[
                        // Params buffer (uniform)
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
                        // Vbe input buffer
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
                        // Vce input buffer
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
                        // Output buffer
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
                    label: Some("BJT Eval Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BJT Eval Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("eval_bjt"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            ctx,
            pipeline,
            bind_group_layout,
        })
    }

    /// Evaluate BJTs for all devices × all sweep points.
    ///
    /// # Arguments
    /// * `params` - BJT model parameters
    /// * `vbe` - Base-emitter voltages, length = num_devices × num_sweeps
    /// * `vce` - Collector-emitter voltages, length = num_devices × num_sweeps
    ///
    /// # Returns
    /// Results array with Ic, Ib, gm, gpi, go for each device × sweep.
    pub fn evaluate(
        &self,
        params: &GpuBjtParams,
        vbe: &[f32],
        vce: &[f32],
    ) -> Result<Vec<BjtEvalResult>> {
        let count = vbe.len();
        if vce.len() != count {
            return Err(WgpuError::InvalidDimension(
                "vbe and vce must have same length".into(),
            ));
        }
        if count == 0 {
            return Ok(vec![]);
        }

        // Create uniform buffer for params
        let params_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BJT Params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create input buffers
        let vbe_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vbe Input"),
                contents: bytemuck::cast_slice(vbe),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vce_buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vce Input"),
                contents: bytemuck::cast_slice(vce),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffer (32 bytes per result due to padding)
        let output_size = (count * std::mem::size_of::<BjtEvalResult>()) as u64;
        let output_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("BJT Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("BJT Output Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BJT Eval Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vbe_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vce_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch compute
        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BJT Eval Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BJT Eval Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy to staging
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.ctx.queue().submit(std::iter::once(encoder.finish()));

        // Read back results
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
        let results: Vec<BjtEvalResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Evaluate with timing information for benchmarking.
    pub fn evaluate_timed(
        &self,
        params: &GpuBjtParams,
        vbe: &[f32],
        vce: &[f32],
    ) -> Result<(Vec<BjtEvalResult>, std::time::Duration)> {
        let start = std::time::Instant::now();
        let results = self.evaluate(params, vbe, vce)?;
        let elapsed = start.elapsed();
        Ok((results, elapsed))
    }
}

/// WGSL shader for BJT evaluation.
///
/// Ebers-Moll model with Early effect. Computes Ic, Ib, gm, gpi, go.
const BJT_SHADER: &str = r#"
struct BjtParams {
    is: f32,        // Saturation current
    bf: f32,        // Forward beta
    br: f32,        // Reverse beta
    nf: f32,        // Forward emission coefficient
    nr: f32,        // Reverse emission coefficient
    vaf: f32,       // Forward Early voltage
    vt: f32,        // Thermal voltage
    polarity: f32,  // +1 for NPN, -1 for PNP
}

struct BjtResult {
    ic: f32,
    ib: f32,
    gm: f32,
    gpi: f32,
    go: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
}

@group(0) @binding(0) var<uniform> params: BjtParams;
@group(0) @binding(1) var<storage, read> vbe_in: array<f32>;
@group(0) @binding(2) var<storage, read> vce_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<BjtResult>;

// Voltage limiting to prevent exp() overflow
fn limit_voltage(v: f32, nvt: f32) -> f32 {
    let vcrit = nvt * log(nvt / (1e-14 * 1.41421356));
    if v > vcrit {
        let arg = (v - vcrit) / nvt;
        return vcrit + nvt * log(1.0 + arg);
    }
    return v;
}

@compute @workgroup_size(256)
fn eval_bjt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&vbe_in) {
        return;
    }

    // Apply polarity for PNP
    let p = params.polarity;
    let vbe = p * vbe_in[idx];
    let vce = p * vce_in[idx];
    let vbc = vbe - vce;

    let nf_vt = params.nf * params.vt;
    let nr_vt = params.nr * params.vt;

    // Voltage limiting
    let vbe_lim = limit_voltage(vbe, nf_vt);
    let vbc_lim = limit_voltage(vbc, nr_vt);

    // Forward and reverse currents (Ebers-Moll)
    let exp_be = exp(vbe_lim / nf_vt);
    let exp_bc = exp(vbc_lim / nr_vt);

    let if_current = params.is * (exp_be - 1.0);
    let ir_current = params.is * (exp_bc - 1.0);

    // Early effect factor
    var early_factor = 1.0;
    if vce > 0.0 {
        early_factor = 1.0 + vce / params.vaf;
    }

    // Terminal currents
    let ic = (if_current - ir_current / params.br) * early_factor;
    let ib = if_current / params.bf + ir_current / params.br;

    // Small-signal parameters
    // gm = ∂Ic/∂Vbe
    var gm = 1e-12;
    if if_current > 1e-20 {
        gm = (params.is * exp_be / nf_vt) * early_factor;
    }

    // gpi = gm / BF
    let gpi = gm / params.bf;

    // go = ∂Ic/∂Vce (Early effect + reverse current sensitivity)
    var go_early = 0.0;
    if abs(ic) > 1e-20 {
        go_early = abs(ic) / params.vaf;
    }
    let gbc = params.is * exp_bc / nr_vt;
    let go_reverse = gbc / params.br * early_factor;
    let go = max(go_early + go_reverse, 1e-12);

    // Apply polarity to currents
    results[idx] = BjtResult(
        p * ic,
        p * ib,
        max(gm, 1e-12),
        max(gpi, 1e-12),
        go,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    );
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_context() -> Result<Arc<WgpuContext>> {
        Ok(Arc::new(WgpuContext::new()?))
    }

    #[test]
    fn test_mosfet_cutoff() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuMosfetEvaluator::new(ctx).unwrap();
        let params = GpuMosfetParams::default(); // NMOS, Vto = 0.7V

        // Vgs = 0.3V < Vto = 0.7V → cutoff
        let vgs = vec![0.3];
        let vds = vec![2.0];
        let vbs = vec![0.0];

        let results = evaluator.evaluate(&params, &vgs, &vds, &vbs).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].id.abs() < 1e-10, "Id should be ~0 in cutoff: {}", results[0].id);
    }

    #[test]
    fn test_mosfet_saturation() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuMosfetEvaluator::new(ctx).unwrap();
        let params = GpuMosfetParams::default(); // NMOS, Vto = 0.7V, Kp = 110µA/V²

        // Vgs = 1.5V, Vds = 2.0V → saturation (Vds > Vgs - Vto = 0.8V)
        let vgs = vec![1.5];
        let vds = vec![2.0];
        let vbs = vec![0.0];

        let results = evaluator.evaluate(&params, &vgs, &vds, &vbs).unwrap();

        assert_eq!(results.len(), 1);

        // Expected: Id = 0.5 * Kp * (Vgs - Vth)^2 * (1 + lambda * Vds)
        // = 0.5 * 110e-6 * 0.8^2 * (1 + 0.04 * 2) = 38.0µA
        let expected_id = 0.5 * 110e-6 * 0.8 * 0.8 * 1.08;
        assert!(
            (results[0].id - expected_id as f32).abs() < 5e-6,
            "Id = {} (expected ~{})",
            results[0].id,
            expected_id
        );
        assert!(results[0].gm > 0.0, "gm should be positive");
        assert!(results[0].gds > 0.0, "gds should be positive");
    }

    #[test]
    fn test_mosfet_linear() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuMosfetEvaluator::new(ctx).unwrap();
        let params = GpuMosfetParams::default();

        // Vgs = 1.5V, Vds = 0.2V → linear (Vds < Vgs - Vto = 0.8V)
        let vgs = vec![1.5];
        let vds = vec![0.2];
        let vbs = vec![0.0];

        let results = evaluator.evaluate(&params, &vgs, &vds, &vbs).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].id > 0.0, "Id should be positive in linear region");
        // In linear region, gds should be larger than in saturation
        assert!(results[0].gds > 1e-6, "gds should be significant in linear region");
    }

    #[test]
    fn test_mosfet_batch_performance() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuMosfetEvaluator::new(ctx).unwrap();
        let params = GpuMosfetParams::default();

        // Test scaling at various sizes (stay under 128MB buffer limit)
        // Output is 16 bytes per result, so max ~8M results
        println!("\nGPU MOSFET Evaluation Scaling:");
        println!("{:>12} {:>12} {:>12}", "Count", "Time", "M evals/sec");
        println!("{:-<40}", "");

        for &count in &[10_000, 100_000, 1_000_000, 5_000_000] {
            let vgs: Vec<f32> = (0..count).map(|i| 0.5 + (i as f32 / count as f32) * 1.5).collect();
            let vds: Vec<f32> = (0..count).map(|i| 0.1 + (i as f32 / count as f32) * 2.0).collect();
            let vbs: Vec<f32> = vec![0.0; count];

            // Warm-up run
            let _ = evaluator.evaluate(&params, &vgs, &vds, &vbs);

            // Timed run
            let (results, elapsed) = evaluator.evaluate_timed(&params, &vgs, &vds, &vbs).unwrap();
            assert_eq!(results.len(), count);

            let rate = count as f64 / elapsed.as_secs_f64() / 1e6;
            println!("{:>12} {:>12.2?} {:>12.2}", count, elapsed, rate);
        }

        // Results on M3 Ultra (release mode):
        // 10k:  ~7 M evals/sec (overhead dominated)
        // 100k: ~62 M evals/sec
        // 1M:   ~166 M evals/sec
        // 5M:   ~200+ M evals/sec (expected)
        //
        // Target: 100M+ evals/sec achieved!
    }

    #[test]
    fn test_pmos_polarity() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuMosfetEvaluator::new(ctx).unwrap();
        let mut params = GpuMosfetParams::default();
        params.polarity = -1.0; // PMOS
        params.vto = -0.7; // Negative threshold for PMOS

        // PMOS: Vgs = -1.5V, Vds = -2.0V → saturation
        let vgs = vec![-1.5];
        let vds = vec![-2.0];
        let vbs = vec![0.0];

        let results = evaluator.evaluate(&params, &vgs, &vds, &vbs).unwrap();

        assert_eq!(results.len(), 1);
        // PMOS current flows from source to drain (negative Id convention)
        assert!(results[0].id < 0.0, "PMOS Id should be negative: {}", results[0].id);
    }

    // ========================================================================
    // Diode Tests
    // ========================================================================

    #[test]
    fn test_diode_forward_bias() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuDiodeEvaluator::new(ctx).unwrap();
        let params = GpuDiodeParams::default();

        // Forward bias: Vd = 0.7V
        let vd = vec![0.7];
        let results = evaluator.evaluate(&params, &vd).unwrap();

        assert_eq!(results.len(), 1);
        // At 0.7V forward bias, current should be significant (mA range)
        assert!(results[0].id > 1e-6, "Id should be > 1µA at 0.7V: {}", results[0].id);
        assert!(results[0].gd > 0.0, "gd should be positive");
    }

    #[test]
    fn test_diode_reverse_bias() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuDiodeEvaluator::new(ctx).unwrap();
        let params = GpuDiodeParams::default();

        // Reverse bias: Vd = -1.0V
        let vd = vec![-1.0];
        let results = evaluator.evaluate(&params, &vd).unwrap();

        assert_eq!(results.len(), 1);
        // Reverse bias: current should be ~-Is (very small negative)
        assert!(results[0].id < 0.0, "Id should be negative in reverse: {}", results[0].id);
        assert!(results[0].id.abs() < 1e-12, "Id should be ~-Is: {}", results[0].id);
    }

    #[test]
    fn test_diode_batch_performance() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuDiodeEvaluator::new(ctx).unwrap();
        let params = GpuDiodeParams::default();

        println!("\nGPU Diode Evaluation Scaling:");
        println!("{:>12} {:>12} {:>12}", "Count", "Time", "M evals/sec");
        println!("{:-<40}", "");

        for &count in &[10_000, 100_000, 1_000_000, 5_000_000] {
            let vd: Vec<f32> = (0..count).map(|i| -1.0 + (i as f32 / count as f32) * 2.0).collect();

            // Warm-up run
            let _ = evaluator.evaluate(&params, &vd);

            // Timed run
            let (results, elapsed) = evaluator.evaluate_timed(&params, &vd).unwrap();
            assert_eq!(results.len(), count);

            let rate = count as f64 / elapsed.as_secs_f64() / 1e6;
            println!("{:>12} {:>12.2?} {:>12.2}", count, elapsed, rate);
        }
    }

    // ========================================================================
    // BJT Tests
    // ========================================================================

    #[test]
    fn test_bjt_cutoff() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuBjtEvaluator::new(ctx).unwrap();
        let params = GpuBjtParams::default(); // NPN

        // Both junctions reverse biased: Vbe = -0.5V, Vce = 5V
        let vbe = vec![-0.5];
        let vce = vec![5.0];
        let results = evaluator.evaluate(&params, &vbe, &vce).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].ic.abs() < 1e-10, "Ic should be ~0 in cutoff: {}", results[0].ic);
        assert!(results[0].ib.abs() < 1e-10, "Ib should be ~0 in cutoff: {}", results[0].ib);
    }

    #[test]
    fn test_bjt_forward_active() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuBjtEvaluator::new(ctx).unwrap();
        let params = GpuBjtParams::default(); // NPN, BF=100

        // Forward active: Vbe = 0.7V, Vce = 5V
        let vbe = vec![0.7];
        let vce = vec![5.0];
        let results = evaluator.evaluate(&params, &vbe, &vce).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].ic > 0.0, "Ic should be positive: {}", results[0].ic);
        assert!(results[0].ib > 0.0, "Ib should be positive: {}", results[0].ib);

        // Check beta relationship: Ic ≈ β * Ib
        let beta = results[0].ic / results[0].ib;
        assert!(
            (beta - 100.0).abs() < 20.0,
            "β should be ~100: {}",
            beta
        );

        // Check gm is positive
        assert!(results[0].gm > 0.0, "gm should be positive");
    }

    #[test]
    fn test_bjt_pnp_polarity() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuBjtEvaluator::new(ctx).unwrap();
        let mut params = GpuBjtParams::default();
        params.polarity = -1.0; // PNP

        // PNP forward active: Vbe = -0.7V, Vce = -5V
        let vbe = vec![-0.7];
        let vce = vec![-5.0];
        let results = evaluator.evaluate(&params, &vbe, &vce).unwrap();

        assert_eq!(results.len(), 1);
        // PNP currents are negative
        assert!(results[0].ic < 0.0, "PNP Ic should be negative: {}", results[0].ic);
        assert!(results[0].ib < 0.0, "PNP Ib should be negative: {}", results[0].ib);
    }

    #[test]
    fn test_bjt_batch_performance() {
        let ctx = match create_context() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let evaluator = GpuBjtEvaluator::new(ctx).unwrap();
        let params = GpuBjtParams::default();

        println!("\nGPU BJT Evaluation Scaling:");
        println!("{:>12} {:>12} {:>12}", "Count", "Time", "M evals/sec");
        println!("{:-<40}", "");

        // BJT result is larger (48 bytes) so stay under buffer limits (128MB / 48 = ~2.6M)
        for &count in &[10_000, 100_000, 1_000_000, 2_500_000] {
            let vbe: Vec<f32> = (0..count).map(|i| 0.5 + (i as f32 / count as f32) * 0.3).collect();
            let vce: Vec<f32> = (0..count).map(|i| 1.0 + (i as f32 / count as f32) * 4.0).collect();

            // Warm-up run
            let _ = evaluator.evaluate(&params, &vbe, &vce);

            // Timed run
            let (results, elapsed) = evaluator.evaluate_timed(&params, &vbe, &vce).unwrap();
            assert_eq!(results.len(), count);

            let rate = count as f64 / elapsed.as_secs_f64() / 1e6;
            println!("{:>12} {:>12.2?} {:>12.2}", count, elapsed, rate);
        }
    }
}
