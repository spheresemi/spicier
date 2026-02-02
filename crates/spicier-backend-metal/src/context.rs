//! WebGPU device context management.

use crate::error::{Result, WgpuError};
use std::sync::Arc;

/// WebGPU context holding device and queue.
///
/// This is the main entry point for GPU operations. Create one context
/// and reuse it for multiple operators to avoid reinitialization overhead.
pub struct WgpuContext {
    /// wgpu device for GPU operations.
    pub(crate) device: Arc<wgpu::Device>,
    /// Command queue for submitting GPU work.
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Adapter info for debugging/logging.
    pub(crate) adapter_info: wgpu::AdapterInfo,
    /// Whether the device supports f64 shader operations.
    pub(crate) supports_f64: bool,
}

impl WgpuContext {
    /// Create a new WebGPU context using the best available adapter.
    ///
    /// Prefers discrete GPUs, falls back to integrated GPUs.
    /// On macOS, this will use Metal as the backend.
    pub fn new() -> Result<Self> {
        Self::with_power_preference(wgpu::PowerPreference::HighPerformance)
    }

    /// Create a new WebGPU context with a specific power preference.
    pub fn with_power_preference(power_preference: wgpu::PowerPreference) -> Result<Self> {
        pollster::block_on(Self::new_async(power_preference))
    }

    async fn new_async(power_preference: wgpu::PowerPreference) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(WgpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Selected GPU adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let supports_f64 = adapter.features().contains(wgpu::Features::SHADER_F64);

        let required_features = if supports_f64 {
            log::info!("GPU supports f64 shader operations");
            wgpu::Features::SHADER_F64
        } else {
            log::warn!(
                "GPU does not support f64 shader operations, using f32 with potential precision loss"
            );
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Spicier Solver Device"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| WgpuError::DeviceInit(e.to_string()))?;

        log::info!("WebGPU device initialized successfully");

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            supports_f64,
        })
    }

    /// Check if a WebGPU-capable GPU is available on this system.
    pub fn is_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
                ..Default::default()
            });
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .is_some()
        })
    }

    /// Get the adapter name.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get the backend being used (Metal, Vulkan, etc.).
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    /// Get the underlying device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get the underlying queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Check if the device supports f64 shader operations.
    pub fn supports_f64(&self) -> bool {
        self.supports_f64
    }

    /// Get the maximum buffer size supported by the device.
    ///
    /// Returns the device's max_buffer_size limit, capped at 256MB as a
    /// practical limit for most operations. This is useful for chunking
    /// large sweep operations.
    pub fn max_buffer_size(&self) -> u64 {
        // Query device limits and cap at practical 256MB limit
        let device_max = self.device.limits().max_buffer_size;
        device_max.min(256 * 1024 * 1024)
    }

    /// Get the maximum storage buffer binding size.
    ///
    /// This is the maximum size for a single storage buffer binding
    /// in a shader.
    pub fn max_storage_buffer_binding_size(&self) -> u32 {
        self.device.limits().max_storage_buffer_binding_size
    }
}
