//! MPS device context management.

#[cfg(target_os = "macos")]
mod macos {
    use crate::error::{MpsError, Result};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLCommandQueue, MTLCopyAllDevices, MTLDevice};

    /// MPS context holding Metal device and command queue.
    ///
    /// This is the main entry point for MPS operations. Create one context
    /// and reuse it for multiple operations to avoid reinitialization overhead.
    pub struct MpsContext {
        /// Metal device.
        pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
        /// Command queue for submitting GPU work.
        pub(crate) command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        /// Device name for debugging/logging.
        pub(crate) device_name: String,
    }

    impl MpsContext {
        /// Create a new MPS context using the default (best) Metal device.
        ///
        /// On systems with multiple GPUs, this typically selects the discrete GPU.
        pub fn new() -> Result<Self> {
            let devices = MTLCopyAllDevices();

            // Try to find a device, preferring discrete GPUs
            let device = devices
                .into_iter()
                .find(|d| !d.isLowPower())
                .or_else(|| {
                    // Fall back to any available device
                    let devices = MTLCopyAllDevices();
                    devices.into_iter().next()
                })
                .ok_or(MpsError::NoDevice)?;

            let device_name = device.name().to_string();
            log::info!("Selected Metal device: {}", device_name);

            let command_queue = device
                .newCommandQueue()
                .ok_or_else(|| MpsError::CommandQueue("Failed to create command queue".into()))?;

            log::info!("MPS context initialized successfully");

            Ok(Self {
                device,
                command_queue,
                device_name,
            })
        }

        /// Check if an MPS-capable GPU is available on this system.
        pub fn is_available() -> bool {
            let devices = MTLCopyAllDevices();
            !devices.is_empty()
        }

        /// Get the device name.
        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        /// Get the underlying Metal device.
        pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
            &self.device
        }

        /// Get the command queue.
        pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
            &self.command_queue
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos::MpsContext;

#[cfg(not(target_os = "macos"))]
mod stub {
    use crate::error::{MpsError, Result};

    /// Stub MPS context for non-macOS platforms.
    pub struct MpsContext;

    impl MpsContext {
        /// Always fails on non-macOS platforms.
        pub fn new() -> Result<Self> {
            Err(MpsError::UnsupportedPlatform)
        }

        /// Always returns false on non-macOS platforms.
        pub fn is_available() -> bool {
            false
        }

        /// Stub method.
        pub fn device_name(&self) -> &str {
            ""
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stub::MpsContext;
