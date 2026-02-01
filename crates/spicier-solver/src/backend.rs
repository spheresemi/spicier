//! Compute backend detection and selection.
//!
//! Provides [`ComputeBackend`] for choosing between CPU, CUDA, and Metal/WebGPU
//! acceleration at runtime. The enum lives in `spicier-solver` so all crates
//! can reference it without circular dependencies. Actual GPU availability
//! probing is done by the CLI or application layer.

use std::fmt;

/// The compute backend to use for matrix operations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    /// CPU-only backend (always available).
    #[default]
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda {
        /// CUDA device ordinal.
        device_id: usize,
    },
    /// Metal/WebGPU backend.
    Metal {
        /// Adapter name (for informational display).
        adapter_name: String,
    },
}

impl ComputeBackend {
    /// Parse a backend name from a CLI argument string.
    ///
    /// Accepts `"auto"`, `"cpu"`, `"cuda"`, or `"metal"`.
    /// For `"auto"`, returns `Cpu` — the caller should probe GPU availability
    /// separately and upgrade the result.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "auto" | "cpu" => Some(ComputeBackend::Cpu),
            "cuda" => Some(ComputeBackend::Cuda { device_id: 0 }),
            "metal" => Some(ComputeBackend::Metal {
                adapter_name: String::new(),
            }),
            _ => None,
        }
    }

    /// Short name for display.
    pub fn name(&self) -> &str {
        match self {
            ComputeBackend::Cpu => "CPU",
            ComputeBackend::Cuda { .. } => "CUDA",
            ComputeBackend::Metal { .. } => "Metal",
        }
    }
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeBackend::Cpu => write!(f, "CPU"),
            ComputeBackend::Cuda { device_id } => write!(f, "CUDA (device {})", device_id),
            ComputeBackend::Metal { adapter_name } if adapter_name.is_empty() => {
                write!(f, "Metal/WebGPU")
            }
            ComputeBackend::Metal { adapter_name } => {
                write!(f, "Metal/WebGPU ({})", adapter_name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cpu() {
        assert_eq!(ComputeBackend::default(), ComputeBackend::Cpu);
    }

    #[test]
    fn from_name_cpu() {
        let b = ComputeBackend::from_name("cpu").unwrap();
        assert_eq!(b, ComputeBackend::Cpu);
    }

    #[test]
    fn from_name_auto() {
        let b = ComputeBackend::from_name("auto").unwrap();
        assert_eq!(b, ComputeBackend::Cpu);
    }

    #[test]
    fn from_name_case_insensitive() {
        let b = ComputeBackend::from_name("CPU").unwrap();
        assert_eq!(b, ComputeBackend::Cpu);
    }

    #[test]
    fn from_name_invalid() {
        assert!(ComputeBackend::from_name("opencl").is_none());
    }

    #[test]
    fn display_cpu() {
        assert_eq!(format!("{}", ComputeBackend::Cpu), "CPU");
    }

    #[test]
    fn display_cuda() {
        let b = ComputeBackend::Cuda { device_id: 0 };
        assert_eq!(format!("{}", b), "CUDA (device 0)");
    }

    #[test]
    fn display_metal() {
        let b = ComputeBackend::Metal {
            adapter_name: "Apple M1".to_string(),
        };
        assert_eq!(format!("{}", b), "Metal/WebGPU (Apple M1)");
    }

    #[test]
    fn display_metal_empty_name() {
        let b = ComputeBackend::Metal {
            adapter_name: String::new(),
        };
        assert_eq!(format!("{}", b), "Metal/WebGPU");
    }

    #[test]
    fn name_method() {
        assert_eq!(ComputeBackend::Cpu.name(), "CPU");
        assert_eq!(ComputeBackend::Cuda { device_id: 0 }.name(), "CUDA");
        assert_eq!(
            ComputeBackend::Metal {
                adapter_name: String::new()
            }
            .name(),
            "Metal"
        );
    }
}
