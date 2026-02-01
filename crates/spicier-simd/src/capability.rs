//! SIMD capability detection.
//!
//! Runtime detection of the best available SIMD instruction set.

/// Detected SIMD capability level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCapability {
    /// AVX-512 with 512-bit vectors (8 f64 per vector, 4 complex per iteration)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
    /// AVX2 with 256-bit vectors (4 f64 per vector, 2 complex per iteration)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    /// Scalar fallback (no SIMD)
    Scalar,
}

impl SimdCapability {
    /// Detect the best available SIMD capability at runtime.
    #[inline]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdCapability::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return SimdCapability::Avx2;
            }
        }
        SimdCapability::Scalar
    }

    /// Check if this capability uses SIMD acceleration.
    #[inline]
    pub fn is_simd(&self) -> bool {
        !matches!(self, SimdCapability::Scalar)
    }

    /// Get a human-readable description of the capability.
    pub fn description(&self) -> &'static str {
        match self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdCapability::Avx512 => "AVX-512 (4 complex/iteration)",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdCapability::Avx2 => "AVX2+FMA (2 complex/iteration)",
            SimdCapability::Scalar => "Scalar (no SIMD)",
        }
    }
}

impl Default for SimdCapability {
    fn default() -> Self {
        Self::detect()
    }
}

impl std::fmt::Display for SimdCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_detection() {
        let cap = SimdCapability::detect();
        let desc = cap.description();
        assert!(!desc.is_empty());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(matches!(
            cap,
            SimdCapability::Scalar | SimdCapability::Avx2 | SimdCapability::Avx512
        ));
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        assert!(matches!(cap, SimdCapability::Scalar));
    }

    #[test]
    fn capability_default() {
        let cap = SimdCapability::default();
        assert_eq!(cap, SimdCapability::detect());
    }

    #[test]
    fn display_impl() {
        let cap = SimdCapability::detect();
        let s = format!("{}", cap);
        assert!(!s.is_empty());
    }
}
