//! SIMD-accelerated numerical kernels for Spicier.
//!
//! Provides runtime-detected SIMD implementations for:
//! - f64 dot products and matrix-vector multiplication (DC/transient)
//! - Complex64 dot products and matrix-vector multiplication (AC)
//! - Conjugate dot products for iterative solvers (GMRES)
//!
//! On x86/x86_64 systems, automatically uses AVX-512 or AVX2+FMA when available.
//! Falls back to scalar on all other architectures (including Apple Silicon via
//! native execution).

pub mod capability;
pub mod complex_dot;
pub mod conjugate_dot;
pub mod real_dot;

pub use capability::SimdCapability;
pub use complex_dot::{complex_dot_product, complex_dot_scalar, complex_matvec, complex_matvec_scalar};
pub use conjugate_dot::{complex_conjugate_dot_product, conjugate_dot_scalar};
pub use real_dot::{real_dot_product, real_dot_scalar, real_matvec, real_matvec_scalar};
