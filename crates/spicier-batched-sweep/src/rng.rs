//! GPU-friendly random number generation for Monte Carlo sweeps.
//!
//! This module provides a hash-based RNG that generates reproducible random
//! numbers based on (seed, sweep_index, parameter_index). The algorithm is
//! designed to be easily portable to GPU compute shaders.
//!
//! # Design
//!
//! Unlike traditional PRNGs that maintain state between calls, this RNG is
//! "stateless" - each random value is computed independently based on its
//! coordinates. This makes it ideal for parallel GPU execution where each
//! thread can compute its own random values without synchronization.
//!
//! # Algorithm
//!
//! Uses SplitMix64 as the hash function, which provides excellent statistical
//! properties and is simple enough for GPU implementation.
//!
//! For Gaussian distribution, uses the Box-Muller transform which requires
//! two uniform values to produce two Gaussian values.

use std::f64::consts::PI;

/// SplitMix64 hash function.
///
/// This is the mixing function from SplitMix64, a fast, high-quality PRNG.
/// It takes a 64-bit input and produces a well-distributed 64-bit output.
///
/// The same algorithm can be implemented in GPU shaders:
/// ```wgsl
/// fn splitmix64(x: u64) -> u64 {
///     var z = x + 0x9e3779b97f4a7c15u;
///     z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9u;
///     z = (z ^ (z >> 27u)) * 0x94d049bb133111ebu;
///     return z ^ (z >> 31u);
/// }
/// ```
#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Combine seed with indices to create a unique hash input.
///
/// Uses a simple but effective mixing strategy that avoids collisions
/// between different (sweep_idx, param_idx) combinations.
#[inline]
pub fn combine_indices(seed: u64, sweep_idx: u32, param_idx: u32) -> u64 {
    // Mix the indices into different bit ranges to avoid collisions
    // Use wrapping_mul to avoid overflow panics in debug mode
    seed ^ (sweep_idx as u64).wrapping_mul(0x517cc1b727220a95)
        ^ (param_idx as u64).wrapping_mul(0x5851f42d4c957f2d)
}

/// Generate a uniform random f64 in [0, 1) from hash coordinates.
///
/// # Arguments
/// * `seed` - Random seed for reproducibility
/// * `sweep_idx` - Index of the sweep point (0..num_sweeps)
/// * `param_idx` - Index of the parameter (0..num_params)
///
/// # Example
/// ```
/// use spicier_batched_sweep::rng::uniform;
///
/// let seed = 42;
/// let value = uniform(seed, 0, 0);
/// assert!(value >= 0.0 && value < 1.0);
///
/// // Same inputs always produce same output
/// assert_eq!(uniform(seed, 0, 0), uniform(seed, 0, 0));
///
/// // Different inputs produce different outputs
/// assert_ne!(uniform(seed, 0, 0), uniform(seed, 0, 1));
/// ```
#[inline]
pub fn uniform(seed: u64, sweep_idx: u32, param_idx: u32) -> f64 {
    let hash = splitmix64(combine_indices(seed, sweep_idx, param_idx));
    // Convert to [0, 1) by dividing by 2^64
    // Use the upper 53 bits for full f64 mantissa precision
    (hash >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Generate a uniform random f32 in [0, 1) from hash coordinates.
///
/// This is the preferred version for GPU computation where f32 is faster.
#[inline]
pub fn uniform_f32(seed: u64, sweep_idx: u32, param_idx: u32) -> f32 {
    let hash = splitmix64(combine_indices(seed, sweep_idx, param_idx));
    // Use upper 24 bits for f32 mantissa precision
    (hash >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

/// Generate a Gaussian random f64 with mean=0, sigma=1 from hash coordinates.
///
/// Uses the Box-Muller transform, which requires two uniform values to
/// produce one Gaussian value. We use `param_idx * 2` and `param_idx * 2 + 1`
/// as the two uniform sources.
///
/// # Arguments
/// * `seed` - Random seed for reproducibility
/// * `sweep_idx` - Index of the sweep point
/// * `param_idx` - Index of the parameter
///
/// # Example
/// ```
/// use spicier_batched_sweep::rng::gaussian;
///
/// let seed = 42;
/// let values: Vec<f64> = (0..1000).map(|i| gaussian(seed, i, 0)).collect();
///
/// // Mean should be close to 0
/// let mean = values.iter().sum::<f64>() / values.len() as f64;
/// assert!(mean.abs() < 0.1);
/// ```
#[inline]
pub fn gaussian(seed: u64, sweep_idx: u32, param_idx: u32) -> f64 {
    // Use two different hash values for Box-Muller
    // Multiply param_idx by 2 to ensure we get independent uniform values
    let u1 = uniform(seed, sweep_idx, param_idx.wrapping_mul(2));
    let u2 = uniform(seed, sweep_idx, param_idx.wrapping_mul(2).wrapping_add(1));

    // Avoid log(0) by clamping u1 away from 0
    let u1 = u1.max(1e-10);

    // Box-Muller transform
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Generate a Gaussian random f32 with mean=0, sigma=1 from hash coordinates.
///
/// This is the preferred version for GPU computation where f32 is faster.
#[inline]
pub fn gaussian_f32(seed: u64, sweep_idx: u32, param_idx: u32) -> f32 {
    let u1 = uniform_f32(seed, sweep_idx, param_idx.wrapping_mul(2));
    let u2 = uniform_f32(seed, sweep_idx, param_idx.wrapping_mul(2).wrapping_add(1));

    let u1 = u1.max(1e-7);

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Generate a Gaussian random value with specified mean and sigma.
#[inline]
pub fn gaussian_scaled(seed: u64, sweep_idx: u32, param_idx: u32, mean: f64, sigma: f64) -> f64 {
    mean + gaussian(seed, sweep_idx, param_idx) * sigma
}

/// Generate a Gaussian random f32 with specified mean and sigma.
#[inline]
pub fn gaussian_scaled_f32(
    seed: u64,
    sweep_idx: u32,
    param_idx: u32,
    mean: f32,
    sigma: f32,
) -> f32 {
    mean + gaussian_f32(seed, sweep_idx, param_idx) * sigma
}

/// GPU-compatible RNG configuration.
///
/// This struct can be passed to GPU shaders as a uniform buffer.
/// The shader then calls the hash functions with this seed.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuRngConfig {
    /// Random seed.
    pub seed: u64,
}

impl GpuRngConfig {
    /// Create a new GPU RNG configuration.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate uniform value (CPU reference implementation).
    #[inline]
    pub fn uniform(&self, sweep_idx: u32, param_idx: u32) -> f64 {
        uniform(self.seed, sweep_idx, param_idx)
    }

    /// Generate Gaussian value (CPU reference implementation).
    #[inline]
    pub fn gaussian(&self, sweep_idx: u32, param_idx: u32) -> f64 {
        gaussian(self.seed, sweep_idx, param_idx)
    }
}

/// Generate all random parameters for a batch of sweep points.
///
/// This is useful for CPU-side generation when GPU RNG is not available.
///
/// # Arguments
/// * `seed` - Random seed
/// * `num_sweeps` - Number of sweep points
/// * `num_params` - Number of parameters per sweep point
/// * `means` - Mean value for each parameter
/// * `sigmas` - Standard deviation for each parameter
///
/// # Returns
/// A flattened vector of size `num_sweeps * num_params` containing
/// the random parameter values in row-major order (sweep-major).
pub fn generate_gaussian_parameters(
    seed: u64,
    num_sweeps: usize,
    means: &[f64],
    sigmas: &[f64],
) -> Vec<f64> {
    let num_params = means.len();
    assert_eq!(means.len(), sigmas.len());

    let mut result = Vec::with_capacity(num_sweeps * num_params);

    for sweep_idx in 0..num_sweeps {
        for (param_idx, (mean, sigma)) in means.iter().zip(sigmas.iter()).enumerate() {
            let value = gaussian_scaled(seed, sweep_idx as u32, param_idx as u32, *mean, *sigma);
            result.push(value);
        }
    }

    result
}

/// Generate all random parameters for a batch of sweep points (f32 version).
pub fn generate_gaussian_parameters_f32(
    seed: u64,
    num_sweeps: usize,
    means: &[f32],
    sigmas: &[f32],
) -> Vec<f32> {
    let num_params = means.len();
    assert_eq!(means.len(), sigmas.len());

    let mut result = Vec::with_capacity(num_sweeps * num_params);

    for sweep_idx in 0..num_sweeps {
        for (param_idx, (mean, sigma)) in means.iter().zip(sigmas.iter()).enumerate() {
            let value =
                gaussian_scaled_f32(seed, sweep_idx as u32, param_idx as u32, *mean, *sigma);
            result.push(value);
        }
    }

    result
}

// ============================================================================
// GPU Shader Code Generation
// ============================================================================

/// WGSL shader code for the RNG functions.
///
/// This can be included in Metal/WebGPU compute shaders.
pub const WGSL_RNG_CODE: &str = r#"
// SplitMix64 hash function
fn splitmix64(x: u64) -> u64 {
    var z = x + 0x9e3779b97f4a7c15lu;
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9lu;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111eblu;
    return z ^ (z >> 31u);
}

// Combine seed with indices
fn combine_indices(seed: u64, sweep_idx: u32, param_idx: u32) -> u64 {
    return seed
        ^ (u64(sweep_idx) * 0x517cc1b727220a95lu)
        ^ (u64(param_idx) * 0x5851f42d4c957f2dlu);
}

// Uniform [0, 1) random number
fn rng_uniform(seed: u64, sweep_idx: u32, param_idx: u32) -> f32 {
    let hash = splitmix64(combine_indices(seed, sweep_idx, param_idx));
    return f32(hash >> 40u) * (1.0 / f32(1u << 24u));
}

// Gaussian N(0,1) random number using Box-Muller
fn rng_gaussian(seed: u64, sweep_idx: u32, param_idx: u32) -> f32 {
    let u1 = max(rng_uniform(seed, sweep_idx, param_idx * 2u), 1e-7);
    let u2 = rng_uniform(seed, sweep_idx, param_idx * 2u + 1u);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
}

// Gaussian N(mean, sigma) random number
fn rng_gaussian_scaled(seed: u64, sweep_idx: u32, param_idx: u32, mean: f32, sigma: f32) -> f32 {
    return mean + rng_gaussian(seed, sweep_idx, param_idx) * sigma;
}
"#;

/// CUDA device code for the RNG functions.
///
/// This can be included in CUDA kernels.
pub const CUDA_RNG_CODE: &str = r#"
// SplitMix64 hash function
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Combine seed with indices
__device__ __forceinline__ uint64_t combine_indices(uint64_t seed, uint32_t sweep_idx, uint32_t param_idx) {
    return seed
        ^ ((uint64_t)sweep_idx * 0x517cc1b727220a95ULL)
        ^ ((uint64_t)param_idx * 0x5851f42d4c957f2dULL);
}

// Uniform [0, 1) random number
__device__ __forceinline__ float rng_uniform(uint64_t seed, uint32_t sweep_idx, uint32_t param_idx) {
    uint64_t hash = splitmix64(combine_indices(seed, sweep_idx, param_idx));
    return (float)(hash >> 40) * (1.0f / (float)(1ULL << 24));
}

// Gaussian N(0,1) random number using Box-Muller
__device__ __forceinline__ float rng_gaussian(uint64_t seed, uint32_t sweep_idx, uint32_t param_idx) {
    float u1 = fmaxf(rng_uniform(seed, sweep_idx, param_idx * 2), 1e-7f);
    float u2 = rng_uniform(seed, sweep_idx, param_idx * 2 + 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265359f * u2);
}

// Gaussian N(mean, sigma) random number
__device__ __forceinline__ float rng_gaussian_scaled(uint64_t seed, uint32_t sweep_idx, uint32_t param_idx, float mean, float sigma) {
    return mean + rng_gaussian(seed, sweep_idx, param_idx) * sigma;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitmix64_deterministic() {
        // Same input should produce same output
        assert_eq!(splitmix64(0), splitmix64(0));
        assert_eq!(splitmix64(12345), splitmix64(12345));

        // Different inputs should produce different outputs
        assert_ne!(splitmix64(0), splitmix64(1));
    }

    #[test]
    fn test_uniform_range() {
        let seed = 42;
        for sweep_idx in 0..1000 {
            for param_idx in 0..10 {
                let value = uniform(seed, sweep_idx, param_idx);
                assert!(value >= 0.0, "uniform should be >= 0");
                assert!(value < 1.0, "uniform should be < 1");
            }
        }
    }

    #[test]
    fn test_uniform_deterministic() {
        let seed = 12345;
        assert_eq!(uniform(seed, 0, 0), uniform(seed, 0, 0));
        assert_eq!(uniform(seed, 100, 5), uniform(seed, 100, 5));
    }

    #[test]
    fn test_uniform_different_indices() {
        let seed = 42;
        // Different sweep indices should give different results
        assert_ne!(uniform(seed, 0, 0), uniform(seed, 1, 0));
        // Different param indices should give different results
        assert_ne!(uniform(seed, 0, 0), uniform(seed, 0, 1));
        // Different seeds should give different results
        assert_ne!(uniform(42, 0, 0), uniform(43, 0, 0));
    }

    #[test]
    fn test_uniform_distribution() {
        // Check that uniform values are roughly uniformly distributed
        let seed = 42;
        let n = 10000;
        let mut buckets = [0u32; 10];

        for i in 0..n {
            let value = uniform(seed, i, 0);
            let bucket = (value * 10.0) as usize;
            buckets[bucket.min(9)] += 1;
        }

        // Each bucket should have roughly n/10 = 1000 values
        // Allow 20% deviation
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count > 800 && count < 1200,
                "bucket {} has {} values (expected ~1000)",
                i,
                count
            );
        }
    }

    #[test]
    fn test_gaussian_mean_and_variance() {
        let seed = 42;
        let n = 10000;
        let values: Vec<f64> = (0..n).map(|i| gaussian(seed, i, 0)).collect();

        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Mean should be close to 0 (within 0.05)
        assert!(mean.abs() < 0.05, "Gaussian mean = {} (expected ~0)", mean);

        // Std dev should be close to 1 (within 0.05)
        assert!(
            (std_dev - 1.0).abs() < 0.05,
            "Gaussian std_dev = {} (expected ~1)",
            std_dev
        );
    }

    #[test]
    fn test_gaussian_scaled() {
        let seed = 42;
        let mean = 1000.0;
        let sigma = 50.0;
        let n = 10000;

        let values: Vec<f64> = (0..n)
            .map(|i| gaussian_scaled(seed, i, 0, mean, sigma))
            .collect();

        let actual_mean = values.iter().sum::<f64>() / n as f64;
        let actual_variance = values
            .iter()
            .map(|x| (x - actual_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let actual_std_dev = actual_variance.sqrt();

        // Mean should be close to specified mean
        assert!(
            (actual_mean - mean).abs() < sigma * 0.1,
            "Scaled Gaussian mean = {} (expected ~{})",
            actual_mean,
            mean
        );

        // Std dev should be close to specified sigma
        assert!(
            (actual_std_dev - sigma).abs() < sigma * 0.1,
            "Scaled Gaussian std_dev = {} (expected ~{})",
            actual_std_dev,
            sigma
        );
    }

    #[test]
    fn test_generate_gaussian_parameters() {
        let seed = 42;
        let num_sweeps = 100;
        let means = vec![1000.0, 2000.0, 500.0];
        let sigmas = vec![50.0, 100.0, 25.0];

        let params = generate_gaussian_parameters(seed, num_sweeps, &means, &sigmas);

        assert_eq!(params.len(), num_sweeps * means.len());

        // Check each parameter's statistics
        for (param_idx, (mean, sigma)) in means.iter().zip(sigmas.iter()).enumerate() {
            let param_values: Vec<f64> = (0..num_sweeps)
                .map(|i| params[i * means.len() + param_idx])
                .collect();

            let actual_mean = param_values.iter().sum::<f64>() / num_sweeps as f64;

            // Mean should be within 2 sigma of expected (with high probability)
            assert!(
                (actual_mean - mean).abs() < 2.0 * sigma,
                "Parameter {} mean = {} (expected ~{})",
                param_idx,
                actual_mean,
                mean
            );
        }
    }

    #[test]
    fn test_f32_versions() {
        let seed = 42;

        // f32 uniform should be in [0, 1)
        let u = uniform_f32(seed, 0, 0);
        assert!((0.0..1.0).contains(&u));

        // f32 gaussian should produce reasonable values
        let g = gaussian_f32(seed, 0, 0);
        assert!(g.is_finite());
        assert!(g.abs() < 10.0); // Very unlikely to be outside [-10, 10]
    }

    #[test]
    fn test_gpu_rng_config() {
        let config = GpuRngConfig::new(42);

        let u = config.uniform(0, 0);
        assert!((0.0..1.0).contains(&u));

        let g = config.gaussian(0, 0);
        assert!(g.is_finite());
    }
}
