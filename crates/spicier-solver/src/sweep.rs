//! Batched parameter sweep infrastructure.
//!
//! Provides efficient multi-point analysis for:
//! - Monte Carlo simulation with random parameter variations
//! - Corner analysis with systematic worst-case combinations
//! - Parameter sweeps with parallel execution

use nalgebra::DVector;
use std::sync::Arc;

use crate::error::Result;
use crate::linear::solve_dense;
use crate::newton::ConvergenceCriteria;

/// A parameter variation for sweep analysis.
#[derive(Debug, Clone)]
pub struct ParameterVariation {
    /// Parameter name (for identification).
    pub name: String,
    /// Nominal value.
    pub nominal: f64,
    /// Minimum value (for corner analysis).
    pub min: f64,
    /// Maximum value (for corner analysis).
    pub max: f64,
    /// Standard deviation (for Monte Carlo, as fraction of nominal).
    pub sigma: f64,
}

impl ParameterVariation {
    /// Create a new parameter variation.
    pub fn new(name: impl Into<String>, nominal: f64) -> Self {
        Self {
            name: name.into(),
            nominal,
            min: nominal * 0.9, // Default ±10%
            max: nominal * 1.1,
            sigma: 0.05, // Default 5% sigma
        }
    }

    /// Set min/max bounds.
    pub fn with_bounds(mut self, min: f64, max: f64) -> Self {
        self.min = min;
        self.max = max;
        self
    }

    /// Set sigma (as fraction of nominal).
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }
}

/// A single sweep point with parameter values.
#[derive(Debug, Clone)]
pub struct SweepPoint {
    /// Parameter values at this point.
    pub parameters: Vec<f64>,
}

/// Result of a batched sweep analysis.
#[derive(Debug, Clone)]
pub struct BatchedSweepResult {
    /// Solutions for each sweep point.
    pub solutions: Vec<DVector<f64>>,
    /// Parameter values for each point.
    pub points: Vec<SweepPoint>,
    /// Number of converged points.
    pub converged_count: usize,
    /// Total number of points.
    pub total_count: usize,
}

impl BatchedSweepResult {
    /// Get the solution at a specific point.
    pub fn solution(&self, index: usize) -> Option<&DVector<f64>> {
        self.solutions.get(index)
    }

    /// Get all node voltages at a specific node across all points.
    pub fn node_voltages(&self, node_index: usize) -> Vec<f64> {
        self.solutions.iter().map(|s| s[node_index]).collect()
    }

    /// Calculate statistics for a node voltage across all points.
    pub fn statistics(&self, node_index: usize) -> SweepStatistics {
        let voltages = self.node_voltages(node_index);
        SweepStatistics::from_samples(&voltages)
    }
}

/// Statistics for a sweep result.
#[derive(Debug, Clone)]
pub struct SweepStatistics {
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Number of samples.
    pub count: usize,
}

impl SweepStatistics {
    /// Calculate statistics from samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }

        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean,
            std_dev,
            min,
            max,
            count,
        }
    }
}

/// Generator for sweep points.
pub trait SweepPointGenerator: Send + Sync {
    /// Generate all sweep points.
    fn generate(&self, variations: &[ParameterVariation]) -> Vec<SweepPoint>;
}

/// Monte Carlo point generator with random sampling.
#[derive(Debug, Clone)]
pub struct MonteCarloGenerator {
    /// Number of samples to generate.
    pub num_samples: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl MonteCarloGenerator {
    /// Create a new Monte Carlo generator.
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            seed: 12345,
        }
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl SweepPointGenerator for MonteCarloGenerator {
    fn generate(&self, variations: &[ParameterVariation]) -> Vec<SweepPoint> {
        // Simple LCG random number generator for reproducibility
        let mut rng_state = self.seed;
        let mut next_random = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [0, 1)
            (rng_state >> 33) as f64 / (1u64 << 31) as f64
        };

        // Box-Muller transform for normal distribution
        let mut next_normal = || -> f64 {
            let u1 = next_random().max(1e-10);
            let u2 = next_random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        (0..self.num_samples)
            .map(|_| {
                let parameters: Vec<f64> = variations
                    .iter()
                    .map(|v| {
                        let z = next_normal();
                        (v.nominal + z * v.sigma * v.nominal).clamp(v.min, v.max)
                    })
                    .collect();
                SweepPoint { parameters }
            })
            .collect()
    }
}

/// Corner analysis generator (all combinations of min/max).
#[derive(Debug, Clone)]
pub struct CornerGenerator;

impl SweepPointGenerator for CornerGenerator {
    fn generate(&self, variations: &[ParameterVariation]) -> Vec<SweepPoint> {
        if variations.is_empty() {
            return vec![SweepPoint { parameters: vec![] }];
        }

        // Generate 2^n corners
        let n = variations.len();
        let num_corners = 1 << n;

        (0..num_corners)
            .map(|i| {
                let parameters: Vec<f64> = variations
                    .iter()
                    .enumerate()
                    .map(|(j, v)| if (i >> j) & 1 == 0 { v.min } else { v.max })
                    .collect();
                SweepPoint { parameters }
            })
            .collect()
    }
}

/// Linear sweep generator for single parameter.
#[derive(Debug, Clone)]
pub struct LinearSweepGenerator {
    /// Number of points.
    pub num_points: usize,
}

impl LinearSweepGenerator {
    /// Create a new linear sweep generator.
    pub fn new(num_points: usize) -> Self {
        Self { num_points }
    }
}

impl SweepPointGenerator for LinearSweepGenerator {
    fn generate(&self, variations: &[ParameterVariation]) -> Vec<SweepPoint> {
        if variations.is_empty() {
            return vec![SweepPoint { parameters: vec![] }];
        }

        // For single parameter, sweep from min to max
        // For multiple parameters, sweep all simultaneously
        (0..self.num_points)
            .map(|i| {
                let t = if self.num_points > 1 {
                    i as f64 / (self.num_points - 1) as f64
                } else {
                    0.5
                };
                let parameters: Vec<f64> = variations
                    .iter()
                    .map(|v| v.min + t * (v.max - v.min))
                    .collect();
                SweepPoint { parameters }
            })
            .collect()
    }
}

/// Stamper factory for creating stampers with varied parameters.
///
/// This trait allows the sweep solver to create new stampers for each
/// parameter combination.
pub trait SweepStamperFactory: Send + Sync {
    /// Create a stamper for the given parameter values.
    fn create_stamper(&self, parameters: &[f64]) -> Arc<dyn SweepStamper>;
}

/// Stamper for a single sweep point.
pub trait SweepStamper: Send + Sync {
    /// Stamp linear devices into the MNA system.
    fn stamp_linear(&self, matrix: &mut nalgebra::DMatrix<f64>, rhs: &mut DVector<f64>);

    /// Number of nodes.
    fn num_nodes(&self) -> usize;

    /// Number of current variables.
    fn num_vsources(&self) -> usize;

    /// Stamp linear devices into sparse triplet format.
    ///
    /// Returns (matrix_triplets, rhs_values) where:
    /// - matrix_triplets: Vec of (row, col, value) for non-zero matrix entries
    /// - rhs_values: Vec of (row, value) for non-zero RHS entries
    ///
    /// Default implementation uses dense stamping and extracts non-zeros.
    /// Override for better performance with direct triplet generation.
    #[allow(clippy::type_complexity)]
    fn stamp_triplets(&self) -> (Vec<(usize, usize, f64)>, Vec<(usize, f64)>) {
        let size = self.num_nodes() + self.num_vsources();
        let mut matrix = nalgebra::DMatrix::zeros(size, size);
        let mut rhs = DVector::zeros(size);

        self.stamp_linear(&mut matrix, &mut rhs);

        // Extract non-zero matrix entries
        let mut mat_triplets = Vec::new();
        for col in 0..size {
            for row in 0..size {
                let val = matrix[(row, col)];
                if val.abs() > 1e-15 {
                    mat_triplets.push((row, col, val));
                }
            }
        }

        // Extract non-zero RHS entries
        let mut rhs_values = Vec::new();
        for row in 0..size {
            let val = rhs[row];
            if val.abs() > 1e-15 {
                rhs_values.push((row, val));
            }
        }

        (mat_triplets, rhs_values)
    }
}

/// Execute a batched sweep analysis.
///
/// This function runs multiple simulations in parallel across sweep points.
///
/// # Arguments
/// * `factory` - Factory for creating stampers with varied parameters
/// * `generator` - Generator for sweep points
/// * `variations` - Parameter variations to sweep
/// * `criteria` - Convergence criteria (for nonlinear circuits)
pub fn solve_batched_sweep(
    factory: &dyn SweepStamperFactory,
    generator: &dyn SweepPointGenerator,
    variations: &[ParameterVariation],
    _criteria: &ConvergenceCriteria,
) -> Result<BatchedSweepResult> {
    let points = generator.generate(variations);
    let total_count = points.len();
    let mut solutions = Vec::with_capacity(total_count);
    let mut converged_count = 0;

    // For linear circuits, we can parallelize across points
    // Note: Using sequential for now; could use rayon for parallel execution
    for point in &points {
        let stamper = factory.create_stamper(&point.parameters);
        let size = stamper.num_nodes() + stamper.num_vsources();

        let mut matrix = nalgebra::DMatrix::zeros(size, size);
        let mut rhs = DVector::zeros(size);

        stamper.stamp_linear(&mut matrix, &mut rhs);

        match solve_dense(&matrix, &rhs) {
            Ok(solution) => {
                solutions.push(solution);
                converged_count += 1;
            }
            Err(_) => {
                // Use zeros for failed point
                solutions.push(DVector::zeros(size));
            }
        }
    }

    Ok(BatchedSweepResult {
        solutions,
        points,
        converged_count,
        total_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_generator() {
        let generator = MonteCarloGenerator::new(100).with_seed(42);
        let variations = vec![
            ParameterVariation::new("R1", 1000.0).with_sigma(0.1),
            ParameterVariation::new("R2", 2000.0).with_sigma(0.05),
        ];

        let points = generator.generate(&variations);
        assert_eq!(points.len(), 100);

        // Check all values are within bounds
        for point in &points {
            assert!(point.parameters[0] >= 900.0 && point.parameters[0] <= 1100.0);
            assert!(point.parameters[1] >= 1800.0 && point.parameters[1] <= 2200.0);
        }

        // Check reproducibility
        let gen2 = MonteCarloGenerator::new(100).with_seed(42);
        let points2 = gen2.generate(&variations);
        assert_eq!(points[0].parameters, points2[0].parameters);
    }

    #[test]
    fn test_corner_generator() {
        let generator = CornerGenerator;
        let variations = vec![
            ParameterVariation::new("R1", 1000.0).with_bounds(900.0, 1100.0),
            ParameterVariation::new("R2", 2000.0).with_bounds(1800.0, 2200.0),
        ];

        let points = generator.generate(&variations);
        assert_eq!(points.len(), 4); // 2^2 corners

        // Check we have all corners
        let mut has_min_min = false;
        let mut has_min_max = false;
        let mut has_max_min = false;
        let mut has_max_max = false;

        for point in &points {
            match (point.parameters[0] as i32, point.parameters[1] as i32) {
                (900, 1800) => has_min_min = true,
                (900, 2200) => has_min_max = true,
                (1100, 1800) => has_max_min = true,
                (1100, 2200) => has_max_max = true,
                _ => panic!("Unexpected corner: {:?}", point.parameters),
            }
        }

        assert!(has_min_min && has_min_max && has_max_min && has_max_max);
    }

    #[test]
    fn test_linear_sweep_generator() {
        let generator = LinearSweepGenerator::new(5);
        let variations = vec![ParameterVariation::new("V1", 5.0).with_bounds(0.0, 10.0)];

        let points = generator.generate(&variations);
        assert_eq!(points.len(), 5);

        // Check sweep is linear
        assert!((points[0].parameters[0] - 0.0).abs() < 1e-10);
        assert!((points[1].parameters[0] - 2.5).abs() < 1e-10);
        assert!((points[2].parameters[0] - 5.0).abs() < 1e-10);
        assert!((points[3].parameters[0] - 7.5).abs() < 1e-10);
        assert!((points[4].parameters[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sweep_statistics() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SweepStatistics::from_samples(&samples);

        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert_eq!(stats.count, 5);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_batched_sweep_simple() {
        // Simple voltage divider sweep

        struct DividerFactory {
            r2_nominal: f64,
        }

        impl SweepStamperFactory for DividerFactory {
            fn create_stamper(&self, parameters: &[f64]) -> Arc<dyn SweepStamper> {
                let r1 = parameters.first().copied().unwrap_or(1000.0);
                Arc::new(DividerStamper {
                    r1,
                    r2: self.r2_nominal,
                    v_source: 10.0,
                })
            }
        }

        struct DividerStamper {
            r1: f64,
            r2: f64,
            v_source: f64,
        }

        impl SweepStamper for DividerStamper {
            fn stamp_linear(&self, matrix: &mut nalgebra::DMatrix<f64>, rhs: &mut DVector<f64>) {
                let g1 = 1.0 / self.r1;
                let g2 = 1.0 / self.r2;

                // Stamp R1 between node 0 and 1
                matrix[(0, 0)] += g1;
                matrix[(1, 1)] += g1;
                matrix[(0, 1)] -= g1;
                matrix[(1, 0)] -= g1;

                // Stamp R2 between node 1 and ground
                matrix[(1, 1)] += g2;

                // Stamp voltage source at node 0 (branch current at index 2)
                matrix[(0, 2)] += 1.0;
                matrix[(2, 0)] += 1.0;
                rhs[2] = self.v_source;
            }

            fn num_nodes(&self) -> usize {
                2
            }

            fn num_vsources(&self) -> usize {
                1
            }
        }

        let factory = DividerFactory { r2_nominal: 1000.0 };
        let generator = LinearSweepGenerator::new(5);
        let variations = vec![
            // Sweep R1 from 500 to 1500 so midpoint is R1=1000=R2
            ParameterVariation::new("R1", 1000.0).with_bounds(500.0, 1500.0),
        ];

        let result = solve_batched_sweep(
            &factory,
            &generator,
            &variations,
            &ConvergenceCriteria::default(),
        )
        .expect("Sweep should succeed");

        assert_eq!(result.total_count, 5);
        assert_eq!(result.converged_count, 5);

        // V(1) should decrease as R1 increases (voltage divider)
        let v1_first = result.solutions[0][1];
        let v1_last = result.solutions[4][1];
        assert!(v1_first > v1_last, "V(1) should decrease as R1 increases");

        // At midpoint R1 = R2 = 1000, V(1) should be 5V (voltage divider)
        let v1_mid = result.solutions[2][1];
        assert!(
            (v1_mid - 5.0).abs() < 0.01,
            "V(1) = {} at R1=R2 (expected 5V)",
            v1_mid
        );
    }
}
