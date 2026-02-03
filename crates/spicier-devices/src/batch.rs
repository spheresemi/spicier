//! Batched device evaluation with SIMD-friendly Structure-of-Arrays layout.
//!
//! When a circuit has many devices of the same type, batching them together
//! allows SIMD instructions to evaluate multiple devices simultaneously.
//!
//! # Structure-of-Arrays (SoA) vs Array-of-Structures (AoS)
//!
//! Traditional AoS layout (one device at a time):
//! ```text
//! Device0: [Is, n, Vt, node_p, node_n]
//! Device1: [Is, n, Vt, node_p, node_n]
//! Device2: [Is, n, Vt, node_p, node_n]
//! ```
//!
//! SIMD-friendly SoA layout (parameters grouped):
//! ```text
//! Is:     [Is0, Is1, Is2, Is3, ...]  <- SIMD can load 4-8 at once
//! n:      [n0,  n1,  n2,  n3,  ...]
//! node_p: [p0,  p1,  p2,  p3,  ...]
//! node_n: [n0,  n1,  n2,  n3,  ...]
//! ```

use spicier_simd::SimdCapability;

/// SIMD lane count for f64 with AVX2.
pub const SIMD_LANES_AVX2: usize = 4;

/// Round up to next multiple of SIMD lane count for padding.
#[inline]
pub fn round_up_to_simd(count: usize) -> usize {
    count.div_ceil(SIMD_LANES_AVX2) * SIMD_LANES_AVX2
}

// ============================================================================
// Diode Batch
// ============================================================================

/// Batched diode parameters in SoA layout.
///
/// All arrays have the same length. For SIMD efficiency, arrays may be
/// padded to a multiple of the SIMD lane count.
#[derive(Debug, Clone)]
pub struct DiodeBatch {
    /// Number of actual diodes (before padding).
    pub count: usize,
    /// Saturation current (A) for each diode.
    pub is: Vec<f64>,
    /// Emission coefficient for each diode.
    pub n: Vec<f64>,
    /// Pre-computed n * Vt for each diode.
    pub nvt: Vec<f64>,
    /// Positive node index (None = ground represented as usize::MAX).
    pub node_pos: Vec<usize>,
    /// Negative node index (None = ground represented as usize::MAX).
    pub node_neg: Vec<usize>,
}

impl DiodeBatch {
    /// Create a new empty diode batch.
    pub fn new() -> Self {
        Self {
            count: 0,
            is: Vec::new(),
            n: Vec::new(),
            nvt: Vec::new(),
            node_pos: Vec::new(),
            node_neg: Vec::new(),
        }
    }

    /// Create a diode batch with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let padded = round_up_to_simd(capacity);
        Self {
            count: 0,
            is: Vec::with_capacity(padded),
            n: Vec::with_capacity(padded),
            nvt: Vec::with_capacity(padded),
            node_pos: Vec::with_capacity(padded),
            node_neg: Vec::with_capacity(padded),
        }
    }

    /// Add a diode to the batch.
    ///
    /// `node_pos` and `node_neg` should be `Some(index)` or `None` for ground.
    pub fn push(&mut self, is: f64, n: f64, node_pos: Option<usize>, node_neg: Option<usize>) {
        const VT: f64 = 0.025851997; // Thermal voltage at 300.15K

        self.is.push(is);
        self.n.push(n);
        self.nvt.push(n * VT);
        self.node_pos.push(node_pos.unwrap_or(usize::MAX));
        self.node_neg.push(node_neg.unwrap_or(usize::MAX));
        self.count += 1;
    }

    /// Pad arrays to SIMD lane count and return actual count.
    ///
    /// Padding uses neutral values (Is=1e-14, n=1, nvt=0.026) that produce
    /// near-zero current and won't affect the circuit.
    pub fn finalize(&mut self) {
        let target = round_up_to_simd(self.count);
        while self.is.len() < target {
            self.is.push(1e-14);
            self.n.push(1.0);
            self.nvt.push(0.026);
            self.node_pos.push(usize::MAX); // Ground
            self.node_neg.push(usize::MAX); // Ground
        }
    }

    /// Evaluate all diodes in the batch.
    ///
    /// # Arguments
    /// - `voltages`: Node voltage solution vector
    /// - `id_out`: Output current for each diode (must be pre-allocated to count)
    /// - `gd_out`: Output conductance for each diode (must be pre-allocated to count)
    /// - `capability`: SIMD capability level
    pub fn evaluate_batch(
        &self,
        voltages: &[f64],
        id_out: &mut [f64],
        gd_out: &mut [f64],
        capability: SimdCapability,
    ) {
        debug_assert!(id_out.len() >= self.count);
        debug_assert!(gd_out.len() >= self.count);

        match capability {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdCapability::Avx512 | SimdCapability::Avx2 => {
                unsafe { self.evaluate_batch_avx2(voltages, id_out, gd_out) };
            }
            SimdCapability::Scalar | _ => {
                self.evaluate_batch_scalar(voltages, id_out, gd_out);
            }
        }
    }

    /// Scalar implementation of batch evaluation.
    pub fn evaluate_batch_scalar(&self, voltages: &[f64], id_out: &mut [f64], gd_out: &mut [f64]) {
        for i in 0..self.count {
            let vp = if self.node_pos[i] == usize::MAX {
                0.0
            } else {
                voltages[self.node_pos[i]]
            };
            let vn = if self.node_neg[i] == usize::MAX {
                0.0
            } else {
                voltages[self.node_neg[i]]
            };
            let vd = vp - vn;

            let nvt = self.nvt[i];
            let vd_limited = limit_voltage_scalar(vd, nvt);
            let exp_term = (vd_limited / nvt).exp();

            id_out[i] = self.is[i] * (exp_term - 1.0);
            gd_out[i] = (self.is[i] * exp_term / nvt).max(1e-12);
        }
    }

    /// AVX2 implementation of batch evaluation.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn evaluate_batch_avx2(&self, voltages: &[f64], id_out: &mut [f64], gd_out: &mut [f64]) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let chunks = self.count / 4;

        for chunk in 0..chunks {
            let base = chunk * 4;

            // Load 4 diode voltages from solution vector
            let vd = self.load_voltages_avx2(base, voltages);

            // Load parameters
            let is = _mm256_loadu_pd(self.is.as_ptr().add(base));
            let nvt = _mm256_loadu_pd(self.nvt.as_ptr().add(base));

            // Apply voltage limiting
            let vd_limited = limit_voltage_avx2(vd, nvt);

            // Compute exp(Vd / nvt) - use scalar exp for each element
            // (AVX2 doesn't have native exp, but we can call scalar exp efficiently)
            let vd_over_nvt = _mm256_div_pd(vd_limited, nvt);
            let mut exp_arr = [0.0f64; 4];
            _mm256_storeu_pd(exp_arr.as_mut_ptr(), vd_over_nvt);
            for j in 0..4 {
                exp_arr[j] = exp_arr[j].exp();
            }
            let exp_term = _mm256_loadu_pd(exp_arr.as_ptr());

            // Id = Is * (exp - 1)
            let one = _mm256_set1_pd(1.0);
            let id = _mm256_mul_pd(is, _mm256_sub_pd(exp_term, one));

            // Gd = Is * exp / nvt
            let gd_raw = _mm256_div_pd(_mm256_mul_pd(is, exp_term), nvt);
            // Apply minimum conductance
            let min_gd = _mm256_set1_pd(1e-12);
            let gd = _mm256_max_pd(gd_raw, min_gd);

            // Store results
            _mm256_storeu_pd(id_out.as_mut_ptr().add(base), id);
            _mm256_storeu_pd(gd_out.as_mut_ptr().add(base), gd);
        }

        // Handle remainder with scalar code
        for i in (chunks * 4)..self.count {
            let vp = if self.node_pos[i] == usize::MAX {
                0.0
            } else {
                voltages[self.node_pos[i]]
            };
            let vn = if self.node_neg[i] == usize::MAX {
                0.0
            } else {
                voltages[self.node_neg[i]]
            };
            let vd = vp - vn;

            let nvt = self.nvt[i];
            let vd_limited = limit_voltage_scalar(vd, nvt);
            let exp_term = (vd_limited / nvt).exp();

            id_out[i] = self.is[i] * (exp_term - 1.0);
            gd_out[i] = (self.is[i] * exp_term / nvt).max(1e-12);
        }
    }

    /// Load 4 diode voltages from the solution vector (AVX2).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn load_voltages_avx2(
        &self,
        base: usize,
        voltages: &[f64],
    ) -> std::arch::x86_64::__m256d {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mut vd = [0.0f64; 4];
        for i in 0..4 {
            let idx = base + i;
            let vp = if self.node_pos[idx] == usize::MAX {
                0.0
            } else {
                voltages[self.node_pos[idx]]
            };
            let vn = if self.node_neg[idx] == usize::MAX {
                0.0
            } else {
                voltages[self.node_neg[idx]]
            };
            vd[i] = vp - vn;
        }
        _mm256_loadu_pd(vd.as_ptr())
    }

    /// Evaluate batch and compute linearized stamp values.
    ///
    /// Returns (id, gd, ieq) where ieq = id - gd * vd is the companion current source.
    pub fn evaluate_linearized_batch(
        &self,
        voltages: &[f64],
        id_out: &mut [f64],
        gd_out: &mut [f64],
        ieq_out: &mut [f64],
        capability: SimdCapability,
    ) {
        debug_assert!(ieq_out.len() >= self.count);

        // First evaluate to get id and gd
        self.evaluate_batch(voltages, id_out, gd_out, capability);

        // Compute ieq = id - gd * vd for each diode
        for i in 0..self.count {
            let vp = if self.node_pos[i] == usize::MAX {
                0.0
            } else {
                voltages[self.node_pos[i]]
            };
            let vn = if self.node_neg[i] == usize::MAX {
                0.0
            } else {
                voltages[self.node_neg[i]]
            };
            let vd = limit_voltage_scalar(vp - vn, self.nvt[i]);
            ieq_out[i] = id_out[i] - gd_out[i] * vd;
        }
    }
}

impl Default for DiodeBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar voltage limiting (for remainder handling).
#[inline]
fn limit_voltage_scalar(vd: f64, nvt: f64) -> f64 {
    let vcrit = nvt * (nvt / (std::f64::consts::SQRT_2 * 1e-14)).ln();
    if vd > vcrit {
        let arg = (vd - vcrit) / nvt;
        vcrit + nvt * (1.0 + arg).ln_1p()
    } else {
        vd
    }
}

/// AVX2 voltage limiting.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn limit_voltage_avx2(
    vd: std::arch::x86_64::__m256d,
    nvt: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // vcrit = nvt * ln(nvt / (sqrt(2) * 1e-14))
    // We need scalar ln, so extract and compute
    let mut nvt_arr = [0.0f64; 4];
    let mut vd_arr = [0.0f64; 4];
    _mm256_storeu_pd(nvt_arr.as_mut_ptr(), nvt);
    _mm256_storeu_pd(vd_arr.as_mut_ptr(), vd);

    let sqrt2_is = std::f64::consts::SQRT_2 * 1e-14;
    let mut result = [0.0f64; 4];
    for i in 0..4 {
        let vcrit = nvt_arr[i] * (nvt_arr[i] / sqrt2_is).ln();
        if vd_arr[i] > vcrit {
            let arg = (vd_arr[i] - vcrit) / nvt_arr[i];
            result[i] = vcrit + nvt_arr[i] * (1.0 + arg).ln_1p();
        } else {
            result[i] = vd_arr[i];
        }
    }
    _mm256_loadu_pd(result.as_ptr())
}

// ============================================================================
// MOSFET Batch
// ============================================================================

/// MOSFET type encoded for batch processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BatchMosfetType {
    Nmos = 0,
    Pmos = 1,
}

/// Batched MOSFET parameters in SoA layout.
#[derive(Debug, Clone)]
pub struct MosfetBatch {
    /// Number of actual MOSFETs (before padding).
    pub count: usize,
    /// MOSFET type (NMOS/PMOS).
    pub mos_type: Vec<BatchMosfetType>,
    /// Threshold voltage magnitude (always positive).
    pub vth: Vec<f64>,
    /// Beta = kp * W / L.
    pub beta: Vec<f64>,
    /// Channel-length modulation parameter.
    pub lambda: Vec<f64>,
    /// Drain node index.
    pub node_drain: Vec<usize>,
    /// Gate node index.
    pub node_gate: Vec<usize>,
    /// Source node index.
    pub node_source: Vec<usize>,
}

impl MosfetBatch {
    /// Create a new empty MOSFET batch.
    pub fn new() -> Self {
        Self {
            count: 0,
            mos_type: Vec::new(),
            vth: Vec::new(),
            beta: Vec::new(),
            lambda: Vec::new(),
            node_drain: Vec::new(),
            node_gate: Vec::new(),
            node_source: Vec::new(),
        }
    }

    /// Create a MOSFET batch with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let padded = round_up_to_simd(capacity);
        Self {
            count: 0,
            mos_type: Vec::with_capacity(padded),
            vth: Vec::with_capacity(padded),
            beta: Vec::with_capacity(padded),
            lambda: Vec::with_capacity(padded),
            node_drain: Vec::with_capacity(padded),
            node_gate: Vec::with_capacity(padded),
            node_source: Vec::with_capacity(padded),
        }
    }

    /// Add a MOSFET to the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        mos_type: BatchMosfetType,
        vth: f64,
        beta: f64,
        lambda: f64,
        node_drain: Option<usize>,
        node_gate: Option<usize>,
        node_source: Option<usize>,
    ) {
        self.mos_type.push(mos_type);
        self.vth.push(vth.abs());
        self.beta.push(beta);
        self.lambda.push(lambda);
        self.node_drain.push(node_drain.unwrap_or(usize::MAX));
        self.node_gate.push(node_gate.unwrap_or(usize::MAX));
        self.node_source.push(node_source.unwrap_or(usize::MAX));
        self.count += 1;
    }

    /// Pad arrays to SIMD lane count.
    pub fn finalize(&mut self) {
        let target = round_up_to_simd(self.count);
        while self.mos_type.len() < target {
            // Padding MOSFETs: NMOS in cutoff (vth very high)
            self.mos_type.push(BatchMosfetType::Nmos);
            self.vth.push(1e6); // Very high threshold = always cutoff
            self.beta.push(0.0);
            self.lambda.push(0.0);
            self.node_drain.push(usize::MAX);
            self.node_gate.push(usize::MAX);
            self.node_source.push(usize::MAX);
        }
    }

    /// Evaluate all MOSFETs in the batch.
    ///
    /// # Arguments
    /// - `voltages`: Node voltage solution vector
    /// - `ids_out`: Drain-source current
    /// - `gds_out`: Output conductance (dIds/dVds)
    /// - `gm_out`: Transconductance (dIds/dVgs)
    /// - `_capability`: SIMD capability (currently scalar only due to branching)
    pub fn evaluate_batch(
        &self,
        voltages: &[f64],
        ids_out: &mut [f64],
        gds_out: &mut [f64],
        gm_out: &mut [f64],
        _capability: SimdCapability,
    ) {
        debug_assert!(ids_out.len() >= self.count);
        debug_assert!(gds_out.len() >= self.count);
        debug_assert!(gm_out.len() >= self.count);

        // MOSFET evaluation has significant branching (cutoff/linear/saturation)
        // A branchless SIMD implementation would require computing all three regions
        // and selecting via masks. For now, use scalar with manual unrolling.
        self.evaluate_batch_scalar(voltages, ids_out, gds_out, gm_out);
    }

    /// Scalar implementation of MOSFET batch evaluation.
    fn evaluate_batch_scalar(
        &self,
        voltages: &[f64],
        ids_out: &mut [f64],
        gds_out: &mut [f64],
        gm_out: &mut [f64],
    ) {
        for i in 0..self.count {
            let vd = self.get_voltage(self.node_drain[i], voltages);
            let vg = self.get_voltage(self.node_gate[i], voltages);
            let vs = self.get_voltage(self.node_source[i], voltages);

            let (vgs, vds) = match self.mos_type[i] {
                BatchMosfetType::Nmos => (vg - vs, vd - vs),
                BatchMosfetType::Pmos => (vs - vg, vs - vd),
            };

            let vth = self.vth[i];
            let beta = self.beta[i];
            let lambda = self.lambda[i];

            let (ids, gds, gm) = if vgs < vth {
                // Cutoff
                (0.0, 1e-12, 0.0)
            } else if vds < vgs - vth {
                // Linear
                let vov = vgs - vth;
                let ids = beta * (vov * vds - 0.5 * vds * vds) * (1.0 + lambda * vds);
                let gds = beta * (vov - vds) * (1.0 + lambda * vds)
                    + beta * (vov * vds - 0.5 * vds * vds) * lambda;
                let gm = beta * vds * (1.0 + lambda * vds);
                (ids, gds.max(1e-12), gm)
            } else {
                // Saturation
                let vov = vgs - vth;
                let ids = 0.5 * beta * vov * vov * (1.0 + lambda * vds);
                let gds = 0.5 * beta * vov * vov * lambda;
                let gm = beta * vov * (1.0 + lambda * vds);
                (ids, gds.max(1e-12), gm)
            };

            // Apply sign for PMOS
            match self.mos_type[i] {
                BatchMosfetType::Nmos => {
                    ids_out[i] = ids;
                    gds_out[i] = gds;
                    gm_out[i] = gm;
                }
                BatchMosfetType::Pmos => {
                    ids_out[i] = -ids;
                    gds_out[i] = gds;
                    gm_out[i] = gm;
                }
            }
        }
    }

    /// Evaluate batch and compute linearized stamp values.
    ///
    /// Returns (ids, gds, gm, ieq) where ieq = ids - gds*vds - gm*vgs.
    pub fn evaluate_linearized_batch(
        &self,
        voltages: &[f64],
        ids_out: &mut [f64],
        gds_out: &mut [f64],
        gm_out: &mut [f64],
        ieq_out: &mut [f64],
        capability: SimdCapability,
    ) {
        self.evaluate_batch(voltages, ids_out, gds_out, gm_out, capability);

        for i in 0..self.count {
            let vd = self.get_voltage(self.node_drain[i], voltages);
            let vg = self.get_voltage(self.node_gate[i], voltages);
            let vs = self.get_voltage(self.node_source[i], voltages);

            let (vgs, vds) = match self.mos_type[i] {
                BatchMosfetType::Nmos => (vg - vs, vd - vs),
                BatchMosfetType::Pmos => (vs - vg, vs - vd),
            };

            // ieq = ids - gds*vds - gm*vgs
            // Note: ids already has the PMOS sign applied
            let ids_unsigned = ids_out[i].abs();
            ieq_out[i] = match self.mos_type[i] {
                BatchMosfetType::Nmos => ids_unsigned - gds_out[i] * vds - gm_out[i] * vgs,
                BatchMosfetType::Pmos => -(ids_unsigned - gds_out[i] * vds - gm_out[i] * vgs),
            };
        }
    }

    #[inline]
    fn get_voltage(&self, node: usize, voltages: &[f64]) -> f64 {
        if node == usize::MAX {
            0.0
        } else {
            voltages[node]
        }
    }
}

impl Default for MosfetBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diode_batch_single() {
        let mut batch = DiodeBatch::new();
        batch.push(1e-14, 1.0, Some(0), None); // Node 0 to ground
        batch.finalize();

        let voltages = [0.7]; // 0.7V forward bias
        let mut id = vec![0.0; batch.count];
        let mut gd = vec![0.0; batch.count];

        batch.evaluate_batch(&voltages, &mut id, &mut gd, SimdCapability::Scalar);

        assert!(id[0] > 0.0, "Forward current should be positive");
        assert!(gd[0] > 0.0, "Conductance should be positive");
    }

    #[test]
    fn test_diode_batch_multiple() {
        let mut batch = DiodeBatch::new();
        // Add 5 diodes with varying Is
        for i in 0..5 {
            batch.push(1e-14 * (i as f64 + 1.0), 1.0, Some(i), None);
        }
        batch.finalize();

        let voltages = [0.6, 0.65, 0.7, 0.75, 0.8];
        let mut id = vec![0.0; batch.count];
        let mut gd = vec![0.0; batch.count];

        batch.evaluate_batch(&voltages, &mut id, &mut gd, SimdCapability::Scalar);

        // All should have positive current at forward bias
        for (i, &current) in id.iter().enumerate().take(5) {
            assert!(current > 0.0, "Diode {} should have positive current", i);
        }
    }

    #[test]
    fn test_diode_batch_simd_vs_scalar() {
        let cap = SimdCapability::detect();

        let mut batch = DiodeBatch::new();
        // Add 17 diodes (not a multiple of 4) to test both SIMD and remainder
        for i in 0..17 {
            batch.push(
                1e-14 * (i as f64 % 5.0 + 1.0),
                1.0 + (i as f64) * 0.02,
                Some(i),
                None,
            );
        }
        batch.finalize();

        let voltages: Vec<f64> = (0..17).map(|i| 0.5 + (i as f64) * 0.02).collect();

        let mut id_scalar = vec![0.0; batch.count];
        let mut gd_scalar = vec![0.0; batch.count];
        let mut id_simd = vec![0.0; batch.count];
        let mut gd_simd = vec![0.0; batch.count];

        batch.evaluate_batch_scalar(&voltages, &mut id_scalar, &mut gd_scalar);
        batch.evaluate_batch(&voltages, &mut id_simd, &mut gd_simd, cap);

        for i in 0..17 {
            let id_diff = (id_scalar[i] - id_simd[i]).abs();
            let gd_diff = (gd_scalar[i] - gd_simd[i]).abs();
            assert!(
                id_diff < 1e-12,
                "Diode {}: Id scalar {} vs simd {} (cap={:?})",
                i,
                id_scalar[i],
                id_simd[i],
                cap
            );
            assert!(
                gd_diff < 1e-12,
                "Diode {}: Gd scalar {} vs simd {}",
                i,
                gd_scalar[i],
                gd_simd[i]
            );
        }
    }

    #[test]
    fn test_diode_batch_linearized() {
        let mut batch = DiodeBatch::new();
        batch.push(1e-14, 1.0, Some(0), None);
        batch.finalize();

        let voltages = [0.7];
        let mut id = vec![0.0; batch.count];
        let mut gd = vec![0.0; batch.count];
        let mut ieq = vec![0.0; batch.count];

        batch.evaluate_linearized_batch(
            &voltages,
            &mut id,
            &mut gd,
            &mut ieq,
            SimdCapability::Scalar,
        );

        assert!(gd[0] > 0.0);
    }

    #[test]
    fn test_mosfet_batch_cutoff() {
        let mut batch = MosfetBatch::new();
        batch.push(
            BatchMosfetType::Nmos,
            0.7,     // vth
            2e-4,    // beta
            0.0,     // lambda
            Some(0), // drain
            Some(1), // gate
            None,    // source = ground
        );
        batch.finalize();

        let voltages = [1.0, 0.3]; // Vds=1.0, Vgs=0.3 < Vth=0.7
        let mut ids = vec![0.0; batch.count];
        let mut gds = vec![0.0; batch.count];
        let mut gm = vec![0.0; batch.count];

        batch.evaluate_batch(
            &voltages,
            &mut ids,
            &mut gds,
            &mut gm,
            SimdCapability::Scalar,
        );

        assert_eq!(ids[0], 0.0, "Cutoff: Ids should be 0");
        assert_eq!(gm[0], 0.0, "Cutoff: gm should be 0");
    }

    #[test]
    fn test_mosfet_batch_saturation() {
        let mut batch = MosfetBatch::new();
        batch.push(
            BatchMosfetType::Nmos,
            0.7,
            2e-4,
            0.0,
            Some(0),
            Some(1),
            None,
        );
        batch.finalize();

        let voltages = [5.0, 2.0]; // Vds=5.0, Vgs=2.0, Vov=1.3, Vds > Vov -> saturation
        let mut ids = vec![0.0; batch.count];
        let mut gds = vec![0.0; batch.count];
        let mut gm = vec![0.0; batch.count];

        batch.evaluate_batch(
            &voltages,
            &mut ids,
            &mut gds,
            &mut gm,
            SimdCapability::Scalar,
        );

        // Ids = beta/2 * Vov^2 = 2e-4/2 * 1.3^2 = 1e-4 * 1.69 = 1.69e-4
        let expected = 0.5 * 2e-4 * 1.3 * 1.3;
        assert!(
            (ids[0] - expected).abs() < 1e-10,
            "Saturation Ids: {} vs expected {}",
            ids[0],
            expected
        );
        assert!(gm[0] > 0.0, "Saturation: gm should be positive");
    }

    #[test]
    fn test_mosfet_batch_linear() {
        let mut batch = MosfetBatch::new();
        batch.push(
            BatchMosfetType::Nmos,
            0.7,
            2e-4,
            0.0,
            Some(0),
            Some(1),
            None,
        );
        batch.finalize();

        let voltages = [0.5, 2.0]; // Vds=0.5, Vgs=2.0, Vov=1.3, Vds < Vov -> linear
        let mut ids = vec![0.0; batch.count];
        let mut gds = vec![0.0; batch.count];
        let mut gm = vec![0.0; batch.count];

        batch.evaluate_batch(
            &voltages,
            &mut ids,
            &mut gds,
            &mut gm,
            SimdCapability::Scalar,
        );

        // Ids = beta * (Vov*Vds - Vds^2/2) = 2e-4 * (1.3*0.5 - 0.125) = 2e-4 * 0.525
        let expected = 2e-4 * (1.3 * 0.5 - 0.5 * 0.5 * 0.5);
        assert!(
            (ids[0] - expected).abs() < 1e-10,
            "Linear Ids: {} vs expected {}",
            ids[0],
            expected
        );
        assert!(gds[0] > 0.0, "Linear: gds should be positive");
    }

    #[test]
    fn test_pmos_batch() {
        let mut batch = MosfetBatch::new();
        batch.push(
            BatchMosfetType::Pmos,
            0.7,
            1e-4,
            0.0,
            Some(0), // drain
            Some(1), // gate
            Some(2), // source
        );
        batch.finalize();

        // PMOS: Vsg = 2V (source at 5V, gate at 3V), Vsd = 3V (source at 5V, drain at 2V)
        let voltages = [2.0, 3.0, 5.0]; // drain, gate, source
        let mut ids = vec![0.0; batch.count];
        let mut gds = vec![0.0; batch.count];
        let mut gm = vec![0.0; batch.count];

        batch.evaluate_batch(
            &voltages,
            &mut ids,
            &mut gds,
            &mut gm,
            SimdCapability::Scalar,
        );

        assert!(ids[0] < 0.0, "PMOS Ids should be negative: {}", ids[0]);
    }

    #[test]
    fn test_round_up_to_simd() {
        assert_eq!(round_up_to_simd(0), 0);
        assert_eq!(round_up_to_simd(1), 4);
        assert_eq!(round_up_to_simd(4), 4);
        assert_eq!(round_up_to_simd(5), 8);
        assert_eq!(round_up_to_simd(8), 8);
        assert_eq!(round_up_to_simd(9), 12);
    }
}
