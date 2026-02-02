//! BJT (Bipolar Junction Transistor) device model using Ebers-Moll equations.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

use crate::diode::thermal_voltage;
use crate::stamp::Stamp;

/// BJT type (NPN or PNP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BjtType {
    /// NPN transistor.
    Npn,
    /// PNP transistor.
    Pnp,
}

/// BJT model parameters (simplified Ebers-Moll / Gummel-Poon).
#[derive(Debug, Clone)]
pub struct BjtParams {
    /// Saturation current (A). Default: 1e-16.
    pub is: f64,
    /// Forward current gain (beta_F). Default: 100.
    pub bf: f64,
    /// Reverse current gain (beta_R). Default: 1.
    pub br: f64,
    /// Forward emission coefficient. Default: 1.0.
    pub nf: f64,
    /// Reverse emission coefficient. Default: 1.0.
    pub nr: f64,
    /// Forward Early voltage (V). Default: infinity (no Early effect).
    pub vaf: f64,
    /// Reverse Early voltage (V). Default: infinity.
    pub var: f64,
    /// Base resistance (ohms). Default: 0.
    pub rb: f64,
    /// Emitter resistance (ohms). Default: 0.
    pub re: f64,
    /// Collector resistance (ohms). Default: 0.
    pub rc: f64,
    /// Base-emitter junction capacitance (F). Default: 0.
    pub cje: f64,
    /// Base-collector junction capacitance (F). Default: 0.
    pub cjc: f64,
    /// Forward transit time (s). Default: 0.
    pub tf: f64,
    /// Reverse transit time (s). Default: 0.
    pub tr: f64,
}

impl BjtParams {
    /// Create default NPN parameters.
    pub fn npn_default() -> Self {
        Self {
            is: 1e-16,
            bf: 100.0,
            br: 1.0,
            nf: 1.0,
            nr: 1.0,
            vaf: f64::INFINITY,
            var: f64::INFINITY,
            rb: 0.0,
            re: 0.0,
            rc: 0.0,
            cje: 0.0,
            cjc: 0.0,
            tf: 0.0,
            tr: 0.0,
        }
    }

    /// Create default PNP parameters.
    pub fn pnp_default() -> Self {
        Self::npn_default() // Same defaults, polarity handled in evaluation
    }
}

impl Default for BjtParams {
    fn default() -> Self {
        Self::npn_default()
    }
}

/// Operating region of the BJT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BjtRegion {
    /// Cutoff: both junctions reverse biased.
    Cutoff,
    /// Forward active: BE forward, BC reverse.
    ForwardActive,
    /// Reverse active: BE reverse, BC forward.
    ReverseActive,
    /// Saturation: both junctions forward biased.
    Saturation,
}

/// A BJT element.
#[derive(Debug, Clone)]
pub struct Bjt {
    /// Device name (e.g., "Q1").
    pub name: String,
    /// Collector node.
    pub node_collector: NodeId,
    /// Base node.
    pub node_base: NodeId,
    /// Emitter node.
    pub node_emitter: NodeId,
    /// BJT type.
    pub bjt_type: BjtType,
    /// Model parameters.
    pub params: BjtParams,
}

impl Bjt {
    /// Create a new NPN transistor with default parameters.
    pub fn npn(
        name: impl Into<String>,
        collector: NodeId,
        base: NodeId,
        emitter: NodeId,
    ) -> Self {
        Self {
            name: name.into(),
            node_collector: collector,
            node_base: base,
            node_emitter: emitter,
            bjt_type: BjtType::Npn,
            params: BjtParams::npn_default(),
        }
    }

    /// Create a new PNP transistor with default parameters.
    pub fn pnp(
        name: impl Into<String>,
        collector: NodeId,
        base: NodeId,
        emitter: NodeId,
    ) -> Self {
        Self {
            name: name.into(),
            node_collector: collector,
            node_base: base,
            node_emitter: emitter,
            bjt_type: BjtType::Pnp,
            params: BjtParams::pnp_default(),
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        name: impl Into<String>,
        collector: NodeId,
        base: NodeId,
        emitter: NodeId,
        bjt_type: BjtType,
        params: BjtParams,
    ) -> Self {
        Self {
            name: name.into(),
            node_collector: collector,
            node_base: base,
            node_emitter: emitter,
            bjt_type,
            params,
        }
    }

    /// Evaluate BJT currents and small-signal parameters using Ebers-Moll model.
    ///
    /// For NPN:
    /// - Forward current: If = Is * (exp(Vbe/(Nf*Vt)) - 1)
    /// - Reverse current: Ir = Is * (exp(Vbc/(Nr*Vt)) - 1)
    /// - Ic = If - Ir/Br (with Early effect: multiply by (1 + Vce/Vaf))
    /// - Ie = If/Bf - Ir
    /// - Ib = If/Bf + Ir/Br
    ///
    /// Returns (ic, ib, ie, gm, gpi, go, region) where:
    /// - ic, ib, ie: terminal currents
    /// - gm: transconductance ∂Ic/∂Vbe
    /// - gpi: input conductance ∂Ib/∂Vbe = gm/β
    /// - go: output conductance ∂Ic/∂Vce
    pub fn evaluate(&self, vbe: f64, vce: f64) -> (f64, f64, f64, f64, f64, f64, BjtRegion) {
        // For PNP, flip polarities
        let (vbe, vce, sign) = match self.bjt_type {
            BjtType::Npn => (vbe, vce, 1.0),
            BjtType::Pnp => (-vbe, -vce, -1.0),
        };

        let vbc = vbe - vce;

        let vt = thermal_voltage(300.15); // Room temperature
        let nf_vt = self.params.nf * vt;
        let nr_vt = self.params.nr * vt;

        // Apply voltage limiting to prevent exp() overflow
        let vbe_limited = limit_voltage(vbe, nf_vt);
        let vbc_limited = limit_voltage(vbc, nr_vt);

        // Forward and reverse currents (Ebers-Moll)
        let exp_be = (vbe_limited / nf_vt).exp();
        let exp_bc = (vbc_limited / nr_vt).exp();

        let if_current = self.params.is * (exp_be - 1.0);
        let ir_current = self.params.is * (exp_bc - 1.0);

        // Early effect factor
        let early_factor = if self.params.vaf.is_finite() && vce > 0.0 {
            1.0 + vce / self.params.vaf
        } else {
            1.0
        };

        // Terminal currents
        let ic = (if_current - ir_current / self.params.br) * early_factor;
        let ie = if_current / self.params.bf + ir_current;
        let ib = if_current / self.params.bf + ir_current / self.params.br;

        // Small-signal parameters
        // gm = ∂Ic/∂Vbe ≈ Ic/Vt (in forward active)
        let gm = if if_current > 1e-20 {
            (self.params.is * exp_be / nf_vt) * early_factor
        } else {
            1e-12
        };

        // gpi = gm / BF
        let gpi = gm / self.params.bf;

        // go = ∂Ic/∂Vce includes:
        // 1. Early effect: (if - ir/βr) * (1/Vaf)
        // 2. Reverse current sensitivity: ∂ir/∂Vbc * ∂Vbc/∂Vce = gbc * (-1)
        //    where gbc = is * exp(vbc/nvt) / nvt
        // In saturation, the reverse path is active, so we need term 2
        let go_early = if self.params.vaf.is_finite() && ic.abs() > 1e-20 {
            ic.abs() / self.params.vaf
        } else {
            0.0
        };
        // Reverse current derivative: ∂ir/∂vce = -is*exp(vbc/nvt)/nvt
        // Its contribution to ic: +(ir/βr contribution) since vbc = vbe - vce
        let gbc = self.params.is * exp_bc / nr_vt;
        let go_reverse = gbc / self.params.br * early_factor;
        let go = (go_early + go_reverse).max(1e-12);

        // Determine operating region
        let region = if vbe < 0.5 && vbc < 0.5 {
            BjtRegion::Cutoff
        } else if vbe > 0.5 && vbc < 0.5 {
            BjtRegion::ForwardActive
        } else if vbe < 0.5 && vbc > 0.5 {
            BjtRegion::ReverseActive
        } else {
            BjtRegion::Saturation
        };

        (
            sign * ic,
            sign * ib,
            sign * ie,
            gm.max(1e-12),
            gpi.max(1e-12),
            go.max(1e-12),
            region,
        )
    }

    /// Stamp the linearized BJT model into the MNA system.
    ///
    /// The BJT is modeled using the hybrid-π equivalent circuit:
    /// - gpi: base-emitter conductance (input)
    /// - gm: voltage-controlled current source (Vbe controls Ic)
    /// - go: collector-emitter conductance (output)
    /// - Ieq terms for Newton-Raphson
    pub fn stamp_linearized_at(&self, mna: &mut MnaSystem, vbe: f64, vce: f64) {
        // Apply voltage limiting consistently with evaluate() to ensure
        // the equivalent current sources match the device currents.
        let vt = crate::diode::thermal_voltage(300.15);
        let nf_vt = self.params.nf * vt;

        // For PNP, flip polarities (same as in evaluate)
        let vbe_internal = match self.bjt_type {
            BjtType::Npn => vbe,
            BjtType::Pnp => -vbe,
        };

        // Limit the BE voltage to prevent overflow
        let vbe_limited = limit_voltage(vbe_internal, nf_vt);

        // Convert back to external polarity for ieq calculation
        let vbe_for_ieq = match self.bjt_type {
            BjtType::Npn => vbe_limited,
            BjtType::Pnp => -vbe_limited,
        };

        let (ic, ib, _ie, gm, gpi, go, _region) = self.evaluate(vbe, vce);

        let c = node_to_index(self.node_collector);
        let b = node_to_index(self.node_base);
        let e = node_to_index(self.node_emitter);

        // Stamp gpi (base-emitter conductance representing base current)
        mna.stamp_conductance(b, e, gpi);

        // Stamp go (collector-emitter output conductance)
        mna.stamp_conductance(c, e, go);

        // Stamp gm (transconductance) as VCCS: I = gm * Vbe flowing from collector to emitter
        // Collector row: +gm*Vb - gm*Ve
        // Emitter row: -gm*Vb + gm*Ve
        if let Some(ci) = c {
            if let Some(bi) = b {
                mna.add_element(ci, bi, gm);
            }
            if let Some(ei) = e {
                mna.add_element(ci, ei, -gm);
            }
        }
        if let Some(ei) = e {
            if let Some(bi) = b {
                mna.add_element(ei, bi, -gm);
            }
            mna.add_element(ei, ei, gm);
        }

        // Equivalent current sources for linearization
        // Use limited vbe for consistency with evaluate()
        // Collector current: ic = gm*Vbe + go*Vce + ic_eq
        // ic_eq = Ic - gm*Vbe - go*Vce
        let ic_eq = ic - gm * vbe_for_ieq - go * vce;
        mna.stamp_current_source(c, e, ic_eq);

        // Base current: ib = gpi*Vbe + ib_eq
        // ib_eq = Ib - gpi*Vbe
        let ib_eq = ib - gpi * vbe_for_ieq;
        mna.stamp_current_source(b, e, ib_eq);
    }
}

/// Voltage limiting to prevent numerical overflow.
fn limit_voltage(v: f64, nvt: f64) -> f64 {
    let vcrit = nvt * (nvt / (std::f64::consts::SQRT_2 * 1e-14)).ln();

    if v > vcrit {
        let arg = (v - vcrit) / nvt;
        vcrit + nvt * (1.0 + arg.ln_1p())
    } else {
        v
    }
}

fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

impl Stamp for Bjt {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Initial stamp: Gmin shunts for numerical stability
        let c = node_to_index(self.node_collector);
        let b = node_to_index(self.node_base);
        let e = node_to_index(self.node_emitter);

        mna.stamp_conductance(b, e, 1e-12);
        mna.stamp_conductance(c, e, 1e-12);
    }
}

impl Element for Bjt {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_collector, self.node_base, self.node_emitter]
    }
}

impl Stamper for Bjt {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn is_nonlinear(&self) -> bool {
        true
    }

    fn stamp_nonlinear(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        let vc = node_to_index(self.node_collector)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vb = node_to_index(self.node_base)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let ve = node_to_index(self.node_emitter)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vbe = vb - ve;
        let vce = vc - ve;

        self.stamp_linearized_at(mna, vbe, vce);
    }

    fn ac_info_at(&self, solution: &DVector<f64>) -> AcDeviceInfo {
        let vc = node_to_index(self.node_collector)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vb = node_to_index(self.node_base)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let ve = node_to_index(self.node_emitter)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vbe = vb - ve;
        let vce = vc - ve;

        let (_ic, _ib, _ie, gm, gpi, go, _region) = self.evaluate(vbe, vce);

        AcDeviceInfo::Bjt {
            collector: node_to_index(self.node_collector),
            base: node_to_index(self.node_base),
            emitter: node_to_index(self.node_emitter),
            gm,
            gpi,
            go,
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npn_cutoff() {
        let q = Bjt::npn("Q1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Both junctions reverse biased: Vbe = -0.5V, Vce = 5V
        let (ic, ib, _ie, _gm, _gpi, _go, region) = q.evaluate(-0.5, 5.0);

        assert_eq!(region, BjtRegion::Cutoff);
        assert!(ic.abs() < 1e-12, "Ic should be ~0 in cutoff: {}", ic);
        assert!(ib.abs() < 1e-12, "Ib should be ~0 in cutoff: {}", ib);
    }

    #[test]
    fn test_npn_forward_active() {
        let q = Bjt::npn("Q1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Forward active: Vbe = 0.7V, Vce = 5V
        let (ic, ib, ie, gm, gpi, _go, region) = q.evaluate(0.7, 5.0);

        assert_eq!(region, BjtRegion::ForwardActive);
        assert!(ic > 0.0, "Ic should be positive: {}", ic);
        assert!(ib > 0.0, "Ib should be positive: {}", ib);
        assert!(ie > 0.0, "Ie should be positive: {}", ie);

        // Beta relationship: Ic ≈ β * Ib
        let beta_measured = ic / ib;
        assert!(
            (beta_measured - 100.0).abs() < 10.0,
            "β should be ~100: {}",
            beta_measured
        );

        // gm ≈ Ic / Vt
        let vt = thermal_voltage(300.15);
        let expected_gm = ic / vt;
        assert!(
            (gm - expected_gm).abs() / expected_gm < 0.1,
            "gm = {} (expected ~{})",
            gm,
            expected_gm
        );

        // gpi = gm / β
        let expected_gpi = gm / 100.0;
        assert!(
            (gpi - expected_gpi).abs() / expected_gpi < 0.1,
            "gpi = {} (expected ~{})",
            gpi,
            expected_gpi
        );
    }

    #[test]
    fn test_npn_saturation() {
        let q = Bjt::npn("Q1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Saturation: both junctions forward biased
        // Vbe = 0.7V, Vce = 0.1V → Vbc = 0.7 - 0.1 = 0.6V > 0.5
        let (_ic, _ib, _ie, _gm, _gpi, _go, region) = q.evaluate(0.7, 0.1);

        assert_eq!(region, BjtRegion::Saturation);
    }

    #[test]
    fn test_pnp_polarity() {
        let q = Bjt::pnp("Q1", NodeId::new(1), NodeId::new(2), NodeId::new(3));

        // For PNP: Veb = 0.7V means Vbe = -0.7V
        // Vce = -5V (collector more negative than emitter)
        let (ic, ib, _ie, _gm, _gpi, _go, region) = q.evaluate(-0.7, -5.0);

        assert_eq!(region, BjtRegion::ForwardActive);
        // PNP currents flow in opposite direction
        assert!(ic < 0.0, "PNP Ic should be negative: {}", ic);
        assert!(ib < 0.0, "PNP Ib should be negative: {}", ib);
    }

    #[test]
    fn test_voltage_limiting() {
        let q = Bjt::npn("Q1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Very large Vbe should not cause overflow
        let (ic, _ib, _ie, _gm, _gpi, _go, _region) = q.evaluate(100.0, 5.0);

        assert!(ic.is_finite(), "Ic should be finite even with large Vbe");
        assert!(ic > 0.0, "Ic should still be positive");
    }

    #[test]
    fn test_early_effect() {
        let mut params = BjtParams::npn_default();
        params.vaf = 100.0; // 100V Early voltage

        let q = Bjt::with_params(
            "Q1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            BjtType::Npn,
            params,
        );

        // Measure Ic at two different Vce values
        let (ic1, _, _, _, _, go1, _) = q.evaluate(0.7, 5.0);
        let (ic2, _, _, _, _, _, _) = q.evaluate(0.7, 10.0);

        // Ic should increase with Vce due to Early effect
        assert!(ic2 > ic1, "Ic should increase with Vce due to Early effect");

        // go = Ic / Vaf
        let expected_go = ic1 / 100.0;
        assert!(
            (go1 - expected_go).abs() / expected_go < 0.2,
            "go = {} (expected ~{})",
            go1,
            expected_go
        );
    }

    #[test]
    fn test_ac_info_at() {
        let q = Bjt::npn("Q1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // DC solution: Vc=5V, Vb=0.7V, Ve=0V
        let solution = DVector::from_vec(vec![5.0, 0.7]);

        let ac_info = q.ac_info_at(&solution);

        match ac_info {
            AcDeviceInfo::Bjt {
                collector,
                base,
                emitter,
                gm,
                gpi,
                go,
            } => {
                assert_eq!(collector, Some(0));
                assert_eq!(base, Some(1));
                assert_eq!(emitter, None);
                assert!(gm > 0.0, "gm should be positive: {}", gm);
                assert!(gpi > 0.0, "gpi should be positive: {}", gpi);
                assert!(go > 0.0, "go should be positive: {}", go);
            }
            _ => panic!("Expected AcDeviceInfo::Bjt"),
        }
    }

    #[test]
    fn test_bjt_stamping() {
        let q = Bjt::npn("Q1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        let mut mna = MnaSystem::new(2, 0);
        q.stamp_linearized_at(&mut mna, 0.7, 5.0);

        let matrix = mna.to_dense_matrix();

        // Matrix should have conductance contributions
        // G[1,1] should have gpi contribution (base-emitter)
        assert!(matrix[(1, 1)] > 0.0, "G[1,1] should have gpi");

        // G[0,0] should have go contribution (collector-emitter)
        assert!(matrix[(0, 0)] > 0.0, "G[0,0] should have go");

        // G[0,1] should have gm contribution
        assert!(matrix[(0, 1)] > 0.0, "G[0,1] should have gm");
    }
}
