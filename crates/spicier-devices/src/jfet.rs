//! JFET (Junction Field Effect Transistor) device model using Shichman-Hodges equations.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// JFET type (N-channel or P-channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JfetType {
    /// N-channel JFET (NJF).
    Njf,
    /// P-channel JFET (PJF).
    Pjf,
}

/// JFET model parameters (Shichman-Hodges model).
#[derive(Debug, Clone)]
pub struct JfetParams {
    /// Threshold voltage (V). Default: -2.0 (NJF), +2.0 (PJF).
    pub vto: f64,
    /// Transconductance parameter (A/V^2). Default: 1e-4.
    pub beta: f64,
    /// Channel-length modulation (1/V). Default: 0.0.
    pub lambda: f64,
    /// Gate saturation current (A). Default: 1e-14.
    pub is: f64,
    /// Gate-source capacitance (F). Default: 0.0.
    pub cgs: f64,
    /// Gate-drain capacitance (F). Default: 0.0.
    pub cgd: f64,
}

impl JfetParams {
    /// Create default N-channel JFET parameters.
    pub fn njf_default() -> Self {
        Self {
            vto: -2.0,
            beta: 1e-4,
            lambda: 0.0,
            is: 1e-14,
            cgs: 0.0,
            cgd: 0.0,
        }
    }

    /// Create default P-channel JFET parameters.
    pub fn pjf_default() -> Self {
        Self {
            vto: 2.0,
            beta: 1e-4,
            lambda: 0.0,
            is: 1e-14,
            cgs: 0.0,
            cgd: 0.0,
        }
    }
}

impl Default for JfetParams {
    fn default() -> Self {
        Self::njf_default()
    }
}

/// Operating region of the JFET.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JfetRegion {
    /// Cutoff: channel is pinched off, no current flows.
    Cutoff,
    /// Linear (triode): channel conducts, current depends on Vds.
    Linear,
    /// Saturation: channel pinched at drain, current independent of Vds.
    Saturation,
}

/// A JFET element.
#[derive(Debug, Clone)]
pub struct Jfet {
    /// Device name (e.g., "J1").
    pub name: String,
    /// Drain node.
    pub node_drain: NodeId,
    /// Gate node.
    pub node_gate: NodeId,
    /// Source node.
    pub node_source: NodeId,
    /// JFET type.
    pub jfet_type: JfetType,
    /// Model parameters.
    pub params: JfetParams,
}

impl Jfet {
    /// Create a new N-channel JFET with default parameters.
    pub fn njf(name: impl Into<String>, drain: NodeId, gate: NodeId, source: NodeId) -> Self {
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            jfet_type: JfetType::Njf,
            params: JfetParams::njf_default(),
        }
    }

    /// Create a new P-channel JFET with default parameters.
    pub fn pjf(name: impl Into<String>, drain: NodeId, gate: NodeId, source: NodeId) -> Self {
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            jfet_type: JfetType::Pjf,
            params: JfetParams::pjf_default(),
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        jfet_type: JfetType,
        params: JfetParams,
    ) -> Self {
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            jfet_type,
            params,
        }
    }

    /// Evaluate drain current and partial derivatives using Shichman-Hodges model.
    ///
    /// For N-channel JFET:
    /// - Cutoff:     Vgs < Vto → Ids = 0
    /// - Linear:     Vds < Vgs - Vto → Ids = β(2(Vgs-Vto)Vds - Vds²)(1 + λVds)
    /// - Saturation: Vds ≥ Vgs - Vto → Ids = β(Vgs-Vto)²(1 + λVds)
    ///
    /// Note: Vto is negative for NJF, so Vgs - Vto is positive when Vgs > Vto.
    ///
    /// Returns (ids, gds, gm, region) where:
    /// - ids: drain-source current
    /// - gds: ∂Ids/∂Vds (output conductance)
    /// - gm:  ∂Ids/∂Vgs (transconductance)
    pub fn evaluate(&self, vgs: f64, vds: f64) -> (f64, f64, f64, JfetRegion) {
        // For PJF, flip signs to use same equations as NJF
        let (vgs, vds, sign) = match self.jfet_type {
            JfetType::Njf => (vgs, vds, 1.0),
            JfetType::Pjf => (-vgs, -vds, -1.0),
        };

        let vto = self.params.vto.abs(); // Use magnitude for calculations
        let beta = self.params.beta;
        let lambda = self.params.lambda;

        // For NJF: Vto is negative, so conducting when Vgs > Vto (i.e., Vgs - Vto > 0)
        // After sign adjustment, Vto is positive, and we check Vgs > -Vto for conduction
        // Actually, let's use the standard formulation where Vp = |Vto| is the pinch-off voltage
        let vp = vto; // Pinch-off voltage (positive)

        // Cutoff: Vgs <= -Vp (for NJF, this means gate is reverse-biased beyond pinch-off)
        if vgs <= -vp {
            return (0.0, 1e-12, 0.0, JfetRegion::Cutoff);
        }

        // Overdrive voltage
        let vov = vgs + vp; // This is (Vgs - Vto) for NJF where Vto = -Vp

        // Handle negative Vds (source-drain swap behavior)
        let vds = vds.max(0.0);

        if vds < vov {
            // Linear region: Ids = β(2*Vov*Vds - Vds²)(1 + λVds)
            let ids_base = beta * (2.0 * vov * vds - vds * vds);
            let ids = ids_base * (1.0 + lambda * vds);

            // ∂Ids/∂Vds = β(2*Vov - 2*Vds)(1 + λVds) + β(2*Vov*Vds - Vds²)λ
            let gds = beta * (2.0 * vov - 2.0 * vds) * (1.0 + lambda * vds)
                + ids_base * lambda;

            // ∂Ids/∂Vgs = β * 2 * Vds * (1 + λVds)
            let gm = beta * 2.0 * vds * (1.0 + lambda * vds);

            (sign * ids, gds.max(1e-12), gm, JfetRegion::Linear)
        } else {
            // Saturation region: Ids = β * Vov² * (1 + λVds)
            let ids_base = beta * vov * vov;
            let ids = ids_base * (1.0 + lambda * vds);

            // ∂Ids/∂Vds = β * Vov² * λ
            let gds = ids_base * lambda;

            // ∂Ids/∂Vgs = β * 2 * Vov * (1 + λVds)
            let gm = beta * 2.0 * vov * (1.0 + lambda * vds);

            (sign * ids, gds.max(1e-12), gm, JfetRegion::Saturation)
        }
    }

    /// Stamp the linearized JFET model into the MNA system.
    ///
    /// The JFET is linearized as:
    /// - gds conductance between drain and source
    /// - gm * Vgs voltage-controlled current source from drain to source
    /// - Ieq = Ids - gds*Vds - gm*Vgs (companion current source)
    pub fn stamp_linearized_at(&self, mna: &mut MnaSystem, vgs: f64, vds: f64) {
        let (ids, gds, gm, _region) = self.evaluate(vgs, vds);

        let d = node_to_index(self.node_drain);
        let g = node_to_index(self.node_gate);
        let s = node_to_index(self.node_source);

        // Stamp gds (drain-source conductance)
        mna.stamp_conductance(d, s, gds);

        // Stamp gm (transconductance) as VCCS: I = gm * Vgs flowing from drain to source
        if let Some(di) = d {
            if let Some(gi) = g {
                mna.add_element(di, gi, gm);
            }
            if let Some(si) = s {
                mna.add_element(di, si, -gm);
            }
        }
        if let Some(si) = s {
            if let Some(gi) = g {
                mna.add_element(si, gi, -gm);
            }
            mna.add_element(si, si, gm);
        }

        // Equivalent current source: Ieq = Ids - gds*Vds - gm*Vgs
        let ieq = ids - gds * vds - gm * vgs;
        mna.stamp_current_source(d, s, ieq);
    }
}

fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

impl Stamp for Jfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Initial stamp: Gmin shunt between drain and source for numerical stability
        let d = node_to_index(self.node_drain);
        let s = node_to_index(self.node_source);
        mna.stamp_conductance(d, s, 1e-12);
    }
}

impl Element for Jfet {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_drain, self.node_gate, self.node_source]
    }
}

impl Stamper for Jfet {
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
        let vg = node_to_index(self.node_gate)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vd = node_to_index(self.node_drain)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vs = node_to_index(self.node_source)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vgs = vg - vs;
        let vds = vd - vs;

        self.stamp_linearized_at(mna, vgs, vds);
    }

    fn ac_info_at(&self, solution: &DVector<f64>) -> AcDeviceInfo {
        let vg = node_to_index(self.node_gate)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vd = node_to_index(self.node_drain)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vs = node_to_index(self.node_source)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vgs = vg - vs;
        let vds = vd - vs;

        let (_ids, gds, gm, _region) = self.evaluate(vgs, vds);

        AcDeviceInfo::Jfet {
            drain: node_to_index(self.node_drain),
            gate: node_to_index(self.node_gate),
            source: node_to_index(self.node_source),
            gds,
            gm,
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
    fn test_njf_cutoff() {
        let j = Jfet::njf("J1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Vgs = -3V < Vto = -2V → cutoff (gate more negative than pinch-off)
        let (ids, _gds, gm, region) = j.evaluate(-3.0, 5.0);

        assert_eq!(region, JfetRegion::Cutoff);
        assert_eq!(ids, 0.0);
        assert_eq!(gm, 0.0);
    }

    #[test]
    fn test_njf_saturation() {
        let j = Jfet::njf("J1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Vgs = 0V, Vds = 5V
        // Vov = Vgs + Vp = 0 + 2 = 2V
        // Vds = 5V > Vov = 2V → saturation
        let (ids, gds, gm, region) = j.evaluate(0.0, 5.0);

        assert_eq!(region, JfetRegion::Saturation);
        assert!(ids > 0.0, "Ids should be positive in saturation: {}", ids);
        assert!(gm > 0.0, "gm should be positive: {}", gm);

        // Ids = β * Vov² = 1e-4 * 2² = 4e-4 A = 0.4 mA (with λ=0)
        let expected_ids = 1e-4 * 2.0 * 2.0;
        assert!(
            (ids - expected_ids).abs() < 1e-10,
            "Ids = {} (expected {})",
            ids,
            expected_ids
        );

        // gm = 2 * β * Vov = 2 * 1e-4 * 2 = 4e-4 S
        let expected_gm = 2.0 * 1e-4 * 2.0;
        assert!(
            (gm - expected_gm).abs() < 1e-10,
            "gm = {} (expected {})",
            gm,
            expected_gm
        );

        // gds should be small (λ=0)
        assert!(gds < 1e-10, "gds should be near zero with λ=0: {}", gds);
    }

    #[test]
    fn test_njf_linear() {
        let j = Jfet::njf("J1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Vgs = 0V, Vds = 1V
        // Vov = 0 + 2 = 2V
        // Vds = 1V < Vov = 2V → linear
        let (ids, gds, gm, region) = j.evaluate(0.0, 1.0);

        assert_eq!(region, JfetRegion::Linear);
        assert!(ids > 0.0, "Ids should be positive in linear: {}", ids);
        assert!(gds > 0.0, "gds should be positive: {}", gds);
        assert!(gm > 0.0, "gm should be positive: {}", gm);

        // Ids = β * (2*Vov*Vds - Vds²) = 1e-4 * (2*2*1 - 1) = 1e-4 * 3 = 3e-4 A
        let expected_ids = 1e-4 * (2.0 * 2.0 * 1.0 - 1.0 * 1.0);
        assert!(
            (ids - expected_ids).abs() < 1e-10,
            "Ids = {} (expected {})",
            ids,
            expected_ids
        );
    }

    #[test]
    fn test_pjf_polarity() {
        let j = Jfet::pjf("J1", NodeId::new(1), NodeId::new(2), NodeId::new(3));

        // For PJF: Vgs = 0V (with respect to source), Vds = -5V (drain more negative)
        // Internal: Vgs' = 0, Vds' = 5V → saturation
        let (ids, _gds, _gm, region) = j.evaluate(0.0, -5.0);

        assert_eq!(region, JfetRegion::Saturation);
        // PJF current flows opposite direction (source to drain)
        assert!(ids < 0.0, "PJF Ids should be negative: {}", ids);
    }

    #[test]
    fn test_jfet_with_lambda() {
        let mut params = JfetParams::njf_default();
        params.lambda = 0.01; // 1% channel-length modulation

        let j = Jfet::with_params(
            "J1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            JfetType::Njf,
            params,
        );

        // Saturation: Vgs = 0, Vds = 5V
        let (ids, gds, _gm, region) = j.evaluate(0.0, 5.0);

        assert_eq!(region, JfetRegion::Saturation);

        // Ids = β * Vov² * (1 + λ*Vds) = 1e-4 * 4 * 1.05 = 4.2e-4 A
        let expected_ids = 1e-4 * 4.0 * (1.0 + 0.01 * 5.0);
        assert!(
            (ids - expected_ids).abs() < 1e-10,
            "Ids = {} (expected {})",
            ids,
            expected_ids
        );

        // gds = β * Vov² * λ = 1e-4 * 4 * 0.01 = 4e-6 S
        let expected_gds = 1e-4 * 4.0 * 0.01;
        assert!(
            (gds - expected_gds).abs() < 1e-10,
            "gds = {} (expected {})",
            gds,
            expected_gds
        );
    }

    #[test]
    fn test_ac_info_at_saturation() {
        let j = Jfet::njf("J1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // DC solution: Vd=5V, Vg=0V, Vs=0V → Vgs=0V, Vds=5V → saturation
        let solution = DVector::from_vec(vec![5.0, 0.0]);

        let ac_info = j.ac_info_at(&solution);

        match ac_info {
            AcDeviceInfo::Jfet {
                drain,
                gate,
                source,
                gds,
                gm,
            } => {
                assert_eq!(drain, Some(0));
                assert_eq!(gate, Some(1));
                assert_eq!(source, None);
                assert!(gm > 1e-6, "gm should be positive in saturation: {}", gm);
                assert!(gds >= 0.0, "gds should be non-negative: {}", gds);
            }
            _ => panic!("Expected AcDeviceInfo::Jfet"),
        }
    }

    #[test]
    fn test_jfet_stamping() {
        let j = Jfet::njf("J1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Test at Vgs=0, Vds=5 (saturation)
        let mut mna = MnaSystem::new(2, 0);
        j.stamp_linearized_at(&mut mna, 0.0, 5.0);

        let matrix = mna.to_dense_matrix();
        let rhs = mna.rhs();

        // Verify gds stamp and gm VCCS pattern
        let (_ids, gds, gm, _) = j.evaluate(0.0, 5.0);

        // Matrix should have gds + gm contributions
        assert!(matrix[(0, 0)] > 0.0, "G[0,0] should have gds");
        assert!(
            (matrix[(0, 1)] - gm).abs() < 1e-15,
            "G[0,1] should be gm: {} vs {}",
            matrix[(0, 1)],
            gm
        );

        // RHS should have ieq contribution
        let ieq = 1e-4 * 4.0 - gds * 5.0 - gm * 0.0;
        assert!(
            (rhs[0] - (-ieq)).abs() < 1e-15,
            "RHS[0] should be -ieq: {} vs {}",
            rhs[0],
            -ieq
        );
    }
}
