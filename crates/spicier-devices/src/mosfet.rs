//! MOSFET Level 1 device model.

use spicier_core::mna::MnaSystem;
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// MOSFET type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosfetType {
    Nmos,
    Pmos,
}

/// MOSFET Level 1 model parameters.
#[derive(Debug, Clone)]
pub struct MosfetParams {
    /// Threshold voltage (V). Default: 0.7 (NMOS), -0.7 (PMOS).
    pub vto: f64,
    /// Transconductance parameter (A/V^2). Default: 2e-5.
    pub kp: f64,
    /// Channel-length modulation (1/V). Default: 0.0.
    pub lambda: f64,
    /// Oxide capacitance per unit area (F/m^2). Default: 0.0.
    pub cox: f64,
    /// Channel width (m). Default: 1e-6.
    pub w: f64,
    /// Channel length (m). Default: 1e-6.
    pub l: f64,
}

impl MosfetParams {
    /// Create default NMOS parameters.
    pub fn nmos_default() -> Self {
        Self {
            vto: 0.7,
            kp: 2e-5,
            lambda: 0.0,
            cox: 0.0,
            w: 10e-6,
            l: 1e-6,
        }
    }

    /// Create default PMOS parameters.
    pub fn pmos_default() -> Self {
        Self {
            vto: -0.7,
            kp: 1e-5,
            lambda: 0.0,
            cox: 0.0,
            w: 10e-6,
            l: 1e-6,
        }
    }

    /// Effective transconductance: beta = kp * W / L.
    pub fn beta(&self) -> f64 {
        self.kp * self.w / self.l
    }
}

/// Operating region of the MOSFET.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosfetRegion {
    Cutoff,
    Linear,
    Saturation,
}

/// A MOSFET element.
#[derive(Debug, Clone)]
pub struct Mosfet {
    /// Device name (e.g., "M1").
    pub name: String,
    /// Drain node.
    pub node_drain: NodeId,
    /// Gate node.
    pub node_gate: NodeId,
    /// Source node.
    pub node_source: NodeId,
    /// MOSFET type.
    pub mos_type: MosfetType,
    /// Model parameters.
    pub params: MosfetParams,
}

impl Mosfet {
    /// Create a new NMOS transistor.
    pub fn nmos(name: impl Into<String>, drain: NodeId, gate: NodeId, source: NodeId) -> Self {
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            mos_type: MosfetType::Nmos,
            params: MosfetParams::nmos_default(),
        }
    }

    /// Create a new PMOS transistor.
    pub fn pmos(name: impl Into<String>, drain: NodeId, gate: NodeId, source: NodeId) -> Self {
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            mos_type: MosfetType::Pmos,
            params: MosfetParams::pmos_default(),
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        mos_type: MosfetType,
        params: MosfetParams,
    ) -> Self {
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            mos_type,
            params,
        }
    }

    /// Evaluate drain current and partial derivatives.
    ///
    /// For NMOS:
    /// - Cutoff:     Vgs < Vth → Ids = 0
    /// - Linear:     Vgs >= Vth, Vds < Vgs - Vth → Ids = beta * ((Vgs-Vth)*Vds - Vds^2/2) * (1 + lambda*Vds)
    /// - Saturation: Vgs >= Vth, Vds >= Vgs - Vth → Ids = beta/2 * (Vgs-Vth)^2 * (1 + lambda*Vds)
    ///
    /// Returns (ids, gds, gm, region) where:
    /// - ids: drain-source current
    /// - gds: ∂Ids/∂Vds (output conductance)
    /// - gm:  ∂Ids/∂Vgs (transconductance)
    pub fn evaluate(&self, vgs: f64, vds: f64) -> (f64, f64, f64, MosfetRegion) {
        let (vgs, vds) = match self.mos_type {
            MosfetType::Nmos => (vgs, vds),
            MosfetType::Pmos => (-vgs, -vds),
        };

        let vth = self.params.vto.abs();
        let beta = self.params.beta();
        let lambda = self.params.lambda;

        if vgs < vth {
            // Cutoff
            (0.0, 1e-12, 0.0, MosfetRegion::Cutoff)
        } else if vds < vgs - vth {
            // Linear (triode)
            let vov = vgs - vth;
            let ids = beta * (vov * vds - 0.5 * vds * vds) * (1.0 + lambda * vds);
            let gds = beta * (vov - vds) * (1.0 + lambda * vds)
                + beta * (vov * vds - 0.5 * vds * vds) * lambda;
            let gm = beta * vds * (1.0 + lambda * vds);

            match self.mos_type {
                MosfetType::Nmos => (ids, gds.max(1e-12), gm, MosfetRegion::Linear),
                MosfetType::Pmos => (-ids, gds.max(1e-12), gm, MosfetRegion::Linear),
            }
        } else {
            // Saturation
            let vov = vgs - vth;
            let ids = 0.5 * beta * vov * vov * (1.0 + lambda * vds);
            let gds = 0.5 * beta * vov * vov * lambda;
            let gm = beta * vov * (1.0 + lambda * vds);

            match self.mos_type {
                MosfetType::Nmos => (ids, gds.max(1e-12), gm, MosfetRegion::Saturation),
                MosfetType::Pmos => (-ids, gds.max(1e-12), gm, MosfetRegion::Saturation),
            }
        }
    }

    /// Stamp the linearized MOSFET model into the MNA system.
    ///
    /// The MOSFET is linearized as:
    /// - gds conductance between drain and source
    /// - gm * Vgs voltage-controlled current source from drain to source
    /// - Ieq = Ids - gds*Vds - gm*Vgs (companion current source)
    pub fn stamp_nonlinear(&self, mna: &mut MnaSystem, vgs: f64, vds: f64) {
        let (ids, gds, gm, _region) = self.evaluate(vgs, vds);

        let d = node_to_index(self.node_drain);
        let g = node_to_index(self.node_gate);
        let s = node_to_index(self.node_source);

        // Stamp gds (drain-source conductance)
        mna.stamp_conductance(d, s, gds);

        // Stamp gm (transconductance) as VCCS: gm * Vgs
        // Current flows from drain to source, controlled by gate-source voltage
        if let Some(di) = d {
            if let Some(gi) = g {
                mna.matrix_mut()[(di, gi)] += gm;
            }
            if let Some(si) = s {
                mna.matrix_mut()[(di, si)] -= gm;
            }
        }
        if let Some(si) = s {
            if let Some(gi) = g {
                mna.matrix_mut()[(si, gi)] -= gm;
            }
            if let Some(si2) = s {
                mna.matrix_mut()[(si, si2)] += gm;
            }
        }

        // Equivalent current source: Ieq = Ids - gds*Vds - gm*Vgs
        let ieq = ids - gds * vds - gm * vgs;
        mna.stamp_current_source(d, s, -ieq);
    }
}

fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

impl Stamp for Mosfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Initial stamp: Gmin shunt between drain and source
        let d = node_to_index(self.node_drain);
        let s = node_to_index(self.node_source);
        mna.stamp_conductance(d, s, 1e-12);
    }
}

impl Element for Mosfet {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.node_drain, self.node_gate, self.node_source]
    }
}

impl Stamper for Mosfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmos_cutoff() {
        let m = Mosfet::nmos("M1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Vgs = 0.3V < Vth = 0.7V → cutoff
        let (ids, _gds, gm, region) = m.evaluate(0.3, 1.0);

        assert_eq!(region, MosfetRegion::Cutoff);
        assert_eq!(ids, 0.0);
        assert_eq!(gm, 0.0);
    }

    #[test]
    fn test_nmos_saturation() {
        let m = Mosfet::nmos("M1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Vgs = 2V, Vds = 5V → Vds > Vgs - Vth = 1.3V → saturation
        let (ids, _gds, gm, region) = m.evaluate(2.0, 5.0);

        assert_eq!(region, MosfetRegion::Saturation);
        assert!(ids > 0.0, "Ids should be positive in saturation: {}", ids);
        assert!(gm > 0.0, "gm should be positive: {}", gm);

        // Ids = beta/2 * (Vgs - Vth)^2 = (2e-5 * 10/1)/2 * 1.3^2 = 1e-4 * 1.69 = 1.69e-4
        let beta = 2e-5 * 10.0; // kp * W/L
        let expected_ids = 0.5 * beta * 1.3 * 1.3;
        assert!(
            (ids - expected_ids).abs() < 1e-10,
            "Ids = {} (expected {})",
            ids,
            expected_ids
        );
    }

    #[test]
    fn test_nmos_linear() {
        let m = Mosfet::nmos("M1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // Vgs = 2V, Vds = 0.5V → Vds < Vgs - Vth = 1.3V → linear
        let (ids, gds, gm, region) = m.evaluate(2.0, 0.5);

        assert_eq!(region, MosfetRegion::Linear);
        assert!(ids > 0.0, "Ids should be positive in linear: {}", ids);
        assert!(gds > 0.0, "gds should be positive: {}", gds);
        assert!(gm > 0.0, "gm should be positive: {}", gm);

        // Ids = beta * ((Vgs-Vth)*Vds - Vds^2/2) = 2e-4 * (1.3*0.5 - 0.125)
        let beta = 2e-5 * 10.0;
        let expected_ids = beta * (1.3 * 0.5 - 0.5 * 0.5 * 0.5);
        assert!(
            (ids - expected_ids).abs() < 1e-10,
            "Ids = {} (expected {})",
            ids,
            expected_ids
        );
    }

    #[test]
    fn test_pmos_saturation() {
        let m = Mosfet::pmos("M1", NodeId::new(1), NodeId::new(2), NodeId::new(3));

        // For PMOS: Vsg = 2V (Vgs = -2), Vsd = 5V (Vds = -5)
        let (ids, _gds, _gm, region) = m.evaluate(-2.0, -5.0);

        assert_eq!(region, MosfetRegion::Saturation);
        assert!(ids < 0.0, "PMOS Ids should be negative: {}", ids);
    }

    #[test]
    fn test_mosfet_params() {
        let params = MosfetParams::nmos_default();
        let beta = params.beta();

        // beta = kp * W/L = 2e-5 * 10e-6 / 1e-6 = 2e-4
        assert!(
            (beta - 2e-4).abs() < 1e-10,
            "beta = {} (expected 2e-4)",
            beta
        );
    }
}
