//! MOSFET Level 1 (Shichman-Hodges) device model.
//!
//! This is a simple square-law model suitable for hand calculations
//! and basic circuit analysis. For short-channel effects, use BSIM3.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// MOSFET type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Number of fingers. Default: 1.0.
    pub nf: f64,
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
            nf: 1.0,
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
            nf: 1.0,
        }
    }

    /// Effective transconductance: beta = kp * (nf * W) / L.
    pub fn beta(&self) -> f64 {
        self.kp * self.nf * self.w / self.l
    }
}

/// Operating region of the MOSFET.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    ///
    /// For PMOS, current flows from source to drain (opposite to NMOS).
    /// The linearization is done in terms of the actual terminal voltages.
    pub fn stamp_linearized_at(&self, mna: &mut MnaSystem, vgs: f64, vds: f64) {
        let (ids, gds, gm, _region) = self.evaluate(vgs, vds);

        let d = node_to_index(self.node_drain);
        let g = node_to_index(self.node_gate);
        let s = node_to_index(self.node_source);

        // Stamp gds (drain-source conductance)
        mna.stamp_conductance(d, s, gds);

        // Stamp gm (transconductance) as VCCS: I = gm * Vgs flowing from drain to source
        //
        // The linearized model: ids = gds*Vds + gm*Vgs + Ieq
        // For MNA (G*V = b), current ids leaving drain means:
        //   Drain row: coefficients give +ids contribution
        //   Source row: coefficients give -ids contribution
        //
        // The gm*Vgs term = gm*(Vg - Vs) contributes:
        //   Drain row: +gm*Vg - gm*Vs → G[d,g] = +gm, G[d,s] = -gm
        //   Source row: -gm*Vg + gm*Vs → G[s,g] = -gm, G[s,s] = +gm
        if let Some(di) = d {
            if let Some(gi) = g {
                mna.add_element(di, gi, gm); // +gm at (drain, gate)
            }
            if let Some(si) = s {
                mna.add_element(di, si, -gm); // -gm at (drain, source)
            }
        }
        if let Some(si) = s {
            if let Some(gi) = g {
                mna.add_element(si, gi, -gm); // -gm at (source, gate)
            }
            // Note: si2 is the same as si (both from s)
            mna.add_element(si, si, gm); // +gm at (source, source)
        }

        // Equivalent current source: Ieq = Ids - gds*Vds - gm*Vgs
        //
        // The drain node equation (current balance, currents leaving = 0):
        //   (1/Rd + gds)*V(d) + gm*(V(g) - V(s)) + ieq = 0
        // Rearranging: matrix_terms = -ieq, so RHS[d] = -ieq
        //
        // stamp_current_source(d, s, I) gives: RHS[d] += -I, RHS[s] += I
        // To get RHS[d] = -ieq, we need stamp_current_source(d, s, ieq)
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
        // Extract operating point voltages from DC solution
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

        // Get small-signal parameters at operating point
        let (_ids, gds, gm, _region) = self.evaluate(vgs, vds);

        AcDeviceInfo::Mosfet {
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

    #[test]
    fn test_ac_info_at_saturation() {
        // NMOS with drain=node1, gate=node2, source=ground
        let m = Mosfet::nmos("M1", NodeId::new(1), NodeId::new(2), NodeId::GROUND);

        // DC solution: Vd=5V, Vg=2V, Vs=0V → Vgs=2V, Vds=5V → saturation
        let solution = DVector::from_vec(vec![5.0, 2.0]);

        let ac_info = m.ac_info_at(&solution);

        match ac_info {
            AcDeviceInfo::Mosfet {
                drain,
                gate,
                source,
                gds,
                gm,
            } => {
                assert_eq!(drain, Some(0));
                assert_eq!(gate, Some(1));
                assert_eq!(source, None);
                // In saturation, gm should be significant
                assert!(gm > 1e-6, "gm should be positive in saturation: {}", gm);
                // gds may be small (lambda=0 by default)
                assert!(gds >= 0.0, "gds should be non-negative: {}", gds);
            }
            _ => panic!("Expected AcDeviceInfo::Mosfet"),
        }
    }

    #[test]
    fn test_pmos_stamping() {
        // PMOS circuit: drain=1, gate=2, source=3
        // V(source) = 5V, V(gate) = 3V, V(drain) = 0V initially
        let mut params = MosfetParams::pmos_default();
        params.vto = -0.7;
        params.kp = 50e-6;
        params.w = 10e-6;
        params.l = 1e-6;
        params.lambda = 0.0;

        let pmos = Mosfet::with_params(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            MosfetType::Pmos,
            params,
        );

        // Operating point: V(gate)=3V, V(source)=5V
        let vgs = 3.0 - 5.0; // -2V
        let vds = 0.0 - 5.0; // -5V

        let (ids, gds, gm, region) = pmos.evaluate(vgs, vds);

        // Expected: PMOS in saturation with ids = -422.5µA, gm = 650µS
        let beta = 50e-6 * 10.0;
        let expected_ids = -0.5 * beta * 1.3 * 1.3; // -422.5µA
        let expected_gm = beta * 1.3; // 650µS

        assert_eq!(region, MosfetRegion::Saturation);
        assert!(
            (ids - expected_ids).abs() < 1e-10,
            "ids={} expected={}",
            ids,
            expected_ids
        );
        assert!(
            (gm - expected_gm).abs() < 1e-10,
            "gm={} expected={}",
            gm,
            expected_gm
        );

        // Test MNA stamping
        let mut mna = MnaSystem::new(3, 0);
        pmos.stamp_linearized_at(&mut mna, vgs, vds);

        let ieq = ids - gds * vds - gm * vgs;
        let matrix = mna.to_dense_matrix();
        let rhs = mna.rhs();

        // Verify the matrix stamps match expected VCCS + ieq pattern
        let eps = 1e-15;
        assert!((matrix[(0, 0)] - gds).abs() < eps, "G[0,0] wrong");
        assert!((matrix[(0, 1)] - gm).abs() < eps, "G[0,1] wrong");
        assert!((matrix[(0, 2)] - (-gds - gm)).abs() < eps, "G[0,2] wrong");
        assert!((matrix[(2, 0)] - (-gds)).abs() < eps, "G[2,0] wrong");
        assert!((matrix[(2, 1)] - (-gm)).abs() < eps, "G[2,1] wrong");
        assert!((matrix[(2, 2)] - (gds + gm)).abs() < eps, "G[2,2] wrong");
        assert!((rhs[0] - (-ieq)).abs() < eps, "RHS[0] wrong");
        assert!((rhs[2] - ieq).abs() < eps, "RHS[2] wrong");
    }

    #[test]
    fn test_pmos_circuit_manual() {
        // Build the PMOS circuit manually and verify Newton-Raphson iteration
        // Circuit: Vdd=5V (node3), Vg=3V (node2), Rd=1k (node1 to gnd), PMOS M1

        let mut params = MosfetParams::pmos_default();
        params.vto = -0.7;
        params.kp = 50e-6;
        params.w = 10e-6;
        params.l = 1e-6;
        params.lambda = 0.0;

        let pmos = Mosfet::with_params(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            MosfetType::Pmos,
            params,
        );

        // Iteration 0: Start with zero solution
        let mut mna = MnaSystem::new(3, 2);
        mna.stamp_conductance(Some(0), None, 0.001); // Rd
        mna.stamp_voltage_source(Some(2), None, 0, 5.0); // Vdd
        mna.stamp_voltage_source(Some(1), None, 1, 3.0); // Vg
        pmos.stamp_linearized_at(&mut mna, 0.0, 0.0); // MOSFET at (0, 0) = cutoff

        let matrix = mna.to_dense_matrix();
        let solution_1 = matrix
            .clone()
            .lu()
            .solve(mna.rhs())
            .expect("LU solve failed");

        // After iteration 0: V(1)≈0, V(2)=3, V(3)=5
        assert!((solution_1[1] - 3.0).abs() < 1e-9, "V(2) should be 3V");
        assert!((solution_1[2] - 5.0).abs() < 1e-9, "V(3) should be 5V");

        // Iteration 1: MOSFET sees vgs=-2, vds≈-5 → saturation
        mna.clear();
        mna.stamp_conductance(Some(0), None, 0.001);
        mna.stamp_voltage_source(Some(2), None, 0, 5.0);
        mna.stamp_voltage_source(Some(1), None, 1, 3.0);

        let vgs_1 = solution_1[1] - solution_1[2]; // 3 - 5 = -2
        let vds_1 = solution_1[0] - solution_1[2]; // ~0 - 5 ≈ -5
        pmos.stamp_linearized_at(&mut mna, vgs_1, vds_1);

        let matrix = mna.to_dense_matrix();
        let solution_2 = matrix
            .clone()
            .lu()
            .solve(mna.rhs())
            .expect("LU solve failed");

        // Expected: V(1) ≈ 0.42V for PMOS in saturation
        assert!(
            solution_2[0] > 0.3 && solution_2[0] < 0.6,
            "V(1) = {} (expected ~0.42V)",
            solution_2[0]
        );
    }

    #[test]
    fn test_pmos_stamp_nonlinear() {
        // Test that stamp_nonlinear correctly extracts voltages from solution
        let mut params = MosfetParams::pmos_default();
        params.vto = -0.7;
        params.kp = 50e-6;
        params.w = 10e-6;
        params.l = 1e-6;
        params.lambda = 0.0;

        let pmos = Mosfet::with_params(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            MosfetType::Pmos,
            params,
        );

        // Test with solution vector: V(1)=0, V(2)=3, V(3)=5
        let solution = DVector::from_vec(vec![0.0, 3.0, 5.0]);

        let mut mna = MnaSystem::new(3, 0);
        pmos.stamp_nonlinear(&mut mna, &solution);

        // For PMOS with vgs=-2, vds=-5 (saturation):
        // ids = -422.5uA, gds = 1e-12, gm = 650uS
        // ieq = -422.5uA - 0 + 1300uA = 877.5uA

        let gm = 650e-6_f64;
        let gds = 1e-12_f64;
        let ieq = 877.5e-6_f64;
        let matrix = mna.to_dense_matrix();

        let eps = 1e-10;
        assert!((matrix[(0, 0)] - gds).abs() < eps, "G[0,0] wrong");
        assert!((matrix[(0, 1)] - gm).abs() < eps, "G[0,1] wrong");
        assert!((matrix[(0, 2)] - (-gds - gm)).abs() < eps, "G[0,2] wrong");
        assert!((mna.rhs()[0] - (-ieq)).abs() < eps, "RHS[0] wrong");
    }
}
