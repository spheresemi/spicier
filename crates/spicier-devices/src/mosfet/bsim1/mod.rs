//! BSIM1 (Level 4) MOSFET model.
//!
//! This module implements the Berkeley Short-channel IGFET Model version 1,
//! an empirical compact model for ~1 µm short-channel MOSFETs.
//!
//! # Features
//!
//! This BSIM1 implementation includes:
//! - Threshold voltage with body effect (K1, K2 parameters)
//! - DIBL (drain-induced barrier lowering) via ETA parameters
//! - Mobility interpolation between Vds=0 and Vds=Vdd
//! - Gate-field mobility degradation (U0, U0B)
//! - Velocity saturation (U1 parameter)
//! - Subthreshold current model (N0, NB, ND)
//! - L/W parameter scaling
//!
//! # Usage
//!
//! ```text
//! .MODEL NMOD NMOS LEVEL=4 TOX=25n MUZ=600 VFB0=-0.9 PHI0=0.6 K10=0.5
//! M1 d g s b NMOD W=10u L=1u
//! ```
//!
//! # Note on Parameters
//!
//! Unlike Level 1 which has meaningful defaults, BSIM1 parameters default to 0.
//! Users must provide extracted parameters from their process.
//!
//! # References
//!
//! - BSIM1 User's Manual, UC Berkeley
//! - B.J. Sheu et al., "BSIM: Berkeley Short-Channel IGFET Model for MOS Transistors",
//!   IEEE JSSC, vol. SC-22, no. 4, Aug. 1987

pub mod derived;
pub mod evaluate;
pub mod params;

pub use derived::Bsim1Derived;
pub use evaluate::{Bsim1EvalResult, Bsim1Region, evaluate as bsim1_evaluate};
pub use params::Bsim1Params;

use super::level1::MosfetType;
use crate::stamp::Stamp;

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

/// A BSIM1 (Level 4) MOSFET device.
#[derive(Debug, Clone)]
pub struct Bsim1Mosfet {
    /// Device name (e.g., "M1").
    pub name: String,
    /// Drain node.
    pub node_drain: NodeId,
    /// Gate node.
    pub node_gate: NodeId,
    /// Source node.
    pub node_source: NodeId,
    /// Bulk (body) node.
    pub node_bulk: NodeId,
    /// Model parameters.
    pub params: Bsim1Params,
    /// Pre-calculated derived parameters.
    derived: Bsim1Derived,
}

impl Bsim1Mosfet {
    /// Create a new BSIM1 NMOS transistor with default parameters.
    pub fn nmos(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
    ) -> Self {
        let params = Bsim1Params::nmos_default();
        let derived = Bsim1Derived::from_params(&params);
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            node_bulk: bulk,
            params,
            derived,
        }
    }

    /// Create a new BSIM1 PMOS transistor with default parameters.
    pub fn pmos(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
    ) -> Self {
        let params = Bsim1Params::pmos_default();
        let derived = Bsim1Derived::from_params(&params);
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            node_bulk: bulk,
            params,
            derived,
        }
    }

    /// Create a BSIM1 MOSFET with custom parameters.
    pub fn with_params(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
        params: Bsim1Params,
    ) -> Self {
        let derived = Bsim1Derived::from_params(&params);
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            node_bulk: bulk,
            params,
            derived,
        }
    }

    /// Update derived parameters after modifying instance parameters.
    pub fn update_derived(&mut self) {
        self.derived = Bsim1Derived::from_params(&self.params);
    }

    /// Get the MOSFET type.
    pub fn mos_type(&self) -> MosfetType {
        self.params.mos_type
    }

    /// Evaluate drain current and conductances at given terminal voltages.
    ///
    /// # Arguments
    /// * `vgs` - Gate-source voltage (V)
    /// * `vds` - Drain-source voltage (V)
    /// * `vbs` - Bulk-source voltage (V)
    ///
    /// # Returns
    /// Evaluation result with Ids, gm, gds, gmbs, operating region.
    pub fn evaluate(&self, vgs: f64, vds: f64, vbs: f64) -> Bsim1EvalResult {
        bsim1_evaluate(&self.params, &self.derived, vgs, vds, vbs)
    }

    /// Stamp the linearized BSIM1 model into the MNA system.
    ///
    /// The MOSFET is linearized as:
    /// - gds conductance between drain and source
    /// - gm * Vgs voltage-controlled current source from drain to source
    /// - gmbs * Vbs voltage-controlled current source from drain to source
    /// - Ieq = Ids - gds*Vds - gm*Vgs - gmbs*Vbs (companion current source)
    pub fn stamp_linearized_at(&self, mna: &mut MnaSystem, vgs: f64, vds: f64, vbs: f64) {
        let result = self.evaluate(vgs, vds, vbs);

        let d = node_to_index(self.node_drain);
        let g = node_to_index(self.node_gate);
        let s = node_to_index(self.node_source);
        let b = node_to_index(self.node_bulk);

        let ids = result.ids;
        let gds = result.gds;
        let gm = result.gm;
        let gmbs = result.gmbs;

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

        // Stamp gmbs (body transconductance) as VCCS: I = gmbs * Vbs
        if let Some(di) = d {
            if let Some(bi) = b {
                mna.add_element(di, bi, gmbs);
            }
            if let Some(si) = s {
                mna.add_element(di, si, -gmbs);
            }
        }
        if let Some(si) = s {
            if let Some(bi) = b {
                mna.add_element(si, bi, -gmbs);
            }
            mna.add_element(si, si, gmbs);
        }

        // Equivalent current source: Ieq = Ids - gds*Vds - gm*Vgs - gmbs*Vbs
        let ieq = ids - gds * vds - gm * vgs - gmbs * vbs;
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

impl Stamp for Bsim1Mosfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Initial stamp: Gmin shunt between drain and source
        let d = node_to_index(self.node_drain);
        let s = node_to_index(self.node_source);
        mna.stamp_conductance(d, s, 1e-12);
    }
}

impl Element for Bsim1Mosfet {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![
            self.node_drain,
            self.node_gate,
            self.node_source,
            self.node_bulk,
        ]
    }
}

impl Stamper for Bsim1Mosfet {
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
        let vb = node_to_index(self.node_bulk)
            .map(|i| solution[i])
            .unwrap_or(0.0);

        let vgs = vg - vs;
        let vds = vd - vs;
        let vbs = vb - vs;

        self.stamp_linearized_at(mna, vgs, vds, vbs);
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
        let vb = node_to_index(self.node_bulk)
            .map(|i| solution[i])
            .unwrap_or(0.0);

        let vgs = vg - vs;
        let vds = vd - vs;
        let vbs = vb - vs;

        let result = self.evaluate(vgs, vds, vbs);

        // Return BSIM1 AC info (no intrinsic capacitances for now - DC model only)
        AcDeviceInfo::Bsim1Mosfet {
            drain: node_to_index(self.node_drain),
            gate: node_to_index(self.node_gate),
            source: node_to_index(self.node_source),
            bulk: node_to_index(self.node_bulk),
            gds: result.gds,
            gm: result.gm,
            gmbs: result.gmbs,
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
    fn test_nmos_creation() {
        let m = Bsim1Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        assert_eq!(m.name, "M1");
        assert_eq!(m.mos_type(), MosfetType::Nmos);
        assert_eq!(m.node_drain, NodeId::new(1));
        assert_eq!(m.node_gate, NodeId::new(2));
        assert_eq!(m.node_source, NodeId::GROUND);
        assert_eq!(m.node_bulk, NodeId::GROUND);
    }

    #[test]
    fn test_pmos_creation() {
        let m = Bsim1Mosfet::pmos(
            "M2",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            NodeId::new(3),
        );

        assert_eq!(m.mos_type(), MosfetType::Pmos);
    }

    #[test]
    fn test_evaluate_saturation() {
        let m = Bsim1Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        // Vgs = 2V, Vds = 2V, Vbs = 0 -> saturation
        let result = m.evaluate(2.0, 2.0, 0.0);

        assert_eq!(result.region, Bsim1Region::Saturation);
        assert!(result.ids > 0.0);
        assert!(result.gm > 0.0);
        assert!(result.gds > 0.0);
    }

    #[test]
    fn test_with_custom_params() {
        let mut params = Bsim1Params::nmos_default();
        params.muz = 700.0;
        params.w = 20e-6;
        params.l = 2e-6;

        let m = Bsim1Mosfet::with_params(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
            params,
        );

        assert_eq!(m.params.muz, 700.0);
        assert_eq!(m.params.w, 20e-6);
        assert_eq!(m.params.l, 2e-6);
    }

    #[test]
    fn test_stamp_linearized() {
        let m = Bsim1Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        let mut mna = MnaSystem::new(2, 0);
        m.stamp_linearized_at(&mut mna, 2.0, 2.0, 0.0);

        // Matrix should have non-zero entries
        let matrix = mna.to_dense_matrix();
        assert!(matrix[(0, 0)].abs() > 1e-15); // gds at drain
        assert!(matrix[(0, 1)].abs() > 1e-15); // gm contribution at drain-gate
    }

    #[test]
    fn test_ac_info() {
        let m = Bsim1Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        // DC solution: Vd=2V, Vg=2V
        let solution = DVector::from_vec(vec![2.0, 2.0]);
        let ac_info = m.ac_info_at(&solution);

        match ac_info {
            AcDeviceInfo::Bsim1Mosfet {
                drain,
                gate,
                source,
                bulk,
                gds,
                gm,
                gmbs,
            } => {
                assert_eq!(drain, Some(0));
                assert_eq!(gate, Some(1));
                assert_eq!(source, None);
                assert_eq!(bulk, None);
                assert!(gm > 0.0);
                assert!(gds > 0.0);
                assert!(gmbs >= 0.0);
            }
            _ => panic!("Expected AcDeviceInfo::Bsim1Mosfet"),
        }
    }
}
