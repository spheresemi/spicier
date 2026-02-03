//! BSIM3v3.3 MOSFET model.
//!
//! This module implements the Berkeley Short-channel IGFET Model version 3.3,
//! an industry-standard compact model for short-channel MOSFETs.
//!
//! # Features
//!
//! This "BSIM3-lite" implementation includes:
//! - Threshold voltage with short-channel effects (SCE)
//! - DIBL (drain-induced barrier lowering)
//! - Mobility degradation (vertical and lateral fields)
//! - Velocity saturation
//! - Channel length modulation
//! - Body effect
//! - Source/drain parasitic resistance
//!
//! # Usage
//!
//! ```text
//! .MODEL NMOD NMOS LEVEL=49 VTH0=0.4 U0=400 TOX=9e-9
//! M1 d g s b NMOD W=1u L=100n
//! ```
//!
//! # References
//!
//! - BSIM3v3.3 Manual: https://bsim.berkeley.edu/models/bsim3/

pub mod derived;
pub mod evaluate;
pub mod params;

pub use derived::Bsim3Derived;
pub use evaluate::{
    Bsim3CapResult, Bsim3EvalResult, Bsim3Region, evaluate as bsim3_evaluate,
    evaluate_capacitances as bsim3_evaluate_caps,
};
pub use params::Bsim3Params;

use super::level1::MosfetType;
use crate::stamp::Stamp;

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

/// A BSIM3v3.3 MOSFET device.
#[derive(Debug, Clone)]
pub struct Bsim3Mosfet {
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
    pub params: Bsim3Params,
    /// Pre-calculated derived parameters.
    derived: Bsim3Derived,
}

impl Bsim3Mosfet {
    /// Create a new BSIM3 NMOS transistor with default parameters.
    pub fn nmos(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
    ) -> Self {
        let params = Bsim3Params::nmos_default();
        let derived = Bsim3Derived::from_params(&params);
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

    /// Create a new BSIM3 PMOS transistor with default parameters.
    pub fn pmos(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
    ) -> Self {
        let params = Bsim3Params::pmos_default();
        let derived = Bsim3Derived::from_params(&params);
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

    /// Create a BSIM3 MOSFET with custom parameters.
    pub fn with_params(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
        params: Bsim3Params,
    ) -> Self {
        let derived = Bsim3Derived::from_params(&params);
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
        self.derived = Bsim3Derived::from_params(&self.params);
    }

    /// Set the operating temperature and update derived parameters.
    ///
    /// This applies BSIM3 temperature scaling to all temperature-dependent
    /// parameters including Vth0, mobility, saturation velocity, and Rds.
    ///
    /// # Arguments
    /// * `temp` - Operating temperature in Kelvin
    pub fn set_temperature(&mut self, temp: f64) {
        self.derived = Bsim3Derived::from_params_at_temp(&self.params, temp);
    }

    /// Get the current operating temperature (K).
    pub fn temperature(&self) -> f64 {
        self.derived.temp
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
    pub fn evaluate(&self, vgs: f64, vds: f64, vbs: f64) -> Bsim3EvalResult {
        bsim3_evaluate(&self.params, &self.derived, vgs, vds, vbs)
    }

    /// Stamp the linearized BSIM3 model into the MNA system.
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

impl Stamp for Bsim3Mosfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Initial stamp: Gmin shunt between drain and source
        let d = node_to_index(self.node_drain);
        let s = node_to_index(self.node_source);
        mna.stamp_conductance(d, s, 1e-12);
    }
}

impl Element for Bsim3Mosfet {
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

impl Stamper for Bsim3Mosfet {
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

        // Calculate capacitances at operating point
        let caps = bsim3_evaluate_caps(
            &self.params,
            &self.derived,
            vgs,
            vds,
            vbs,
            result.region,
            result.vth,
            result.vdsat,
        );

        AcDeviceInfo::Bsim3Mosfet {
            drain: node_to_index(self.node_drain),
            gate: node_to_index(self.node_gate),
            source: node_to_index(self.node_source),
            bulk: node_to_index(self.node_bulk),
            gds: result.gds,
            gm: result.gm,
            gmbs: result.gmbs,
            cgs: caps.cgs,
            cgd: caps.cgd,
            cgb: caps.cgb,
            cbs: caps.cbs,
            cbd: caps.cbd,
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
        let m = Bsim3Mosfet::nmos(
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
        let m = Bsim3Mosfet::pmos(
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
        let m = Bsim3Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        // Vgs = 1.0V, Vds = 1.0V, Vbs = 0 -> saturation
        let result = m.evaluate(1.0, 1.0, 0.0);

        assert_eq!(result.region, Bsim3Region::Saturation);
        assert!(result.ids > 0.0);
        assert!(result.gm > 0.0);
        assert!(result.gds > 0.0);
    }

    #[test]
    fn test_with_custom_params() {
        let mut params = Bsim3Params::nmos_default();
        params.vth0 = 0.5;
        params.w = 2e-6;
        params.l = 50e-9;

        let m = Bsim3Mosfet::with_params(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
            params,
        );

        assert_eq!(m.params.vth0, 0.5);
        assert_eq!(m.params.w, 2e-6);
        assert_eq!(m.params.l, 50e-9);
    }

    #[test]
    fn test_stamp_linearized() {
        let m = Bsim3Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        let mut mna = MnaSystem::new(2, 0);
        m.stamp_linearized_at(&mut mna, 1.0, 1.0, 0.0);

        // Matrix should have non-zero entries
        let matrix = mna.to_dense_matrix();
        assert!(matrix[(0, 0)].abs() > 1e-15); // gds at drain
        assert!(matrix[(0, 1)].abs() > 1e-15); // gm contribution at drain-gate
    }

    #[test]
    fn test_ac_info() {
        let m = Bsim3Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        // DC solution: Vd=1V, Vg=1V
        let solution = DVector::from_vec(vec![1.0, 1.0]);
        let ac_info = m.ac_info_at(&solution);

        match ac_info {
            AcDeviceInfo::Bsim3Mosfet {
                drain,
                gate,
                source,
                bulk,
                gds,
                gm,
                gmbs,
                cgs,
                cgd,
                cgb,
                cbs,
                cbd,
            } => {
                assert_eq!(drain, Some(0));
                assert_eq!(gate, Some(1));
                assert_eq!(source, None);
                assert_eq!(bulk, None);
                assert!(gm > 0.0);
                assert!(gds > 0.0);
                assert!(gmbs >= 0.0);
                // Capacitances should be non-negative
                assert!(cgs >= 0.0);
                assert!(cgd >= 0.0);
                assert!(cgb >= 0.0);
                assert!(cbs >= 0.0);
                assert!(cbd >= 0.0);
            }
            _ => panic!("Expected AcDeviceInfo::Bsim3Mosfet"),
        }
    }
}
