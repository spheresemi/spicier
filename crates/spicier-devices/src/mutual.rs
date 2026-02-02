//! Mutual Inductance (K element) for coupling between inductors.
//!
//! Mutual inductance represents magnetic coupling between two inductors.
//! The mutual inductance M is defined as: M = k * sqrt(L1 * L2)
//! where k is the coupling coefficient (0 < k <= 1).

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// Mutual inductance coupling between two inductors.
#[derive(Debug, Clone)]
pub struct MutualInductance {
    /// Device name (e.g., "K1").
    pub name: String,
    /// Name of the first inductor.
    pub inductor1_name: String,
    /// Name of the second inductor.
    pub inductor2_name: String,
    /// Coupling coefficient k (0 < k <= 1).
    pub coupling_coeff: f64,
    /// Branch current index of the first inductor (resolved after parsing).
    pub l1_branch_idx: Option<usize>,
    /// Branch current index of the second inductor (resolved after parsing).
    pub l2_branch_idx: Option<usize>,
    /// Inductance value of L1 (resolved after parsing).
    pub l1_value: f64,
    /// Inductance value of L2 (resolved after parsing).
    pub l2_value: f64,
}

impl MutualInductance {
    /// Create a new mutual inductance coupling.
    ///
    /// The inductor names are stored and branch indices must be resolved
    /// after the netlist is fully parsed.
    pub fn new(
        name: impl Into<String>,
        inductor1_name: impl Into<String>,
        inductor2_name: impl Into<String>,
        coupling_coeff: f64,
    ) -> Self {
        Self {
            name: name.into(),
            inductor1_name: inductor1_name.into(),
            inductor2_name: inductor2_name.into(),
            coupling_coeff: coupling_coeff.clamp(0.0, 1.0),
            l1_branch_idx: None,
            l2_branch_idx: None,
            l1_value: 0.0,
            l2_value: 0.0,
        }
    }

    /// Create with resolved branch indices and inductance values.
    pub fn with_resolved(
        name: impl Into<String>,
        inductor1_name: impl Into<String>,
        inductor2_name: impl Into<String>,
        coupling_coeff: f64,
        l1_branch_idx: usize,
        l2_branch_idx: usize,
        l1_value: f64,
        l2_value: f64,
    ) -> Self {
        Self {
            name: name.into(),
            inductor1_name: inductor1_name.into(),
            inductor2_name: inductor2_name.into(),
            coupling_coeff: coupling_coeff.clamp(0.0, 1.0),
            l1_branch_idx: Some(l1_branch_idx),
            l2_branch_idx: Some(l2_branch_idx),
            l1_value,
            l2_value,
        }
    }

    /// Calculate mutual inductance M = k * sqrt(L1 * L2).
    pub fn mutual_inductance(&self) -> f64 {
        self.coupling_coeff * (self.l1_value * self.l2_value).sqrt()
    }

    /// Resolve inductor references by setting branch indices and values.
    pub fn resolve(
        &mut self,
        l1_branch_idx: usize,
        l2_branch_idx: usize,
        l1_value: f64,
        l2_value: f64,
    ) {
        self.l1_branch_idx = Some(l1_branch_idx);
        self.l2_branch_idx = Some(l2_branch_idx);
        self.l1_value = l1_value;
        self.l2_value = l2_value;
    }

    /// Check if the inductor references have been resolved.
    pub fn is_resolved(&self) -> bool {
        self.l1_branch_idx.is_some() && self.l2_branch_idx.is_some()
    }
}

impl Stamp for MutualInductance {
    fn stamp(&self, _mna: &mut MnaSystem) {
        // For DC analysis, inductors are short circuits and mutual inductance
        // has no effect. The coupling only matters for AC and transient analysis.
        //
        // The actual coupling stamps are added by the AC/transient analysis
        // engines which use the ac_info() and transient_info() methods.
    }
}

impl Element for MutualInductance {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        // K element doesn't connect to nodes directly; it couples inductors
        Vec::new()
    }
}

impl Stamper for MutualInductance {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn is_nonlinear(&self) -> bool {
        false
    }

    fn ac_info(&self) -> AcDeviceInfo {
        if let (Some(l1_idx), Some(l2_idx)) = (self.l1_branch_idx, self.l2_branch_idx) {
            AcDeviceInfo::MutualInductance {
                l1_branch_idx: l1_idx,
                l2_branch_idx: l2_idx,
                mutual_inductance: self.mutual_inductance(),
            }
        } else {
            AcDeviceInfo::None
        }
    }

    fn ac_info_at(&self, _solution: &DVector<f64>) -> AcDeviceInfo {
        self.ac_info()
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        if let (Some(l1_idx), Some(l2_idx)) = (self.l1_branch_idx, self.l2_branch_idx) {
            TransientDeviceInfo::MutualInductance {
                l1_branch_idx: l1_idx,
                l2_branch_idx: l2_idx,
                mutual_inductance: self.mutual_inductance(),
            }
        } else {
            TransientDeviceInfo::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutual_inductance_calculation() {
        let k = MutualInductance::with_resolved(
            "K1", "L1", "L2", 0.9,
            0, 1,  // Branch indices
            1e-3, 1e-3,  // 1mH each
        );

        // M = k * sqrt(L1 * L2) = 0.9 * sqrt(1e-3 * 1e-3) = 0.9e-3 = 0.9 mH
        let m = k.mutual_inductance();
        assert!(
            (m - 0.9e-3).abs() < 1e-10,
            "M = {} (expected 0.9e-3)",
            m
        );
    }

    #[test]
    fn test_coupling_coefficient_clamping() {
        // k > 1 should be clamped to 1
        let k1 = MutualInductance::new("K1", "L1", "L2", 1.5);
        assert_eq!(k1.coupling_coeff, 1.0);

        // k < 0 should be clamped to 0
        let k2 = MutualInductance::new("K2", "L1", "L2", -0.5);
        assert_eq!(k2.coupling_coeff, 0.0);

        // Valid k should pass through
        let k3 = MutualInductance::new("K3", "L1", "L2", 0.8);
        assert_eq!(k3.coupling_coeff, 0.8);
    }

    #[test]
    fn test_unresolved_ac_info() {
        let k = MutualInductance::new("K1", "L1", "L2", 0.9);

        assert!(!k.is_resolved());

        let ac_info = k.ac_info();
        assert!(matches!(ac_info, AcDeviceInfo::None));
    }

    #[test]
    fn test_resolved_ac_info() {
        let k = MutualInductance::with_resolved(
            "K1", "L1", "L2", 0.9,
            0, 1,
            1e-3, 1e-3,
        );

        assert!(k.is_resolved());

        let ac_info = k.ac_info();
        match ac_info {
            AcDeviceInfo::MutualInductance {
                l1_branch_idx,
                l2_branch_idx,
                mutual_inductance,
            } => {
                assert_eq!(l1_branch_idx, 0);
                assert_eq!(l2_branch_idx, 1);
                assert!((mutual_inductance - 0.9e-3).abs() < 1e-10);
            }
            _ => panic!("Expected AcDeviceInfo::MutualInductance"),
        }
    }

    #[test]
    fn test_resolve() {
        let mut k = MutualInductance::new("K1", "L1", "L2", 0.9);
        assert!(!k.is_resolved());

        k.resolve(0, 1, 1e-3, 1e-3);
        assert!(k.is_resolved());
        assert_eq!(k.l1_branch_idx, Some(0));
        assert_eq!(k.l2_branch_idx, Some(1));
        assert_eq!(k.l1_value, 1e-3);
        assert_eq!(k.l2_value, 1e-3);
    }

    #[test]
    fn test_different_inductance_values() {
        let k = MutualInductance::with_resolved(
            "K1", "L1", "L2", 1.0,  // Perfect coupling
            0, 1,
            1e-3, 4e-3,  // L1=1mH, L2=4mH
        );

        // M = 1.0 * sqrt(1e-3 * 4e-3) = 1.0 * 2e-3 = 2mH
        let m = k.mutual_inductance();
        assert!(
            (m - 2e-3).abs() < 1e-10,
            "M = {} (expected 2e-3)",
            m
        );
    }

    #[test]
    fn test_nodes_empty() {
        let k = MutualInductance::new("K1", "L1", "L2", 0.9);
        assert!(k.nodes().is_empty());
    }

    #[test]
    fn test_transient_info() {
        let k = MutualInductance::with_resolved(
            "K1", "L1", "L2", 0.9,
            0, 1,
            1e-3, 1e-3,
        );

        let info = k.transient_info();
        match info {
            TransientDeviceInfo::MutualInductance {
                l1_branch_idx,
                l2_branch_idx,
                mutual_inductance,
            } => {
                assert_eq!(l1_branch_idx, 0);
                assert_eq!(l2_branch_idx, 1);
                assert!((mutual_inductance - 0.9e-3).abs() < 1e-10);
            }
            _ => panic!("Expected TransientDeviceInfo::MutualInductance"),
        }
    }
}
