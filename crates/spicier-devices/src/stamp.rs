//! MNA stamping trait and utilities.

use spicier_core::mna::MnaSystem;

/// Trait for devices that can stamp into an MNA matrix.
pub trait Stamp {
    /// Stamp this device's contribution into the MNA system.
    ///
    /// The device should add its conductance/coefficient contributions
    /// to the matrix and its source contributions to the RHS vector.
    fn stamp(&self, mna: &mut MnaSystem);

    /// Clear and re-stamp (useful for nonlinear iteration).
    fn restamp(&self, mna: &mut MnaSystem) {
        self.stamp(mna);
    }
}
