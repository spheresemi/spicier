//! MOSFET device models.
//!
//! This module provides MOSFET models at different levels:
//! - Level 1 (Shichman-Hodges): Simple square-law model for hand analysis
//! - BSIM3v3 (Level 49/8): Industry-standard short-channel model

pub mod bsim3;
pub mod level1;

// Re-export Level 1 types (for backwards compatibility)
pub use level1::{Mosfet, MosfetParams, MosfetRegion, MosfetType};

// Re-export BSIM3 types
pub use bsim3::{Bsim3Mosfet, Bsim3Params};

/// MOSFET model level identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MosfetLevel {
    /// Level 1: Shichman-Hodges square-law model
    #[default]
    Level1,
    /// BSIM3v3.3: Berkeley short-channel IGFET model (LEVEL=49 or 8)
    Bsim3,
}
