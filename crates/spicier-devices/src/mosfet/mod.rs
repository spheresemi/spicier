//! MOSFET device models.
//!
//! This module provides MOSFET models at different levels:
//! - Level 1 (Shichman-Hodges): Simple square-law model for hand analysis
//! - Level 4 (BSIM1): Empirical model for ~1 µm short-channel MOSFETs
//! - BSIM3v3 (Level 49/8): Industry-standard short-channel model
//! - BSIM4 (Level 54): Advanced model with QM effects, stress, gate current

pub mod bsim1;
pub mod bsim3;
pub mod bsim4;
pub mod level1;

// Re-export Level 1 types (for backwards compatibility)
pub use level1::{Mosfet, MosfetParams, MosfetRegion, MosfetType};

// Re-export BSIM1 types
pub use bsim1::{Bsim1Mosfet, Bsim1Params, Bsim1Region};

// Re-export BSIM3 types
pub use bsim3::{Bsim3Mosfet, Bsim3Params};

// Re-export BSIM4 types
pub use bsim4::{Bsim4Mosfet, Bsim4Params};

/// MOSFET model level identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MosfetLevel {
    /// Level 1: Shichman-Hodges square-law model
    #[default]
    Level1,
    /// Level 4: BSIM1 empirical model for ~1 µm devices
    Bsim1,
    /// BSIM3v3.3: Berkeley short-channel IGFET model (LEVEL=49 or 8)
    Bsim3,
    /// BSIM4: Advanced model with QM effects, stress, gate current (LEVEL=54)
    Bsim4,
}
