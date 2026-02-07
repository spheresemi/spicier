//! BSIM1 (Level 4) model parameters.
//!
//! This module defines the parameter struct for the BSIM1 (Berkeley Short-channel
//! IGFET Model v1) which targets ~1 um short-channel MOSFETs. BSIM1 uses polynomial
//! bias-dependent parameters with L/W scaling.
//!
//! Each BSIM1 parameter follows the scaling formula:
//!   P_eff = P0 + PL/Leff + PW/Weff
//!
//! where P0 is the base value, PL is the length coefficient, and PW is the width coefficient.

use super::super::level1::MosfetType;

/// BSIM1 model parameters.
///
/// BSIM1 is an empirical model for short-channel MOSFETs. Unlike BSIM3, all parameters
/// default to 0.0 - users must provide extracted parameters from their process.
///
/// # Parameter Naming Convention
/// - Base parameters end with `0` (e.g., `vfb0`)
/// - Length coefficients end with `l` (e.g., `vfb_l`)
/// - Width coefficients end with `w` (e.g., `vfb_w`)
#[derive(Debug, Clone)]
pub struct Bsim1Params {
    // ========================================
    // Process Parameters
    // ========================================
    /// Gate oxide thickness (m). Required parameter.
    pub tox: f64,
    /// Nominal temperature (K). Default: 300.15 K (27°C)
    pub temp: f64,
    /// Supply voltage for beta interpolation (V). Default: 5.0V
    pub vdd: f64,
    /// Channel length reduction DL (m). Default: 0.0
    pub dl: f64,
    /// Channel width reduction DW (m). Default: 0.0
    pub dw: f64,

    // ========================================
    // Threshold Voltage Parameters
    // ========================================
    /// Flat-band voltage base (V)
    pub vfb0: f64,
    /// Flat-band voltage length coefficient (V·m)
    pub vfb_l: f64,
    /// Flat-band voltage width coefficient (V·m)
    pub vfb_w: f64,

    /// Surface potential PHI base (V)
    pub phi0: f64,
    /// Surface potential length coefficient (V·m)
    pub phi_l: f64,
    /// Surface potential width coefficient (V·m)
    pub phi_w: f64,

    /// Body effect coefficient K1 base (V^0.5)
    pub k10: f64,
    /// K1 length coefficient (V^0.5·m)
    pub k1_l: f64,
    /// K1 width coefficient (V^0.5·m)
    pub k1_w: f64,

    /// Drain/gate bias dependence K2 base
    pub k20: f64,
    /// K2 length coefficient (m)
    pub k2_l: f64,
    /// K2 width coefficient (m)
    pub k2_w: f64,

    // ========================================
    // DIBL (Drain-Induced Barrier Lowering) Parameters
    // ========================================
    /// DIBL coefficient ETA base
    pub eta0: f64,
    /// ETA length coefficient (m)
    pub eta_l: f64,
    /// ETA width coefficient (m)
    pub eta_w: f64,

    /// Body-bias sensitivity of ETA base (1/V)
    pub etab0: f64,
    /// ETAB length coefficient (m/V)
    pub etab_l: f64,
    /// ETAB width coefficient (m/V)
    pub etab_w: f64,

    /// Drain-bias sensitivity of ETA base (1/V)
    pub etad0: f64,
    /// ETAD length coefficient (m/V)
    pub etad_l: f64,
    /// ETAD width coefficient (m/V)
    pub etad_w: f64,

    // ========================================
    // Mobility Parameters
    // ========================================
    /// Zero-bias mobility MUZ (cm²/V·s). No L/W scaling - single value.
    pub muz: f64,

    /// Vbs dependence of mobility at Vds=0 (X2MZ) base (cm²/V²·s)
    pub x2mz0: f64,
    /// X2MZ length coefficient (cm²·m/V²·s)
    pub x2mz_l: f64,
    /// X2MZ width coefficient (cm²·m/V²·s)
    pub x2mz_w: f64,

    /// Mobility at Vds=Vdd (MUS) base (cm²/V·s)
    pub mus0: f64,
    /// MUS length coefficient (cm²·m/V·s)
    pub mus_l: f64,
    /// MUS width coefficient (cm²·m/V·s)
    pub mus_w: f64,

    /// Vbs dependence of MUS (X2MS) base (cm²/V²·s)
    pub x2ms0: f64,
    /// X2MS length coefficient (cm²·m/V²·s)
    pub x2ms_l: f64,
    /// X2MS width coefficient (cm²·m/V²·s)
    pub x2ms_w: f64,

    /// Vds dependence of MUS (X3MS) base (cm²/V²·s)
    pub x3ms0: f64,
    /// X3MS length coefficient (cm²·m/V²·s)
    pub x3ms_l: f64,
    /// X3MS width coefficient (cm²·m/V²·s)
    pub x3ms_w: f64,

    // ========================================
    // Gate-Field Mobility Degradation Parameters
    // ========================================
    /// Gate-field degradation U0 base (m/V)
    pub u00: f64,
    /// U0 length coefficient (m²/V)
    pub u0_l: f64,
    /// U0 width coefficient (m²/V)
    pub u0_w: f64,

    /// Body-bias dependence of U0 (U0B) base (m/V²)
    pub u0b0: f64,
    /// U0B length coefficient (m²/V²)
    pub u0b_l: f64,
    /// U0B width coefficient (m²/V²)
    pub u0b_w: f64,

    // ========================================
    // Velocity Saturation Parameters
    // ========================================
    /// Velocity saturation U1 base (m/V)
    pub u10: f64,
    /// U1 length coefficient (m²/V)
    pub u1_l: f64,
    /// U1 width coefficient (m²/V)
    pub u1_w: f64,

    /// Body-bias dependence of U1 (U1B) base (m/V²)
    pub u1b0: f64,
    /// U1B length coefficient (m²/V²)
    pub u1b_l: f64,
    /// U1B width coefficient (m²/V²)
    pub u1b_w: f64,

    /// Drain-bias dependence of U1 (U1D) base (m/V²)
    pub u1d0: f64,
    /// U1D length coefficient (m²/V²)
    pub u1d_l: f64,
    /// U1D width coefficient (m²/V²)
    pub u1d_w: f64,

    // ========================================
    // Subthreshold Parameters
    // ========================================
    /// Subthreshold swing factor N0 base
    pub n00: f64,
    /// N0 length coefficient (m)
    pub n0_l: f64,
    /// N0 width coefficient (m)
    pub n0_w: f64,

    /// Body-bias sensitivity of N (NB) base (1/V)
    pub nb0: f64,
    /// NB length coefficient (m/V)
    pub nb_l: f64,
    /// NB width coefficient (m/V)
    pub nb_w: f64,

    /// Drain-bias sensitivity of N (ND) base (1/V)
    pub nd0: f64,
    /// ND length coefficient (m/V)
    pub nd_l: f64,
    /// ND width coefficient (m/V)
    pub nd_w: f64,

    // ========================================
    // Overlap Capacitances
    // ========================================
    /// Gate-source overlap capacitance per unit width (F/m)
    pub cgso: f64,
    /// Gate-drain overlap capacitance per unit width (F/m)
    pub cgdo: f64,
    /// Gate-bulk overlap capacitance per unit length (F/m)
    pub cgbo: f64,

    // ========================================
    // Junction Parameters
    // ========================================
    /// Sheet resistance of source/drain (Ohm/sq)
    pub rsh: f64,
    /// Saturation current density (A/m²)
    pub js: f64,
    /// Bulk junction bottom potential (V)
    pub pb: f64,
    /// Bulk junction bottom grading coefficient
    pub mj: f64,
    /// Bulk junction sidewall potential (V)
    pub pbsw: f64,
    /// Bulk junction sidewall grading coefficient
    pub mjsw: f64,
    /// Zero-bias bulk junction bottom capacitance per area (F/m²)
    pub cj: f64,
    /// Zero-bias bulk junction sidewall capacitance per length (F/m)
    pub cjsw: f64,

    // ========================================
    // Instance Parameters
    // ========================================
    /// Channel width (m). Required for device instances.
    pub w: f64,
    /// Channel length (m). Required for device instances.
    pub l: f64,
    /// Source diffusion area (m²). Default: 0.0 (auto-calculated if zero)
    pub as_: f64,
    /// Drain diffusion area (m²). Default: 0.0 (auto-calculated if zero)
    pub ad: f64,
    /// Source diffusion perimeter (m). Default: 0.0 (auto-calculated if zero)
    pub ps: f64,
    /// Drain diffusion perimeter (m). Default: 0.0 (auto-calculated if zero)
    pub pd: f64,
    /// Device type (NMOS or PMOS)
    pub mos_type: MosfetType,
}

impl Bsim1Params {
    /// Physical constants
    pub const Q: f64 = 1.602176634e-19; // Elementary charge (C)
    pub const KB: f64 = 1.380649e-23; // Boltzmann constant (J/K)
    pub const EPS_SI: f64 = 1.03594e-10; // Permittivity of Si (F/m) = 11.7 * eps0
    pub const EPS_OX: f64 = 3.45314e-11; // Permittivity of SiO2 (F/m) = 3.9 * eps0
    pub const T_NOM: f64 = 300.15; // Nominal temperature (K)

    /// Create default NMOS BSIM1 parameters.
    ///
    /// All parameters default to 0.0 except for required physical parameters.
    /// This is the BSIM1 convention - users must provide extracted parameters.
    pub fn nmos_default() -> Self {
        Self {
            // Process parameters - minimal defaults
            tox: 25e-9, // 25nm oxide (typical for BSIM1 era)
            temp: Self::T_NOM,
            vdd: 5.0,
            dl: 0.0,
            dw: 0.0,

            // Threshold voltage - all zeros
            vfb0: -0.9,
            vfb_l: 0.0,
            vfb_w: 0.0,
            phi0: 0.6,
            phi_l: 0.0,
            phi_w: 0.0,
            k10: 0.5,
            k1_l: 0.0,
            k1_w: 0.0,
            k20: 0.0,
            k2_l: 0.0,
            k2_w: 0.0,

            // DIBL - all zeros
            eta0: 0.0,
            eta_l: 0.0,
            eta_w: 0.0,
            etab0: 0.0,
            etab_l: 0.0,
            etab_w: 0.0,
            etad0: 0.0,
            etad_l: 0.0,
            etad_w: 0.0,

            // Mobility - reasonable defaults for NMOS
            muz: 600.0, // cm²/V·s
            x2mz0: 0.0,
            x2mz_l: 0.0,
            x2mz_w: 0.0,
            mus0: 500.0,
            mus_l: 0.0,
            mus_w: 0.0,
            x2ms0: 0.0,
            x2ms_l: 0.0,
            x2ms_w: 0.0,
            x3ms0: 0.0,
            x3ms_l: 0.0,
            x3ms_w: 0.0,

            // Gate-field degradation - all zeros
            u00: 0.0,
            u0_l: 0.0,
            u0_w: 0.0,
            u0b0: 0.0,
            u0b_l: 0.0,
            u0b_w: 0.0,

            // Velocity saturation - all zeros
            u10: 0.0,
            u1_l: 0.0,
            u1_w: 0.0,
            u1b0: 0.0,
            u1b_l: 0.0,
            u1b_w: 0.0,
            u1d0: 0.0,
            u1d_l: 0.0,
            u1d_w: 0.0,

            // Subthreshold - reasonable defaults
            n00: 1.0,
            n0_l: 0.0,
            n0_w: 0.0,
            nb0: 0.0,
            nb_l: 0.0,
            nb_w: 0.0,
            nd0: 0.0,
            nd_l: 0.0,
            nd_w: 0.0,

            // Overlap capacitances
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,

            // Junction parameters
            rsh: 0.0,
            js: 1e-4,
            pb: 0.8,
            mj: 0.5,
            pbsw: 0.8,
            mjsw: 0.33,
            cj: 0.0,
            cjsw: 0.0,

            // Instance defaults
            w: 10e-6, // 10 µm
            l: 1e-6,  // 1 µm
            as_: 0.0,
            ad: 0.0,
            ps: 0.0,
            pd: 0.0,
            mos_type: MosfetType::Nmos,
        }
    }

    /// Create default PMOS BSIM1 parameters.
    ///
    /// Similar to NMOS but with appropriate sign changes for PMOS operation.
    pub fn pmos_default() -> Self {
        Self {
            // Process parameters
            tox: 25e-9,
            temp: Self::T_NOM,
            vdd: 5.0,
            dl: 0.0,
            dw: 0.0,

            // Threshold voltage - adjusted for PMOS
            vfb0: 0.1,
            vfb_l: 0.0,
            vfb_w: 0.0,
            phi0: 0.6,
            phi_l: 0.0,
            phi_w: 0.0,
            k10: 0.5,
            k1_l: 0.0,
            k1_w: 0.0,
            k20: 0.0,
            k2_l: 0.0,
            k2_w: 0.0,

            // DIBL - all zeros
            eta0: 0.0,
            eta_l: 0.0,
            eta_w: 0.0,
            etab0: 0.0,
            etab_l: 0.0,
            etab_w: 0.0,
            etad0: 0.0,
            etad_l: 0.0,
            etad_w: 0.0,

            // Mobility - lower for PMOS
            muz: 250.0, // cm²/V·s (holes have lower mobility)
            x2mz0: 0.0,
            x2mz_l: 0.0,
            x2mz_w: 0.0,
            mus0: 200.0,
            mus_l: 0.0,
            mus_w: 0.0,
            x2ms0: 0.0,
            x2ms_l: 0.0,
            x2ms_w: 0.0,
            x3ms0: 0.0,
            x3ms_l: 0.0,
            x3ms_w: 0.0,

            // Gate-field degradation - all zeros
            u00: 0.0,
            u0_l: 0.0,
            u0_w: 0.0,
            u0b0: 0.0,
            u0b_l: 0.0,
            u0b_w: 0.0,

            // Velocity saturation - all zeros
            u10: 0.0,
            u1_l: 0.0,
            u1_w: 0.0,
            u1b0: 0.0,
            u1b_l: 0.0,
            u1b_w: 0.0,
            u1d0: 0.0,
            u1d_l: 0.0,
            u1d_w: 0.0,

            // Subthreshold
            n00: 1.0,
            n0_l: 0.0,
            n0_w: 0.0,
            nb0: 0.0,
            nb_l: 0.0,
            nb_w: 0.0,
            nd0: 0.0,
            nd_l: 0.0,
            nd_w: 0.0,

            // Overlap capacitances
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,

            // Junction parameters
            rsh: 0.0,
            js: 1e-4,
            pb: 0.8,
            mj: 0.5,
            pbsw: 0.8,
            mjsw: 0.33,
            cj: 0.0,
            cjsw: 0.0,

            // Instance defaults
            w: 10e-6,
            l: 1e-6,
            as_: 0.0,
            ad: 0.0,
            ps: 0.0,
            pd: 0.0,
            mos_type: MosfetType::Pmos,
        }
    }

    /// Calculate thermal voltage at operating temperature.
    #[inline]
    pub fn vt(&self) -> f64 {
        Self::KB * self.temp / Self::Q
    }

    /// Calculate oxide capacitance per unit area (F/m²).
    #[inline]
    pub fn cox(&self) -> f64 {
        Self::EPS_OX / self.tox
    }

    /// Calculate effective channel length (m).
    #[inline]
    pub fn leff(&self) -> f64 {
        (self.l - self.dl).max(1e-9)
    }

    /// Calculate effective channel width (m).
    #[inline]
    pub fn weff(&self) -> f64 {
        (self.w - self.dw).max(1e-9)
    }

    /// Apply L/W scaling to a parameter.
    ///
    /// P_eff = P0 + PL/Leff + PW/Weff
    #[inline]
    pub fn scaled_param(p0: f64, pl: f64, pw: f64, leff: f64, weff: f64) -> f64 {
        p0 + pl / leff + pw / weff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmos_defaults() {
        let params = Bsim1Params::nmos_default();
        assert_eq!(params.mos_type, MosfetType::Nmos);
        assert!(params.muz > 0.0, "MUZ should have a default value");
        assert!(params.tox > 0.0, "TOX should have a default value");
    }

    #[test]
    fn test_pmos_defaults() {
        let params = Bsim1Params::pmos_default();
        assert_eq!(params.mos_type, MosfetType::Pmos);
        assert!(
            params.muz < Bsim1Params::nmos_default().muz,
            "PMOS mobility should be lower"
        );
    }

    #[test]
    fn test_thermal_voltage() {
        let params = Bsim1Params::nmos_default();
        let vt = params.vt();
        // At 300K, Vt should be approximately 25.9 mV
        assert!((vt - 0.0259).abs() < 0.001);
    }

    #[test]
    fn test_oxide_capacitance() {
        let params = Bsim1Params::nmos_default();
        let cox = params.cox();
        // Cox = eps_ox / tox = 3.45e-11 / 25e-9 ≈ 1.38e-3 F/m²
        assert!(cox > 1e-3 && cox < 2e-3, "Cox should be ~1.38 mF/m²");
    }

    #[test]
    fn test_effective_dimensions() {
        let mut params = Bsim1Params::nmos_default();
        params.w = 10e-6;
        params.l = 1e-6;
        params.dl = 0.1e-6;
        params.dw = 0.2e-6;

        let leff = params.leff();
        let weff = params.weff();

        // Leff = 1um - 0.1um = 0.9um
        assert!((leff - 0.9e-6).abs() < 1e-12);
        // Weff = 10um - 0.2um = 9.8um
        assert!((weff - 9.8e-6).abs() < 1e-12);
    }

    #[test]
    fn test_scaled_param() {
        let leff = 1e-6;
        let weff = 10e-6;

        // Test with simple values
        let p0 = 1.0;
        let pl = 1e-6; // Will contribute 1.0 when divided by Leff=1um
        let pw = 10e-6; // Will contribute 1.0 when divided by Weff=10um

        let result = Bsim1Params::scaled_param(p0, pl, pw, leff, weff);
        // P_eff = 1.0 + 1e-6/1e-6 + 10e-6/10e-6 = 1.0 + 1.0 + 1.0 = 3.0
        assert!((result - 3.0).abs() < 1e-10);
    }
}
