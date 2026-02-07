//! BSIM1 derived (pre-calculated) parameters.
//!
//! These are parameters computed once from the model/instance parameters
//! and reused during device evaluation. This avoids redundant calculations
//! during Newton-Raphson iterations.

use super::params::Bsim1Params;

/// Pre-calculated BSIM1 parameters derived from model parameters.
///
/// These values are computed once when the device is created and
/// stored for efficient reuse during simulation.
#[derive(Debug, Clone)]
pub struct Bsim1Derived {
    /// Effective channel length (m)
    pub leff: f64,
    /// Effective channel width (m)
    pub weff: f64,
    /// Oxide capacitance per unit area (F/m²)
    pub cox: f64,
    /// Thermal voltage (V) at operating temperature
    pub vt: f64,

    // ========================================
    // Scaled Threshold Voltage Parameters
    // ========================================
    /// Effective flat-band voltage (V)
    pub vfb: f64,
    /// Effective surface potential PHI (V)
    pub phi: f64,
    /// Effective body effect coefficient K1 (V^0.5)
    pub k1: f64,
    /// Effective drain/gate bias coefficient K2
    pub k2: f64,

    // ========================================
    // Scaled DIBL Parameters
    // ========================================
    /// Effective DIBL coefficient ETA
    pub eta: f64,
    /// Effective body-bias sensitivity of ETA (1/V)
    pub etab: f64,
    /// Effective drain-bias sensitivity of ETA (1/V)
    pub etad: f64,

    // ========================================
    // Scaled Mobility Parameters (in m²/V·s)
    // ========================================
    /// Zero-bias mobility (m²/V·s) - converted from cm²/V·s
    pub mu_z: f64,
    /// Vbs dependence of mobility at Vds=0 (m²/V²·s)
    pub x2mz: f64,
    /// Mobility at Vds=Vdd (m²/V·s)
    pub mu_s: f64,
    /// Vbs dependence of MUS (m²/V²·s)
    pub x2ms: f64,
    /// Vds dependence of MUS (m²/V²·s)
    pub x3ms: f64,

    // ========================================
    // Scaled Gate-Field Degradation Parameters
    // ========================================
    /// Effective gate-field degradation U0 (m/V)
    pub u0: f64,
    /// Effective body-bias dependence of U0 (m/V²)
    pub u0b: f64,

    // ========================================
    // Scaled Velocity Saturation Parameters
    // ========================================
    /// Effective velocity saturation U1 (m/V)
    pub u1: f64,
    /// Effective body-bias dependence of U1 (m/V²)
    pub u1b: f64,
    /// Effective drain-bias dependence of U1 (m/V²)
    pub u1d: f64,

    // ========================================
    // Scaled Subthreshold Parameters
    // ========================================
    /// Effective subthreshold swing factor N0
    pub n0: f64,
    /// Effective body-bias sensitivity of N (1/V)
    pub nb: f64,
    /// Effective drain-bias sensitivity of N (1/V)
    pub nd: f64,

    // ========================================
    // Transconductance Parameters
    // ========================================
    /// Beta at Vds=0 (A/V²): beta_zero = mu_z * Cox * Weff / Leff
    pub beta_zero: f64,
    /// Beta at Vds=Vdd (A/V²): beta_vdd = mu_s * Cox * Weff / Leff
    pub beta_vdd: f64,

    // ========================================
    // Capacitance-related derived values
    // ========================================
    /// Total gate-source overlap capacitance (F)
    pub cgs_ov: f64,
    /// Total gate-drain overlap capacitance (F)
    pub cgd_ov: f64,
    /// Total gate-bulk overlap capacitance (F)
    pub cgb_ov: f64,
    /// Source diffusion area (m²) - auto-calculated if zero
    pub as_eff: f64,
    /// Drain diffusion area (m²) - auto-calculated if zero
    pub ad_eff: f64,
    /// Source diffusion perimeter (m) - auto-calculated if zero
    pub ps_eff: f64,
    /// Drain diffusion perimeter (m) - auto-calculated if zero
    pub pd_eff: f64,
}

impl Bsim1Derived {
    /// Compute derived parameters from model parameters.
    pub fn from_params(p: &Bsim1Params) -> Self {
        let leff = p.leff();
        let weff = p.weff();
        let cox = p.cox();
        let vt = p.vt();

        // Apply L/W scaling to all parameters
        let vfb = Bsim1Params::scaled_param(p.vfb0, p.vfb_l, p.vfb_w, leff, weff);
        let phi = Bsim1Params::scaled_param(p.phi0, p.phi_l, p.phi_w, leff, weff);
        let k1 = Bsim1Params::scaled_param(p.k10, p.k1_l, p.k1_w, leff, weff);
        let k2 = Bsim1Params::scaled_param(p.k20, p.k2_l, p.k2_w, leff, weff);

        let eta = Bsim1Params::scaled_param(p.eta0, p.eta_l, p.eta_w, leff, weff);
        let etab = Bsim1Params::scaled_param(p.etab0, p.etab_l, p.etab_w, leff, weff);
        let etad = Bsim1Params::scaled_param(p.etad0, p.etad_l, p.etad_w, leff, weff);

        // Mobility parameters - convert cm²/V·s to m²/V·s
        let mu_z = p.muz * 1e-4;
        let x2mz = Bsim1Params::scaled_param(p.x2mz0, p.x2mz_l, p.x2mz_w, leff, weff) * 1e-4;
        let mu_s = Bsim1Params::scaled_param(p.mus0, p.mus_l, p.mus_w, leff, weff) * 1e-4;
        let x2ms = Bsim1Params::scaled_param(p.x2ms0, p.x2ms_l, p.x2ms_w, leff, weff) * 1e-4;
        let x3ms = Bsim1Params::scaled_param(p.x3ms0, p.x3ms_l, p.x3ms_w, leff, weff) * 1e-4;

        let u0 = Bsim1Params::scaled_param(p.u00, p.u0_l, p.u0_w, leff, weff);
        let u0b = Bsim1Params::scaled_param(p.u0b0, p.u0b_l, p.u0b_w, leff, weff);

        let u1 = Bsim1Params::scaled_param(p.u10, p.u1_l, p.u1_w, leff, weff);
        let u1b = Bsim1Params::scaled_param(p.u1b0, p.u1b_l, p.u1b_w, leff, weff);
        let u1d = Bsim1Params::scaled_param(p.u1d0, p.u1d_l, p.u1d_w, leff, weff);

        let n0 = Bsim1Params::scaled_param(p.n00, p.n0_l, p.n0_w, leff, weff);
        let nb = Bsim1Params::scaled_param(p.nb0, p.nb_l, p.nb_w, leff, weff);
        let nd = Bsim1Params::scaled_param(p.nd0, p.nd_l, p.nd_w, leff, weff);

        // Beta calculations
        // Beta = mu * Cox * W / L
        let beta_zero = mu_z * cox * weff / leff;
        let beta_vdd = mu_s * cox * weff / leff;

        // Overlap capacitances
        let cgs_ov = p.cgso * weff;
        let cgd_ov = p.cgdo * weff;
        let cgb_ov = p.cgbo * leff;

        // Source/drain diffusion areas and perimeters
        // If not specified (zero), use typical estimates based on width
        let diff_length = 0.5e-6; // 0.5um default diffusion length
        let as_eff = if p.as_ > 0.0 {
            p.as_
        } else {
            weff * diff_length
        };
        let ad_eff = if p.ad > 0.0 {
            p.ad
        } else {
            weff * diff_length
        };
        let ps_eff = if p.ps > 0.0 {
            p.ps
        } else {
            2.0 * (weff + diff_length)
        };
        let pd_eff = if p.pd > 0.0 {
            p.pd
        } else {
            2.0 * (weff + diff_length)
        };

        Self {
            leff,
            weff,
            cox,
            vt,
            vfb,
            phi,
            k1,
            k2,
            eta,
            etab,
            etad,
            mu_z,
            x2mz,
            mu_s,
            x2ms,
            x3ms,
            u0,
            u0b,
            u1,
            u1b,
            u1d,
            n0,
            nb,
            nd,
            beta_zero,
            beta_vdd,
            cgs_ov,
            cgd_ov,
            cgb_ov,
            as_eff,
            ad_eff,
            ps_eff,
            pd_eff,
        }
    }

    /// Calculate junction capacitance with voltage dependence.
    ///
    /// Cj(V) = Cj0 / (1 - V/Pb)^Mj  for V < fc*Pb
    /// Cj(V) = Cj0 * (1 + Mj*(V - fc*Pb)/(Pb*(1-fc)^(1+Mj)))  for V >= fc*Pb
    ///
    /// where fc is the forward-bias capacitance coefficient (typically 0.5)
    pub fn junction_cap(cj0: f64, v: f64, pb: f64, mj: f64) -> f64 {
        const FC: f64 = 0.5; // Forward-bias coefficient

        if cj0 <= 0.0 {
            return 0.0;
        }

        // Limit voltage to avoid numerical issues
        let v = v.min(pb * 0.95);

        if v < FC * pb {
            // Reverse bias or small forward bias
            let ratio = 1.0 - v / pb;
            cj0 / ratio.powf(mj)
        } else {
            // Strong forward bias - linear extrapolation
            let f1 = (1.0 - FC).powf(1.0 + mj);
            let f2 = 1.0 + mj;
            let f3 = 1.0 - FC * (1.0 + mj);
            cj0 / f1 * (f3 + mj * v / pb / f2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derived_nmos() {
        let params = Bsim1Params::nmos_default();
        let derived = Bsim1Derived::from_params(&params);

        // Check effective dimensions
        assert!((derived.leff - 1e-6).abs() < 1e-12);
        assert!((derived.weff - 10e-6).abs() < 1e-12);

        // Check thermal voltage (~25.9mV at 300K)
        assert!((derived.vt - 0.0259).abs() < 0.001);

        // Check oxide capacitance
        assert!(derived.cox > 1e-3 && derived.cox < 2e-3);

        // Check mobility conversion (cm²/V·s to m²/V·s)
        assert!((derived.mu_z - 600.0 * 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_derived_with_dl_dw() {
        let mut params = Bsim1Params::nmos_default();
        params.l = 2e-6;
        params.w = 20e-6;
        params.dl = 0.2e-6;
        params.dw = 0.4e-6;

        let derived = Bsim1Derived::from_params(&params);

        // Leff = 2um - 0.2um = 1.8um
        assert!((derived.leff - 1.8e-6).abs() < 1e-12);
        // Weff = 20um - 0.4um = 19.6um
        assert!((derived.weff - 19.6e-6).abs() < 1e-12);
    }

    #[test]
    fn test_scaled_parameters() {
        let mut params = Bsim1Params::nmos_default();
        params.vfb0 = -0.5;
        params.vfb_l = 0.5e-6; // Will add 0.5 when Leff=1um
        params.vfb_w = 5e-6; // Will add 0.5 when Weff=10um

        let derived = Bsim1Derived::from_params(&params);

        // VFB = -0.5 + 0.5e-6/1e-6 + 5e-6/10e-6 = -0.5 + 0.5 + 0.5 = 0.5
        assert!((derived.vfb - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_beta_calculation() {
        let params = Bsim1Params::nmos_default();
        let derived = Bsim1Derived::from_params(&params);

        // beta_zero = mu_z * Cox * Weff / Leff
        // With mu_z=600 cm²/V·s = 0.06 m²/V·s
        // Cox ≈ 1.38e-3 F/m², Weff=10um, Leff=1um
        // beta_zero ≈ 0.06 * 1.38e-3 * 10e-6 / 1e-6 ≈ 8.28e-4 A/V²
        assert!(derived.beta_zero > 1e-4 && derived.beta_zero < 1e-3);
    }

    #[test]
    fn test_junction_capacitance() {
        // Test reverse bias
        let cj_rev = Bsim1Derived::junction_cap(1e-15, -1.0, 0.8, 0.5);
        // With reverse bias, capacitance decreases
        assert!(cj_rev < 1e-15);

        // Test zero bias
        let cj_zero = Bsim1Derived::junction_cap(1e-15, 0.0, 0.8, 0.5);
        assert!((cj_zero - 1e-15).abs() < 1e-20);

        // Test forward bias (should increase)
        let cj_fwd = Bsim1Derived::junction_cap(1e-15, 0.3, 0.8, 0.5);
        assert!(cj_fwd > 1e-15);
    }
}
