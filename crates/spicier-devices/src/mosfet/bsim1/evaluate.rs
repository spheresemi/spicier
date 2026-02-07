//! BSIM1 device evaluation.
//!
//! This module implements the core BSIM1 DC equations including:
//! - Threshold voltage with DIBL
//! - Mobility interpolation between Vds=0 and Vds=Vdd
//! - Gate-field mobility degradation
//! - Velocity saturation
//! - Subthreshold current model
//!
//! The BSIM1 model is based on empirical polynomial equations derived from
//! measured data, rather than physical device equations like BSIM3.

use super::super::level1::MosfetType;
use super::derived::Bsim1Derived;
use super::params::Bsim1Params;

/// Operating region of the BSIM1 MOSFET.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bsim1Region {
    /// Cutoff (Vgs < Vth)
    Cutoff,
    /// Subthreshold (weak inversion)
    Subthreshold,
    /// Linear (triode) region
    Linear,
    /// Saturation region
    Saturation,
}

/// Result of BSIM1 device evaluation.
#[derive(Debug, Clone)]
pub struct Bsim1EvalResult {
    /// Drain current (A)
    pub ids: f64,
    /// Output conductance dIds/dVds (S)
    pub gds: f64,
    /// Transconductance dIds/dVgs (S)
    pub gm: f64,
    /// Body transconductance dIds/dVbs (S)
    pub gmbs: f64,
    /// Operating region
    pub region: Bsim1Region,
    /// Effective threshold voltage (V)
    pub vth: f64,
    /// Saturation voltage Vdsat (V)
    pub vdsat: f64,
}

/// Evaluate BSIM1 MOSFET drain current and conductances.
///
/// # Arguments
/// * `params` - Model parameters
/// * `derived` - Pre-calculated derived parameters
/// * `vgs` - Gate-source voltage (V)
/// * `vds` - Drain-source voltage (V)
/// * `vbs` - Bulk-source voltage (V)
///
/// # Returns
/// Evaluation result containing Ids, gm, gds, gmbs, and operating region.
pub fn evaluate(
    params: &Bsim1Params,
    derived: &Bsim1Derived,
    vgs: f64,
    vds: f64,
    vbs: f64,
) -> Bsim1EvalResult {
    // Handle PMOS by flipping voltages (work in absolute values)
    let (vgs, vds, vbs, sign) = match params.mos_type {
        MosfetType::Nmos => (vgs, vds, vbs, 1.0),
        MosfetType::Pmos => (-vgs, -vds, -vbs, -1.0),
    };

    // Ensure non-negative Vds (source-drain swap for reverse bias)
    let (vgs, vds, vbs, swap) = if vds < 0.0 {
        (vgs - vds, -vds, vbs - vds, true)
    } else {
        (vgs, vds, vbs, false)
    };

    // Calculate threshold voltage with DIBL
    let vth = calc_threshold(params, derived, vds, vbs);

    // Gate overdrive voltage
    let vgst = vgs - vth;

    // Check for cutoff/subthreshold
    if vgst < 0.0 {
        // Subthreshold region - exponential current
        let (ids_sub, gm_sub, gds_sub, gmbs_sub) =
            calc_subthreshold(params, derived, vgs, vds, vbs, vth, vgst);

        return Bsim1EvalResult {
            ids: sign * ids_sub,
            gds: gds_sub.max(1e-12),
            gm: if swap { 0.0 } else { gm_sub } * sign.abs(),
            gmbs: gmbs_sub * sign.abs(),
            region: if vgst < -3.0 * derived.vt {
                Bsim1Region::Cutoff
            } else {
                Bsim1Region::Subthreshold
            },
            vth: sign * vth,
            vdsat: 0.0,
        };
    }

    // Calculate effective mobility with bias dependence
    let (beta, dbeta_dvgs, dbeta_dvds, dbeta_dvbs) =
        calc_mobility_and_beta(params, derived, vgs, vds, vbs, vgst);

    // Calculate body effect factor A
    let (a_factor, da_dvbs) = calc_body_effect_factor(derived, vbs);

    // Calculate saturation voltage Vdsat
    let (vdsat, dvdsat_dvgs, dvdsat_dvbs) = calc_vdsat(derived, vgst, a_factor);

    // Determine operating region and calculate current
    let (ids, gm, gds, gmbs, region) = if vds < vdsat {
        // Linear region
        calc_linear(
            derived, vgst, vds, vbs, beta, a_factor, vdsat, dbeta_dvgs, dbeta_dvds, dbeta_dvbs,
            da_dvbs, dvdsat_dvgs, dvdsat_dvbs,
        )
    } else {
        // Saturation region
        calc_saturation(
            derived, vgst, vds, vbs, beta, a_factor, vdsat, dbeta_dvgs, dbeta_dvds, dbeta_dvbs,
            da_dvbs, dvdsat_dvgs, dvdsat_dvbs,
        )
    };

    // Handle source-drain swap
    let (gm, gds) = if swap { (gm - gds, gds) } else { (gm, gds) };

    Bsim1EvalResult {
        ids: sign * ids,
        gds: gds.max(1e-12),
        gm: gm * sign.abs(),
        gmbs: gmbs * sign.abs(),
        region,
        vth: sign * vth,
        vdsat,
    }
}

/// Calculate threshold voltage with DIBL effect.
///
/// BSIM1 threshold voltage:
///   Von = VFB + PHI + K1*sqrt(PHI - Vbs) - K2*(PHI - Vbs) - ETA_eff*Vds
///
/// where ETA_eff = ETA + ETAB*Vbs + ETAD*Vds
fn calc_threshold(_params: &Bsim1Params, derived: &Bsim1Derived, vds: f64, vbs: f64) -> f64 {
    let d = derived;

    // Body effect term
    let phi_vbs = (d.phi - vbs).max(0.01);
    let sqrt_phi_vbs = phi_vbs.sqrt();
    let sqrt_phi = d.phi.max(0.01).sqrt();

    // Basic threshold: VFB + PHI + K1*(sqrt(PHI-Vbs) - sqrt(PHI)) - K2*(PHI-Vbs - PHI)
    let body_effect = d.k1 * (sqrt_phi_vbs - sqrt_phi) - d.k2 * (-vbs);

    // DIBL: ETA_eff * Vds reduces threshold at higher Vds
    // ETA_eff = ETA + ETAB*Vbs + ETAD*Vds
    let eta_eff = d.eta + d.etab * vbs + d.etad * vds;
    let dibl = eta_eff * vds;

    // Total threshold
    let vfb = d.vfb;
    let vth0 = vfb + d.phi + d.k1 * sqrt_phi;

    vth0 + body_effect - dibl
}

/// Calculate subthreshold current (weak inversion).
///
/// In subthreshold, current follows exponential behavior:
///   Isub = I0 * exp((Vgs - Vth) / (n * Vt)) * (1 - exp(-Vds/Vt))
fn calc_subthreshold(
    _params: &Bsim1Params,
    derived: &Bsim1Derived,
    _vgs: f64,
    vds: f64,
    vbs: f64,
    _vth: f64,
    vgst: f64,
) -> (f64, f64, f64, f64) {
    let d = derived;

    // Subthreshold swing factor: n = N0 + NB*Vbs + ND*Vds
    let n = (d.n0 + d.nb * vbs + d.nd * vds).max(1.0);

    // Thermal voltage times n
    let nvt = n * d.vt;

    // Subthreshold current: Ids = I0 * exp(Vgst / (n * Vt)) * (1 - exp(-Vds/Vt))
    // I0 is approximately W/L * mu * Cox * (n * Vt)^2
    let i0 = d.beta_zero * nvt * nvt;

    // Exponential factor (limited to avoid overflow)
    let exp_factor = (vgst / nvt).clamp(-40.0, 20.0).exp();

    // Drain voltage factor
    let vds_factor = 1.0 - (-vds / d.vt).exp();

    let ids = i0 * exp_factor * vds_factor;

    // Derivatives
    let gm = ids / nvt;
    let gds = i0 * exp_factor * (-vds / d.vt).exp() / d.vt + ids * 0.01; // Add small CLM

    // Body transconductance (through n dependence)
    let dn_dvbs = d.nb;
    let gmbs = -ids * dn_dvbs * vgst / (n * nvt);

    (ids.max(0.0), gm.max(0.0), gds.max(1e-12), gmbs.abs())
}

/// Calculate effective mobility and beta with all bias dependencies.
///
/// BSIM1 uses quadratic interpolation for mobility between Vds=0 and Vds=Vdd:
///   Beta(Vds) = Beta0 + (BetaVdd - Beta0) * (3 - 2*Vds/Vdd) * (Vds/Vdd)^2
///
/// Then apply gate-field degradation:
///   Beta_eff = Beta / (1 + U0*Eeff + U0B*Eeff*Vbs)
///
/// where Eeff = (Vgs - Vth) / Tox
#[allow(clippy::type_complexity)]
fn calc_mobility_and_beta(
    params: &Bsim1Params,
    derived: &Bsim1Derived,
    _vgs: f64,
    vds: f64,
    vbs: f64,
    vgst: f64,
) -> (f64, f64, f64, f64) {
    let d = derived;
    let p = params;

    // Bias-dependent mobility at Vds=0
    // mu_zero = MUZ + X2MZ * Vbs
    let mu_zero = d.mu_z + d.x2mz * vbs;

    // Bias-dependent mobility at Vds=Vdd
    // mu_vdd = MUS + X2MS * Vbs + X3MS * Vds
    let mu_vdd = d.mu_s + d.x2ms * vbs + d.x3ms * vds;

    // Quadratic interpolation for mobility between Vds=0 and Vds=Vdd
    // Factor x = Vds / Vdd (clamped to [0, 1])
    let x = (vds / p.vdd).clamp(0.0, 1.0);
    let x2 = x * x;
    let interp = (3.0 - 2.0 * x) * x2;
    let mu_interp = mu_zero + (mu_vdd - mu_zero) * interp;

    // Calculate beta from interpolated mobility
    // beta = mu * Cox * W / L
    let beta_base = mu_interp * d.cox * d.weff / d.leff;

    // Gate-field mobility degradation
    // Eeff = (Vgs - Vth) / Tox (effective vertical field)
    let eeff = vgst / p.tox;

    // Degradation factor: 1 + (U0 + U0B*Vbs) * Eeff
    let u0_eff = d.u0 + d.u0b * vbs;
    let degradation = 1.0 + u0_eff * eeff;

    let beta = beta_base / degradation.max(0.1);

    // Derivatives for Newton-Raphson
    // dbeta/dvgs = -beta * u0_eff / (Tox * degradation)
    let dbeta_dvgs = -beta * u0_eff / (p.tox * degradation);

    // dbeta/dvds comes from mobility interpolation
    let dinterp_dx = 6.0 * x * (1.0 - x);
    let dmu_dvds = (mu_vdd - mu_zero) * dinterp_dx / p.vdd + d.x3ms * interp;
    let dbeta_dvds = (dmu_dvds * d.cox * d.weff / d.leff) / degradation;

    // dbeta/dvbs from mu_zero, mu_vdd, and u0b terms
    let dmu_dvbs = d.x2mz * (1.0 - interp) + d.x2ms * interp;
    let dbeta_base_dvbs = dmu_dvbs * d.cox * d.weff / d.leff;
    let ddeg_dvbs = d.u0b * eeff;
    let dbeta_dvbs = (dbeta_base_dvbs - beta_base * ddeg_dvbs / degradation) / degradation;

    (beta, dbeta_dvgs, dbeta_dvds, dbeta_dvbs)
}

/// Calculate body effect factor A.
///
/// A = 1 + 0.5 * G * K1 / sqrt(PHI - Vbs)
///
/// where G is a geometry-dependent factor (simplified to 1 here)
fn calc_body_effect_factor(derived: &Bsim1Derived, vbs: f64) -> (f64, f64) {
    let d = derived;

    let phi_vbs = (d.phi - vbs).max(0.01);
    let sqrt_phi_vbs = phi_vbs.sqrt();

    // G factor (simplified - could be expanded for better accuracy)
    let g = 1.0;

    let a = 1.0 + 0.5 * g * d.k1 / sqrt_phi_vbs;

    // da/dvbs = 0.5 * G * K1 * 0.5 / (PHI - Vbs)^1.5
    let da_dvbs = 0.25 * g * d.k1 / (phi_vbs * sqrt_phi_vbs);

    (a, da_dvbs)
}

/// Calculate saturation voltage Vdsat.
///
/// BSIM1 uses velocity saturation model:
///   Vdsat = Vgst / (A * sqrt(1 + (Vgst / (A * Esat * Leff))^2))
///
/// Simplified for when U1 ≈ 0:
///   Vdsat ≈ Vgst / A
fn calc_vdsat(derived: &Bsim1Derived, vgst: f64, a_factor: f64) -> (f64, f64, f64) {
    let d = derived;
    let vgst_eff = vgst.max(0.001);

    // Include velocity saturation if U1 is non-zero
    // Ksat = 1 + U1 * Vgst / Leff
    let u1_eff = d.u1 / d.leff;
    let ksat = 1.0 + u1_eff * vgst_eff;

    // Vdsat = Vgst / (A * sqrt(Ksat))
    let vdsat = vgst_eff / (a_factor * ksat.sqrt());

    // dvdsat/dvgs
    let dksat_dvgs = u1_eff;
    let dvdsat_dvgs = 1.0 / (a_factor * ksat.sqrt())
        - 0.5 * vgst_eff * dksat_dvgs / (a_factor * ksat.powf(1.5));

    // dvdsat/dvbs (through A factor)
    let dvdsat_dvbs = 0.0; // Simplified - could include da/dvbs effect

    (vdsat.max(0.001), dvdsat_dvgs, dvdsat_dvbs)
}

/// Calculate linear region current and conductances.
#[allow(clippy::too_many_arguments)]
fn calc_linear(
    derived: &Bsim1Derived,
    vgst: f64,
    vds: f64,
    vbs: f64,
    beta: f64,
    a_factor: f64,
    _vdsat: f64,
    dbeta_dvgs: f64,
    dbeta_dvds: f64,
    dbeta_dvbs: f64,
    _da_dvbs: f64,
    _dvdsat_dvgs: f64,
    _dvdsat_dvbs: f64,
) -> (f64, f64, f64, f64, Bsim1Region) {
    let d = derived;

    // Velocity saturation factor in linear region
    // Uds = 1 + (U1 + U1B*Vbs + U1D*Vds) * Vds / Leff
    let u1_total = d.u1 + d.u1b * vbs + d.u1d * vds;
    let uds = 1.0 + u1_total * vds / d.leff;

    // Linear region current:
    // Ids = Beta * (Vgst - 0.5*A*Vds) * Vds / Uds
    let veff = (vgst - 0.5 * a_factor * vds).max(0.001);
    let ids = beta * veff * vds / uds;

    // Transconductance: gm = dIds/dVgs
    let gm = beta * vds / uds + dbeta_dvgs * veff * vds / uds;

    // Output conductance: gds = dIds/dVds
    let duds_dvds = (u1_total + d.u1d * vds) / d.leff;
    let gds = beta * (vgst - a_factor * vds) / uds - ids * duds_dvds / uds + dbeta_dvds * veff * vds / uds;

    // Body transconductance: gmbs = dIds/dVbs
    let duds_dvbs = d.u1b * vds / d.leff;
    let gmbs = (dbeta_dvbs * veff * vds - ids * duds_dvbs) / uds;

    (
        ids.max(0.0),
        gm.max(0.0),
        gds.max(1e-12),
        gmbs.abs(),
        Bsim1Region::Linear,
    )
}

/// Calculate saturation region current and conductances.
#[allow(clippy::too_many_arguments)]
fn calc_saturation(
    derived: &Bsim1Derived,
    vgst: f64,
    vds: f64,
    vbs: f64,
    beta: f64,
    a_factor: f64,
    vdsat: f64,
    dbeta_dvgs: f64,
    _dbeta_dvds: f64,
    dbeta_dvbs: f64,
    _da_dvbs: f64,
    dvdsat_dvgs: f64,
    _dvdsat_dvbs: f64,
) -> (f64, f64, f64, f64, Bsim1Region) {
    let d = derived;

    // Velocity saturation factor at Vdsat
    let u1_total = d.u1 + d.u1b * vbs + d.u1d * vdsat;
    let uds = 1.0 + u1_total * vdsat / d.leff;

    // Saturation current:
    // Ids_sat = 0.5 * Beta * Vgst^2 / (A * Uds)
    let ids_sat = 0.5 * beta * vgst * vgst / (a_factor * uds);

    // Channel length modulation (simple model)
    // Va = early voltage, approximated based on device length
    let va = 10.0 * d.leff / 1e-6; // ~10V per um
    let clm = 1.0 + (vds - vdsat).max(0.0) / va;

    let ids = ids_sat * clm;

    // Transconductance: gm = dIds/dVgs (in saturation)
    let duds_dvgst = d.u1 / d.leff * dvdsat_dvgs;
    let gm = beta * vgst / (a_factor * uds) * clm
        + dbeta_dvgs * 0.5 * vgst * vgst / (a_factor * uds) * clm
        - ids_sat * duds_dvgst / uds * clm;

    // Output conductance: gds = Ids_sat / Va (CLM effect)
    let gds = ids_sat / va;

    // Body transconductance: gmbs
    let duds_dvbs = d.u1b * vdsat / d.leff;
    let gmbs = (dbeta_dvbs * 0.5 * vgst * vgst - ids_sat * duds_dvbs) / (a_factor * uds) * clm;

    (
        ids.max(0.0),
        gm.max(0.0),
        gds.max(1e-12),
        gmbs.abs(),
        Bsim1Region::Saturation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_nmos() -> (Bsim1Params, Bsim1Derived) {
        let params = Bsim1Params::nmos_default();
        let derived = Bsim1Derived::from_params(&params);
        (params, derived)
    }

    #[test]
    fn test_cutoff() {
        let (params, derived) = setup_nmos();

        // Very low Vgs should be cutoff
        let result = evaluate(&params, &derived, 0.0, 1.0, 0.0);

        assert!(
            matches!(result.region, Bsim1Region::Cutoff | Bsim1Region::Subthreshold),
            "Expected cutoff/subthreshold at Vgs=0"
        );
        assert!(
            result.ids.abs() < 1e-6,
            "Current should be very small in cutoff"
        );
    }

    #[test]
    fn test_subthreshold() {
        let (params, derived) = setup_nmos();

        // Just below threshold (within 3*Vt to be in subthreshold, not cutoff)
        // Vt ~ 0.026V, so 3*Vt ~ 0.078V. Use 0.05V below Vth.
        let vth_approx = derived.vfb + derived.phi + derived.k1 * derived.phi.sqrt();
        let vgs = vth_approx - 0.05;
        let result = evaluate(&params, &derived, vgs, 0.5, 0.0);

        assert_eq!(result.region, Bsim1Region::Subthreshold);
        assert!(result.ids > 0.0, "Subthreshold current should be positive");
        assert!(
            result.ids < 1e-5,
            "Subthreshold current should be small: {}",
            result.ids
        );
    }

    #[test]
    fn test_saturation() {
        let (params, derived) = setup_nmos();

        // Vgs = 2V, Vds = 2V should be in saturation
        let result = evaluate(&params, &derived, 2.0, 2.0, 0.0);

        assert_eq!(result.region, Bsim1Region::Saturation);
        assert!(result.ids > 0.0);
        assert!(result.gm > 0.0);
        assert!(result.gds > 0.0);
    }

    #[test]
    fn test_linear() {
        let (params, derived) = setup_nmos();

        // Vgs = 2V, Vds = 0.2V should be in linear
        let result = evaluate(&params, &derived, 2.0, 0.2, 0.0);

        assert_eq!(result.region, Bsim1Region::Linear);
        assert!(result.ids > 0.0);
        assert!(result.gds > result.gm * 0.1, "High gds expected in linear region");
    }

    #[test]
    fn test_body_effect() {
        let (params, derived) = setup_nmos();

        // Test threshold voltage shift with body bias
        let result_vbs0 = evaluate(&params, &derived, 2.0, 2.0, 0.0);
        let result_vbs_neg = evaluate(&params, &derived, 2.0, 2.0, -1.0);

        // Negative Vbs should increase Vth, reducing current
        assert!(
            result_vbs_neg.vth > result_vbs0.vth,
            "Vth should increase with negative Vbs"
        );
        assert!(
            result_vbs_neg.ids < result_vbs0.ids,
            "Current should decrease with negative Vbs"
        );
    }

    #[test]
    fn test_pmos() {
        let params = Bsim1Params::pmos_default();
        let derived = Bsim1Derived::from_params(&params);

        // PMOS: Vgs = -2V, Vds = -2V -> saturation
        let result = evaluate(&params, &derived, -2.0, -2.0, 0.0);

        assert_eq!(result.region, Bsim1Region::Saturation);
        assert!(
            result.ids < 0.0,
            "PMOS Ids should be negative: {}",
            result.ids
        );
    }

    #[test]
    fn test_continuity_at_vdsat() {
        let (params, derived) = setup_nmos();

        // Test current continuity at Vdsat boundary
        let vgs = 2.0;

        // Evaluate at a point known to be in saturation to get Vdsat
        let result_sat = evaluate(&params, &derived, vgs, 2.0, 0.0);
        let vdsat = result_sat.vdsat;

        // Evaluate just below and just above Vdsat
        let result_below = evaluate(&params, &derived, vgs, vdsat * 0.95, 0.0);
        let result_above = evaluate(&params, &derived, vgs, vdsat * 1.05, 0.0);

        assert_eq!(result_below.region, Bsim1Region::Linear);
        assert_eq!(result_above.region, Bsim1Region::Saturation);

        // Current should be reasonably continuous (within 30%)
        let ratio = result_above.ids / result_below.ids;
        assert!(
            ratio > 0.7 && ratio < 1.3,
            "Current discontinuity at Vdsat: {} vs {} (ratio={})",
            result_below.ids,
            result_above.ids,
            ratio
        );
    }

    #[test]
    fn test_dibl() {
        let mut params = Bsim1Params::nmos_default();
        params.eta0 = 0.1; // Enable DIBL
        let derived = Bsim1Derived::from_params(&params);

        // DIBL: higher Vds should reduce Vth
        let result_low_vds = evaluate(&params, &derived, 1.0, 0.1, 0.0);
        let result_high_vds = evaluate(&params, &derived, 1.0, 2.0, 0.0);

        // With DIBL, threshold should be lower at high Vds
        assert!(
            result_high_vds.vth < result_low_vds.vth,
            "DIBL should reduce Vth: low_vds={}, high_vds={}",
            result_low_vds.vth,
            result_high_vds.vth
        );
    }

    #[test]
    fn test_positive_conductances() {
        let (params, derived) = setup_nmos();

        // All conductances should be positive in normal operation
        for vgs in [1.0, 1.5, 2.0, 2.5] {
            for vds in [0.1, 0.5, 1.0, 2.0] {
                let result = evaluate(&params, &derived, vgs, vds, 0.0);

                assert!(result.gm >= 0.0, "gm should be non-negative");
                assert!(result.gds > 0.0, "gds should be positive");
                assert!(result.gmbs >= 0.0, "gmbs should be non-negative");
            }
        }
    }
}
