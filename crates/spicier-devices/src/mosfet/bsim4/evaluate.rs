//! BSIM4 device evaluation.
//!
//! This module implements the core BSIM4 DC equations including:
//! - All BSIM3-compatible physics (SCE, DIBL, mobility, velocity saturation, CLM)
//! - Quantum mechanical corrections (poly depletion, inversion quantization)
//! - Stress effects on Vth and mobility
//! - Gate-induced drain leakage (IGIDL/IGISL)

use super::super::level1::MosfetType;
use super::derived::Bsim4Derived;
use super::params::Bsim4Params;

/// Operating region of the BSIM4 MOSFET.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bsim4Region {
    /// Subthreshold (weak inversion)
    Subthreshold,
    /// Linear (triode) region
    Linear,
    /// Saturation region
    Saturation,
}

/// Result of BSIM4 device evaluation.
#[derive(Debug, Clone)]
pub struct Bsim4EvalResult {
    /// Drain current (A)
    pub ids: f64,
    /// Output conductance dIds/dVds (S)
    pub gds: f64,
    /// Transconductance dIds/dVgs (S)
    pub gm: f64,
    /// Body transconductance dIds/dVbs (S)
    pub gmbs: f64,
    /// Operating region
    pub region: Bsim4Region,
    /// Effective threshold voltage (V)
    pub vth: f64,
    /// Saturation voltage Vdsat (V)
    pub vdsat: f64,
    /// Effective mobility (m^2/V-s)
    pub ueff: f64,
    /// Substrate current from impact ionization (A)
    pub isub: f64,
    /// Gate-induced drain leakage current (A)
    pub igidl: f64,
    /// Gate-induced source leakage current (A)
    pub igisl: f64,
}

/// Result of BSIM4 capacitance evaluation.
#[derive(Debug, Clone)]
pub struct Bsim4CapResult {
    /// Gate-source capacitance (F) - intrinsic + overlap.
    pub cgs: f64,
    /// Gate-drain capacitance (F) - intrinsic + overlap.
    pub cgd: f64,
    /// Gate-bulk capacitance (F) - overlap only.
    pub cgb: f64,
    /// Bulk-source junction capacitance (F).
    pub cbs: f64,
    /// Bulk-drain junction capacitance (F).
    pub cbd: f64,
}

/// Evaluate BSIM4 MOSFET capacitances at given terminal voltages.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_capacitances(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vgs: f64,
    vds: f64,
    vbs: f64,
    region: Bsim4Region,
    _vth: f64,
    _vdsat: f64,
) -> Bsim4CapResult {
    // Handle PMOS by working with absolute voltages
    let (_vgs, vds, vbs) = match params.mos_type {
        MosfetType::Nmos => (vgs, vds, vbs),
        MosfetType::Pmos => (-vgs, -vds, -vbs),
    };

    // Total gate capacitance (intrinsic) using electrical oxide thickness
    let cox_total = derived.coxe * derived.weff * derived.leff;

    // Meyer's model for intrinsic capacitances
    let (cgs_int, cgd_int, cgb_int) = match region {
        Bsim4Region::Subthreshold => (0.0, 0.0, cox_total),
        Bsim4Region::Linear => {
            let c_half = cox_total / 2.0;
            (c_half, c_half, 0.0)
        }
        Bsim4Region::Saturation => {
            let cgs = cox_total * 2.0 / 3.0;
            (cgs, 0.0, 0.0)
        }
    };

    // Add overlap capacitances
    let cgs = cgs_int + derived.cgs_ov;
    let cgd = cgd_int + derived.cgd_ov;
    let cgb = cgb_int + derived.cgb_ov;

    // Junction capacitances
    let vbd = vbs - vds;

    let cbs_area =
        Bsim4Derived::junction_cap(params.cj * derived.as_eff, vbs, params.pb, params.mj);
    let cbs_sw =
        Bsim4Derived::junction_cap(params.cjsw * derived.ps_eff, vbs, params.pbsw, params.mjsw);
    let cbs_swg =
        Bsim4Derived::junction_cap(params.cjswg * derived.weff, vbs, params.pbswg, params.mjswg);
    let cbs = cbs_area + cbs_sw + cbs_swg;

    let cbd_area =
        Bsim4Derived::junction_cap(params.cj * derived.ad_eff, vbd, params.pb, params.mj);
    let cbd_sw =
        Bsim4Derived::junction_cap(params.cjsw * derived.pd_eff, vbd, params.pbsw, params.mjsw);
    let cbd_swg =
        Bsim4Derived::junction_cap(params.cjswg * derived.weff, vbd, params.pbswg, params.mjswg);
    let cbd = cbd_area + cbd_sw + cbd_swg;

    // Apply MULT scaling: M parallel devices
    let mult = params.mult;

    Bsim4CapResult {
        cgs: cgs * mult,
        cgd: cgd * mult,
        cgb: cgb * mult,
        cbs: cbs * mult,
        cbd: cbd * mult,
    }
}

/// Evaluate BSIM4 MOSFET drain current and conductances.
pub fn evaluate(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vgs: f64,
    vds: f64,
    vbs: f64,
) -> Bsim4EvalResult {
    // Handle PMOS by flipping voltages
    let (vgs, vds, vbs, sign) = match params.mos_type {
        MosfetType::Nmos => (vgs, vds, vbs, 1.0),
        MosfetType::Pmos => (-vgs, -vds, -vbs, -1.0),
    };

    // Ensure non-negative Vds (source-drain swap)
    let (vgs, vds, vbs, swap) = if vds < 0.0 {
        (vgs - vds, -vds, vbs - vds, true)
    } else {
        (vgs, vds, vbs, false)
    };

    // Calculate threshold voltage with BSIM4 effects
    let vth = calc_threshold(params, derived, vds, vbs);

    // Gate overdrive
    let vgst = vgs - vth;

    // Calculate IGIDL/IGISL (independent of operating region)
    let (igidl, igisl) = calc_igidl(params, derived, vgs, vds, vbs);

    // Check for subthreshold
    if vgst < 0.0 {
        let ids_sub = calc_subthreshold(params, derived, vds, vth, vgst);
        let gm_sub = ids_sub / (params.nfactor * derived.vt);
        let gds_sub = ids_sub * 0.01;

        return Bsim4EvalResult {
            ids: sign * ids_sub,
            gds: gds_sub.max(1e-12),
            gm: if swap { 0.0 } else { gm_sub } * sign.abs(),
            gmbs: 0.0,
            region: Bsim4Region::Subthreshold,
            vth: sign * vth,
            vdsat: 0.0,
            ueff: derived.u0_si * derived.stress_mu_factor,
            isub: 0.0,
            igidl: sign * igidl,
            igisl: sign * igisl,
        };
    }

    // Calculate effective mobility with QM and stress corrections
    let ueff = calc_mobility(params, derived, vgst, vbs);

    // Calculate Abulk
    let abulk = calc_abulk(params, derived, vbs);

    // Calculate saturation voltage
    let vdsat = calc_vdsat(params, derived, vgst, ueff, abulk);

    // Calculate early voltage
    let va = calc_early_voltage(params, derived, vds, vdsat, vbs, vgst);

    // Determine operating region and calculate current
    let (ids, gm, gds, gmbs, region) = if vds < vdsat {
        calc_linear(params, derived, vgst, vds, vbs, ueff, abulk, vdsat, va)
    } else {
        calc_saturation(params, derived, vgst, vds, vbs, ueff, abulk, vdsat, va)
    };

    // Substrate current from impact ionization
    let isub = if vds > vdsat && params.alpha0 > 0.0 {
        let vds_eff = (vds - vdsat).max(0.01);
        let exp_term = (-params.beta0 / vds_eff).exp();
        params.alpha0 / derived.leff * ids * vds_eff * exp_term
    } else {
        0.0
    };

    // Handle source-drain swap
    let (gm, gds) = if swap { (gm - gds, gds) } else { (gm, gds) };

    // Apply MULT scaling: M parallel devices
    let mult = params.mult;

    Bsim4EvalResult {
        ids: sign * ids * mult,
        gds: gds.max(1e-12) * mult,
        gm: gm * sign.abs() * mult,
        gmbs: gmbs * sign.abs() * mult,
        region,
        vth: sign * vth,
        vdsat,
        ueff,
        isub: sign.abs() * isub * mult,
        igidl: sign * igidl * mult,
        igisl: sign * igisl * mult,
    }
}

/// Calculate threshold voltage with BSIM4 effects.
///
/// Adds to BSIM3 base: quantum mechanical correction, poly depletion, stress shift.
fn calc_threshold(params: &Bsim4Params, derived: &Bsim4Derived, vds: f64, vbs: f64) -> f64 {
    let p = params;
    let d = derived;

    // Body effect: dVth = K1 * (sqrt(phi - Vbs) - sqrt(phi)) - K2 * Vbs
    let phi_vbs = (d.phi - vbs).max(0.01);
    let body_effect = p.k1 * (phi_vbs.sqrt() - d.sqrt_phi) - p.k2 * vbs;

    // Short-channel effect (same as BSIM3)
    let lt_ratio = p.dvt1 * d.leff / (2.0 * d.lt);
    let exp_term = (-lt_ratio.min(20.0)).exp();
    let theta = exp_term / (1.0 + 2.0 * exp_term);
    let dvth_sce = -p.dvt0 * theta * d.vt * (1.0 + p.dvt2 * vbs);

    // Narrow-width effect
    let dvth_nwe = if d.weff > 1e-9 {
        let tox_weff = p.toxe / d.weff;
        p.k3 * tox_weff * phi_vbs.sqrt() + p.k3b * vbs / d.weff
    } else {
        0.0
    };

    // Narrow-width SCE
    let wt = (Bsim4Params::EPS_SI * p.toxe / Bsim4Params::EPS_OX).sqrt();
    let wt_ratio = p.dvt1w * d.weff / (2.0 * wt + p.w0);
    let exp_w = (-wt_ratio.min(20.0)).exp();
    let theta_w = exp_w / (1.0 + 2.0 * exp_w);
    let dvth_nwe_sce = -p.dvt0w * theta_w * d.vt * (1.0 + p.dvt2w * vbs);

    // DIBL effect
    let dibl_lt_ratio = p.dsub * d.leff / (2.0 * d.lt);
    let theta_dibl = (-dibl_lt_ratio.min(20.0)).exp();
    let dvth_dibl = -(p.eta0 + p.etab * vbs) * vds * theta_dibl;

    // Temperature effect
    let dvth_temp = d.vth0_temp + p.kt2 * vbs * (d.temp_ratio - 1.0);

    // BSIM4-specific: Quantum mechanical correction
    let dvth_qm = d.dvth_qm;

    // BSIM4-specific: Poly depletion
    let dvth_poly = d.dvth_poly;

    // BSIM4-specific: Stress effect on Vth
    let dvth_stress = d.stress_vth_shift;

    // Total threshold voltage
    let vth0_abs = p.vth0.abs();
    vth0_abs + body_effect + dvth_sce + dvth_nwe + dvth_nwe_sce + dvth_dibl + dvth_temp + dvth_qm
        + dvth_poly
        + dvth_stress
}

/// Calculate subthreshold current.
fn calc_subthreshold(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vds: f64,
    _vth: f64,
    vgst: f64,
) -> f64 {
    let d = derived;
    let n = params.nfactor;
    let nvt = n * d.vt;

    // Apply stress factor to mobility
    let u0_eff = d.u0_si * d.stress_mu_factor;

    let i0 = (d.weff / d.leff) * u0_eff * d.coxe * nvt * nvt;
    let exp_factor = (vgst / nvt).min(20.0).exp();
    let vds_factor = 1.0 - (-vds / d.vt).exp();

    i0 * exp_factor * vds_factor
}

/// Calculate effective mobility with BSIM4 enhancements.
///
/// Adds stress-dependent mobility correction on top of BSIM3 mobility model.
fn calc_mobility(params: &Bsim4Params, derived: &Bsim4Derived, vgst: f64, vbs: f64) -> f64 {
    let p = params;
    let d = derived;

    let vgst_eff = vgst.max(0.01);
    let vth_eff = p.vth0.abs();

    // Effective vertical field (same as BSIM3)
    let eeff = (vgst_eff + 3.2 * vth_eff) / p.toxe;

    // Temperature-scaled mobility degradation
    let degradation = 1.0 + (d.ua_temp + d.uc_temp * vbs) * eeff + d.ub_temp * eeff * eeff;

    // Base mobility with degradation
    let u_base = d.u0_si / degradation.max(0.1);

    // Apply BSIM4 stress factor
    u_base * d.stress_mu_factor
}

/// Calculate bulk charge effect coefficient Abulk.
fn calc_abulk(params: &Bsim4Params, derived: &Bsim4Derived, vbs: f64) -> f64 {
    let p = params;
    let d = derived;

    let phi_vbs = (d.phi - vbs).max(0.01);
    let abulk_base = 1.0 + p.k1 / (2.0 * phi_vbs.sqrt()) - p.k2;

    let sce_factor = 1.0 - 0.3 * (d.lt / d.leff).min(1.0);
    let abulk = abulk_base * sce_factor;

    abulk.clamp(1.0, 2.5)
}

/// Calculate saturation voltage Vdsat.
fn calc_vdsat(
    _params: &Bsim4Params,
    derived: &Bsim4Derived,
    vgst: f64,
    ueff: f64,
    abulk: f64,
) -> f64 {
    let d = derived;

    let esat = 2.0 * d.vsat_temp / ueff;
    let esat_l = esat * d.leff;

    let vgst_eff = vgst.max(0.001);
    let vdsat = vgst_eff / (abulk + vgst_eff / esat_l);

    vdsat.max(0.001)
}

/// Calculate Early voltage for channel length modulation.
fn calc_early_voltage(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vds: f64,
    vdsat: f64,
    vbs: f64,
    vgst: f64,
) -> f64 {
    let p = params;
    let d = derived;

    let esat = 2.0 * p.vsat / d.u0_si;
    let va_base = d.leff * esat * 4.0 / (p.pclm.max(0.1));

    let pdiblc_eff = (p.pdiblc1 + p.pdiblcb * vbs).abs() + p.pdiblc2;
    let va_dibl_factor = 1.0 / (1.0 + pdiblc_eff * (vds / (vdsat + 0.1)));

    let drout_factor = 1.0 / (1.0 + p.drout * vds / (vgst + 0.1));
    let pvag_factor = 1.0 + p.pvag * vgst / (vdsat + 0.1);

    let va = va_base * va_dibl_factor * drout_factor * pvag_factor;

    va.clamp(0.5, 10.0)
}

/// Calculate IGIDL and IGISL (gate-induced drain/source leakage).
///
/// BSIM4 IGIDL model:
/// IGIDL = AGIDL * Weff * NF * (Vds - Vgs + EGIDL) * exp(-BGIDL / (Vds - Vgs + EGIDL))
/// Only significant when Vdg is large (drain at higher voltage than gate).
fn calc_igidl(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vgs: f64,
    vds: f64,
    _vbs: f64,
) -> (f64, f64) {
    if params.agidl == 0.0 {
        return (0.0, 0.0);
    }

    let weff_nf = derived.weff;

    // IGIDL: leakage at drain end
    // Triggered when Vdg = Vds - Vgs is positive and large
    let vdg = vds - vgs;
    let igidl = if vdg > 0.1 {
        let v_eff = (vdg + params.egidl).max(0.01);
        let exp_term = (-params.bgidl / v_eff).min(20.0).exp();
        params.agidl * weff_nf * v_eff * exp_term
    } else {
        0.0
    };

    // IGISL: leakage at source end (symmetric to IGIDL for Vsg)
    // For normal operation (Vds > 0), IGISL is typically negligible
    let igisl = 0.0;

    (igidl, igisl)
}

/// Calculate linear region current and conductances.
#[allow(clippy::too_many_arguments)]
fn calc_linear(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vgst: f64,
    vds: f64,
    vbs: f64,
    ueff: f64,
    abulk: f64,
    _vdsat: f64,
    va: f64,
) -> (f64, f64, f64, f64, Bsim4Region) {
    let d = derived;
    let vgst_eff = vgst.max(0.001);
    let beta = (d.weff / d.leff) * ueff * d.coxe;

    let esat = 2.0 * params.vsat / ueff;
    let esat_l = esat * d.leff;

    let vgst_eff_lin = (vgst_eff - abulk * vds / 2.0).max(0.001);
    let vs_factor = 1.0 + vds / esat_l;

    let ids_base = beta * vgst_eff_lin * vds / vs_factor;
    let ids = ids_base * (1.0 + vds / va * 0.1);

    let gm = beta * vds / vs_factor * (1.0 + vds / va * 0.1);

    let gds = beta * (vgst_eff - abulk * vds) / vs_factor
        + beta * vgst_eff_lin * vds / (vs_factor * esat_l * vs_factor)
        + ids_base / va * 0.1;

    let phi_vbs = (d.phi - vbs).max(0.01);
    let dvth_dvbs = params.k1 / (2.0 * phi_vbs.sqrt()) - params.k2;
    let gmbs = (gm * dvth_dvbs.abs() * 0.5).abs();

    (
        ids.max(0.0),
        gm.max(0.0),
        gds.max(1e-12),
        gmbs,
        Bsim4Region::Linear,
    )
}

/// Calculate saturation region current and conductances.
#[allow(clippy::too_many_arguments)]
fn calc_saturation(
    params: &Bsim4Params,
    derived: &Bsim4Derived,
    vgst: f64,
    vds: f64,
    vbs: f64,
    ueff: f64,
    abulk: f64,
    vdsat: f64,
    va: f64,
) -> (f64, f64, f64, f64, Bsim4Region) {
    let d = derived;
    let vgst_eff = vgst.max(0.001);
    let beta = (d.weff / d.leff) * ueff * d.coxe;

    let esat = 2.0 * params.vsat / ueff;
    let esat_l = esat * d.leff;

    let denom = 2.0 * (abulk + vgst_eff / esat_l);
    let ids_dsat = beta * vgst_eff * vgst_eff / denom;

    let clm_factor = 1.0 + (vds - vdsat).max(0.0) / va;
    let ids = ids_dsat * clm_factor;

    let denom_sq = (abulk + vgst_eff / esat_l).powi(2);
    let gm = beta * vgst_eff * (2.0 * abulk + vgst_eff / esat_l) / (2.0 * denom_sq) * clm_factor;

    let gds = ids_dsat / va;

    let phi_vbs = (d.phi - vbs).max(0.01);
    let dvth_dvbs = params.k1 / (2.0 * phi_vbs.sqrt()) - params.k2 - params.etab * vds;
    let gmbs = (gm * dvth_dvbs.abs()).abs();

    (
        ids.max(0.0),
        gm.max(0.0),
        gds.max(1e-12),
        gmbs,
        Bsim4Region::Saturation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_nmos() -> (Bsim4Params, Bsim4Derived) {
        let params = Bsim4Params::nmos_default();
        let derived = Bsim4Derived::from_params(&params);
        (params, derived)
    }

    #[test]
    fn test_subthreshold() {
        let (params, derived) = setup_nmos();

        let result_ref = evaluate(&params, &derived, 0.5, 0.1, 0.0);
        let vth_approx = result_ref.vth;

        let vgs_sub = (vth_approx - 0.2).min(0.1);
        let result = evaluate(&params, &derived, vgs_sub, 0.1, 0.0);

        assert_eq!(result.region, Bsim4Region::Subthreshold);
        assert!(result.ids > 0.0);
        assert!(result.ids < 1e-5);
    }

    #[test]
    fn test_saturation() {
        let (params, derived) = setup_nmos();

        let result = evaluate(&params, &derived, 1.0, 1.0, 0.0);

        assert_eq!(result.region, Bsim4Region::Saturation);
        assert!(result.ids > 0.0);
        assert!(result.gm > 0.0);
        assert!(result.gds > 0.0);
        assert!(result.ids > 10e-6);
        assert!(result.ids < 10e-3);
    }

    #[test]
    fn test_linear() {
        let (params, derived) = setup_nmos();

        let result = evaluate(&params, &derived, 1.0, 0.1, 0.0);

        assert_eq!(result.region, Bsim4Region::Linear);
        assert!(result.ids > 0.0);
        assert!(result.gds > result.gm * 0.1);
    }

    #[test]
    fn test_body_effect() {
        let (params, derived) = setup_nmos();

        let result_vbs0 = evaluate(&params, &derived, 1.0, 1.0, 0.0);
        let result_vbs_neg = evaluate(&params, &derived, 1.0, 1.0, -1.0);

        assert!(result_vbs_neg.vth > result_vbs0.vth);
        assert!(result_vbs_neg.ids < result_vbs0.ids);
    }

    #[test]
    fn test_pmos() {
        let params = Bsim4Params::pmos_default();
        let derived = Bsim4Derived::from_params(&params);

        let result = evaluate(&params, &derived, -1.0, -1.0, 0.0);

        assert_eq!(result.region, Bsim4Region::Saturation);
        assert!(result.ids < 0.0);
    }

    #[test]
    fn test_continuity_at_vdsat() {
        let (params, derived) = setup_nmos();

        let vgs = 1.0;
        let result_sat = evaluate(&params, &derived, vgs, 1.0, 0.0);
        let vdsat = result_sat.vdsat;

        let result_below = evaluate(&params, &derived, vgs, vdsat * 0.95, 0.0);
        let result_above = evaluate(&params, &derived, vgs, vdsat * 1.05, 0.0);

        assert_eq!(result_below.region, Bsim4Region::Linear);
        assert_eq!(result_above.region, Bsim4Region::Saturation);

        let ratio = result_above.ids / result_below.ids;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "Current discontinuity at Vdsat: {} vs {} (ratio={})",
            result_below.ids,
            result_above.ids,
            ratio
        );
    }

    #[test]
    fn test_short_channel_effect() {
        let mut params_short = Bsim4Params::nmos_default();
        params_short.l = 50e-9;

        let mut params_long = Bsim4Params::nmos_default();
        params_long.l = 500e-9;

        let derived_short = Bsim4Derived::from_params(&params_short);
        let derived_long = Bsim4Derived::from_params(&params_long);

        let result_short = evaluate(&params_short, &derived_short, 1.0, 0.1, 0.0);
        let result_long = evaluate(&params_long, &derived_long, 1.0, 0.1, 0.0);

        assert!(result_short.vth < result_long.vth);
    }

    #[test]
    fn test_dibl() {
        let (params, derived) = setup_nmos();

        let result_low_vds = evaluate(&params, &derived, 0.6, 0.1, 0.0);
        let result_high_vds = evaluate(&params, &derived, 0.6, 1.0, 0.0);

        assert!(result_high_vds.vth < result_low_vds.vth);
    }

    // ========================================
    // BSIM4-specific tests
    // ========================================

    #[test]
    fn test_quantum_effect_on_threshold() {
        let mut params_no_qm = Bsim4Params::nmos_default();
        params_no_qm.qme1 = 0.0;

        let mut params_qm = Bsim4Params::nmos_default();
        params_qm.qme1 = 0.5;

        let derived_no_qm = Bsim4Derived::from_params(&params_no_qm);
        let derived_qm = Bsim4Derived::from_params(&params_qm);

        let result_no_qm = evaluate(&params_no_qm, &derived_no_qm, 1.0, 1.0, 0.0);
        let result_qm = evaluate(&params_qm, &derived_qm, 1.0, 1.0, 0.0);

        // QM correction should increase Vth
        assert!(
            result_qm.vth > result_no_qm.vth,
            "QM should increase Vth: with_qm={:.4}V > no_qm={:.4}V",
            result_qm.vth,
            result_no_qm.vth
        );
    }

    #[test]
    fn test_stress_effect_on_current() {
        let mut params = Bsim4Params::nmos_default();
        params.sa = 0.3e-6; // Closer to gate -> more stress
        params.sb = 0.3e-6;
        params.saref = 1e-6;
        params.sbref = 1e-6;
        params.ku0 = 5e-7; // Positive: stress increases mobility

        let derived_stress = Bsim4Derived::from_params(&params);
        let result_stress = evaluate(&params, &derived_stress, 1.0, 1.0, 0.0);

        let params_no_stress = Bsim4Params::nmos_default();
        let derived_no_stress = Bsim4Derived::from_params(&params_no_stress);
        let result_no_stress = evaluate(&params_no_stress, &derived_no_stress, 1.0, 1.0, 0.0);

        // With positive ku0 and SA < SAref, stress should enhance mobility
        assert!(
            result_stress.ids != result_no_stress.ids,
            "Stress should change Ids"
        );
    }

    #[test]
    fn test_igidl_off_by_default() {
        let (params, derived) = setup_nmos();
        let result = evaluate(&params, &derived, 0.0, 2.0, 0.0);
        assert_eq!(result.igidl, 0.0);
    }

    #[test]
    fn test_igidl_with_large_vdg() {
        let mut params = Bsim4Params::nmos_default();
        params.agidl = 1e-12;
        params.bgidl = 0.5;
        params.egidl = 0.3;
        let derived = Bsim4Derived::from_params(&params);

        // High Vds, low Vgs -> large Vdg -> IGIDL should be non-zero
        let result = evaluate(&params, &derived, 0.0, 2.0, 0.0);

        assert!(
            result.igidl > 0.0,
            "IGIDL should be non-zero for large Vdg: {}",
            result.igidl
        );
    }

    #[test]
    fn test_temperature_threshold_shift() {
        let params = Bsim4Params::nmos_default();

        let derived_nom = Bsim4Derived::from_params(&params);
        let result_nom = evaluate(&params, &derived_nom, 1.0, 0.5, 0.0);

        let derived_hot = Bsim4Derived::from_params_at_temp(&params, 400.0);
        let result_hot = evaluate(&params, &derived_hot, 1.0, 0.5, 0.0);

        assert!(
            result_hot.vth < result_nom.vth,
            "Vth should decrease with temperature"
        );
    }

    #[test]
    fn test_temperature_current_decrease() {
        let params = Bsim4Params::nmos_default();

        let derived_nom = Bsim4Derived::from_params(&params);
        let result_nom = evaluate(&params, &derived_nom, 1.0, 1.0, 0.0);

        let derived_hot = Bsim4Derived::from_params_at_temp(&params, 400.0);
        let result_hot = evaluate(&params, &derived_hot, 1.0, 1.0, 0.0);

        assert_eq!(result_nom.region, Bsim4Region::Saturation);
        assert_eq!(result_hot.region, Bsim4Region::Saturation);
        assert!(result_hot.ids < result_nom.ids);
    }

    #[test]
    fn test_capacitance_saturation() {
        let (params, derived) = setup_nmos();

        let result = evaluate(&params, &derived, 1.0, 1.0, 0.0);
        assert_eq!(result.region, Bsim4Region::Saturation);

        let caps = evaluate_capacitances(
            &params,
            &derived,
            1.0,
            1.0,
            0.0,
            result.region,
            result.vth,
            result.vdsat,
        );

        let cox_total = derived.coxe * derived.weff * derived.leff;
        let cgs_intrinsic = cox_total * 2.0 / 3.0;

        assert!(caps.cgs >= cgs_intrinsic * 0.9);
        assert!(caps.cgd >= 0.0);
        assert!(caps.cbs > 0.0);
        assert!(caps.cbd > 0.0);
    }

    #[test]
    fn test_substrate_current() {
        let mut params = Bsim4Params::nmos_default();
        params.alpha0 = 1e-6;
        params.beta0 = 20.0;
        let derived = Bsim4Derived::from_params(&params);

        let result_low_vds = evaluate(&params, &derived, 1.0, 0.5, 0.0);
        let result_high_vds = evaluate(&params, &derived, 1.0, 1.5, 0.0);

        assert_eq!(result_low_vds.region, Bsim4Region::Saturation);
        assert_eq!(result_high_vds.region, Bsim4Region::Saturation);

        assert!(result_high_vds.isub > result_low_vds.isub);
        assert!(result_high_vds.isub < result_high_vds.ids * 0.1);
    }
}
