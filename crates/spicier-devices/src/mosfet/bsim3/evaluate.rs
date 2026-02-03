//! BSIM3v3.3 device evaluation.
//!
//! This module implements the core BSIM3 DC equations including:
//! - Threshold voltage with short-channel effects
//! - Mobility degradation
//! - Velocity saturation
//! - DIBL (drain-induced barrier lowering)
//! - Channel length modulation

use super::super::level1::MosfetType;
use super::derived::Bsim3Derived;
use super::params::Bsim3Params;

/// Operating region of the BSIM3 MOSFET.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bsim3Region {
    /// Subthreshold (weak inversion)
    Subthreshold,
    /// Linear (triode) region
    Linear,
    /// Saturation region
    Saturation,
}

/// Result of BSIM3 device evaluation.
#[derive(Debug, Clone)]
pub struct Bsim3EvalResult {
    /// Drain current (A)
    pub ids: f64,
    /// Output conductance dIds/dVds (S)
    pub gds: f64,
    /// Transconductance dIds/dVgs (S)
    pub gm: f64,
    /// Body transconductance dIds/dVbs (S)
    pub gmbs: f64,
    /// Operating region
    pub region: Bsim3Region,
    /// Effective threshold voltage (V)
    pub vth: f64,
    /// Saturation voltage Vdsat (V)
    pub vdsat: f64,
    /// Effective mobility (m^2/V-s)
    pub ueff: f64,
    /// Substrate current from impact ionization (A) - Phase 2
    pub isub: f64,
}

/// Result of BSIM3 capacitance evaluation.
#[derive(Debug, Clone)]
pub struct Bsim3CapResult {
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

/// Evaluate BSIM3 MOSFET capacitances at given terminal voltages.
///
/// Uses Meyer's model for intrinsic capacitances and voltage-dependent
/// junction capacitances for bulk-source and bulk-drain.
///
/// # Arguments
/// * `params` - Model parameters
/// * `derived` - Pre-calculated derived parameters
/// * `vgs` - Gate-source voltage (V)
/// * `vds` - Drain-source voltage (V)
/// * `vbs` - Bulk-source voltage (V)
/// * `region` - Operating region from evaluate()
/// * `vth` - Threshold voltage from evaluate()
/// * `vdsat` - Saturation voltage from evaluate()
///
/// # Returns
/// Capacitance result with Cgs, Cgd, Cgb, Cbs, Cbd.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_capacitances(
    params: &Bsim3Params,
    derived: &Bsim3Derived,
    vgs: f64,
    vds: f64,
    vbs: f64,
    region: Bsim3Region,
    _vth: f64,
    _vdsat: f64,
) -> Bsim3CapResult {
    // Handle PMOS by working with absolute voltages
    let (_vgs, vds, vbs) = match params.mos_type {
        MosfetType::Nmos => (vgs, vds, vbs),
        MosfetType::Pmos => (-vgs, -vds, -vbs),
    };

    // Total gate capacitance (intrinsic)
    let cox_total = derived.cox * derived.weff * derived.leff;

    // Meyer's model for intrinsic capacitances
    // The intrinsic capacitance is partitioned between gate-source and gate-drain
    // based on operating region
    let (cgs_int, cgd_int, cgb_int) = match region {
        Bsim3Region::Subthreshold => {
            // In subthreshold, most gate capacitance goes to bulk
            (0.0, 0.0, cox_total)
        }
        Bsim3Region::Linear => {
            // Linear region: symmetric partition between source and drain
            // Cgs = Cgd = Cox * W * L / 2
            let c_half = cox_total / 2.0;
            (c_half, c_half, 0.0)
        }
        Bsim3Region::Saturation => {
            // Saturation region: 2/3 to source, drain is pinched off
            // Cgs = (2/3) * Cox * W * L
            // Cgd = 0 (pinch-off)
            let cgs = cox_total * 2.0 / 3.0;
            (cgs, 0.0, 0.0)
        }
    };

    // Add overlap capacitances
    let cgs = cgs_int + derived.cgs_ov;
    let cgd = cgd_int + derived.cgd_ov;
    let cgb = cgb_int + derived.cgb_ov;

    // Junction capacitances (bulk-source, bulk-drain)
    // Vbs and Vbd are negative in normal operation (reverse-biased junctions)
    let vbd = vbs - vds;

    // Source junction: area + sidewall + gate-edge sidewall
    let cbs_area =
        Bsim3Derived::junction_cap(params.cj * derived.as_eff, vbs, params.pb, params.mj);
    let cbs_sw =
        Bsim3Derived::junction_cap(params.cjsw * derived.ps_eff, vbs, params.pbsw, params.mjsw);
    let cbs_swg =
        Bsim3Derived::junction_cap(params.cjswg * derived.weff, vbs, params.pbswg, params.mjswg);
    let cbs = cbs_area + cbs_sw + cbs_swg;

    // Drain junction: area + sidewall + gate-edge sidewall
    let cbd_area =
        Bsim3Derived::junction_cap(params.cj * derived.ad_eff, vbd, params.pb, params.mj);
    let cbd_sw =
        Bsim3Derived::junction_cap(params.cjsw * derived.pd_eff, vbd, params.pbsw, params.mjsw);
    let cbd_swg =
        Bsim3Derived::junction_cap(params.cjswg * derived.weff, vbd, params.pbswg, params.mjswg);
    let cbd = cbd_area + cbd_sw + cbd_swg;

    Bsim3CapResult {
        cgs,
        cgd,
        cgb,
        cbs,
        cbd,
    }
}

/// Evaluate BSIM3 MOSFET drain current and conductances.
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
    params: &Bsim3Params,
    derived: &Bsim3Derived,
    vgs: f64,
    vds: f64,
    vbs: f64,
) -> Bsim3EvalResult {
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

    // Calculate threshold voltage with all effects
    let vth = calc_threshold(params, derived, vds, vbs);

    // Gate overdrive voltage
    let vgst = vgs - vth;

    // Check for subthreshold
    if vgst < 0.0 {
        // Subthreshold region - exponential current
        let ids_sub = calc_subthreshold(params, derived, vgs, vds, vbs, vth, vgst);
        let gm_sub = ids_sub / (params.nfactor * derived.vt);
        let gds_sub = ids_sub * 0.01; // Small output conductance in subthreshold

        return Bsim3EvalResult {
            ids: sign * ids_sub,
            gds: gds_sub.max(1e-12),
            gm: if swap { 0.0 } else { gm_sub } * sign.abs(),
            gmbs: 0.0,
            region: Bsim3Region::Subthreshold,
            vth: sign * vth,
            vdsat: 0.0,
            ueff: derived.u0_si,
            isub: 0.0, // Negligible substrate current in subthreshold
        };
    }

    // Calculate effective mobility with degradation
    let ueff = calc_mobility(params, derived, vgst, vbs);

    // Calculate Abulk (bulk charge effect)
    let abulk = calc_abulk(params, derived, vbs);

    // Calculate saturation voltage
    let vdsat = calc_vdsat(params, derived, vgst, ueff, abulk);

    // Calculate early voltage for CLM (with Phase 2 enhanced DIBL)
    let va = calc_early_voltage(params, derived, vds, vdsat, vbs, vgst);

    // Determine operating region and calculate current
    let (ids, gm, gds, gmbs, region) = if vds < vdsat {
        // Linear region
        calc_linear(params, derived, vgst, vds, vbs, ueff, abulk, vdsat, va)
    } else {
        // Saturation region
        calc_saturation(params, derived, vgst, vds, vbs, ueff, abulk, vdsat, va)
    };

    // Calculate substrate current from impact ionization (Phase 2)
    // BSIM3 formula: Isub = ALPHA0 * (Ids/Leff) * (Vds - Vdsat) * exp(-BETA0 / (Vds - Vdsat))
    // ALPHA0 is dimensionless, BETA0 is in volts (typically ~30V)
    // Only significant in saturation when Vds > Vdsat
    let isub = if vds > vdsat && params.alpha0 > 0.0 {
        let vds_eff = (vds - vdsat).max(0.01);
        // BETA0 is already in volts - do NOT multiply by Leff
        let exp_term = (-params.beta0 / vds_eff).exp();
        params.alpha0 / derived.leff * ids * vds_eff * exp_term
    } else {
        0.0
    };

    // Handle source-drain swap
    let (gm, gds) = if swap { (gm - gds, gds) } else { (gm, gds) };

    Bsim3EvalResult {
        ids: sign * ids,
        gds: gds.max(1e-12),
        gm: gm * sign.abs(),
        gmbs: gmbs * sign.abs(),
        region,
        vth: sign * vth,
        vdsat,
        ueff,
        isub: sign.abs() * isub, // Substrate current always flows to substrate
    }
}

/// Calculate threshold voltage with short-channel and narrow-width effects.
fn calc_threshold(params: &Bsim3Params, derived: &Bsim3Derived, vds: f64, vbs: f64) -> f64 {
    let p = params;
    let d = derived;

    // Body effect: dVth = K1 * (sqrt(phi - Vbs) - sqrt(phi)) - K2 * Vbs
    let phi_vbs = (d.phi - vbs).max(0.01);
    let body_effect = p.k1 * (phi_vbs.sqrt() - d.sqrt_phi) - p.k2 * vbs;

    // Short-channel effect (SCE)
    // BSIM3 SCE formula: ΔVth,SCE = -DVT0 * Theta * Vt * (1 + DVT2*Vbs)
    // where Theta = exp(-DVT1 * Leff / (2*lt)) / [1 + 2*exp(-DVT1 * Leff / (2*lt))]
    //
    // In the full BSIM3 model, DVT0 is a dimensionless fitting parameter that,
    // when multiplied by the thermal voltage (Vt ≈ 26mV), gives the SCE in volts.
    // This matches the typical magnitude of SCE (tens of mV) and allows DVT0
    // values of 1-5 to give reasonable results.
    let lt_ratio = p.dvt1 * d.leff / (2.0 * d.lt);
    let exp_term = (-lt_ratio.min(20.0)).exp();
    // Theta smoothly approaches 0 for long channels and 1/3 for very short channels
    let theta = exp_term / (1.0 + 2.0 * exp_term);
    // Scale by Vt to get proper magnitude (DVT0 ~ 2 gives ~17mV at theta=0.33)
    let dvth_sce = -p.dvt0 * theta * d.vt * (1.0 + p.dvt2 * vbs);

    // Narrow-width effect (Phase 2)
    // ΔVth,NWE = K3 * Tox / Weff * (phi - Vbs) + K3B * (phi - Vbs) / Weff
    // This increases Vth for narrow devices
    let dvth_nwe = if d.weff > 1e-9 {
        let tox_weff = p.tox / d.weff;
        p.k3 * tox_weff * phi_vbs.sqrt() + p.k3b * vbs / d.weff
    } else {
        0.0
    };

    // Narrow-width SCE (Phase 2)
    // Similar to length SCE but for width direction
    // Uses characteristic width instead of lt
    let wt = (Bsim3Params::EPS_SI * p.tox / Bsim3Params::EPS_OX).sqrt(); // characteristic width
    let wt_ratio = p.dvt1w * d.weff / (2.0 * wt + p.w0);
    let exp_w = (-wt_ratio.min(20.0)).exp();
    let theta_w = exp_w / (1.0 + 2.0 * exp_w);
    let dvth_nwe_sce = -p.dvt0w * theta_w * d.vt * (1.0 + p.dvt2w * vbs);

    // DIBL effect: dVth_DIBL = -(ETA0 + ETAB * Vbs) * Vds * theta_dibl
    // In BSIM3, DIBL has length dependence similar to SCE
    // The DSUB parameter controls how DIBL diminishes for longer channels
    // theta_dibl = exp(-DSUB * Leff / (2*lt)) for smooth length scaling
    let dibl_lt_ratio = p.dsub * d.leff / (2.0 * d.lt);
    let theta_dibl = (-dibl_lt_ratio.min(20.0)).exp();
    let dvth_dibl = -(p.eta0 + p.etab * vbs) * vds * theta_dibl;

    // Temperature effect on threshold (Phase 4)
    // dVth_T = KT1*(T/Tnom - 1) + KT1L/Leff*(T/Tnom - 1) + KT2*Vbs*(T/Tnom - 1)
    let dvth_temp = d.vth0_temp + p.kt2 * vbs * (d.temp_ratio - 1.0);

    // Total threshold voltage (use absolute Vth0 for internal calculations)
    let vth0_abs = p.vth0.abs();
    vth0_abs + body_effect + dvth_sce + dvth_nwe + dvth_nwe_sce + dvth_dibl + dvth_temp
}

/// Calculate subthreshold current.
fn calc_subthreshold(
    params: &Bsim3Params,
    derived: &Bsim3Derived,
    _vgs: f64,
    vds: f64,
    _vbs: f64,
    _vth: f64,
    vgst: f64,
) -> f64 {
    let d = derived;

    // Subthreshold swing: n = 1 + NFACTOR
    let n = params.nfactor;

    // Thermal voltage
    let nvt = n * d.vt;

    // Subthreshold current: Ids = I0 * exp(Vgst / (n * Vt)) * (1 - exp(-Vds/Vt))
    // I0 is approximately W/L * u0 * Cox * (n * Vt)^2
    let i0 = (d.weff / d.leff) * d.u0_si * d.cox * nvt * nvt;

    // Exponential factor (limited to avoid overflow)
    let exp_factor = (vgst / nvt).min(20.0).exp();

    // Drain voltage factor
    let vds_factor = 1.0 - (-vds / d.vt).exp();

    i0 * exp_factor * vds_factor
}

/// Calculate effective mobility with degradation.
fn calc_mobility(params: &Bsim3Params, derived: &Bsim3Derived, vgst: f64, vbs: f64) -> f64 {
    let p = params;
    let d = derived;

    // BSIM3 effective vertical field
    // Eeff = (Vgst + 2*Vth) / (3 * Tox) for surface roughness scattering
    // This gives a more accurate representation of the inversion layer field
    // The factor of 3 comes from the triangular approximation of the inversion layer
    let vth_eff = p.vth0.abs(); // Use nominal Vth for field calculation
    let vgst_eff = vgst.max(0.01);
    let eeff = (vgst_eff + 2.0 * vth_eff) / (3.0 * p.tox);

    // Use temperature-scaled mobility degradation coefficients (Phase 4)
    // UA(T), UB(T), UC(T) are stored in derived
    // μeff = μ0 / (1 + (UA + UC*Vbs)*Eeff + UB*Eeff²)
    let degradation = 1.0 + (d.ua_temp + d.uc_temp * vbs) * eeff + d.ub_temp * eeff * eeff;

    d.u0_si / degradation.max(0.1)
}

/// Calculate bulk charge effect coefficient Abulk.
fn calc_abulk(params: &Bsim3Params, derived: &Bsim3Derived, vbs: f64) -> f64 {
    let p = params;
    let d = derived;

    // BSIM3 Abulk model (simplified)
    // Abulk = 1 + K1 / (2 * sqrt(Phi - Vbs)) - K2
    // The K2 term reduces Abulk for heavily doped substrates
    let phi_vbs = (d.phi - vbs).max(0.01);
    let abulk_base = 1.0 + p.k1 / (2.0 * phi_vbs.sqrt()) - p.k2;

    // Short-channel Abulk reduction
    // For short channels, fringing fields reduce bulk charge effect
    // Simplified: use a length-dependent factor
    let sce_factor = 1.0 - 0.3 * (d.lt / d.leff).min(1.0);

    let abulk = abulk_base * sce_factor;

    // Clamp to reasonable range
    abulk.clamp(1.0, 2.5)
}

/// Calculate saturation voltage Vdsat.
fn calc_vdsat(
    _params: &Bsim3Params,
    derived: &Bsim3Derived,
    vgst: f64,
    ueff: f64,
    abulk: f64,
) -> f64 {
    let d = derived;

    // Velocity saturation: Esat = 2 * VSAT / ueff
    // Use temperature-scaled saturation velocity (Phase 4)
    let esat = 2.0 * d.vsat_temp / ueff;

    // EsatL = Esat * Leff
    let esat_l = esat * d.leff;

    // Vdsat = Vgst / (Abulk + Vgst / EsatL)
    // This smoothly transitions from long-channel (Vdsat = Vgst/Abulk)
    // to short-channel (Vdsat ~ EsatL) behavior
    let vgst_eff = vgst.max(0.001);
    let vdsat = vgst_eff / (abulk + vgst_eff / esat_l);

    vdsat.max(0.001)
}

/// Calculate Early voltage for channel length modulation (with Phase 2 enhancements).
fn calc_early_voltage(
    params: &Bsim3Params,
    derived: &Bsim3Derived,
    vds: f64,
    vdsat: f64,
    vbs: f64,
    vgst: f64,
) -> f64 {
    let p = params;
    let d = derived;

    // BSIM3 simplified Early voltage model
    // Va represents the voltage at which the output current would extrapolate to zero
    //
    // For a 100nm device at Vds=1V, ngspice gives:
    //   Ids = 280 µA, gds = 159 µS
    //   => Va = Ids/gds ≈ 1.76V
    //
    // The BSIM3 output conductance is primarily from:
    // 1. Channel length modulation (CLM): gds_clm = Ids / Va_clm
    // 2. DIBL effect on Rout: gds_dibl = Ids / Va_dibl
    //
    // Va_clm scales roughly as: PCLM * Leff / sqrt(Vds - Vdsat)
    // For short channels, Va_clm ~ Leff / PCLM
    //
    // Simplified model that matches typical BSIM3 behavior:
    // Va = (Leff / PCLM) * Esat * adjustment_factors
    // where Esat ≈ 2*Vsat/μeff and typical adjustment gives Va ~ 1-3V

    // Base Early voltage from channel length modulation
    // Va_base = Leff * Esat * scaling_factor
    // where Esat = 2*Vsat/ueff is the saturation field
    //
    // For ngspice reference: 100nm device, Vds=1V
    //   Va ≈ 1.76V, Leff = 1e-7m, Esat ≈ 5e6 V/m
    //   => scaling = 1.76 / (1e-7 * 5e6) = 3.5
    //
    // The scaling factor accounts for PCLM and other effects
    // PCLM typically ranges 0.5-2.0, so use 4/PCLM as scaling
    let esat = 2.0 * p.vsat / d.u0_si; // Saturation field in V/m
    let va_base = d.leff * esat * 4.0 / (p.pclm.max(0.1));

    // DIBL reduction of Early voltage
    // Higher PDIBLC means lower Va
    let pdiblc_eff = (p.pdiblc1 + p.pdiblcb * vbs).abs() + p.pdiblc2;
    let va_dibl_factor = 1.0 / (1.0 + pdiblc_eff * (vds / (vdsat + 0.1)));

    // DROUT: length dependence of DIBL output resistance
    let drout_factor = 1.0 / (1.0 + p.drout * vds / (vgst + 0.1));

    // PVAG: gate overdrive increases output resistance
    let pvag_factor = 1.0 + p.pvag * vgst / (vdsat + 0.1);

    // Combine factors
    let va = va_base * va_dibl_factor * drout_factor * pvag_factor;

    // Ensure reasonable range (0.5V to 10V for typical devices)
    va.clamp(0.5, 10.0)
}

/// Calculate linear region current and conductances.
#[allow(clippy::too_many_arguments)]
fn calc_linear(
    params: &Bsim3Params,
    derived: &Bsim3Derived,
    vgst: f64,
    vds: f64,
    vbs: f64,
    ueff: f64,
    abulk: f64,
    _vdsat: f64,
    va: f64,
) -> (f64, f64, f64, f64, Bsim3Region) {
    let d = derived;
    let vgst_eff = vgst.max(0.001);
    let beta = (d.weff / d.leff) * ueff * d.cox;

    // Velocity saturation parameters
    let esat = 2.0 * params.vsat / ueff;
    let esat_l = esat * d.leff;

    // BSIM3 linear region current:
    // Ids = beta * [(Vgst - Abulk*Vds/2) * Vds] / (1 + Vds/EsatL)
    // This gives resistive behavior for small Vds
    let vgst_eff_lin = (vgst_eff - abulk * vds / 2.0).max(0.001);
    let vs_factor = 1.0 + vds / esat_l;

    let ids_base = beta * vgst_eff_lin * vds / vs_factor;

    // Channel length modulation (smaller effect in linear)
    let ids = ids_base * (1.0 + vds / va * 0.1);

    // Transconductance: gm = dIds/dVgs = beta * Vds / vs_factor
    let gm = beta * vds / vs_factor * (1.0 + vds / va * 0.1);

    // Output conductance in linear region
    let gds = beta * (vgst_eff - abulk * vds) / vs_factor
        + beta * vgst_eff_lin * vds / (vs_factor * esat_l * vs_factor)
        + ids_base / va * 0.1;

    // Body transconductance
    let phi_vbs = (d.phi - vbs).max(0.01);
    let dvth_dvbs = params.k1 / (2.0 * phi_vbs.sqrt()) - params.k2;
    let gmbs = (gm * dvth_dvbs.abs() * 0.5).abs();

    (
        ids.max(0.0),
        gm.max(0.0),
        gds.max(1e-12),
        gmbs,
        Bsim3Region::Linear,
    )
}

/// Calculate saturation region current and conductances.
#[allow(clippy::too_many_arguments)]
fn calc_saturation(
    params: &Bsim3Params,
    derived: &Bsim3Derived,
    vgst: f64,
    vds: f64,
    vbs: f64,
    ueff: f64,
    abulk: f64,
    vdsat: f64,
    va: f64,
) -> (f64, f64, f64, f64, Bsim3Region) {
    let d = derived;
    let vgst_eff = vgst.max(0.001);
    let beta = (d.weff / d.leff) * ueff * d.cox;

    // Velocity saturation parameters
    let esat = 2.0 * params.vsat / ueff;
    let esat_l = esat * d.leff;

    // BSIM3 saturation current:
    // Ids = beta * Vgst^2 / [2 * (Abulk + Vgst/EsatL)]
    // This is the correct square-law with velocity saturation
    let denom = 2.0 * (abulk + vgst_eff / esat_l);
    let ids_dsat = beta * vgst_eff * vgst_eff / denom;

    // Channel length modulation for Vds > Vdsat
    let clm_factor = 1.0 + (vds - vdsat).max(0.0) / va;
    let ids = ids_dsat * clm_factor;

    // Transconductance: gm = dIds/dVgst
    // d/dVgst [Vgst^2 / (Abulk + Vgst/EsatL)] = Vgst * (2*Abulk + Vgst/EsatL) / (Abulk + Vgst/EsatL)^2
    let denom_sq = (abulk + vgst_eff / esat_l).powi(2);
    let gm = beta * vgst_eff * (2.0 * abulk + vgst_eff / esat_l) / (2.0 * denom_sq) * clm_factor;

    // Output conductance: gds = Ids_dsat / Va (CLM effect)
    let gds = ids_dsat / va;

    // Body transconductance (through threshold voltage modulation)
    let phi_vbs = (d.phi - vbs).max(0.01);
    let dvth_dvbs = params.k1 / (2.0 * phi_vbs.sqrt()) - params.k2 - params.etab * vds;
    let gmbs = (gm * dvth_dvbs.abs()).abs();

    (
        ids.max(0.0),
        gm.max(0.0),
        gds.max(1e-12),
        gmbs,
        Bsim3Region::Saturation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_nmos() -> (Bsim3Params, Bsim3Derived) {
        let params = Bsim3Params::nmos_default();
        let derived = Bsim3Derived::from_params(&params);
        (params, derived)
    }

    #[test]
    fn test_subthreshold() {
        let (params, derived) = setup_nmos();

        // First, find what the threshold is at low Vds to minimize DIBL
        let result_ref = evaluate(&params, &derived, 0.5, 0.1, 0.0);
        let vth_approx = result_ref.vth;

        // Use a gate voltage clearly below threshold (Vth - 0.2V)
        // Also use low Vds to minimize DIBL effect on Vth
        let vgs_sub = (vth_approx - 0.2).min(0.1);
        let result = evaluate(&params, &derived, vgs_sub, 0.1, 0.0);

        assert_eq!(
            result.region,
            Bsim3Region::Subthreshold,
            "Expected subthreshold at Vgs={}, Vth={}",
            vgs_sub,
            result.vth
        );
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

        // Vgs = 1.0V, Vds = 1.0V -> saturation (Vds > Vdsat)
        let result = evaluate(&params, &derived, 1.0, 1.0, 0.0);

        assert_eq!(result.region, Bsim3Region::Saturation);
        assert!(result.ids > 0.0);
        assert!(result.gm > 0.0);
        assert!(result.gds > 0.0);

        // Current should be reasonable for W=1um, L=100nm
        assert!(result.ids > 10e-6); // > 10uA
        assert!(result.ids < 10e-3); // < 10mA
    }

    /// Compare against ngspice BSIM3v3.3 reference values.
    ///
    /// ngspice-45 with identical parameters:
    /// - Ids = 280.769 µA
    /// - gm = 502.261 µS
    /// - gds = 159.053 µS
    /// - Vth = 0.344833 V
    /// - Vdsat = 0.313407 V
    #[test]
    fn test_ngspice_comparison() {
        let (params, derived) = setup_nmos();

        // Same conditions as ngspice test
        let result = evaluate(&params, &derived, 1.0, 1.0, 0.0);

        // ngspice reference values (from ngspice-45 with identical parameters)
        let ngspice_ids = 280.769e-6;
        let _ngspice_gm = 502.261e-6;
        let _ngspice_gds = 159.053e-6;
        let ngspice_vth = 0.344833;
        let ngspice_vdsat = 0.313407;

        // Check Vth is close (should be within 10%)
        let vth_error = (result.vth - ngspice_vth).abs() / ngspice_vth;
        assert!(
            vth_error < 0.10,
            "Vth error {:.1}% exceeds 10%: spicier={:.4}V, ngspice={:.4}V",
            vth_error * 100.0,
            result.vth,
            ngspice_vth
        );

        // Check Vdsat is close (should be within 20%)
        let vdsat_error = (result.vdsat - ngspice_vdsat).abs() / ngspice_vdsat;
        assert!(
            vdsat_error < 0.20,
            "Vdsat error {:.1}% exceeds 20%: spicier={:.4}V, ngspice={:.4}V",
            vdsat_error * 100.0,
            result.vdsat,
            ngspice_vdsat
        );

        // Check Ids is within 50% (we expect some difference due to simplified model)
        // TODO: Improve to within 25%
        let ids_error = (result.ids - ngspice_ids).abs() / ngspice_ids;
        assert!(
            ids_error < 0.50,
            "Ids error {:.1}% exceeds 50%: spicier={:.3e}A, ngspice={:.3e}A",
            ids_error * 100.0,
            result.ids,
            ngspice_ids
        );
    }

    #[test]
    fn test_linear() {
        let (params, derived) = setup_nmos();

        // Vgs = 1.0V, Vds = 0.1V -> linear (Vds < Vdsat)
        let result = evaluate(&params, &derived, 1.0, 0.1, 0.0);

        assert_eq!(result.region, Bsim3Region::Linear);
        assert!(result.ids > 0.0);
        assert!(result.gds > result.gm * 0.1); // High gds in linear region
    }

    #[test]
    fn test_body_effect() {
        let (params, derived) = setup_nmos();

        // Test threshold voltage shift with body bias
        let result_vbs0 = evaluate(&params, &derived, 1.0, 1.0, 0.0);
        let result_vbs_neg = evaluate(&params, &derived, 1.0, 1.0, -1.0);

        // Negative Vbs should increase Vth, reducing current
        assert!(result_vbs_neg.vth > result_vbs0.vth);
        assert!(result_vbs_neg.ids < result_vbs0.ids);
    }

    #[test]
    fn test_pmos() {
        let params = Bsim3Params::pmos_default();
        let derived = Bsim3Derived::from_params(&params);

        // PMOS: Vgs = -1.0V, Vds = -1.0V -> saturation
        let result = evaluate(&params, &derived, -1.0, -1.0, 0.0);

        assert_eq!(result.region, Bsim3Region::Saturation);
        assert!(
            result.ids < 0.0,
            "PMOS Ids should be negative: {}",
            result.ids
        );
        // Note: vth is returned as the internal (NMOS-equivalent) threshold used in calculations
        // The sign convention varies by implementation; what matters is the current direction
    }

    #[test]
    fn test_continuity_at_vdsat() {
        let (params, derived) = setup_nmos();

        // Test current continuity at Vdsat boundary
        let vgs = 1.0;

        // Evaluate at a point known to be in saturation to get Vdsat
        let result_sat = evaluate(&params, &derived, vgs, 1.0, 0.0);
        let vdsat = result_sat.vdsat;

        // Evaluate just below and just above Vdsat
        let result_below = evaluate(&params, &derived, vgs, vdsat * 0.95, 0.0);
        let result_above = evaluate(&params, &derived, vgs, vdsat * 1.05, 0.0);

        assert_eq!(result_below.region, Bsim3Region::Linear);
        assert_eq!(result_above.region, Bsim3Region::Saturation);

        // Current should be reasonably continuous (within 20%)
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
        let mut params_short = Bsim3Params::nmos_default();
        params_short.l = 50e-9; // 50nm

        let mut params_long = Bsim3Params::nmos_default();
        params_long.l = 500e-9; // 500nm

        let derived_short = Bsim3Derived::from_params(&params_short);
        let derived_long = Bsim3Derived::from_params(&params_long);

        // Use lower Vds to reduce DIBL contribution for clearer SCE comparison
        let result_short = evaluate(&params_short, &derived_short, 1.0, 0.1, 0.0);
        let result_long = evaluate(&params_long, &derived_long, 1.0, 0.1, 0.0);

        // Short channel should have lower Vth due to SCE (Vth roll-off)
        // The SCE term is: -dvt0 * dvt1 * exp(-Leff/(2*lt))
        // Shorter channel -> smaller Leff -> larger exp term -> more negative contribution -> lower Vth
        assert!(
            result_short.vth < result_long.vth,
            "SCE should reduce Vth for short channels: short={}, long={}",
            result_short.vth,
            result_long.vth
        );

        // Verify both devices are above threshold
        assert!(result_short.ids > 0.0);
        assert!(result_long.ids > 0.0);
    }

    #[test]
    fn test_dibl() {
        let (params, derived) = setup_nmos();

        // DIBL: higher Vds should reduce Vth
        let result_low_vds = evaluate(&params, &derived, 0.6, 0.1, 0.0);
        let result_high_vds = evaluate(&params, &derived, 0.6, 1.0, 0.0);

        // With DIBL, threshold should be lower at high Vds
        assert!(
            result_high_vds.vth < result_low_vds.vth,
            "DIBL should reduce Vth: low_vds={}, high_vds={}",
            result_low_vds.vth,
            result_high_vds.vth
        );
    }

    // ========================================
    // Phase 2 Tests
    // ========================================

    #[test]
    fn test_narrow_width_effect() {
        // Test narrow width effect on threshold voltage
        // K3 causes Vth to increase for narrow devices

        let mut params_wide = Bsim3Params::nmos_default();
        params_wide.w = 1e-6; // 1um wide
        params_wide.k3 = 10.0; // Enable narrow width effect

        let mut params_narrow = Bsim3Params::nmos_default();
        params_narrow.w = 100e-9; // 100nm narrow
        params_narrow.k3 = 10.0; // Same K3

        let derived_wide = super::Bsim3Derived::from_params(&params_wide);
        let derived_narrow = super::Bsim3Derived::from_params(&params_narrow);

        let result_wide = evaluate(&params_wide, &derived_wide, 1.0, 0.5, 0.0);
        let result_narrow = evaluate(&params_narrow, &derived_narrow, 1.0, 0.5, 0.0);

        // Narrow device should have higher Vth due to K3 effect
        assert!(
            result_narrow.vth > result_wide.vth,
            "Narrow width should increase Vth: wide={:.3}V, narrow={:.3}V",
            result_wide.vth,
            result_narrow.vth
        );

        // Consequently, narrow device should have less current
        assert!(
            result_narrow.ids < result_wide.ids,
            "Narrow device should have less current due to higher Vth"
        );
    }

    #[test]
    fn test_substrate_current() {
        // Test impact ionization substrate current
        // Isub increases with (Vds - Vdsat) in saturation

        let mut params = Bsim3Params::nmos_default();
        params.alpha0 = 1e-6; // Enable substrate current
        params.beta0 = 20.0;
        let derived = super::Bsim3Derived::from_params(&params);

        // Low Vds (close to Vdsat) - minimal impact ionization
        let result_low_vds = evaluate(&params, &derived, 1.0, 0.5, 0.0);

        // High Vds (well above Vdsat) - significant impact ionization
        let result_high_vds = evaluate(&params, &derived, 1.0, 1.5, 0.0);

        // Both should be in saturation
        assert_eq!(result_low_vds.region, Bsim3Region::Saturation);
        assert_eq!(result_high_vds.region, Bsim3Region::Saturation);

        // Substrate current should increase with Vds
        assert!(
            result_high_vds.isub > result_low_vds.isub,
            "Isub should increase with Vds: low={:.3e}A, high={:.3e}A",
            result_low_vds.isub,
            result_high_vds.isub
        );

        // Substrate current should be small compared to drain current
        assert!(
            result_high_vds.isub < result_high_vds.ids * 0.1,
            "Isub should be << Ids"
        );
    }

    #[test]
    fn test_enhanced_dibl_body_effect() {
        // Test PDIBLCB: body-bias dependence of DIBL output resistance

        let mut params = Bsim3Params::nmos_default();
        params.pdiblcb = 0.5; // Enable body effect on DIBL
        let derived = super::Bsim3Derived::from_params(&params);

        // Vbs = 0
        let result_vbs0 = evaluate(&params, &derived, 1.0, 1.0, 0.0);

        // Negative Vbs (reverse body bias)
        let result_vbs_neg = evaluate(&params, &derived, 1.0, 1.0, -0.5);

        // Both should produce positive current in saturation
        assert!(result_vbs0.ids > 0.0);
        assert!(result_vbs_neg.ids > 0.0);

        // gds (output conductance) may be affected by body bias via PDIBLCB
        // The effect should be observable (though the sign depends on parameter values)
        let _gds_diff = (result_vbs_neg.gds - result_vbs0.gds).abs();
        // Just verify gds is positive (showing PDIBLCB computation doesn't break anything)
        assert!(
            result_vbs0.gds > 0.0 && result_vbs_neg.gds > 0.0,
            "gds should be positive"
        );
    }

    // ========================================
    // Phase 3 Tests (Capacitances)
    // ========================================

    #[test]
    fn test_capacitance_saturation() {
        // Test capacitance calculation in saturation
        // In saturation, Cgs = 2/3 * Cox * W * L, Cgd ≈ 0

        let (params, derived) = setup_nmos();

        let result = evaluate(&params, &derived, 1.0, 1.0, 0.0);
        assert_eq!(result.region, Bsim3Region::Saturation);

        let caps = evaluate_capacitances(
            &params,
            &derived,
            1.0, // vgs
            1.0, // vds
            0.0, // vbs
            result.region,
            result.vth,
            result.vdsat,
        );

        // Cox total = Cox * Weff * Leff
        let cox_total = derived.cox * derived.weff * derived.leff;

        // In saturation, Cgs_intrinsic ≈ (2/3) * Cox_total
        // Plus overlap capacitance
        let cgs_intrinsic = cox_total * 2.0 / 3.0;
        assert!(
            caps.cgs >= cgs_intrinsic * 0.9,
            "Cgs in saturation should be >= (2/3)*Cox*W*L: {:.3e} >= {:.3e}",
            caps.cgs,
            cgs_intrinsic
        );

        // In saturation, Cgd_intrinsic ≈ 0 (pinch-off), only overlap
        assert!(
            caps.cgd >= 0.0,
            "Cgd should be non-negative: {:.3e}",
            caps.cgd
        );

        // Junction capacitances should be positive
        assert!(caps.cbs > 0.0, "Cbs should be positive");
        assert!(caps.cbd > 0.0, "Cbd should be positive");
    }

    #[test]
    fn test_capacitance_linear() {
        // Test capacitance calculation in linear region
        // In linear, Cgs ≈ Cgd ≈ Cox * W * L / 2

        let (params, derived) = setup_nmos();

        let result = evaluate(&params, &derived, 1.0, 0.1, 0.0);
        assert_eq!(result.region, Bsim3Region::Linear);

        let caps = evaluate_capacitances(
            &params,
            &derived,
            1.0, // vgs
            0.1, // vds (low, linear region)
            0.0, // vbs
            result.region,
            result.vth,
            result.vdsat,
        );

        // In linear, Cgs ≈ Cgd ≈ Cox_total / 2
        let cox_total = derived.cox * derived.weff * derived.leff;
        let c_half = cox_total / 2.0;

        assert!(
            (caps.cgs - caps.cgd).abs() < c_half * 0.2,
            "In linear, Cgs ≈ Cgd: Cgs={:.3e}, Cgd={:.3e}",
            caps.cgs,
            caps.cgd
        );
    }

    #[test]
    fn test_capacitance_subthreshold() {
        // Test capacitance calculation in subthreshold
        // In subthreshold, gate capacitance goes to bulk

        let (params, derived) = setup_nmos();

        let result = evaluate(&params, &derived, 0.2, 0.5, 0.0);
        assert_eq!(result.region, Bsim3Region::Subthreshold);

        let caps = evaluate_capacitances(
            &params,
            &derived,
            0.2, // vgs (below threshold)
            0.5, // vds
            0.0, // vbs
            result.region,
            result.vth,
            result.vdsat,
        );

        // In subthreshold, Cgb should dominate intrinsic capacitance
        let cox_total = derived.cox * derived.weff * derived.leff;

        // Cgb_intrinsic ≈ Cox_total in subthreshold
        assert!(
            caps.cgb >= cox_total * 0.8,
            "In subthreshold, Cgb should be ≈ Cox*W*L: {:.3e} >= {:.3e}",
            caps.cgb,
            cox_total
        );
    }

    #[test]
    fn test_junction_capacitance_voltage_dependence() {
        // Test that junction capacitance increases with forward bias

        let (params, derived) = setup_nmos();

        // Zero body bias (junctions reverse-biased in normal operation)
        let result0 = evaluate(&params, &derived, 1.0, 1.0, 0.0);
        let caps0 = evaluate_capacitances(
            &params,
            &derived,
            1.0,
            1.0,
            0.0,
            result0.region,
            result0.vth,
            result0.vdsat,
        );

        // Slight forward bias on source junction
        let result_fwd = evaluate(&params, &derived, 1.0, 1.0, 0.3);
        let caps_fwd = evaluate_capacitances(
            &params,
            &derived,
            1.0,
            1.0,
            0.3,
            result_fwd.region,
            result_fwd.vth,
            result_fwd.vdsat,
        );

        // Forward bias should increase junction capacitance
        assert!(
            caps_fwd.cbs > caps0.cbs,
            "Forward bias should increase Cbs: fwd={:.3e} > rev={:.3e}",
            caps_fwd.cbs,
            caps0.cbs
        );
    }

    #[test]
    fn test_overlap_capacitances() {
        // Test that overlap capacitances are correctly added

        let mut params = Bsim3Params::nmos_default();
        params.cgso = 2e-10; // 200 fF/um
        params.cgdo = 2e-10;
        params.cgbo = 1e-10;
        let derived = Bsim3Derived::from_params(&params);

        let result = evaluate(&params, &derived, 1.0, 1.0, 0.0);
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

        // Overlap capacitances: Cov = C_per_um * Weff
        let cgs_ov_expected = params.cgso * derived.weff;
        let cgd_ov_expected = params.cgdo * derived.weff;
        let _cgb_ov_expected = params.cgbo * derived.leff;

        // Cgs should be >= overlap (intrinsic adds to it)
        assert!(
            caps.cgs >= cgs_ov_expected * 0.9,
            "Cgs should include overlap: {:.3e} >= {:.3e}",
            caps.cgs,
            cgs_ov_expected
        );

        // In saturation, Cgd ≈ overlap only (intrinsic = 0)
        assert!(
            (caps.cgd - cgd_ov_expected).abs() < cgd_ov_expected * 0.5,
            "Cgd in saturation should be ≈ overlap: {:.3e} ≈ {:.3e}",
            caps.cgd,
            cgd_ov_expected
        );
    }

    // ========================================
    // Phase 4 Tests (Temperature)
    // ========================================

    #[test]
    fn test_temperature_threshold_shift() {
        // Test that threshold voltage decreases with increasing temperature
        // (KT1 is typically negative, ~-0.11 V/decade)

        let params = Bsim3Params::nmos_default();

        // At nominal temperature (300K)
        let derived_nom = Bsim3Derived::from_params(&params);
        let result_nom = evaluate(&params, &derived_nom, 1.0, 0.5, 0.0);

        // At elevated temperature (400K)
        let derived_hot = Bsim3Derived::from_params_at_temp(&params, 400.0);
        let result_hot = evaluate(&params, &derived_hot, 1.0, 0.5, 0.0);

        // Vth should decrease with temperature (KT1 < 0)
        assert!(
            result_hot.vth < result_nom.vth,
            "Vth should decrease with temperature: Vth(400K)={:.3}V < Vth(300K)={:.3}V",
            result_hot.vth,
            result_nom.vth
        );

        // Verify the shift is in the right ballpark
        // dVth ≈ KT1 * (T/Tnom - 1) ≈ -0.11 * (400/300 - 1) ≈ -0.037V
        let expected_shift = params.kt1 * (400.0 / params.tnom - 1.0);
        let actual_shift = result_hot.vth - result_nom.vth;
        assert!(
            (actual_shift - expected_shift).abs() < 0.05,
            "Vth shift: actual={:.4}V, expected≈{:.4}V",
            actual_shift,
            expected_shift
        );
    }

    #[test]
    fn test_temperature_mobility_degradation() {
        // Test that mobility decreases with increasing temperature
        // u0(T) = u0(Tnom) * (T/Tnom)^UTE, where UTE is typically -1.5

        let params = Bsim3Params::nmos_default();

        // At nominal temperature (300K)
        let derived_nom = Bsim3Derived::from_params(&params);

        // At elevated temperature (400K)
        let derived_hot = Bsim3Derived::from_params_at_temp(&params, 400.0);

        // Mobility should decrease with temperature (UTE < 0)
        assert!(
            derived_hot.u0_si < derived_nom.u0_si,
            "Mobility should decrease with temperature: u0(400K)={:.3e} < u0(300K)={:.3e}",
            derived_hot.u0_si,
            derived_nom.u0_si
        );

        // Verify the scaling follows (T/Tnom)^UTE
        let temp_ratio = 400.0 / params.tnom;
        let expected_ratio = temp_ratio.powf(params.ute);
        let actual_ratio = derived_hot.u0_si / derived_nom.u0_si;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Mobility ratio: actual={:.4}, expected={:.4}",
            actual_ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_temperature_current_decrease() {
        // Test that drain current generally decreases with temperature
        // (due to mobility degradation, partially offset by Vth decrease)

        let params = Bsim3Params::nmos_default();

        // At nominal temperature (300K)
        let derived_nom = Bsim3Derived::from_params(&params);
        let result_nom = evaluate(&params, &derived_nom, 1.0, 1.0, 0.0);

        // At elevated temperature (400K)
        let derived_hot = Bsim3Derived::from_params_at_temp(&params, 400.0);
        let result_hot = evaluate(&params, &derived_hot, 1.0, 1.0, 0.0);

        // Both should be in saturation
        assert_eq!(result_nom.region, Bsim3Region::Saturation);
        assert_eq!(result_hot.region, Bsim3Region::Saturation);

        // Current should decrease due to mobility degradation
        // (the effect of lower mobility usually dominates over lower Vth)
        assert!(
            result_hot.ids < result_nom.ids,
            "Ids should decrease with temperature: Ids(400K)={:.3e}A < Ids(300K)={:.3e}A",
            result_hot.ids,
            result_nom.ids
        );
    }

    #[test]
    fn test_temperature_saturation_velocity() {
        // Test that saturation velocity decreases with temperature
        // Vsat(T) = Vsat - AT*(T - Tnom)

        let params = Bsim3Params::nmos_default();

        // At nominal temperature
        let derived_nom = Bsim3Derived::from_params(&params);

        // At elevated temperature
        let derived_hot = Bsim3Derived::from_params_at_temp(&params, 400.0);

        // Vsat should decrease with temperature (AT > 0)
        assert!(
            derived_hot.vsat_temp < derived_nom.vsat_temp,
            "Vsat should decrease with temperature: {:.2e} < {:.2e}",
            derived_hot.vsat_temp,
            derived_nom.vsat_temp
        );

        // Verify the change
        let delta_t = 400.0 - params.tnom;
        let expected_vsat = params.vsat - params.at * delta_t;
        assert!(
            (derived_hot.vsat_temp - expected_vsat).abs() < 1e3,
            "Vsat at 400K: actual={:.2e}, expected={:.2e}",
            derived_hot.vsat_temp,
            expected_vsat
        );
    }

    #[test]
    fn test_cold_temperature_operation() {
        // Test operation at cold temperature (200K)
        // Should see higher mobility and higher Vth

        let params = Bsim3Params::nmos_default();

        // At nominal temperature (300K)
        let derived_nom = Bsim3Derived::from_params(&params);
        let result_nom = evaluate(&params, &derived_nom, 1.0, 1.0, 0.0);

        // At cold temperature (200K)
        let derived_cold = Bsim3Derived::from_params_at_temp(&params, 200.0);
        let result_cold = evaluate(&params, &derived_cold, 1.0, 1.0, 0.0);

        // Vth should increase at cold temperature (opposite of hot)
        assert!(
            result_cold.vth > result_nom.vth,
            "Vth should increase at cold: Vth(200K)={:.3}V > Vth(300K)={:.3}V",
            result_cold.vth,
            result_nom.vth
        );

        // Mobility should increase at cold temperature
        assert!(
            derived_cold.u0_si > derived_nom.u0_si,
            "Mobility should increase at cold: u0(200K)={:.3e} > u0(300K)={:.3e}",
            derived_cold.u0_si,
            derived_nom.u0_si
        );
    }
}
