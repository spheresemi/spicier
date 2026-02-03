//! BSIM3v3.3 derived (pre-calculated) parameters.
//!
//! These are parameters that are computed once from the model/instance
//! parameters and reused during device evaluation. This avoids redundant
//! calculations during Newton-Raphson iterations.

use super::params::Bsim3Params;

/// Pre-calculated BSIM3 parameters derived from model parameters.
///
/// These values are computed once when the device is created and
/// stored for efficient reuse during simulation.
#[derive(Debug, Clone)]
pub struct Bsim3Derived {
    /// Effective channel length (m)
    pub leff: f64,
    /// Effective channel width (m)
    pub weff: f64,
    /// Oxide capacitance per unit area (F/m^2)
    pub cox: f64,
    /// Thermal voltage (V) at operating temperature
    pub vt: f64,
    /// sqrt(2 * q * eps_si * Nch)
    pub sqrtk1: f64,
    /// Surface potential at strong inversion (~2*phi_s)
    pub phi: f64,
    /// Sqrt of surface potential
    pub sqrt_phi: f64,
    /// Channel doping term for body effect
    pub k1ox: f64,
    /// Low-field mobility in m^2/V-s (converted from cm^2/V-s)
    pub u0_si: f64,
    /// Pre-computed characteristic length for SCE
    pub lt: f64,
    /// Source/drain resistance (ohms)
    pub rds: f64,
    /// BSIM3 Abulk coefficient for bulk charge effect
    pub abulk0: f64,
    /// Built-in potential Vbi (V) for SCE calculation
    pub vbi: f64,

    // ========================================
    // Capacitance-related derived values (Phase 3)
    // ========================================
    /// Total gate-source overlap capacitance (F)
    pub cgs_ov: f64,
    /// Total gate-drain overlap capacitance (F)
    pub cgd_ov: f64,
    /// Total gate-bulk overlap capacitance (F)
    pub cgb_ov: f64,
    /// Source diffusion area (m^2) - auto-calculated if zero
    pub as_eff: f64,
    /// Drain diffusion area (m^2) - auto-calculated if zero
    pub ad_eff: f64,
    /// Source diffusion perimeter (m) - auto-calculated if zero
    pub ps_eff: f64,
    /// Drain diffusion perimeter (m) - auto-calculated if zero
    pub pd_eff: f64,

    // ========================================
    // Temperature-scaled parameters (Phase 4)
    // ========================================
    /// Operating temperature (K)
    pub temp: f64,
    /// Temperature ratio T/Tnom
    pub temp_ratio: f64,
    /// Temperature-scaled threshold voltage offset (V)
    pub vth0_temp: f64,
    /// Temperature-scaled saturation velocity (m/s)
    pub vsat_temp: f64,
    /// Temperature-scaled UA mobility degradation (m/V)
    pub ua_temp: f64,
    /// Temperature-scaled UB mobility degradation (m/V)^2
    pub ub_temp: f64,
    /// Temperature-scaled UC mobility degradation (m/V^2)
    pub uc_temp: f64,
}

impl Bsim3Derived {
    /// Compute derived parameters from model parameters.
    pub fn from_params(p: &Bsim3Params) -> Self {
        let leff = p.leff();
        let weff = p.weff();
        let cox = p.cox();
        let vt = p.vt();

        // Surface potential: phi = 2 * vt * ln(Nch / ni)
        // At 300K with Nch = 1.7e17: phi ~ 0.88V
        let phi = 2.0 * vt * (p.nch / Bsim3Params::NI).ln();
        let sqrt_phi = phi.sqrt();

        // Body effect coefficient
        // sqrtk1 = sqrt(2 * q * eps_si * Nch) for body effect calculation
        let sqrtk1 = (2.0 * Bsim3Params::Q * Bsim3Params::EPS_SI * p.nch * 1e6).sqrt();
        let k1ox = sqrtk1 / cox;

        // Convert mobility from cm^2/V-s to m^2/V-s
        let u0_si = p.u0 * 1e-4;

        // Characteristic length for SCE
        // lt = sqrt(eps_si * Xdep / Cox) where Xdep ~ sqrt(2*eps_si*phi/(q*Nch))
        // Simplified: lt ~ sqrt(eps_si * tox / eps_ox) * sqrt(phi * eps_si / (q * Nch))
        // For typical values, lt ~ 20-50nm
        let xdep = (2.0 * Bsim3Params::EPS_SI * phi / (Bsim3Params::Q * p.nch * 1e6)).sqrt();
        let lt = (Bsim3Params::EPS_SI * xdep * p.tox / Bsim3Params::EPS_OX).sqrt();

        // Source/drain resistance: Rds = rdsw / Weff
        // rdsw is in ohm-um, so convert Weff to um
        let rds = p.rdsw / (weff * 1e6);

        // Bulk charge effect coefficient (simplified)
        // In full BSIM3, this is more complex with A0, AGS, B0, B1
        // For now, use simplified: abulk0 = 1 + k1ox / (2 * sqrt_phi)
        let abulk0 = 1.0 + k1ox / (2.0 * sqrt_phi);

        // Built-in potential for source/drain junctions
        // Vbi = Vt * ln(Nch * Ngate / ni²) for n+/p junction
        // Simplified: Vbi ≈ 0.9V for typical doping levels
        // This is used in the SCE formula: ΔVth_SCE = -2*(Vbi - φs)*DVT0*θ*(1+DVT2*Vbs)
        // where φs is half the surface potential at threshold
        let vbi = vt * ((p.nch * 1e6) / Bsim3Params::NI).ln() + 0.56; // Add half bandgap

        // Overlap capacitances (Phase 3)
        // Cgs_ov = CGSO * Weff, Cgd_ov = CGDO * Weff, Cgb_ov = CGBO * Leff
        let cgs_ov = p.cgso * weff;
        let cgd_ov = p.cgdo * weff;
        let cgb_ov = p.cgbo * leff;

        // Source/drain diffusion areas and perimeters
        // If not specified (zero), use typical estimates based on width
        // Typical diffusion length is ~0.5um for modern processes
        let diff_length = 0.5e-6; // 0.5um default diffusion length
        let as_eff = if p.as_ > 0.0 {
            p.as_
        } else {
            weff * diff_length
        };
        let ad_eff = if p.ad > 0.0 { p.ad } else { weff * diff_length };
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

        // Temperature scaling (Phase 4)
        // Default: use nominal temperature
        let temp = p.tnom;
        let temp_ratio = 1.0;

        // At nominal temperature, no scaling applied
        let vth0_temp = 0.0; // No Vth shift at Tnom
        let vsat_temp = p.vsat;
        let ua_temp = p.ua;
        let ub_temp = p.ub;
        let uc_temp = p.uc;

        Self {
            leff,
            weff,
            cox,
            vt,
            sqrtk1,
            phi,
            sqrt_phi,
            k1ox,
            u0_si,
            lt,
            rds,
            abulk0,
            vbi,
            cgs_ov,
            cgd_ov,
            cgb_ov,
            as_eff,
            ad_eff,
            ps_eff,
            pd_eff,
            temp,
            temp_ratio,
            vth0_temp,
            vsat_temp,
            ua_temp,
            ub_temp,
            uc_temp,
        }
    }

    /// Compute derived parameters at a specific operating temperature.
    ///
    /// This applies BSIM3 temperature scaling equations:
    /// - Vth(T) = Vth0 + KT1*(T/Tnom - 1) + KT1L/Leff*(T/Tnom - 1) + KT2*Vbs*(T/Tnom - 1)
    /// - u0(T) = u0 * (T/Tnom)^UTE
    /// - Vsat(T) = Vsat - AT*(T - Tnom)
    /// - UA(T) = UA + UA1*(T - Tnom)
    /// - UB(T) = UB + UB1*(T - Tnom)
    /// - UC(T) = UC + UC1*(T - Tnom)
    /// - Rds(T) = Rds * (1 + PRT*(T - Tnom))
    pub fn from_params_at_temp(p: &Bsim3Params, temp: f64) -> Self {
        let leff = p.leff();
        let weff = p.weff();
        let cox = p.cox();

        // Thermal voltage at operating temperature
        let vt = p.vt_at(temp);

        // Temperature ratio
        let temp_ratio = temp / p.tnom;
        let delta_t = temp - p.tnom;

        // Surface potential scales with temperature
        // phi(T) ≈ phi(Tnom) * T/Tnom (simplified)
        let phi_tnom = 2.0 * p.vt() * (p.nch / Bsim3Params::NI).ln();
        let phi = phi_tnom * temp_ratio;
        let sqrt_phi = phi.sqrt();

        // Body effect coefficient
        let sqrtk1 = (2.0 * Bsim3Params::Q * Bsim3Params::EPS_SI * p.nch * 1e6).sqrt();
        let k1ox = sqrtk1 / cox;

        // Temperature-scaled mobility
        // u0(T) = u0(Tnom) * (T/Tnom)^UTE
        // UTE is typically negative (-1.5), so mobility decreases with temperature
        let u0_scaled = p.u0 * temp_ratio.powf(p.ute);
        let u0_si = u0_scaled * 1e-4;

        // Characteristic length for SCE
        let xdep = (2.0 * Bsim3Params::EPS_SI * phi / (Bsim3Params::Q * p.nch * 1e6)).sqrt();
        let lt = (Bsim3Params::EPS_SI * xdep * p.tox / Bsim3Params::EPS_OX).sqrt();

        // Temperature-scaled source/drain resistance
        // Rds(T) = Rds(Tnom) * (1 + PRT * (T - Tnom))
        let rds_tnom = p.rdsw / (weff * 1e6);
        let rds = rds_tnom * (1.0 + p.prt * delta_t);

        // Bulk charge effect coefficient
        let abulk0 = 1.0 + k1ox / (2.0 * sqrt_phi);

        // Built-in potential (temperature-dependent via Vt)
        let vbi = vt * ((p.nch * 1e6) / Bsim3Params::NI).ln() + 0.56;

        // Overlap capacitances (not temperature dependent)
        let cgs_ov = p.cgso * weff;
        let cgd_ov = p.cgdo * weff;
        let cgb_ov = p.cgbo * leff;

        // Diffusion areas and perimeters
        let diff_length = 0.5e-6;
        let as_eff = if p.as_ > 0.0 {
            p.as_
        } else {
            weff * diff_length
        };
        let ad_eff = if p.ad > 0.0 { p.ad } else { weff * diff_length };
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

        // Temperature-scaled threshold voltage shift
        // dVth = KT1*(T/Tnom - 1) + KT1L/Leff*(T/Tnom - 1)
        let vth0_temp = p.kt1 * (temp_ratio - 1.0) + p.kt1l / leff * (temp_ratio - 1.0);

        // Temperature-scaled saturation velocity
        // Vsat(T) = Vsat - AT*(T - Tnom)
        let vsat_temp = (p.vsat - p.at * delta_t).max(1e4);

        // Temperature-scaled mobility degradation coefficients
        // UA(T) = UA + UA1*(T - Tnom)
        // UB(T) = UB + UB1*(T - Tnom)
        // UC(T) = UC + UC1*(T - Tnom)
        let ua_temp = p.ua + p.ua1 * delta_t;
        let ub_temp = p.ub + p.ub1 * delta_t;
        let uc_temp = p.uc + p.uc1 * delta_t;

        Self {
            leff,
            weff,
            cox,
            vt,
            sqrtk1,
            phi,
            sqrt_phi,
            k1ox,
            u0_si,
            lt,
            rds,
            abulk0,
            vbi,
            cgs_ov,
            cgd_ov,
            cgb_ov,
            as_eff,
            ad_eff,
            ps_eff,
            pd_eff,
            temp,
            temp_ratio,
            vth0_temp,
            vsat_temp,
            ua_temp,
            ub_temp,
            uc_temp,
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
        let params = Bsim3Params::nmos_default();
        let derived = Bsim3Derived::from_params(&params);

        // Check effective dimensions
        assert!((derived.leff - 100e-9).abs() < 1e-12);
        assert!((derived.weff - 1e-6).abs() < 1e-12);

        // Check thermal voltage (~25.9mV at 300K)
        assert!((derived.vt - 0.0259).abs() < 0.001);

        // Check oxide capacitance
        assert!(derived.cox > 3e-3 && derived.cox < 4e-3);

        // Check surface potential (should be ~0.8-1.0V for typical doping)
        assert!(derived.phi > 0.7 && derived.phi < 1.1);

        // Check characteristic length (should be ~20-50nm for modern processes)
        assert!(derived.lt > 10e-9 && derived.lt < 100e-9);
    }

    #[test]
    fn test_derived_with_lint() {
        let mut params = Bsim3Params::nmos_default();
        params.l = 100e-9;
        params.lint = 10e-9;
        params.w = 1e-6;
        params.wint = 20e-9;

        let derived = Bsim3Derived::from_params(&params);

        // Leff = 100nm - 2*10nm = 80nm
        assert!((derived.leff - 80e-9).abs() < 1e-12);
        // Weff = 1um - 2*20nm = 960nm
        assert!((derived.weff - 960e-9).abs() < 1e-12);
    }

    #[test]
    fn test_source_drain_resistance() {
        let mut params = Bsim3Params::nmos_default();
        params.rdsw = 200.0; // ohm-um
        params.w = 1e-6; // 1um

        let derived = Bsim3Derived::from_params(&params);

        // Rds = 200 ohm-um / 1um = 200 ohms
        assert!((derived.rds - 200.0).abs() < 1e-6);
    }
}
