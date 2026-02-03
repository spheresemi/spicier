//! BSIM3v3.3 model parameters.
//!
//! This module defines the parameter structs for the BSIM3v3.3 model.
//! Phase 1 (BSIM3-lite): ~30 core parameters
//! Phase 2 (Enhanced DC): Additional DIBL, width effects, substrate current

use super::super::level1::MosfetType;

/// BSIM3v3.3 model parameters.
///
/// This struct contains the BSIM3 parameters needed for DC analysis
/// with short-channel effects. The full BSIM3 model has 200+ parameters;
/// this is a subset covering the essential physics.
#[derive(Debug, Clone)]
pub struct Bsim3Params {
    // ========================================
    // Geometry Parameters
    // ========================================
    /// Gate oxide thickness (m). Default: 9e-9 (9 nm)
    pub tox: f64,
    /// Channel length reduction per side (m). Default: 0.0
    pub lint: f64,
    /// Channel width reduction per side (m). Default: 0.0
    pub wint: f64,

    // ========================================
    // Threshold Voltage Parameters
    // ========================================
    /// Zero-bias threshold voltage (V). Default: 0.7 (NMOS), -0.7 (PMOS)
    pub vth0: f64,
    /// First-order body effect coefficient (V^0.5). Default: 0.5
    pub k1: f64,
    /// Second-order body effect coefficient. Default: 0.0
    pub k2: f64,
    /// Short-channel effect coefficient 0. Default: 2.2
    pub dvt0: f64,
    /// Short-channel effect coefficient 1. Default: 0.53
    pub dvt1: f64,
    /// Body-bias coefficient for SCE. Default: -0.032
    pub dvt2: f64,
    /// Lateral non-uniform doping parameter (m). Default: 1.74e-7
    pub nlx: f64,
    /// Threshold voltage offset in subthreshold (V). Default: -0.11
    pub voff: f64,
    /// Subthreshold swing factor. Default: 1.0
    pub nfactor: f64,

    // ========================================
    // Width Effect Parameters (Phase 2)
    // ========================================
    /// Narrow width threshold voltage coefficient. Default: 0.0 (disable by default)
    pub k3: f64,
    /// Body effect coefficient for narrow width. Default: 0.0
    pub k3b: f64,
    /// Narrow width parameter (m). Default: 0.0
    pub w0: f64,
    /// Narrow width SCE coefficient 0. Default: 0.0
    pub dvt0w: f64,
    /// Narrow width SCE coefficient 1. Default: 0.0 (disable by default)
    pub dvt1w: f64,
    /// Narrow width SCE body coefficient. Default: 0.0
    pub dvt2w: f64,

    // ========================================
    // Mobility Parameters
    // ========================================
    /// Low-field mobility (cm^2/V-s). Default: 670 (NMOS), 250 (PMOS)
    pub u0: f64,
    /// First-order mobility degradation (m/V). Default: 2.25e-9
    pub ua: f64,
    /// Second-order mobility degradation (m/V)^2. Default: 5.87e-19
    pub ub: f64,
    /// Body-bias mobility degradation (m/V^2). Default: -4.65e-11
    pub uc: f64,
    /// Saturation velocity (m/s). Default: 1.5e5
    pub vsat: f64,

    // ========================================
    // Output Conductance Parameters
    // ========================================
    /// Channel length modulation parameter. Default: 1.3
    pub pclm: f64,
    /// DIBL coefficient 1. Default: 0.39
    pub pdiblc1: f64,
    /// DIBL coefficient 2. Default: 0.0086
    pub pdiblc2: f64,
    /// DIBL output resistance parameter. Default: 0.56
    pub drout: f64,
    /// Effective Vds smoothing parameter (V). Default: 0.01
    pub delta: f64,

    // ========================================
    // DIBL Parameters
    // ========================================
    /// DIBL coefficient. Default: 0.08
    pub eta0: f64,
    /// Body-bias DIBL coefficient (1/V). Default: -0.07
    pub etab: f64,
    /// DIBL coefficient in subthreshold. Default: 0.56
    pub dsub: f64,

    // ========================================
    // Enhanced DIBL Parameters (Phase 2)
    // ========================================
    /// Body effect on DIBL output resistance. Default: 0.0
    pub pdiblcb: f64,
    /// DIBL to Rout parameter. Default: 0.0
    pub fprout: f64,
    /// DIBL multiplier for Vth. Default: 1.0
    pub pvag: f64,

    // ========================================
    // Substrate Current Parameters (Phase 2)
    // ========================================
    /// Substrate current impact ionization coefficient. Default: 0.0
    pub alpha0: f64,
    /// Substrate current exponent. Default: 30.0
    pub beta0: f64,

    // ========================================
    // Parasitic Resistance
    // ========================================
    /// Source/drain resistance per width (ohm-um). Default: 200
    pub rdsw: f64,
    /// Zero-bias drain resistance. Default: 0.0
    pub rd: f64,
    /// Zero-bias source resistance. Default: 0.0
    pub rs: f64,
    /// Width dependence of Rds. Default: 0.0
    pub prwb: f64,
    /// Gate-bias dependence of Rds. Default: 0.0
    pub prwg: f64,

    // ========================================
    // Process Parameters
    // ========================================
    /// Channel doping concentration (cm^-3). Default: 1.7e17
    pub nch: f64,
    /// Gate doping concentration (cm^-3). Default: 1e20
    pub ngate: f64,
    /// Substrate doping concentration (cm^-3). Default: 1e15
    pub nsub: f64,
    /// Vertical doping profile characteristic length (m). Default: 0.0
    pub xt: f64,

    // ========================================
    // Capacitance Parameters (Phase 3)
    // ========================================
    /// Gate-source overlap capacitance per unit width (F/m). Default: 0.0
    pub cgso: f64,
    /// Gate-drain overlap capacitance per unit width (F/m). Default: 0.0
    pub cgdo: f64,
    /// Gate-bulk overlap capacitance per unit length (F/m). Default: 0.0
    pub cgbo: f64,
    /// Zero-bias bulk-drain junction capacitance per unit area (F/m^2). Default: 5e-4
    pub cj: f64,
    /// Zero-bias bulk-drain sidewall capacitance per unit length (F/m). Default: 5e-10
    pub cjsw: f64,
    /// Zero-bias gate-edge sidewall capacitance per unit length (F/m). Default: 0.0
    pub cjswg: f64,
    /// Bulk junction bottom grading coefficient. Default: 0.5
    pub mj: f64,
    /// Bulk junction sidewall grading coefficient. Default: 0.33
    pub mjsw: f64,
    /// Bulk junction gate-side sidewall grading coefficient. Default: 0.33
    pub mjswg: f64,
    /// Bulk junction built-in potential (V). Default: 1.0
    pub pb: f64,
    /// Bulk junction sidewall built-in potential (V). Default: 1.0
    pub pbsw: f64,
    /// Bulk junction gate-side sidewall built-in potential (V). Default: 1.0
    pub pbswg: f64,

    // ========================================
    // Temperature Parameters (Phase 4)
    // ========================================
    /// Nominal temperature for parameter extraction (K). Default: 300.15
    pub tnom: f64,
    /// First-order Vth temperature coefficient (V/K). Default: -0.11
    pub kt1: f64,
    /// Body-bias Vth temperature coefficient (V/K). Default: 0.022
    pub kt1l: f64,
    /// Second-order Vth temperature coefficient (V/K^2). Default: 0.022
    pub kt2: f64,
    /// Mobility temperature exponent. Default: -1.5
    pub ute: f64,
    /// UA temperature coefficient (m/V/K). Default: 4.31e-9
    pub ua1: f64,
    /// UB temperature coefficient ((m/V)^2/K). Default: -7.61e-18
    pub ub1: f64,
    /// UC temperature coefficient (m/V^2/K). Default: -5.6e-11
    pub uc1: f64,
    /// Saturation velocity temperature coefficient (m/s/K). Default: 3.3e2
    pub at: f64,
    /// RDSW temperature coefficient (ohm-um/K). Default: 0.0
    pub prt: f64,

    // ========================================
    // Instance Parameters (set per device)
    // ========================================
    /// Channel width (m). Default: 1e-6
    pub w: f64,
    /// Channel length (m). Default: 100e-9
    pub l: f64,
    /// Number of fingers. Default: 1
    pub nf: f64,
    /// Source diffusion area (m^2). Default: 0.0 (auto-calculated if zero)
    pub as_: f64,
    /// Drain diffusion area (m^2). Default: 0.0 (auto-calculated if zero)
    pub ad: f64,
    /// Source diffusion perimeter (m). Default: 0.0 (auto-calculated if zero)
    pub ps: f64,
    /// Drain diffusion perimeter (m). Default: 0.0 (auto-calculated if zero)
    pub pd: f64,
    /// Device type (set by model type)
    pub mos_type: MosfetType,
}

impl Bsim3Params {
    /// Physical constants used in BSIM3 calculations.
    pub const Q: f64 = 1.602176634e-19; // Elementary charge (C)
    pub const KB: f64 = 1.380649e-23; // Boltzmann constant (J/K)
    pub const EPS_SI: f64 = 1.03594e-10; // Permittivity of Si (F/m) = 11.7 * eps0
    pub const EPS_OX: f64 = 3.45314e-11; // Permittivity of SiO2 (F/m) = 3.9 * eps0
    pub const NI: f64 = 1.45e10; // Intrinsic carrier concentration at 300K (cm^-3)
    pub const T_NOM: f64 = 300.15; // Nominal temperature (K)

    /// Create default NMOS BSIM3 parameters.
    pub fn nmos_default() -> Self {
        Self {
            // Geometry
            tox: 9e-9,
            lint: 0.0,
            wint: 0.0,

            // Threshold voltage
            vth0: 0.4,
            k1: 0.5,
            k2: 0.0,
            dvt0: 2.2,
            dvt1: 0.53,
            dvt2: -0.032,
            nlx: 1.74e-7,
            voff: -0.11,
            nfactor: 1.0,

            // Width effects (Phase 2) - disabled by default for wide devices
            k3: 0.0,
            k3b: 0.0,
            w0: 0.0,
            dvt0w: 0.0,
            dvt1w: 0.0,
            dvt2w: 0.0,

            // Mobility
            u0: 670.0,
            ua: 2.25e-9,
            ub: 5.87e-19,
            uc: -4.65e-11,
            vsat: 1.5e5,

            // Output conductance
            pclm: 1.3,
            pdiblc1: 0.39,
            pdiblc2: 0.0086,
            drout: 0.56,
            delta: 0.01,

            // DIBL
            eta0: 0.08,
            etab: -0.07,
            dsub: 0.56,

            // Enhanced DIBL (Phase 2)
            pdiblcb: 0.0,
            fprout: 0.0,
            pvag: 1.0,

            // Substrate current (Phase 2)
            alpha0: 0.0,
            beta0: 30.0,

            // Parasitic resistance
            rdsw: 200.0,
            rd: 0.0,
            rs: 0.0,
            prwb: 0.0,
            prwg: 0.0,

            // Process
            nch: 1.7e17,
            ngate: 1e20,
            nsub: 1e15,
            xt: 0.0,

            // Capacitances (Phase 3)
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,
            cj: 5e-4,
            cjsw: 5e-10,
            cjswg: 0.0,
            mj: 0.5,
            mjsw: 0.33,
            mjswg: 0.33,
            pb: 1.0,
            pbsw: 1.0,
            pbswg: 1.0,

            // Temperature (Phase 4)
            tnom: 300.15,
            kt1: -0.11,
            kt1l: 0.0,
            kt2: 0.022,
            ute: -1.5,
            ua1: 4.31e-9,
            ub1: -7.61e-18,
            uc1: -5.6e-11,
            at: 3.3e2, // Reasonable: ~22% Vsat reduction over 100K
            prt: 0.0,

            // Instance (defaults)
            w: 1e-6,
            l: 100e-9,
            nf: 1.0,
            as_: 0.0,
            ad: 0.0,
            ps: 0.0,
            pd: 0.0,
            mos_type: MosfetType::Nmos,
        }
    }

    /// Create default PMOS BSIM3 parameters.
    pub fn pmos_default() -> Self {
        Self {
            // Geometry
            tox: 9e-9,
            lint: 0.0,
            wint: 0.0,

            // Threshold voltage (negative for PMOS)
            vth0: -0.4,
            k1: 0.5,
            k2: 0.0,
            dvt0: 2.2,
            dvt1: 0.53,
            dvt2: -0.032,
            nlx: 1.74e-7,
            voff: -0.11,
            nfactor: 1.0,

            // Width effects (Phase 2) - disabled by default for wide devices
            k3: 0.0,
            k3b: 0.0,
            w0: 0.0,
            dvt0w: 0.0,
            dvt1w: 0.0,
            dvt2w: 0.0,

            // Mobility (lower for PMOS)
            u0: 250.0,
            ua: 2.25e-9,
            ub: 5.87e-19,
            uc: -4.65e-11,
            vsat: 1.0e5,

            // Output conductance
            pclm: 1.3,
            pdiblc1: 0.39,
            pdiblc2: 0.0086,
            drout: 0.56,
            delta: 0.01,

            // DIBL
            eta0: 0.08,
            etab: -0.07,
            dsub: 0.56,

            // Enhanced DIBL (Phase 2)
            pdiblcb: 0.0,
            fprout: 0.0,
            pvag: 1.0,

            // Substrate current (Phase 2)
            alpha0: 0.0,
            beta0: 30.0,

            // Parasitic resistance (higher for PMOS)
            rdsw: 300.0,
            rd: 0.0,
            rs: 0.0,
            prwb: 0.0,
            prwg: 0.0,

            // Process
            nch: 1.7e17,
            ngate: 1e20,
            nsub: 1e15,
            xt: 0.0,

            // Capacitances (Phase 3)
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,
            cj: 5e-4,
            cjsw: 5e-10,
            cjswg: 0.0,
            mj: 0.5,
            mjsw: 0.33,
            mjswg: 0.33,
            pb: 1.0,
            pbsw: 1.0,
            pbswg: 1.0,

            // Temperature (Phase 4)
            tnom: 300.15,
            kt1: -0.11,
            kt1l: 0.0,
            kt2: 0.022,
            ute: -1.5,
            ua1: 4.31e-9,
            ub1: -7.61e-18,
            uc1: -5.6e-11,
            at: 3.3e2, // Reasonable: ~22% Vsat reduction over 100K
            prt: 0.0,

            // Instance (defaults)
            w: 1e-6,
            l: 100e-9,
            nf: 1.0,
            as_: 0.0,
            ad: 0.0,
            ps: 0.0,
            pd: 0.0,
            mos_type: MosfetType::Pmos,
        }
    }

    /// Calculate thermal voltage at nominal temperature.
    #[inline]
    pub fn vt(&self) -> f64 {
        Self::KB * self.tnom / Self::Q
    }

    /// Calculate thermal voltage at a given temperature (K).
    #[inline]
    pub fn vt_at(&self, temp: f64) -> f64 {
        Self::KB * temp / Self::Q
    }

    /// Calculate oxide capacitance per unit area (F/m^2).
    #[inline]
    pub fn cox(&self) -> f64 {
        Self::EPS_OX / self.tox
    }

    /// Calculate effective channel length (m).
    #[inline]
    pub fn leff(&self) -> f64 {
        self.l - 2.0 * self.lint
    }

    /// Calculate effective channel width (m).
    #[inline]
    pub fn weff(&self) -> f64 {
        (self.w - 2.0 * self.wint) * self.nf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmos_defaults() {
        let params = Bsim3Params::nmos_default();
        assert_eq!(params.vth0, 0.4);
        assert_eq!(params.u0, 670.0);
        assert_eq!(params.tox, 9e-9);
        assert_eq!(params.mos_type, MosfetType::Nmos);
    }

    #[test]
    fn test_pmos_defaults() {
        let params = Bsim3Params::pmos_default();
        assert_eq!(params.vth0, -0.4);
        assert_eq!(params.u0, 250.0);
        assert_eq!(params.mos_type, MosfetType::Pmos);
    }

    #[test]
    fn test_thermal_voltage() {
        let params = Bsim3Params::nmos_default();
        let vt = params.vt();
        // Vt at 300K should be approximately 25.9 mV
        assert!((vt - 0.0259).abs() < 0.001);
    }

    #[test]
    fn test_oxide_capacitance() {
        let params = Bsim3Params::nmos_default();
        let cox = params.cox();
        // Cox = eps_ox / tox = 3.45e-11 / 9e-9 = 3.84e-3 F/m^2
        assert!((cox - 3.84e-3).abs() < 1e-4);
    }

    #[test]
    fn test_effective_dimensions() {
        let mut params = Bsim3Params::nmos_default();
        params.w = 1e-6;
        params.l = 100e-9;
        params.lint = 5e-9;
        params.wint = 10e-9;
        params.nf = 2.0;

        let leff = params.leff();
        let weff = params.weff();

        // Leff = 100nm - 2*5nm = 90nm
        assert!((leff - 90e-9).abs() < 1e-12);
        // Weff = (1um - 2*10nm) * 2 = 1.96um
        assert!((weff - 1.96e-6).abs() < 1e-12);
    }
}
