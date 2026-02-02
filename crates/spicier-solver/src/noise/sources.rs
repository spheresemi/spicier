//! Noise source definitions and spectral density calculations.

/// Boltzmann constant (J/K)
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Elementary charge (C)
pub const ELECTRON_CHARGE: f64 = 1.602176634e-19;

/// Type of noise source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSourceType {
    /// Thermal noise from a resistor: Sv = 4kTR
    Thermal,
    /// Shot noise from current flow: Si = 2qI
    Shot,
    /// Flicker (1/f) noise: Si = Kf * I^Af / f
    Flicker,
}

/// A noise source in the circuit.
#[derive(Debug, Clone)]
pub struct NoiseSource {
    /// Name/identifier of the noise source (e.g., "R1", "D1_shot").
    pub name: String,
    /// Type of noise.
    pub source_type: NoiseSourceType,
    /// Node where the noise current is injected (positive terminal).
    pub node_pos: Option<usize>,
    /// Node where the noise current returns (negative terminal).
    pub node_neg: Option<usize>,
    /// For thermal noise: resistance in Ohms.
    /// For shot noise: DC current in Amps.
    /// For flicker noise: coefficient Kf.
    pub value: f64,
    /// For flicker noise: current exponent Af (typically 1 or 2).
    pub flicker_af: f64,
    /// For flicker noise: DC current through device.
    pub flicker_current: f64,
}

impl NoiseSource {
    /// Create a thermal noise source (resistor).
    ///
    /// # Arguments
    /// * `name` - Source identifier
    /// * `node_pos` - Positive node (None for ground)
    /// * `node_neg` - Negative node (None for ground)
    /// * `resistance` - Resistance in Ohms
    pub fn thermal(
        name: impl Into<String>,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        resistance: f64,
    ) -> Self {
        Self {
            name: name.into(),
            source_type: NoiseSourceType::Thermal,
            node_pos,
            node_neg,
            value: resistance,
            flicker_af: 0.0,
            flicker_current: 0.0,
        }
    }

    /// Create a shot noise source (diode, BJT junction).
    ///
    /// # Arguments
    /// * `name` - Source identifier
    /// * `node_pos` - Positive node (None for ground)
    /// * `node_neg` - Negative node (None for ground)
    /// * `current` - DC current through junction in Amps
    pub fn shot(
        name: impl Into<String>,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        current: f64,
    ) -> Self {
        Self {
            name: name.into(),
            source_type: NoiseSourceType::Shot,
            node_pos,
            node_neg,
            value: current.abs(), // Use absolute value of current
            flicker_af: 0.0,
            flicker_current: 0.0,
        }
    }

    /// Create a flicker (1/f) noise source.
    ///
    /// # Arguments
    /// * `name` - Source identifier
    /// * `node_pos` - Positive node (None for ground)
    /// * `node_neg` - Negative node (None for ground)
    /// * `kf` - Flicker noise coefficient
    /// * `af` - Current exponent (typically 1 or 2)
    /// * `current` - DC current through device
    pub fn flicker(
        name: impl Into<String>,
        node_pos: Option<usize>,
        node_neg: Option<usize>,
        kf: f64,
        af: f64,
        current: f64,
    ) -> Self {
        Self {
            name: name.into(),
            source_type: NoiseSourceType::Flicker,
            node_pos,
            node_neg,
            value: kf,
            flicker_af: af,
            flicker_current: current.abs(),
        }
    }

    /// Compute the noise current spectral density Si (A²/Hz) at a given frequency.
    ///
    /// # Arguments
    /// * `frequency` - Frequency in Hz
    /// * `temperature` - Temperature in Kelvin
    ///
    /// # Returns
    /// Current noise spectral density in A²/Hz
    pub fn current_spectral_density(&self, frequency: f64, temperature: f64) -> f64 {
        match self.source_type {
            NoiseSourceType::Thermal => {
                // Thermal noise: Si = 4kT/R (A²/Hz)
                // Sv = 4kTR, Si = Sv/R² = 4kT/R
                if self.value > 0.0 {
                    4.0 * BOLTZMANN * temperature / self.value
                } else {
                    0.0
                }
            }
            NoiseSourceType::Shot => {
                // Shot noise: Si = 2qI (A²/Hz)
                2.0 * ELECTRON_CHARGE * self.value
            }
            NoiseSourceType::Flicker => {
                // Flicker noise: Si = Kf * I^Af / f (A²/Hz)
                if frequency > 0.0 {
                    self.value * self.flicker_current.powf(self.flicker_af) / frequency
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute the equivalent noise current (RMS) in A/√Hz.
    pub fn noise_current_density(&self, frequency: f64, temperature: f64) -> f64 {
        self.current_spectral_density(frequency, temperature).sqrt()
    }

    /// Compute the noise voltage spectral density Sv (V²/Hz) for a resistor.
    ///
    /// Only valid for thermal noise sources.
    pub fn voltage_spectral_density(&self, temperature: f64) -> f64 {
        match self.source_type {
            NoiseSourceType::Thermal => {
                // Thermal noise: Sv = 4kTR (V²/Hz)
                4.0 * BOLTZMANN * temperature * self.value
            }
            _ => {
                // For current noise sources, need transfer function to convert to voltage
                0.0
            }
        }
    }
}

/// Compute thermal noise voltage density for a resistor.
///
/// # Arguments
/// * `resistance` - Resistance in Ohms
/// * `temperature` - Temperature in Kelvin (default 300K = 27°C)
///
/// # Returns
/// Noise voltage density in V/√Hz
pub fn thermal_noise_voltage(resistance: f64, temperature: f64) -> f64 {
    (4.0 * BOLTZMANN * temperature * resistance).sqrt()
}

/// Compute shot noise current density.
///
/// # Arguments
/// * `current` - DC current in Amps
///
/// # Returns
/// Noise current density in A/√Hz
pub fn shot_noise_current(current: f64) -> f64 {
    (2.0 * ELECTRON_CHARGE * current.abs()).sqrt()
}

/// Compute flicker noise current density at a given frequency.
///
/// # Arguments
/// * `kf` - Flicker noise coefficient
/// * `af` - Current exponent
/// * `current` - DC current in Amps
/// * `frequency` - Frequency in Hz
///
/// # Returns
/// Noise current density in A/√Hz
pub fn flicker_noise_current(kf: f64, af: f64, current: f64, frequency: f64) -> f64 {
    if frequency > 0.0 {
        (kf * current.abs().powf(af) / frequency).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_noise_1k_resistor() {
        // 1kΩ resistor at 300K should have ~4.07 nV/√Hz
        let vn = thermal_noise_voltage(1000.0, 300.0);
        assert!((vn - 4.07e-9).abs() < 0.1e-9, "Expected ~4.07 nV/√Hz, got {}", vn);
    }

    #[test]
    fn test_thermal_noise_source() {
        let source = NoiseSource::thermal("R1", Some(0), Some(1), 1000.0);
        let sv = source.voltage_spectral_density(300.0);
        // Sv = 4kTR = 4 * 1.38e-23 * 300 * 1000 = 1.656e-17 V²/Hz
        assert!((sv - 1.656e-17).abs() < 0.01e-17);
    }

    #[test]
    fn test_shot_noise_1ma() {
        // 1mA current should have ~17.9 pA/√Hz shot noise
        let in_noise = shot_noise_current(1e-3);
        assert!((in_noise - 17.9e-12).abs() < 0.5e-12, "Expected ~17.9 pA/√Hz, got {}", in_noise);
    }

    #[test]
    fn test_shot_noise_source() {
        let source = NoiseSource::shot("D1", Some(0), Some(1), 1e-3);
        let si = source.current_spectral_density(1000.0, 300.0);
        // Si = 2qI = 2 * 1.6e-19 * 1e-3 = 3.2e-22 A²/Hz
        assert!((si - 3.2e-22).abs() < 0.1e-22);
    }

    #[test]
    fn test_flicker_noise() {
        // Flicker noise with Kf=1e-24, Af=1, I=1mA at 100Hz
        let source = NoiseSource::flicker("M1", Some(0), Some(1), 1e-24, 1.0, 1e-3);
        let si = source.current_spectral_density(100.0, 300.0);
        // Si = Kf * I^Af / f = 1e-24 * 1e-3 / 100 = 1e-29 A²/Hz
        assert!((si - 1e-29).abs() < 1e-31);
    }

    #[test]
    fn test_flicker_noise_frequency_dependence() {
        let source = NoiseSource::flicker("M1", Some(0), Some(1), 1e-24, 1.0, 1e-3);
        let si_100 = source.current_spectral_density(100.0, 300.0);
        let si_1000 = source.current_spectral_density(1000.0, 300.0);
        // Flicker noise should be 10x higher at 100Hz vs 1000Hz
        assert!((si_100 / si_1000 - 10.0).abs() < 0.01);
    }
}
