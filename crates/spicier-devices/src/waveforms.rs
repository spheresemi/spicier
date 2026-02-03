//! Time-varying source waveforms for transient analysis.
//!
//! This module provides waveform types that can be used with voltage and current sources
//! for transient simulations.

use std::f64::consts::PI;

/// A time-varying waveform specification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Waveform {
    /// Constant DC value (time-independent).
    Dc(f64),

    /// Pulse waveform: PULSE(V1 V2 TD TR TF PW PER)
    ///
    /// - V1: Initial value
    /// - V2: Pulsed value
    /// - TD: Delay time (before first pulse)
    /// - TR: Rise time
    /// - TF: Fall time
    /// - PW: Pulse width (at V2)
    /// - PER: Period (0 for single pulse)
    Pulse {
        v1: f64,
        v2: f64,
        td: f64,
        tr: f64,
        tf: f64,
        pw: f64,
        per: f64,
    },

    /// Sinusoidal waveform: SIN(VO VA FREQ TD THETA PHASE)
    ///
    /// - VO: DC offset
    /// - VA: Amplitude
    /// - FREQ: Frequency in Hz
    /// - TD: Delay time (before sinusoid starts)
    /// - THETA: Damping factor (1/s), 0 for undamped
    /// - PHASE: Phase in degrees
    Sin {
        vo: f64,
        va: f64,
        freq: f64,
        td: f64,
        theta: f64,
        phase: f64,
    },

    /// Piecewise linear waveform: PWL(T1 V1 T2 V2 ...)
    ///
    /// Linear interpolation between specified (time, value) points.
    Pwl {
        /// Time-value pairs, sorted by time.
        points: Vec<(f64, f64)>,
    },
}

impl Waveform {
    /// Create a DC waveform.
    pub fn dc(value: f64) -> Self {
        Waveform::Dc(value)
    }

    /// Create a pulse waveform with common defaults.
    ///
    /// - `v1`: Initial value
    /// - `v2`: Pulsed value
    /// - `td`: Delay time
    /// - `tr`: Rise time
    /// - `tf`: Fall time
    /// - `pw`: Pulse width
    /// - `per`: Period (0 for single pulse)
    pub fn pulse(v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64) -> Self {
        Waveform::Pulse {
            v1,
            v2,
            td,
            tr,
            tf,
            pw,
            per,
        }
    }

    /// Create a sinusoidal waveform.
    ///
    /// - `vo`: DC offset
    /// - `va`: Amplitude
    /// - `freq`: Frequency in Hz
    pub fn sin(vo: f64, va: f64, freq: f64) -> Self {
        Waveform::Sin {
            vo,
            va,
            freq,
            td: 0.0,
            theta: 0.0,
            phase: 0.0,
        }
    }

    /// Create a sinusoidal waveform with full parameters.
    pub fn sin_full(vo: f64, va: f64, freq: f64, td: f64, theta: f64, phase: f64) -> Self {
        Waveform::Sin {
            vo,
            va,
            freq,
            td,
            theta,
            phase,
        }
    }

    /// Create a piecewise linear waveform.
    pub fn pwl(points: Vec<(f64, f64)>) -> Self {
        Waveform::Pwl { points }
    }

    /// Evaluate the waveform at a given time.
    pub fn value_at(&self, time: f64) -> f64 {
        match self {
            Waveform::Dc(v) => *v,
            Waveform::Pulse {
                v1,
                v2,
                td,
                tr,
                tf,
                pw,
                per,
            } => eval_pulse(*v1, *v2, *td, *tr, *tf, *pw, *per, time),
            Waveform::Sin {
                vo,
                va,
                freq,
                td,
                theta,
                phase,
            } => eval_sin(*vo, *va, *freq, *td, *theta, *phase, time),
            Waveform::Pwl { points } => eval_pwl(points, time),
        }
    }

    /// Get the DC value (for operating point calculation).
    ///
    /// For PULSE, returns V1. For SIN, returns VO. For PWL, returns the first value.
    pub fn dc_value(&self) -> f64 {
        match self {
            Waveform::Dc(v) => *v,
            Waveform::Pulse { v1, .. } => *v1,
            Waveform::Sin { vo, .. } => *vo,
            Waveform::Pwl { points } => points.first().map(|(_, v)| *v).unwrap_or(0.0),
        }
    }
}

/// Evaluate a pulse waveform at time t.
#[allow(clippy::too_many_arguments)]
fn eval_pulse(v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64, t: f64) -> f64 {
    if t < td {
        return v1;
    }

    // Time within the period (or from delay if per=0)
    let t_rel = if per > 0.0 { (t - td) % per } else { t - td };

    // Pulse shape: rise -> high -> fall -> low
    if t_rel < tr {
        // Rising edge
        v1 + (v2 - v1) * t_rel / tr
    } else if t_rel < tr + pw {
        // Pulse high
        v2
    } else if t_rel < tr + pw + tf {
        // Falling edge
        v2 - (v2 - v1) * (t_rel - tr - pw) / tf
    } else {
        // Pulse low
        v1
    }
}

/// Evaluate a sinusoidal waveform at time t.
fn eval_sin(vo: f64, va: f64, freq: f64, td: f64, theta: f64, phase: f64, t: f64) -> f64 {
    if t < td {
        return vo;
    }

    let t_rel = t - td;
    let phase_rad = phase * PI / 180.0;

    // Damped sinusoid: vo + va * exp(-theta * t) * sin(2*pi*freq*t + phase)
    let damping = if theta > 0.0 {
        (-theta * t_rel).exp()
    } else {
        1.0
    };

    vo + va * damping * (2.0 * PI * freq * t_rel + phase_rad).sin()
}

/// Evaluate a piecewise linear waveform at time t.
fn eval_pwl(points: &[(f64, f64)], t: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Before first point: use first value
    if t <= points[0].0 {
        return points[0].1;
    }

    // After last point: use last value
    if t >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }

    // Find the interval containing t and interpolate
    for i in 0..points.len() - 1 {
        let (t0, v0) = points[i];
        let (t1, v1) = points[i + 1];

        if t >= t0 && t <= t1 {
            // Linear interpolation
            let frac = (t - t0) / (t1 - t0);
            return v0 + frac * (v1 - v0);
        }
    }

    // Shouldn't reach here, but return last value as fallback
    points[points.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_waveform() {
        let w = Waveform::dc(5.0);
        assert_eq!(w.value_at(0.0), 5.0);
        assert_eq!(w.value_at(1.0), 5.0);
        assert_eq!(w.dc_value(), 5.0);
    }

    #[test]
    fn test_pulse_waveform() {
        // PULSE(0 5 1m 0.1m 0.1m 1m 5m)
        // v1=0, v2=5, td=1ms, tr=0.1ms, tf=0.1ms, pw=1ms, per=5ms
        let w = Waveform::pulse(0.0, 5.0, 1e-3, 0.1e-3, 0.1e-3, 1e-3, 5e-3);

        // Before delay: v1
        assert_eq!(w.value_at(0.5e-3), 0.0);

        // At start of rise
        assert!((w.value_at(1e-3) - 0.0).abs() < 1e-10);

        // Middle of rise (50%)
        assert!((w.value_at(1.05e-3) - 2.5).abs() < 1e-10);

        // At peak (after rise)
        assert!((w.value_at(1.2e-3) - 5.0).abs() < 1e-10);

        // During pulse width
        assert!((w.value_at(1.5e-3) - 5.0).abs() < 1e-10);

        // After fall
        assert!((w.value_at(2.5e-3) - 0.0).abs() < 1e-10);

        // DC value is v1
        assert_eq!(w.dc_value(), 0.0);
    }

    #[test]
    fn test_sin_waveform() {
        // SIN(0 1 1k) - 1V amplitude sine at 1kHz
        let w = Waveform::sin(0.0, 1.0, 1000.0);

        // At t=0: sin(0) = 0
        assert!((w.value_at(0.0) - 0.0).abs() < 1e-10);

        // At t=0.25ms (quarter period): sin(pi/2) = 1
        assert!((w.value_at(0.25e-3) - 1.0).abs() < 1e-10);

        // At t=0.5ms (half period): sin(pi) = 0
        assert!((w.value_at(0.5e-3) - 0.0).abs() < 1e-10);

        // At t=0.75ms (3/4 period): sin(3*pi/2) = -1
        assert!((w.value_at(0.75e-3) - (-1.0)).abs() < 1e-10);

        // DC value is offset
        assert_eq!(w.dc_value(), 0.0);
    }

    #[test]
    fn test_pwl_waveform() {
        // PWL(0 0 1m 5 2m 5 3m 0)
        let w = Waveform::pwl(vec![(0.0, 0.0), (1e-3, 5.0), (2e-3, 5.0), (3e-3, 0.0)]);

        // At t=0
        assert_eq!(w.value_at(0.0), 0.0);

        // At t=0.5ms (midway to first point)
        assert!((w.value_at(0.5e-3) - 2.5).abs() < 1e-10);

        // At t=1ms
        assert!((w.value_at(1e-3) - 5.0).abs() < 1e-10);

        // Flat region
        assert!((w.value_at(1.5e-3) - 5.0).abs() < 1e-10);

        // Falling
        assert!((w.value_at(2.5e-3) - 2.5).abs() < 1e-10);

        // At end
        assert!((w.value_at(3e-3) - 0.0).abs() < 1e-10);

        // After end: hold last value
        assert!((w.value_at(5e-3) - 0.0).abs() < 1e-10);
    }
}
