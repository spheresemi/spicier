//! Engineering units and SI prefix handling.

/// Parse a SPICE-style value with optional SI suffix.
///
/// Supported suffixes:
/// - T (tera, 1e12)
/// - G (giga, 1e9)
/// - MEG (mega, 1e6)
/// - K (kilo, 1e3)
/// - M (milli, 1e-3)
/// - U (micro, 1e-6)
/// - N (nano, 1e-9)
/// - P (pico, 1e-12)
/// - F (femto, 1e-15)
pub fn parse_value(s: &str) -> Option<f64> {
    let s = s.trim().to_uppercase();

    // Try to parse as plain number first
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }

    // Find where the numeric part ends
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+' && c != 'E')
        .unwrap_or(s.len());

    if num_end == 0 {
        return None;
    }

    let (num_str, suffix) = s.split_at(num_end);
    let value: f64 = num_str.parse().ok()?;

    let multiplier = match suffix {
        "T" => 1e12,
        "G" => 1e9,
        "MEG" => 1e6,
        "K" => 1e3,
        "" => 1.0,
        "M" => 1e-3,
        "MIL" => 25.4e-6, // mil = 1/1000 inch
        "U" => 1e-6,
        "N" => 1e-9,
        "P" => 1e-12,
        "F" => 1e-15,
        _ => return None,
    };

    Some(value * multiplier)
}

/// Format a value with appropriate SI prefix.
pub fn format_value(value: f64) -> String {
    let abs_value = value.abs();

    let (scaled, suffix) = if abs_value >= 1e12 {
        (value / 1e12, "T")
    } else if abs_value >= 1e9 {
        (value / 1e9, "G")
    } else if abs_value >= 1e6 {
        (value / 1e6, "M")
    } else if abs_value >= 1e3 {
        (value / 1e3, "k")
    } else if abs_value >= 1.0 {
        (value, "")
    } else if abs_value >= 1e-3 {
        (value * 1e3, "m")
    } else if abs_value >= 1e-6 {
        (value * 1e6, "u")
    } else if abs_value >= 1e-9 {
        (value * 1e9, "n")
    } else if abs_value >= 1e-12 {
        (value * 1e12, "p")
    } else if abs_value >= 1e-15 {
        (value * 1e15, "f")
    } else if abs_value == 0.0 {
        (0.0, "")
    } else {
        (value, "")
    };

    format!("{:.4}{}", scaled, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_number() {
        assert_eq!(parse_value("1.5"), Some(1.5));
        assert_eq!(parse_value("-2.5"), Some(-2.5));
        assert_eq!(parse_value("1e-3"), Some(1e-3));
    }

    #[test]
    fn test_parse_with_suffix() {
        fn approx_eq(a: Option<f64>, b: f64) -> bool {
            a.is_some_and(|v| (v - b).abs() < b.abs() * 1e-10 + 1e-20)
        }

        assert!(approx_eq(parse_value("1k"), 1e3));
        assert!(approx_eq(parse_value("4.7K"), 4.7e3));
        assert!(approx_eq(parse_value("10M"), 10e-3));
        assert!(approx_eq(parse_value("10MEG"), 10e6));
        assert!(approx_eq(parse_value("100n"), 100e-9));
        assert!(approx_eq(parse_value("1u"), 1e-6));
        assert!(approx_eq(parse_value("10p"), 10e-12));
    }

    #[test]
    fn test_parse_invalid() {
        assert_eq!(parse_value("abc"), None);
        assert_eq!(parse_value(""), None);
    }

    #[test]
    fn test_format_value() {
        assert_eq!(format_value(1000.0), "1.0000k");
        assert_eq!(format_value(0.001), "1.0000m");
        assert_eq!(format_value(1e-9), "1.0000n");
    }
}
