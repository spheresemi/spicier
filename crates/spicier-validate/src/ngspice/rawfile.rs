//! Parser for ngspice rawfile format.
//!
//! ngspice outputs simulation data in a "rawfile" format that can be either ASCII
//! or binary. The format consists of a header section followed by data values.
//!
//! Header fields:
//! - Title: simulation title
//! - Plotname: type of analysis
//! - Flags: real or complex
//! - No. Variables: number of data columns
//! - No. Points: number of data rows
//! - Variables: list of variable names and types
//! - Values: (ASCII) or Binary: (binary) marker before data

use crate::error::{Error, Result};
use crate::ngspice::types::{RawVariable, RawfileData, RawfileHeader};

/// Parse a rawfile from bytes.
pub fn parse_rawfile(data: &[u8]) -> Result<RawfileData> {
    // Try to find the header end (either "Values:" or "Binary:")
    let data_str = String::from_utf8_lossy(data);

    // Parse header
    let header = parse_header(&data_str)?;

    if header.is_binary {
        parse_binary_data(data, &header)
    } else {
        parse_ascii_data(&data_str, &header)
    }
}

/// Parse the header section of the rawfile.
fn parse_header(data: &str) -> Result<RawfileHeader> {
    let mut title = String::new();
    let mut plotname = String::new();
    let mut flags = String::new();
    let mut num_variables = 0usize;
    let mut num_points = 0usize;
    let mut variables = Vec::new();
    let mut is_binary = false;
    let mut in_variables = false;

    for line in data.lines() {
        let line = line.trim();

        if line.starts_with("Title:") {
            title = line.strip_prefix("Title:").unwrap().trim().to_string();
        } else if line.starts_with("Plotname:") {
            plotname = line.strip_prefix("Plotname:").unwrap().trim().to_string();
        } else if line.starts_with("Flags:") {
            flags = line.strip_prefix("Flags:").unwrap().trim().to_string();
        } else if line.starts_with("No. Variables:") {
            let val = line.strip_prefix("No. Variables:").unwrap().trim();
            num_variables = val.parse().map_err(|_| {
                Error::RawfileParseError(format!("invalid No. Variables: {}", val))
            })?;
        } else if line.starts_with("No. Points:") {
            let val = line.strip_prefix("No. Points:").unwrap().trim();
            num_points = val
                .parse()
                .map_err(|_| Error::RawfileParseError(format!("invalid No. Points: {}", val)))?;
        } else if line.starts_with("Variables:") {
            in_variables = true;
        } else if line.starts_with("Values:") || line.starts_with("Binary:") {
            is_binary = line.starts_with("Binary:");
            break;
        } else if in_variables && !line.is_empty() {
            // Parse variable line: "index name type"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let index: usize = parts[0].parse().map_err(|_| {
                    Error::RawfileParseError(format!("invalid variable index: {}", parts[0]))
                })?;
                let name = parts[1].to_string();
                let var_type = parts[2].to_string();
                variables.push(RawVariable {
                    index,
                    name,
                    var_type,
                });
            }
        }
    }

    let is_complex = flags.to_lowercase().contains("complex");

    Ok(RawfileHeader {
        title,
        plotname,
        flags,
        num_variables,
        num_points,
        variables,
        is_complex,
        is_binary,
    })
}

/// Parse ASCII format data section.
fn parse_ascii_data(data: &str, header: &RawfileHeader) -> Result<RawfileData> {
    let mut real_data: Vec<Vec<f64>> = Vec::with_capacity(header.num_points);
    let mut imag_data: Option<Vec<Vec<f64>>> = if header.is_complex {
        Some(Vec::with_capacity(header.num_points))
    } else {
        None
    };

    // Find "Values:" line and parse data after it
    let values_start = data
        .find("Values:")
        .ok_or_else(|| Error::RawfileParseError("Values: marker not found".to_string()))?;

    let data_section = &data[values_start + "Values:".len()..];

    // ASCII format: each point is a block starting with point index,
    // followed by value lines (one per variable)
    let mut current_point_real: Vec<f64> = Vec::with_capacity(header.num_variables);
    let mut current_point_imag: Vec<f64> = Vec::with_capacity(header.num_variables);
    let mut expecting_index = true;

    for line in data_section.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if expecting_index {
            // Point index line (may have tab-separated first value)
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // First part is index, second (if present) is first value
            if parts.len() > 1 {
                if header.is_complex {
                    // Complex: value,value format
                    let val_str = parts[1];
                    if let Some((re, im)) = parse_complex_value(val_str) {
                        current_point_real.push(re);
                        current_point_imag.push(im);
                    }
                } else {
                    if let Ok(v) = parts[1].parse::<f64>() {
                        current_point_real.push(v);
                    }
                }
            }
            expecting_index = false;
        } else {
            // Value line
            if header.is_complex {
                if let Some((re, im)) = parse_complex_value(line) {
                    current_point_real.push(re);
                    current_point_imag.push(im);
                }
            } else if let Ok(v) = line.parse::<f64>() {
                current_point_real.push(v);
            }
        }

        // Check if we've collected all variables for this point
        if current_point_real.len() == header.num_variables {
            real_data.push(std::mem::take(&mut current_point_real));
            if header.is_complex {
                imag_data
                    .as_mut()
                    .unwrap()
                    .push(std::mem::take(&mut current_point_imag));
            }
            current_point_real = Vec::with_capacity(header.num_variables);
            current_point_imag = Vec::with_capacity(header.num_variables);
            expecting_index = true;
        }
    }

    Ok(RawfileData {
        header: header.clone(),
        real_data,
        imag_data,
    })
}

/// Parse a complex value in ngspice format: "real,imag" or "real, imag".
fn parse_complex_value(s: &str) -> Option<(f64, f64)> {
    let s = s.trim();
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() == 2 {
        let re = parts[0].trim().parse::<f64>().ok()?;
        let im = parts[1].trim().parse::<f64>().ok()?;
        Some((re, im))
    } else {
        // Try as just a real value
        let v = s.parse::<f64>().ok()?;
        Some((v, 0.0))
    }
}

/// Parse binary format data section.
fn parse_binary_data(data: &[u8], header: &RawfileHeader) -> Result<RawfileData> {
    // Find "Binary:" marker followed by newline
    let marker = b"Binary:\n";
    let marker_pos = find_bytes(data, marker).ok_or_else(|| {
        // Try without newline
        Error::RawfileParseError("Binary: marker not found".to_string())
    })?;

    let binary_start = marker_pos + marker.len();
    let binary_data = &data[binary_start..];

    // Each point contains num_variables values
    // For real data: num_variables * 8 bytes (f64) per point
    // For complex data: num_variables * 16 bytes (two f64s) per point
    let bytes_per_value = if header.is_complex { 16 } else { 8 };
    let bytes_per_point = header.num_variables * bytes_per_value;

    let mut real_data: Vec<Vec<f64>> = Vec::with_capacity(header.num_points);
    let mut imag_data: Option<Vec<Vec<f64>>> = if header.is_complex {
        Some(Vec::with_capacity(header.num_points))
    } else {
        None
    };

    for point_idx in 0..header.num_points {
        let point_start = point_idx * bytes_per_point;
        if point_start + bytes_per_point > binary_data.len() {
            break; // Not enough data
        }

        let mut point_real = Vec::with_capacity(header.num_variables);
        let mut point_imag = Vec::with_capacity(header.num_variables);

        for var_idx in 0..header.num_variables {
            let val_start = point_start + var_idx * bytes_per_value;

            if header.is_complex {
                // Read two f64s: real, then imag
                let re = read_f64_le(&binary_data[val_start..val_start + 8]);
                let im = read_f64_le(&binary_data[val_start + 8..val_start + 16]);
                point_real.push(re);
                point_imag.push(im);
            } else {
                let v = read_f64_le(&binary_data[val_start..val_start + 8]);
                point_real.push(v);
            }
        }

        real_data.push(point_real);
        if header.is_complex {
            imag_data.as_mut().unwrap().push(point_imag);
        }
    }

    Ok(RawfileData {
        header: header.clone(),
        real_data,
        imag_data,
    })
}

/// Read a little-endian f64 from bytes.
fn read_f64_le(data: &[u8]) -> f64 {
    let bytes: [u8; 8] = data[..8].try_into().unwrap_or([0; 8]);
    f64::from_le_bytes(bytes)
}

/// Find a byte sequence in a slice.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header() {
        let data = r#"Title: Test Circuit
Plotname: DC transfer characteristic
Flags: real
No. Variables: 3
No. Points: 11
Variables:
	0	v-sweep	voltage
	1	V(1)	voltage
	2	I(V1)	current
Values:
"#;
        let header = parse_header(data).unwrap();
        assert_eq!(header.title, "Test Circuit");
        assert_eq!(header.plotname, "DC transfer characteristic");
        assert_eq!(header.num_variables, 3);
        assert_eq!(header.num_points, 11);
        assert!(!header.is_complex);
        assert!(!header.is_binary);
        assert_eq!(header.variables.len(), 3);
        assert_eq!(header.variables[0].name, "v-sweep");
        assert_eq!(header.variables[1].name, "V(1)");
        assert_eq!(header.variables[2].name, "I(V1)");
    }

    #[test]
    fn test_parse_complex_value() {
        assert_eq!(parse_complex_value("1.0,2.0"), Some((1.0, 2.0)));
        assert_eq!(parse_complex_value("1.0, 2.0"), Some((1.0, 2.0)));
        assert_eq!(parse_complex_value("3.14"), Some((3.14, 0.0)));
    }
}
