//! Types for ngspice simulation results.

use num_complex::Complex;
use std::collections::HashMap;

/// Analysis type parsed from rawfile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisType {
    /// DC operating point.
    DcOp,
    /// DC sweep.
    DcSweep,
    /// AC analysis.
    Ac,
    /// Transient analysis.
    Transient,
}

/// A variable in the rawfile (column in the data).
#[derive(Debug, Clone)]
pub struct RawVariable {
    /// Variable index (0-based).
    pub index: usize,
    /// Variable name (e.g., "V(1)", "I(V1)").
    pub name: String,
    /// Variable type (e.g., "voltage", "current", "time", "frequency").
    pub var_type: String,
}

/// Parsed rawfile header information.
#[derive(Debug, Clone)]
pub struct RawfileHeader {
    /// Title of the simulation.
    pub title: String,
    /// Plot name (e.g., "DC transfer characteristic", "AC Analysis").
    pub plotname: String,
    /// Flags (e.g., "real", "complex").
    pub flags: String,
    /// Number of variables.
    pub num_variables: usize,
    /// Number of data points.
    pub num_points: usize,
    /// Variable definitions.
    pub variables: Vec<RawVariable>,
    /// Whether data is complex (AC analysis).
    pub is_complex: bool,
    /// Whether data is binary format.
    pub is_binary: bool,
}

/// Result of parsing a rawfile.
#[derive(Debug, Clone)]
pub struct RawfileData {
    /// Header information.
    pub header: RawfileHeader,
    /// Real data values (num_points x num_variables).
    /// For complex data, this contains only the real parts.
    pub real_data: Vec<Vec<f64>>,
    /// Imaginary data values (only for complex data).
    pub imag_data: Option<Vec<Vec<f64>>>,
}

impl RawfileData {
    /// Get the analysis type from the plotname.
    pub fn analysis_type(&self) -> AnalysisType {
        let plotname = self.header.plotname.to_lowercase();
        if plotname.contains("operating point") {
            AnalysisType::DcOp
        } else if plotname.contains("dc transfer") || plotname.contains("dc analysis") {
            AnalysisType::DcSweep
        } else if plotname.contains("ac analysis") {
            AnalysisType::Ac
        } else if plotname.contains("transient") {
            AnalysisType::Transient
        } else {
            // Default to DC op for unknown types
            AnalysisType::DcOp
        }
    }

    /// Find a variable by name (case-insensitive).
    pub fn find_variable(&self, name: &str) -> Option<&RawVariable> {
        let name_lower = name.to_lowercase();
        self.header
            .variables
            .iter()
            .find(|v| v.name.to_lowercase() == name_lower)
    }

    /// Get real values for a variable across all points.
    pub fn get_real_values(&self, var_index: usize) -> Option<Vec<f64>> {
        if var_index >= self.header.num_variables {
            return None;
        }
        Some(self.real_data.iter().map(|row| row[var_index]).collect())
    }

    /// Get complex values for a variable across all points.
    pub fn get_complex_values(&self, var_index: usize) -> Option<Vec<Complex<f64>>> {
        if var_index >= self.header.num_variables {
            return None;
        }
        let imag = self.imag_data.as_ref()?;
        Some(
            self.real_data
                .iter()
                .zip(imag.iter())
                .map(|(re_row, im_row)| Complex::new(re_row[var_index], im_row[var_index]))
                .collect(),
        )
    }
}

/// DC operating point result from ngspice.
#[derive(Debug, Clone)]
pub struct NgspiceDcOp {
    /// Node voltages and branch currents.
    pub values: HashMap<String, f64>,
}

impl NgspiceDcOp {
    /// Create from rawfile data.
    pub fn from_rawfile(data: &RawfileData) -> Self {
        let mut values = HashMap::new();

        // DC op should have exactly one point
        if !data.real_data.is_empty() {
            let point = &data.real_data[0];
            for var in &data.header.variables {
                values.insert(var.name.clone(), point[var.index]);
            }
        }

        NgspiceDcOp { values }
    }

    /// Get a voltage value.
    pub fn voltage(&self, name: &str) -> Option<f64> {
        // Try exact match first
        if let Some(&v) = self.values.get(name) {
            return Some(v);
        }
        // Try case-insensitive
        let name_lower = name.to_lowercase();
        self.values
            .iter()
            .find(|(k, _)| k.to_lowercase() == name_lower)
            .map(|(_, &v)| v)
    }

    /// Get a current value.
    pub fn current(&self, name: &str) -> Option<f64> {
        self.voltage(name) // Same lookup mechanism
    }
}

/// AC analysis result from ngspice.
#[derive(Debug, Clone)]
pub struct NgspiceAc {
    /// Frequencies.
    pub frequencies: Vec<f64>,
    /// Complex values for each variable (variable name -> values at each frequency).
    pub values: HashMap<String, Vec<Complex<f64>>>,
}

impl NgspiceAc {
    /// Create from rawfile data.
    pub fn from_rawfile(data: &RawfileData) -> Self {
        let mut frequencies = Vec::new();
        let mut values: HashMap<String, Vec<Complex<f64>>> = HashMap::new();

        // Find frequency variable (usually index 0)
        let freq_idx = data
            .header
            .variables
            .iter()
            .find(|v| v.var_type.to_lowercase() == "frequency")
            .map(|v| v.index)
            .unwrap_or(0);

        for var in &data.header.variables {
            values.insert(var.name.clone(), Vec::new());
        }

        for (i, re_row) in data.real_data.iter().enumerate() {
            frequencies.push(re_row[freq_idx]);

            for var in &data.header.variables {
                let re = re_row[var.index];
                let im = data
                    .imag_data
                    .as_ref()
                    .map(|im| im[i][var.index])
                    .unwrap_or(0.0);
                values
                    .get_mut(&var.name)
                    .unwrap()
                    .push(Complex::new(re, im));
            }
        }

        NgspiceAc {
            frequencies,
            values,
        }
    }

    /// Get magnitude in dB for a variable.
    pub fn magnitude_db(&self, name: &str) -> Option<Vec<f64>> {
        self.values.get(name).map(|vals| {
            vals.iter()
                .map(|c| 20.0 * c.norm().log10())
                .collect()
        })
    }

    /// Get phase in degrees for a variable.
    pub fn phase_deg(&self, name: &str) -> Option<Vec<f64>> {
        self.values.get(name).map(|vals| {
            vals.iter()
                .map(|c| c.arg().to_degrees())
                .collect()
        })
    }
}

/// Transient analysis result from ngspice.
#[derive(Debug, Clone)]
pub struct NgspiceTransient {
    /// Time values.
    pub times: Vec<f64>,
    /// Values for each variable (variable name -> values at each time).
    pub values: HashMap<String, Vec<f64>>,
}

impl NgspiceTransient {
    /// Create from rawfile data.
    pub fn from_rawfile(data: &RawfileData) -> Self {
        let mut times = Vec::new();
        let mut values: HashMap<String, Vec<f64>> = HashMap::new();

        // Find time variable (usually index 0)
        let time_idx = data
            .header
            .variables
            .iter()
            .find(|v| v.var_type.to_lowercase() == "time")
            .map(|v| v.index)
            .unwrap_or(0);

        for var in &data.header.variables {
            values.insert(var.name.clone(), Vec::new());
        }

        for row in &data.real_data {
            times.push(row[time_idx]);

            for var in &data.header.variables {
                values.get_mut(&var.name).unwrap().push(row[var.index]);
            }
        }

        NgspiceTransient { times, values }
    }

    /// Get values at a specific time using linear interpolation.
    pub fn interpolate_at(&self, time: f64, name: &str) -> Option<f64> {
        let vals = self.values.get(name)?;
        if self.times.is_empty() || vals.is_empty() {
            return None;
        }

        // Handle boundary cases
        if time <= self.times[0] {
            return Some(vals[0]);
        }
        if time >= *self.times.last().unwrap() {
            return Some(*vals.last().unwrap());
        }

        // Find interval and interpolate
        for i in 0..self.times.len() - 1 {
            if time >= self.times[i] && time <= self.times[i + 1] {
                let t0 = self.times[i];
                let t1 = self.times[i + 1];
                let alpha = (time - t0) / (t1 - t0);
                return Some(vals[i] * (1.0 - alpha) + vals[i + 1] * alpha);
            }
        }

        None
    }
}

/// Unified ngspice result type.
#[derive(Debug, Clone)]
pub enum NgspiceResult {
    /// DC operating point.
    DcOp(NgspiceDcOp),
    /// AC analysis.
    Ac(NgspiceAc),
    /// Transient analysis.
    Transient(NgspiceTransient),
}

impl NgspiceResult {
    /// Create from rawfile data.
    pub fn from_rawfile(data: &RawfileData) -> Self {
        match data.analysis_type() {
            AnalysisType::DcOp | AnalysisType::DcSweep => {
                NgspiceResult::DcOp(NgspiceDcOp::from_rawfile(data))
            }
            AnalysisType::Ac => NgspiceResult::Ac(NgspiceAc::from_rawfile(data)),
            AnalysisType::Transient => {
                NgspiceResult::Transient(NgspiceTransient::from_rawfile(data))
            }
        }
    }
}
