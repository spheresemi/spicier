//! ngspice process runner.
//!
//! This module handles invoking ngspice as a subprocess to run simulations.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use tempfile::NamedTempFile;

use crate::error::{Error, Result};
use crate::ngspice::rawfile::parse_rawfile;
use crate::ngspice::types::RawfileData;

/// Configuration for the ngspice runner.
#[derive(Debug, Clone)]
pub struct NgspiceConfig {
    /// Path to ngspice executable (default: "ngspice" in PATH).
    pub executable: String,
    /// Timeout for ngspice execution in seconds.
    pub timeout_secs: u64,
}

impl Default for NgspiceConfig {
    fn default() -> Self {
        Self {
            executable: "ngspice".to_string(),
            timeout_secs: 60,
        }
    }
}

/// Check if ngspice is available.
pub fn is_ngspice_available(config: &NgspiceConfig) -> bool {
    Command::new(&config.executable)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Get ngspice version string.
pub fn ngspice_version(config: &NgspiceConfig) -> Result<String> {
    let output = Command::new(&config.executable)
        .arg("--version")
        .output()
        .map_err(|e| Error::NgspiceNotFound(e.to_string()))?;

    if !output.status.success() {
        return Err(Error::NgspiceNotFound("--version failed".to_string()));
    }

    // ngspice prints version to stdout
    let version = String::from_utf8_lossy(&output.stdout);
    // Extract first line
    Ok(version.lines().next().unwrap_or("unknown").to_string())
}

/// Run a netlist through ngspice and return the raw results.
pub fn run_ngspice(netlist: &str, config: &NgspiceConfig) -> Result<RawfileData> {
    // Create temp files for netlist and output
    let mut netlist_file =
        NamedTempFile::new().map_err(|e| Error::TempFile(e.to_string()))?;

    // Ensure netlist ends with .end if not present
    let netlist = if !netlist.to_lowercase().contains(".end") {
        format!("{}\n.end\n", netlist.trim())
    } else {
        netlist.to_string()
    };

    netlist_file
        .write_all(netlist.as_bytes())
        .map_err(|e| Error::TempFile(e.to_string()))?;

    let raw_file = NamedTempFile::new().map_err(|e| Error::TempFile(e.to_string()))?;

    // Run ngspice in batch mode
    // -b: batch mode
    // -r output.raw: write raw output
    // -o /dev/null: suppress log output
    let mut cmd = Command::new(&config.executable);
    cmd.arg("-b")
        .arg("-r")
        .arg(raw_file.path())
        .arg(netlist_file.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = cmd.spawn().map_err(|e| Error::NgspiceNotFound(e.to_string()))?;

    // Wait with timeout
    let output = wait_with_timeout(child, Duration::from_secs(config.timeout_secs))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(Error::NgspiceExecutionFailed(format!(
            "ngspice exited with {}\nstderr: {}\nstdout: {}",
            output.status, stderr, stdout
        )));
    }

    // Read and parse the raw file
    let raw_data = std::fs::read(raw_file.path()).map_err(|e| {
        Error::RawfileParseError(format!("failed to read rawfile: {}", e))
    })?;

    if raw_data.is_empty() {
        return Err(Error::RawfileParseError(
            "ngspice produced empty rawfile".to_string(),
        ));
    }

    parse_rawfile(&raw_data)
}

/// Wait for a child process with timeout.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output> {
    use std::thread;

    let start = std::time::Instant::now();
    let poll_interval = Duration::from_millis(100);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited, collect output
                let stdout = child.stdout.take().map(|mut s| {
                    let mut buf = Vec::new();
                    std::io::Read::read_to_end(&mut s, &mut buf).ok();
                    buf
                }).unwrap_or_default();

                let stderr = child.stderr.take().map(|mut s| {
                    let mut buf = Vec::new();
                    std::io::Read::read_to_end(&mut s, &mut buf).ok();
                    buf
                }).unwrap_or_default();

                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                // Still running
                if start.elapsed() > timeout {
                    // Kill the process
                    let _ = child.kill();
                    return Err(Error::NgspiceTimeout(timeout.as_secs()));
                }
                thread::sleep(poll_interval);
            }
            Err(e) => {
                return Err(Error::NgspiceExecutionFailed(e.to_string()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NgspiceConfig::default();
        assert_eq!(config.executable, "ngspice");
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    #[ignore] // Requires ngspice to be installed
    fn test_ngspice_available() {
        let config = NgspiceConfig::default();
        if is_ngspice_available(&config) {
            let version = ngspice_version(&config).unwrap();
            assert!(!version.is_empty());
        }
    }

    #[test]
    #[ignore] // Requires ngspice to be installed
    fn test_run_simple_circuit() {
        let config = NgspiceConfig::default();
        if !is_ngspice_available(&config) {
            return;
        }

        let netlist = r#"Voltage Divider
V1 1 0 DC 10
R1 1 2 1k
R2 2 0 1k
.op
.end
"#;

        let result = run_ngspice(netlist, &config).unwrap();
        assert!(!result.real_data.is_empty());
    }
}
