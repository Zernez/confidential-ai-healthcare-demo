//! Python bindings for TEE Attestation
//! 
//! Provides the same attestation capabilities used by WASM modules,
//! exposed to Python via PyO3.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use serde::{Deserialize, Serialize};
use std::process::Command;

// ═══════════════════════════════════════════════════════════════════════════
// Data Structures (matching WASM host)
// ═══════════════════════════════════════════════════════════════════════════

/// TEE type detection result
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeeInfo {
    #[pyo3(get)]
    pub tee_type: String,
    #[pyo3(get)]
    pub supports_attestation: bool,
}

#[pymethods]
impl TeeInfo {
    fn __repr__(&self) -> String {
        format!("TeeInfo(tee_type='{}', supports_attestation={})", 
                self.tee_type, self.supports_attestation)
    }
}

/// Attestation result
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttestationResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub token: Option<String>,
    #[pyo3(get)]
    pub evidence: Option<String>,
    #[pyo3(get)]
    pub error: Option<String>,
    #[pyo3(get)]
    pub tee_type: Option<String>,
}

#[pymethods]
impl AttestationResult {
    fn __repr__(&self) -> String {
        if self.success {
            format!("AttestationResult(success=True, token_len={})", 
                    self.token.as_ref().map(|t| t.len()).unwrap_or(0))
        } else {
            format!("AttestationResult(success=False, error='{}')", 
                    self.error.as_ref().unwrap_or(&"Unknown".to_string()))
        }
    }
    
    #[getter]
    fn token_length(&self) -> usize {
        self.token.as_ref().map(|t| t.len()).unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TEE Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Detect TEE type (AMD SEV-SNP, Intel TDX, or None)
#[pyfunction]
fn detect_tee() -> PyResult<TeeInfo> {
    // Try AMD SEV-SNP via vTPM
    #[cfg(feature = "attestation-amd")]
    {
        use vtpm_attestation::{vtpm, hcl};
        
        if let Ok(report_bytes) = vtpm::get_report() {
            if let Ok(hcl_report) = hcl::HclReport::new(report_bytes) {
                let report_type = hcl_report.report_type();
                match report_type {
                    hcl::ReportType::Snp => {
                        return Ok(TeeInfo {
                            tee_type: "AMD SEV-SNP".to_string(),
                            supports_attestation: true,
                        });
                    }
                    hcl::ReportType::Tdx => {
                        return Ok(TeeInfo {
                            tee_type: "Intel TDX".to_string(),
                            supports_attestation: true,
                        });
                    }
                }
            }
        }
    }
    
    // Check for AMD SEV without SNP
    if std::path::Path::new("/sys/firmware/sev").exists() {
        return Ok(TeeInfo {
            tee_type: "AMD SEV".to_string(),
            supports_attestation: false,
        });
    }
    
    // Check for TDX device
    if std::path::Path::new("/dev/tdx_guest").exists() ||
       std::path::Path::new("/dev/tdx-guest").exists() {
        return Ok(TeeInfo {
            tee_type: "Intel TDX".to_string(),
            supports_attestation: true,
        });
    }
    
    Ok(TeeInfo {
        tee_type: "None".to_string(),
        supports_attestation: false,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// VM Attestation (AMD SEV-SNP)
// ═══════════════════════════════════════════════════════════════════════════

/// Attest the VM (AMD SEV-SNP or Intel TDX)
#[pyfunction]
fn attest_vm() -> PyResult<AttestationResult> {
    #[cfg(feature = "attestation-amd")]
    {
        return attest_amd_sev_snp();
    }
    
    #[cfg(not(feature = "attestation-amd"))]
    {
        Ok(AttestationResult {
            success: false,
            token: None,
            evidence: None,
            error: Some("AMD attestation not compiled in".to_string()),
            tee_type: None,
        })
    }
}

#[cfg(feature = "attestation-amd")]
fn attest_amd_sev_snp() -> PyResult<AttestationResult> {
    use vtpm_attestation::{vtpm, hcl};
    use sev::firmware::guest::AttestationReport as SnpReport;
    use std::convert::TryFrom;
    
    // Step 1: Get HCL report from vTPM
    let report_bytes = vtpm::get_report()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get vTPM report: {}", e)))?;
    
    let hcl_report = hcl::HclReport::new(report_bytes.clone())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse HCL report: {}", e)))?;
    
    // Verify it's SNP
    if hcl_report.report_type() != hcl::ReportType::Snp {
        return Ok(AttestationResult {
            success: false,
            token: None,
            evidence: None,
            error: Some(format!("Expected SNP report, got {:?}", hcl_report.report_type())),
            tee_type: None,
        });
    }
    
    // Step 2: Extract SNP report
    let snp_report = SnpReport::try_from(&hcl_report)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract SNP report: {}", e)))?;
    
    // Step 3: Get vTPM quote
    let nonce = b"python-baseline-attestation";
    let quote = vtpm::get_quote(nonce)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get vTPM quote: {}", e)))?;
    
    // Step 4: Get AK public key
    let ak_pub = vtpm::get_ak_pub()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get AK public key: {}", e)))?;
    
    // Step 5: Get VCEK certificate chain from Azure IMDS (blocking)
    let certs = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?
        .block_on(async {
            amd_vtpm::imds::get_certs().await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get VCEK certs: {}", e)))?;
    
    // Build evidence JSON
    let evidence = serde_json::json!({
        "tee_type": "AMD_SEV_SNP",
        "platform": "Azure",
        "snp_report": {
            "version": snp_report.version,
            "guest_svn": snp_report.guest_svn,
            "policy": format!("0x{:016x}", snp_report.policy.0),
            "measurement": hex::encode(&snp_report.measurement),
            "host_data": hex::encode(&snp_report.host_data),
            "report_data": hex::encode(&snp_report.report_data),
        },
        "hcl_report": {
            "raw_size": report_bytes.len(),
            "var_data_hash": hex::encode(hcl_report.var_data_sha256()),
        },
        "vtpm_quote": {
            "signature_size": quote.signature.len(),
            "message_size": quote.message.len(),
            "pcr_count": quote.pcrs.len(),
        },
        "ak_pub": {
            "modulus_size": ak_pub.modulus().len(),
            "exponent_size": ak_pub.exponent().len(),
        },
        "vcek_certs": {
            "vcek_cert_size": certs.vcek.len(),
            "chain_size": certs.amd_chain.len(),
        },
    });
    
    let evidence_str = serde_json::to_string_pretty(&evidence)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize evidence: {}", e)))?;
    
    // Create attestation bundle as token
    let attestation_bundle = serde_json::json!({
        "hcl_report": base64_encode(&report_bytes),
        "vtpm_quote": {
            "signature": base64_encode(&quote.signature),
            "message": base64_encode(&quote.message),
        },
        "vcek_cert": base64_encode(certs.vcek.as_bytes()),
        "cert_chain": base64_encode(certs.amd_chain.as_bytes()),
    });
    let token = base64_encode(&serde_json::to_vec(&attestation_bundle).unwrap_or_default());
    
    Ok(AttestationResult {
        success: true,
        token: Some(token),
        evidence: Some(evidence_str),
        error: None,
        tee_type: Some("AMD SEV-SNP".to_string()),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Attestation (NVIDIA)
// ═══════════════════════════════════════════════════════════════════════════

/// Attest GPU using nvattest CLI (LOCAL verification)
#[pyfunction]
#[pyo3(signature = (gpu_index=0))]
fn attest_gpu(gpu_index: u32) -> PyResult<AttestationResult> {
    // Use nvattest CLI for local attestation (same as WASM host)
    let output = Command::new("nvattest")
        .args(["attest", "--device", "gpu", "--verifier", "local"])
        .output();
    
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            return Ok(AttestationResult {
                success: false,
                token: None,
                evidence: None,
                error: Some(format!("Failed to run nvattest: {}. Is it installed?", e)),
                tee_type: None,
            });
        }
    };
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Ok(AttestationResult {
            success: false,
            token: None,
            evidence: None,
            error: Some(format!("nvattest failed: {}", stderr)),
            tee_type: None,
        });
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Parse JSON output
    let json: serde_json::Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AttestationResult {
                success: false,
                token: None,
                evidence: None,
                error: Some(format!("Failed to parse nvattest output: {}", e)),
                tee_type: None,
            });
        }
    };
    
    // Check result code
    let result_code = json.get("result_code")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);
    
    if result_code != 0 {
        let msg = json.get("result_message")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        return Ok(AttestationResult {
            success: false,
            token: None,
            evidence: None,
            error: Some(format!("Attestation failed: {}", msg)),
            tee_type: None,
        });
    }
    
    // Extract JWT token
    let token = json.get("detached_eat")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.get(0))
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.get(1))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    
    // Extract claims as evidence
    let evidence = json.get("claims")
        .map(|v| serde_json::to_string_pretty(v).unwrap_or_default());
    
    // Check measurement result
    let meas_result = json.get("claims")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.get(0))
        .and_then(|v| v.get("measres"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    
    if meas_result != "success" {
        return Ok(AttestationResult {
            success: false,
            token: None,
            evidence: evidence,
            error: Some(format!("GPU measurement verification failed: {}", meas_result)),
            tee_type: None,
        });
    }
    
    Ok(AttestationResult {
        success: true,
        token: token,
        evidence: evidence,
        error: None,
        tee_type: Some("NVIDIA H100 CC".to_string()),
    })
}

/// Get GPU information from nvidia-smi
#[pyfunction]
fn get_gpu_info() -> PyResult<(String, String, u64)> {
    // Query GPU name
    let name_output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .map_err(|e| PyRuntimeError::new_err(format!("nvidia-smi failed: {}", e)))?;
    
    let name = String::from_utf8_lossy(&name_output.stdout).trim().to_string();
    
    // Query GPU memory
    let mem_output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| PyRuntimeError::new_err(format!("nvidia-smi failed: {}", e)))?;
    
    let mem_str = String::from_utf8_lossy(&mem_output.stdout).trim().to_string();
    let memory_mb: u64 = mem_str.parse().unwrap_or(0);
    
    Ok((name, "cuda".to_string(), memory_mb))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(data)
}

// ═══════════════════════════════════════════════════════════════════════════
// Python Module
// ═══════════════════════════════════════════════════════════════════════════

/// TEE Attestation module for Python
/// 
/// Provides the same attestation capabilities used by WASM modules,
/// allowing fair performance comparisons.
#[pymodule]
fn tee_attestation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TeeInfo>()?;
    m.add_class::<AttestationResult>()?;
    m.add_function(wrap_pyfunction!(detect_tee, m)?)?;
    m.add_function(wrap_pyfunction!(attest_vm, m)?)?;
    m.add_function(wrap_pyfunction!(attest_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_info, m)?)?;
    Ok(())
}
