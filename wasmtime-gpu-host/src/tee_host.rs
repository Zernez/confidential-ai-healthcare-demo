/**
 * TEE Host - wasmtime:attestation Implementation
 * 
 * Provides attestation capabilities for WASM guests running in Confidential VMs
 * with GPU support. This module exposes host functions for:
 * - VM attestation (TDX or AMD SEV-SNP via vTPM)
 * - GPU attestation (NVIDIA LOCAL via nvattest, with NRAS fallback)
 * - Token verification
 * - Evidence collection
 */

use anyhow::{Context, Result};
use tracing::{debug, info, warn, error};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use wasmtime::{Caller, Linker};
use tokio::runtime::{Runtime, Handle};

// Conditional imports based on features
#[cfg(feature = "attestation-tdx")]
use lunal_attestation;

#[cfg(feature = "attestation-amd")]
use vtpm_attestation::{vtpm, hcl};

#[cfg(feature = "attestation-amd")]
use amd_vtpm::imds;

/// Detected TEE type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeeType {
    None,
    IntelTdx,
    AmdSevSnp,
    AmdSev,  // SEV without SNP (no attestation, just encryption)
}

impl std::fmt::Display for TeeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeeType::None => write!(f, "None"),
            TeeType::IntelTdx => write!(f, "Intel TDX"),
            TeeType::AmdSevSnp => write!(f, "AMD SEV-SNP"),
            TeeType::AmdSev => write!(f, "AMD SEV"),
        }
    }
}

/// Attestation result returned to WASM guest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Success flag
    pub success: bool,
    /// JWT token (if successful) - for GPU attestation
    pub token: Option<String>,
    /// Evidence JSON (if requested) - for VM attestation
    pub evidence: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Timestamp
    pub timestamp: i64,
    /// TEE type detected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tee_type: Option<String>,
}

impl AttestationResult {
    pub fn success(token: String) -> Self {
        Self {
            success: true,
            token: Some(token),
            evidence: None,
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            tee_type: None,
        }
    }

    pub fn success_with_tee(token: String, tee_type: TeeType) -> Self {
        Self {
            success: true,
            token: Some(token),
            evidence: None,
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            tee_type: Some(tee_type.to_string()),
        }
    }

    pub fn with_evidence(token: String, evidence: String) -> Self {
        Self {
            success: true,
            token: Some(token),
            evidence: Some(evidence),
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            tee_type: None,
        }
    }

    pub fn with_evidence_and_tee(token: String, evidence: String, tee_type: TeeType) -> Self {
        Self {
            success: true,
            token: Some(token),
            evidence: Some(evidence),
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            tee_type: Some(tee_type.to_string()),
        }
    }

    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            token: None,
            evidence: None,
            error: Some(error),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            tee_type: None,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(r#"{{"success":false,"error":"JSON serialization failed"}}"#)
        })
    }
}

/// TEE Host state (wrapped in Arc for sharing with closures)
#[derive(Clone)]
pub struct TeeHost {
    inner: Arc<TeeHostInner>,
}

struct TeeHostInner {
    /// Cached VM attestation token/evidence
    vm_token_cache: Mutex<Option<String>>,
    /// Cached GPU attestation token
    gpu_token_cache: Mutex<Option<String>>,
    /// NRAS endpoint (optional override)
    nras_endpoint: Option<String>,
    /// Detected TEE type (cached)
    detected_tee: Mutex<Option<TeeType>>,
}

impl TeeHost {
    pub fn new() -> Self {
        info!("Initializing TEE Host (wasmtime:attestation)");
        Self {
            inner: Arc::new(TeeHostInner {
                vm_token_cache: Mutex::new(None),
                gpu_token_cache: Mutex::new(None),
                nras_endpoint: None,
                detected_tee: Mutex::new(None),
            }),
        }
    }

    pub fn with_nras_endpoint(mut self, endpoint: String) -> Self {
        self.inner = Arc::new(TeeHostInner {
            vm_token_cache: Mutex::new(None),
            gpu_token_cache: Mutex::new(None),
            nras_endpoint: Some(endpoint),
            detected_tee: Mutex::new(None),
        });
        self
    }

    /// Detect the TEE type of the current environment
    pub fn detect_tee_type(&self) -> TeeType {
        // Check cache first
        {
            let cache = self.inner.detected_tee.lock().unwrap();
            if let Some(tee_type) = *cache {
                return tee_type;
            }
        }

        let tee_type = self.detect_tee_type_internal();
        
        // Cache the result
        {
            let mut cache = self.inner.detected_tee.lock().unwrap();
            *cache = Some(tee_type);
        }
        
        tee_type
    }

    fn detect_tee_type_internal(&self) -> TeeType {
        info!("Detecting TEE type...");

        // Try AMD SEV-SNP first (via vTPM HCL report)
        #[cfg(feature = "attestation-amd")]
        {
            debug!("Checking for AMD SEV-SNP via vTPM...");
            match vtpm::get_report() {
                Ok(report_bytes) => {
                    match hcl::HclReport::new(report_bytes) {
                        Ok(hcl_report) => {
                            let report_type = hcl_report.report_type();
                            match report_type {
                                hcl::ReportType::Snp => {
                                    info!("✓ Detected AMD SEV-SNP (via vTPM HCL report)");
                                    return TeeType::AmdSevSnp;
                                }
                                hcl::ReportType::Tdx => {
                                    info!("✓ Detected Intel TDX (via vTPM HCL report)");
                                    return TeeType::IntelTdx;
                                }
                            }
                        }
                        Err(e) => {
                            debug!("Failed to parse HCL report: {}", e);
                        }
                    }
                }
                Err(e) => {
                    debug!("Failed to get vTPM report: {}", e);
                }
            }
        }

        // Check for AMD SEV (without SNP) via /sys
        if std::path::Path::new("/sys/firmware/sev").exists() {
            info!("✓ Detected AMD SEV (memory encryption only, no attestation)");
            return TeeType::AmdSev;
        }

        // Try TDX directly
        #[cfg(feature = "attestation-tdx")]
        {
            debug!("Checking for Intel TDX...");
            if std::path::Path::new("/dev/tdx_guest").exists() ||
               std::path::Path::new("/dev/tdx-guest").exists() {
                info!("✓ Detected Intel TDX");
                return TeeType::IntelTdx;
            }
        }

        warn!("No supported TEE detected");
        TeeType::None
    }

    /// Attest VM - automatically detects TEE type (TDX or AMD SEV-SNP)
    pub fn attest_vm_sync(&self) -> AttestationResult {
        info!("Starting VM attestation...");

        // Check cache first
        {
            let cache = self.inner.vm_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("Using cached VM attestation");
                return AttestationResult::success(cached_token.clone());
            }
        }

        // Detect TEE type
        let tee_type = self.detect_tee_type();
        info!("TEE type: {}", tee_type);

        match tee_type {
            TeeType::AmdSevSnp => {
                #[cfg(feature = "attestation-amd")]
                {
                    match self.attest_amd_sev_snp_sync() {
                        Ok(result) => {
                            info!("AMD SEV-SNP attestation successful");
                            if let Some(ref evidence) = result.evidence {
                                let mut cache = self.inner.vm_token_cache.lock().unwrap();
                                *cache = Some(evidence.clone());
                            }
                            return result;
                        }
                        Err(e) => {
                            error!("AMD SEV-SNP attestation failed: {}", e);
                            return AttestationResult::failure(format!("AMD SEV-SNP attestation failed: {}", e));
                        }
                    }
                }
                #[cfg(not(feature = "attestation-amd"))]
                {
                    return AttestationResult::failure(
                        "AMD SEV-SNP attestation not compiled in. Rebuild with --features attestation-amd".to_string()
                    );
                }
            }
            TeeType::IntelTdx => {
                #[cfg(feature = "attestation-tdx")]
                {
                    match self.attest_tdx_sync() {
                        Ok(result) => {
                            info!("TDX attestation successful");
                            if let Some(ref token) = result.token {
                                let mut cache = self.inner.vm_token_cache.lock().unwrap();
                                *cache = Some(token.clone());
                            }
                            return result;
                        }
                        Err(e) => {
                            error!("TDX attestation failed: {}", e);
                            return AttestationResult::failure(format!("TDX attestation failed: {}", e));
                        }
                    }
                }
                #[cfg(not(feature = "attestation-tdx"))]
                {
                    return AttestationResult::failure(
                        "TDX attestation not compiled in. Rebuild with --features attestation-tdx".to_string()
                    );
                }
            }
            TeeType::AmdSev => {
                warn!("AMD SEV detected but SNP not available - cannot perform remote attestation");
                return AttestationResult::failure(
                    "AMD SEV detected (memory encrypted) but SEV-SNP not available for attestation. \
                     This VM may not be a full Confidential VM with attestation support.".to_string()
                );
            }
            TeeType::None => {
                warn!("No TEE attestation available");
                return AttestationResult::failure(
                    "No TEE attestation available. Not running in a Confidential VM (Intel TDX or AMD SEV-SNP)?".to_string()
                );
            }
        }
    }

    /// AMD SEV-SNP attestation via vTPM
    #[cfg(feature = "attestation-amd")]
    fn attest_amd_sev_snp_sync(&self) -> Result<AttestationResult> {
        info!("Performing AMD SEV-SNP attestation via vTPM...");

        // Step 1: Get HCL report from vTPM
        info!("  Step 1: Reading HCL report from vTPM...");
        let report_bytes = vtpm::get_report()
            .map_err(|e| anyhow::anyhow!("Failed to get vTPM report: {}", e))?;
        
        let hcl_report = hcl::HclReport::new(report_bytes.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse HCL report: {}", e))?;

        // Verify it's SNP
        if hcl_report.report_type() != hcl::ReportType::Snp {
            return Err(anyhow::anyhow!("Expected SNP report, got {:?}", hcl_report.report_type()));
        }
        info!("  ✓ HCL report type: SNP");

        // Step 2: Get SNP report from HCL
        info!("  Step 2: Extracting SNP attestation report...");
        use sev::firmware::guest::AttestationReport as SnpReport;
        use std::convert::TryFrom;
        
        let snp_report = SnpReport::try_from(&hcl_report)
            .map_err(|e| anyhow::anyhow!("Failed to extract SNP report: {}", e))?;
        
        info!("  ✓ SNP report extracted");
        info!("    Version: {}", snp_report.version);
        info!("    Guest SVN: {}", snp_report.guest_svn);
        info!("    Policy: 0x{:016x}", snp_report.policy.0);

        // Step 3: Get vTPM quote (optional, for additional binding)
        info!("  Step 3: Getting vTPM quote...");
        let nonce = b"wasmtime-attestation-nonce";
        let quote = vtpm::get_quote(nonce)
            .map_err(|e| anyhow::anyhow!("Failed to get vTPM quote: {}", e))?;
        info!("  ✓ vTPM quote obtained ({} bytes signature)", quote.signature.len());

        // Step 4: Get AK public key
        info!("  Step 4: Getting AK public key...");
        let ak_pub = vtpm::get_ak_pub()
            .map_err(|e| anyhow::anyhow!("Failed to get AK public key: {}", e))?;
        info!("  ✓ AK public key obtained");

        // Step 5: Get VCEK certificate chain from IMDS (async)
        info!("  Step 5: Fetching VCEK certificate chain from Azure IMDS...");
        let certs = {
            // Create a temporary tokio runtime for the async IMDS call
            let rt = Runtime::new()
                .map_err(|e| anyhow::anyhow!("Failed to create tokio runtime: {}", e))?;
            rt.block_on(async {
                imds::get_certs().await
            })
        }.map_err(|e| anyhow::anyhow!("Failed to get VCEK certs from IMDS: {}", e))?;
        info!("  ✓ VCEK certificate chain obtained");
        info!("    VCEK cert: {} bytes", certs.vcek.len());
        info!("    Cert chain: {} bytes", certs.amd_chain.len());

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
            .map_err(|e| anyhow::anyhow!("Failed to serialize evidence: {}", e))?;

        // Create a "token" that contains the base64-encoded full attestation bundle
        let attestation_bundle = serde_json::json!({
            "hcl_report": base64::encode(&report_bytes),
            "vtpm_quote": {
                "signature": base64::encode(&quote.signature),
                "message": base64::encode(&quote.message),
            },
            "vcek_cert": base64::encode(certs.vcek.as_bytes()),
            "cert_chain": base64::encode(certs.amd_chain.as_bytes()),
        });
        let token = base64::encode(&serde_json::to_vec(&attestation_bundle).unwrap_or_default());

        info!("✓ AMD SEV-SNP attestation completed successfully");
        info!("  Evidence size: {} bytes", evidence_str.len());
        info!("  Token size: {} bytes", token.len());

        Ok(AttestationResult::with_evidence_and_tee(token, evidence_str, TeeType::AmdSevSnp))
    }

    /// Attest GPU - tries LOCAL first (nvattest CLI), then NRAS as fallback
    pub fn attest_gpu_sync(&self, gpu_index: u32) -> AttestationResult {
        info!("Starting GPU attestation for device {}...", gpu_index);

        // Check cache first
        {
            let cache = self.inner.gpu_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("Using cached GPU attestation token");
                return AttestationResult::success(cached_token.clone());
            }
        }

        // Check if NRAS is preferred via environment variable
        let prefer_nras = std::env::var("ATTESTATION_PREFER_NRAS")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if prefer_nras {
            info!("ATTESTATION_PREFER_NRAS=1, trying NRAS first...");
            #[cfg(feature = "attestation-nvidia")]
            {
                match self.attest_gpu_nras(gpu_index) {
                    Ok(result) => return result,
                    Err(e) => {
                        warn!("NRAS attestation failed: {}, falling back to local", e);
                    }
                }
            }
        }

        // Strategy: Try LOCAL attestation first (works on Azure/cloud without NRAS subscription)
        info!("Attempting LOCAL GPU attestation via nvattest CLI...");
        match self.attest_gpu_local(gpu_index) {
            Ok(result) => {
                info!("Local GPU attestation successful!");
                if let Some(token) = &result.token {
                    let mut cache = self.inner.gpu_token_cache.lock().unwrap();
                    *cache = Some(token.clone());
                }
                return result;
            }
            Err(e) => {
                warn!("Local attestation failed: {}", e);
                info!("Falling back to NRAS remote attestation...");
            }
        }

        // Fallback: Try NRAS remote attestation
        #[cfg(feature = "attestation-nvidia")]
        {
            match self.attest_gpu_nras(gpu_index) {
                Ok(result) => return result,
                Err(e) => {
                    error!("NRAS attestation also failed: {}", e);
                    return AttestationResult::failure(format!("GPU attestation failed (local and NRAS): {}", e));
                }
            }
        }

        #[cfg(not(feature = "attestation-nvidia"))]
        {
            error!("GPU attestation not compiled in");
            AttestationResult::failure("GPU attestation not available (local failed, NRAS not compiled)".to_string())
        }
    }

    /// NRAS remote attestation
    #[cfg(feature = "attestation-nvidia")]
    fn attest_gpu_nras(&self, gpu_index: u32) -> Result<AttestationResult> {
        let api_key = std::env::var("NVIDIA_API_KEY").ok();
        
        if api_key.is_none() {
            warn!("NVIDIA_API_KEY not set - NRAS remote attestation may fail");
            warn!("Get your API key from: https://org.ngc.nvidia.com/setup/api-key");
        }
        
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(async {
                lunal_attestation::nvidia::attest::attest_remote_token(
                    gpu_index,
                    None,
                    api_key,
                ).await
            })
        });
        
        match result {
            Ok(token) => {
                info!("GPU attestation successful (NRAS token)");
                info!("  Token length: {} chars", token.len());
                
                let mut cache = self.inner.gpu_token_cache.lock().unwrap();
                *cache = Some(token.clone());
                
                Ok(AttestationResult::success(token))
            }
            Err(e) => {
                let error_str = format!("{}", e);
                if error_str.contains("403") || error_str.contains("Forbidden") {
                    Err(anyhow::anyhow!("NRAS returned 403 Forbidden - API key may not have attestation permissions"))
                } else if error_str.contains("NVIDIA_API_KEY") {
                    Err(anyhow::anyhow!("NVIDIA_API_KEY not set. Get your key from https://org.ngc.nvidia.com/setup/api-key"))
                } else {
                    Err(anyhow::anyhow!("NRAS attestation failed: {}", e))
                }
            }
        }
    }

    /// Perform LOCAL GPU attestation using nvattest CLI
    fn attest_gpu_local(&self, _gpu_index: u32) -> Result<AttestationResult> {
        use std::process::Command;
        
        info!("Running: nvattest attest --device gpu --verifier local");
        
        let output = Command::new("nvattest")
            .args(["attest", "--device", "gpu", "--verifier", "local"])
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to run nvattest: {}. Is it installed?", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("nvattest failed: {}", stderr));
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Parse JSON output
        let json: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| anyhow::anyhow!("Failed to parse nvattest output: {}", e))?;
        
        // Check result code
        let result_code = json.get("result_code")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1);
        
        if result_code != 0 {
            let msg = json.get("result_message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            return Err(anyhow::anyhow!("Attestation failed: {}", msg));
        }
        
        // Extract JWT token from detached_eat array
        let token = json.get("detached_eat")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(0))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(1))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Extract claims for evidence
        let claims = json.get("claims")
            .map(|v| serde_json::to_string_pretty(v).unwrap_or_default())
            .unwrap_or_else(|| "{}".to_string());
        
        // Check measurement result from claims
        let meas_result = json.get("claims")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(0))
            .and_then(|v| v.get("measres"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        
        if meas_result != "success" {
            return Err(anyhow::anyhow!("GPU measurement verification failed: {}", meas_result));
        }

        // Extract additional info for logging
        let hw_model = json.get("claims")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(0))
            .and_then(|v| v.get("hwmodel"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let driver_version = json.get("claims")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(0))
            .and_then(|v| v.get("x-nvidia-gpu-driver-version"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        
        info!("✓ Local attestation completed");
        info!("  Verifier: LOCAL (nvattest)");
        info!("  Hardware: {}", hw_model);
        info!("  Driver: {}", driver_version);
        info!("  Measurement: {}", meas_result);
        
        if let Some(ref t) = token {
            info!("  JWT Token: {} chars", t.len());
        }
        
        Ok(AttestationResult {
            success: true,
            token: token,
            evidence: Some(claims),
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            tee_type: None,
        })
    }

    /// Verify a JWT token (basic validation)
    pub fn verify_token(&self, token: &str) -> bool {
        debug!("Verifying token...");
        
        if token.is_empty() {
            return false;
        }

        // Basic JWT structure validation (header.payload.signature)
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            // Could be base64-encoded attestation bundle (for AMD SEV-SNP)
            if let Ok(decoded) = base64::decode(token) {
                if serde_json::from_slice::<serde_json::Value>(&decoded).is_ok() {
                    debug!("Token is valid base64-encoded JSON attestation bundle");
                    return true;
                }
            }
            warn!("Invalid token format (not JWT or attestation bundle)");
            return false;
        }

        debug!("Token structure valid (JWT format)");
        true
    }

    /// Clear all cached tokens
    pub fn clear_cache(&self) {
        info!("Clearing attestation token cache");
        let mut vm_cache = self.inner.vm_token_cache.lock().unwrap();
        *vm_cache = None;
        let mut gpu_cache = self.inner.gpu_token_cache.lock().unwrap();
        *gpu_cache = None;
        let mut tee_cache = self.inner.detected_tee.lock().unwrap();
        *tee_cache = None;
    }

    /// Register host functions with Wasmtime linker
    pub fn register_functions<T>(&self, linker: &mut Linker<T>) -> Result<()>
    where
        T: AsTeeHost + Send + 'static,
    {
        info!("Registering wasmtime:attestation functions...");

        // attest_vm() -> pointer to JSON string
        linker.func_wrap(
            "wasmtime_attestation",
            "attest_vm",
            |mut caller: Caller<'_, T>| -> i32 {
                info!("[WASM] attest_vm() called");
                
                let result = {
                    let tee_host = caller.data().as_tee_host();
                    tee_host.attest_vm_sync()
                };
                
                let json = result.to_json();
                write_string_to_wasm(&mut caller, &json)
            },
        )?;

        // attest_gpu(gpu_index: u32) -> pointer to JSON string
        linker.func_wrap(
            "wasmtime_attestation",
            "attest_gpu",
            |mut caller: Caller<'_, T>, gpu_index: u32| -> i32 {
                info!("[WASM] attest_gpu({}) called", gpu_index);
                
                let result = {
                    let tee_host = caller.data().as_tee_host();
                    tee_host.attest_gpu_sync(gpu_index)
                };
                
                let json = result.to_json();
                write_string_to_wasm(&mut caller, &json)
            },
        )?;

        // verify_token(token_ptr: i32, token_len: i32) -> bool
        linker.func_wrap(
            "wasmtime_attestation",
            "verify_token",
            |mut caller: Caller<'_, T>, token_ptr: i32, token_len: i32| -> i32 {
                debug!("[WASM] verify_token() called");
                
                match read_string_from_wasm(&mut caller, token_ptr, token_len) {
                    Ok(token) => {
                        let tee_host = caller.data().as_tee_host();
                        if tee_host.verify_token(&token) { 1 } else { 0 }
                    }
                    Err(_) => 0,
                }
            },
        )?;

        // clear_cache()
        linker.func_wrap(
            "wasmtime_attestation",
            "clear_cache",
            |caller: Caller<'_, T>| {
                info!("[WASM] clear_cache() called");
                let tee_host = caller.data().as_tee_host();
                tee_host.clear_cache();
            },
        )?;

        // detect_tee() -> pointer to JSON string with TEE type
        linker.func_wrap(
            "wasmtime_attestation",
            "detect_tee",
            |mut caller: Caller<'_, T>| -> i32 {
                info!("[WASM] detect_tee() called");
                
                let tee_type = {
                    let tee_host = caller.data().as_tee_host();
                    tee_host.detect_tee_type()
                };
                
                let result = serde_json::json!({
                    "tee_type": tee_type.to_string(),
                    "supports_attestation": matches!(tee_type, TeeType::AmdSevSnp | TeeType::IntelTdx),
                });
                
                let json = serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());
                write_string_to_wasm(&mut caller, &json)
            },
        )?;

        info!("wasmtime:attestation functions registered");
        Ok(())
    }

    #[cfg(feature = "attestation-tdx")]
    fn attest_tdx_sync(&self) -> Result<AttestationResult> {
        let quote = lunal_attestation::attestation::get_raw_attestation_report()
            .map_err(|e| anyhow::anyhow!("Failed to generate TDX quote: {}", e))?;
        
        let evidence_json = serde_json::json!({
            "tee_type": "TDX",
            "quote": hex::encode(&quote),
            "quote_size": quote.len(),
        }).to_string();
        
        Ok(AttestationResult::with_evidence_and_tee(
            "tdx-quote".to_string(),
            evidence_json,
            TeeType::IntelTdx,
        ))
    }
}

impl Default for TeeHost {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can provide access to TeeHost
pub trait AsTeeHost {
    fn as_tee_host(&self) -> &TeeHost;
}

/// Helper: Write string to WASM memory using cabi_realloc
/// This allocates memory in the guest using the Component Model ABI
fn write_string_to_wasm<T>(caller: &mut Caller<'_, T>, s: &str) -> i32 {
    let bytes = s.as_bytes();
    let total_len = 4 + bytes.len();  // 4 bytes for length + data
    
    // Try to get cabi_realloc from the guest
    let cabi_realloc = match caller.get_export("cabi_realloc") {
        Some(wasmtime::Extern::Func(func)) => func,
        _ => {
            // Fallback: use fixed offset if cabi_realloc not available
            debug!("cabi_realloc not found, using fixed offset");
            return write_string_to_wasm_fixed(caller, s);
        }
    };
    
    // Call cabi_realloc(old_ptr=0, old_size=0, align=1, new_size=total_len)
    let params = [
        wasmtime::Val::I32(0),           // old_ptr
        wasmtime::Val::I32(0),           // old_size  
        wasmtime::Val::I32(1),           // align
        wasmtime::Val::I32(total_len as i32),  // new_size
    ];
    let mut results = [wasmtime::Val::I32(0)];
    
    if let Err(e) = cabi_realloc.call(&mut *caller, &params, &mut results) {
        error!("cabi_realloc call failed: {}", e);
        return write_string_to_wasm_fixed(caller, s);
    }
    
    let ptr = match results[0] {
        wasmtime::Val::I32(p) => p as usize,
        _ => {
            error!("cabi_realloc returned unexpected type");
            return write_string_to_wasm_fixed(caller, s);
        }
    };
    
    if ptr == 0 {
        error!("cabi_realloc returned null");
        return write_string_to_wasm_fixed(caller, s);
    }
    
    // Get memory and write data
    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(mem)) => mem,
        _ => {
            error!("Failed to get WASM memory export");
            return 0;
        }
    };
    
    let data = memory.data_mut(caller);
    if data.len() < ptr + total_len {
        error!("WASM memory too small after cabi_realloc");
        return 0;
    }
    
    // Write: [len: 4 bytes][data: len bytes]
    data[ptr..ptr+4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
    data[ptr+4..ptr+4+bytes.len()].copy_from_slice(bytes);
    
    debug!("Allocated {} bytes at offset {} via cabi_realloc", total_len, ptr);
    ptr as i32
}

/// Fallback: Write to fixed offset (for modules without cabi_realloc)
fn write_string_to_wasm_fixed<T>(caller: &mut Caller<'_, T>, s: &str) -> i32 {
    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(mem)) => mem,
        _ => {
            error!("Failed to get WASM memory export");
            return 0;
        }
    };

    let bytes = s.as_bytes();
    let len = bytes.len();
    
    // Use high fixed offset to avoid conflicts
    let offset = 1048576;  // 1MB
    
    let data = memory.data_mut(caller);
    if data.len() < offset + len + 8 {
        error!("WASM memory too small for attestation response");
        return 0;
    }
    
    data[offset..offset+4].copy_from_slice(&(len as u32).to_le_bytes());
    data[offset+4..offset+4+len].copy_from_slice(bytes);
    
    offset as i32
}

/// Helper: Read string from WASM memory
fn read_string_from_wasm<T>(caller: &mut Caller<'_, T>, ptr: i32, len: i32) -> Result<String> {
    if ptr < 0 || len < 0 {
        anyhow::bail!("Invalid pointer or length");
    }

    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(mem)) => mem,
        _ => anyhow::bail!("Failed to get WASM memory export"),
    };

    let data = memory.data(caller);
    let start = ptr as usize;
    let end = start + (len as usize);

    if end > data.len() {
        anyhow::bail!("Read out of bounds");
    }

    String::from_utf8(data[start..end].to_vec())
        .context("Invalid UTF-8 in WASM memory")
}

// Re-export base64 functions
mod base64 {
    use base64::{Engine as _, engine::general_purpose};
    
    pub fn encode(data: &[u8]) -> String {
        general_purpose::STANDARD.encode(data)
    }
    
    pub fn decode(s: &str) -> Result<Vec<u8>, base64::DecodeError> {
        general_purpose::STANDARD.decode(s)
    }
}

// use ::base64 as base64_crate;
