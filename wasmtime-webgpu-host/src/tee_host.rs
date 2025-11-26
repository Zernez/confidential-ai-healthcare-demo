/**
 * TEE Host - wasmtime:attestation Implementation
 * 
 * Provides attestation capabilities for WASM guests running in Confidential VMs
 * with GPU support. This module exposes host functions for:
 * - VM attestation (TDX/SEV-SNP)
 * - GPU attestation (NVIDIA LOCAL via nvattest, with NRAS fallback)
 * - Token verification
 * - Evidence collection
 */

use anyhow::{Context, Result};
use log::{debug, info, warn, error};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use wasmtime::{Caller, Linker};
use tokio::runtime::Handle;

// Conditional import based on features
#[cfg(feature = "lunal-attestation")]
use lunal_attestation;

/// Attestation result returned to WASM guest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Success flag
    pub success: bool,
    /// JWT token (if successful)
    pub token: Option<String>,
    /// Evidence JSON (if requested)
    pub evidence: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Timestamp
    pub timestamp: i64,
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
    /// Cached VM attestation token
    vm_token_cache: Mutex<Option<String>>,
    /// Cached GPU attestation token
    gpu_token_cache: Mutex<Option<String>>,
    /// NRAS endpoint (optional override)
    nras_endpoint: Option<String>,
}

impl TeeHost {
    pub fn new() -> Self {
        info!("Initializing TEE Host (wasmtime:attestation)");
        Self {
            inner: Arc::new(TeeHostInner {
                vm_token_cache: Mutex::new(None),
                gpu_token_cache: Mutex::new(None),
                nras_endpoint: None,
            }),
        }
    }

    pub fn with_nras_endpoint(mut self, endpoint: String) -> Self {
        self.inner = Arc::new(TeeHostInner {
            vm_token_cache: Mutex::new(None),
            gpu_token_cache: Mutex::new(None),
            nras_endpoint: Some(endpoint),
        });
        self
    }

    /// Attest VM (TDX) - synchronous wrapper
    pub fn attest_vm_sync(&self) -> AttestationResult {
        info!("Starting VM attestation...");

        // Check cache first
        {
            let cache = self.inner.vm_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("Using cached VM attestation token");
                return AttestationResult::success(cached_token.clone());
            }
        }

        // Try TDX
        #[cfg(feature = "attestation-tdx")]
        {
            debug!("Attempting TDX attestation...");
            match self.attest_tdx_sync() {
                Ok(result) => {
                    info!("TDX attestation successful");
                    if let Some(token) = &result.token {
                        let mut cache = self.inner.vm_token_cache.lock().unwrap();
                        *cache = Some(token.clone());
                    }
                    return result;
                }
                Err(e) => {
                    debug!("TDX attestation failed: {}", e);
                }
            }
        }

        // No TEE available
        warn!("No TEE attestation available (TDX not found or not in CVM)");
        AttestationResult::failure(
            "No TEE attestation available. Not running in Intel TDX confidential VM?".to_string()
        )
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

        // Strategy: Try LOCAL attestation first (works on Azure/cloud without NRAS subscription)
        // Then fall back to NRAS if local fails
        
        info!("Attempting LOCAL GPU attestation via nvattest CLI...");
        match self.attest_gpu_local(gpu_index) {
            Ok(result) => {
                info!("Local GPU attestation successful!");
                // Cache the token
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
            let nras_endpoint = self.inner.nras_endpoint.clone();
            
            let result = tokio::task::block_in_place(|| {
                Handle::current().block_on(async {
                    lunal_attestation::nvidia::attest::attest_remote_token(
                        gpu_index,
                        None,
                        nras_endpoint,
                    ).await
                })
            });
            
            match result {
                Ok(token) => {
                    info!("GPU attestation successful (NRAS token)");
                    info!("  Token length: {} chars", token.len());
                    
                    let mut cache = self.inner.gpu_token_cache.lock().unwrap();
                    *cache = Some(token.clone());
                    
                    return AttestationResult::success(token);
                }
                Err(e) => {
                    let error_str = format!("{}", e);
                    if error_str.contains("403") || error_str.contains("Forbidden") {
                        error!("NRAS returned 403 Forbidden - API key may not have attestation permissions");
                    } else {
                        error!("NRAS attestation failed: {}", e);
                    }
                    return AttestationResult::failure(format!("GPU attestation failed: {}", e));
                }
            }
        }

        #[cfg(not(feature = "attestation-nvidia"))]
        {
            error!("GPU attestation not compiled in");
            AttestationResult::failure("GPU attestation not available".to_string())
        }
    }

    /// Perform LOCAL GPU attestation using nvattest CLI
    /// This works on Azure/cloud VMs without NRAS subscription!
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
        // Format: [["JWT", "eyJhbG..."], {"GPU-0": "..."}]
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
        
        // Log success details
        info!("✓ Local attestation completed");
        info!("  Verifier: LOCAL (nvattest)");
        info!("  Hardware: {}", hw_model);
        info!("  Driver: {}", driver_version);
        info!("  Measurement: {}", meas_result);
        
        if let Some(ref t) = token {
            info!("  JWT Token: {} chars", t.len());
        }
        
        // Return result with token and evidence
        Ok(AttestationResult {
            success: true,
            token: token,
            evidence: Some(claims),
            error: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
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
            warn!("Invalid JWT format (expected 3 parts, got {})", parts.len());
            return false;
        }

        // For local verifier tokens, signature may be empty but structure is valid
        debug!("Token structure valid");
        true
    }

    /// Clear all cached tokens
    pub fn clear_cache(&self) {
        info!("Clearing attestation token cache");
        let mut vm_cache = self.inner.vm_token_cache.lock().unwrap();
        *vm_cache = None;
        let mut gpu_cache = self.inner.gpu_token_cache.lock().unwrap();
        *gpu_cache = None;
    }

    /// Register host functions with Wasmtime linker
    pub fn register_functions<T>(&self, linker: &mut Linker<T>) -> Result<()>
    where
        T: AsTeeHost + Send + 'static,
    {
        info!("Registering wasmtime:attestation functions...");

        // wasmtime_attestation_vm() -> pointer to JSON string
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

        // wasmtime_attestation_gpu(gpu_index: u32) -> pointer to JSON string
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

        // wasmtime_attestation_verify(token_ptr: i32, token_len: i32) -> bool
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

        // wasmtime_attestation_clear_cache()
        linker.func_wrap(
            "wasmtime_attestation",
            "clear_cache",
            |caller: Caller<'_, T>| {
                info!("[WASM] clear_cache() called");
                let tee_host = caller.data().as_tee_host();
                tee_host.clear_cache();
            },
        )?;

        info!("wasmtime:attestation functions registered");
        Ok(())
    }

    // ============================================
    // Platform-specific attestation implementations
    // ============================================

    #[cfg(feature = "attestation-tdx")]
    fn attest_tdx_sync(&self) -> Result<AttestationResult> {
        let quote = lunal_attestation::attestation::get_raw_attestation_report()
            .map_err(|e| anyhow::anyhow!("Failed to generate TDX quote: {}", e))?;
        
        let evidence_json = serde_json::json!({
            "tee_type": "TDX",
            "quote": hex::encode(&quote),
            "quote_size": quote.len(),
        }).to_string();
        
        Ok(AttestationResult::with_evidence(
            "tdx-quote".to_string(),
            evidence_json,
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

/// Helper: Write string to WASM memory
fn write_string_to_wasm<T>(caller: &mut Caller<'_, T>, s: &str) -> i32 {
    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(mem)) => mem,
        _ => {
            error!("Failed to get WASM memory export");
            return 0;
        }
    };

    let bytes = s.as_bytes();
    let len = bytes.len();
    
    // Write: [len: 4 bytes][data: len bytes] at a fixed offset
    let offset = 1024;
    
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
