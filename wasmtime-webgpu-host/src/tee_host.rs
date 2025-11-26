/**
 * TEE Host - wasmtime:attestation Implementation
 * 
 * Provides attestation capabilities for WASM guests running in Confidential VMs
 * with GPU support. This module exposes host functions for:
 * - VM attestation (TDX/SEV-SNP)
 * - GPU attestation (NVIDIA NRAS)
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
        // Note: This requires creating a new Arc, which is fine for initialization
        self.inner = Arc::new(TeeHostInner {
            vm_token_cache: Mutex::new(None),
            gpu_token_cache: Mutex::new(None),
            nras_endpoint: Some(endpoint),
        });
        self
    }

    /// Attest VM (TDX) - synchronous wrapper
    pub fn attest_vm_sync(&self) -> AttestationResult {
        info!("🔐 Starting VM attestation...");

        // Check cache first
        {
            let cache = self.inner.vm_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("✓ Using cached VM attestation token");
                return AttestationResult::success(cached_token.clone());
            }
        }

        // Try TDX
        #[cfg(feature = "attestation-tdx")]
        {
            debug!("Attempting TDX attestation...");
            match self.attest_tdx_sync() {
                Ok(result) => {
                    info!("✓ TDX attestation successful");
                    // Cache the token
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
        warn!("⚠️  No TEE attestation available (TDX not found or not in CVM)");
        AttestationResult::failure(
            "No TEE attestation available. Not running in Intel TDX confidential VM?".to_string()
        )
    }

    /// Attest GPU via NVIDIA NRAS - synchronous wrapper using block_on
    pub fn attest_gpu_sync(&self, gpu_index: u32) -> AttestationResult {
        info!("🔐 Starting GPU attestation for device {}...", gpu_index);

        // Check cache first
        {
            let cache = self.inner.gpu_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("✓ Using cached GPU attestation token");
                return AttestationResult::success(cached_token.clone());
            }
        }

        #[cfg(feature = "attestation-nvidia")]
        {
            debug!("Collecting GPU evidence and requesting token from NRAS...");
            
            // Get the tokio runtime handle and run async code synchronously
            let nras_endpoint = self.inner.nras_endpoint.clone();
            
            // Use tokio's block_in_place to run async code from sync context
            let result = tokio::task::block_in_place(|| {
                Handle::current().block_on(async {
                    lunal_attestation::nvidia::attest::attest_remote_token(
                        gpu_index,
                        None, // Use default nonce
                        nras_endpoint,
                    ).await
                })
            });
            
            match result {
                Ok(token) => {
                    info!("✓ GPU attestation successful");
                    info!("  Token length: {} chars", token.len());
                    
                    // Cache the token
                    let mut cache = self.inner.gpu_token_cache.lock().unwrap();
                    *cache = Some(token.clone());
                    
                    return AttestationResult::success(token);
                }
                Err(e) => {
                    error!("❌ GPU attestation failed: {}", e);
                    return AttestationResult::failure(format!("GPU attestation failed: {}", e));
                }
            }
        }

        #[cfg(not(feature = "attestation-nvidia"))]
        {
            error!("❌ GPU attestation not compiled in (missing attestation-nvidia feature)");
            AttestationResult::failure(
                "GPU attestation not available. Recompile with --features attestation-nvidia".to_string()
            )
        }
    }

    /// Verify a JWT token (basic validation)
    pub fn verify_token(&self, token: &str) -> bool {
        debug!("Verifying token...");
        
        if token.is_empty() {
            return false;
        }

        // Basic JWT structure validation
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            warn!("Invalid JWT format (expected 3 parts, got {})", parts.len());
            return false;
        }

        // TODO: Full JWT signature verification with public key
        debug!("✓ Token structure valid");
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
    /// T must provide access to TeeHost via the get_tee_host trait
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
                
                // Get TeeHost from caller's state and perform attestation
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
                
                // Get TeeHost from caller's state and perform attestation
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
                
                // Read token from WASM memory
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

        info!("✓ wasmtime:attestation functions registered");
        Ok(())
    }

    // ============================================
    // Platform-specific attestation implementations
    // ============================================

    #[cfg(feature = "attestation-tdx")]
    fn attest_tdx_sync(&self) -> Result<AttestationResult> {
        // Use the attestation module functions
        let report_data = vec![0u8; 64];
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

    #[cfg(feature = "attestation-tdx")]
    fn collect_tdx_evidence(&self) -> Result<String> {
        let quote = lunal_attestation::attestation::get_raw_attestation_report()
            .map_err(|e| anyhow::anyhow!("Failed to generate TDX evidence: {}", e))?;
        
        Ok(serde_json::json!({
            "tee_type": "TDX",
            "quote": hex::encode(&quote),
        }).to_string())
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
    // Get WASM memory
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
    let offset = 1024; // Fixed offset for attestation responses
    
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
