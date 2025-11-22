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

/// TEE Host state
pub struct TeeHost {
    /// Cached VM attestation token
    vm_token_cache: Arc<Mutex<Option<String>>>,
    /// Cached GPU attestation token
    gpu_token_cache: Arc<Mutex<Option<String>>>,
    /// NRAS endpoint (optional override)
    nras_endpoint: Option<String>,
}

impl TeeHost {
    pub fn new() -> Self {
        info!("Initializing TEE Host (wasmtime:attestation)");
        Self {
            vm_token_cache: Arc::new(Mutex::new(None)),
            gpu_token_cache: Arc::new(Mutex::new(None)),
            nras_endpoint: None,
        }
    }

    pub fn with_nras_endpoint(mut self, endpoint: String) -> Self {
        self.nras_endpoint = Some(endpoint);
        self
    }

    /// Attest VM (TDX or SEV-SNP)
    pub async fn attest_vm(&self) -> Result<AttestationResult> {
        info!("🔐 Starting VM attestation...");

        // Check cache first
        {
            let cache = self.vm_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("✓ Using cached VM attestation token");
                return Ok(AttestationResult::success(cached_token.clone()));
            }
        }

        // Try TDX first
        #[cfg(feature = "attestation-tdx")]
        {
            debug!("Attempting TDX attestation...");
            match self.attest_tdx().await {
                Ok(result) => {
                    info!("✓ TDX attestation successful");
                    // Cache the token
                    if let Some(token) = &result.token {
                        let mut cache = self.vm_token_cache.lock().unwrap();
                        *cache = Some(token.clone());
                    }
                    return Ok(result);
                }
                Err(e) => {
                    debug!("TDX attestation failed: {}", e);
                }
            }
        }

        // Try SEV-SNP
        // Disabled: requires TPM libraries
        // #[cfg(feature = "attestation")]
        // {
        //     debug!("Attempting SEV-SNP attestation...");
        //     ...
        // }

        // No TEE available
        error!("❌ No TEE attestation available (TDX not found)");
        Ok(AttestationResult::failure(
            "No TEE attestation available. Not running in Intel TDX confidential VM?".to_string()
        ))
    }

    /// Attest GPU via NVIDIA NRAS
    pub async fn attest_gpu(&self, gpu_index: u32) -> Result<AttestationResult> {
        info!("🔐 Starting GPU attestation for device {}...", gpu_index);

        // Check cache first
        {
            let cache = self.gpu_token_cache.lock().unwrap();
            if let Some(cached_token) = cache.as_ref() {
                info!("✓ Using cached GPU attestation token");
                return Ok(AttestationResult::success(cached_token.clone()));
            }
        }

        #[cfg(feature = "attestation-nvidia")]
        {
            debug!("Collecting GPU evidence...");
            
            // Call attestation-rs library
            match lunal_attestation::nvidia::attest::attest_remote_token(
                gpu_index,
                None, // Use default nonce
                self.nras_endpoint.clone(), // Clone the Option<String>
            ).await {
                Ok(token) => {
                    info!("✓ GPU attestation successful");
                    info!("  Token length: {} chars", token.len());
                    
                    // Cache the token
                    let mut cache = self.gpu_token_cache.lock().unwrap();
                    *cache = Some(token.clone());
                    
                    Ok(AttestationResult::success(token))
                }
                Err(e) => {
                    error!("❌ GPU attestation failed: {}", e);
                    Ok(AttestationResult::failure(format!("GPU attestation failed: {}", e)))
                }
            }
        }

        #[cfg(not(feature = "attestation-nvidia"))]
        {
            error!("❌ GPU attestation not compiled in (missing attestation-nvidia feature)");
            Ok(AttestationResult::failure(
                "GPU attestation not available. Recompile with --features attestation-nvidia".to_string()
            ))
        }
    }

    /// Get VM evidence without remote attestation
    pub async fn get_vm_evidence(&self) -> Result<AttestationResult> {
        info!("📄 Collecting VM evidence...");

        #[cfg(feature = "attestation-tdx")]
        {
            debug!("Collecting TDX evidence...");
            match self.collect_tdx_evidence().await {
                Ok(evidence_json) => {
                    info!("✓ TDX evidence collected");
                    return Ok(AttestationResult::with_evidence(
                        "evidence-only".to_string(),
                        evidence_json,
                    ));
                }
                Err(e) => {
                    debug!("TDX evidence collection failed: {}", e);
                }
            }
        }

        // Disabled: SEV-SNP requires TPM
        // #[cfg(feature = "attestation")]
        // {
        //     debug!("Collecting SEV-SNP evidence...");
        //     ...
        // }

        error!("❌ No TEE evidence available");
        Ok(AttestationResult::failure("No TEE available".to_string()))
    }

    /// Get GPU evidence without remote attestation
    pub async fn get_gpu_evidence(&self, gpu_index: u32) -> Result<AttestationResult> {
        info!("📄 Collecting GPU evidence for device {}...", gpu_index);

        #[cfg(feature = "attestation-nvidia")]
        {
            match lunal_attestation::nvidia::attest::collect_evidence(gpu_index, None) {
                Ok(evidence) => {
                    info!("✓ GPU evidence collected");
                    let evidence_json = evidence.to_json_pretty()
                        .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e));
                    
                    Ok(AttestationResult::with_evidence(
                        "evidence-only".to_string(),
                        evidence_json,
                    ))
                }
                Err(e) => {
                    error!("❌ GPU evidence collection failed: {}", e);
                    Ok(AttestationResult::failure(format!("GPU evidence collection failed: {}", e)))
                }
            }
        }

        #[cfg(not(feature = "attestation-nvidia"))]
        {
            Ok(AttestationResult::failure("GPU evidence not available".to_string()))
        }
    }

    /// Verify a JWT token (basic validation)
    pub fn verify_token(&self, token: &str) -> Result<bool> {
        debug!("Verifying token...");
        
        if token.is_empty() {
            return Ok(false);
        }

        // Basic JWT structure validation
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            warn!("Invalid JWT format (expected 3 parts, got {})", parts.len());
            return Ok(false);
        }

        // TODO: Full JWT signature verification with public key
        // For now, just check structure
        debug!("✓ Token structure valid");
        Ok(true)
    }

    /// Clear all cached tokens
    pub fn clear_cache(&self) {
        info!("Clearing attestation token cache");
        let mut vm_cache = self.vm_token_cache.lock().unwrap();
        *vm_cache = None;
        let mut gpu_cache = self.gpu_token_cache.lock().unwrap();
        *gpu_cache = None;
    }

    /// Register host functions with Wasmtime linker
    pub fn register_functions<T>(&self, linker: &mut Linker<T>) -> Result<()>
    where
        T: Send + 'static,
    {
        info!("Registering wasmtime:attestation functions...");

        // wasmtime_attestation_vm() -> pointer to JSON string
        linker.func_wrap(
            "wasmtime_attestation",
            "attest_vm",
            |mut caller: Caller<'_, T>| -> i32 {
                info!("[WASM] attest_vm() called");
                
                // We need to make this work with the sync context
                // For now, return a placeholder
                let result = AttestationResult::failure(
                    "VM attestation requires async runtime - use attest_vm_async".to_string()
                );
                let json = result.to_json();
                
                // Allocate memory in WASM and write JSON
                Self::write_string_to_wasm(&mut caller, &json)
            },
        )?;

        // wasmtime_attestation_gpu(gpu_index: u32) -> pointer to JSON string
        linker.func_wrap(
            "wasmtime_attestation",
            "attest_gpu",
            |mut caller: Caller<'_, T>, gpu_index: u32| -> i32 {
                info!("[WASM] attest_gpu({}) called", gpu_index);
                
                let result = AttestationResult::failure(
                    "GPU attestation requires async runtime - use attest_gpu_async".to_string()
                );
                let json = result.to_json();
                
                Self::write_string_to_wasm(&mut caller, &json)
            },
        )?;

        // wasmtime_attestation_verify(token_ptr: i32, token_len: i32) -> bool
        linker.func_wrap(
            "wasmtime_attestation",
            "verify_token",
            |mut caller: Caller<'_, T>, token_ptr: i32, token_len: i32| -> i32 {
                debug!("[WASM] verify_token() called");
                
                // Read token from WASM memory
                match Self::read_string_from_wasm(&mut caller, token_ptr, token_len) {
                    Ok(token) => {
                        // Basic verification
                        let parts: Vec<&str> = token.split('.').collect();
                        if parts.len() == 3 { 1 } else { 0 }
                    }
                    Err(_) => 0,
                }
            },
        )?;

        // wasmtime_attestation_clear_cache()
        linker.func_wrap(
            "wasmtime_attestation",
            "clear_cache",
            |_caller: Caller<'_, T>| {
                info!("[WASM] clear_cache() called");
                // We can't access self here, so this is a no-op for now
                // TODO: Use a static or pass TeeHost through HostState
            },
        )?;

        info!("✓ wasmtime:attestation functions registered");
        Ok(())
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

        // Allocate memory in WASM (call __wasm_malloc if available)
        // For now, we'll use a simple approach: write to a fixed location
        // TODO: Proper memory allocation via WASM allocator
        
        let bytes = s.as_bytes();
        let len = bytes.len();
        
        // Write length at offset 0
        let data = memory.data_mut(caller);
        if data.len() < len + 8 {
            error!("WASM memory too small");
            return 0;
        }
        
        // Write: [len: 4 bytes][data: len bytes]
        let offset = 1024; // Fixed offset for now
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

    // ============================================
    // Platform-specific attestation implementations
    // ============================================

    #[cfg(feature = "attestation-tdx")]
    async fn attest_tdx(&self) -> Result<AttestationResult> {
        // Use the attestation module functions
        let report_data = vec![0u8; 64];
        let quote = lunal_attestation::attestation::get_raw_attestation_report()
            .context("Failed to generate TDX quote")?;
        
        let evidence_json = serde_json::json!({
            "tee_type": "TDX",
            "quote": hex::encode(&quote),
        }).to_string();
        
        // TODO: Send to remote attestation service
        Ok(AttestationResult::with_evidence(
            "tdx-quote".to_string(),
            evidence_json,
        ))
    }

    // SEV-SNP functions disabled (require TPM)
    /*
    #[cfg(feature = "attestation")]
    async fn attest_sev_snp(&self) -> Result<AttestationResult> {
        use lunal_attestation::attestation::sev_snp;
        
        // Generate SEV-SNP attestation report
        let report_data = vec![0u8; 64];
        let report = sev_snp::get_report(&report_data)
            .context("Failed to generate SEV-SNP report")?;
        
        let evidence_json = serde_json::json!({
            "tee_type": "SEV-SNP",
            "report": hex::encode(&report),
            "report_data": hex::encode(&report_data),
        }).to_string();
        
        // TODO: Send to Azure attestation service
        Ok(AttestationResult::with_evidence(
            "sev-snp-report".to_string(),
            evidence_json,
        ))
    }
    */

    #[cfg(feature = "attestation-tdx")]
    async fn collect_tdx_evidence(&self) -> Result<String> {
        let quote = lunal_attestation::attestation::get_raw_attestation_report()?;
        
        Ok(serde_json::json!({
            "tee_type": "TDX",
            "quote": hex::encode(&quote),
        }).to_string())
    }

    // SEV evidence collection disabled
    /*
    #[cfg(feature = "attestation")]
    async fn collect_sev_evidence(&self) -> Result<String> {
        use lunal_attestation::attestation::sev_snp;
        
        let report_data = vec![0u8; 64];
        let report = sev_snp::get_report(&report_data)?;
        
        Ok(serde_json::json!({
            "tee_type": "SEV-SNP",
            "report": hex::encode(&report),
        }).to_string())
    }
    */
}

impl Default for TeeHost {
    fn default() -> Self {
        Self::new()
    }
}
