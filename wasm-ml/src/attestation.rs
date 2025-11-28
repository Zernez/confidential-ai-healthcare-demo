/// Rust bindings for wasmtime:attestation
/// 
/// Provides safe Rust wrappers around the host functions exposed by
/// the wasmtime:attestation runtime extension.

use serde::{Deserialize, Serialize};

/// Attestation result from host
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    pub success: bool,
    pub token: Option<String>,
    pub evidence: Option<String>,
    pub error: Option<String>,
    pub timestamp: i64,
    #[serde(default)]
    pub tee_type: Option<String>,
}

/// TEE detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeDetectionResult {
    pub tee_type: String,
    pub supports_attestation: bool,
}

// External declarations for host functions
#[link(wasm_import_module = "wasmtime_attestation")]
extern "C" {
    /// Attest the VM (TDX/SEV-SNP)
    /// Returns pointer to JSON AttestationResult
    fn attest_vm() -> i32;
    
    /// Attest the GPU (NVIDIA NRAS)
    /// Returns pointer to JSON AttestationResult
    fn attest_gpu(gpu_index: u32) -> i32;
    
    /// Verify a JWT token
    /// Returns 1 if valid, 0 if invalid
    fn verify_token(token_ptr: *const u8, token_len: i32) -> i32;
    
    /// Clear cached attestation tokens
    fn clear_cache();

    /// Detect TEE type without performing attestation
    /// Returns pointer to JSON with tee_type and supports_attestation
    fn detect_tee() -> i32;
}

/// Detect the TEE type of the current environment
pub fn detect_tee_type() -> Result<TeeDetectionResult, String> {
    unsafe {
        let ptr = detect_tee();
        let result = read_json_from_host::<TeeDetectionResult>(ptr)?;
        Ok(result)
    }
}

/// Attest the VM and return result
pub fn attest_vm_token() -> Result<AttestationResult, String> {
    unsafe {
        let ptr = attest_vm();
        let result = read_json_from_host::<AttestationResult>(ptr)?;
        
        if result.success {
            Ok(result)
        } else {
            Err(result.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }
}

/// Attest the GPU and return result
pub fn attest_gpu_token(gpu_index: u32) -> Result<AttestationResult, String> {
    unsafe {
        let ptr = attest_gpu(gpu_index);
        let result = read_json_from_host::<AttestationResult>(ptr)?;
        
        if result.success {
            Ok(result)
        } else {
            Err(result.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }
}

/// Verify a JWT token or attestation bundle
pub fn verify_attestation_token(token: &str) -> bool {
    unsafe {
        let result = verify_token(token.as_ptr(), token.len() as i32);
        result == 1
    }
}

/// Clear all cached attestation tokens
pub fn clear_attestation_cache() {
    unsafe {
        clear_cache();
    }
}

/// Helper: Read JSON from host memory
unsafe fn read_json_from_host<T: for<'de> Deserialize<'de>>(ptr: i32) -> Result<T, String> {
    if ptr == 0 {
        return Err("Host returned null pointer".to_string());
    }
    
    // Read: [len: 4 bytes][data: len bytes]
    let len_bytes = std::slice::from_raw_parts(ptr as *const u8, 4);
    let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
    
    let data_bytes = std::slice::from_raw_parts((ptr + 4) as *const u8, len);
    let json_str = std::str::from_utf8(data_bytes)
        .map_err(|e| format!("Invalid UTF-8: {}", e))?;
    
    serde_json::from_str(json_str)
        .map_err(|e| format!("JSON parse error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_verify_token_format() {
        // Valid JWT format
        let valid = "eyJhbGc.eyJzdWI.signature";
        assert!(verify_attestation_token(valid));
        
        // Invalid format
        let invalid = "not-a-jwt";
        assert!(!verify_attestation_token(invalid));
    }
}
