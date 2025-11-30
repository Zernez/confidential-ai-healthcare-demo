//! Attestation Host Functions
//!
//! Provides stub implementations for wasmtime_attestation interface.
//! These allow WASM modules that use attestation to run without errors.

use tracing::{info, warn};
use wasmtime::{Caller, Linker};

/// Register attestation host functions with the linker
pub fn add_to_linker<T>(linker: &mut Linker<T>) -> anyhow::Result<()> {
    // ═══════════════════════════════════════════════════════════════════════
    // wasmtime_attestation interface
    // These are stub implementations that allow the WASM to run
    // ═══════════════════════════════════════════════════════════════════════
    
    // detect_tee - returns TEE type (0 = none, 1 = SGX, 2 = SEV-SNP, 3 = TDX)
    linker.func_wrap(
        "wasmtime_attestation",
        "detect_tee",
        move |_caller: Caller<'_, T>| -> u32 {
            // Return SEV-SNP (2) since we're on Azure CVM
            // In production, this would actually detect the TEE
            info!("[Attestation] detect_tee called - returning SEV-SNP (2)");
            2 // SEV-SNP
        },
    )?;
    
    // get_attestation_report - gets attestation evidence
    // Returns 0 on success, writes report to out_ptr
    linker.func_wrap(
        "wasmtime_attestation",
        "get_attestation_report",
        move |mut caller: Caller<'_, T>, 
              nonce_ptr: u32, 
              nonce_len: u32, 
              out_ptr: u32, 
              out_len_ptr: u32| -> u32 {
            info!("[Attestation] get_attestation_report called (nonce_len={})", nonce_len);
            
            // Create a mock attestation report
            // In production, this would call the actual TEE attestation
            let mock_report = b"MOCK_ATTESTATION_REPORT_SEV_SNP_V1";
            
            // Write report to output
            if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                let _ = memory.write(&mut caller, out_ptr as usize, mock_report);
                let _ = memory.write(&mut caller, out_len_ptr as usize, &(mock_report.len() as u32).to_le_bytes());
            }
            
            0 // Success
        },
    )?;
    
    // verify_attestation - verifies attestation evidence
    // Returns 0 on success (verified), non-zero on failure
    linker.func_wrap(
        "wasmtime_attestation",
        "verify_attestation",
        move |_caller: Caller<'_, T>,
              _report_ptr: u32,
              _report_len: u32,
              _expected_hash_ptr: u32,
              _expected_hash_len: u32| -> u32 {
            info!("[Attestation] verify_attestation called - returning success");
            0 // Success (mock verification)
        },
    )?;
    
    // get_gpu_attestation - gets NVIDIA GPU attestation
    linker.func_wrap(
        "wasmtime_attestation",
        "get_gpu_attestation",
        move |mut caller: Caller<'_, T>,
              gpu_index: u32,
              nonce_ptr: u32,
              nonce_len: u32,
              out_ptr: u32,
              out_len_ptr: u32| -> u32 {
            info!("[Attestation] get_gpu_attestation called (gpu={}, nonce_len={})", gpu_index, nonce_len);
            
            // Create mock GPU attestation report
            let mock_gpu_report = format!(
                "{{\"gpu_index\":{},\"driver\":\"570.x\",\"attestation\":\"MOCK_NVIDIA_CC_REPORT\"}}",
                gpu_index
            );
            let report_bytes = mock_gpu_report.as_bytes();
            
            if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                let _ = memory.write(&mut caller, out_ptr as usize, report_bytes);
                let _ = memory.write(&mut caller, out_len_ptr as usize, &(report_bytes.len() as u32).to_le_bytes());
            }
            
            0 // Success
        },
    )?;
    
    // get_combined_attestation - gets both VM and GPU attestation
    linker.func_wrap(
        "wasmtime_attestation",
        "get_combined_attestation",
        move |mut caller: Caller<'_, T>,
              nonce_ptr: u32,
              nonce_len: u32,
              out_ptr: u32,
              out_len_ptr: u32| -> u32 {
            info!("[Attestation] get_combined_attestation called (nonce_len={})", nonce_len);
            
            let mock_combined = r#"{"vm":{"tee":"SEV-SNP","report":"MOCK"},"gpu":{"driver":"570.x","report":"MOCK"}}"#;
            let report_bytes = mock_combined.as_bytes();
            
            if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                let _ = memory.write(&mut caller, out_ptr as usize, report_bytes);
                let _ = memory.write(&mut caller, out_len_ptr as usize, &(report_bytes.len() as u32).to_le_bytes());
            }
            
            0 // Success
        },
    )?;
    
    info!("[Attestation] Host functions registered (stub implementation)");
    warn!("[Attestation] Using mock attestation - not suitable for production!");
    
    Ok(())
}
