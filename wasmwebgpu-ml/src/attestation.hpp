/**
 * @file attestation.hpp
 * @brief TEE Attestation bindings for wasmtime:attestation host functions
 * 
 * This provides C++ bindings for the TEE attestation interface implemented
 * by the wasmtime-gpu-host runtime.
 */

#ifndef ATTESTATION_HPP
#define ATTESTATION_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Raw ABI declarations for wasmtime:attestation host functions
// ═══════════════════════════════════════════════════════════════════════════

/// Result structure for TEE type detection
/// Layout: tee_type_ptr(4), tee_type_len(4), supports_attestation(4)
typedef struct {
    uint32_t tee_type_ptr;
    uint32_t tee_type_len;
    uint32_t supports_attestation;
} attestation_tee_info_raw_t;

/// Result structure for attestation token
/// Layout: is_err(4), token_ptr(4), token_len(4) OR is_err(4), error_ptr(4), error_len(4)
typedef struct {
    uint32_t is_err;
    uint32_t data_ptr;
    uint32_t data_len;
} attestation_result_raw_t;

/// Detect TEE type (raw ABI)
__attribute__((import_module("wasmtime:attestation")))
__attribute__((import_name("detect-tee-type")))
void __attestation_detect_tee_type_raw(attestation_tee_info_raw_t* retptr);

/// Attest VM and get token (raw ABI)
__attribute__((import_module("wasmtime:attestation")))
__attribute__((import_name("attest-vm-token")))
void __attestation_attest_vm_token_raw(attestation_result_raw_t* retptr);

/// Attest GPU and get token (raw ABI)
__attribute__((import_module("wasmtime:attestation")))
__attribute__((import_name("attest-gpu-token")))
void __attestation_attest_gpu_token_raw(uint32_t gpu_index, attestation_result_raw_t* retptr);

#ifdef __cplusplus
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// C++ Wrapper Classes
// ═══════════════════════════════════════════════════════════════════════════

namespace attestation {

/**
 * @struct TeeInfo
 * @brief Information about the TEE environment
 */
struct TeeInfo {
    std::string tee_type;
    bool supports_attestation;
};

/**
 * @struct AttestationResult
 * @brief Result of an attestation operation
 */
struct AttestationResult {
    bool success;
    std::string token;      // On success: the attestation token
    std::string error;      // On failure: error message
};

/**
 * @brief Detect the TEE type (AMD SEV-SNP, Intel TDX, etc.)
 */
inline TeeInfo detect_tee_type() {
    TeeInfo info;
    attestation_tee_info_raw_t raw;
    
    __attestation_detect_tee_type_raw(&raw);
    
    if (raw.tee_type_ptr != 0 && raw.tee_type_len > 0) {
        const char* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(raw.tee_type_ptr));
        info.tee_type = std::string(src, raw.tee_type_len);
    } else {
        info.tee_type = "Unknown";
    }
    
    info.supports_attestation = (raw.supports_attestation != 0);
    
    return info;
}

/**
 * @brief Attest the VM and get a token
 */
inline AttestationResult attest_vm() {
    AttestationResult result;
    attestation_result_raw_t raw;
    
    __attestation_attest_vm_token_raw(&raw);
    
    if (raw.is_err == 0) {
        result.success = true;
        if (raw.data_ptr != 0 && raw.data_len > 0) {
            const char* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(raw.data_ptr));
            result.token = std::string(src, raw.data_len);
        }
    } else {
        result.success = false;
        if (raw.data_ptr != 0 && raw.data_len > 0) {
            const char* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(raw.data_ptr));
            result.error = std::string(src, raw.data_len);
        } else {
            result.error = "Unknown error";
        }
    }
    
    return result;
}

/**
 * @brief Attest a GPU and get a token
 */
inline AttestationResult attest_gpu(uint32_t gpu_index = 0) {
    AttestationResult result;
    attestation_result_raw_t raw;
    
    __attestation_attest_gpu_token_raw(gpu_index, &raw);
    
    if (raw.is_err == 0) {
        result.success = true;
        if (raw.data_ptr != 0 && raw.data_len > 0) {
            const char* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(raw.data_ptr));
            result.token = std::string(src, raw.data_len);
        }
    } else {
        result.success = false;
        if (raw.data_ptr != 0 && raw.data_len > 0) {
            const char* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(raw.data_ptr));
            result.error = std::string(src, raw.data_len);
        } else {
            result.error = "Unknown error";
        }
    }
    
    return result;
}

} // namespace attestation

#endif // ATTESTATION_HPP
