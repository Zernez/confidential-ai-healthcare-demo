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
// Raw ABI declarations for wasmtime_attestation host functions
// NOTE: Module name uses underscore (wasmtime_attestation) not colon
// ═══════════════════════════════════════════════════════════════════════════

/// Detect TEE type - returns pointer to JSON string in WASM memory
/// JSON format: {"tee_type": "AMD SEV-SNP", "supports_attestation": true}
__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("detect_tee")))
int32_t __attestation_detect_tee_raw(void);

/// Attest VM - returns pointer to JSON string in WASM memory
/// JSON format: {"success": true, "token": "...", "evidence": "...", ...}
__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("attest_vm")))
int32_t __attestation_attest_vm_raw(void);

/// Attest GPU - returns pointer to JSON string in WASM memory
__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("attest_gpu")))
int32_t __attestation_attest_gpu_raw(uint32_t gpu_index);

#ifdef __cplusplus
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// C++ Wrapper Classes
// ═══════════════════════════════════════════════════════════════════════════

/// Helper to read a JSON string from WASM memory pointer
/// Format at ptr: [len: 4 bytes][data: len bytes]
inline std::string read_json_from_ptr(int32_t ptr) {
    if (ptr == 0) return "{}";
    
    // Read length (4 bytes at ptr)
    const uint32_t* len_ptr = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(ptr));
    uint32_t len = *len_ptr;
    
    if (len == 0 || len > 1024 * 1024) return "{}"; // Sanity check
    
    // Read data (starts at ptr + 4)
    const char* data_ptr = reinterpret_cast<const char*>(static_cast<uintptr_t>(ptr + 4));
    return std::string(data_ptr, len);
}

/// Simple JSON value extractor (avoids full JSON parser dependency)
inline std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    
    // Skip whitespace and find start of value
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    if (pos >= json.size()) return "";
    
    if (json[pos] == '"') {
        // String value
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    } else {
        // Non-string value (bool, number)
        size_t end = json.find_first_of(",}", pos);
        if (end == std::string::npos) end = json.size();
        std::string val = json.substr(pos, end - pos);
        // Trim whitespace
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
        return val;
    }
}

inline bool extract_json_bool(const std::string& json, const std::string& key) {
    std::string val = extract_json_string(json, key);
    return val == "true";
}

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
    
    int32_t ptr = __attestation_detect_tee_raw();
    std::string json = read_json_from_ptr(ptr);
    
    info.tee_type = extract_json_string(json, "tee_type");
    info.supports_attestation = extract_json_bool(json, "supports_attestation");
    
    if (info.tee_type.empty()) {
        info.tee_type = "Unknown";
    }
    
    return info;
}

/**
 * @brief Attest the VM and get a token
 */
inline AttestationResult attest_vm() {
    AttestationResult result;
    
    int32_t ptr = __attestation_attest_vm_raw();
    std::string json = read_json_from_ptr(ptr);
    
    result.success = extract_json_bool(json, "success");
    
    if (result.success) {
        result.token = extract_json_string(json, "token");
        if (result.token.empty()) {
            result.token = extract_json_string(json, "evidence");
        }
    } else {
        result.error = extract_json_string(json, "error");
        if (result.error.empty()) {
            result.error = "Attestation failed";
        }
    }
    
    return result;
}

/**
 * @brief Attest a GPU and get a token
 */
inline AttestationResult attest_gpu(uint32_t gpu_index = 0) {
    AttestationResult result;
    
    int32_t ptr = __attestation_attest_gpu_raw(gpu_index);
    std::string json = read_json_from_ptr(ptr);
    
    result.success = extract_json_bool(json, "success");
    
    if (result.success) {
        result.token = extract_json_string(json, "token");
    } else {
        result.error = extract_json_string(json, "error");
        if (result.error.empty()) {
            result.error = "GPU attestation failed";
        }
    }
    
    return result;
}

} // namespace attestation

#endif // ATTESTATION_HPP
