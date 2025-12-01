/**
 * @file attestation.hpp
 * @brief TEE Attestation bindings for wasmtime_attestation host functions
 * 
 * Provides C++ bindings for the TEE attestation interface implemented
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
// Module name uses underscore (wasmtime_attestation), function names use underscore
// ═══════════════════════════════════════════════════════════════════════════

/// Detect TEE type - returns pointer to JSON string in WASM memory
__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("detect_tee")))
int32_t __attestation_detect_tee(void);

/// Attest VM - returns pointer to JSON string in WASM memory
__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("attest_vm")))
int32_t __attestation_attest_vm(void);

/// Attest GPU - returns pointer to JSON string in WASM memory
__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("attest_gpu")))
int32_t __attestation_attest_gpu(uint32_t gpu_index);

#ifdef __cplusplus
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// C++ Wrapper Classes
// ═══════════════════════════════════════════════════════════════════════════

namespace attestation {

/// Helper to read a JSON string from WASM memory pointer
/// The host writes: [len: 4 bytes][data: len bytes] at the returned pointer
/// In WASM32, the returned int32_t IS the memory address
inline std::string read_json_from_ptr(int32_t ptr) {
    if (ptr == 0) return "{}";
    
    // In WASM32, the pointer value IS the memory address
    // Read length (first 4 bytes)
    const uint8_t* base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptr));
    uint32_t len = base[0] | (base[1] << 8) | (base[2] << 16) | (base[3] << 24);
    
    if (len == 0 || len > 1024 * 1024) return "{}"; // Sanity check
    
    // Read data (starts after the 4-byte length)
    return std::string(reinterpret_cast<const char*>(base + 4), len);
}

/// Simple JSON string value extractor
inline std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    
    // Skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    
    if (pos >= json.size()) return "";
    
    if (json[pos] == '"') {
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    } else {
        // Non-string value (bool, number, null)
        size_t end = json.find_first_of(",}\n", pos);
        if (end == std::string::npos) end = json.size();
        std::string val = json.substr(pos, end - pos);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
        return val;
    }
}

inline bool json_get_bool(const std::string& json, const std::string& key) {
    return json_get_string(json, key) == "true";
}

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
    std::string token;
    std::string error;
};

/**
 * @brief Detect the TEE type (AMD SEV-SNP, Intel TDX, etc.)
 */
inline TeeInfo detect_tee_type() {
    TeeInfo info;
    
    int32_t ptr = __attestation_detect_tee();
    std::string json = read_json_from_ptr(ptr);
    
    info.tee_type = json_get_string(json, "tee_type");
    info.supports_attestation = json_get_bool(json, "supports_attestation");
    
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
    
    int32_t ptr = __attestation_attest_vm();
    std::string json = read_json_from_ptr(ptr);
    
    result.success = json_get_bool(json, "success");
    
    if (result.success) {
        result.token = json_get_string(json, "token");
        if (result.token.empty()) {
            result.token = json_get_string(json, "evidence");
        }
    } else {
        result.error = json_get_string(json, "error");
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
    
    int32_t ptr = __attestation_attest_gpu(gpu_index);
    std::string json = read_json_from_ptr(ptr);
    
    result.success = json_get_bool(json, "success");
    
    if (result.success) {
        result.token = json_get_string(json, "token");
    } else {
        result.error = json_get_string(json, "error");
        if (result.error.empty()) {
            result.error = "GPU attestation failed";
        }
    }
    
    return result;
}

} // namespace attestation

#endif // ATTESTATION_HPP
