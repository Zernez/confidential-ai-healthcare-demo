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
// ═══════════════════════════════════════════════════════════════════════════

__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("detect_tee")))
int32_t __attestation_detect_tee(void);

__attribute__((import_module("wasmtime_attestation")))
__attribute__((import_name("attest_vm")))
int32_t __attestation_attest_vm(void);

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

/**
 * Helper to read JSON string from WASM linear memory.
 * Host writes: [len: 4 bytes little-endian][data: len bytes] at returned pointer.
 * In WASM32, the returned i32 IS the memory address (like Rust's `ptr as *const u8`).
 */
inline std::string read_json_from_host(int32_t ptr) {
    if (ptr == 0) {
        return "{}";
    }
    
    // In WASM32, ptr value IS the address in linear memory
    // This is exactly how Rust does it: `ptr as *const u8`
    const uint8_t* mem = reinterpret_cast<const uint8_t*>(ptr);
    
    // Read length (4 bytes, little-endian)
    uint32_t len = mem[0] | (mem[1] << 8) | (mem[2] << 16) | (mem[3] << 24);
    
    if (len == 0 || len > 100000) {
        return "{}";  // Sanity check
    }
    
    // Read data (starts at ptr + 4)
    return std::string(reinterpret_cast<const char*>(mem + 4), len);
}

/// Simple JSON string extractor
inline std::string json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;
    
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    if (pos >= json.size()) return "";
    
    if (json[pos] == '"') {
        pos++;
        size_t end = pos;
        while (end < json.size() && json[end] != '"') end++;
        return json.substr(pos, end - pos);
    } else {
        size_t end = json.find_first_of(",}\n", pos);
        if (end == std::string::npos) end = json.size();
        std::string val = json.substr(pos, end - pos);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
        return val;
    }
}

inline bool json_bool(const std::string& json, const std::string& key) {
    return json_string(json, key) == "true";
}

struct TeeInfo {
    std::string tee_type;
    bool supports_attestation;
};

struct AttestationResult {
    bool success;
    std::string token;
    std::string error;
};

/**
 * Detect TEE type by calling host function
 */
inline TeeInfo detect_tee_type() {
    TeeInfo info;
    
    int32_t ptr = __attestation_detect_tee();
    std::string json = read_json_from_host(ptr);
    
    info.tee_type = json_string(json, "tee_type");
    info.supports_attestation = json_bool(json, "supports_attestation");
    
    if (info.tee_type.empty()) {
        info.tee_type = "Unknown";
    }
    
    return info;
}

/**
 * Attest VM by calling host function
 */
inline AttestationResult attest_vm() {
    AttestationResult result;
    
    int32_t ptr = __attestation_attest_vm();
    std::string json = read_json_from_host(ptr);
    
    result.success = json_bool(json, "success");
    
    if (result.success) {
        result.token = json_string(json, "token");
        if (result.token.empty()) {
            result.token = json_string(json, "evidence");
        }
    } else {
        result.error = json_string(json, "error");
        if (result.error.empty()) {
            result.error = "Attestation failed";
        }
    }
    
    return result;
}

/**
 * Attest GPU by calling host function
 */
inline AttestationResult attest_gpu(uint32_t gpu_index = 0) {
    AttestationResult result;
    
    int32_t ptr = __attestation_attest_gpu(gpu_index);
    std::string json = read_json_from_host(ptr);
    
    result.success = json_bool(json, "success");
    
    if (result.success) {
        result.token = json_string(json, "token");
    } else {
        result.error = json_string(json, "error");
        if (result.error.empty()) {
            result.error = "GPU attestation failed";
        }
    }
    
    return result;
}

} // namespace attestation

#endif // ATTESTATION_HPP
