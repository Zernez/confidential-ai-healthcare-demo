/**
 * @file attestation.hpp
 * @brief C++ bindings for wasmtime:attestation runtime extension
 * 
 * Provides C++ wrappers around the wasmtime:attestation host functions
 * for TEE (TDX/SEV-SNP) and GPU (NVIDIA NRAS) attestation.
 * 
 * Usage:
 *   #include "attestation.hpp"
 *   
 *   // Attest VM
 *   auto vm_result = wasmtime_attestation::attest_vm();
 *   if (!vm_result.success) {
 *       fprintf(stderr, "VM attestation failed: %s\n", vm_result.error.c_str());
 *       return 1;
 *   }
 *   
 *   // Attest GPU
 *   auto gpu_result = wasmtime_attestation::attest_gpu(0);
 *   if (!gpu_result.success) {
 *       fprintf(stderr, "GPU attestation failed: %s\n", gpu_result.error.c_str());
 *       return 1;
 *   }
 *   
 *   // Verify tokens
 *   if (!wasmtime_attestation::verify_token(vm_result.token.value())) {
 *       fprintf(stderr, "VM token verification failed!\n");
 *       return 1;
 *   }
 */

#pragma once

#include <string>
#include <optional>
#include <cstdint>
#include "external/json.hpp"

namespace wasmtime_attestation {

using json = nlohmann::json;

/**
 * @brief Attestation result from host
 */
struct AttestationResult {
    bool success;
    std::optional<std::string> token;
    std::optional<std::string> evidence;
    std::optional<std::string> error;
    int64_t timestamp;
    
    /**
     * @brief Parse from JSON
     */
    static AttestationResult from_json(const std::string& json_str) {
        AttestationResult result;
        try {
            auto j = json::parse(json_str);
            result.success = j["success"].get<bool>();
            result.timestamp = j["timestamp"].get<int64_t>();
            
            if (j.contains("token") && !j["token"].is_null()) {
                result.token = j["token"].get<std::string>();
            }
            if (j.contains("evidence") && !j["evidence"].is_null()) {
                result.evidence = j["evidence"].get<std::string>();
            }
            if (j.contains("error") && !j["error"].is_null()) {
                result.error = j["error"].get<std::string>();
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.error = std::string("JSON parse error: ") + e.what();
            result.timestamp = 0;
        }
        return result;
    }
};

// External C declarations for host functions
extern "C" {
    /**
     * @brief Attest the VM (TDX/SEV-SNP)
     * @return Pointer to JSON AttestationResult in host memory
     */
    int32_t __attribute__((
        __import_module__("wasmtime_attestation"),
        __import_name__("attest_vm")
    )) wasmtime_attest_vm();
    
    /**
     * @brief Attest the GPU (NVIDIA NRAS)
     * @param gpu_index GPU device index (0-based)
     * @return Pointer to JSON AttestationResult in host memory
     */
    int32_t __attribute__((
        __import_module__("wasmtime_attestation"),
        __import_name__("attest_gpu")
    )) wasmtime_attest_gpu(uint32_t gpu_index);
    
    /**
     * @brief Verify a JWT token
     * @param token_ptr Pointer to token string in WASM memory
     * @param token_len Length of token string
     * @return 1 if valid, 0 if invalid
     */
    int32_t __attribute__((
        __import_module__("wasmtime_attestation"),
        __import_name__("verify_token")
    )) wasmtime_verify_token(const char* token_ptr, int32_t token_len);
    
    /**
     * @brief Clear cached attestation tokens
     */
    void __attribute__((
        __import_module__("wasmtime_attestation"),
        __import_name__("clear_cache")
    )) wasmtime_clear_cache();
}

/**
 * @brief Read JSON string from host memory
 * @param ptr Pointer returned by host function
 * @return JSON string
 */
inline std::string read_json_from_host(int32_t ptr) {
    if (ptr == 0) {
        return R"({"success":false,"error":"Host returned null pointer"})";
    }
    
    // Read: [len: 4 bytes][data: len bytes]
    const uint8_t* base = reinterpret_cast<const uint8_t*>(ptr);
    uint32_t len = *reinterpret_cast<const uint32_t*>(base);
    const char* data = reinterpret_cast<const char*>(base + 4);
    
    return std::string(data, len);
}

/**
 * @brief Attest VM and return result
 * @return AttestationResult
 */
inline AttestationResult attest_vm() {
    int32_t ptr = wasmtime_attest_vm();
    std::string json_str = read_json_from_host(ptr);
    return AttestationResult::from_json(json_str);
}

/**
 * @brief Attest GPU and return result
 * @param gpu_index GPU device index (typically 0)
 * @return AttestationResult
 */
inline AttestationResult attest_gpu(uint32_t gpu_index) {
    int32_t ptr = wasmtime_attest_gpu(gpu_index);
    std::string json_str = read_json_from_host(ptr);
    return AttestationResult::from_json(json_str);
}

/**
 * @brief Verify a JWT token
 * @param token JWT token string
 * @return true if valid, false otherwise
 */
inline bool verify_token(const std::string& token) {
    int32_t result = wasmtime_verify_token(token.c_str(), token.length());
    return result == 1;
}

/**
 * @brief Clear all cached attestation tokens
 */
inline void clear_cache() {
    wasmtime_clear_cache();
}

/**
 * @brief Helper: Run full attestation workflow (VM + GPU)
 * @param gpu_index GPU device index
 * @return true if all attestations passed, false otherwise
 */
inline bool attest_all(uint32_t gpu_index = 0) {
    // Attest VM
    printf("🔐 Attesting VM (TDX/SEV-SNP)...\n");
    auto vm_result = attest_vm();
    if (!vm_result.success) {
        fprintf(stderr, "❌ VM attestation failed: %s\n", 
                vm_result.error.value_or("Unknown error").c_str());
        return false;
    }
    printf("✓ VM attestation successful\n");
    
    // Attest GPU
    printf("🔐 Attesting GPU (device %u)...\n", gpu_index);
    auto gpu_result = attest_gpu(gpu_index);
    if (!gpu_result.success) {
        fprintf(stderr, "❌ GPU attestation failed: %s\n",
                gpu_result.error.value_or("Unknown error").c_str());
        return false;
    }
    printf("✓ GPU attestation successful\n");
    
    // Verify VM token
    printf("🔍 Verifying VM token...\n");
    if (!vm_result.token.has_value() || !verify_token(vm_result.token.value())) {
        fprintf(stderr, "❌ VM token verification failed\n");
        return false;
    }
    printf("✓ VM token verified\n");
    
    // Verify GPU token
    printf("🔍 Verifying GPU token...\n");
    if (!gpu_result.token.has_value() || !verify_token(gpu_result.token.value())) {
        fprintf(stderr, "❌ GPU token verification failed\n");
        return false;
    }
    printf("✓ GPU token verified\n");
    
    printf("\n✅ All attestations passed!\n\n");
    return true;
}

} // namespace wasmtime_attestation
