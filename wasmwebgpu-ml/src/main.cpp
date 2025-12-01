/**
 * @file main.cpp
 * @brief WASM ML Benchmark - Diabetes Prediction (C++ + wasi:gpu + TEE)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sstream>

#include "dataset.hpp"
#include "random_forest.hpp"
#include "gpu_executor.hpp"

// Model parameters
constexpr size_t N_ESTIMATORS = 200;
constexpr size_t MAX_DEPTH = 16;
constexpr size_t N_FEATURES = 10;
const std::string TRAIN_CSV = "data/diabetes_train.csv";
const std::string TEST_CSV = "data/diabetes_test.csv";

// ═══════════════════════════════════════════════════════════════════════════
// External declarations for wasmtime_attestation
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {
    __attribute__((import_module("wasmtime_attestation")))
    __attribute__((import_name("detect_tee")))
    int32_t wasm_detect_tee(void);

    __attribute__((import_module("wasmtime_attestation")))
    __attribute__((import_name("attest_vm")))
    int32_t wasm_attest_vm(void);

    __attribute__((import_module("wasmtime_attestation")))
    __attribute__((import_name("attest_gpu")))
    int32_t wasm_attest_gpu(uint32_t gpu_index);
}

// ═══════════════════════════════════════════════════════════════════════════
// Simple JSON field extraction without std::string allocation for large data
// Works directly on the memory buffer
// ═══════════════════════════════════════════════════════════════════════════

// Get length of JSON at pointer
uint32_t get_json_len(int32_t ptr) {
    if (ptr == 0 || ptr < 0) return 0;
    const unsigned char* p = (const unsigned char*)(uintptr_t)ptr;
    return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}

// Check if JSON contains "success":true
bool json_has_success(int32_t ptr) {
    if (ptr == 0 || ptr < 0) return false;
    const char* p = (const char*)(uintptr_t)ptr;
    uint32_t len = get_json_len(ptr);
    if (len == 0 || len > 100000) return false;
    
    // Search for "success":true in first 200 bytes
    const char* data = p + 4;
    size_t search_len = (len < 200) ? len : 200;
    for (size_t i = 0; i + 14 < search_len; i++) {
        if (data[i] == '"' && data[i+1] == 's' && data[i+2] == 'u' &&
            data[i+3] == 'c' && data[i+4] == 'c' && data[i+5] == 'e' &&
            data[i+6] == 's' && data[i+7] == 's' && data[i+8] == '"') {
            // Found "success", now look for :true
            for (size_t j = i + 9; j + 4 < search_len; j++) {
                if (data[j] == 't' && data[j+1] == 'r' && 
                    data[j+2] == 'u' && data[j+3] == 'e') {
                    return true;
                }
                if (data[j] == 'f' && data[j+1] == 'a' && 
                    data[j+2] == 'l' && data[j+3] == 's' && data[j+4] == 'e') {
                    return false;
                }
            }
        }
    }
    return false;
}

// Extract small string field (tee_type, etc.) - max 100 chars
std::string json_get_small(int32_t ptr, const char* key) {
    if (ptr == 0 || ptr < 0) return "";
    const char* p = (const char*)(uintptr_t)ptr;
    uint32_t len = get_json_len(ptr);
    if (len == 0 || len > 100000) return "";
    
    const char* data = p + 4;
    size_t keylen = 0;
    while (key[keylen]) keylen++;
    
    // Search for "key": in first 500 bytes
    size_t search_len = (len < 500) ? len : 500;
    for (size_t i = 0; i + keylen + 3 < search_len; i++) {
        if (data[i] == '"') {
            bool match = true;
            for (size_t k = 0; k < keylen && match; k++) {
                if (data[i + 1 + k] != key[k]) match = false;
            }
            if (match && data[i + 1 + keylen] == '"') {
                // Found key, look for value
                size_t j = i + 2 + keylen;
                while (j < search_len && data[j] != ':') j++;
                j++; // skip ':'
                while (j < search_len && (data[j] == ' ' || data[j] == '\t')) j++;
                
                if (data[j] == '"') {
                    j++; // skip opening quote
                    size_t start = j;
                    while (j < search_len && data[j] != '"' && j - start < 100) j++;
                    return std::string(data + start, j - start);
                } else {
                    // Non-string value
                    size_t start = j;
                    while (j < search_len && data[j] != ',' && data[j] != '}' && j - start < 50) j++;
                    return std::string(data + start, j - start);
                }
            }
        }
    }
    return "";
}

// Get token length by finding "token":"..." and counting
size_t json_get_token_len(int32_t ptr) {
    if (ptr == 0 || ptr < 0) return 0;
    const char* p = (const char*)(uintptr_t)ptr;
    uint32_t len = get_json_len(ptr);
    if (len == 0 || len > 100000) return 0;
    
    const char* data = p + 4;
    
    // Search for "token":" 
    for (size_t i = 0; i + 9 < len; i++) {
        if (data[i] == '"' && data[i+1] == 't' && data[i+2] == 'o' &&
            data[i+3] == 'k' && data[i+4] == 'e' && data[i+5] == 'n' &&
            data[i+6] == '"') {
            // Found "token", look for :"
            size_t j = i + 7;
            while (j < len && data[j] != '"') j++;
            if (j >= len) return 0;
            j++; // skip opening quote
            size_t start = j;
            while (j < len && data[j] != '"') j++;
            return j - start;
        }
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Results
// ═══════════════════════════════════════════════════════════════════════════

struct BenchmarkResults {
    std::string language = "cpp";
    std::string gpu_device, gpu_backend, tee_type;
    bool gpu_available = false, tee_available = false;
    double attestation_ms = 0, training_ms = 0, inference_ms = 0;
    float mse = 0;
    size_t train_samples = 0, test_samples = 0;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\"language\":\"" << language << "\",";
        ss << "\"gpu_device\":\"" << gpu_device << "\",";
        ss << "\"gpu_backend\":\"" << gpu_backend << "\",";
        ss << "\"tee_type\":\"" << tee_type << "\",";
        ss << "\"gpu_available\":" << (gpu_available ? "true" : "false") << ",";
        ss << "\"tee_available\":" << (tee_available ? "true" : "false") << ",";
        ss << "\"attestation_ms\":" << attestation_ms << ",";
        ss << "\"training_ms\":" << training_ms << ",";
        ss << "\"inference_ms\":" << inference_ms << ",";
        ss << std::setprecision(4) << "\"mse\":" << mse << ",";
        ss << "\"train_samples\":" << train_samples << ",";
        ss << "\"test_samples\":" << test_samples << "}";
        return ss.str();
    }
};

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start_).count();
    }
};

float calculate_mse(const std::vector<float>& pred, const std::vector<float>& actual) {
    float sum = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
        float d = pred[i] - actual[i];
        sum += d * d;
    }
    return sum / pred.size();
}

int main() {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    BenchmarkResults results;
    
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   WASM ML Benchmark - Diabetes Prediction                ║\n";
    std::cout << "║   C++ + wasi:gpu + TEE Attestation                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    // GPU Info
    std::cout << "\n=== GPU INFORMATION ===\n";
    {
        ml::GpuExecutor executor;
        if (executor.is_available()) {
            results.gpu_available = true;
            results.gpu_device = executor.device_name();
            results.gpu_backend = executor.backend();
            std::cout << "[GPU] Device: " << results.gpu_device << "\n";
            std::cout << "[GPU] Backend: " << results.gpu_backend << "\n";
            std::cout << "[GPU] Memory: " << (executor.device_info().total_memory / (1024*1024)) << " MB\n";
            std::cout << "[GPU] Hardware: " << (executor.is_hardware_gpu() ? "YES ✓" : "NO") << "\n";
        }
    }
    
    // TEE Attestation
    std::cout << "\n=== TEE ATTESTATION ===\n";
    Timer att_timer;
    
    // 1. Detect TEE
    {
        int32_t ptr = wasm_detect_tee();
        results.tee_type = json_get_small(ptr, "tee_type");
        std::string supports = json_get_small(ptr, "supports_attestation");
        results.tee_available = (supports == "true");
        std::cout << "[TEE] Type: " << results.tee_type << "\n";
        std::cout << "[TEE] Supports attestation: " << (results.tee_available ? "YES" : "NO") << "\n";
    }
    
    // 2. VM attestation
    {
        int32_t ptr = wasm_attest_vm();
        bool success = json_has_success(ptr);
        size_t token_len = success ? json_get_token_len(ptr) : 0;
        std::cout << "[TEE] VM attestation: " << (success ? "OK" : "FAILED") 
                  << " (token: " << token_len << " chars)\n";
    }
    
    // 3. GPU attestation
    {
        int32_t ptr = wasm_attest_gpu(0);
        bool success = json_has_success(ptr);
        size_t token_len = success ? json_get_token_len(ptr) : 0;
        std::cout << "[TEE] GPU attestation: " << (success ? "OK" : "FAILED")
                  << " (token: " << token_len << " chars)\n";
    }
    
    results.attestation_ms = att_timer.elapsed_ms();
    std::cout << "[TIMING] Attestation: " << std::fixed << std::setprecision(2) 
              << results.attestation_ms << " ms\n";
    
    // Training
    std::cout << "\n=== TRAINING ===\n";
    auto train_ds = ml::Dataset::from_csv(TRAIN_CSV, N_FEATURES);
    results.train_samples = train_ds.size();
    
    std::cout << "[TRAIN] Dataset: " << train_ds.size() << " samples, " 
              << train_ds.n_features() << " features\n";
    std::cout << "[TRAIN] Model: RandomForest (" << N_ESTIMATORS 
              << " trees, depth " << MAX_DEPTH << ")\n";
    
    ml::RandomForest rf(N_ESTIMATORS, MAX_DEPTH);
    ml::GpuTrainer gpu_trainer;
    
    Timer train_timer;
    if (gpu_trainer.is_available()) {
        std::cout << "[TRAIN] Accelerator: GPU\n";
        
        std::vector<float> data, labels;
        for (size_t i = 0; i < train_ds.size(); ++i) {
            const float* s = train_ds.get_sample(i);
            data.insert(data.end(), s, s + train_ds.n_features());
            labels.push_back(train_ds.get_label(i));
        }
        gpu_trainer.upload_training_data(data, labels, train_ds.size(), train_ds.n_features());
        
        rf.train_with_gpu(train_ds, gpu_trainer, [](size_t done, size_t total) {
            if (done % 10 == 0) std::cerr << "Trained " << done << "/" << total << " trees (GPU)\n";
        });
        gpu_trainer.cleanup();
    } else {
        std::cout << "[TRAIN] Accelerator: CPU\n";
        rf.train_cpu(train_ds);
    }
    results.training_ms = train_timer.elapsed_ms();
    std::cout << "[TIMING] Training: " << std::fixed << std::setprecision(2) 
              << results.training_ms << " ms\n";
    
    // Inference
    std::cout << "\n=== INFERENCE ===\n";
    auto test_ds = ml::Dataset::from_csv(TEST_CSV, N_FEATURES);
    results.test_samples = test_ds.size();
    std::cout << "[INFER] Test set: " << test_ds.size() << " samples\n";
    
    std::vector<float> test_data, test_labels;
    for (size_t i = 0; i < test_ds.size(); ++i) {
        const float* s = test_ds.get_sample(i);
        test_data.insert(test_data.end(), s, s + test_ds.n_features());
        test_labels.push_back(test_ds.get_label(i));
    }
    
    ml::GpuPredictor predictor;
    Timer infer_timer;
    std::vector<float> preds;
    if (predictor.is_available()) {
        std::cout << "[INFER] Accelerator: GPU\n";
        preds = rf.predict_with_gpu(test_data, test_ds.size(), test_ds.n_features(), predictor);
    } else {
        std::cout << "[INFER] Accelerator: CPU\n";
        preds = rf.predict_cpu(test_data, test_ds.size(), test_ds.n_features());
    }
    results.inference_ms = infer_timer.elapsed_ms();
    results.mse = calculate_mse(preds, test_labels);
    
    std::cout << "[TIMING] Inference: " << std::fixed << std::setprecision(2) 
              << results.inference_ms << " ms\n";
    std::cout << "[INFER] MSE: " << std::fixed << std::setprecision(4) << results.mse << "\n";
    
    // Summary
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    BENCHMARK RESULTS                      ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Language:       C++                                      ║\n";
    std::cout << "║  Attestation:  " << std::setw(10) << results.attestation_ms << " ms                           ║\n";
    std::cout << "║  Training:     " << std::setw(10) << results.training_ms << " ms                           ║\n";
    std::cout << "║  Inference:    " << std::setw(10) << results.inference_ms << " ms                           ║\n";
    std::cout << "║  MSE:          " << std::setw(10) << std::setprecision(4) << results.mse << "                             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    std::cout << "\n### BENCHMARK_JSON ###\n" << results.to_json() << "\n### END_BENCHMARK_JSON ###\n";
    std::cout << "\n✅ Benchmark completed successfully!\n";
    
    return 0;
}
