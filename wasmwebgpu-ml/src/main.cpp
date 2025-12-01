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
// Read JSON from pointer - MUST be called immediately after host function
// because host reuses the same memory location (offset 1024)
// ═══════════════════════════════════════════════════════════════════════════

struct HostJson {
    bool valid;
    std::string data;
    
    std::string get(const std::string& key) const {
        if (!valid) return "";
        size_t pos = data.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = data.find(':', pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < data.size() && data[pos] == ' ') pos++;
        if (pos >= data.size()) return "";
        if (data[pos] == '"') {
            pos++;
            size_t end = json_find_string_end(pos);
            return data.substr(pos, end - pos);
        }
        size_t end = data.find_first_of(",}", pos);
        std::string val = data.substr(pos, end - pos);
        while (!val.empty() && val.back() == ' ') val.pop_back();
        return val;
    }
    
    bool getBool(const std::string& key) const {
        return get(key) == "true";
    }
    
private:
    size_t json_find_string_end(size_t start) const {
        for (size_t i = start; i < data.size(); i++) {
            if (data[i] == '"' && (i == 0 || data[i-1] != '\\')) {
                return i;
            }
        }
        return data.size();
    }
};

HostJson read_host_result(int32_t ptr) {
    HostJson result;
    result.valid = false;
    
    if (ptr <= 0 || ptr > 0x100000) {
        return result;
    }
    
    // Read directly from WASM linear memory at offset ptr
    // Host writes: [len: 4 bytes LE][data: len bytes]
    const uint8_t* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptr));
    
    uint32_t len = static_cast<uint32_t>(p[0]) | 
                   (static_cast<uint32_t>(p[1]) << 8) | 
                   (static_cast<uint32_t>(p[2]) << 16) | 
                   (static_cast<uint32_t>(p[3]) << 24);
    
    // Sanity check - don't read garbage
    if (len == 0 || len > 100000) {
        return result;
    }
    
    result.data = std::string(reinterpret_cast<const char*>(p + 4), len);
    result.valid = true;
    return result;
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
    
    // TEE Attestation - read result IMMEDIATELY after each call
    std::cout << "\n=== TEE ATTESTATION ===\n";
    Timer att_timer;
    
    size_t vm_token_len = 0;
    size_t gpu_token_len = 0;
    
    // 1. Detect TEE - call and read immediately
    {
        int32_t ptr = wasm_detect_tee();
        HostJson json = read_host_result(ptr);  // Read NOW before next call
        if (json.valid) {
            results.tee_type = json.get("tee_type");
            results.tee_available = json.getBool("supports_attestation");
        } else {
            results.tee_type = "Unknown";
            results.tee_available = false;
        }
    }
    std::cout << "[TEE] Type: " << results.tee_type << "\n";
    std::cout << "[TEE] Supports attestation: " << (results.tee_available ? "YES" : "NO") << "\n";
    
    // 2. Attest VM - call and read immediately
    {
        int32_t ptr = wasm_attest_vm();
        HostJson json = read_host_result(ptr);  // Read NOW before next call
        if (json.valid && json.getBool("success")) {
            vm_token_len = json.get("token").length();
            std::cout << "[TEE] VM attestation: OK (token: " << vm_token_len << " chars)\n";
        } else {
            std::string err = json.valid ? json.get("error") : "read failed";
            std::cout << "[TEE] VM attestation: FAILED (" << err << ")\n";
        }
    }
    
    // 3. Attest GPU - call and read immediately
    {
        int32_t ptr = wasm_attest_gpu(0);
        HostJson json = read_host_result(ptr);  // Read NOW before next call
        if (json.valid && json.getBool("success")) {
            gpu_token_len = json.get("token").length();
            std::cout << "[TEE] GPU attestation: OK (token: " << gpu_token_len << " chars)\n";
        } else {
            std::string err = json.valid ? json.get("error") : "read failed";
            std::cout << "[TEE] GPU attestation: FAILED (" << err << ")\n";
        }
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
