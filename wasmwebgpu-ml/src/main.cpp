/**
 * @file main.cpp
 * @brief WASM ML Benchmark - Diabetes Prediction (C++ + wasi:gpu + TEE)
 * 
 * Unified benchmark with:
 * - TEE attestation (AMD SEV-SNP / Intel TDX)
 * - GPU acceleration via wasi:gpu
 * - Structured JSON output for benchmark aggregation
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
#include "attestation.hpp"

// Model parameters - MUST match Python/Rust configuration
constexpr size_t N_ESTIMATORS = 200;
constexpr size_t MAX_DEPTH = 16;
constexpr size_t N_FEATURES = 10;
const std::string TRAIN_CSV = "data/diabetes_train.csv";
const std::string TEST_CSV = "data/diabetes_test.csv";

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Results Structure
// ═══════════════════════════════════════════════════════════════════════════

struct BenchmarkResults {
    std::string language = "cpp";
    std::string gpu_device;
    std::string gpu_backend;
    std::string tee_type;
    bool gpu_available = false;
    bool tee_available = false;
    double attestation_ms = 0.0;
    double training_ms = 0.0;
    double inference_ms = 0.0;
    float mse = 0.0f;
    size_t train_samples = 0;
    size_t test_samples = 0;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{";
        ss << "\"language\":\"" << language << "\",";
        ss << "\"gpu_device\":\"" << gpu_device << "\",";
        ss << "\"gpu_backend\":\"" << gpu_backend << "\",";
        ss << "\"tee_type\":\"" << tee_type << "\",";
        ss << "\"gpu_available\":" << (gpu_available ? "true" : "false") << ",";
        ss << "\"tee_available\":" << (tee_available ? "true" : "false") << ",";
        ss << "\"attestation_ms\":" << attestation_ms << ",";
        ss << "\"training_ms\":" << training_ms << ",";
        ss << "\"inference_ms\":" << inference_ms << ",";
        ss << std::setprecision(4);
        ss << "\"mse\":" << mse << ",";
        ss << "\"train_samples\":" << train_samples << ",";
        ss << "\"test_samples\":" << test_samples;
        ss << "}";
        return ss.str();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Timer Utility
// ═══════════════════════════════════════════════════════════════════════════

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ═══════════════════════════════════════════════════════════════════════════
// Calculate Mean Squared Error
// ═══════════════════════════════════════════════════════════════════════════

float calculate_mse(const std::vector<float>& predictions, 
                   const std::vector<float>& actual) {
    if (predictions.size() != actual.size()) {
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    
    return sum / predictions.size();
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Entry Point
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    // Force unbuffered output for WASM
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    BenchmarkResults results;
    
    // ═══════════════════════════════════════════════════════════════════
    // HEADER
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   WASM ML Benchmark - Diabetes Prediction                ║" << std::endl;
    std::cout << "║   C++ + wasi:gpu + TEE Attestation                       ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // GPU INFORMATION
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n=== GPU INFORMATION ===" << std::endl;
    
    {
        ml::GpuExecutor executor;
        if (executor.is_available()) {
            results.gpu_available = true;
            results.gpu_device = executor.device_name();
            results.gpu_backend = executor.backend();
            
            std::cout << "[GPU] Device: " << results.gpu_device << std::endl;
            std::cout << "[GPU] Backend: " << results.gpu_backend << std::endl;
            std::cout << "[GPU] Memory: " << (executor.device_info().total_memory / (1024*1024)) << " MB" << std::endl;
            std::cout << "[GPU] Hardware: " << (executor.is_hardware_gpu() ? "YES ✓" : "NO (software)") << std::endl;
        } else {
            std::cout << "[GPU] Not available - will use CPU" << std::endl;
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // TEE ATTESTATION
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n=== TEE ATTESTATION ===" << std::endl;
    
    Timer attestation_timer;
    
    // Detect TEE type
    auto tee_info = attestation::detect_tee_type();
    results.tee_type = tee_info.tee_type;
    results.tee_available = tee_info.supports_attestation;
    
    std::cout << "[TEE] Type: " << tee_info.tee_type << std::endl;
    std::cout << "[TEE] Supports attestation: " << (tee_info.supports_attestation ? "YES" : "NO") << std::endl;
    
    // Attest VM
    auto vm_result = attestation::attest_vm();
    if (vm_result.success) {
        std::cout << "[TEE] VM attestation: OK (token: " << vm_result.token.length() << " chars)" << std::endl;
    } else {
        std::cout << "[TEE] VM attestation: FAILED (" << vm_result.error << ")" << std::endl;
    }
    
    // Attest GPU
    auto gpu_attest_result = attestation::attest_gpu(0);
    if (gpu_attest_result.success) {
        std::cout << "[TEE] GPU attestation: OK (token: " << gpu_attest_result.token.length() << " chars)" << std::endl;
    } else {
        std::cout << "[TEE] GPU attestation: FAILED (" << gpu_attest_result.error << ")" << std::endl;
    }
    
    results.attestation_ms = attestation_timer.elapsed_ms();
    std::cout << "[TIMING] Attestation: " << std::fixed << std::setprecision(2) 
              << results.attestation_ms << " ms" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // TRAINING PHASE
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n=== TRAINING ===" << std::endl;
    
    // Load training data
    auto train_dataset = ml::Dataset::from_csv(TRAIN_CSV, N_FEATURES);
    results.train_samples = train_dataset.size();
    
    std::cout << "[TRAIN] Dataset: " << train_dataset.size() << " samples, "
              << train_dataset.n_features() << " features" << std::endl;
    std::cout << "[TRAIN] Model: RandomForest (" << N_ESTIMATORS 
              << " trees, depth " << MAX_DEPTH << ")" << std::endl;
    
    // Create RandomForest
    ml::RandomForest rf(N_ESTIMATORS, MAX_DEPTH);
    
    // Initialize GPU trainer
    ml::GpuTrainer gpu_trainer;
    
    Timer train_timer;
    
    if (gpu_trainer.is_available()) {
        std::cout << "[TRAIN] Accelerator: GPU" << std::endl;
        
        // Get data vectors for upload
        std::vector<float> all_data;
        std::vector<float> all_labels;
        all_data.reserve(train_dataset.size() * train_dataset.n_features());
        all_labels.reserve(train_dataset.size());
        
        for (size_t i = 0; i < train_dataset.size(); ++i) {
            const float* sample = train_dataset.get_sample(i);
            all_data.insert(all_data.end(), sample, sample + train_dataset.n_features());
            all_labels.push_back(train_dataset.get_label(i));
        }
        
        gpu_trainer.upload_training_data(all_data, all_labels, 
                                          train_dataset.size(), 
                                          train_dataset.n_features());
        
        // Train with GPU (with progress output matching Rust)
        rf.train_with_gpu(train_dataset, gpu_trainer, [](size_t trained, size_t total) {
            if (trained % 10 == 0) {
                std::cerr << "Trained " << trained << "/" << total << " trees (GPU)" << std::endl;
            }
        });
        
        // Cleanup GPU resources
        gpu_trainer.cleanup();
    } else {
        std::cout << "[TRAIN] Accelerator: CPU" << std::endl;
        rf.train_cpu(train_dataset);
    }
    
    results.training_ms = train_timer.elapsed_ms();
    std::cout << "[TIMING] Training: " << std::fixed << std::setprecision(2) 
              << results.training_ms << " ms" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // INFERENCE PHASE
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n=== INFERENCE ===" << std::endl;
    
    // Load test data
    auto test_dataset = ml::Dataset::from_csv(TEST_CSV, N_FEATURES);
    results.test_samples = test_dataset.size();
    
    std::cout << "[INFER] Test set: " << test_dataset.size() << " samples" << std::endl;
    
    // Prepare test data vectors
    std::vector<float> test_data;
    std::vector<float> test_labels;
    test_data.reserve(test_dataset.size() * test_dataset.n_features());
    test_labels.reserve(test_dataset.size());
    
    for (size_t i = 0; i < test_dataset.size(); ++i) {
        const float* sample = test_dataset.get_sample(i);
        test_data.insert(test_data.end(), sample, sample + test_dataset.n_features());
        test_labels.push_back(test_dataset.get_label(i));
    }
    
    // Predict
    ml::GpuPredictor gpu_predictor;
    std::vector<float> predictions;
    
    Timer infer_timer;
    
    if (gpu_predictor.is_available()) {
        std::cout << "[INFER] Accelerator: GPU" << std::endl;
        predictions = rf.predict_with_gpu(test_data, test_dataset.size(), 
                                           test_dataset.n_features(), gpu_predictor);
    } else {
        std::cout << "[INFER] Accelerator: CPU" << std::endl;
        predictions = rf.predict_cpu(test_data, test_dataset.size(), 
                                      test_dataset.n_features());
    }
    
    results.inference_ms = infer_timer.elapsed_ms();
    std::cout << "[TIMING] Inference: " << std::fixed << std::setprecision(2) 
              << results.inference_ms << " ms" << std::endl;
    
    // Calculate MSE
    results.mse = calculate_mse(predictions, test_labels);
    std::cout << "[INFER] MSE: " << std::fixed << std::setprecision(4) << results.mse << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK SUMMARY
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    BENCHMARK RESULTS                      ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Language:       C++                                      ║" << std::endl;
    std::cout << "║  Attestation:  " << std::setw(10) << std::fixed << std::setprecision(2) 
              << results.attestation_ms << " ms                           ║" << std::endl;
    std::cout << "║  Training:     " << std::setw(10) << std::fixed << std::setprecision(2) 
              << results.training_ms << " ms                           ║" << std::endl;
    std::cout << "║  Inference:    " << std::setw(10) << std::fixed << std::setprecision(2) 
              << results.inference_ms << " ms                           ║" << std::endl;
    std::cout << "║  MSE:          " << std::setw(10) << std::fixed << std::setprecision(4) 
              << results.mse << "                             ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // JSON OUTPUT FOR BENCHMARK AGGREGATION
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n### BENCHMARK_JSON ###" << std::endl;
    std::cout << results.to_json() << std::endl;
    std::cout << "### END_BENCHMARK_JSON ###" << std::endl;
    
    std::cout << "\n✅ Benchmark completed successfully!" << std::endl;
    
    return 0;
}
