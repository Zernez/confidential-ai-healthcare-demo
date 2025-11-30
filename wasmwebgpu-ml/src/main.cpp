/**
 * @file main.cpp
 * @brief WASM ML Benchmark - Diabetes Prediction (C++ + wasi:gpu)
 * 
 * This program replicates the Python ML pipeline with GPU acceleration via wasi:gpu:
 * 
 * 0. [TEE] Perform attestation (if available)
 * 1. Load training data from CSV
 * 2. Train RandomForest (200 trees, depth 16) - GPU accelerated
 * 3. Load test data from CSV
 * 4. Predict on test set - GPU accelerated
 * 5. Calculate and print MSE
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

#include "dataset.hpp"
#include "random_forest.hpp"
#include "gpu_executor.hpp"

// Model parameters - MUST match Python configuration
constexpr size_t N_ESTIMATORS = 200;
constexpr size_t MAX_DEPTH = 16;
constexpr size_t N_FEATURES = 10;
const std::string TRAIN_CSV = "data/diabetes_train.csv";
const std::string TEST_CSV = "data/diabetes_test.csv";

/**
 * @brief Timer utility
 */
class Timer {
public:
    Timer(const std::string& label) : label_(label) {
        std::cout << "[TIMING] Starting: " << label_ << std::endl;
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::cout << "[TIMING] " << label_ << ": " << std::fixed 
                  << std::setprecision(2) << ms << " ms" << std::endl;
        return ms;
    }
    
private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Calculate Mean Squared Error
 */
float calculate_mse(const std::vector<float>& predictions, 
                   const std::vector<float>& actual) {
    if (predictions.size() != actual.size()) {
        std::cerr << "ERROR: Prediction and actual length mismatch" << std::endl;
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    
    return sum / predictions.size();
}

/**
 * @brief Print GPU information
 */
void print_gpu_info() {
    std::cout << "\n=== GPU INFORMATION ===\n" << std::endl;
    
    ml::GpuExecutor executor;
    if (executor.is_available()) {
        std::cout << "[GPU] Device: " << executor.device_name() << std::endl;
        std::cout << "[GPU] Backend: " << executor.backend() << std::endl;
        std::cout << "[GPU] Hardware GPU: " 
                  << (executor.is_hardware_gpu() ? "YES ✓" : "NO (software)") << std::endl;
    } else {
        std::cout << "[GPU] Not available - will use CPU" << std::endl;
    }
}

/**
 * @brief Main entry point
 */
int main(int argc, char** argv) {
    // Force unbuffered output for WASM
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   WASM ML Benchmark - Diabetes Prediction                ║" << std::endl;
    std::cout << "║   C++ + wasi:gpu (GPU-accelerated via host functions)    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    
    // Print GPU info
    print_gpu_info();
    
    double train_time = 0.0;
    double infer_time = 0.0;
    float mse = 0.0f;
    
    // ═══════════════════════════════════════════════════════════════════
    // TRAINING PHASE
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n=== TRAINING PHASE ===\n" << std::endl;
    
    // Load training data
    std::cout << "[LOADING] Reading training data from: " << TRAIN_CSV << std::endl;
    
    auto train_dataset = ml::Dataset::from_csv(TRAIN_CSV, N_FEATURES);
    
    std::cout << "[TRAINING] Dataset: " << train_dataset.size() << " samples, "
              << train_dataset.n_features() << " features" << std::endl;
    
    // Create RandomForest
    std::cout << "[TRAINING] Creating RandomForest: " << N_ESTIMATORS 
              << " trees, max_depth " << MAX_DEPTH << std::endl;
    
    ml::RandomForest rf(N_ESTIMATORS, MAX_DEPTH);
    
    // Initialize GPU trainer
    ml::GpuTrainer gpu_trainer;
    
    if (gpu_trainer.is_available()) {
        std::cout << "[TRAINING] Uploading data to GPU..." << std::endl;
        
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
        
        // Train with GPU
        Timer train_timer("GPU training");
        rf.train_with_gpu(train_dataset, gpu_trainer);
        train_time = train_timer.stop();
        
        // Cleanup GPU resources
        gpu_trainer.cleanup();
    } else {
        std::cout << "[TRAINING] Using CPU (GPU not available)..." << std::endl;
        
        Timer train_timer("CPU training");
        rf.train_cpu(train_dataset);
        train_time = train_timer.stop();
    }
    
    std::cout << "[TRAINING] Training completed!" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // INFERENCE PHASE
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n=== INFERENCE PHASE ===\n" << std::endl;
    
    // Load test data
    std::cout << "[LOADING] Reading test data from: " << TEST_CSV << std::endl;
    
    auto test_dataset = ml::Dataset::from_csv(TEST_CSV, N_FEATURES);
    
    std::cout << "[INFERENCE] Test set: " << test_dataset.size() << " samples" << std::endl;
    
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
    
    if (gpu_predictor.is_available()) {
        Timer infer_timer("GPU inference");
        predictions = rf.predict_with_gpu(test_data, test_dataset.size(), 
                                           test_dataset.n_features(), gpu_predictor);
        infer_time = infer_timer.stop();
    } else {
        Timer infer_timer("CPU inference");
        predictions = rf.predict_cpu(test_data, test_dataset.size(), 
                                      test_dataset.n_features());
        infer_time = infer_timer.stop();
    }
    
    // Calculate MSE
    mse = calculate_mse(predictions, test_labels);
    
    std::cout << "[INFERENCE] Samples: " << test_dataset.size() << std::endl;
    std::cout << "[INFERENCE] Mean Squared Error: " << std::fixed 
              << std::setprecision(4) << mse << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    BENCHMARK SUMMARY                      ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Training time:  " << std::setw(10) << std::fixed 
              << std::setprecision(2) << train_time << " ms                          ║" << std::endl;
    std::cout << "║  Inference time: " << std::setw(10) << std::fixed 
              << std::setprecision(2) << infer_time << " ms                          ║" << std::endl;
    std::cout << "║  MSE:            " << std::setw(10) << std::fixed 
              << std::setprecision(4) << mse << "                             ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\n✅ Benchmark completed successfully!" << std::endl;
    
    return 0;
}
