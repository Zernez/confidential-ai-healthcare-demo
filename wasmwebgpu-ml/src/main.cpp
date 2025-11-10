/**
 * @file main.cpp
 * @brief WASM ML Benchmark - Diabetes Prediction (C++ + wasi:webgpu)
 * 
 * This program replicates the Python ML pipeline:
 * 1. Load training data from CSV
 * 2. Train RandomForest (200 trees, depth 16)
 * 3. Load test data from CSV
 * 4. Predict on test set
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
 * @brief Calculate Mean Squared Error
 */
float calculate_mse(const std::vector<float>& predictions, 
                   const std::vector<float>& actual) {
    if (predictions.size() != actual.size()) {
        fprintf(stderr, "Prediction and actual length mismatch\n");
        exit(1);
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    
    return sum / predictions.size();
}

/**
 * @brief Main entry point - matches main.py sequence
 */
int main(int argc, char** argv) {
    // Force unbuffered output for better WASM compatibility
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    std::cout << "[STARTUP] C++ WASM ML Benchmark starting..." << std::endl;
    std::cout << std::flush;

    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   WASM ML Benchmark - Diabetes Prediction     ║" << std::endl;
    std::cout << "║   C++ + wasi:webgpu implementation            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::flush;
    
    // ═══════════════════════════════════════════════════════════
    // TRAINING PHASE
    // ═══════════════════════════════════════════════════════════
    
    std::cout << "\n=== TRAINING PHASE ===\n" << std::endl;
    std::cout << std::flush;
    
    // Load training data
    std::cout << "[LOADING] Reading training data from: " << TRAIN_CSV << std::endl;
    std::cout << std::flush;
    
    auto train_dataset = ml::Dataset::from_csv(TRAIN_CSV, N_FEATURES);
    
    std::cout << "[TRAINING] Dataset loaded: " 
                << train_dataset.size() << " samples, "
                << train_dataset.n_features() << " features" << std::endl;
    std::cout << std::flush;
    
    // Create and train RandomForest
    std::cout << "[TRAINING] Creating RandomForest with " << N_ESTIMATORS 
                << " estimators, max_depth " << MAX_DEPTH << std::endl;
    std::cout << std::flush;
    
    ml::RandomForest rf(N_ESTIMATORS, MAX_DEPTH);
    
    // Try GPU training if available, otherwise CPU
    ml::GpuExecutor gpu;
    
    auto train_start = std::chrono::high_resolution_clock::now();
    
    if (gpu.is_available()) {
        std::cout << "[TRAINING] Starting GPU training..." << std::endl;
        std::cout << std::flush;
        rf.train_gpu(train_dataset, gpu);
    } else {
        std::cout << "[TRAINING] GPU not available, starting CPU training..." << std::endl;
        std::cout << "[TRAINING] (this may take 1-2 minutes)..." << std::endl;
        std::cout << std::flush;
        rf.train_cpu(train_dataset);
    }
    
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
    
    std::cout << "\n[TRAINING] Training completed!" << std::endl;
    std::cout << "[TRAINING] Training time: " << train_duration.count() << " ms" << std::endl;
    std::cout << "[TRAINING] Number of trees: " << rf.n_trees() << std::endl;
    std::cout << std::flush;
    
    // ═══════════════════════════════════════════════════════════
    // INFERENCE PHASE
    // ═══════════════════════════════════════════════════════════
    
    std::cout << "\n=== INFERENCE PHASE ===\n" << std::endl;
    std::cout << std::flush;
    
    // Load test data
    std::cout << "[LOADING] Reading test data from: " << TEST_CSV << std::endl;
    std::cout << std::flush;
    
    auto test_dataset = ml::Dataset::from_csv(TEST_CSV, N_FEATURES);
    
    std::cout << "[INFERENCE] Test dataset loaded: " 
                << test_dataset.size() << " samples" << std::endl;
    std::cout << std::flush;
    
    // Predict on test set
    std::cout << "[INFERENCE] Running predictions on " 
                << test_dataset.size() << " test samples..." << std::endl;
    std::cout << std::flush;
    
    auto infer_start = std::chrono::high_resolution_clock::now();
    
    // Use the trained model (no save/load needed for benchmarking)
    std::vector<float> predictions;
    
    if (gpu.is_available()) {
        std::cout << "[INFERENCE] Using GPU for prediction..." << std::endl;
        std::cout << std::flush;
        predictions = gpu.predict(rf, test_dataset.data(), N_FEATURES);
    } else {
        std::cout << "[INFERENCE] Using CPU for prediction..." << std::endl;
        std::cout << std::flush;
        predictions = rf.predict_cpu(test_dataset.data(), 
                                        test_dataset.size(), 
                                        N_FEATURES);
    }
    
    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start);
    
    // Calculate MSE
    float mse = calculate_mse(predictions, test_dataset.labels());
    
    // Print results - same format as Python
    std::cout << "\n[INFERENCE] Results:" << std::endl;
    std::cout << "[INFERENCE] Test samples: " << test_dataset.size() << std::endl;
    std::cout << "[INFERENCE] Inference time: " << infer_duration.count() << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[INFERENCE] Mean Squared Error: " << mse << std::endl;
    std::cout << std::flush;
    
    // ═══════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Training samples:  " << train_dataset.size() << std::endl;
    std::cout << "Test samples:      " << test_dataset.size() << std::endl;
    std::cout << "Features:          " << N_FEATURES << std::endl;
    std::cout << "Trees:             " << N_ESTIMATORS << std::endl;
    std::cout << "Max depth:         " << MAX_DEPTH << std::endl;
    std::cout << "Training time:     " << train_duration.count() << " ms" << std::endl;
    std::cout << "Inference time:    " << infer_duration.count() << " ms" << std::endl;
    std::cout << "Mean Squared Error: " << std::fixed << std::setprecision(4) << mse << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << std::flush;
    
    std::cout << "Benchmark completed successfully!" << std::endl;
    std::cout << std::flush;
    
    return 0;
}
