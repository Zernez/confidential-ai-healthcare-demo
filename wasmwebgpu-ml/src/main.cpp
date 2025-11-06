/**
 * @file main.cpp
 * @brief WASM ML Benchmark - Diabetes Prediction (C++ + wasi:webgpu)
 * 
 * This program replicates the Python ML pipeline:
 * 1. Load training data from CSV
 * 2. Train RandomForest (200 trees, depth 16)
 * 3. Save model
 * 4. Load test data from CSV
 * 5. Load model
 * 6. Predict on test set
 * 7. Calculate and print MSE
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
const std::string MODEL_PATH = "data/model_diabetes_cpp.json";
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
 * @brief Training phase - matches train_model.py
 */
void train_and_save() {
    std::cout << "\n=== TRAINING PHASE ===\n" << std::endl;
    std::cout << std::flush;
    
    // Load training data
    std::cout << "[DEBUG] About to load training data from: " << TRAIN_CSV << std::endl;
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
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (gpu.is_available()) {
        std::cout << "[TRAINING] Starting GPU training..." << std::endl;
        std::cout << std::flush;
        rf.train_gpu(train_dataset, gpu);
    } else {
        std::cout << "[TRAINING] GPU not available, starting CPU training..." << std::endl;
        std::cout << "[TRAINING] (this may take a while)..." << std::endl;
        std::cout << std::flush;
        rf.train_cpu(train_dataset);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "[TRAINING] Training completed in " 
              << duration.count() << " ms!" << std::endl;
    std::cout << std::flush;
    
    // Serialize and save model
    std::string model_json = rf.to_json();
    
    std::ofstream out(MODEL_PATH);
    if (!out) {
        fprintf(stderr, "Cannot write model file: %s\n", MODEL_PATH.c_str());
        exit(1);
    }
    out << model_json;
    out.close();
    
    std::cout << "[TRAINING] Model saved to: " << MODEL_PATH << std::endl;
    std::cout << "[TRAINING] Model size: " << model_json.size() << " bytes" << std::endl;
    std::cout << std::flush;
}

/**
 * @brief Inference phase - matches infer_model.py
 */
void load_and_infer() {
    std::cout << "\n=== INFERENCE PHASE ===\n" << std::endl;
    std::cout << std::flush;
    
    // Load test data
    std::cout << "[DEBUG] About to load test data from: " << TEST_CSV << std::endl;
    std::cout << std::flush;
    
    auto test_dataset = ml::Dataset::from_csv(TEST_CSV, N_FEATURES);
    
    std::cout << "[INFERENCE] Test dataset loaded: " 
              << test_dataset.size() << " samples" << std::endl;
    std::cout << std::flush;
    
    // Load model
    std::cout << "[INFERENCE] Loading model from: " << MODEL_PATH << std::endl;
    std::cout << std::flush;
    
    std::ifstream in(MODEL_PATH);
    if (!in) {
        fprintf(stderr, "Cannot read model file: %s\n", MODEL_PATH.c_str());
        exit(1);
    }
    std::string model_json((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    in.close();
    
    ml::RandomForest rf = ml::RandomForest::from_json(model_json);
    
    std::cout << "[INFERENCE] Model loaded successfully" << std::endl;
    std::cout << "[INFERENCE] Number of trees: " << rf.n_trees() << std::endl;
    std::cout << std::flush;
    
    // Predict on test set
    std::cout << "[INFERENCE] Running predictions on " 
              << test_dataset.size() << " test samples..." << std::endl;
    std::cout << std::flush;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Try GPU prediction if available, otherwise CPU
    ml::GpuExecutor gpu;
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
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Calculate MSE
    float mse = calculate_mse(predictions, test_dataset.labels());
    
    // Print results - same format as Python
    std::cout << "\n[INFERENCE] Results:" << std::endl;
    std::cout << "[INFERENCE] Samples: " << test_dataset.size() << std::endl;
    std::cout << "[INFERENCE] Inference time: " << duration.count() << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[INFERENCE] Mean Squared Error: " << mse << std::endl;
    std::cout << std::flush;
}

/**
 * @brief Main entry point - matches main.py sequence
 */
int main(int argc, char** argv) {
    // Force unbuffered output for better WASM compatibility
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    std::cout << "[STARTUP] C++ WASM ML Benchmark starting..." << std::endl;
    std::cout << "[STARTUP] Checking file paths..." << std::endl;
    std::cout << std::flush;
    
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   WASM ML Benchmark - Diabetes Prediction     ║" << std::endl;
    std::cout << "║   C++ + wasi:webgpu implementation            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::flush;

    // Step 1: Training (matches MLTrainer.train_and_split())
    train_and_save();

    // Step 2: Inference (matches MLInferencer.run_inference())
    load_and_infer();

    std::cout << "\n✅ Benchmark completed successfully!" << std::endl;
    std::cout << std::flush;

    return 0;
}
