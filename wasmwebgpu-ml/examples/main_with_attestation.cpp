/**
 * @file main_with_attestation.cpp
 * @brief WASM ML Benchmark with TEE Attestation - Diabetes Prediction
 * 
 * This program demonstrates confidential ML with runtime attestation:
 * 1. Attest VM (TDX/SEV-SNP)
 * 2. Attest GPU (NVIDIA H100)
 * 3. Verify attestation tokens
 * 4. Only if attestation passes → proceed with ML
 * 5. Load training data from CSV
 * 6. Train RandomForest (200 trees, depth 16)
 * 7. Load test data from CSV
 * 8. Predict on test set
 * 9. Calculate and print MSE
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
#include "attestation.hpp"  // NEW: TEE attestation

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
 * @brief Main entry point with attestation
 */
int main() {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Confidential ML with TEE Attestation         ║\n");
    printf("║  WASM + C++ + wasmtime:attestation            ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");
    
    // ═══════════════════════════════════════════════════════
    // PHASE 1: ATTESTATION (Critical Security Step)
    // ═══════════════════════════════════════════════════════
    
    printf("━━━ Phase 1: TEE Attestation ━━━\n\n");
    
    // Run full attestation workflow (VM + GPU)
    if (!wasmtime_attestation::attest_all(0)) {
        fprintf(stderr, "\n❌ Attestation failed! Aborting ML execution.\n");
        return 1;
    }
    
    // ═══════════════════════════════════════════════════════
    // PHASE 2: ML TRAINING (Only executed if attestation passed)
    // ═══════════════════════════════════════════════════════
    
    printf("━━━ Phase 2: ML Training ━━━\n\n");
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load training data
    printf("[LOADING] Reading training CSV: %s\n", TRAIN_CSV.c_str());
    Dataset train_dataset;
    try {
        train_dataset = Dataset::load_csv(TRAIN_CSV, N_FEATURES);
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to load training data: %s\n", e.what());
        return 1;
    }
    printf("[LOADING] Loaded %zu training samples with %zu features\n", 
           train_dataset.n_samples(), N_FEATURES);
    
    // Create and train RandomForest
    printf("\n[TRAINING] Creating RandomForest with %zu estimators, max_depth %zu\n",
           N_ESTIMATORS, MAX_DEPTH);
    printf("[TRAINING] Starting training (this may take a while)...\n");
    
    RandomForest rf(N_ESTIMATORS, MAX_DEPTH);
    try {
        rf.train(train_dataset);
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Training failed: %s\n", e.what());
        return 1;
    }
    
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        train_end - start_time).count();
    
    printf("[TRAINING] Training completed in %.2f seconds!\n", train_time / 1000.0);
    
    // ═══════════════════════════════════════════════════════
    // PHASE 3: INFERENCE
    // ═══════════════════════════════════════════════════════
    
    printf("\n━━━ Phase 3: Inference ━━━\n\n");
    
    // Load test data
    printf("[LOADING] Reading test CSV: %s\n", TEST_CSV.c_str());
    Dataset test_dataset;
    try {
        test_dataset = Dataset::load_csv(TEST_CSV, N_FEATURES);
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to load test data: %s\n", e.what());
        return 1;
    }
    printf("[LOADING] Loaded %zu test samples\n", test_dataset.n_samples());
    
    // Predict
    printf("\n[INFERENCE] Running predictions...\n");
    auto predict_start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> predictions;
    try {
        predictions = rf.predict(test_dataset);
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Prediction failed: %s\n", e.what());
        return 1;
    }
    
    auto predict_end = std::chrono::high_resolution_clock::now();
    auto predict_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        predict_end - predict_start).count();
    
    printf("[INFERENCE] Predictions completed in %ld ms\n", predict_time);
    
    // ═══════════════════════════════════════════════════════
    // PHASE 4: EVALUATION & RESULTS
    // ═══════════════════════════════════════════════════════
    
    printf("\n━━━ Phase 4: Results ━━━\n\n");
    
    // Calculate MSE
    float mse = calculate_mse(predictions, test_dataset.get_targets());
    
    // Calculate total time
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        predict_end - start_time).count();
    
    // Print results
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Performance Metrics                           ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║  Mean Squared Error: %-25.2f ║\n", mse);
    printf("║  Training Time:      %-25.2f s ║\n", train_time / 1000.0);
    printf("║  Inference Time:     %-25ld ms ║\n", predict_time);
    printf("║  Total Time:         %-25.2f s ║\n", total_time / 1000.0);
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║  Model Configuration                           ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║  Number of Trees:    %-25zu ║\n", N_ESTIMATORS);
    printf("║  Max Depth:          %-25zu ║\n", MAX_DEPTH);
    printf("║  Features:           %-25zu ║\n", N_FEATURES);
    printf("║  Training Samples:   %-25zu ║\n", train_dataset.n_samples());
    printf("║  Test Samples:       %-25zu ║\n", test_dataset.n_samples());
    printf("╚════════════════════════════════════════════════╝\n");
    
    printf("\n✅ Confidential ML workflow completed successfully!\n");
    printf("   • VM attestation: PASSED\n");
    printf("   • GPU attestation: PASSED\n");
    printf("   • ML training: COMPLETED\n");
    printf("   • Inference: COMPLETED\n");
    
    return 0;
}
