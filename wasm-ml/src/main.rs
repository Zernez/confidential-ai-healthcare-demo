//! WASM ML Benchmark - Diabetes Prediction with TEE Attestation
//! 
//! This program replicates the exact behavior of the Python ML pipeline
//! with added TEE (Trusted Execution Environment) attestation:
//! 
//! 0. [NEW] Attest VM and GPU before processing sensitive data
//! 1. Load training data from CSV
//! 2. Train RandomForest (200 trees, depth 16)
//! 3. Save model
//! 4. Load test data from CSV
//! 5. Load model
//! 6. Predict on test set
//! 7. Calculate and print MSE

use std::fs;
use std::error::Error;
use csv::ReaderBuilder;

// Import from library
use wasm_ml::random_forest::RandomForest;
use wasm_ml::data::Dataset;
use wasm_ml::gpu_compute::GpuExecutor;

// Import attestation module (only for WASM target)
#[cfg(target_arch = "wasm32")]
use wasm_ml::attestation::{attest_vm_token, attest_gpu_token, verify_attestation_token, detect_tee_type};

// Model parameters - MUST match Python configuration
const N_ESTIMATORS: usize = 200;
const MAX_DEPTH: usize = 16;
const N_FEATURES: usize = 10;
const MODEL_PATH: &str = "data/model_diabetes_wasm.bin";

/// Perform TEE attestation before processing sensitive data
#[cfg(target_arch = "wasm32")]
fn perform_attestation() -> Result<(), Box<dyn Error>> {
    println!("\n=== TEE ATTESTATION PHASE ===\n");
    
    // Step 0: Detect TEE type
    println!("[🔍 DETECTION] Detecting TEE environment...");
    match detect_tee_type() {
        Ok(tee_info) => {
            println!("[✓ TEE] Detected: {}", tee_info.tee_type);
            println!("  Supports attestation: {}", if tee_info.supports_attestation { "YES" } else { "NO" });
        }
        Err(e) => {
            println!("[⚠️  TEE] Detection failed: {}", e);
        }
    }
    
    // Step 1: Attest VM (TDX or AMD SEV-SNP)
    println!("\n[🔐 ATTESTATION] Attesting VM (TDX/SEV-SNP)...");
    match attest_vm_token() {
        Ok(result) => {
            println!("[✓ VM] Attestation successful!");
            if let Some(tee_type) = &result.tee_type {
                println!("  TEE Type: {}", tee_type);
            }
            if let Some(token) = &result.token {
                println!("  Token length: {} chars", token.len());
                
                // Verify the token
                if verify_attestation_token(token) {
                    println!("  Token verification: PASSED");
                } else {
                    println!("  Token verification: FAILED (but continuing)");
                }
            }
            if let Some(evidence) = &result.evidence {
                println!("  Evidence available: {} bytes", evidence.len());
                // Print first few lines of evidence for debugging
                let preview: String = evidence.lines().take(5).collect::<Vec<_>>().join("\n");
                println!("  Evidence preview:\n{}", preview);
            }
        }
        Err(e) => {
            println!("[⚠️  VM] Attestation failed: {}", e);
            println!("  Note: This is expected if not running in a Confidential VM");
        }
    }
    
    // Step 2: Attest GPU (NVIDIA H100 via LOCAL or NRAS)
    println!("\n[🔐 ATTESTATION] Attesting GPU (NVIDIA H100)...");
    match attest_gpu_token(0) {
        Ok(result) => {
            println!("[✓ GPU] Attestation successful!");
            if let Some(token) = &result.token {
                println!("  Token length: {} chars", token.len());
                
                // Verify the token
                if verify_attestation_token(token) {
                    println!("  Token verification: PASSED");
                } else {
                    println!("  Token verification: FAILED (but continuing)");
                }
            }
            if let Some(evidence) = &result.evidence {
                println!("  Evidence available: {} bytes", evidence.len());
            }
        }
        Err(e) => {
            println!("[⚠️  GPU] Attestation failed: {}", e);
            println!("  Note: This is expected if NVIDIA driver doesn't support attestation");
        }
    }
    
    println!("\n[✓ ATTESTATION] Phase completed - proceeding with ML training");
    Ok(())
}

/// Placeholder for non-WASM builds
#[cfg(not(target_arch = "wasm32"))]
fn perform_attestation() -> Result<(), Box<dyn Error>> {
    println!("\n=== TEE ATTESTATION PHASE ===\n");
    println!("[⚠️  SKIP] Attestation skipped (not running in WASM)\n");
    Ok(())
}

/// Load diabetes dataset from CSV
fn load_csv(path: &str) -> Result<(Vec<f32>, Vec<f32>, usize), Box<dyn Error>> {
    println!("[LOADING] Reading CSV: {}", path);
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    let mut data = Vec::new();
    let mut labels = Vec::new();
    let mut n_samples = 0;
    
    for result in reader.records() {
        let record = result?;
        
        // First 10 columns are features, last is target
        for i in 0..N_FEATURES {
            let value: f32 = record[i].parse()?;
            data.push(value);
        }
        
        let target: f32 = record[N_FEATURES].parse()?;
        labels.push(target);
        
        n_samples += 1;
    }
    
    println!("[LOADING] Loaded {} samples with {} features", n_samples, N_FEATURES);
    
    Ok((data, labels, n_samples))
}

/// Calculate Mean Squared Error
fn calculate_mse(predictions: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(predictions.len(), actual.len(), "Prediction and actual length mismatch");
    
    let sum: f32 = predictions.iter()
        .zip(actual.iter())
        .map(|(pred, act)| {
            let diff = pred - act;
            diff * diff
        })
        .sum();
    
    sum / predictions.len() as f32
}

/// Main training function - matches train_model.py
fn train_and_save() -> Result<(), Box<dyn Error>> {
    println!("\n=== TRAINING PHASE ===\n");
    
    // Load training data
    let (train_data, train_labels, n_samples) = load_csv("data/diabetes_train.csv")?;
    
    // Create dataset
    let dataset = Dataset::new(train_data, train_labels, n_samples, N_FEATURES)?;
    
    // Create and train RandomForest
    println!("[TRAINING] Creating RandomForest with {} estimators, max_depth {}", 
             N_ESTIMATORS, MAX_DEPTH);
    
    let mut rf = RandomForest::new(N_ESTIMATORS, MAX_DEPTH);
    
    println!("[TRAINING] Starting training on CPU (this may take a while)...");
    rf.train(&dataset)?;
    
    println!("[TRAINING] Training completed!");
    
    // Serialize and save model
    let model_bytes = bincode::serialize(&rf)?;
    fs::write(MODEL_PATH, &model_bytes)?;
    
    println!("[TRAINING] Model saved to: {}", MODEL_PATH);
    println!("[TRAINING] Model size: {} bytes", model_bytes.len());
    
    Ok(())
}

/// Main inference function - matches infer_model.py
fn load_and_infer() -> Result<(), Box<dyn Error>> {
    println!("\n=== INFERENCE PHASE ===\n");
    
    // Load test data
    let (test_data, test_labels, n_samples) = load_csv("data/diabetes_test.csv")?;
    
    // Load model
    println!("[INFERENCE] Loading model from: {}", MODEL_PATH);
    let model_bytes = fs::read(MODEL_PATH)?;
    let rf: RandomForest = bincode::deserialize(&model_bytes)?;
    
    println!("[INFERENCE] Model loaded successfully");
    println!("[INFERENCE] Number of trees: {}", rf.n_trees());
    
    // Predict on test set (CPU)
    println!("[INFERENCE] Running predictions on {} test samples...", n_samples);
    let predictions = rf.predict_cpu(&test_data, n_samples, N_FEATURES)?;
    
    // Calculate MSE
    let mse = calculate_mse(&predictions, &test_labels);
    
    // Print results - same format as Python
    println!("[INFERENCE] Samples: {}", n_samples);
    println!("[INFERENCE] Mean Squared Error (CPU): {:.4}", mse);
    
    Ok(())
}

/// Main entry point - matches main.py sequence with attestation
fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   WASM ML Benchmark - Diabetes Prediction     ║");
    println!("║   With TEE Attestation (VM + GPU)             ║");
    println!("╚════════════════════════════════════════════════╝");
    
    // Step 0: TEE Attestation (NEW)
    perform_attestation()?;
    
    // Step 1: Training (matches MLTrainer.train_and_split())
    train_and_save()?;
    
    // Step 2: Inference (matches MLInferencer.run_inference())
    load_and_infer()?;
    
    println!("\n✅ Benchmark completed successfully!");
    
    Ok(())
}
