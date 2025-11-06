//! WASM ML Benchmark - Diabetes Prediction
//! 
//! This program replicates the exact behavior of the Python ML pipeline:
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

// Model parameters - MUST match Python configuration
const N_ESTIMATORS: usize = 200;
const MAX_DEPTH: usize = 16;
const N_FEATURES: usize = 10;
const MODEL_PATH: &str = "data/model_diabetes_wasm.bin";

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

/// Main entry point - matches main.py sequence
fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   WASM ML Benchmark - Diabetes Prediction     ║");
    println!("║   Matching Python RAPIDS implementation       ║");
    println!("╚════════════════════════════════════════════════╝");
    
    // Step 1: Training (matches MLTrainer.train_and_split())
    train_and_save()?;
    
    // Step 2: Inference (matches MLInferencer.run_inference())
    load_and_infer()?;
    
    println!("\n✅ Benchmark completed successfully!");
    
    Ok(())
}
