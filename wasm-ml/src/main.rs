//! WASM ML Benchmark - Diabetes Prediction with TEE Attestation and GPU Acceleration
//! 
//! This program replicates the exact behavior of the Python ML pipeline
//! with added TEE (Trusted Execution Environment) attestation and GPU acceleration:
//! 
//! 0. [TEE] Attest VM and GPU before processing sensitive data
//! 1. Load training data from CSV
//! 2. Train RandomForest (200 trees, depth 16) - GPU accelerated via wasi:gpu
//! 3. Save model
//! 4. Load test data from CSV
//! 5. Load model
//! 6. Predict on test set - GPU accelerated
//! 7. Calculate and print MSE

use std::fs;
use std::error::Error;
use std::time::Instant;
use csv::ReaderBuilder;

// Import from library
use wasm_ml::random_forest::RandomForest;
use wasm_ml::data::Dataset;
use wasm_ml::gpu_wasi::{GpuTrainer, GpuPredictor, GpuExecutor};

// Import attestation module (only for WASM target)
#[cfg(target_arch = "wasm32")]
use wasm_ml::attestation::{attest_vm_token, attest_gpu_token, verify_attestation_token, detect_tee_type};

// Model parameters - MUST match Python configuration
const N_ESTIMATORS: usize = 200;
const MAX_DEPTH: usize = 16;
const N_FEATURES: usize = 10;
const MODEL_PATH: &str = "data/model_diabetes_wasm.bin";

/// Timing helper
struct Timer {
    start: Instant,
    label: String,
}

impl Timer {
    fn new(label: &str) -> Self {
        println!("[TIMING] Starting: {}", label);
        Self {
            start: Instant::now(),
            label: label.to_string(),
        }
    }
    
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    fn stop(&self) -> f64 {
        let elapsed = self.elapsed_ms();
        println!("[TIMING] {}: {:.2} ms", self.label, elapsed);
        elapsed
    }
}

/// Print GPU information
fn print_gpu_info() -> Result<(), Box<dyn Error>> {
    println!("\n=== GPU INFORMATION ===\n");
    
    match GpuExecutor::new() {
        Ok(executor) => {
            println!("[GPU] Device: {}", executor.device_name());
            println!("[GPU] Backend: {}", executor.backend());
            println!("[GPU] Hardware GPU: {}", if executor.is_hardware_gpu() { "YES ✓" } else { "NO (software)" });
        }
        Err(e) => {
            println!("[GPU] Error initializing: {}", e);
            println!("[GPU] Will fall back to CPU");
        }
    }
    
    Ok(())
}

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
                if verify_attestation_token(token) {
                    println!("  Token verification: PASSED");
                } else {
                    println!("  Token verification: FAILED (but continuing)");
                }
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
                if verify_attestation_token(token) {
                    println!("  Token verification: PASSED");
                } else {
                    println!("  Token verification: FAILED (but continuing)");
                }
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

/// Train with GPU acceleration via wasi:gpu
fn train_and_save_gpu() -> Result<f64, Box<dyn Error>> {
    println!("\n=== TRAINING PHASE (GPU via wasi:gpu) ===\n");
    
    // Load training data
    let (train_data, train_labels, n_samples) = load_csv("data/diabetes_train.csv")?;
    
    // Create dataset
    let dataset = Dataset::new(train_data.clone(), train_labels.clone(), n_samples, N_FEATURES)?;
    
    // Initialize GPU trainer
    let mut gpu_trainer = GpuTrainer::new()?;
    
    // Upload data to GPU
    let upload_timer = Timer::new("GPU data upload");
    gpu_trainer.upload_training_data(&train_data, &train_labels, n_samples, N_FEATURES)?;
    upload_timer.stop();
    
    // Create RandomForest
    println!("[TRAINING] Creating RandomForest with {} estimators, max_depth {}", 
             N_ESTIMATORS, MAX_DEPTH);
    
    let mut rf = RandomForest::new(N_ESTIMATORS, MAX_DEPTH);
    
    // Train with GPU
    let train_timer = Timer::new("GPU training");
    rf.train_with_gpu(&dataset, &gpu_trainer)?;
    let train_time = train_timer.stop();
    
    // Cleanup GPU
    gpu_trainer.cleanup()?;
    
    println!("[TRAINING] Training completed!");
    
    // Serialize and save model
    let model_bytes = bincode::serialize(&rf)?;
    fs::write(MODEL_PATH, &model_bytes)?;
    
    println!("[TRAINING] Model saved to: {}", MODEL_PATH);
    println!("[TRAINING] Model size: {} bytes", model_bytes.len());
    
    Ok(train_time)
}

/// Train with CPU only (for comparison)
fn train_and_save_cpu() -> Result<f64, Box<dyn Error>> {
    println!("\n=== TRAINING PHASE (CPU) ===\n");
    
    // Load training data
    let (train_data, train_labels, n_samples) = load_csv("data/diabetes_train.csv")?;
    
    // Create dataset
    let dataset = Dataset::new(train_data, train_labels, n_samples, N_FEATURES)?;
    
    // Create and train RandomForest
    println!("[TRAINING] Creating RandomForest with {} estimators, max_depth {}", 
             N_ESTIMATORS, MAX_DEPTH);
    
    let mut rf = RandomForest::new(N_ESTIMATORS, MAX_DEPTH);
    
    let train_timer = Timer::new("CPU training");
    rf.train(&dataset)?;
    let train_time = train_timer.stop();
    
    println!("[TRAINING] Training completed!");
    
    // Serialize and save model
    let model_bytes = bincode::serialize(&rf)?;
    fs::write(MODEL_PATH, &model_bytes)?;
    
    println!("[TRAINING] Model saved to: {}", MODEL_PATH);
    println!("[TRAINING] Model size: {} bytes", model_bytes.len());
    
    Ok(train_time)
}

/// Inference with GPU acceleration
fn load_and_infer_gpu() -> Result<(f32, f64), Box<dyn Error>> {
    println!("\n=== INFERENCE PHASE (GPU via wasi:gpu) ===\n");
    
    // Load test data
    let (test_data, test_labels, n_samples) = load_csv("data/diabetes_test.csv")?;
    
    // Load model
    println!("[INFERENCE] Loading model from: {}", MODEL_PATH);
    let model_bytes = fs::read(MODEL_PATH)?;
    let rf: RandomForest = bincode::deserialize(&model_bytes)?;
    
    println!("[INFERENCE] Model loaded successfully");
    println!("[INFERENCE] Number of trees: {}", rf.n_trees());
    
    // Initialize GPU predictor
    let predictor = GpuPredictor::new()?;
    
    // Get tree predictions (on CPU) then average on GPU
    let infer_timer = Timer::new("GPU inference");
    
    let n_trees = rf.n_trees();
    let mut tree_predictions = Vec::with_capacity(n_samples * n_trees);
    
    for sample_idx in 0..n_samples {
        let start = sample_idx * N_FEATURES;
        let end = start + N_FEATURES;
        let sample = &test_data[start..end];
        
        let preds = rf.get_tree_predictions(sample);
        tree_predictions.extend(preds);
    }
    
    // Average on GPU
    let predictions = predictor.average_predictions(&tree_predictions, n_samples, n_trees)?;
    let infer_time = infer_timer.stop();
    
    // Calculate MSE
    let mse = calculate_mse(&predictions, &test_labels);
    
    println!("[INFERENCE] Samples: {}", n_samples);
    println!("[INFERENCE] Mean Squared Error: {:.4}", mse);
    
    Ok((mse, infer_time))
}

/// Inference with CPU only (for comparison)
fn load_and_infer_cpu() -> Result<(f32, f64), Box<dyn Error>> {
    println!("\n=== INFERENCE PHASE (CPU) ===\n");
    
    // Load test data
    let (test_data, test_labels, n_samples) = load_csv("data/diabetes_test.csv")?;
    
    // Load model
    println!("[INFERENCE] Loading model from: {}", MODEL_PATH);
    let model_bytes = fs::read(MODEL_PATH)?;
    let rf: RandomForest = bincode::deserialize(&model_bytes)?;
    
    println!("[INFERENCE] Model loaded successfully");
    println!("[INFERENCE] Number of trees: {}", rf.n_trees());
    
    // Predict on CPU
    let infer_timer = Timer::new("CPU inference");
    let predictions = rf.predict_cpu(&test_data, n_samples, N_FEATURES)?;
    let infer_time = infer_timer.stop();
    
    // Calculate MSE
    let mse = calculate_mse(&predictions, &test_labels);
    
    println!("[INFERENCE] Samples: {}", n_samples);
    println!("[INFERENCE] Mean Squared Error: {:.4}", mse);
    
    Ok((mse, infer_time))
}

/// Main entry point
fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   WASM ML Benchmark - Diabetes Prediction                ║");
    println!("║   With TEE Attestation and GPU Acceleration (wasi:gpu)   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    // Step 0: Print GPU info
    print_gpu_info()?;
    
    // Step 1: TEE Attestation
    perform_attestation()?;
    
    // Step 2: Training - try GPU first, fall back to CPU
    let train_time = match train_and_save_gpu() {
        Ok(time) => {
            println!("\n[✓] GPU training succeeded");
            time
        }
        Err(e) => {
            println!("\n[⚠️] GPU training failed: {}", e);
            println!("[⚠️] Falling back to CPU training...");
            train_and_save_cpu()?
        }
    };
    
    // Step 3: Inference - try GPU first, fall back to CPU
    let (mse, infer_time) = match load_and_infer_gpu() {
        Ok(result) => {
            println!("\n[✓] GPU inference succeeded");
            result
        }
        Err(e) => {
            println!("\n[⚠️] GPU inference failed: {}", e);
            println!("[⚠️] Falling back to CPU inference...");
            load_and_infer_cpu()?
        }
    };
    
    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK SUMMARY                      ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Training time:  {:>10.2} ms                          ║", train_time);
    println!("║  Inference time: {:>10.2} ms                          ║", infer_time);
    println!("║  MSE:            {:>10.4}                             ║", mse);
    println!("╚══════════════════════════════════════════════════════════╝");
    
    println!("\n✅ Benchmark completed successfully!");
    
    Ok(())
}
