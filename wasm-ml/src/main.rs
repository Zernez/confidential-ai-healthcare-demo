//! WASM ML Benchmark - Diabetes Prediction with TEE Attestation and GPU Acceleration
//! 
//! Unified benchmark with:
//! - TEE attestation (AMD SEV-SNP / Intel TDX)
//! - GPU acceleration via wasi:gpu
//! - Structured JSON output for benchmark aggregation

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
use wasm_ml::attestation::{attest_vm_token, attest_gpu_token, detect_tee_type};

// Model parameters - MUST match Python/C++ configuration
const N_ESTIMATORS: usize = 200;
const MAX_DEPTH: usize = 16;
const N_FEATURES: usize = 10;
const MODEL_PATH: &str = "data/model_diabetes_wasm.bin";

/// Benchmark results structure
struct BenchmarkResults {
    language: String,
    gpu_device: String,
    gpu_backend: String,
    tee_type: String,
    gpu_available: bool,
    tee_available: bool,
    attestation_ms: f64,
    training_ms: f64,
    inference_ms: f64,
    mse: f32,
    train_samples: usize,
    test_samples: usize,
}

impl BenchmarkResults {
    fn new() -> Self {
        Self {
            language: "rust".to_string(),
            gpu_device: String::new(),
            gpu_backend: String::new(),
            tee_type: String::new(),
            gpu_available: false,
            tee_available: false,
            attestation_ms: 0.0,
            training_ms: 0.0,
            inference_ms: 0.0,
            mse: 0.0,
            train_samples: 0,
            test_samples: 0,
        }
    }
    
    fn to_json(&self) -> String {
        format!(
            r#"{{"language":"{}","gpu_device":"{}","gpu_backend":"{}","tee_type":"{}","gpu_available":{},"tee_available":{},"attestation_ms":{:.2},"training_ms":{:.2},"inference_ms":{:.2},"mse":{:.4},"train_samples":{},"test_samples":{}}}"#,
            self.language,
            self.gpu_device,
            self.gpu_backend,
            self.tee_type,
            self.gpu_available,
            self.tee_available,
            self.attestation_ms,
            self.training_ms,
            self.inference_ms,
            self.mse,
            self.train_samples,
            self.test_samples
        )
    }
}

/// Timer utility
struct Timer {
    start: Instant,
}

impl Timer {
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

/// Perform TEE attestation
#[cfg(target_arch = "wasm32")]
fn perform_attestation(results: &mut BenchmarkResults) {
    println!("\n=== TEE ATTESTATION ===");
    
    let timer = Timer::new();
    
    // Detect TEE type
    match detect_tee_type() {
        Ok(tee_info) => {
            results.tee_type = tee_info.tee_type.clone();
            results.tee_available = tee_info.supports_attestation;
            println!("[TEE] Type: {}", tee_info.tee_type);
            println!("[TEE] Supports attestation: {}", if tee_info.supports_attestation { "YES" } else { "NO" });
        }
        Err(e) => {
            println!("[TEE] Detection failed: {}", e);
        }
    }
    
    // Attest VM
    match attest_vm_token() {
        Ok(result) => {
            if let Some(token) = &result.token {
                println!("[TEE] VM attestation: OK (token: {} chars)", token.len());
            } else {
                println!("[TEE] VM attestation: OK (no token)");
            }
        }
        Err(e) => {
            println!("[TEE] VM attestation: SKIPPED ({})", e);
        }
    }
    
    // Attest GPU
    match attest_gpu_token(0) {
        Ok(result) => {
            if let Some(token) = &result.token {
                println!("[TEE] GPU attestation: OK (token: {} chars)", token.len());
            } else {
                println!("[TEE] GPU attestation: OK (no token)");
            }
        }
        Err(e) => {
            println!("[TEE] GPU attestation: SKIPPED ({})", e);
        }
    }
    
    results.attestation_ms = timer.elapsed_ms();
    println!("[TIMING] Attestation: {:.2} ms", results.attestation_ms);
}

/// Placeholder for non-WASM builds
#[cfg(not(target_arch = "wasm32"))]
fn perform_attestation(results: &mut BenchmarkResults) {
    println!("\n=== TEE ATTESTATION ===");
    println!("[TEE] Type: None (not running in WASM)");
    println!("[TEE] Supports attestation: NO");
    println!("[TIMING] Attestation: 0.00 ms");
}

/// Load diabetes dataset from CSV
fn load_csv(path: &str) -> Result<(Vec<f32>, Vec<f32>, usize), Box<dyn Error>> {
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

/// Main entry point
fn main() -> Result<(), Box<dyn Error>> {
    let mut results = BenchmarkResults::new();
    
    // ═══════════════════════════════════════════════════════════════════
    // HEADER
    // ═══════════════════════════════════════════════════════════════════
    
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   WASM ML Benchmark - Diabetes Prediction                ║");
    println!("║   Rust + wasi:gpu + TEE Attestation                      ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    // ═══════════════════════════════════════════════════════════════════
    // GPU INFORMATION
    // ═══════════════════════════════════════════════════════════════════
    
    println!("\n=== GPU INFORMATION ===");
    
    match GpuExecutor::new() {
        Ok(executor) => {
            results.gpu_available = true;
            results.gpu_device = executor.device_name().to_string();
            results.gpu_backend = executor.backend().to_string();
            
            println!("[GPU] Device: {}", results.gpu_device);
            println!("[GPU] Backend: {}", results.gpu_backend);
            println!("[GPU] Memory: {} MB", executor.total_memory() / (1024 * 1024));
            println!("[GPU] Hardware: {}", if executor.is_hardware_gpu() { "YES ✓" } else { "NO (software)" });
        }
        Err(_) => {
            println!("[GPU] Not available - will use CPU");
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // TEE ATTESTATION
    // ═══════════════════════════════════════════════════════════════════
    
    perform_attestation(&mut results);
    
    // ═══════════════════════════════════════════════════════════════════
    // TRAINING PHASE
    // ═══════════════════════════════════════════════════════════════════
    
    println!("\n=== TRAINING ===");
    
    // Load training data
    let (train_data, train_labels, n_samples) = load_csv("data/diabetes_train.csv")?;
    results.train_samples = n_samples;
    
    println!("[TRAIN] Dataset: {} samples, {} features", n_samples, N_FEATURES);
    println!("[TRAIN] Model: RandomForest ({} trees, depth {})", N_ESTIMATORS, MAX_DEPTH);
    
    // Create dataset
    let dataset = Dataset::new(train_data.clone(), train_labels.clone(), n_samples, N_FEATURES)?;
    
    // Create RandomForest
    let mut rf = RandomForest::new(N_ESTIMATORS, MAX_DEPTH);
    
    let train_timer = Timer::new();
    
    // Try GPU training first
    let gpu_training_ok = if let Ok(mut gpu_trainer) = GpuTrainer::new() {
        if gpu_trainer.upload_training_data(&train_data, &train_labels, n_samples, N_FEATURES).is_ok() {
            println!("[TRAIN] Accelerator: GPU");
            if rf.train_with_gpu(&dataset, &gpu_trainer).is_ok() {
                let _ = gpu_trainer.cleanup();
                true
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };
    
    if !gpu_training_ok {
        println!("[TRAIN] Accelerator: CPU");
        rf.train(&dataset)?;
    }
    
    results.training_ms = train_timer.elapsed_ms();
    println!("[TIMING] Training: {:.2} ms", results.training_ms);
    
    // Save model
    let model_bytes = bincode::serialize(&rf)?;
    fs::write(MODEL_PATH, &model_bytes)?;
    
    // ═══════════════════════════════════════════════════════════════════
    // INFERENCE PHASE
    // ═══════════════════════════════════════════════════════════════════
    
    println!("\n=== INFERENCE ===");
    
    // Load test data
    let (test_data, test_labels, n_test_samples) = load_csv("data/diabetes_test.csv")?;
    results.test_samples = n_test_samples;
    
    println!("[INFER] Test set: {} samples", n_test_samples);
    
    let infer_timer = Timer::new();
    
    // Try GPU inference
    let predictions = if let Ok(predictor) = GpuPredictor::new() {
        println!("[INFER] Accelerator: GPU");
        
        let n_trees = rf.n_trees();
        let mut tree_predictions = Vec::with_capacity(n_test_samples * n_trees);
        
        for sample_idx in 0..n_test_samples {
            let start = sample_idx * N_FEATURES;
            let end = start + N_FEATURES;
            let sample = &test_data[start..end];
            
            let preds = rf.get_tree_predictions(sample);
            tree_predictions.extend(preds);
        }
        
        predictor.average_predictions(&tree_predictions, n_test_samples, n_trees)?
    } else {
        println!("[INFER] Accelerator: CPU");
        rf.predict_cpu(&test_data, n_test_samples, N_FEATURES)?
    };
    
    results.inference_ms = infer_timer.elapsed_ms();
    println!("[TIMING] Inference: {:.2} ms", results.inference_ms);
    
    // Calculate MSE
    results.mse = calculate_mse(&predictions, &test_labels);
    println!("[INFER] MSE: {:.4}", results.mse);
    
    // ═══════════════════════════════════════════════════════════════════
    // BENCHMARK SUMMARY
    // ═══════════════════════════════════════════════════════════════════
    
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                      ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Language:       Rust                                     ║");
    println!("║  Attestation:  {:>10.2} ms                           ║", results.attestation_ms);
    println!("║  Training:     {:>10.2} ms                           ║", results.training_ms);
    println!("║  Inference:    {:>10.2} ms                           ║", results.inference_ms);
    println!("║  MSE:          {:>10.4}                             ║", results.mse);
    println!("╚══════════════════════════════════════════════════════════╝");
    
    // ═══════════════════════════════════════════════════════════════════
    // JSON OUTPUT FOR BENCHMARK AGGREGATION
    // ═══════════════════════════════════════════════════════════════════
    
    println!("\n### BENCHMARK_JSON ###");
    println!("{}", results.to_json());
    println!("### END_BENCHMARK_JSON ###");
    
    println!("\n✅ Benchmark completed successfully!");
    
    Ok(())
}
