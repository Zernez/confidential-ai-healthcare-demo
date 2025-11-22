/// Example: WASM ML with TEE Attestation
/// 
/// This demonstrates how to use wasmtime:attestation in a WASM guest module.
/// 
/// Flow:
/// 1. Attest VM (TDX/SEV-SNP)
/// 2. Attest GPU (NVIDIA H100)
/// 3. Verify tokens
/// 4. Only if attestation passes → proceed with ML training
/// 5. Train RandomForest on GPU
/// 6. Run inference

#[cfg(target_arch = "wasm32")]
use wasm_ml::attestation::{attest_vm_token, attest_gpu_token, verify_attestation_token};

use std::error::Error;
use wasm_ml::random_forest::RandomForest;
use wasm_ml::data::Dataset;

/// Main entry point with attestation
fn main() -> Result<(), Box<dyn Error>> {
    println!("╔════════════════════════════════════════════════╗");
    println!("║  Confidential ML with TEE Attestation         ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════
    // PHASE 1: ATTESTATION (Critical Security Step)
    // ═══════════════════════════════════════════════════════
    
    #[cfg(target_arch = "wasm32")]
    {
        println!("━━━ Phase 1: Attestation ━━━\n");
        
        // Step 1: Attest VM
        println!("🔐 [1/4] Attesting VM (TDX/SEV-SNP)...");
        let vm_result = attest_vm_token().map_err(|e| {
            eprintln!("❌ VM attestation failed: {}", e);
            e
        })?;
        
        println!("✓ VM attestation successful!");
        if let Some(token) = &vm_result.token {
            println!("  Token length: {} chars", token.len());
            println!("  Timestamp: {}", vm_result.timestamp);
        }
        
        // Step 2: Attest GPU
        println!("\n🔐 [2/4] Attesting GPU (NVIDIA H100)...");
        let gpu_result = attest_gpu_token(0).map_err(|e| {
            eprintln!("❌ GPU attestation failed: {}", e);
            e
        })?;
        
        println!("✓ GPU attestation successful!");
        if let Some(token) = &gpu_result.token {
            println!("  Token length: {} chars", token.len());
            println!("  Timestamp: {}", gpu_result.timestamp);
        }
        
        // Step 3: Verify VM token
        println!("\n🔍 [3/4] Verifying VM token...");
        let vm_token = vm_result.token.as_ref().unwrap();
        if !verify_attestation_token(vm_token) {
            eprintln!("❌ VM token verification failed!");
            return Err("Invalid VM attestation token".into());
        }
        println!("✓ VM token verified!");
        
        // Step 4: Verify GPU token
        println!("\n🔍 [4/4] Verifying GPU token...");
        let gpu_token = gpu_result.token.as_ref().unwrap();
        if !verify_attestation_token(gpu_token) {
            eprintln!("❌ GPU token verification failed!");
            return Err("Invalid GPU attestation token".into());
        }
        println!("✓ GPU token verified!");
        
        println!("\n✅ All attestations passed! Proceeding with ML training...\n");
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("⚠️  Skipping attestation (not running in WASM)\n");
    }

    // ═══════════════════════════════════════════════════════
    // PHASE 2: ML TRAINING (Only executed if attestation passed)
    // ═══════════════════════════════════════════════════════
    
    println!("━━━ Phase 2: ML Training ━━━\n");
    
    // Load training data
    println!("📂 Loading training data...");
    let (train_data, train_labels) = load_sample_data();
    let n_samples = train_labels.len();
    let n_features = 10;
    
    let dataset = Dataset::new(train_data, train_labels, n_samples, n_features)?;
    println!("✓ Loaded {} samples with {} features\n", n_samples, n_features);
    
    // Train RandomForest
    println!("🌲 Training RandomForest (200 trees, depth 16)...");
    let n_estimators = 200;
    let max_depth = 16;
    
    let mut rf = RandomForest::new(n_estimators, max_depth);
    rf.train(&dataset)?;
    
    println!("✓ Training completed!\n");
    
    // ═══════════════════════════════════════════════════════
    // PHASE 3: INFERENCE
    // ═══════════════════════════════════════════════════════
    
    println!("━━━ Phase 3: Inference ━━━\n");
    
    // Load test data
    let (test_data, test_labels) = load_sample_test_data();
    let test_n_samples = test_labels.len();
    
    println!("📂 Running predictions on {} test samples...", test_n_samples);
    let predictions = rf.predict_cpu(&test_data, test_n_samples, n_features)?;
    
    // Calculate MSE
    let mse = calculate_mse(&predictions, &test_labels);
    println!("✓ Predictions completed!\n");
    
    // ═══════════════════════════════════════════════════════
    // PHASE 4: RESULTS
    // ═══════════════════════════════════════════════════════
    
    println!("━━━ Results ━━━\n");
    println!("Mean Squared Error: {:.2}", mse);
    println!("Number of Trees: {}", n_estimators);
    println!("Max Depth: {}", max_depth);
    
    println!("\n✅ Confidential ML workflow completed successfully!");
    
    Ok(())
}

/// Load sample training data (simplified for example)
fn load_sample_data() -> (Vec<f32>, Vec<f32>) {
    // In real implementation, load from CSV via WASI filesystem
    let train_data = vec![
        // Sample 1
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        // Sample 2
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        // Sample 3
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        // Sample 4
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    
    let train_labels = vec![100.0, 120.0, 140.0, 160.0];
    
    (train_data, train_labels)
}

/// Load sample test data
fn load_sample_test_data() -> (Vec<f32>, Vec<f32>) {
    let test_data = vec![
        2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
    ];
    
    let test_labels = vec![125.0];
    
    (test_data, test_labels)
}

/// Calculate Mean Squared Error
fn calculate_mse(predictions: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(predictions.len(), actual.len());
    
    let sum: f32 = predictions.iter()
        .zip(actual.iter())
        .map(|(pred, act)| {
            let diff = pred - act;
            diff * diff
        })
        .sum();
    
    sum / predictions.len() as f32
}
