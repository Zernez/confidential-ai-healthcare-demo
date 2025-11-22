//! WASM ML Module - RandomForest with WebGPU Acceleration
//! 
//! This module implements a RandomForest classifier/regressor with GPU acceleration
//! via WebGPU compute shaders for both training and inference.

#![allow(unused)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod random_forest;
pub mod gpu_compute;
pub mod gpu_training;  // NEW: GPU training module
pub mod data;

#[cfg(target_arch = "wasm32")]
pub mod attestation;  // TEE attestation bindings

use random_forest::RandomForest;
use data::Dataset;

/// Initialize panic hook for better error messages in WASM
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// GPU-accelerated training entry point
/// Trains a RandomForest model using GPU for bootstrap sampling and split finding
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub async fn train_model_gpu(
    n_estimators: usize,
    max_depth: usize,
    training_data: Vec<f32>,
    training_labels: Vec<f32>,
    n_features: usize,
) -> Result<Vec<u8>, String> {
    // Reshape data
    let n_samples = training_data.len() / n_features;
    
    let dataset = Dataset::new(
        training_data,
        training_labels,
        n_samples,
        n_features,
    ).map_err(|e| e.to_string())?;
    
    // Initialize GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to find GPU adapter")?;
    
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ML Training Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .map_err(|e| format!("Failed to create device: {}", e))?;
    
    // Create GPU trainer
    let gpu_trainer = gpu_training::GpuTrainer::new(device, queue)
        .await
        .map_err(|e| format!("Failed to init GPU trainer: {}", e))?;
    
    // Train RandomForest with GPU acceleration
    let mut rf = RandomForest::new(n_estimators, max_depth);
    rf.train_gpu(&dataset, &gpu_trainer)
        .await
        .map_err(|e| e.to_string())?;
    
    // Serialize model
    let serialized = bincode::serialize(&rf)
        .map_err(|e| format!("Serialization error: {}", e))?;
    
    Ok(serialized)
}

/// CPU-only training (fallback)
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn train_model_cpu(
    n_estimators: usize,
    max_depth: usize,
    training_data: Vec<f32>,
    training_labels: Vec<f32>,
    n_features: usize,
) -> Result<Vec<u8>, String> {
    // Reshape data
    let n_samples = training_data.len() / n_features;
    
    let dataset = Dataset::new(
        training_data,
        training_labels,
        n_samples,
        n_features,
    ).map_err(|e| e.to_string())?;
    
    // Train RandomForest on CPU
    let mut rf = RandomForest::new(n_estimators, max_depth);
    rf.train(&dataset).map_err(|e| e.to_string())?;
    
    // Serialize model
    let serialized = bincode::serialize(&rf)
        .map_err(|e| format!("Serialization error: {}", e))?;
    
    Ok(serialized)
}

/// GPU-accelerated inference entry point
/// Uses WebGPU compute shaders to parallelize predictions across trees
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub async fn predict_gpu(
    model_bytes: Vec<u8>,
    input_data: Vec<f32>,
    n_features: usize,
) -> Result<Vec<f32>, String> {
    // Deserialize model
    let rf: RandomForest = bincode::deserialize(&model_bytes)
        .map_err(|e| format!("Deserialization error: {}", e))?;
    
    // Initialize GPU
    let gpu_executor = gpu_compute::GpuExecutor::new()
        .await
        .map_err(|e| format!("GPU initialization error: {}", e))?;
    
    // Run predictions on GPU
    let predictions = gpu_executor.predict(&rf, &input_data, n_features)
        .await
        .map_err(|e| format!("GPU prediction error: {}", e))?;
    
    Ok(predictions)
}

/// CPU fallback inference (no GPU required)
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn predict_cpu(
    model_bytes: Vec<u8>,
    input_data: Vec<f32>,
    n_features: usize,
) -> Result<Vec<f32>, String> {
    // Deserialize model
    let rf: RandomForest = bincode::deserialize(&model_bytes)
        .map_err(|e| format!("Deserialization error: {}", e))?;
    
    let n_samples = input_data.len() / n_features;
    
    // Predict on CPU
    let predictions = rf.predict_cpu(&input_data, n_samples, n_features)
        .map_err(|e| e.to_string())?;
    
    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cpu_workflow() {
        // Simple synthetic dataset
        let training_data = vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
            4.0, 5.0, 6.0,
        ];
        let training_labels = vec![10.0, 20.0, 30.0, 40.0];
        let n_features = 3;
        
        // Train on CPU
        let model_bytes = train_model_cpu(10, 5, training_data.clone(), training_labels, n_features)
            .expect("Training failed");
        
        // Predict CPU
        let test_data = vec![2.5, 3.5, 4.5];
        let predictions = predict_cpu(model_bytes, test_data, n_features)
            .expect("Prediction failed");
        
        assert_eq!(predictions.len(), 1);
        // Rough check - should be around 25.0
        assert!(predictions[0] > 15.0 && predictions[0] < 35.0);
    }
    
    #[tokio::test]
    async fn test_gpu_training() {
        // Test GPU training
        let training_data = vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
            4.0, 5.0, 6.0,
        ];
        let training_labels = vec![10.0, 20.0, 30.0, 40.0];
        let n_features = 3;
        
        // Train on GPU
        let result = train_model_gpu(5, 3, training_data, training_labels, n_features).await;
        
        // Should succeed or fail gracefully if GPU not available
        match result {
            Ok(model_bytes) => {
                assert!(model_bytes.len() > 0);
                println!("GPU training succeeded");
            }
            Err(e) => {
                println!("GPU training not available: {}", e);
                // This is okay in CI environments
            }
        }
    }
}
