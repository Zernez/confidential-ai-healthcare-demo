//! WASM ML Module - RandomForest with GPU Acceleration via wasi:gpu
//! 
//! This module implements a RandomForest classifier/regressor with GPU acceleration.
//! GPU operations are abstracted through the wasi:gpu interface, allowing the host
//! runtime to use either WebGPU/Vulkan or CUDA as the backend.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         WASM Module (this)          │
//! │  - RandomForest algorithm           │
//! │  - Training logic                   │
//! │  - Prediction logic                 │
//! └─────────────────┬───────────────────┘
//!                   │ wasi:gpu calls
//!                   ▼
//! ┌─────────────────────────────────────┐
//! │         Host Runtime                │
//! │  ┌─────────────┐ ┌───────────────┐  │
//! │  │   WebGPU    │ │     CUDA      │  │
//! │  │  (Vulkan)   │ │   (cuBLAS)    │  │
//! │  └─────────────┘ └───────────────┘  │
//! └─────────────────────────────────────┘
//! ```

#![allow(unused)]

pub mod random_forest;
pub mod gpu_wasi;
pub mod data;

#[cfg(target_arch = "wasm32")]
pub mod attestation;

use random_forest::RandomForest;
use data::Dataset;
use gpu_wasi::{GpuTrainer, GpuPredictor, GpuExecutor};

/// GPU-accelerated training entry point
/// 
/// Trains a RandomForest model using GPU for:
/// - Bootstrap sampling (parallel random index generation)
/// - Split finding (parallel MSE computation)
/// 
/// The GPU backend is automatically selected by the host runtime.
pub fn train_model_gpu(
    n_estimators: usize,
    max_depth: usize,
    training_data: Vec<f32>,
    training_labels: Vec<f32>,
    n_features: usize,
) -> Result<Vec<u8>, String> {
    let n_samples = training_data.len() / n_features;
    
    eprintln!("[wasm-ml] Starting GPU training: {} trees, depth {}, {} samples x {} features",
              n_estimators, max_depth, n_samples, n_features);
    
    // Create dataset
    let dataset = Dataset::new(
        training_data.clone(),
        training_labels.clone(),
        n_samples,
        n_features,
    ).map_err(|e| e.to_string())?;
    
    // Initialize GPU trainer
    let mut gpu_trainer = GpuTrainer::new()?;
    
    // Upload training data to GPU (once)
    gpu_trainer.upload_training_data(&training_data, &training_labels, n_samples, n_features)?;
    
    // Train RandomForest with GPU acceleration
    let mut rf = RandomForest::new(n_estimators, max_depth);
    rf.train_with_gpu(&dataset, &gpu_trainer)?;
    
    // Cleanup GPU resources
    gpu_trainer.cleanup()?;
    
    // Serialize model
    let serialized = bincode::serialize(&rf)
        .map_err(|e| format!("Serialization error: {}", e))?;
    
    eprintln!("[wasm-ml] Training complete, model size: {} bytes", serialized.len());
    
    Ok(serialized)
}

/// CPU-only training (fallback when GPU unavailable or for comparison)
pub fn train_model_cpu(
    n_estimators: usize,
    max_depth: usize,
    training_data: Vec<f32>,
    training_labels: Vec<f32>,
    n_features: usize,
) -> Result<Vec<u8>, String> {
    let n_samples = training_data.len() / n_features;
    
    eprintln!("[wasm-ml] Starting CPU training: {} trees, depth {}, {} samples x {} features",
              n_estimators, max_depth, n_samples, n_features);
    
    let dataset = Dataset::new(
        training_data,
        training_labels,
        n_samples,
        n_features,
    ).map_err(|e| e.to_string())?;
    
    let mut rf = RandomForest::new(n_estimators, max_depth);
    rf.train(&dataset).map_err(|e| e.to_string())?;
    
    let serialized = bincode::serialize(&rf)
        .map_err(|e| format!("Serialization error: {}", e))?;
    
    eprintln!("[wasm-ml] Training complete, model size: {} bytes", serialized.len());
    
    Ok(serialized)
}

/// GPU-accelerated inference
/// 
/// Uses GPU for parallel tree prediction averaging.
pub fn predict_gpu(
    model_bytes: Vec<u8>,
    input_data: Vec<f32>,
    n_features: usize,
) -> Result<Vec<f32>, String> {
    let rf: RandomForest = bincode::deserialize(&model_bytes)
        .map_err(|e| format!("Deserialization error: {}", e))?;
    
    let n_samples = input_data.len() / n_features;
    eprintln!("[wasm-ml] GPU prediction: {} samples", n_samples);
    
    // Get predictions from each tree for each sample
    let n_trees = rf.n_trees();
    let mut tree_predictions = Vec::with_capacity(n_samples * n_trees);
    
    for sample_idx in 0..n_samples {
        let start = sample_idx * n_features;
        let end = start + n_features;
        let sample = &input_data[start..end];
        
        let preds = rf.get_tree_predictions(sample);
        tree_predictions.extend(preds);
    }
    
    // Use GPU to average predictions
    let predictor = GpuPredictor::new()?;
    let predictions = predictor.average_predictions(&tree_predictions, n_samples, n_trees)?;
    
    Ok(predictions)
}

/// CPU fallback inference
pub fn predict_cpu(
    model_bytes: Vec<u8>,
    input_data: Vec<f32>,
    n_features: usize,
) -> Result<Vec<f32>, String> {
    let rf: RandomForest = bincode::deserialize(&model_bytes)
        .map_err(|e| format!("Deserialization error: {}", e))?;
    
    let n_samples = input_data.len() / n_features;
    eprintln!("[wasm-ml] CPU prediction: {} samples", n_samples);
    
    rf.predict_cpu(&input_data, n_samples, n_features)
}

/// Get GPU device information
pub fn get_gpu_info() -> Result<String, String> {
    let executor = GpuExecutor::new()?;
    Ok(format!(
        "Device: {}, Backend: {}, Hardware: {}",
        executor.device_name(),
        executor.backend(),
        executor.is_hardware_gpu()
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Component Model ABI - cabi_realloc
// Required for host to allocate memory in the guest
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn cabi_realloc(
    old_ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    use std::alloc::{alloc, dealloc, realloc, Layout};
    
    if new_size == 0 {
        if !old_ptr.is_null() && old_size > 0 {
            let layout = Layout::from_size_align_unchecked(old_size, align);
            dealloc(old_ptr, layout);
        }
        return std::ptr::null_mut();
    }
    
    let new_layout = Layout::from_size_align_unchecked(new_size, align);
    
    if old_ptr.is_null() || old_size == 0 {
        alloc(new_layout)
    } else {
        let old_layout = Layout::from_size_align_unchecked(old_size, align);
        realloc(old_ptr, old_layout, new_size)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Component Model Exports
// ═══════════════════════════════════════════════════════════════════════════

/// Export functions for WASM component model
#[cfg(target_arch = "wasm32")]
mod exports {
    use super::*;
    
    #[no_mangle]
    pub extern "C" fn wasm_ml_train_gpu(
        n_estimators: u32,
        max_depth: u32,
        data_ptr: *const f32,
        data_len: u32,
        labels_ptr: *const f32,
        labels_len: u32,
        n_features: u32,
        output_ptr: *mut u8,
        output_capacity: u32,
    ) -> i32 {
        let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
        let labels = unsafe { std::slice::from_raw_parts(labels_ptr, labels_len as usize) };
        
        match train_model_gpu(
            n_estimators as usize,
            max_depth as usize,
            data.to_vec(),
            labels.to_vec(),
            n_features as usize,
        ) {
            Ok(model_bytes) => {
                let len = model_bytes.len().min(output_capacity as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(model_bytes.as_ptr(), output_ptr, len);
                }
                len as i32
            }
            Err(_) => -1,
        }
    }
    
    #[no_mangle]
    pub extern "C" fn wasm_ml_predict_gpu(
        model_ptr: *const u8,
        model_len: u32,
        data_ptr: *const f32,
        data_len: u32,
        n_features: u32,
        output_ptr: *mut f32,
        output_capacity: u32,
    ) -> i32 {
        let model = unsafe { std::slice::from_raw_parts(model_ptr, model_len as usize) };
        let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len as usize) };
        
        match predict_gpu(model.to_vec(), data.to_vec(), n_features as usize) {
            Ok(predictions) => {
                let len = predictions.len().min(output_capacity as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(predictions.as_ptr(), output_ptr, len);
                }
                len as i32
            }
            Err(_) => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_workflow() {
        let training_data = vec![
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
            4.0, 5.0, 6.0,
        ];
        let training_labels = vec![10.0, 20.0, 30.0, 40.0];
        let n_features = 3;
        
        let model_bytes = train_model_cpu(10, 5, training_data, training_labels, n_features)
            .expect("Training failed");
        
        let test_data = vec![2.5, 3.5, 4.5];
        let predictions = predict_cpu(model_bytes, test_data, n_features)
            .expect("Prediction failed");
        
        assert_eq!(predictions.len(), 1);
        assert!(predictions[0] > 15.0 && predictions[0] < 35.0);
    }
}
