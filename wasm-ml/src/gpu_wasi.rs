//! GPU Compute via wasi:gpu host functions
//!
//! This module provides GPU acceleration by calling host-provided functions.
//! The host runtime implements these using either WebGPU/Vulkan or CUDA.
//!
//! The WASM module is completely agnostic to the underlying GPU backend.

use bytemuck::cast_slice;

// Generate bindings from WIT
// This creates the `wasi::gpu::compute` and `wasi::gpu::ml_kernels` modules
wit_bindgen::generate!({
    world: "ml-compute",
    path: "wit",
});

// Re-export generated types for convenience
pub use wasi::gpu::compute::{
    DeviceInfo, BufferUsage, BufferId, GpuError,
    get_device_info, buffer_create, buffer_write, buffer_read, buffer_destroy, sync,
};
pub use wasi::gpu::ml_kernels::{
    BootstrapParams, FindSplitParams, AverageParams, MatmulParams,
    kernel_bootstrap_sample, kernel_find_split, kernel_average, kernel_matmul,
};

/// High-level GPU executor wrapping wasi:gpu calls
pub struct GpuExecutor {
    device_info: DeviceInfo,
}

impl GpuExecutor {
    /// Initialize GPU executor by querying device info
    pub fn new() -> Result<Self, String> {
        let device_info = get_device_info();
        
        eprintln!("[wasi:gpu] Connected to GPU: {} ({})", 
                  device_info.name, device_info.backend);
        eprintln!("[wasi:gpu] Memory: {} MB, Hardware: {}", 
                  device_info.total_memory / (1024 * 1024),
                  device_info.is_hardware);
        
        if !device_info.is_hardware {
            eprintln!("[wasi:gpu] WARNING: Using software renderer, performance will be slow");
        }
        
        Ok(Self { device_info })
    }
    
    /// Check if using hardware GPU
    pub fn is_hardware_gpu(&self) -> bool {
        self.device_info.is_hardware
    }
    
    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.device_info.name
    }
    
    /// Get backend type ("cuda", "vulkan", "webgpu")
    pub fn backend(&self) -> &str {
        &self.device_info.backend
    }
    
    /// Allocate a GPU buffer
    pub fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<BufferId, String> {
        buffer_create(size, usage)
            .map_err(|e| format!("Buffer creation failed: {:?}", e))
    }
    
    /// Write data to GPU buffer
    pub fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), String> {
        buffer_write(buffer, offset, data)
            .map_err(|e| format!("Buffer write failed: {:?}", e))
    }
    
    /// Read data from GPU buffer
    pub fn read_buffer(&self, buffer: BufferId, offset: u64, size: u32) -> Result<Vec<u8>, String> {
        buffer_read(buffer, offset, size)
            .map_err(|e| format!("Buffer read failed: {:?}", e))
    }
    
    /// Free GPU buffer
    pub fn destroy_buffer(&self, buffer: BufferId) -> Result<(), String> {
        buffer_destroy(buffer)
            .map_err(|e| format!("Buffer destroy failed: {:?}", e))
    }
    
    /// Synchronize GPU operations
    pub fn sync(&self) {
        sync();
    }
}

/// GPU-accelerated training operations
pub struct GpuTrainer {
    executor: GpuExecutor,
    // Persistent buffers for training data
    data_buffer: Option<BufferId>,
    labels_buffer: Option<BufferId>,
    n_samples: usize,
    n_features: usize,
}

impl GpuTrainer {
    /// Create a new GPU trainer
    pub fn new() -> Result<Self, String> {
        let executor = GpuExecutor::new()?;
        Ok(Self {
            executor,
            data_buffer: None,
            labels_buffer: None,
            n_samples: 0,
            n_features: 0,
        })
    }
    
    /// Upload training data to GPU (call once before training)
    pub fn upload_training_data(
        &mut self,
        data: &[f32],
        labels: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Result<(), String> {
        self.n_samples = n_samples;
        self.n_features = n_features;
        
        // Create and upload data buffer
        let data_size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let data_buffer = self.executor.create_buffer(
            data_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(data_buffer, 0, cast_slice(data))?;
        self.data_buffer = Some(data_buffer);
        
        // Create and upload labels buffer
        let labels_size = (labels.len() * std::mem::size_of::<f32>()) as u64;
        let labels_buffer = self.executor.create_buffer(
            labels_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(labels_buffer, 0, cast_slice(labels))?;
        self.labels_buffer = Some(labels_buffer);
        
        eprintln!("[GpuTrainer] Uploaded {} samples x {} features to GPU", n_samples, n_features);
        
        Ok(())
    }
    
    /// Generate bootstrap sample indices on GPU
    pub fn bootstrap_sample(&self, n_samples: usize, seed: u32) -> Result<Vec<u32>, String> {
        // Create output buffer
        let output_size = (n_samples * std::mem::size_of::<u32>()) as u64;
        let output_buffer = self.executor.create_buffer(
            output_size,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )?;
        
        // Call kernel
        let params = BootstrapParams {
            n_samples: n_samples as u32,
            seed,
            max_index: self.n_samples as u32,
        };
        
        kernel_bootstrap_sample(params, output_buffer)
            .map_err(|e| format!("Bootstrap kernel failed: {:?}", e))?;
        
        // Read results
        let result_bytes = self.executor.read_buffer(output_buffer, 0, output_size as u32)?;
        let indices: Vec<u32> = cast_slice(&result_bytes).to_vec();
        
        // Cleanup
        self.executor.destroy_buffer(output_buffer)?;
        
        Ok(indices)
    }
    
    /// Find the best split for a feature on GPU
    pub fn find_best_split(
        &self,
        indices: &[u32],
        feature_idx: usize,
        thresholds: &[f32],
    ) -> Result<(f32, f32), String> {
        let data_buffer = self.data_buffer
            .ok_or("Training data not uploaded")?;
        let labels_buffer = self.labels_buffer
            .ok_or("Labels not uploaded")?;
        
        let n_thresholds = thresholds.len();
        
        // Create indices buffer
        let indices_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
        let indices_buffer = self.executor.create_buffer(
            indices_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(indices_buffer, 0, cast_slice(indices))?;
        
        // Create thresholds buffer
        let thresholds_size = (n_thresholds * std::mem::size_of::<f32>()) as u64;
        let thresholds_buffer = self.executor.create_buffer(
            thresholds_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(thresholds_buffer, 0, cast_slice(thresholds))?;
        
        // Create output scores buffer
        let scores_size = (n_thresholds * std::mem::size_of::<f32>()) as u64;
        let scores_buffer = self.executor.create_buffer(
            scores_size,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )?;
        
        // Call kernel
        let params = FindSplitParams {
            n_samples: indices.len() as u32,
            n_features: self.n_features as u32,
            feature_idx: feature_idx as u32,
            n_thresholds: n_thresholds as u32,
        };
        
        kernel_find_split(
            params,
            data_buffer,
            labels_buffer,
            indices_buffer,
            thresholds_buffer,
            scores_buffer,
        ).map_err(|e| format!("Find split kernel failed: {:?}", e))?;
        
        // Read scores
        let scores_bytes = self.executor.read_buffer(scores_buffer, 0, scores_size as u32)?;
        let scores: Vec<f32> = cast_slice(&scores_bytes).to_vec();
        
        // Find best threshold
        let (best_idx, &best_score) = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or("No valid split found")?;
        
        let best_threshold = thresholds[best_idx];
        
        // Cleanup temporary buffers
        self.executor.destroy_buffer(indices_buffer)?;
        self.executor.destroy_buffer(thresholds_buffer)?;
        self.executor.destroy_buffer(scores_buffer)?;
        
        Ok((best_threshold, best_score))
    }
    
    /// Cleanup GPU resources
    pub fn cleanup(&mut self) -> Result<(), String> {
        if let Some(buf) = self.data_buffer.take() {
            self.executor.destroy_buffer(buf)?;
        }
        if let Some(buf) = self.labels_buffer.take() {
            self.executor.destroy_buffer(buf)?;
        }
        Ok(())
    }
}

impl Drop for GpuTrainer {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// GPU-accelerated inference
pub struct GpuPredictor {
    executor: GpuExecutor,
}

impl GpuPredictor {
    pub fn new() -> Result<Self, String> {
        let executor = GpuExecutor::new()?;
        Ok(Self { executor })
    }
    
    /// Average tree predictions on GPU
    pub fn average_predictions(
        &self,
        tree_predictions: &[f32],
        n_samples: usize,
        n_trees: usize,
    ) -> Result<Vec<f32>, String> {
        // Create input buffer
        let input_size = (tree_predictions.len() * std::mem::size_of::<f32>()) as u64;
        let input_buffer = self.executor.create_buffer(
            input_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(input_buffer, 0, cast_slice(tree_predictions))?;
        
        // Create output buffer
        let output_size = (n_samples * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.executor.create_buffer(
            output_size,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )?;
        
        // Call kernel
        let params = AverageParams {
            n_trees: n_trees as u32,
            n_samples: n_samples as u32,
        };
        
        kernel_average(params, input_buffer, output_buffer)
            .map_err(|e| format!("Average kernel failed: {:?}", e))?;
        
        // Read results
        let result_bytes = self.executor.read_buffer(output_buffer, 0, output_size as u32)?;
        let predictions: Vec<f32> = cast_slice(&result_bytes).to_vec();
        
        // Cleanup
        self.executor.destroy_buffer(input_buffer)?;
        self.executor.destroy_buffer(output_buffer)?;
        
        Ok(predictions)
    }
    
    /// Matrix multiplication on GPU (for future neural network support)
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        // Create buffers
        let a_size = (a.len() * std::mem::size_of::<f32>()) as u64;
        let b_size = (b.len() * std::mem::size_of::<f32>()) as u64;
        let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
        
        let a_buffer = self.executor.create_buffer(
            a_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(a_buffer, 0, cast_slice(a))?;
        
        let b_buffer = self.executor.create_buffer(
            b_size,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        self.executor.write_buffer(b_buffer, 0, cast_slice(b))?;
        
        let c_buffer = self.executor.create_buffer(
            c_size,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        )?;
        // Zero-initialize C
        let zeros = vec![0u8; c_size as usize];
        self.executor.write_buffer(c_buffer, 0, &zeros)?;
        
        // Call kernel
        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 0.0,
        };
        
        kernel_matmul(params, a_buffer, b_buffer, c_buffer)
            .map_err(|e| format!("Matmul kernel failed: {:?}", e))?;
        
        // Read results
        let result_bytes = self.executor.read_buffer(c_buffer, 0, c_size as u32)?;
        let c: Vec<f32> = cast_slice(&result_bytes).to_vec();
        
        // Cleanup
        self.executor.destroy_buffer(a_buffer)?;
        self.executor.destroy_buffer(b_buffer)?;
        self.executor.destroy_buffer(c_buffer)?;
        
        Ok(c)
    }
}
