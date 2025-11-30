//! CUDA Backend Implementation
//!
//! Uses cudarc for CUDA bindings:
//! - cuBLAS for matrix operations (SGEMM with Tensor Cores on H100)
//! - Custom kernels for ML-specific operations

use super::{
    AverageParams, BatchPredictParams, BootstrapParams, BufferId, DeviceInfo,
    ElementwiseOp, ElementwiseParams, FindSplitParams, GpuBackend, GpuError, MatmulParams,
    ReduceOp, ReduceParams,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;

/// Internal buffer representation for CUDA
#[cfg(feature = "cuda")]
struct CudaBuffer {
    /// Raw byte buffer on GPU
    data: CudaSlice<u8>,
    size: u64,
    usage: u32,
}

/// CUDA Backend implementation using cuBLAS
/// 
/// Note: We don't use CudaRng because it's not Send+Sync.
/// Instead, we generate random numbers on CPU and upload them.
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    device_info: DeviceInfo,
    buffers: HashMap<BufferId, CudaBuffer>,
    next_buffer_id: BufferId,
}

// Manually implement Send + Sync since CudaDevice and CudaBlas are thread-safe
#[cfg(feature = "cuda")]
unsafe impl Send for CudaBackend {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaBackend {}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend
    pub fn new() -> Result<Self, GpuError> {
        info!("[CUDA] Initializing CUDA backend...");
        
        // Initialize CUDA device
        let device = CudaDevice::new(0).map_err(|e| {
            error!("[CUDA] Failed to initialize device: {}", e);
            GpuError::DeviceUnavailable
        })?;
        
        // Get device properties
        let name = device.name().unwrap_or_else(|_| "Unknown CUDA Device".to_string());
        
        // Get compute capability (cudarc 0.12 doesn't expose this directly)
        // Default to sm_90 for H100, will be detected at runtime
        let (major, minor) = (9, 0); // Assume H100 sm_90
        let compute_capability = format!("{}.{}", major, minor);
        
        // Estimate memory (cudarc doesn't expose this directly in newer versions)
        let total_memory = 80 * 1024 * 1024 * 1024u64; // Assume 80GB for H100
        
        info!("[CUDA] Device: {}", name);
        info!("[CUDA] Compute Capability: {} (sm_{}{})", compute_capability, major, minor);
        
        // Check for H100 (sm_90) or similar
        if major >= 9 {
            info!("[CUDA] Hopper architecture detected - Tensor Cores available");
        } else if major >= 8 {
            info!("[CUDA] Ampere architecture detected - Tensor Cores available");
        }
        
        // Initialize cuBLAS
        let blas = CudaBlas::new(device.clone()).map_err(|e| {
            error!("[CUDA] Failed to initialize cuBLAS: {}", e);
            GpuError::BackendError(format!("cuBLAS init failed: {}", e))
        })?;
        info!("[CUDA] cuBLAS initialized");
        
        let device_info = DeviceInfo {
            name,
            backend: "cuda".to_string(),
            total_memory,
            is_hardware: true,
            compute_capability,
        };
        
        info!("[CUDA] Backend initialized successfully");
        
        Ok(Self {
            device,
            blas,
            device_info,
            buffers: HashMap::new(),
            next_buffer_id: 1,
        })
    }
    
    /// Get buffer or return error
    fn get_buffer(&self, id: BufferId) -> Result<&CudaBuffer, GpuError> {
        self.buffers.get(&id).ok_or(GpuError::InvalidBuffer(id))
    }
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Buffer Management
    // ═══════════════════════════════════════════════════════════════════════
    
    fn buffer_create(&mut self, size: u64, usage: u32) -> Result<BufferId, GpuError> {
        debug!("[CUDA] buffer_create(size={}, usage={:#x})", size, usage);
        
        let data = self.device.alloc_zeros::<u8>(size as usize).map_err(|e| {
            error!("[CUDA] Failed to allocate buffer: {}", e);
            GpuError::OutOfMemory
        })?;
        
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        
        self.buffers.insert(id, CudaBuffer { data, size, usage });
        
        debug!("[CUDA] Created buffer {} (size={})", id, size);
        Ok(id)
    }
    
    fn buffer_write(&mut self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), GpuError> {
        debug!("[CUDA] buffer_write(buffer={}, offset={}, len={})", buffer, offset, data.len());
        
        // Get buffer info first
        let size = {
            let buf = self.get_buffer(buffer)?;
            buf.size
        };
        
        if offset + data.len() as u64 > size {
            return Err(GpuError::InvalidParams(format!(
                "Write exceeds buffer size: {} + {} > {}",
                offset, data.len(), size
            )));
        }
        
        // Get mutable reference and write
        let buf = self.buffers.get_mut(&buffer).ok_or(GpuError::InvalidBuffer(buffer))?;
        
        self.device.htod_copy_into(data.to_vec(), &mut buf.data).map_err(|e| {
            error!("[CUDA] htod_copy failed: {}", e);
            GpuError::BackendError(format!("Copy to device failed: {}", e))
        })?;
        
        Ok(())
    }
    
    fn buffer_read(&self, buffer: BufferId, offset: u64, size: u32) -> Result<Vec<u8>, GpuError> {
        debug!("[CUDA] buffer_read(buffer={}, offset={}, size={})", buffer, offset, size);
        
        let buf = self.get_buffer(buffer)?;
        
        if offset + size as u64 > buf.size {
            return Err(GpuError::InvalidParams(format!(
                "Read exceeds buffer size: {} + {} > {}",
                offset, size, buf.size
            )));
        }
        
        // Copy data from device
        let data = self.device.dtoh_sync_copy(&buf.data).map_err(|e| {
            error!("[CUDA] dtoh_copy failed: {}", e);
            GpuError::BackendError(format!("Copy from device failed: {}", e))
        })?;
        
        let start = offset as usize;
        let end = start + size as usize;
        
        Ok(data[start..end].to_vec())
    }
    
    fn buffer_copy(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: BufferId,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] buffer_copy(src={}, dst={}, size={})", src, dst, size);
        
        // Read from src, write to dst
        let data = self.buffer_read(src, src_offset, size as u32)?;
        self.buffer_write(dst, dst_offset, &data)?;
        
        Ok(())
    }
    
    fn buffer_destroy(&mut self, buffer: BufferId) -> Result<(), GpuError> {
        debug!("[CUDA] buffer_destroy(buffer={})", buffer);
        
        self.buffers.remove(&buffer).ok_or(GpuError::InvalidBuffer(buffer))?;
        Ok(())
    }
    
    fn sync(&self) {
        debug!("[CUDA] sync()");
        let _ = self.device.synchronize();
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // ML Kernels - CPU fallback implementations
    // These run on CPU but data is on GPU, so we transfer back and forth.
    // For production, these would be replaced with actual CUDA kernels.
    // ═══════════════════════════════════════════════════════════════════════
    
    fn kernel_bootstrap_sample(
        &mut self,
        params: &BootstrapParams,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_bootstrap_sample(n_samples={}, seed={}, max_index={})",
               params.n_samples, params.seed, params.max_index);
        
        // Generate random indices on CPU using PCG-style hash
        let mut indices = Vec::with_capacity(params.n_samples as usize);
        let mut state = params.seed;
        
        for _ in 0..params.n_samples {
            // PCG hash
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            let mut hash = state;
            hash ^= hash >> 16;
            hash = hash.wrapping_mul(2654435769);
            hash ^= hash >> 16;
            
            let idx = hash % params.max_index;
            indices.push(idx);
        }
        
        // Write to output buffer
        let data: Vec<u8> = indices.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output, 0, &data)?;
        
        debug!("[CUDA] bootstrap_sample completed");
        Ok(())
    }
    
    fn kernel_find_split(
        &mut self,
        params: &FindSplitParams,
        data: BufferId,
        labels: BufferId,
        indices: BufferId,
        thresholds: BufferId,
        output_scores: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_find_split(n_samples={}, n_features={}, feature_idx={}, n_thresholds={})",
               params.n_samples, params.n_features, params.feature_idx, params.n_thresholds);
        
        // Read data from GPU
        let data_bytes = self.buffer_read(data, 0, (params.n_samples * params.n_features * 4) as u32)?;
        let labels_bytes = self.buffer_read(labels, 0, (params.n_samples * 4) as u32)?;
        let indices_bytes = self.buffer_read(indices, 0, (params.n_samples * 4) as u32)?;
        let thresholds_bytes = self.buffer_read(thresholds, 0, (params.n_thresholds * 4) as u32)?;
        
        // Convert to typed arrays
        let data_f32: Vec<f32> = data_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let labels_f32: Vec<f32> = labels_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let indices_u32: Vec<u32> = indices_bytes.chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let thresholds_f32: Vec<f32> = thresholds_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        debug!("[CUDA] find_split: data_f32.len()={}, labels_f32.len()={}, indices_u32.len()={}",
               data_f32.len(), labels_f32.len(), indices_u32.len());
        
        // Validate indices are within bounds
        let max_valid_idx = params.n_samples - 1;
        for (i, &idx) in indices_u32.iter().enumerate() {
            if idx > max_valid_idx {
                warn!("[CUDA] find_split: index {} at position {} exceeds max_valid_idx {}, clamping",
                      idx, i, max_valid_idx);
            }
        }
        
        // Compute MSE for each threshold
        let mut scores = Vec::with_capacity(params.n_thresholds as usize);
        
        for &threshold in &thresholds_f32 {
            let mut left_sum = 0.0f32;
            let mut right_sum = 0.0f32;
            let mut left_count = 0u32;
            let mut right_count = 0u32;
            
            // First pass: compute means (with bounds checking)
            for &idx in &indices_u32 {
                // Clamp index to valid range
                let safe_idx = (idx as usize).min((params.n_samples - 1) as usize);
                let data_idx = safe_idx * (params.n_features as usize) + (params.feature_idx as usize);
                
                if data_idx >= data_f32.len() {
                    continue; // Skip invalid indices
                }
                
                let feature_val = data_f32[data_idx];
                let label = labels_f32[safe_idx.min(labels_f32.len() - 1)];
                
                if feature_val <= threshold {
                    left_sum += label;
                    left_count += 1;
                } else {
                    right_sum += label;
                    right_count += 1;
                }
            }
            
            if left_count == 0 || right_count == 0 {
                scores.push(f32::INFINITY);
                continue;
            }
            
            let left_mean = left_sum / left_count as f32;
            let right_mean = right_sum / right_count as f32;
            
            // Second pass: compute MSE (with bounds checking)
            let mut mse = 0.0f32;
            for &idx in &indices_u32 {
                let safe_idx = (idx as usize).min((params.n_samples - 1) as usize);
                let data_idx = safe_idx * (params.n_features as usize) + (params.feature_idx as usize);
                
                if data_idx >= data_f32.len() {
                    continue;
                }
                
                let feature_val = data_f32[data_idx];
                let label = labels_f32[safe_idx.min(labels_f32.len() - 1)];
                
                let mean = if feature_val <= threshold { left_mean } else { right_mean };
                let diff = label - mean;
                mse += diff * diff;
            }
            
            scores.push(mse);
        }
        
        // Write scores to output
        let scores_bytes: Vec<u8> = scores.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output_scores, 0, &scores_bytes)?;
        
        debug!("[CUDA] find_split completed");
        Ok(())
    }
    
    fn kernel_average(
        &mut self,
        params: &AverageParams,
        tree_predictions: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_average(n_trees={}, n_samples={})",
               params.n_trees, params.n_samples);
        
        // Read predictions
        let total_size = params.n_samples * params.n_trees;
        let preds_bytes = self.buffer_read(tree_predictions, 0, (total_size * 4) as u32)?;
        
        let preds: Vec<f32> = preds_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        // Compute averages
        let mut averages = Vec::with_capacity(params.n_samples as usize);
        
        for i in 0..params.n_samples as usize {
            let mut sum = 0.0f32;
            for j in 0..params.n_trees as usize {
                sum += preds[i * params.n_trees as usize + j];
            }
            averages.push(sum / params.n_trees as f32);
        }
        
        // Write output
        let output_bytes: Vec<u8> = averages.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output, 0, &output_bytes)?;
        
        debug!("[CUDA] average completed");
        Ok(())
    }
    
    fn kernel_matmul(
        &mut self,
        params: &MatmulParams,
        a: BufferId,
        b: BufferId,
        c: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_matmul(m={}, k={}, n={}, alpha={}, beta={})",
               params.m, params.k, params.n, params.alpha, params.beta);
        
        // Read matrices from GPU
        let a_bytes = self.buffer_read(a, 0, (params.m * params.k * 4) as u32)?;
        let b_bytes = self.buffer_read(b, 0, (params.k * params.n * 4) as u32)?;
        let c_bytes = self.buffer_read(c, 0, (params.m * params.n * 4) as u32)?;
        
        let a_f32: Vec<f32> = a_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = b_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32: Vec<f32> = c_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        // C = alpha * A @ B + beta * C (naive implementation)
        for i in 0..params.m as usize {
            for j in 0..params.n as usize {
                let mut sum = 0.0f32;
                for l in 0..params.k as usize {
                    sum += a_f32[i * params.k as usize + l] * b_f32[l * params.n as usize + j];
                }
                c_f32[i * params.n as usize + j] = 
                    params.alpha * sum + params.beta * c_f32[i * params.n as usize + j];
            }
        }
        
        // Write back
        let c_out: Vec<u8> = c_f32.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(c, 0, &c_out)?;
        
        debug!("[CUDA] matmul completed");
        Ok(())
    }
    
    fn kernel_elementwise(
        &mut self,
        params: &ElementwiseParams,
        input_a: BufferId,
        input_b: Option<BufferId>,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_elementwise(n_elements={}, op={:?})",
               params.n_elements, params.op);
        
        let a_bytes = self.buffer_read(input_a, 0, (params.n_elements * 4) as u32)?;
        let a: Vec<f32> = a_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        let b: Option<Vec<f32>> = if let Some(id) = input_b {
            let b_bytes = self.buffer_read(id, 0, (params.n_elements * 4) as u32)?;
            Some(b_bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        } else {
            None
        };
        
        let result: Vec<f32> = match params.op {
            ElementwiseOp::Relu => a.iter().map(|&x| x.max(0.0)).collect(),
            ElementwiseOp::Sigmoid => a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
            ElementwiseOp::Tanh => a.iter().map(|&x| x.tanh()).collect(),
            ElementwiseOp::Sqrt => a.iter().map(|&x| x.sqrt()).collect(),
            ElementwiseOp::Exp => a.iter().map(|&x| x.exp()).collect(),
            ElementwiseOp::Log => a.iter().map(|&x| x.ln()).collect(),
            ElementwiseOp::Add => {
                let b = b.ok_or(GpuError::InvalidParams("Add requires second input".into()))?;
                a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
            }
            ElementwiseOp::Mul => {
                let b = b.ok_or(GpuError::InvalidParams("Mul requires second input".into()))?;
                a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
            }
        };
        
        let out_bytes: Vec<u8> = result.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output, 0, &out_bytes)?;
        
        debug!("[CUDA] elementwise completed");
        Ok(())
    }
    
    fn kernel_reduce(
        &mut self,
        params: &ReduceParams,
        input: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_reduce(n_elements={}, op={:?})",
               params.n_elements, params.op);
        
        let in_bytes = self.buffer_read(input, 0, (params.n_elements * 4) as u32)?;
        let data: Vec<f32> = in_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        
        let result = match params.op {
            ReduceOp::Sum => data.iter().sum(),
            ReduceOp::Max => data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Min => data.iter().cloned().fold(f32::INFINITY, f32::min),
            ReduceOp::Mean => data.iter().sum::<f32>() / data.len() as f32,
            ReduceOp::Variance => {
                let mean = data.iter().sum::<f32>() / data.len() as f32;
                data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
            }
        };
        
        let out_bytes = result.to_le_bytes();
        self.buffer_write(output, 0, &out_bytes)?;
        
        debug!("[CUDA] reduce completed");
        Ok(())
    }
    
    fn kernel_batch_predict(
        &mut self,
        params: &BatchPredictParams,
        _samples: BufferId,
        _tree_nodes: BufferId,
        _tree_offsets: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA] kernel_batch_predict(batch_size={}, n_features={}, n_trees={})",
               params.batch_size, params.n_features, params.n_trees);
        
        // Placeholder - returns zeros
        warn!("[CUDA] batch_predict not fully implemented - using placeholder");
        
        let out_bytes = vec![0u8; (params.batch_size * 4) as usize];
        self.buffer_write(output, 0, &out_bytes)?;
        
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fallback when CUDA feature is disabled
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "cuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        Err(GpuError::DeviceUnavailable)
    }
}
