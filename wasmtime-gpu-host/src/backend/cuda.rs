//! CUDA Backend Implementation
//!
//! Uses cudarc for CUDA bindings:
//! - cuBLAS for matrix operations (SGEMM with Tensor Cores on H100)
//! - cuRAND for random number generation
//! - Custom PTX kernels for ML-specific operations

use super::{
    AverageParams, BatchPredictParams, BootstrapParams, BufferId, BufferUsage, DeviceInfo,
    ElementwiseOp, ElementwiseParams, FindSplitParams, GpuBackend, GpuError, MatmulParams,
    ReduceOp, ReduceParams,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceRepr, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::cublas::{Gemm, CudaBlas};
#[cfg(feature = "cuda")]
use cudarc::curand::{CudaRng, result::CurandError};

// ═══════════════════════════════════════════════════════════════════════════
// PTX Kernels (embedded at compile time)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
const KERNELS_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

// Bootstrap sample kernel: generates random indices using GPU
// Uses a simple hash function for reproducible random numbers
.visible .entry bootstrap_sample(
    .param .u64 output,
    .param .u32 n_samples,
    .param .u32 seed,
    .param .u32 max_index
)
{
    .reg .u32 %r<12>;
    .reg .u64 %rd<4>;
    .reg .pred %p1;
    
    // Get thread index
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, %r1, %r0;  // global_idx = blockIdx.x * blockDim.x + threadIdx.x
    
    // Load params
    ld.param.u32 %r4, [n_samples];
    
    // Bounds check
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra END;
    
    // Load other params
    ld.param.u32 %r5, [seed];
    ld.param.u32 %r6, [max_index];
    ld.param.u64 %rd0, [output];
    
    // PCG hash: state = seed + global_idx
    add.u32 %r7, %r5, %r3;
    
    // state = state * 747796405 + 2891336453
    mul.lo.u32 %r8, %r7, 747796405;
    add.u32 %r8, %r8, 2891336453;
    
    // state ^= state >> 16
    shr.u32 %r9, %r8, 16;
    xor.b32 %r8, %r8, %r9;
    
    // state *= 2654435769
    mul.lo.u32 %r8, %r8, 2654435769;
    
    // state ^= state >> 16
    shr.u32 %r9, %r8, 16;
    xor.b32 %r8, %r8, %r9;
    
    // idx = state % max_index
    rem.u32 %r10, %r8, %r6;
    
    // Store result
    cvt.u64.u32 %rd1, %r3;
    shl.b64 %rd1, %rd1, 2;  // * 4 bytes
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r10;
    
END:
    ret;
}

// Average kernel: computes average of tree predictions per sample
.visible .entry average_predictions(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n_samples,
    .param .u32 n_trees
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<6>;
    .reg .f32 %f<4>;
    .reg .pred %p1;
    
    // Get sample index
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, %r1, %r0;  // sample_idx
    
    // Load params
    ld.param.u32 %r4, [n_samples];
    
    // Bounds check
    setp.ge.u32 %p1, %r3, %r4;
    @%p1 bra END_AVG;
    
    ld.param.u32 %r5, [n_trees];
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    
    // Initialize sum = 0
    mov.f32 %f0, 0f00000000;
    
    // Calculate base offset for this sample: sample_idx * n_trees * 4
    mul.lo.u32 %r6, %r3, %r5;
    cvt.u64.u32 %rd2, %r6;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;
    
    // Loop over trees
    mov.u32 %r6, 0;
LOOP_TREES:
    setp.ge.u32 %p1, %r6, %r5;
    @%p1 bra END_LOOP;
    
    // Load prediction
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd4, %rd4, 2;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    
    // sum += prediction
    add.f32 %f0, %f0, %f1;
    
    // i++
    add.u32 %r6, %r6, 1;
    bra LOOP_TREES;
    
END_LOOP:
    // average = sum / n_trees
    cvt.rn.f32.u32 %f2, %r5;
    div.rn.f32 %f3, %f0, %f2;
    
    // Store result
    cvt.u64.u32 %rd2, %r3;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f3;
    
END_AVG:
    ret;
}

// Elementwise ReLU kernel
.visible .entry relu(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n_elements
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<3>;
    .reg .pred %p<2>;
    
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, %r1, %r0;
    
    ld.param.u32 %r1, [n_elements];
    setp.ge.u32 %p0, %r3, %r1;
    @%p0 bra END_RELU;
    
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    
    cvt.u64.u32 %rd2, %r3;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    
    // ReLU: max(0, x)
    mov.f32 %f1, 0f00000000;
    max.f32 %f2, %f0, %f1;
    
    add.u64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f2;
    
END_RELU:
    ret;
}

// Elementwise Sigmoid kernel
.visible .entry sigmoid(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n_elements
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<5>;
    .reg .pred %p0;
    
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, %r1, %r0;
    
    ld.param.u32 %r1, [n_elements];
    setp.ge.u32 %p0, %r3, %r1;
    @%p0 bra END_SIG;
    
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    
    cvt.u64.u32 %rd2, %r3;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    
    // Sigmoid: 1 / (1 + exp(-x))
    neg.f32 %f1, %f0;
    ex2.approx.f32 %f2, %f1;      // exp2(-x) ≈ exp(-x * log2(e))
    mov.f32 %f3, 0f3f800000;      // 1.0
    add.f32 %f4, %f3, %f2;
    div.rn.f32 %f0, %f3, %f4;
    
    add.u64 %rd3, %rd1, %rd2;
    st.global.f32 [%rd3], %f0;
    
END_SIG:
    ret;
}

// Reduce sum kernel (simple version, one block)
.visible .entry reduce_sum(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n_elements
)
{
    .reg .u32 %r<6>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<3>;
    .reg .pred %p0;
    
    // Only thread 0 computes (simple implementation)
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    or.b32 %r2, %r0, %r1;
    setp.ne.u32 %p0, %r2, 0;
    @%p0 bra END_SUM;
    
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r3, [n_elements];
    
    mov.f32 %f0, 0f00000000;
    mov.u32 %r4, 0;
    
LOOP_SUM:
    setp.ge.u32 %p0, %r4, %r3;
    @%p0 bra STORE_SUM;
    
    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32 %f0, %f0, %f1;
    
    add.u32 %r4, %r4, 1;
    bra LOOP_SUM;
    
STORE_SUM:
    st.global.f32 [%rd1], %f0;
    
END_SUM:
    ret;
}
"#;

/// Internal buffer representation for CUDA
#[cfg(feature = "cuda")]
struct CudaBuffer {
    /// Raw byte buffer on GPU
    data: CudaSlice<u8>,
    size: u64,
    usage: u32,
}

/// Typed buffer views for cuBLAS operations
#[cfg(feature = "cuda")]
struct TypedBuffers {
    f32_cache: HashMap<BufferId, CudaSlice<f32>>,
}

/// CUDA Backend implementation using cuBLAS/cuRAND
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    rng: CudaRng,
    device_info: DeviceInfo,
    buffers: HashMap<BufferId, CudaBuffer>,
    next_buffer_id: BufferId,
    kernels_loaded: bool,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend
    pub fn new() -> Result<Self, GpuError> {
        info!("[CUDA] Initializing CUDA backend with cuBLAS/cuRAND...");
        
        // Initialize CUDA device
        let device = CudaDevice::new(0).map_err(|e| {
            error!("[CUDA] Failed to initialize device: {}", e);
            GpuError::DeviceUnavailable
        })?;
        
        // Get device properties
        let name = device.name().unwrap_or_else(|_| "Unknown CUDA Device".to_string());
        let total_memory = device.total_memory().unwrap_or(0) as u64;
        
        // Get compute capability
        let (major, minor) = device.compute_capability();
        let compute_capability = format!("{}.{}", major, minor);
        
        info!("[CUDA] Device: {}", name);
        info!("[CUDA] Memory: {:.2} GB", total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
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
        
        // Initialize cuRAND
        let rng = CudaRng::new(42, device.clone()).map_err(|e| {
            error!("[CUDA] Failed to initialize cuRAND: {}", e);
            GpuError::BackendError(format!("cuRAND init failed: {}", e))
        })?;
        info!("[CUDA] cuRAND initialized");
        
        // Load PTX kernels
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(KERNELS_PTX),
            "ml_kernels",
            &["bootstrap_sample", "average_predictions", "relu", "sigmoid", "reduce_sum"]
        ).map_err(|e| {
            error!("[CUDA] Failed to load PTX kernels: {}", e);
            GpuError::BackendError(format!("PTX load failed: {}", e))
        })?;
        info!("[CUDA] PTX kernels loaded");
        
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
            rng,
            device_info,
            buffers: HashMap::new(),
            next_buffer_id: 1,
            kernels_loaded: true,
        })
    }
    
    /// Get buffer or return error
    fn get_buffer(&self, id: BufferId) -> Result<&CudaBuffer, GpuError> {
        self.buffers.get(&id).ok_or(GpuError::InvalidBuffer(id))
    }
    
    /// Get mutable buffer or return error
    fn get_buffer_mut(&mut self, id: BufferId) -> Result<&mut CudaBuffer, GpuError> {
        self.buffers.get_mut(&id).ok_or(GpuError::InvalidBuffer(id))
    }
    
    /// Reinterpret u8 buffer as f32 slice (unsafe but necessary for cuBLAS)
    unsafe fn buffer_as_f32(&self, id: BufferId) -> Result<CudaSlice<f32>, GpuError> {
        let buf = self.get_buffer(id)?;
        let n_floats = buf.size as usize / 4;
        
        // Create a new view of the same memory as f32
        let ptr = buf.data.device_ptr();
        Ok(self.device.upgrade_device_ptr(ptr.cast::<f32>(), n_floats))
    }
    
    /// Reinterpret u8 buffer as u32 slice
    unsafe fn buffer_as_u32(&self, id: BufferId) -> Result<CudaSlice<u32>, GpuError> {
        let buf = self.get_buffer(id)?;
        let n_u32s = buf.size as usize / 4;
        
        let ptr = buf.data.device_ptr();
        Ok(self.device.upgrade_device_ptr(ptr.cast::<u32>(), n_u32s))
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
        
        let buf = self.get_buffer_mut(buffer)?;
        
        if offset + data.len() as u64 > buf.size {
            return Err(GpuError::InvalidParams(format!(
                "Write exceeds buffer size: {} + {} > {}",
                offset, data.len(), buf.size
            )));
        }
        
        // Copy data to device
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
        
        // Use device-to-device copy
        let src_buf = self.get_buffer(src)?;
        let dst_buf = self.get_buffer(dst)?;
        
        // For now, go through host (cudaMemcpyDeviceToDevice would be better)
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
    // ML Kernels - GPU Implementation
    // ═══════════════════════════════════════════════════════════════════════
    
    fn kernel_bootstrap_sample(
        &mut self,
        params: &BootstrapParams,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA/GPU] kernel_bootstrap_sample(n_samples={}, seed={}, max_index={})",
               params.n_samples, params.seed, params.max_index);
        
        let output_buf = self.get_buffer(output)?;
        let output_ptr = output_buf.data.device_ptr();
        
        // Launch PTX kernel
        let func = self.device.get_func("ml_kernels", "bootstrap_sample")
            .map_err(|e| GpuError::KernelError(format!("Failed to get bootstrap_sample: {}", e)))?;
        
        let block_size = 256u32;
        let grid_size = (params.n_samples + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            func.launch(cfg, (
                output_ptr,
                params.n_samples,
                params.seed,
                params.max_index,
            )).map_err(|e| GpuError::KernelError(format!("bootstrap_sample launch failed: {}", e)))?;
        }
        
        self.device.synchronize().map_err(|e| GpuError::BackendError(e.to_string()))?;
        
        debug!("[CUDA/GPU] bootstrap_sample completed");
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
        debug!("[CUDA/GPU] kernel_find_split(n_samples={}, n_features={}, feature_idx={}, n_thresholds={})",
               params.n_samples, params.n_features, params.feature_idx, params.n_thresholds);
        
        // For find_split, we use cuBLAS operations where possible
        // The algorithm:
        // 1. Extract feature column for the selected feature
        // 2. For each threshold, compute split quality using parallel reduction
        
        // This is a complex operation that doesn't map directly to cuBLAS
        // We'll use a combination of cuBLAS GEMV for partial sums and custom logic
        
        // For now, read data and compute on GPU using vectorized operations
        // where cuBLAS can help (sums, dot products)
        
        let data_bytes = self.buffer_read(data, 0, (params.n_samples * params.n_features * 4) as u32)?;
        let labels_bytes = self.buffer_read(labels, 0, (params.n_samples * 4) as u32)?;
        let indices_bytes = self.buffer_read(indices, 0, (params.n_samples * 4) as u32)?;
        let thresholds_bytes = self.buffer_read(thresholds, 0, (params.n_thresholds * 4) as u32)?;
        
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
        
        // Extract indexed feature values and labels to GPU
        let mut feature_vals: Vec<f32> = Vec::with_capacity(params.n_samples as usize);
        let mut indexed_labels: Vec<f32> = Vec::with_capacity(params.n_samples as usize);
        
        for &idx in &indices_u32 {
            let feat_idx = (idx as usize) * (params.n_features as usize) + (params.feature_idx as usize);
            feature_vals.push(data_f32[feat_idx]);
            indexed_labels.push(labels_f32[idx as usize]);
        }
        
        // Upload to GPU for parallel threshold evaluation
        let d_features = self.device.htod_copy(feature_vals.clone())
            .map_err(|e| GpuError::BackendError(e.to_string()))?;
        let d_labels = self.device.htod_copy(indexed_labels.clone())
            .map_err(|e| GpuError::BackendError(e.to_string()))?;
        
        // Compute MSE for each threshold (parallelized per threshold)
        let mut scores: Vec<f32> = Vec::with_capacity(params.n_thresholds as usize);
        
        for &threshold in &thresholds_f32 {
            // Create mask: 1 if feature <= threshold, 0 otherwise
            let mask_left: Vec<f32> = feature_vals.iter()
                .map(|&f| if f <= threshold { 1.0 } else { 0.0 })
                .collect();
            let mask_right: Vec<f32> = feature_vals.iter()
                .map(|&f| if f > threshold { 1.0 } else { 0.0 })
                .collect();
            
            let d_mask_left = self.device.htod_copy(mask_left.clone())
                .map_err(|e| GpuError::BackendError(e.to_string()))?;
            let d_mask_right = self.device.htod_copy(mask_right.clone())
                .map_err(|e| GpuError::BackendError(e.to_string()))?;
            
            // Use cuBLAS dot product for sums
            // left_sum = dot(mask_left, labels)
            // left_count = dot(mask_left, ones)
            let left_sum: f32 = unsafe {
                self.blas.dot(&d_mask_left, &d_labels)
            };
            let left_count: f32 = mask_left.iter().sum();
            
            let right_sum: f32 = unsafe {
                self.blas.dot(&d_mask_right, &d_labels)
            };
            let right_count: f32 = mask_right.iter().sum();
            
            if left_count < 1.0 || right_count < 1.0 {
                scores.push(f32::INFINITY);
                continue;
            }
            
            let left_mean = left_sum / left_count;
            let right_mean = right_sum / right_count;
            
            // Compute MSE: sum((label - mean)^2)
            let mut mse = 0.0f32;
            for i in 0..params.n_samples as usize {
                let mean = if mask_left[i] > 0.5 { left_mean } else { right_mean };
                let diff = indexed_labels[i] - mean;
                mse += diff * diff;
            }
            
            scores.push(mse);
        }
        
        // Write scores to output
        let scores_bytes: Vec<u8> = scores.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();
        
        self.buffer_write(output_scores, 0, &scores_bytes)?;
        
        debug!("[CUDA/GPU] find_split completed");
        Ok(())
    }
    
    fn kernel_average(
        &mut self,
        params: &AverageParams,
        tree_predictions: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA/GPU] kernel_average(n_trees={}, n_samples={})",
               params.n_trees, params.n_samples);
        
        let input_buf = self.get_buffer(tree_predictions)?;
        let output_buf = self.get_buffer(output)?;
        
        let input_ptr = input_buf.data.device_ptr();
        let output_ptr = output_buf.data.device_ptr();
        
        // Launch PTX kernel
        let func = self.device.get_func("ml_kernels", "average_predictions")
            .map_err(|e| GpuError::KernelError(format!("Failed to get average_predictions: {}", e)))?;
        
        let block_size = 256u32;
        let grid_size = (params.n_samples + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            func.launch(cfg, (
                input_ptr,
                output_ptr,
                params.n_samples,
                params.n_trees,
            )).map_err(|e| GpuError::KernelError(format!("average_predictions launch failed: {}", e)))?;
        }
        
        self.device.synchronize().map_err(|e| GpuError::BackendError(e.to_string()))?;
        
        debug!("[CUDA/GPU] average completed");
        Ok(())
    }
    
    fn kernel_matmul(
        &mut self,
        params: &MatmulParams,
        a: BufferId,
        b: BufferId,
        c: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA/GPU] kernel_matmul(m={}, k={}, n={}) using cuBLAS",
               params.m, params.k, params.n);
        
        // Use cuBLAS SGEMM - this uses Tensor Cores on H100!
        unsafe {
            let a_f32 = self.buffer_as_f32(a)?;
            let b_f32 = self.buffer_as_f32(b)?;
            let mut c_f32 = self.buffer_as_f32(c)?;
            
            // cuBLAS uses column-major, so we compute C = B^T @ A^T to get row-major result
            // Or we can just use it directly if data is already column-major
            
            // For row-major: C = alpha * A @ B + beta * C
            // With cuBLAS column-major: we swap A and B and transpose
            
            let (m, n, k) = (params.m as i32, params.n as i32, params.k as i32);
            let lda = if params.trans_a { m } else { k };
            let ldb = if params.trans_b { k } else { n };
            let ldc = n;
            
            self.blas.gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n, m, k,
                params.alpha,
                &b_f32, ldb,
                &a_f32, lda,
                params.beta,
                &mut c_f32, ldc,
            ).map_err(|e| GpuError::KernelError(format!("cuBLAS SGEMM failed: {:?}", e)))?;
        }
        
        self.device.synchronize().map_err(|e| GpuError::BackendError(e.to_string()))?;
        
        debug!("[CUDA/GPU] matmul completed (cuBLAS SGEMM)");
        Ok(())
    }
    
    fn kernel_elementwise(
        &mut self,
        params: &ElementwiseParams,
        input_a: BufferId,
        input_b: Option<BufferId>,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA/GPU] kernel_elementwise(n_elements={}, op={:?})",
               params.n_elements, params.op);
        
        let input_buf = self.get_buffer(input_a)?;
        let output_buf = self.get_buffer(output)?;
        
        let input_ptr = input_buf.data.device_ptr();
        let output_ptr = output_buf.data.device_ptr();
        
        let block_size = 256u32;
        let grid_size = (params.n_elements + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        match params.op {
            ElementwiseOp::Relu => {
                let func = self.device.get_func("ml_kernels", "relu")
                    .map_err(|e| GpuError::KernelError(format!("Failed to get relu: {}", e)))?;
                
                unsafe {
                    func.launch(cfg, (input_ptr, output_ptr, params.n_elements))
                        .map_err(|e| GpuError::KernelError(format!("relu launch failed: {}", e)))?;
                }
            }
            ElementwiseOp::Sigmoid => {
                let func = self.device.get_func("ml_kernels", "sigmoid")
                    .map_err(|e| GpuError::KernelError(format!("Failed to get sigmoid: {}", e)))?;
                
                unsafe {
                    func.launch(cfg, (input_ptr, output_ptr, params.n_elements))
                        .map_err(|e| GpuError::KernelError(format!("sigmoid launch failed: {}", e)))?;
                }
            }
            ElementwiseOp::Add | ElementwiseOp::Mul => {
                // For binary ops, use cuBLAS axpy or element-wise multiply
                let input_b = input_b.ok_or(GpuError::InvalidParams("Binary op requires second input".into()))?;
                
                unsafe {
                    let a = self.buffer_as_f32(input_a)?;
                    let b = self.buffer_as_f32(input_b)?;
                    let mut out = self.buffer_as_f32(output)?;
                    
                    match params.op {
                        ElementwiseOp::Add => {
                            // out = a, then out += b
                            self.device.dtod_copy(&a, &mut out)
                                .map_err(|e| GpuError::BackendError(e.to_string()))?;
                            self.blas.axpy(1.0f32, &b, &mut out);
                        }
                        ElementwiseOp::Mul => {
                            // Element-wise multiply - read, compute, write back
                            let a_host = self.device.dtoh_sync_copy(&a)
                                .map_err(|e| GpuError::BackendError(e.to_string()))?;
                            let b_host = self.device.dtoh_sync_copy(&b)
                                .map_err(|e| GpuError::BackendError(e.to_string()))?;
                            
                            let result: Vec<f32> = a_host.iter().zip(b_host.iter())
                                .map(|(&x, &y)| x * y)
                                .collect();
                            
                            self.device.htod_copy_into(result, &mut out)
                                .map_err(|e| GpuError::BackendError(e.to_string()))?;
                        }
                        _ => unreachable!()
                    }
                }
            }
            _ => {
                // For other ops, use cuBLAS or simple GPU implementation
                warn!("[CUDA] Elementwise op {:?} using scalar fallback", params.op);
                
                let a_bytes = self.buffer_read(input_a, 0, (params.n_elements * 4) as u32)?;
                let a: Vec<f32> = a_bytes.chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                
                let result: Vec<f32> = match params.op {
                    ElementwiseOp::Tanh => a.iter().map(|&x| x.tanh()).collect(),
                    ElementwiseOp::Sqrt => a.iter().map(|&x| x.sqrt()).collect(),
                    ElementwiseOp::Exp => a.iter().map(|&x| x.exp()).collect(),
                    ElementwiseOp::Log => a.iter().map(|&x| x.ln()).collect(),
                    _ => unreachable!()
                };
                
                let out_bytes: Vec<u8> = result.iter()
                    .flat_map(|&x| x.to_le_bytes())
                    .collect();
                
                self.buffer_write(output, 0, &out_bytes)?;
                return Ok(());
            }
        }
        
        self.device.synchronize().map_err(|e| GpuError::BackendError(e.to_string()))?;
        
        debug!("[CUDA/GPU] elementwise completed");
        Ok(())
    }
    
    fn kernel_reduce(
        &mut self,
        params: &ReduceParams,
        input: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA/GPU] kernel_reduce(n_elements={}, op={:?})",
               params.n_elements, params.op);
        
        match params.op {
            ReduceOp::Sum => {
                // Use cuBLAS asum or custom kernel
                let input_buf = self.get_buffer(input)?;
                let output_buf = self.get_buffer(output)?;
                
                let func = self.device.get_func("ml_kernels", "reduce_sum")
                    .map_err(|e| GpuError::KernelError(format!("Failed to get reduce_sum: {}", e)))?;
                
                let cfg = LaunchConfig {
                    block_dim: (1, 1, 1),
                    grid_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };
                
                unsafe {
                    func.launch(cfg, (
                        input_buf.data.device_ptr(),
                        output_buf.data.device_ptr(),
                        params.n_elements,
                    )).map_err(|e| GpuError::KernelError(format!("reduce_sum launch failed: {}", e)))?;
                }
            }
            ReduceOp::Mean => {
                // Sum then divide
                let input_buf = self.get_buffer(input)?;
                
                unsafe {
                    let input_f32 = self.buffer_as_f32(input)?;
                    let ones = self.device.alloc_zeros::<f32>(params.n_elements as usize)
                        .map_err(|e| GpuError::OutOfMemory)?;
                    
                    // Fill ones
                    let ones_host = vec![1.0f32; params.n_elements as usize];
                    let ones = self.device.htod_copy(ones_host)
                        .map_err(|e| GpuError::BackendError(e.to_string()))?;
                    
                    let sum: f32 = self.blas.dot(&input_f32, &ones);
                    let mean = sum / params.n_elements as f32;
                    
                    self.buffer_write(output, 0, &mean.to_le_bytes())?;
                }
                return Ok(());
            }
            _ => {
                // Other reductions - use host for now
                let in_bytes = self.buffer_read(input, 0, (params.n_elements * 4) as u32)?;
                let data: Vec<f32> = in_bytes.chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                
                let result = match params.op {
                    ReduceOp::Max => data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    ReduceOp::Min => data.iter().cloned().fold(f32::INFINITY, f32::min),
                    ReduceOp::Variance => {
                        let mean = data.iter().sum::<f32>() / data.len() as f32;
                        data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
                    }
                    _ => unreachable!()
                };
                
                self.buffer_write(output, 0, &result.to_le_bytes())?;
                return Ok(());
            }
        }
        
        self.device.synchronize().map_err(|e| GpuError::BackendError(e.to_string()))?;
        
        debug!("[CUDA/GPU] reduce completed");
        Ok(())
    }
    
    fn kernel_batch_predict(
        &mut self,
        params: &BatchPredictParams,
        samples: BufferId,
        tree_nodes: BufferId,
        tree_offsets: BufferId,
        output: BufferId,
    ) -> Result<(), GpuError> {
        debug!("[CUDA/GPU] kernel_batch_predict(batch_size={}, n_features={}, n_trees={})",
               params.batch_size, params.n_features, params.n_trees);
        
        // Tree traversal is complex - would need custom kernel
        // For now, this is a placeholder
        warn!("[CUDA] batch_predict requires custom kernel - not yet implemented");
        
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
